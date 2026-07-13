"""Plate Tectonics: voronoi plates drift, collide into mountains, tear open rifts."""

import numpy as np
import taichi as ti

N = 256
N_PLATES = 7
DT = 0.02
DRIFT_STEP = 1.0
DRIFT_EVERY = 6
UPLIFT = 0.22
RIFT = 0.15
EROSION = 0.10
SEA = 0.48
NEW_CRUST = 0.25
RELIEF = 40.0

QUAKE_CONV = 0.8
QUAKE_PROB = 0.002
ACTIVITY_DECAY = 0.92

plate_id = None
plate_id_next = None
height = None
height_next = None
plate_vel = None
activity = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global plate_id, plate_id_next, height, height_next, plate_vel, activity, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    plate_id = ti.field(ti.i32, shape=(N, N))
    plate_id_next = ti.field(ti.i32, shape=(N, N))
    height = ti.field(ti.f32, shape=(N, N))
    height_next = ti.field(ti.f32, shape=(N, N))
    plate_vel = ti.Vector.field(2, ti.f32, shape=N_PLATES)
    activity = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def voronoi_plates(n, seeds):
    """Pure numpy: nearest-seed labeling with toroidal (wraparound) distance."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    best = np.full((n, n), 1e18)
    pid = np.zeros((n, n), dtype=np.int32)
    for k in range(len(seeds)):
        dx = np.abs(ii - seeds[k, 0])
        dy = np.abs(jj - seeds[k, 1])
        dx = np.minimum(dx, n - dx)
        dy = np.minimum(dy, n - dy)
        d = dx * dx + dy * dy
        m = d < best
        best[m] = d[m]
        pid[m] = k
    return pid


def seed_world(rng_seed=0):
    """Pure numpy: plates from voronoi, continents vs oceans, one drift vector per plate."""
    rng = np.random.default_rng(rng_seed)
    seeds = rng.uniform(0, N, size=(N_PLATES, 2)).astype(np.float32)
    pid = voronoi_plates(N, seeds)
    is_continent = rng.random(N_PLATES) < 0.4
    h = np.where(is_continent[pid], 0.58, 0.30).astype(np.float32)
    h += rng.normal(0, 0.02, (N, N)).astype(np.float32)
    v = rng.uniform(-1, 1, size=(N_PLATES, 2)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return pid, h, v


def apply_seed(rng_seed=0):
    pid, h, v = seed_world(rng_seed)
    plate_id.from_numpy(pid)
    height.from_numpy(h)
    plate_vel.from_numpy(v)
    activity.fill(0.0)


@ti.func
def wrap(i):
    return ((i % N) + N) % N


@ti.kernel
def boundary_forces():
    for i, j in height:
        me = plate_id[i, j]
        delta = 0.0
        for k in ti.static(range(4)):
            di = (1, -1, 0, 0)[k]
            dj = (0, 0, 1, -1)[k]
            ni, nj = wrap(i + di), wrap(j + dj)
            other = plate_id[ni, nj]
            if other != me:
                rel = plate_vel[me] - plate_vel[other]
                conv = rel[0] * di + rel[1] * dj
                if conv > 0:
                    delta += UPLIFT * conv * DT * (1.0 - height[i, j])
                else:
                    delta += RIFT * conv * DT * height[i, j]
                if ti.abs(conv) > QUAKE_CONV and ti.random() < QUAKE_PROB:
                    activity[i, j] = 1.0
        height_next[i, j] = ti.math.clamp(height[i, j] + delta, 0.0, 1.0)


@ti.kernel
def erode():
    for i, j in height:
        avg = 0.25 * (
            height_next[wrap(i + 1), j]
            + height_next[wrap(i - 1), j]
            + height_next[i, wrap(j + 1)]
            + height_next[i, wrap(j - 1)]
        )
        height[i, j] = height_next[i, j] * (1 - EROSION) + avg * EROSION


@ti.kernel
def drift():
    for i, j in plate_id:
        best_h = -1.0
        best_id = -1
        for k in range(N_PLATES):
            si = wrap(i - ti.cast(ti.round(plate_vel[k][0] * DRIFT_STEP), ti.i32))
            sj = wrap(j - ti.cast(ti.round(plate_vel[k][1] * DRIFT_STEP), ti.i32))
            if plate_id[si, sj] == k:
                if height[si, sj] > best_h:
                    best_h = height[si, sj]
                    best_id = k
        if best_id >= 0:
            plate_id_next[i, j] = best_id
            height_next[i, j] = best_h
        else:
            plate_id_next[i, j] = plate_id[i, j]
            height_next[i, j] = NEW_CRUST


@ti.kernel
def copy_drift():
    for i, j in plate_id:
        plate_id[i, j] = plate_id_next[i, j]
        height[i, j] = height_next[i, j]


@ti.kernel
def smooth_after_drift():
    for i, j in height:
        height_next[i, j] = height[i, j]
    for i, j in height:
        avg = 0.25 * (
            height_next[wrap(i + 1), j]
            + height_next[wrap(i - 1), j]
            + height_next[i, wrap(j + 1)]
            + height_next[i, wrap(j - 1)]
        )
        height[i, j] = height_next[i, j] * 0.5 + avg * 0.5


@ti.kernel
def decay_activity():
    for i, j in activity:
        activity[i, j] *= ACTIVITY_DECAY


@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t


@ti.kernel
def render():
    for i, j in pixels:
        hh = height[i, j]
        c = ti.Vector([0.05, 0.15, 0.4])
        if hh > SEA:
            land = (hh - SEA) / (1.0 - SEA)
            c = band(ti.Vector([0.55, 0.6, 0.3]), ti.Vector([0.45, 0.4, 0.35]), land, 0.1, 0.5)
            c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.55, 0.8)
        else:
            c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), hh / SEA, 0.3, 1.0)
        dhdx = (height[wrap(i + 1), j] - hh) * RELIEF
        dhdy = (height[i, wrap(j + 1)] - hh) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.4 + 0.6 * normal.dot(light), 0.0, 1.0)
        c = c * shade
        c += activity[i, j] * ti.Vector([1.0, 0.35, 0.05])
        pixels[i, j] = ti.math.clamp(c, 0.0, 1.0)


def step(frame):
    boundary_forces()
    erode()
    if frame % DRIFT_EVERY == 0:
        drift()
        copy_drift()
        smooth_after_drift()
    decay_activity()


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Plate Tectonics — taichi-academy", res=N, background_color=0x0A0A12)
    frame = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        step(frame)
        frame += 1
        render()
        gui.set_image(pixels)
        gui.text("mountains rise where plates meet", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new world", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

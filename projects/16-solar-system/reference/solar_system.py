"""Solar System: real 1/r^2 gravity, a leapfrog integrator, and orbits that actually hold."""

import numpy as np
import taichi as ti

RES = 512
GM = 1.0
DT = 0.002
SUBSTEPS = 4

N_PLANETS = 6
N_BELT = 4000
N_COMETS = 24
N = N_PLANETS + N_BELT + N_COMETS
PLANET_BASE = 0
BELT_BASE = N_PLANETS
COMET_BASE = N_PLANETS + N_BELT

BELT_R = (0.30, 0.36)
COMET_PERI = 0.06
COMET_APO = 0.46
VIEW_SCALE = 0.95
CANVAS_FADE = 0.90

PLANET_COLORS = np.array(
    [
        [0.75, 0.72, 0.68],
        [0.95, 0.85, 0.55],
        [0.30, 0.55, 0.95],
        [0.90, 0.45, 0.25],
        [0.85, 0.75, 0.55],
        [0.60, 0.80, 0.90],
    ],
    dtype=np.float32,
)

pos = None
vel = None
color = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, color, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    color = ti.Vector.field(3, ti.f32, shape=N)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def circular_velocity(p):
    """Pure numpy: the speed that makes gravity exactly the centripetal force — one orbit, forever."""
    r = np.linalg.norm(p, axis=-1, keepdims=True)
    speed = np.sqrt(GM / r.squeeze(-1))
    tangent = np.stack([-p[..., 1], p[..., 0]], axis=-1) / r
    return tangent * speed[..., None]


def seed_planets(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    radii = np.linspace(0.10, 0.42, N_PLANETS).astype(np.float32)
    ang = rng.uniform(0.0, 2 * np.pi, N_PLANETS)
    p = np.stack([radii * np.cos(ang), radii * np.sin(ang)], axis=1).astype(np.float32)
    return p, circular_velocity(p).astype(np.float32), PLANET_COLORS.copy()


def seed_belt(rng_seed=0):
    rng = np.random.default_rng(rng_seed + 1)
    r = rng.uniform(BELT_R[0], BELT_R[1], N_BELT)
    ang = rng.uniform(0.0, 2 * np.pi, N_BELT)
    p = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)
    v = circular_velocity(p).astype(np.float32)
    col = np.full((N_BELT, 3), (0.35, 0.32, 0.28), dtype=np.float32)
    col *= rng.uniform(0.5, 1.0, (N_BELT, 1)).astype(np.float32)
    return p, v, col


def comet_aphelion_velocity(r_apo, r_peri):
    """Pure numpy: vis-viva at aphelion for an ellipse with the given extremes."""
    a = 0.5 * (r_apo + r_peri)
    return np.sqrt(GM * (2.0 / r_apo - 1.0 / a))


def seed_comets(rng_seed=0):
    rng = np.random.default_rng(rng_seed + 2)
    ang = rng.uniform(0.0, 2 * np.pi, N_COMETS)
    r_apo = rng.uniform(COMET_APO * 0.8, COMET_APO, N_COMETS)
    p = np.stack([r_apo * np.cos(ang), r_apo * np.sin(ang)], axis=1).astype(np.float32)
    speed = comet_aphelion_velocity(r_apo, COMET_PERI)
    tangent = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    v = (tangent * speed[:, None]).astype(np.float32)
    col = np.full((N_COMETS, 3), (0.55, 0.85, 0.95), dtype=np.float32)
    return p, v, col


def apply_seed(rng_seed=0):
    parts = [seed_planets(rng_seed), seed_belt(rng_seed), seed_comets(rng_seed)]
    pos.from_numpy(np.concatenate([p for p, _v, _c in parts]))
    vel.from_numpy(np.concatenate([v for _p, v, _c in parts]))
    color.from_numpy(np.concatenate([c for _p, _v, c in parts]))
    pixels.fill(0.0)


@ti.func
def accel(p):
    r2 = p.dot(p) + 1e-6
    r = ti.sqrt(r2)
    return -GM * p / (r2 * r)


@ti.kernel
def leapfrog():
    for b in pos:
        vel[b] += 0.5 * DT * accel(pos[b])
        pos[b] += DT * vel[b]
        vel[b] += 0.5 * DT * accel(pos[b])


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    cx = RES // 2
    for _ in range(1):
        for di, dj in ti.ndrange((-4, 5), (-4, 5)):
            w = ti.exp(-(di * di + dj * dj) / 6.0)
            pixels[cx + di, cx + dj] += w * ti.Vector([1.0, 0.85, 0.4])

    for b in pos:
        x = 0.5 + pos[b][0] * VIEW_SCALE
        y = 0.5 + pos[b][1] * VIEW_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 1 <= xi < RES - 1 and 1 <= yi < RES - 1:
            gain = 1.0
            if b >= BELT_BASE and b < COMET_BASE:
                gain = 0.35
            pixels[xi, yi] += color[b] * gain
            if b < N_PLANETS or b >= COMET_BASE:
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    pixels[xi + di, yi + dj] += color[b] * 0.3


@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)


def step():
    for _ in range(SUBSTEPS):
        leapfrog()
    render()
    clamp_pixels()


def total_energy():
    """Pure numpy: kinetic + potential per body — the quantity leapfrog protects."""
    p = pos.to_numpy()
    v = vel.to_numpy()
    ke = 0.5 * (v**2).sum(axis=1)
    pe = -GM / np.linalg.norm(p, axis=1)
    return ke + pe


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Solar System — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        step()
        gui.set_image(pixels)
        gui.text("planets, belt, comets — leapfrog keeps them honest", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] rescatter", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

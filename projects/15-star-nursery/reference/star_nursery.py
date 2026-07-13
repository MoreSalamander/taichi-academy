"""Star Nursery: a molecular cloud collapses under its own gravity and ignites into stars."""

import numpy as np
import taichi as ti

RES = 512
GRID = 128
N_GAS = 40000
MAX_STARS = 400

GRAVITY_PULL = 120.0
DAMPING = 0.96
IGNITE_DENSITY = 22.0
IGNITE_PROB = 0.001
RADIATION = 250.0
RADIATION_R = 0.05
DT = 0.004

GAS_COLOR = (0.10, 0.08, 0.20)
STAR_COLOR = (1.0, 0.9, 0.7)
CANVAS_FADE = 0.85

pos = None
vel = None
alive = None
star_pos = None
star_age = None
n_stars = None
density = None
density_blur = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, alive, star_pos, star_age, n_stars, density, density_blur, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N_GAS)
    vel = ti.Vector.field(2, ti.f32, shape=N_GAS)
    alive = ti.field(ti.i32, shape=N_GAS)
    star_pos = ti.Vector.field(2, ti.f32, shape=MAX_STARS)
    star_age = ti.field(ti.f32, shape=MAX_STARS)
    n_stars = ti.field(ti.i32, shape=())
    density = ti.field(ti.f32, shape=(GRID, GRID))
    density_blur = ti.field(ti.f32, shape=(GRID, GRID))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def seed_gas(n, rng_seed=0, blobs=4):
    """Pure numpy: a few overlapping gaussian gas clouds."""
    rng = np.random.default_rng(rng_seed)
    centers = rng.uniform(0.25, 0.75, size=(blobs, 2))
    which = rng.integers(0, blobs, n)
    p = centers[which] + rng.normal(0, 0.09, size=(n, 2))
    return np.clip(p, 0.02, 0.98).astype(np.float32)


def apply_seed(rng_seed=0):
    pos.from_numpy(seed_gas(N_GAS, rng_seed))
    vel.fill(0.0)
    alive.fill(1)
    n_stars[None] = 0
    pixels.fill(0.0)


@ti.kernel
def clear_density():
    for i, j in density:
        density[i, j] = 0.0


@ti.kernel
def deposit():
    for p in pos:
        if alive[p] == 1:
            gi = ti.cast(pos[p][0] * GRID, ti.i32)
            gj = ti.cast(pos[p][1] * GRID, ti.i32)
            if 0 <= gi < GRID and 0 <= gj < GRID:
                density[gi, gj] += 1.0


@ti.kernel
def blur():
    for i, j in density_blur:
        acc = 0.0
        cnt = 0.0
        for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                acc += density[ni, nj]
                cnt += 1.0
        density_blur[i, j] = acc / cnt


@ti.kernel
def gravity():
    for p in pos:
        if alive[p] == 1:
            gi = ti.min(ti.max(ti.cast(pos[p][0] * GRID, ti.i32), 1), GRID - 2)
            gj = ti.min(ti.max(ti.cast(pos[p][1] * GRID, ti.i32), 1), GRID - 2)
            gx = (density_blur[gi + 1, gj] - density_blur[gi - 1, gj]) * 0.5
            gy = (density_blur[gi, gj + 1] - density_blur[gi, gj - 1]) * 0.5
            vel[p] += DT * GRAVITY_PULL * ti.Vector([gx, gy]) / GRID


@ti.kernel
def radiation():
    for p in pos:
        if alive[p] == 1:
            f = ti.Vector([0.0, 0.0])
            for s in range(n_stars[None]):
                d = pos[p] - star_pos[s]
                r2 = d.dot(d)
                if r2 < RADIATION_R * RADIATION_R:
                    r = ti.sqrt(r2) + 1e-4
                    f += RADIATION * (1.0 - r / RADIATION_R) * d / r
            vel[p] += DT * f


@ti.kernel
def integrate():
    for p in pos:
        if alive[p] == 1:
            vel[p] *= DAMPING
            newp = pos[p] + DT * vel[p]
            for a in ti.static(range(2)):
                if newp[a] < 0.01:
                    newp[a] = 0.01
                    vel[p][a] *= -0.5
                if newp[a] > 0.99:
                    newp[a] = 0.99
                    vel[p][a] *= -0.5
            pos[p] = newp


@ti.kernel
def ignite():
    for p in pos:
        if alive[p] == 1 and n_stars[None] < MAX_STARS:
            gi = ti.min(ti.max(ti.cast(pos[p][0] * GRID, ti.i32), 0), GRID - 1)
            gj = ti.min(ti.max(ti.cast(pos[p][1] * GRID, ti.i32), 0), GRID - 1)
            if density_blur[gi, gj] > IGNITE_DENSITY:
                if ti.random() < IGNITE_PROB:
                    s = ti.atomic_add(n_stars[None], 1)
                    if s < MAX_STARS:
                        star_pos[s] = pos[p]
                        star_age[s] = 0.0
                        alive[p] = 0


@ti.kernel
def age_stars(dt: ti.f32):
    for s in range(n_stars[None]):
        star_age[s] += dt


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for p in pos:
        if alive[p] == 1:
            xi = ti.cast(pos[p][0] * RES, ti.i32)
            yi = ti.cast(pos[p][1] * RES, ti.i32)
            if 0 <= xi < RES and 0 <= yi < RES:
                pixels[xi, yi] += ti.Vector([GAS_COLOR[0], GAS_COLOR[1], GAS_COLOR[2]])

    for s in range(n_stars[None]):
        cx = star_pos[s][0] * RES
        cy = star_pos[s][1] * RES
        glow = ti.min(star_age[s] * 2.0, 1.0)
        for di, dj in ti.ndrange((-3, 4), (-3, 4)):
            xi = ti.cast(cx, ti.i32) + di
            yi = ti.cast(cy, ti.i32) + dj
            if 0 <= xi < RES and 0 <= yi < RES:
                w = ti.exp(-(di * di + dj * dj) / 4.0)
                pixels[xi, yi] += glow * w * ti.Vector([STAR_COLOR[0], STAR_COLOR[1], STAR_COLOR[2]])


@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)


def step():
    clear_density()
    deposit()
    blur()
    gravity()
    radiation()
    integrate()
    ignite()
    age_stars(DT)
    render()
    clamp_pixels()


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Star Nursery — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        step()
        gui.set_image(pixels)
        gui.text(f"stars born: {n_stars[None]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new cloud", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

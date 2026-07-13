"""Galaxy Creator: star particles on spiral arms, differential rotation, additive light."""

import numpy as np
import taichi as ti

RES = 512
N_STARS = 60000

SPIRAL, ELLIPTICAL, RING = 0, 1, 2
NAMES = {SPIRAL: "spiral", ELLIPTICAL: "elliptical", RING: "ring"}

FADE = 0.88
SPLAT_GAIN = 0.35
DISK_SCALE = 0.55
ROT_SPEED = 0.35
ROT_SOFTEN = 0.05
DT = 0.016

radius_f = None
angle_f = None
color_f = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global radius_f, angle_f, color_f, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    radius_f = ti.field(ti.f32, shape=N_STARS)
    angle_f = ti.field(ti.f32, shape=N_STARS)
    color_f = ti.Vector.field(3, ti.f32, shape=N_STARS)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def star_colors(r, rng, core_scale=0.12, core_col=(0.7, 0.6, 0.4), arm_col=(0.5, 0.6, 1.0)):
    """Pure numpy: blend core color to arm color by radius, dimmed per-star at random."""
    n = len(r)
    core = np.exp(-r / core_scale)
    col = np.zeros((n, 3), dtype=np.float32)
    for ch in range(3):
        col[:, ch] = core_col[ch] * core + arm_col[ch] * (1 - core)
    brightness = rng.uniform(0.3, 1.0, n) ** 2
    return (col * brightness[:, None]).astype(np.float32)


def disk_radii(n, rng, scale=0.18, r_min=0.01, r_max=0.85):
    """Pure numpy: exponential-falloff radii, re-rolling any that land outside the disk."""
    r = rng.exponential(scale, n)
    bad = (r < r_min) | (r > r_max)
    r[bad] = rng.uniform(r_min, r_max, bad.sum())
    return r.astype(np.float32)


def seed_spiral(n, rng_seed=0, arms=2, twist=3.5):
    """Pure numpy: stars scattered along logarithmic spiral arms."""
    rng = np.random.default_rng(rng_seed)
    r = disk_radii(n, rng)
    arm = rng.integers(0, arms, n)
    theta = arm * (2 * np.pi / arms) + twist * np.log(r / 0.01)
    theta = theta + rng.normal(0, 0.25, n) * (0.3 + r)
    return r, theta.astype(np.float32), star_colors(r, rng)


def seed_elliptical(n, rng_seed=0):
    """Pure numpy: a smooth, armless, golden-old-star blob."""
    rng = np.random.default_rng(rng_seed)
    r = disk_radii(n, rng, scale=0.22)
    theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
    col = star_colors(r, rng, core_scale=0.3, core_col=(0.9, 0.75, 0.5), arm_col=(0.8, 0.6, 0.4))
    return r, theta, col


def seed_ring(n, rng_seed=0):
    """Pure numpy: a thin ring of hot blue stars with a sparse old core."""
    rng = np.random.default_rng(rng_seed)
    n_core = n // 5
    n_ring = n - n_core
    r_ring = rng.normal(0.55, 0.045, n_ring)
    r_core = rng.exponential(0.06, n_core)
    r = np.clip(np.concatenate([r_ring, r_core]), 0.01, 0.85).astype(np.float32)
    theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
    col = star_colors(r, rng, core_scale=0.1, core_col=(0.9, 0.8, 0.6), arm_col=(0.4, 0.65, 1.0))
    return r, theta, col


def seed_galaxy(kind, rng_seed=0):
    if kind == SPIRAL:
        return seed_spiral(N_STARS, rng_seed)
    if kind == ELLIPTICAL:
        return seed_elliptical(N_STARS, rng_seed)
    return seed_ring(N_STARS, rng_seed)


def apply_seed(seed):
    r, theta, col = seed
    radius_f.from_numpy(r)
    angle_f.from_numpy(theta)
    color_f.from_numpy(col)
    pixels.fill(0.0)


@ti.kernel
def rotate(dt: ti.f32):
    for s in radius_f:
        omega = ROT_SPEED / (radius_f[s] + ROT_SOFTEN)
        angle_f[s] += omega * dt


@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE


@ti.kernel
def splat():
    for s in radius_f:
        r = radius_f[s]
        a = angle_f[s]
        x = 0.5 + r * ti.cos(a) * DISK_SCALE
        y = 0.5 + r * ti.sin(a) * DISK_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] += color_f[s] * SPLAT_GAIN


@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)


def step(dt=DT):
    rotate(dt)
    fade()
    splat()
    clamp_pixels()


def main():
    init_sim()
    kind = SPIRAL
    apply_seed(seed_galaxy(kind))
    gui = ti.GUI("Galaxy Creator — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "123":
                kind = int(e.key) - 1
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "r":
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))
        step()
        gui.set_image(pixels)
        gui.text(f"galaxy: {NAMES[kind]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1] spiral  [2] elliptical  [3] ring  [r] reroll", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

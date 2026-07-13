"""Strange Attractors: 200,000 points fall onto the same impossible shape, every time."""

import numpy as np
import taichi as ti

RES = 512
N_PTS = 200000
FADE = 0.90
ROT_SPEED = 0.01
SETTLE_STEPS = 2000

LORENZ, THOMAS, AIZAWA, CLIFFORD = 0, 1, 2, 3
NAMES = {LORENZ: "lorenz", THOMAS: "thomas", AIZAWA: "aizawa", CLIFFORD: "clifford"}

# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
    AIZAWA: (0.28, 0.0, 0.0, 0.0, 0.01, 0.015, 0.4, 0.35),
    CLIFFORD: (0.20, 0.0, 0.0, 0.0, 1.0, 0.04, 1.0, 0.0),
}

pos = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(3, ti.f32, shape=N_PTS)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def seed_points(n, seed_scale, rng_seed=0):
    """Pure numpy: a random cloud sized to land inside the attractor's basin."""
    rng = np.random.default_rng(rng_seed)
    return (rng.uniform(-1.0, 1.0, size=(n, 3)) * seed_scale).astype(np.float32)


@ti.func
def deriv_lorenz(p):
    return ti.Vector([10.0 * (p[1] - p[0]), p[0] * (28.0 - p[2]) - p[1], p[0] * p[1] - 8.0 / 3.0 * p[2]])


@ti.func
def deriv_thomas(p):
    b = 0.19
    return ti.Vector([ti.sin(p[1]) - b * p[0], ti.sin(p[2]) - b * p[1], ti.sin(p[0]) - b * p[2]])


@ti.func
def deriv_aizawa(p):
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
    x, y, z = p[0], p[1], p[2]
    return ti.Vector([
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z**3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x**3,
    ])


@ti.func
def map_clifford(p):
    a, b, c, d = -1.4, 1.6, 1.0, 0.7
    return ti.Vector([ti.sin(a * p[1]) + c * ti.cos(a * p[0]), ti.sin(b * p[0]) + d * ti.cos(b * p[1]), 0.0])


@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        elif kind == THOMAS:
            pos[i] = p + dt * deriv_thomas(p)
        elif kind == AIZAWA:
            pos[i] = p + dt * deriv_aizawa(p)
        else:
            pos[i] = map_clifford(p)


def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale, _spd = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    for _ in range(SETTLE_STEPS):
        step_attractor(kind, dt)
    pixels.fill(0.0)


@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE


@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        if kind == CLIFFORD:
            rx = p[0]
            ry = p[1]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            elif kind == THOMAS:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            elif kind == AIZAWA:
                spd = deriv_aizawa(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain


@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)


def step(kind, angle):
    scale, cx, cy, cz, dt, gain, _seed, speed_scale = FRAME[kind]
    step_attractor(kind, dt)
    fade()
    splat(kind, angle, scale, cx, cy, cz, gain, speed_scale)
    clamp_pixels()


def main():
    init_sim()
    kind = LORENZ
    apply_seed(kind)
    gui = ti.GUI("Strange Attractors — taichi-academy", res=RES, background_color=0x000000)
    angle = 0.0
    pmx = None
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "1234":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                angle -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            angle += ROT_SPEED
        step(kind, angle)
        gui.set_image(pixels)
        gui.text(f"attractor: {NAMES[kind]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1] lorenz  [2] thomas  [3] aizawa  [4] clifford  drag to spin  [r] rescatter", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

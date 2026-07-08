"""Branching lightning: recursive bolts, blue afterglow, storm flashes."""

import numpy as np
import taichi as ti

N = 512
FADE = 0.90
GLOW_FADE = 0.96
GLOW_SPREAD = 0.2
FLASH_FADE = 0.85
BRANCH_CHANCE = 0.35
STORM_PERIOD = 90

bolt = None
deposit = None
glow = None
glow_next = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global bolt, deposit, glow, glow_next, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    bolt = ti.field(ti.f32, shape=(N, N))
    deposit = ti.field(ti.f32, shape=(N, N))
    glow = ti.field(ti.f32, shape=(N, N))
    glow_next = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def deposit_segment(field, p0, p1, bright):
    """Pure numpy: stamp a straight bright segment into a (n, n) array."""
    n = field.shape[0]
    length = float(np.hypot(*(p1 - p0)))
    steps = max(2, int(length * 2))
    ts = np.linspace(0.0, 1.0, steps)
    xs = np.clip(p0[0] + (p1[0] - p0[0]) * ts, 0, n - 1).astype(np.int32)
    ys = np.clip(p0[1] + (p1[1] - p0[1]) * ts, 0, n - 1).astype(np.int32)
    field[xs, ys] = np.maximum(field[xs, ys], bright)


def generate_bolt(n, x_frac, rng_seed=0):
    """Pure numpy + recursion: a jagged, branching bolt as a (n, n) brightness array."""
    rng = np.random.default_rng(rng_seed)
    field = np.zeros((n, n), dtype=np.float32)

    def jag(p0, p1, bright, depth):
        d = p1 - p0
        length = float(np.hypot(*d))
        if length < 8.0 or depth > 10:
            deposit_segment(field, p0, p1, bright)
            return
        mid = (p0 + p1) / 2
        perp = np.array([-d[1], d[0]]) / (length + 1e-9)
        mid = mid + perp * rng.uniform(-0.25, 0.25) * length
        jag(p0, mid, bright, depth + 1)
        jag(mid, p1, bright, depth + 1)
        if depth <= 4 and rng.random() < BRANCH_CHANCE:
            dirv = mid - p0
            ang = rng.uniform(-0.7, 0.7)
            ca, sa = np.cos(ang), np.sin(ang)
            rot = np.array([dirv[0] * ca - dirv[1] * sa, dirv[0] * sa + dirv[1] * ca])
            jag(mid, mid + rot * 0.7, bright * 0.45, depth + 1)

    start = np.array([x_frac * n, n - 1.0])
    end = np.array([x_frac * n + rng.uniform(-0.15, 0.15) * n, 0.0])
    jag(start, end, 1.0, 0)
    return field


@ti.kernel
def absorb():
    for i, j in bolt:
        bolt[i, j] = ti.max(bolt[i, j], deposit[i, j])
        glow[i, j] += deposit[i, j]


def strike(x_frac, rng_seed=0):
    deposit.from_numpy(generate_bolt(N, x_frac, rng_seed))
    absorb()


@ti.kernel
def fade():
    for i, j in bolt:
        bolt[i, j] *= FADE
        glow[i, j] *= GLOW_FADE


@ti.kernel
def diffuse_glow():
    for i, j in glow:
        lap = (
            glow[(i + 1) % N, j]
            + glow[(i - 1) % N, j]
            + glow[i, (j + 1) % N]
            + glow[i, (j - 1) % N]
            - 4.0 * glow[i, j]
        )
        glow_next[i, j] = glow[i, j] + GLOW_SPREAD * lap


@ti.kernel
def copy_glow():
    for i, j in glow:
        glow[i, j] = glow_next[i, j]


@ti.kernel
def clear_fields():
    for i, j in bolt:
        bolt[i, j] = 0.0
        glow[i, j] = 0.0


def step():
    fade()
    diffuse_glow()
    copy_glow()


@ti.kernel
def render(flash: ti.f32):
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        g = ti.min(glow[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04]) + flash * ti.Vector([0.06, 0.08, 0.16])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        halo = g * ti.Vector([0.25, 0.40, 0.95])
        pixels[i, j] = ti.math.clamp(sky + halo + core, 0.0, 1.0)


def main():
    init_sim()
    gui = ti.GUI("Lightning — taichi-academy", res=(N, N))
    storm_on = True
    flash = 0.0
    frame = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
            elif e.key == ti.GUI.SPACE:
                storm_on = not storm_on
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0
        if storm_on and frame % STORM_PERIOD == 0:
            strike(np.random.random(), np.random.randint(1_000_000))
            flash = 1.0
        step()
        frame += 1
        render(flash)
        flash *= FLASH_FADE
        gui.set_image(pixels)
        storm = "on" if storm_on else "off"
        gui.text(f"storm: {storm}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("click to strike  [space] storm  [r] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

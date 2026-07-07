"""Gray-Scott reaction-diffusion: two chemicals paint living patterns."""

import numpy as np
import taichi as ti

N = 512
DU = 1.0
DV = 0.5
DT = 1.0
SUBSTEPS = 12
SEED_SIZE = 24
BRUSH_RADIUS = 8.0

PRESETS = [
    ("coral", 0.0545, 0.0620),
    ("mitosis", 0.0367, 0.0649),
    ("worms", 0.0780, 0.0610),
    ("waves", 0.0140, 0.0450),
    ("solitons", 0.0300, 0.0600),
]

N_STOPS = 5
PALETTES = np.array(
    [
        [[0.00, 0.00, 0.05], [0.10, 0.00, 0.30], [0.80, 0.20, 0.10], [1.00, 0.70, 0.10], [1.00, 1.00, 0.90]],
        [[0.00, 0.02, 0.08], [0.00, 0.20, 0.45], [0.00, 0.60, 0.70], [0.40, 0.90, 0.85], [0.95, 1.00, 1.00]],
        [[0.02, 0.00, 0.05], [0.25, 0.00, 0.40], [0.10, 0.55, 0.20], [0.70, 0.95, 0.20], [1.00, 1.00, 0.75]],
    ],
    dtype=np.float32,
)

u = None
v = None
u_next = None
v_next = None
pixels = None
pal_stops = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global u, v, u_next, v_next, pixels, pal_stops
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    u = ti.field(ti.f32, shape=(N, N))
    v = ti.field(ti.f32, shape=(N, N))
    u_next = ti.field(ti.f32, shape=(N, N))
    v_next = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))
    pal_stops = ti.Vector.field(3, ti.f32, shape=(len(PALETTES), N_STOPS))
    pal_stops.from_numpy(PALETTES)


def seed_pattern(n, size=SEED_SIZE, rng_seed=0, extra_spots=4):
    """Pure numpy: U everywhere, plus a center square of V and a few random spots."""
    u0 = np.ones((n, n), dtype=np.float32)
    v0 = np.zeros((n, n), dtype=np.float32)
    half = size // 2
    c = n // 2
    v0[c - half : c + half, c - half : c + half] = 1.0
    u0[c - half : c + half, c - half : c + half] = 0.5
    rng = np.random.default_rng(rng_seed)
    for _ in range(extra_spots):
        x, y = rng.integers(half, n - half, size=2)
        v0[x - half : x + half, y - half : y + half] = 1.0
        u0[x - half : x + half, y - half : y + half] = 0.5
    return u0, v0


def apply_seed(seed):
    u0, v0 = seed
    u.from_numpy(u0)
    v.from_numpy(v0)


@ti.func
def laplacian(f: ti.template(), i, j):
    side = f[(i + 1) % N, j] + f[(i - 1) % N, j] + f[i, (j + 1) % N] + f[i, (j - 1) % N]
    corner = (
        f[(i + 1) % N, (j + 1) % N]
        + f[(i + 1) % N, (j - 1) % N]
        + f[(i - 1) % N, (j + 1) % N]
        + f[(i - 1) % N, (j - 1) % N]
    )
    return 0.2 * side + 0.05 * corner - f[i, j]


@ti.kernel
def update(feed: ti.f32, kill: ti.f32):
    for i, j in u:
        reaction = u[i, j] * v[i, j] * v[i, j]
        u_next[i, j] = u[i, j] + DT * (DU * laplacian(u, i, j) - reaction + feed * (1.0 - u[i, j]))
        v_next[i, j] = v[i, j] + DT * (DV * laplacian(v, i, j) + reaction - (feed + kill) * v[i, j])


@ti.kernel
def copy_back():
    for i, j in u:
        u[i, j] = u_next[i, j]
        v[i, j] = v_next[i, j]


def step(feed, kill):
    update(feed, kill)
    copy_back()


@ti.kernel
def splat(x: ti.f32, y: ti.f32, radius: ti.f32):
    for i, j in v:
        dx = i - x * N
        dy = j - y * N
        if dx * dx + dy * dy < radius * radius:
            v[i, j] = 1.0
            u[i, j] = 0.5


@ti.kernel
def render(pal: ti.i32):
    for i, j in pixels:
        t = ti.math.clamp(v[i, j] / 0.4, 0.0, 1.0)
        x = t * (N_STOPS - 1)
        s = ti.min(int(x), N_STOPS - 2)
        f = x - s
        pixels[i, j] = pal_stops[pal, s] * (1.0 - f) + pal_stops[pal, s + 1] * f


def main():
    init_sim()
    apply_seed(seed_pattern(N))
    gui = ti.GUI("Gray-Scott — taichi-academy", res=(N, N))
    preset = 0
    pal = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "p":
                pal = (pal + 1) % len(PALETTES)
            elif e.key in "12345":
                preset = int(e.key) - 1
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            splat(mx, my, BRUSH_RADIUS)
        name, feed, kill = PRESETS[preset]
        for _ in range(SUBSTEPS):
            step(feed, kill)
        render(pal)
        gui.set_image(pixels)
        gui.text(f"{name}  F={feed:.4f} k={kill:.4f}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1-5] preset  [r] reseed  [p] palette  paint with mouse", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

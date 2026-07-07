"""Stable fluids: stir a box of incompressible ink with your mouse."""

import numpy as np
import taichi as ti

N = 512
DT = 1.0
JACOBI_ITERS = 40
DYE_DECAY = 0.995
VEL_DECAY = 0.999
BRUSH_RADIUS = 14.0
FORCE_SCALE = 300.0
CURL_STRENGTH = 2.0

DYE_COLORS = [
    ("ember", 1.00, 0.35, 0.10),
    ("sky", 0.15, 0.55, 1.00),
    ("mint", 0.20, 1.00, 0.45),
    ("violet", 0.70, 0.30, 1.00),
    ("gold", 1.00, 0.85, 0.25),
]

vel = None
vel_next = None
dye = None
dye_next = None
pressure = None
pressure_next = None
divergence = None
curl = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, curl, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    vel = ti.Vector.field(2, ti.f32, shape=(N, N))
    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))
    dye = ti.Vector.field(3, ti.f32, shape=(N, N))
    dye_next = ti.Vector.field(3, ti.f32, shape=(N, N))
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))
    curl = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def seed_pattern(n, rng_seed=0, blobs=3):
    """Pure numpy: a few soft ink blobs to start with."""
    dye0 = np.zeros((n, n, 3), dtype=np.float32)
    colors = [(1.0, 0.35, 0.1), (0.15, 0.55, 1.0), (0.2, 1.0, 0.45)]
    rng = np.random.default_rng(rng_seed)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sigma = n / 14.0
    for k in range(blobs):
        cx, cy = rng.integers(n // 4, 3 * n // 4, size=2)
        w = np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (sigma * sigma))
        for ch in range(3):
            dye0[:, :, ch] += w * colors[k % 3][ch]
    return dye0.clip(0.0, 1.0).astype(np.float32)


def apply_seed(dye0):
    dye.from_numpy(dye0)


@ti.func
def sample(f: ti.template(), i, j):
    return f[((i % N) + N) % N, ((j % N) + N) % N]


@ti.func
def bilerp(f: ti.template(), x, y):
    x0 = int(ti.floor(x))
    y0 = int(ti.floor(y))
    fx = x - x0
    fy = y - y0
    a = sample(f, x0, y0)
    b = sample(f, x0 + 1, y0)
    c = sample(f, x0, y0 + 1)
    d = sample(f, x0 + 1, y0 + 1)
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy


@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)


@ti.kernel
def copy_back():
    for i, j in dye:
        dye[i, j] = dye_next[i, j]
        vel[i, j] = vel_next[i, j]


@ti.kernel
def clear_fields():
    for i, j in dye:
        dye[i, j] = ti.Vector([0.0, 0.0, 0.0])
        vel[i, j] = ti.Vector([0.0, 0.0])
        pressure[i, j] = 0.0


@ti.kernel
def splat(x: ti.f32, y: ti.f32, fx: ti.f32, fy: ti.f32, r: ti.f32, g: ti.f32, b: ti.f32):
    for i, j in dye:
        dx = i - x * N
        dy = j - y * N
        w = ti.exp(-(dx * dx + dy * dy) / (BRUSH_RADIUS * BRUSH_RADIUS))
        dye[i, j] += w * ti.Vector([r, g, b])
        vel[i, j] += w * ti.Vector([fx, fy])


@ti.kernel
def decay():
    for i, j in dye:
        dye[i, j] *= DYE_DECAY
        vel[i, j] *= VEL_DECAY


@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0]
            - vel[i, j][0]
            + sample(vel, i, j + 1)[1]
            - vel[i, j][1]
        )


@ti.kernel
def pressure_jacobi():
    for i, j in pressure:
        pressure_next[i, j] = (
            sample(pressure, i + 1, j)
            + sample(pressure, i - 1, j)
            + sample(pressure, i, j + 1)
            + sample(pressure, i, j - 1)
            - divergence[i, j]
        ) * 0.25


@ti.kernel
def copy_pressure():
    for i, j in pressure:
        pressure[i, j] = pressure_next[i, j]


@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector([
            pressure[i, j] - sample(pressure, i - 1, j),
            pressure[i, j] - sample(pressure, i, j - 1),
        ])
        vel[i, j] -= grad


def project():
    compute_divergence()
    for _ in range(JACOBI_ITERS):
        pressure_jacobi()
        copy_pressure()
    subtract_gradient()


@ti.kernel
def compute_curl():
    for i, j in vel:
        curl[i, j] = (
            sample(vel, i + 1, j)[1]
            - sample(vel, i - 1, j)[1]
            - sample(vel, i, j + 1)[0]
            + sample(vel, i, j - 1)[0]
        ) * 0.5


@ti.kernel
def apply_vorticity(strength: ti.f32):
    for i, j in vel:
        grad = ti.Vector([
            ti.abs(sample(curl, i + 1, j)) - ti.abs(sample(curl, i - 1, j)),
            ti.abs(sample(curl, i, j + 1)) - ti.abs(sample(curl, i, j - 1)),
        ]) * 0.5
        n = grad / (grad.norm() + 1e-5)
        vel[i, j] += DT * strength * curl[i, j] * ti.Vector([n[1], -n[0]])


def step(curl_strength):
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    if curl_strength > 0.0:
        compute_curl()
        apply_vorticity(curl_strength)
    project()
    decay()


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.math.clamp(dye[i, j], 0.0, 1.0)


def main():
    init_sim()
    apply_seed(seed_pattern(N))
    gui = ti.GUI("Stable Fluids — taichi-academy", res=(N, N))
    color_idx = 0
    curls_on = True
    pmx, pmy = 0.0, 0.0
    dragging = False
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "c":
                color_idx = (color_idx + 1) % len(DYE_COLORS)
            elif e.key == "v":
                curls_on = not curls_on
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                name, r, g, b = DYE_COLORS[color_idx]
                splat(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE, r, g, b)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False
        step(CURL_STRENGTH if curls_on else 0.0)
        render()
        gui.set_image(pixels)
        name = DYE_COLORS[color_idx][0]
        curls = "on" if curls_on else "off"
        gui.text(f"dye: {name}  curls: {curls}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to stir  [c] color  [v] curls  [r] reset", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

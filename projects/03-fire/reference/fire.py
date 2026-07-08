"""Fire and smoke: heat rises, flames lick, smoke shrouds the glow."""

import numpy as np
import taichi as ti

N = 512
DT = 1.0
BUOYANCY = 0.05
COOLING = 0.985
SMOKE_DECAY = 0.992
VEL_DECAY = 0.99
SOURCE_RADIUS = 40.0
TORCH_RADIUS = 10.0
FORCE_SCALE = 300.0
JACOBI_ITERS = 40
CURL_STRENGTH = 2.0

vel = None
vel_next = None
temp = None
temp_next = None
smoke = None
smoke_next = None
pressure = None
pressure_next = None
divergence = None
curl = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, smoke, smoke_next
    global pressure, pressure_next, divergence, curl, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    vel = ti.Vector.field(2, ti.f32, shape=(N, N))
    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))
    temp = ti.field(ti.f32, shape=(N, N))
    temp_next = ti.field(ti.f32, shape=(N, N))
    smoke = ti.field(ti.f32, shape=(N, N))
    smoke_next = ti.field(ti.f32, shape=(N, N))
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))
    curl = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def seed_pattern(n, rng_seed=0):
    """Pure numpy: one hot ember blob low in the box."""
    rng = np.random.default_rng(rng_seed)
    cx = n // 2 + int(rng.integers(-n // 8, n // 8))
    cy = n // 4
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sigma = n / 16.0
    t0 = np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (sigma * sigma))
    return t0.astype(np.float32)


def apply_seed(t0):
    temp.from_numpy(t0)


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
    for i, j in temp:
        temp[i, j] = temp_next[i, j]
        smoke[i, j] = smoke_next[i, j]
        vel[i, j] = vel_next[i, j]


@ti.kernel
def clear_fields():
    for i, j in temp:
        temp[i, j] = 0.0
        smoke[i, j] = 0.0
        vel[i, j] = ti.Vector([0.0, 0.0])
        pressure[i, j] = 0.0


@ti.kernel
def apply_buoyancy():
    for i, j in vel:
        vel[i, j][1] += DT * BUOYANCY * temp[i, j]


@ti.kernel
def burn_source(t: ti.f32):
    for i, j in temp:
        dx = i - N / 2
        dy = j - 12.0
        flick = 1.0 + 0.35 * ti.sin(0.31 * t + 0.05 * i)
        w = ti.exp(-(dx * dx + dy * dy) / (SOURCE_RADIUS * SOURCE_RADIUS)) * flick
        temp[i, j] = ti.min(temp[i, j] + 0.8 * w, 1.5)
        smoke[i, j] = ti.min(smoke[i, j] + 0.03 * w, 1.0)


@ti.kernel
def torch(x: ti.f32, y: ti.f32, fx: ti.f32, fy: ti.f32):
    for i, j in temp:
        dx = i - x * N
        dy = j - y * N
        w = ti.exp(-(dx * dx + dy * dy) / (TORCH_RADIUS * TORCH_RADIUS))
        temp[i, j] = ti.min(temp[i, j] + 0.9 * w, 1.5)
        smoke[i, j] = ti.min(smoke[i, j] + 0.05 * w, 1.0)
        vel[i, j] += w * ti.Vector([fx, fy])


@ti.kernel
def cool():
    for i, j in temp:
        temp[i, j] *= COOLING
        smoke[i, j] *= SMOKE_DECAY
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
    advect(temp, temp_next)
    advect(smoke, smoke_next)
    advect(vel, vel_next)
    copy_back()
    apply_buoyancy()
    if curl_strength > 0.0:
        compute_curl()
        apply_vorticity(curl_strength)
    project()
    cool()


@ti.kernel
def render():
    for i, j in pixels:
        t = ti.math.clamp(temp[i, j], 0.0, 1.0)
        fire = ti.Vector([1.6 * t, 1.2 * t * t, t * t * t])
        s = smoke[i, j] * 0.25
        pixels[i, j] = ti.math.clamp(fire + ti.Vector([s, s, s]), 0.0, 1.0)


def main():
    init_sim()
    apply_seed(seed_pattern(N))
    gui = ti.GUI("Fire & Smoke — taichi-academy", res=(N, N))
    fire_on = True
    curls_on = True
    frame = 0
    pmx, pmy = 0.0, 0.0
    dragging = False
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on
            elif e.key == "v":
                curls_on = not curls_on
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                torch(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE)
            else:
                torch(mx, my, 0.0, 0.0)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False
        if fire_on:
            burn_source(float(frame))
        step(CURL_STRENGTH if curls_on else 0.0)
        frame += 1
        render()
        gui.set_image(pixels)
        bonfire = "lit" if fire_on else "out"
        curls = "on" if curls_on else "off"
        gui.text(f"bonfire: {bonfire}  curls: {curls}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to torch  [space] bonfire  [v] curls  [r] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

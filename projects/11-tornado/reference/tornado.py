"""Tornado: a self-sustaining vortex in a stable-fluids grid, with debris riding the wind."""

import numpy as np
import taichi as ti

N = 512
DT = 1.0
JACOBI_ITERS = 40
DYE_DECAY = 0.985
VEL_DECAY = 0.97
CURL_STRENGTH = 0.3

CX, CY = N * 0.5, N * 0.5
CORE_R = 60.0
TANGENT_STRENGTH = 1.0
INFLOW_STRENGTH = 0.2

N_DEBRIS = 3000
DRAG = 0.12
DEBRIS_PULL = 0.015
DEBRIS_HOME_R = N * 0.4

STIR_RADIUS = 20.0
STIR_FORCE = 200.0

vel = None
vel_next = None
dye = None
dye_next = None
pressure = None
pressure_next = None
divergence = None
curl = None
dpos = None
dvel = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, curl
    global dpos, dvel
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
    dpos = ti.Vector.field(2, ti.f32, shape=N_DEBRIS)
    dvel = ti.Vector.field(2, ti.f32, shape=N_DEBRIS)


def seed_debris(rng_seed=0):
    """Pure numpy: N_DEBRIS points scattered in a ring around the vortex core."""
    rng = np.random.default_rng(rng_seed)
    ang = rng.uniform(0.0, 2 * np.pi, N_DEBRIS)
    rad = rng.uniform(CORE_R * 1.2, N * 0.45, N_DEBRIS)
    x = CX + rad * np.cos(ang)
    y = CY + rad * np.sin(ang)
    return np.stack([x, y], axis=1).astype(np.float32)


def apply_seed(rng_seed=0):
    dye.fill(0.0)
    vel.fill(0.0)
    pressure.fill(0.0)
    dpos.from_numpy(seed_debris(rng_seed))
    dvel.fill(0.0)


@ti.func
def sample(f: ti.template(), i, j):
    ci = min(max(i, 0), N - 1)
    cj = min(max(j, 0), N - 1)
    return f[ci, cj]


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
def vortex_forcing():
    for i, j in vel:
        rx, ry = float(i) - CX, float(j) - CY
        r = ti.sqrt(rx * rx + ry * ry) + 1e-3
        falloff = r / (r * r + CORE_R * CORE_R) * CORE_R
        tangent = ti.Vector([-ry, rx]) / r
        radial_in = ti.Vector([-rx, -ry]) / r
        vel[i, j] += DT * falloff * (TANGENT_STRENGTH * tangent + INFLOW_STRENGTH * radial_in)


@ti.kernel
def seed_dye():
    for i, j in dye:
        rx, ry = float(i) - CX, float(j) - CY
        r2 = rx * rx + ry * ry
        if r2 < (CORE_R * 1.5) ** 2:
            w = ti.exp(-r2 / (CORE_R * CORE_R))
            dye[i, j] = ti.min(dye[i, j] + 0.02 * w * ti.Vector([0.8, 0.75, 0.6]), 1.0)


@ti.kernel
def stir(mx: ti.f32, my: ti.f32, fx: ti.f32, fy: ti.f32):
    for i, j in vel:
        dx, dy = float(i) - mx * N, float(j) - my * N
        w = ti.exp(-(dx * dx + dy * dy) / (STIR_RADIUS * STIR_RADIUS))
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
            sample(vel, i + 1, j)[0] - vel[i, j][0] + sample(vel, i, j + 1)[1] - vel[i, j][1]
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
        grad = ti.Vector(
            [pressure[i, j] - sample(pressure, i - 1, j), pressure[i, j] - sample(pressure, i, j - 1)]
        )
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
            sample(vel, i + 1, j)[1] - sample(vel, i - 1, j)[1] - sample(vel, i, j + 1)[0] + sample(vel, i, j - 1)[0]
        ) * 0.5


@ti.kernel
def apply_vorticity(strength: ti.f32):
    for i, j in vel:
        grad = (
            ti.Vector(
                [
                    ti.abs(sample(curl, i + 1, j)) - ti.abs(sample(curl, i - 1, j)),
                    ti.abs(sample(curl, i, j + 1)) - ti.abs(sample(curl, i, j - 1)),
                ]
            )
            * 0.5
        )
        n = grad / (grad.norm() + 1e-5)
        vel[i, j] += DT * strength * curl[i, j] * ti.Vector([n[1], -n[0]])


@ti.kernel
def advect_debris():
    for p in dpos:
        fluid_v = bilerp(vel, dpos[p][0], dpos[p][1])
        dvel[p] += (fluid_v - dvel[p]) * DRAG
        offset = dpos[p] - ti.Vector([CX, CY])
        r = offset.norm() + 1e-3
        if r > DEBRIS_HOME_R:
            dvel[p] -= DEBRIS_PULL * (r - DEBRIS_HOME_R) * (offset / r)
        dpos[p] += DT * dvel[p]
        for a in ti.static(range(2)):
            if dpos[p][a] < 0:
                dpos[p][a] = 0
                dvel[p][a] *= -0.5
            if dpos[p][a] >= N:
                dpos[p][a] = N - 1
                dvel[p][a] *= -0.5


def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    compute_curl()
    apply_vorticity(CURL_STRENGTH)
    project()
    decay()
    advect_debris()


@ti.kernel
def render(pixels: ti.template()):
    for i, j in pixels:
        pixels[i, j] = ti.math.clamp(dye[i, j], 0.0, 1.0)


def main():
    init_sim()
    apply_seed()
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))
    gui = ti.GUI("Tornado — taichi-academy", res=N, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            fx, fy = np.random.uniform(-1, 1, 2) * STIR_FORCE
            stir(mx, my, float(fx), float(fy))
        step()
        render(pixels)
        gui.set_image(pixels)
        gui.circles(dpos.to_numpy() / N, radius=1.5, color=0xFFFFFF)
        gui.text("drag to stir  [r] new debris", (0.02, 0.98), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

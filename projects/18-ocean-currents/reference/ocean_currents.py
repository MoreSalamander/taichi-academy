"""Ocean Currents: wind bands + Coriolis + continents turn a fluid box into a climate map."""

import numpy as np
import taichi as ti

N = 256
DT = 1.0
JACOBI = 30
OCEAN_FRACTION = 0.65
WIND = 0.06
CORIOLIS = 0.05
TEMP_RELAX = 0.005
VEL_DECAY = 0.995
PI = 3.14159265

STORM_STRENGTH = 1.2
STORM_RADIUS = 14.0

vel = None
vel_next = None
temp = None
temp_next = None
pressure = None
pressure_next = None
divergence = None
land = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, pressure, pressure_next, divergence, land, pixels
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
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))
    land = ti.field(ti.i32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def resize_bilinear(a, n):
    """Pure numpy: smoothly resize a small square array up to n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1 - f)[:, None] + a[i1] * f[:, None]
    a = a[:, i0] * (1 - f)[None, :] + a[:, i1] * f[None, :]
    return a


def fbm2d(n, rng_seed=0, octaves=5, roughness=0.55):
    """Pure numpy: fractal 2D noise — octaves of noise, each finer and fainter."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n, n), dtype=np.float32)
    amp, res = 1.0, 4
    for _ in range(octaves):
        layer = rng.uniform(0, 1, size=(res, res)).astype(np.float32)
        out += amp * resize_bilinear(layer, n)
        amp *= roughness
        res *= 2
    out -= out.min()
    out /= out.max()
    return out


def seed_continents(n, rng_seed=0):
    """Pure numpy: fbm noise thresholded at a fixed ocean fraction — the land mask."""
    noise = fbm2d(n, rng_seed)
    sea = np.quantile(noise, OCEAN_FRACTION)
    return (noise > sea).astype(np.int32)


def seed_temperature(n):
    """Pure numpy: warm equator, cold poles — the sun's job, one line of latitude math."""
    jj = np.arange(n)
    lat = np.abs(jj - n / 2) / (n / 2)
    return ((1.0 - lat)[None, :] * np.ones((n, n))).astype(np.float32)


def apply_seed(rng_seed=0):
    land.from_numpy(seed_continents(N, rng_seed))
    temp.from_numpy(seed_temperature(N))
    vel.fill(0.0)
    pressure.fill(0.0)


@ti.func
def latitude(j):
    return (j - N / 2.0) / (N / 2.0)


@ti.func
def sample(f: ti.template(), i, j):
    ci = ((i % N) + N) % N
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
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@ti.kernel
def wind_forcing():
    for i, j in vel:
        if land[i, j] == 0:
            zonal = -ti.cos(latitude(j) * 3.0 * PI)
            vel[i, j][0] += DT * WIND * zonal


@ti.kernel
def coriolis():
    for i, j in vel:
        if land[i, j] == 0:
            f = CORIOLIS * latitude(j)
            v = vel[i, j]
            vel[i, j] += DT * f * ti.Vector([v[1], -v[0]])


@ti.kernel
def storm(mx: ti.f32, my: ti.f32):
    ci = mx * N
    cj = my * N
    spin = 1.0
    if latitude(cj) < 0:
        spin = -1.0
    for i, j in vel:
        if land[i, j] == 0:
            dx = i - ci
            dy = j - cj
            r2 = dx * dx + dy * dy
            w = ti.exp(-r2 / (STORM_RADIUS * STORM_RADIUS))
            vel[i, j] += spin * STORM_STRENGTH * w * ti.Vector([-dy, dx]) / STORM_RADIUS


@ti.kernel
def advect_all():
    for i, j in vel:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        vel_next[i, j] = bilerp(vel, x, y)
        temp_next[i, j] = bilerp(temp, x, y)


@ti.kernel
def copy_back():
    for i, j in vel:
        vel[i, j] = vel_next[i, j] * VEL_DECAY
        temp[i, j] = temp_next[i, j]


@ti.kernel
def enforce_land():
    for i, j in vel:
        if land[i, j] == 1:
            vel[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def relax_temp():
    for i, j in temp:
        target = 1.0 - ti.abs(latitude(j))
        temp[i, j] += TEMP_RELAX * (target - temp[i, j])


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
    for _ in range(JACOBI):
        pressure_jacobi()
        copy_pressure()
    subtract_gradient()


@ti.kernel
def render():
    for i, j in pixels:
        if land[i, j] == 1:
            pixels[i, j] = ti.Vector([0.25, 0.22, 0.18])
        else:
            t = ti.math.clamp(temp[i, j], 0.0, 1.0)
            cold = ti.Vector([0.05, 0.15, 0.45])
            warm = ti.Vector([0.9, 0.35, 0.15])
            c = cold * (1 - t) + warm * t
            spd = vel[i, j].norm()
            c += ti.min(spd * 0.5, 0.35) * ti.Vector([1.0, 1.0, 1.0])
            pixels[i, j] = ti.math.clamp(c, 0.0, 1.0)


def step():
    wind_forcing()
    coriolis()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()
    relax_temp()


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Ocean Currents — taichi-academy", res=N, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            storm(mx, my)
        step()
        render()
        gui.set_image(pixels)
        gui.text("click the sea to spawn a storm", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new continents", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

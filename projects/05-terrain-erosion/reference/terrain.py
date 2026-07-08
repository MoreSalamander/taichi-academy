"""Terrain erosion: fractal mountains weathered by rain, rivers, and time."""

import numpy as np
import taichi as ti

N = 512
RAIN = 0.0002
EVAP = 0.004
KC = 1.0
KE = 0.05
KD = 0.05
TALUS = 0.008
THERMAL_RATE = 0.25
RELIEF = 300.0
WATER_VIS = 0.002

h = None
h_next = None
w = None
w_next = None
s = None
s_next = None
flux = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global h, h_next, w, w_next, s, s_next, flux, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    h = ti.field(ti.f32, shape=(N, N))
    h_next = ti.field(ti.f32, shape=(N, N))
    w = ti.field(ti.f32, shape=(N, N))
    w_next = ti.field(ti.f32, shape=(N, N))
    s = ti.field(ti.f32, shape=(N, N))
    s_next = ti.field(ti.f32, shape=(N, N))
    flux = ti.Vector.field(4, ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def resize_bilinear(a, n):
    """Pure numpy: smoothly resize a small square array up to n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1.0 - f)[:, None] + a[i1] * f[:, None]
    a = a[:, i0] * (1.0 - f)[None, :] + a[:, i1] * f[None, :]
    return a


def fbm_terrain(n, rng_seed=0, octaves=7, roughness=0.55):
    """Pure numpy: fractal terrain — octaves of noise, each finer and fainter."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n, n), dtype=np.float32)
    amp = 1.0
    res = 4
    for _ in range(octaves):
        layer = rng.uniform(-1.0, 1.0, size=(res, res)).astype(np.float32)
        out += amp * resize_bilinear(layer, n)
        amp *= roughness
        res *= 2
    out -= out.min()
    out /= out.max()
    return out.astype(np.float32)


def apply_seed(h0):
    h.from_numpy(h0)


DI = (1, -1, 0, 0)
DJ = (0, 0, 1, -1)
OPP = (1, 0, 3, 2)


@ti.kernel
def rain():
    for i, j in w:
        w[i, j] += RAIN


@ti.kernel
def compute_flux():
    for i, j in flux:
        total_h = h[i, j] + w[i, j]
        f = ti.Vector([0.0, 0.0, 0.0, 0.0])
        total = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                d = total_h - (h[ni, nj] + w[ni, nj])
                if d > 0.0:
                    f[k] = d
                    total += d
        scale = 0.0
        if total > 1e-9:
            scale = ti.min(w[i, j], 0.5 * total) / total
        flux[i, j] = f * scale


@ti.kernel
def erode_deposit():
    for i, j in h:
        flow = flux[i, j].sum()
        cap = KC * flow
        if s[i, j] < cap:
            amount = ti.min(KE * (cap - s[i, j]), h[i, j])
            h[i, j] -= amount
            s[i, j] += amount
        else:
            amount = KD * (s[i, j] - cap)
            h[i, j] += amount
            s[i, j] -= amount


@ti.kernel
def move_water():
    for i, j in w:
        inflow = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                inflow += flux[ni, nj][OPP[k]]
        w_next[i, j] = (w[i, j] - flux[i, j].sum() + inflow) * (1.0 - EVAP)


@ti.kernel
def move_sediment():
    for i, j in s:
        kept = s[i, j]
        if w[i, j] > 1e-9:
            kept = s[i, j] * (1.0 - flux[i, j].sum() / w[i, j])
        arriving = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                if w[ni, nj] > 1e-9:
                    arriving += s[ni, nj] * flux[ni, nj][OPP[k]] / w[ni, nj]
        s_next[i, j] = kept + arriving


@ti.kernel
def copy_wet():
    for i, j in w:
        w[i, j] = w_next[i, j]
        s[i, j] = s_next[i, j]


@ti.kernel
def thermal():
    for i, j in h:
        delta = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                d = h[i, j] - h[ni, nj]
                if d > TALUS:
                    delta -= (d - TALUS) * 0.5 * THERMAL_RATE
                elif d < -TALUS:
                    delta += (-d - TALUS) * 0.5 * THERMAL_RATE
        h_next[i, j] = h[i, j] + delta


@ti.kernel
def copy_height():
    for i, j in h:
        h[i, j] = h_next[i, j]


def step():
    rain()
    compute_flux()
    erode_deposit()
    move_water()
    move_sediment()
    copy_wet()
    thermal()
    copy_height()


@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t


@ti.kernel
def render():
    for i, j in pixels:
        ip = ti.min(i + 1, N - 1)
        jp = ti.min(j + 1, N - 1)
        dhdx = (h[ip, j] - h[i, j]) * RELIEF
        dhdy = (h[i, jp] - h[i, j]) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.35 + 0.65 * normal.dot(light), 0.0, 1.0)
        hh = h[i, j]
        c = band(ti.Vector([0.76, 0.70, 0.50]), ti.Vector([0.30, 0.55, 0.25]), hh, 0.05, 0.35)
        c = band(c, ti.Vector([0.45, 0.42, 0.40]), hh, 0.45, 0.75)
        c = band(c, ti.Vector([0.95, 0.95, 0.98]), hh, 0.78, 0.92)
        wet = ti.math.clamp(w[i, j] / WATER_VIS * 0.2, 0.0, 0.85)
        c = c * (1.0 - wet) + ti.Vector([0.15, 0.35, 0.70]) * wet
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)


def main():
    init_sim()
    apply_seed(fbm_terrain(N))
    gui = ti.GUI("Terrain Erosion — taichi-academy", res=(N, N))
    raining = True
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))
                w.fill(0.0)
                s.fill(0.0)
            elif e.key == ti.GUI.SPACE:
                raining = not raining
        if raining:
            step()
        render()
        gui.set_image(pixels)
        sky = "raining" if raining else "paused"
        gui.text(f"weather: {sky}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[space] rain on/off  [r] new mountains", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

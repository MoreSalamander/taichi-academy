"""Code SOT for project 05 — terrain erosion.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 05-terrain-erosion`.

Evolutions: render climbs grayscale→hillshade→color bands→water tint; the
'r' event branch grows a fill line as each new wet field arrives; init_sim
gains fields per chapter; step accretes the erosion pipeline call by call.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="05-terrain-erosion",
    default_file="terrain.py",
    reference={"terrain.py": PROJECT_DIR / "reference" / "terrain.py"},
    chapter_steps={1: 5, 2: 3, 3: 7, 4: 3, 5: 3},
)
frag = SPEC.frag

# --- module head -----------------------------------------------------------------

frag(((1, 1), '"""Terrain erosion: fractal mountains weathered by rain, rivers, and time."""'))
frag(((1, 3), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants ---------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((3, 2), "RAIN = 0.0002\nEVAP = 0.004"))
frag(((4, 1), "KC = 1.0\nKE = 0.05\nKD = 0.05"))
frag(((5, 1), "TALUS = 0.008\nTHERMAL_RATE = 0.25"))
frag(((2, 1), "RELIEF = 300.0"))
frag(((3, 7), "WATER_VIS = 0.002"))

# --- module-level fields --------------------------------------------------------------

frag(
    ((1, 2), "h = None\npixels = None"),
    ((3, 1), "h = None\nw = None\nw_next = None\nflux = None\npixels = None"),
    ((4, 1), "h = None\nw = None\nw_next = None\ns = None\ns_next = None\nflux = None\npixels = None"),
    ((5, 1), "h = None\nh_next = None\nw = None\nw_next = None\ns = None\ns_next = None\nflux = None\npixels = None"),
)

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global h, pixels"),
    ((3, 1), f"def init_sim(arch=None):\n{DOC}\n    global h, w, w_next, flux, pixels"),
    ((4, 1), f"def init_sim(arch=None):\n{DOC}\n    global h, w, w_next, s, s_next, flux, pixels"),
    ((5, 1), f"def init_sim(arch=None):\n{DOC}\n    global h, h_next, w, w_next, s, s_next, flux, pixels"),
)
frag(
    (
        (1, 2),
        "    if arch is None:\n"
        "        try:\n"
        "            ti.init(arch=ti.gpu)\n"
        "        except Exception:\n"
        "            ti.init(arch=ti.cpu)\n"
        "    else:\n"
        "        ti.init(arch=arch)",
    )
)
frag(((1, 2), "    h = ti.field(ti.f32, shape=(N, N))"))
frag(((5, 1), "    h_next = ti.field(ti.f32, shape=(N, N))"))
frag(((3, 1), "    w = ti.field(ti.f32, shape=(N, N))\n    w_next = ti.field(ti.f32, shape=(N, N))"))
frag(((4, 1), "    s = ti.field(ti.f32, shape=(N, N))\n    s_next = ti.field(ti.f32, shape=(N, N))"))
frag(((3, 1), "    flux = ti.Vector.field(4, ti.f32, shape=(N, N))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- pure numpy generation --------------------------------------------------------------

RESIZE = '''def resize_bilinear(a, n):
    """Pure numpy: smoothly resize a small square array up to n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1.0 - f)[:, None] + a[i1] * f[:, None]
    a = a[:, i0] * (1.0 - f)[None, :] + a[:, i1] * f[None, :]
    return a'''

frag(((1, 3), RESIZE))

FBM = '''def fbm_terrain(n, rng_seed=0, octaves=7, roughness=0.55):
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
    return out.astype(np.float32)'''

frag(((1, 4), FBM))
frag(((1, 4), "def apply_seed(h0):\n    h.from_numpy(h0)"))

# --- neighbor bookkeeping ----------------------------------------------------------------

frag(((3, 1), "DI = (1, -1, 0, 0)\nDJ = (0, 0, 1, -1)\nOPP = (1, 0, 3, 2)"))

# --- water -----------------------------------------------------------------------------

frag(((3, 2), "@ti.kernel\ndef rain():\n    for i, j in w:\n        w[i, j] += RAIN"))

FLUX = """@ti.kernel
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
        flux[i, j] = f * scale"""

frag(((3, 3), FLUX))

ERODE = """@ti.kernel
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
            s[i, j] -= amount"""

frag(((4, 2), ERODE))

MOVE_WATER = """@ti.kernel
def move_water():
    for i, j in w:
        inflow = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                inflow += flux[ni, nj][OPP[k]]
        w_next[i, j] = (w[i, j] - flux[i, j].sum() + inflow) * (1.0 - EVAP)"""

frag(((3, 4), MOVE_WATER))

MOVE_SED = """@ti.kernel
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
        s_next[i, j] = kept + arriving"""

frag(((4, 3), MOVE_SED))

frag(
    ((3, 4), "@ti.kernel\ndef copy_wet():\n    for i, j in w:\n        w[i, j] = w_next[i, j]"),
    ((4, 3), "@ti.kernel\ndef copy_wet():\n    for i, j in w:\n        w[i, j] = w_next[i, j]\n        s[i, j] = s_next[i, j]"),
)

# --- thermal ---------------------------------------------------------------------------

THERMAL = """@ti.kernel
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
        h_next[i, j] = h[i, j] + delta"""

frag(((5, 1), THERMAL))
frag(((5, 1), "@ti.kernel\ndef copy_height():\n    for i, j in h:\n        h[i, j] = h_next[i, j]"))

# --- the tick ---------------------------------------------------------------------------

frag(
    ((3, 5), "def step():\n    rain()\n    compute_flux()\n    move_water()\n    copy_wet()"),
    ((4, 2), "def step():\n    rain()\n    compute_flux()\n    erode_deposit()\n    move_water()\n    copy_wet()"),
    ((4, 3), "def step():\n    rain()\n    compute_flux()\n    erode_deposit()\n    move_water()\n    move_sediment()\n    copy_wet()"),
    ((5, 2), "def step():\n    rain()\n    compute_flux()\n    erode_deposit()\n    move_water()\n    move_sediment()\n    copy_wet()\n    thermal()\n    copy_height()"),
)

# --- rendering --------------------------------------------------------------------------

BAND = """@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t"""

frag(((2, 2), BAND))

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        hh = h[i, j]
        pixels[i, j] = ti.Vector([hh, hh, hh])"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        ip = ti.min(i + 1, N - 1)
        jp = ti.min(j + 1, N - 1)
        dhdx = (h[ip, j] - h[i, j]) * RELIEF
        dhdy = (h[i, jp] - h[i, j]) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.35 + 0.65 * normal.dot(light), 0.0, 1.0)
        pixels[i, j] = ti.Vector([shade, shade, shade])"""

RENDER_V3 = """@ti.kernel
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
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)"""

RENDER_V4 = """@ti.kernel
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
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)"""

frag(((1, 5), RENDER_V1), ((2, 1), RENDER_V2), ((2, 2), RENDER_V3), ((3, 7), RENDER_V4))

# --- main loop (ordered sub-fragments) --------------------------------------------------------

frag(((1, 5), "def main():\n    init_sim()\n    apply_seed(fbm_terrain(N))"))
frag(((1, 5), '    gui = ti.GUI("Terrain Erosion — taichi-academy", res=(N, N))'))
frag(((3, 5), "    raining = True"))
frag(((1, 5), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))
                w.fill(0.0)
            elif e.key == ti.GUI.SPACE:
                raining = not raining'''

EVENTS_V4 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))
                w.fill(0.0)
                s.fill(0.0)
            elif e.key == ti.GUI.SPACE:
                raining = not raining'''

frag(((1, 5), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 6), EVENTS_V3), ((4, 3), EVENTS_V4))

frag(((3, 5), "        if raining:\n            step()"))

SHOW_V1 = """        render()
        gui.set_image(pixels)
        gui.show()"""

SHOW_V2 = '''        render()
        gui.set_image(pixels)
        sky = "raining" if raining else "paused"
        gui.text(f"weather: {sky}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[space] rain on/off  [r] new mountains", (0.02, 0.94), color=0xAAAAAA)
        gui.show()'''

frag(((1, 5), SHOW_V1), ((5, 3), SHOW_V2))

frag(((1, 5), 'if __name__ == "__main__":\n    main()'))

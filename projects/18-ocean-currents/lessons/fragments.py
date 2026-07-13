"""Code SOT for project 18 — ocean currents.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 18-ocean-currents`.

Evolutions: step() accretes the climate machinery — wind+advection+projection
+land enforcement (chapter 2), +temperature relaxation (chapter 2's last step),
+coriolis inserted after wind (chapter 3). The storm kernel and its mouse
wiring arrive in chapter 4. The fluid transport core (sample/bilerp/advect/
projection) is projects 02/11's, with sample() changed to wrap-x/clamp-y —
longitude wraps around the globe, latitude ends at the poles.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="18-ocean-currents",
    default_file="ocean_currents.py",
    reference={"ocean_currents.py": PROJECT_DIR / "reference" / "ocean_currents.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Ocean Currents: wind bands + Coriolis + continents turn a fluid box into a climate map."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "N = 256"))
frag(((2, 1), "DT = 1.0"))
frag(((2, 2), "JACOBI = 30"))
frag(((1, 3), "OCEAN_FRACTION = 0.65"))
frag(((2, 1), "WIND = 0.06"))
frag(((3, 1), "CORIOLIS = 0.05"))
frag(((2, 3), "TEMP_RELAX = 0.005"))
frag(((2, 1), "VEL_DECAY = 0.995"))
frag(((2, 1), "PI = 3.14159265"))

frag(((4, 1), "STORM_STRENGTH = 1.2"))
frag(((4, 1), "STORM_RADIUS = 14.0"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "vel = None"))
frag(((2, 1), "vel_next = None"))
frag(((1, 2), "temp = None"))
frag(((2, 1), "temp_next = None"))
frag(((2, 2), "pressure = None"))
frag(((2, 2), "pressure_next = None"))
frag(((2, 2), "divergence = None"))
frag(((1, 2), "land = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global vel, temp, land, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global vel, vel_next, temp, temp_next, land, pixels"),
    (
        (2, 2),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global vel, vel_next, temp, temp_next, pressure, pressure_next, divergence, land, pixels",
    ),
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
frag(((1, 2), "    vel = ti.Vector.field(2, ti.f32, shape=(N, N))"))
frag(((2, 1), "    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))"))
frag(((1, 2), "    temp = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 1), "    temp_next = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 2), "    pressure = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 2), "    pressure_next = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 2), "    divergence = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    land = ti.field(ti.i32, shape=(N, N))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- pure numpy generation ---------------------------------------------------------

RESIZE = '''def resize_bilinear(a, n):
    """Pure numpy: smoothly resize a small square array up to n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1 - f)[:, None] + a[i1] * f[:, None]
    a = a[:, i0] * (1 - f)[None, :] + a[:, i1] * f[None, :]
    return a'''

frag(((1, 3), RESIZE))

FBM2D = '''def fbm2d(n, rng_seed=0, octaves=5, roughness=0.55):
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
    return out'''

frag(((1, 3), FBM2D))

SEED_CONTINENTS = '''def seed_continents(n, rng_seed=0):
    """Pure numpy: fbm noise thresholded at a fixed ocean fraction — the land mask."""
    noise = fbm2d(n, rng_seed)
    sea = np.quantile(noise, OCEAN_FRACTION)
    return (noise > sea).astype(np.int32)'''

frag(((1, 3), SEED_CONTINENTS))

SEED_TEMPERATURE = '''def seed_temperature(n):
    """Pure numpy: warm equator, cold poles — the sun's job, one line of latitude math."""
    jj = np.arange(n)
    lat = np.abs(jj - n / 2) / (n / 2)
    return ((1.0 - lat)[None, :] * np.ones((n, n))).astype(np.float32)'''

frag(((1, 3), SEED_TEMPERATURE))

APPLY_SEED = """def apply_seed(rng_seed=0):
    land.from_numpy(seed_continents(N, rng_seed))
    temp.from_numpy(seed_temperature(N))
    vel.fill(0.0)
    pressure.fill(0.0)"""

APPLY_SEED_V1 = """def apply_seed(rng_seed=0):
    land.from_numpy(seed_continents(N, rng_seed))
    temp.from_numpy(seed_temperature(N))
    vel.fill(0.0)"""

frag(((1, 3), APPLY_SEED_V1), ((2, 2), APPLY_SEED))

frag(((1, 3), "@ti.func\ndef latitude(j):\n    return (j - N / 2.0) / (N / 2.0)"))

SAMPLE = """@ti.func
def sample(f: ti.template(), i, j):
    ci = ((i % N) + N) % N
    cj = min(max(j, 0), N - 1)
    return f[ci, cj]"""

frag(((2, 1), SAMPLE))

BILERP = """@ti.func
def bilerp(f: ti.template(), x, y):
    x0 = int(ti.floor(x))
    y0 = int(ti.floor(y))
    fx = x - x0
    fy = y - y0
    a = sample(f, x0, y0)
    b = sample(f, x0 + 1, y0)
    c = sample(f, x0, y0 + 1)
    d = sample(f, x0 + 1, y0 + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy"""

frag(((2, 1), BILERP))

WIND_FORCING = """@ti.kernel
def wind_forcing():
    for i, j in vel:
        if land[i, j] == 0:
            zonal = -ti.cos(latitude(j) * 3.0 * PI)
            vel[i, j][0] += DT * WIND * zonal"""

frag(((2, 1), WIND_FORCING))

CORIOLIS_K = """@ti.kernel
def coriolis():
    for i, j in vel:
        if land[i, j] == 0:
            f = CORIOLIS * latitude(j)
            v = vel[i, j]
            vel[i, j] += DT * f * ti.Vector([v[1], -v[0]])"""

frag(((3, 1), CORIOLIS_K))

STORM = """@ti.kernel
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
            vel[i, j] += spin * STORM_STRENGTH * w * ti.Vector([-dy, dx]) / STORM_RADIUS"""

frag(((4, 1), STORM))

ADVECT_ALL = """@ti.kernel
def advect_all():
    for i, j in vel:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        vel_next[i, j] = bilerp(vel, x, y)
        temp_next[i, j] = bilerp(temp, x, y)"""

frag(((2, 1), ADVECT_ALL))

COPY_BACK = """@ti.kernel
def copy_back():
    for i, j in vel:
        vel[i, j] = vel_next[i, j] * VEL_DECAY
        temp[i, j] = temp_next[i, j]"""

frag(((2, 1), COPY_BACK))

ENFORCE_LAND = """@ti.kernel
def enforce_land():
    for i, j in vel:
        if land[i, j] == 1:
            vel[i, j] = ti.Vector([0.0, 0.0])"""

frag(((2, 1), ENFORCE_LAND))

RELAX_TEMP = """@ti.kernel
def relax_temp():
    for i, j in temp:
        target = 1.0 - ti.abs(latitude(j))
        temp[i, j] += TEMP_RELAX * (target - temp[i, j])"""

frag(((2, 3), RELAX_TEMP))

DIVERGENCE = """@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0] - vel[i, j][0] + sample(vel, i, j + 1)[1] - vel[i, j][1]
        )"""

frag(((2, 2), DIVERGENCE))

JACOBI_K = """@ti.kernel
def pressure_jacobi():
    for i, j in pressure:
        pressure_next[i, j] = (
            sample(pressure, i + 1, j)
            + sample(pressure, i - 1, j)
            + sample(pressure, i, j + 1)
            + sample(pressure, i, j - 1)
            - divergence[i, j]
        ) * 0.25"""

frag(((2, 2), JACOBI_K))

frag(((2, 2), "@ti.kernel\ndef copy_pressure():\n    for i, j in pressure:\n        pressure[i, j] = pressure_next[i, j]"))

SUBTRACT = """@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector(
            [pressure[i, j] - sample(pressure, i - 1, j), pressure[i, j] - sample(pressure, i, j - 1)]
        )
        vel[i, j] -= grad"""

frag(((2, 2), SUBTRACT))

frag(((2, 2), "def project():\n    compute_divergence()\n    for _ in range(JACOBI):\n        pressure_jacobi()\n        copy_pressure()\n    subtract_gradient()"))

RENDER = """@ti.kernel
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
            pixels[i, j] = ti.math.clamp(c, 0.0, 1.0)"""

frag(((1, 3), RENDER))

# --- the tick ----------------------------------------------------------------------

STEP_V1 = """def step():
    wind_forcing()
    advect_all()
    copy_back()
    enforce_land()"""

STEP_V2 = """def step():
    wind_forcing()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()"""

STEP_V3 = """def step():
    wind_forcing()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()
    relax_temp()"""

STEP_V4 = """def step():
    wind_forcing()
    coriolis()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()
    relax_temp()"""

frag(((2, 1), STEP_V1), ((2, 2), STEP_V2), ((2, 3), STEP_V3), ((3, 2), STEP_V4))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Ocean Currents — taichi-academy", res=N, background_color=0x0A0A12)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((4, 2), EVENTS_V2))

STORM_WIRE = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            storm(mx, my)"""

frag(((4, 1), STORM_WIRE))

frag(((2, 3), "        step()"))
frag(((1, 3), "        render()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((4, 2), '        gui.text("click the sea to spawn a storm", (0.02, 0.98), color=0xFFFFFF)'))
frag(((4, 2), '        gui.text("[r] new continents", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

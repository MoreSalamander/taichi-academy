"""Code SOT for project 29 — earth simulator.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 29-earth-simulator`.

Arc: chapter 1 is the energy balance alone — sunlight in by latitude and albedo, heat out by
radiation — giving a static, far-too-steep equator-to-pole gradient. Chapter 2 adds the moving
parts: heat diffusion and prevailing winds (poleward transport that flattens the gradient) plus
the axial-tilt seasonal cycle. Chapter 3 adds the water cycle — evaporation, rain, and the
vegetation it grows — the finished living planet.

diffuse/advect/moisture are keyed to later chapters even though they sit mid-file, so the tick
(step) grows across all three chapters.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="29-earth-simulator",
    default_file="earth_simulator.py",
    reference={"earth_simulator.py": PROJECT_DIR / "reference" / "earth_simulator.py"},
    chapter_steps={1: 6, 2: 2, 3: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Earth Simulator: an energy-balance climate on a lat-lon grid — sunlight, radiation, winds,\nice-albedo feedback, and a water cycle conspire into climate bands, ice caps, and seasons."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants + fields --------------------------------------------------------------

frag((
    (1, 2),
    "W, H = 256, 128          # longitude x latitude\n"
    "YEAR = 240.0             # steps per orbit\n"
    "TILT = 23.5 * np.pi / 180\n"
    "SOLAR = 350.0            # peak insolation scale\n"
    "A_OLR = 193.0            # outgoing-radiation offset (W/m2); the greenhouse knob (lower = warmer)\n"
    "B_OLR = 2.2              # outgoing radiation per degC\n"
    "C_OCEAN = 80.0           # heat capacity — ocean is a slow flywheel\n"
    "C_LAND = 20.0            # land heats and cools fast\n"
    "DIFF = 0.22              # heat diffusion per pass (2D explicit limit is 0.25)\n"
    "DIFF_ITERS = 6           # diffusion passes per step -> strong poleward transport\n"
    "WIND_AMP = 0.5           # zonal wind speed (cells/step); CFL keeps it < 1\n"
    "FREEZE = -2.0            # below this, a cell counts as ice\n"
    "ALB_OCEAN, ALB_LAND, ALB_ICE = 0.08, 0.25, 0.42  # weak ice contrast avoids a snowball runaway",
))

for _name in ("T", "T2", "land", "moist", "moist2", "veg", "pixels", "clock"):
    frag(((1, 2), f"{_name} = None"))

# --- init ----------------------------------------------------------------------------

INIT_SIM = '''def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global T, T2, land, moist, moist2, veg, pixels, clock
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    T = ti.field(ti.f32, shape=(W, H))
    T2 = ti.field(ti.f32, shape=(W, H))
    land = ti.field(ti.i32, shape=(W, H))
    moist = ti.field(ti.f32, shape=(W, H))
    moist2 = ti.field(ti.f32, shape=(W, H))
    veg = ti.field(ti.f32, shape=(W, H))
    pixels = ti.Vector.field(3, ti.f32, shape=(W, H))
    clock = ti.field(ti.f32, shape=())'''

frag(((1, 3), INIT_SIM))

# --- geography + seed ----------------------------------------------------------------

MAKE_LAND = '''def make_land(seed=3):
    """Pure numpy: low-frequency noise thresholded into continents, with open polar oceans."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((W // 16, H // 16)).astype(np.float32)
    big = np.kron(base, np.ones((16, 16), np.float32))[:W, :H]
    for _ in range(4):
        big = 0.25 * (np.roll(big, 1, 0) + np.roll(big, -1, 0) + np.roll(big, 1, 1) + np.roll(big, -1, 1))
    m = (big > 0.15).astype(np.int32)
    m[:, :6] = 0
    m[:, -6:] = 0
    return m'''

frag(((1, 4), MAKE_LAND))

SEED_FIELDS = '''@ti.kernel
def _seed_fields():
    for i, j in T:
        lat = lat_of(j)
        T[i, j] = 30.0 - 60.0 * (lat / (0.5 * np.pi)) ** 2
        moist[i, j] = 0.0
        veg[i, j] = 0.0'''

frag(((1, 4), SEED_FIELDS))

APPLY_SEED = '''def apply_seed(seed=3):
    """Lay down continents and a warm-equator/cold-pole starting climate; reset the calendar."""
    land.from_numpy(make_land(seed))
    _seed_fields()
    clock[None] = 0.0'''

frag(((1, 4), APPLY_SEED))

# --- geometry + surface properties ---------------------------------------------------

frag(((1, 5), "@ti.func\ndef lat_of(j):\n    return (ti.cast(j, ti.f32) / (H - 1) - 0.5) * np.pi"))

ALBEDO = '''@ti.func
def albedo(i, j):
    base = ALB_LAND if land[i, j] == 1 else ALB_OCEAN
    # ice albedo ramps in smoothly from +2C down to -10C — a soft feedback, not a hard tip
    icefrac = ti.max(0.0, ti.min((2.0 - T[i, j]) / 12.0, 1.0))
    return base + icefrac * (ALB_ICE - base)'''

frag(((1, 5), ALBEDO))

ZONAL_WIND = '''@ti.func
def zonal_wind(lat):
    # easterly trade winds in the tropics, westerlies in the mid-latitudes
    return -WIND_AMP * ti.cos(3.0 * lat)'''

frag(((2, 1), ZONAL_WIND))

# --- the energy balance --------------------------------------------------------------

RADIATE = '''@ti.kernel
def radiate_step(decl: ti.f32, a_olr: ti.f32):
    """Absorb sunlight (by latitude, season, and albedo), radiate heat to space, update temperature."""
    for i, j in T:
        s = ti.max(0.0, ti.cos(lat_of(j) - decl))     # noon sun height at this latitude
        absorbed = SOLAR * s * (1.0 - albedo(i, j))
        olr = a_olr + B_OLR * T[i, j]
        cap = C_LAND if land[i, j] == 1 else C_OCEAN
        T[i, j] = T[i, j] + (absorbed - olr) / cap'''

frag(((1, 5), RADIATE))

# --- transport: diffusion + winds (chapter 2) ----------------------------------------

DIFFUSE = '''@ti.kernel
def diffuse_step():
    """One pass of heat diffusion — the atmosphere and oceans smearing warmth toward the poles."""
    for i, j in T:
        jm = ti.max(j - 1, 0)
        jp = ti.min(j + 1, H - 1)
        im = (i - 1 + W) % W
        ip = (i + 1) % W
        lap = T[im, j] + T[ip, j] + T[i, jm] + T[i, jp] - 4.0 * T[i, j]
        T2[i, j] = T[i, j] + DIFF * lap
    for i, j in T:
        T[i, j] = T2[i, j]'''

frag(((2, 1), DIFFUSE))

ADVECT = '''@ti.kernel
def advect_step():
    """Prevailing winds carry heat around each latitude circle (upwind, longitude wraps)."""
    for i, j in T:
        im = (i - 1 + W) % W
        ip = (i + 1) % W
        u = zonal_wind(lat_of(j))
        adv = 0.0
        if u > 0:
            adv = -u * (T[i, j] - T[im, j])
        else:
            adv = -u * (T[ip, j] - T[i, j])
        T2[i, j] = T[i, j] + adv
    for i, j in T:
        T[i, j] = T2[i, j]'''

frag(((2, 1), ADVECT))

# --- the water cycle (chapter 3) -----------------------------------------------------

MOISTURE = '''@ti.kernel
def moisture_step():
    """Oceans evaporate, winds carry vapor, it rains where the air is over-saturated, and rain
    plus warmth grows vegetation on land."""
    for i, j in moist:
        u = zonal_wind(lat_of(j))
        im = (i - 1 + W) % W
        ip = (i + 1) % W
        evap = 0.0
        if land[i, j] == 0 and T[i, j] > 0.0:
            evap = 0.04 * T[i, j]
        cap = ti.max(0.0, 2.0 + 0.2 * T[i, j])         # warm air holds more vapor
        adv = 0.0
        if u > 0:
            adv = -u * (moist[i, j] - moist[im, j])
        else:
            adv = -u * (moist[ip, j] - moist[i, j])
        m = moist[i, j] + evap + adv
        rain = 0.0
        if m > cap:
            rain = 0.4 * (m - cap)
            m -= rain
        moist2[i, j] = ti.max(m, 0.0)
        if land[i, j] == 1:
            suit = 0.0
            if 2.0 < T[i, j] < 42.0:
                suit = ti.min(rain * 6.0, 1.0)
            veg[i, j] += 0.03 * (suit - veg[i, j])
    for i, j in moist:
        moist[i, j] = moist2[i, j]'''

frag(((3, 1), MOISTURE))

# --- the tick: three versions --------------------------------------------------------

STEP_V1 = """def step(a_olr=A_OLR):
    radiate_step(0.0, a_olr)"""

STEP_V2 = """def step(a_olr=A_OLR):
    decl = TILT * float(np.sin(2 * np.pi * clock[None] / YEAR))
    radiate_step(decl, a_olr)
    for _ in range(DIFF_ITERS):
        diffuse_step()
    advect_step()
    clock[None] += 1.0"""

STEP_V3 = """def step(a_olr=A_OLR):
    decl = TILT * float(np.sin(2 * np.pi * clock[None] / YEAR))
    radiate_step(decl, a_olr)
    for _ in range(DIFF_ITERS):
        diffuse_step()
    advect_step()
    moisture_step()
    clock[None] += 1.0"""

frag(((1, 6), STEP_V1), ((2, 2), STEP_V2), ((3, 2), STEP_V3))

# --- probes --------------------------------------------------------------------------

frag(((1, 6), 'def ice_fraction():\n    """Pure numpy: fraction of the surface frozen (below FREEZE)."""\n    return float((T.to_numpy() < FREEZE).mean())'))
frag(((1, 6), 'def band_temp(j0, j1):\n    """Pure numpy: mean temperature over a band of latitudes."""\n    return float(T.to_numpy()[:, j0:j1].mean())'))

# --- render --------------------------------------------------------------------------

RENDER = '''@ti.kernel
def render():
    for i, j in pixels:
        t = T[i, j]
        col = ti.Vector([0.9, 0.93, 0.97])                # ice
        if t >= FREEZE:
            if land[i, j] == 1:
                green = ti.Vector([0.25, 0.55, 0.2])
                desert = ti.Vector([0.7, 0.62, 0.4])
                v = veg[i, j]
                col = desert * (1.0 - v) + green * v
            else:
                warm = ti.Vector([0.1, 0.5, 0.75])
                cold = ti.Vector([0.03, 0.1, 0.4])
                f = ti.max(ti.min((t + 5.0) / 35.0, 1.0), 0.0)
                col = cold * (1.0 - f) + warm * f
        pixels[i, j] = col'''

frag(((1, 6), RENDER))

# --- main ----------------------------------------------------------------------------

MAIN = '''def main():
    init_sim()
    apply_seed()
    a_olr = A_OLR
    gui = ti.GUI("Earth Simulator — taichi-academy", res=(W, H), background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "=":
                a_olr = max(a_olr - 3.0, 170.0)   # more CO2 -> warmer
            elif e.key == "-":
                a_olr = min(a_olr + 3.0, 215.0)   # less CO2 -> colder
            elif e.key == "r":
                apply_seed(np.random.randint(1_000_000))
                a_olr = A_OLR
        step(a_olr)
        render()
        gui.set_image(pixels)
        day = clock[None] % YEAR
        eq = band_temp(H // 2 - 4, H // 2 + 4)
        gui.text(f"day {day:.0f}/{YEAR:.0f}   equator {eq:.0f}C   ice {ice_fraction() * 100:.0f}%   "
                 f"greenhouse [=/-]  [r] new world", (0.02, 0.98), color=0xFFFFFF)
        gui.show()'''

frag(((1, 6), MAIN))

frag(((1, 6), 'if __name__ == "__main__":\n    main()'))

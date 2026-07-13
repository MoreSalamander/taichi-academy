"""Code SOT for project 16 — solar system.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 16-solar-system`.

Evolutions: the integrator is the star of the show — chapter 2 first ships a
naive euler_step whose orbits visibly spiral outward, then REPLACES it with
leapfrog in the next step (euler_step is deleted; the lesson keeps its lesson).
apply_seed grows from planets-only to planets+belt (chapter 3) to
planets+belt+comets (chapter 3 step 2). render() has two versions: bodies-only,
then +the sun glow and per-population styling.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="16-solar-system",
    default_file="solar_system.py",
    reference={"solar_system.py": PROJECT_DIR / "reference" / "solar_system.py"},
    chapter_steps={1: 3, 2: 3, 3: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Solar System: real 1/r^2 gravity, a leapfrog integrator, and orbits that actually hold."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "GM = 1.0"))
frag(((2, 1), "DT = 0.002"))
frag(((2, 3), "SUBSTEPS = 4"))

frag(((1, 2), "N_PLANETS = 6"))
frag(((3, 1), "N_BELT = 4000"))
frag(((3, 2), "N_COMETS = 24"))
frag(
    ((1, 2), "N = N_PLANETS"),
    ((3, 1), "N = N_PLANETS + N_BELT"),
    ((3, 2), "N = N_PLANETS + N_BELT + N_COMETS"),
)
frag(((1, 2), "PLANET_BASE = 0"))
frag(((3, 1), "BELT_BASE = N_PLANETS"))
frag(((3, 2), "COMET_BASE = N_PLANETS + N_BELT"))

frag(((3, 1), "BELT_R = (0.30, 0.36)"))
frag(((3, 2), "COMET_PERI = 0.06"))
frag(((3, 2), "COMET_APO = 0.46"))
frag(((1, 3), "VIEW_SCALE = 0.95"))
frag(((1, 3), "CANVAS_FADE = 0.90"))

PLANET_COLORS = """PLANET_COLORS = np.array(
    [
        [0.75, 0.72, 0.68],
        [0.95, 0.85, 0.55],
        [0.30, 0.55, 0.95],
        [0.90, 0.45, 0.25],
        [0.85, 0.75, 0.55],
        [0.60, 0.80, 0.90],
    ],
    dtype=np.float32,
)"""

frag(((1, 2), PLANET_COLORS))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "vel = None"))
frag(((1, 2), "color = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, color, pixels"))
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
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    vel = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    color = ti.Vector.field(3, ti.f32, shape=N)"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- pure numpy generation ---------------------------------------------------------

CIRCULAR_V = '''def circular_velocity(p):
    """Pure numpy: the speed that makes gravity exactly the centripetal force — one orbit, forever."""
    r = np.linalg.norm(p, axis=-1, keepdims=True)
    speed = np.sqrt(GM / r.squeeze(-1))
    tangent = np.stack([-p[..., 1], p[..., 0]], axis=-1) / r
    return tangent * speed[..., None]'''

frag(((1, 3), CIRCULAR_V))

SEED_PLANETS = """def seed_planets(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    radii = np.linspace(0.10, 0.42, N_PLANETS).astype(np.float32)
    ang = rng.uniform(0.0, 2 * np.pi, N_PLANETS)
    p = np.stack([radii * np.cos(ang), radii * np.sin(ang)], axis=1).astype(np.float32)
    return p, circular_velocity(p).astype(np.float32), PLANET_COLORS.copy()"""

frag(((1, 3), SEED_PLANETS))

SEED_BELT = """def seed_belt(rng_seed=0):
    rng = np.random.default_rng(rng_seed + 1)
    r = rng.uniform(BELT_R[0], BELT_R[1], N_BELT)
    ang = rng.uniform(0.0, 2 * np.pi, N_BELT)
    p = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)
    v = circular_velocity(p).astype(np.float32)
    col = np.full((N_BELT, 3), (0.35, 0.32, 0.28), dtype=np.float32)
    col *= rng.uniform(0.5, 1.0, (N_BELT, 1)).astype(np.float32)
    return p, v, col"""

frag(((3, 1), SEED_BELT))

VIS_VIVA = '''def comet_aphelion_velocity(r_apo, r_peri):
    """Pure numpy: vis-viva at aphelion for an ellipse with the given extremes."""
    a = 0.5 * (r_apo + r_peri)
    return np.sqrt(GM * (2.0 / r_apo - 1.0 / a))'''

frag(((3, 2), VIS_VIVA))

SEED_COMETS = """def seed_comets(rng_seed=0):
    rng = np.random.default_rng(rng_seed + 2)
    ang = rng.uniform(0.0, 2 * np.pi, N_COMETS)
    r_apo = rng.uniform(COMET_APO * 0.8, COMET_APO, N_COMETS)
    p = np.stack([r_apo * np.cos(ang), r_apo * np.sin(ang)], axis=1).astype(np.float32)
    speed = comet_aphelion_velocity(r_apo, COMET_PERI)
    tangent = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    v = (tangent * speed[:, None]).astype(np.float32)
    col = np.full((N_COMETS, 3), (0.55, 0.85, 0.95), dtype=np.float32)
    return p, v, col"""

frag(((3, 2), SEED_COMETS))

APPLY_V1 = """def apply_seed(rng_seed=0):
    p, v, c = seed_planets(rng_seed)
    pos.from_numpy(p)
    vel.from_numpy(v)
    color.from_numpy(c)
    pixels.fill(0.0)"""

APPLY_V2 = """def apply_seed(rng_seed=0):
    parts = [seed_planets(rng_seed), seed_belt(rng_seed)]
    pos.from_numpy(np.concatenate([p for p, _v, _c in parts]))
    vel.from_numpy(np.concatenate([v for _p, v, _c in parts]))
    color.from_numpy(np.concatenate([c for _p, _v, c in parts]))
    pixels.fill(0.0)"""

APPLY_V3 = """def apply_seed(rng_seed=0):
    parts = [seed_planets(rng_seed), seed_belt(rng_seed), seed_comets(rng_seed)]
    pos.from_numpy(np.concatenate([p for p, _v, _c in parts]))
    vel.from_numpy(np.concatenate([v for _p, v, _c in parts]))
    color.from_numpy(np.concatenate([c for _p, _v, c in parts]))
    pixels.fill(0.0)"""

frag(((1, 3), APPLY_V1), ((3, 1), APPLY_V2), ((3, 2), APPLY_V3))

# --- gravity + integrators -------------------------------------------------------------

ACCEL = """@ti.func
def accel(p):
    r2 = p.dot(p) + 1e-6
    r = ti.sqrt(r2)
    return -GM * p / (r2 * r)"""

frag(((2, 1), ACCEL))

EULER = """@ti.kernel
def euler_step():
    for b in pos:
        a = accel(pos[b])
        pos[b] += DT * vel[b]
        vel[b] += DT * a"""

LEAPFROG = """@ti.kernel
def leapfrog():
    for b in pos:
        vel[b] += 0.5 * DT * accel(pos[b])
        pos[b] += DT * vel[b]
        vel[b] += 0.5 * DT * accel(pos[b])"""

frag(((2, 1), EULER), ((2, 2), LEAPFROG))

# --- render: two versions -------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for b in pos:
        x = 0.5 + pos[b][0] * VIEW_SCALE
        y = 0.5 + pos[b][1] * VIEW_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 1 <= xi < RES - 1 and 1 <= yi < RES - 1:
            pixels[xi, yi] += color[b]"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    cx = RES // 2
    for _ in range(1):
        for di, dj in ti.ndrange((-4, 5), (-4, 5)):
            w = ti.exp(-(di * di + dj * dj) / 6.0)
            pixels[cx + di, cx + dj] += w * ti.Vector([1.0, 0.85, 0.4])

    for b in pos:
        x = 0.5 + pos[b][0] * VIEW_SCALE
        y = 0.5 + pos[b][1] * VIEW_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 1 <= xi < RES - 1 and 1 <= yi < RES - 1:
            gain = 1.0
            if b >= BELT_BASE and b < COMET_BASE:
                gain = 0.35
            pixels[xi, yi] += color[b] * gain
            if b < N_PLANETS or b >= COMET_BASE:
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    pixels[xi + di, yi + dj] += color[b] * 0.3"""

frag(((1, 3), RENDER_V1), ((3, 3), RENDER_V2))

CLAMP = """@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)"""

frag(((1, 3), CLAMP))

STEP_V1 = """def step():
    render()
    clamp_pixels()"""

STEP_V2 = """def step():
    euler_step()
    render()
    clamp_pixels()"""

STEP_V3 = """def step():
    for _ in range(SUBSTEPS):
        leapfrog()
    render()
    clamp_pixels()"""

frag(((1, 3), STEP_V1), ((2, 1), STEP_V2), ((2, 3), STEP_V3))

ENERGY = '''def total_energy():
    """Pure numpy: kinetic + potential per body — the quantity leapfrog protects."""
    p = pos.to_numpy()
    v = vel.to_numpy()
    ke = 0.5 * (v**2).sum(axis=1)
    pe = -GM / np.linalg.norm(p, axis=1)
    return ke + pe'''

frag(((2, 2), ENERGY))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Solar System — taichi-academy", res=RES, background_color=0x000000)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((3, 3), EVENTS_V2))

frag(((1, 3), "        step()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((3, 3), '        gui.text("planets, belt, comets — leapfrog keeps them honest", (0.02, 0.98), color=0xFFFFFF)'))
frag(((3, 3), '        gui.text("[r] rescatter", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

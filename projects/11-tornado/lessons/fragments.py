"""Code SOT for project 11 — tornado.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 11-tornado`.

Evolutions: step() grows in three passes — the base vortex loop (chapter 2),
+debris advection (chapter 3), +vorticity confinement inserted in the middle
(chapter 4, the only version where insertion order matters: curl must be
computed from the just-copied-back velocity, before the pressure project).
sample/bilerp/advect/the pressure-projection block are reused nearly verbatim
from project 02 — this project's new material is the vortex forcing and the
debris-in-a-fluid coupling, not the solver itself.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="11-tornado",
    default_file="tornado.py",
    reference={"tornado.py": PROJECT_DIR / "reference" / "tornado.py"},
    chapter_steps={1: 3, 2: 4, 3: 2, 4: 4},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Tornado: a self-sustaining vortex in a stable-fluids grid, with debris riding the wind."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((2, 1), "DT = 1.0"))
frag(((2, 3), "JACOBI_ITERS = 40"))
frag(((2, 4), "DYE_DECAY = 0.985"))
frag(((2, 4), "VEL_DECAY = 0.97"))
frag(((4, 1), "CURL_STRENGTH = 0.3"))

frag(((1, 2), "CX, CY = N * 0.5, N * 0.5"))
frag(((1, 2), "CORE_R = 60.0"))
frag(((2, 2), "TANGENT_STRENGTH = 1.0"))
frag(((2, 2), "INFLOW_STRENGTH = 0.2"))

frag(((1, 2), "N_DEBRIS = 3000"))
frag(((3, 1), "DRAG = 0.12"))
frag(((3, 1), "DEBRIS_PULL = 0.015"))
frag(((3, 1), "DEBRIS_HOME_R = N * 0.4"))

frag(((4, 3), "STIR_RADIUS = 20.0"))
frag(((4, 3), "STIR_FORCE = 200.0"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "vel = None"))
frag(((1, 2), "vel_next = None"))
frag(((1, 2), "dye = None"))
frag(((1, 2), "dye_next = None"))
frag(((1, 2), "pressure = None"))
frag(((1, 2), "pressure_next = None"))
frag(((1, 2), "divergence = None"))
frag(((4, 1), "curl = None"))
frag(((1, 2), "dpos = None"))
frag(((1, 2), "dvel = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    (
        (1, 2),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence\n"
        "    global dpos, dvel",
    ),
    (
        (4, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, curl\n"
        "    global dpos, dvel",
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
frag(((1, 2), "    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))"))
frag(((1, 2), "    dye = ti.Vector.field(3, ti.f32, shape=(N, N))"))
frag(((1, 2), "    dye_next = ti.Vector.field(3, ti.f32, shape=(N, N))"))
frag(((1, 2), "    pressure = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    pressure_next = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    divergence = ti.field(ti.f32, shape=(N, N))"))
frag(((4, 1), "    curl = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    dpos = ti.Vector.field(2, ti.f32, shape=N_DEBRIS)"))
frag(((1, 2), "    dvel = ti.Vector.field(2, ti.f32, shape=N_DEBRIS)"))

# --- seeding -------------------------------------------------------------------------

SEED_DEBRIS = '''def seed_debris(rng_seed=0):
    """Pure numpy: N_DEBRIS points scattered in a ring around the vortex core."""
    rng = np.random.default_rng(rng_seed)
    ang = rng.uniform(0.0, 2 * np.pi, N_DEBRIS)
    rad = rng.uniform(CORE_R * 1.2, N * 0.45, N_DEBRIS)
    x = CX + rad * np.cos(ang)
    y = CY + rad * np.sin(ang)
    return np.stack([x, y], axis=1).astype(np.float32)'''

frag(((1, 3), SEED_DEBRIS))

APPLY_SEED = """def apply_seed(rng_seed=0):
    dye.fill(0.0)
    vel.fill(0.0)
    pressure.fill(0.0)
    dpos.from_numpy(seed_debris(rng_seed))
    dvel.fill(0.0)"""

frag(((1, 3), APPLY_SEED))

# --- fluid transport (reused from project 02) -----------------------------------------

SAMPLE = """@ti.func
def sample(f: ti.template(), i, j):
    ci = min(max(i, 0), N - 1)
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
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy"""

frag(((2, 1), BILERP))

ADVECT = """@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)"""

frag(((2, 1), ADVECT))

frag(((2, 1), "@ti.kernel\ndef copy_back():\n    for i, j in dye:\n        dye[i, j] = dye_next[i, j]\n        vel[i, j] = vel_next[i, j]"))

# --- the vortex --------------------------------------------------------------------

VORTEX_FORCING = """@ti.kernel
def vortex_forcing():
    for i, j in vel:
        rx, ry = float(i) - CX, float(j) - CY
        r = ti.sqrt(rx * rx + ry * ry) + 1e-3
        falloff = r / (r * r + CORE_R * CORE_R) * CORE_R
        tangent = ti.Vector([-ry, rx]) / r
        radial_in = ti.Vector([-rx, -ry]) / r
        vel[i, j] += DT * falloff * (TANGENT_STRENGTH * tangent + INFLOW_STRENGTH * radial_in)"""

frag(((2, 2), VORTEX_FORCING))

SEED_DYE = """@ti.kernel
def seed_dye():
    for i, j in dye:
        rx, ry = float(i) - CX, float(j) - CY
        r2 = rx * rx + ry * ry
        if r2 < (CORE_R * 1.5) ** 2:
            w = ti.exp(-r2 / (CORE_R * CORE_R))
            dye[i, j] = ti.min(dye[i, j] + 0.02 * w * ti.Vector([0.8, 0.75, 0.6]), 1.0)"""

frag(((2, 2), SEED_DYE))

# --- stirring (added later) ---------------------------------------------------------

STIR = """@ti.kernel
def stir(mx: ti.f32, my: ti.f32, fx: ti.f32, fy: ti.f32):
    for i, j in vel:
        dx, dy = float(i) - mx * N, float(j) - my * N
        w = ti.exp(-(dx * dx + dy * dy) / (STIR_RADIUS * STIR_RADIUS))
        vel[i, j] += w * ti.Vector([fx, fy])"""

frag(((4, 3), STIR))

frag(((2, 4), "@ti.kernel\ndef decay():\n    for i, j in dye:\n        dye[i, j] *= DYE_DECAY\n        vel[i, j] *= VEL_DECAY"))

# --- pressure projection (reused from project 02) -------------------------------------

DIVERGENCE = """@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0] - vel[i, j][0] + sample(vel, i, j + 1)[1] - vel[i, j][1]
        )"""

frag(((2, 3), DIVERGENCE))

JACOBI = """@ti.kernel
def pressure_jacobi():
    for i, j in pressure:
        pressure_next[i, j] = (
            sample(pressure, i + 1, j)
            + sample(pressure, i - 1, j)
            + sample(pressure, i, j + 1)
            + sample(pressure, i, j - 1)
            - divergence[i, j]
        ) * 0.25"""

frag(((2, 3), JACOBI))

frag(((2, 3), "@ti.kernel\ndef copy_pressure():\n    for i, j in pressure:\n        pressure[i, j] = pressure_next[i, j]"))

SUBTRACT_GRADIENT = """@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector(
            [pressure[i, j] - sample(pressure, i - 1, j), pressure[i, j] - sample(pressure, i, j - 1)]
        )
        vel[i, j] -= grad"""

frag(((2, 3), SUBTRACT_GRADIENT))

frag(((2, 3), "def project():\n    compute_divergence()\n    for _ in range(JACOBI_ITERS):\n        pressure_jacobi()\n        copy_pressure()\n    subtract_gradient()"))

# --- vorticity confinement (added later) -----------------------------------------------

COMPUTE_CURL = """@ti.kernel
def compute_curl():
    for i, j in vel:
        curl[i, j] = (
            sample(vel, i + 1, j)[1] - sample(vel, i - 1, j)[1] - sample(vel, i, j + 1)[0] + sample(vel, i, j - 1)[0]
        ) * 0.5"""

frag(((4, 1), COMPUTE_CURL))

APPLY_VORTICITY = """@ti.kernel
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
        vel[i, j] += DT * strength * curl[i, j] * ti.Vector([n[1], -n[0]])"""

frag(((4, 1), APPLY_VORTICITY))

# --- debris --------------------------------------------------------------------------

ADVECT_DEBRIS = """@ti.kernel
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
                dvel[p][a] *= -0.5"""

frag(((3, 1), ADVECT_DEBRIS))

# --- the tick ----------------------------------------------------------------------

STEP_V1 = """def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    project()
    decay()"""

STEP_V2 = """def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    project()
    decay()
    advect_debris()"""

STEP_V3 = """def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    compute_curl()
    apply_vorticity(CURL_STRENGTH)
    project()
    decay()
    advect_debris()"""

frag(((2, 4), STEP_V1), ((3, 2), STEP_V2), ((4, 2), STEP_V3))

frag(((2, 4), "@ti.kernel\ndef render(pixels: ti.template()):\n    for i, j in pixels:\n        pixels[i, j] = ti.math.clamp(dye[i, j], 0.0, 1.0)"))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((2, 4), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))
frag(((1, 3), '    gui = ti.GUI("Tornado — taichi-academy", res=N, background_color=0x0A0A12)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((4, 4), EVENTS_V2))

STIR_BLOCK = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            fx, fy = np.random.uniform(-1, 1, 2) * STIR_FORCE
            stir(mx, my, float(fx), float(fy))"""

frag(((4, 3), STIR_BLOCK))

frag(((2, 4), "        step()"))
frag(((2, 4), "        render(pixels)"))
frag(((2, 4), "        gui.set_image(pixels)"))
frag(((1, 3), "        gui.circles(dpos.to_numpy() / N, radius=1.5, color=0xFFFFFF)"))
frag(((4, 4), '        gui.text("drag to stir  [r] new debris", (0.02, 0.98), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

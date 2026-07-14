"""Code SOT for project 25 — molecular dynamics.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 25-molecular-dynamics`.

Evolutions: chapter 1 is pure NVE dynamics — LJ forces + velocity-Verlet on the
spatial hash — with a plain white render and no thermostat (step's thermo flag
defaults on, but ch1's main never changes the target, so it just holds the seed
temperature). Chapter 2 adds measure_temp, the thermostat, heat/cool keys, and
the coordination-colored render (the SAME render as the reference, so final ==
reference). Chapter 3 adds the heat gun, reseed, and the crystalline HUD.
The compute_forces kernel counts coordination from chapter 1 (cheap), colored
only from chapter 2.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="25-molecular-dynamics",
    default_file="molecular_dynamics.py",
    reference={"molecular_dynamics.py": PROJECT_DIR / "reference" / "molecular_dynamics.py"},
    chapter_steps={1: 3, 2: 3, 3: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Molecular Dynamics: one pair force + Verlet, and matter melts, freezes, and crystallizes."""'))
frag(((1, 1), "import math"))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 600"))
frag(((1, 2), "L = 40.0"))
frag(((1, 2), "N = 1400"))
frag(((1, 2), "SIGMA = 1.0"))
frag(((1, 2), "EPS = 1.0"))
frag(((1, 2), "RCUT = 2.5"))
frag(((1, 2), "COORD_SHELL = 1.3"))
frag(((1, 2), "DT = 0.004"))
frag(((1, 2), "GRID = 16"))
frag(((1, 2), "CELL = L / GRID"))
frag(((1, 2), "NCELLS = GRID * GRID"))

frag(((2, 2), "THERMO_RATE = 0.05"))
frag(((2, 2), "TEMP_STEP = 0.1"))
frag(((2, 2), "TEMP_MIN = 0.02"))
frag(((2, 2), "TEMP_MAX = 4.0"))
frag(((3, 1), "HEAT_RADIUS = 4.0"))
frag(((3, 1), "HEAT_BOOST = 1.4"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "vel = None"))
frag(((1, 2), "acc = None"))
frag(((1, 2), "coord = None"))
frag(((1, 2), "cell_count = None"))
frag(((1, 2), "cell_start = None"))
frag(((1, 2), "cell_cursor = None"))
frag(((1, 2), "sorted_idx = None"))
frag(((2, 2), "temp_target = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    (
        (1, 2),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, vel, acc, coord, cell_count, cell_start, cell_cursor, sorted_idx, pixels",
    ),
    (
        (2, 2),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, vel, acc, coord, cell_count, cell_start, cell_cursor, sorted_idx, temp_target, pixels",
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
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    vel = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    acc = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    coord = ti.field(ti.i32, shape=N)"))
frag(((1, 2), "    cell_count = ti.field(ti.i32, shape=NCELLS)"))
frag(((1, 2), "    cell_start = ti.field(ti.i32, shape=NCELLS)"))
frag(((1, 2), "    cell_cursor = ti.field(ti.i32, shape=NCELLS)"))
frag(((1, 2), "    sorted_idx = ti.field(ti.i32, shape=N)"))
frag(((2, 2), "    temp_target = ti.field(ti.f32, shape=())"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- pure numpy generation ---------------------------------------------------------

LATTICE = '''def lattice_positions(n, box):
    """Pure numpy: n atoms on a square grid filling the box, jittered off perfection."""
    side = int(math.ceil(math.sqrt(n)))
    spacing = box / side
    xy = np.array([[(i + 0.5) * spacing, (j + 0.5) * spacing] for i in range(side) for j in range(side)])
    return xy[:n].astype(np.float32)'''

frag(((1, 3), LATTICE))

MAXWELL = '''def maxwell_velocities(n, temperature, rng):
    """Pure numpy: gaussian velocities at a temperature, with net momentum removed."""
    v = rng.normal(0.0, math.sqrt(temperature), (n, 2)).astype(np.float32)
    v -= v.mean(axis=0)
    return v'''

frag(((1, 3), MAXWELL))

APPLY_V1 = """def apply_seed(rng_seed=0, temperature=1.0):
    rng = np.random.default_rng(rng_seed)
    xy = lattice_positions(N, L) + rng.normal(0, 0.05, (N, 2)).astype(np.float32)
    pos.from_numpy(xy % L)
    vel.from_numpy(maxwell_velocities(N, temperature, rng))
    acc.fill(0.0)
    build_grid()
    compute_forces()"""

APPLY_V2 = """def apply_seed(rng_seed=0, temperature=1.0):
    rng = np.random.default_rng(rng_seed)
    xy = lattice_positions(N, L) + rng.normal(0, 0.05, (N, 2)).astype(np.float32)
    pos.from_numpy(xy % L)
    vel.from_numpy(maxwell_velocities(N, temperature, rng))
    acc.fill(0.0)
    temp_target[None] = temperature
    build_grid()
    compute_forces()"""

frag(((1, 3), APPLY_V1), ((2, 2), APPLY_V2))

# --- spatial hash (project 06) --------------------------------------------------------

FLAT_CELL = """@ti.func
def flat_cell(p):
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj"""

frag(((1, 3), FLAT_CELL))

frag(((1, 3), "@ti.kernel\ndef count_cells():\n    for c in cell_count:\n        cell_count[c] = 0\n    for p in pos:\n        cell_count[flat_cell(p)] += 1"))
frag(((1, 3), "@ti.kernel\ndef prefix_sum():\n    for _ in range(1):\n        a = 0\n        for c in range(NCELLS):\n            cell_start[c] = a\n            a += cell_count[c]"))
frag(((1, 3), "@ti.kernel\ndef scatter():\n    for c in cell_cursor:\n        cell_cursor[c] = cell_start[c]\n    for p in pos:\n        idx = flat_cell(p)\n        slot = ti.atomic_add(cell_cursor[idx], 1)\n        sorted_idx[slot] = p"))
frag(((1, 3), "def build_grid():\n    count_cells()\n    prefix_sum()\n    scatter()"))

WRAPD = """@ti.func
def wrapd(a, b):
    d = a - b
    if d > 0.5 * L:
        d -= L
    if d < -0.5 * L:
        d += L
    return d"""

frag(((1, 3), WRAPD))

# --- the Lennard-Jones force ---------------------------------------------------------

COMPUTE_FORCES = """@ti.kernel
def compute_forces():
    for p in pos:
        f = ti.Vector([0.0, 0.0])
        nc = 0
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni = (ci + di) % GRID
            nj = (cj + dj) % GRID
            nidx = ni * GRID + nj
            for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                q = sorted_idx[s]
                if q != p:
                    dx = wrapd(pos[p][0], pos[q][0])
                    dy = wrapd(pos[p][1], pos[q][1])
                    r2 = dx * dx + dy * dy
                    if 1e-4 < r2 < RCUT * RCUT:
                        inv2 = SIGMA * SIGMA / r2
                        inv6 = inv2 * inv2 * inv2
                        fmag = 24.0 * EPS * (2.0 * inv6 * inv6 - inv6) / r2
                        f += fmag * ti.Vector([dx, dy])
                        if r2 < COORD_SHELL * COORD_SHELL:
                            nc += 1
        acc[p] = f
        coord[p] = nc"""

frag(((1, 3), COMPUTE_FORCES))

# --- velocity-Verlet ------------------------------------------------------------------

frag(((1, 3), "@ti.kernel\ndef half_kick():\n    for p in vel:\n        vel[p] += 0.5 * DT * acc[p]"))
frag(((1, 3), "@ti.kernel\ndef drift():\n    for p in pos:\n        newp = pos[p] + DT * vel[p]\n        pos[p] = ti.Vector([newp[0] % L, newp[1] % L])"))

# --- temperature + thermostat (chapter 2) --------------------------------------------

MEASURE_TEMP = """@ti.kernel
def measure_temp() -> ti.f32:
    ke = 0.0
    for p in vel:
        ke += 0.5 * vel[p].dot(vel[p])
    return ke / N"""

frag(((2, 1), MEASURE_TEMP))

THERMOSTAT = """@ti.kernel
def thermostat(cur_temp: ti.f32):
    scale = 1.0
    if cur_temp > 1e-6:
        scale = ti.sqrt(temp_target[None] / cur_temp)
    s = 1.0 + THERMO_RATE * (scale - 1.0)
    for p in vel:
        vel[p] *= s"""

frag(((2, 2), THERMOSTAT))

# --- heat gun (chapter 3) -------------------------------------------------------------

HEAT = """@ti.kernel
def heat(mx: ti.f32, my: ti.f32):
    for p in pos:
        dx = wrapd(pos[p][0], mx * L)
        dy = wrapd(pos[p][1], my * L)
        if dx * dx + dy * dy < HEAT_RADIUS * HEAT_RADIUS:
            vel[p] *= HEAT_BOOST"""

frag(((3, 1), HEAT))

# --- the tick: two versions -----------------------------------------------------------

STEP_V1 = """def step():
    half_kick()
    drift()
    build_grid()
    compute_forces()
    half_kick()"""

STEP_V2 = """def step(thermo=True):
    half_kick()
    drift()
    build_grid()
    compute_forces()
    half_kick()
    if thermo:
        thermostat(measure_temp())"""

frag(((1, 3), STEP_V1), ((2, 2), STEP_V2))

CRYST_FRAC = '''def crystalline_fraction():
    """Pure numpy: fraction of atoms with a full hexagonal shell of 6 close neighbors."""
    return float((coord.to_numpy() == 6).mean())'''

frag(((2, 3), CRYST_FRAC))

# --- render: two versions -------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.03, 0.03, 0.05])
    for p in pos:
        xi = ti.cast(pos[p][0] / L * RES, ti.i32)
        yi = ti.cast(pos[p][1] / L * RES, ti.i32)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = ti.Vector([0.7, 0.8, 1.0])"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.03, 0.03, 0.05])
    for p in pos:
        xi = ti.cast(pos[p][0] / L * RES, ti.i32)
        yi = ti.cast(pos[p][1] / L * RES, ti.i32)
        c = coord[p]
        col = ti.Vector([0.3, 0.5, 1.0])
        if c >= 6:
            col = ti.Vector([1.0, 0.85, 0.3])
        elif c >= 4:
            col = ti.Vector([0.5, 0.9, 0.5])
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = col"""

frag(((1, 3), RENDER_V1), ((2, 1), RENDER_V2))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed(temperature=1.0)"))
frag(((1, 3), '    gui = ti.GUI("Molecular Dynamics — taichi-academy", res=RES, background_color=0x08080F)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                temp_target[None] = min(temp_target[None] + TEMP_STEP, TEMP_MAX)
            elif e.key == ti.GUI.DOWN:
                temp_target[None] = max(temp_target[None] - TEMP_STEP, TEMP_MIN)'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                temp_target[None] = min(temp_target[None] + TEMP_STEP, TEMP_MAX)
            elif e.key == ti.GUI.DOWN:
                temp_target[None] = max(temp_target[None] - TEMP_STEP, TEMP_MIN)
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000), temperature=1.0)'''

frag(((1, 3), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 2), EVENTS_V3))

HEAT_WIRE = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            heat(mx, my)"""

frag(((3, 1), HEAT_WIRE))

frag(((1, 3), "        step()"))
frag(((1, 3), "        render()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(
    (
        (2, 3),
        '        gui.text(\n'
        '            f"target T {temp_target[None]:.2f}   crystalline {crystalline_fraction() * 100:.0f}%",\n'
        '            (0.02, 0.98), color=0xFFFFFF,\n'
        '        )',
    )
)
frag(((3, 2), '        gui.text("[up/down] heat/cool  drag: heat gun  [r] remelt   gold: crystal, blue: fluid", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

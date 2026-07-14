"""Code SOT for project 26 — destruction engine.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 26-destruction`.

Arc: chapter 1 raises a city that STANDS — a braced lattice of distance bonds solved
with over-relaxed Jacobi PBD, gravity, and a floor. No fracture, plain grey render.
Chapter 2 makes it DESTRUCTIBLE: bonds that snap past a breaking strain, a radial
explosion, the spatial-hash self-collision that lets rubble pile, a stress-coloured
render, and click-to-detonate. Chapter 3 SHAKES it: the quake kernel, a damage HUD,
and the earthquake / rebuild controls — the finished reference.

Kernels are registered in reference document order (predict, then the spatial hash,
then break/solve/collision), so their (chapter, step) keys are deliberately out of
order — the solver pieces are chapter 1 even though the hash between them is chapter 2.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="26-destruction",
    default_file="destruction.py",
    reference={"destruction.py": PROJECT_DIR / "reference" / "destruction.py"},
    chapter_steps={1: 6, 2: 3, 3: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Destruction Engine: buildings are lattices of breakable bonds — explosions and quakes fracture them."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -----------------------------------------------------------------------

frag(((1, 2), "RES = 640"))
frag(((1, 2), "GRAVITY = 0.5"))
frag(((1, 2), "DT = 1.0 / 60"))
frag(((1, 2), "ITERS = 24"))
frag(((1, 2), "OMEGA = 1.7             # over-relaxation: pushes the Jacobi solver toward rigid faster"))
frag(((1, 2), "FLOOR = 0.04"))
frag(((1, 2), "RADIUS = 0.006          # particle radius, for rubble self-collision"))
frag(((1, 2), "SPACING = 0.014         # rest gap between lattice neighbours"))
frag(((1, 2), "BREAK_STRAIN = 1.4      # a bond snaps once stretched past 1.4x its rest length"))
frag(((1, 2), "DAMP = 0.99"))

frag(((1, 2), "GRID = 64"))
frag(((1, 2), "CELL = 1.0 / GRID"))
frag(((1, 2), "NCELLS = GRID * GRID"))

frag(((1, 2), "MAX_P = 2000"))
frag(((1, 2), "MAX_B = 8000"))

frag((
    (1, 2),
    "# buildings: (base_x, width_cols, height_rows) — blocky walls stand; thin spires pancake\n"
    "BUILDINGS = [(0.08, 22, 18), (0.45, 16, 24), (0.72, 20, 14)]",
))

frag(((2, 1), "EXPLODE_POWER = 4.5"))
frag(((2, 1), "EXPLODE_RADIUS = 0.16"))
frag(((3, 1), "QUAKE_AMP = 0.012"))
frag(((3, 1), "QUAKE_FREQ = 38.0"))

# --- module-level fields -------------------------------------------------------------

for _name in (
    "pos", "prev", "delta", "dn", "stress", "b_a", "b_b", "b_rest", "b_broken",
    "n_p", "n_b", "cell_count", "cell_start", "cell_cursor", "sorted_idx", "pixels",
):
    frag(((1, 2), f"{_name} = None"))

# --- init: allocate every field once (Metal can't free fields) -----------------------

INIT_SIM = '''def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, prev, delta, dn, stress, b_a, b_b, b_rest, b_broken, n_p, n_b
    global cell_count, cell_start, cell_cursor, sorted_idx, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=MAX_P)
    prev = ti.Vector.field(2, ti.f32, shape=MAX_P)
    delta = ti.Vector.field(2, ti.f32, shape=MAX_P)
    dn = ti.field(ti.f32, shape=MAX_P)
    stress = ti.field(ti.f32, shape=MAX_P)
    b_a = ti.field(ti.i32, shape=MAX_B)
    b_b = ti.field(ti.i32, shape=MAX_B)
    b_rest = ti.field(ti.f32, shape=MAX_B)
    b_broken = ti.field(ti.i32, shape=MAX_B)
    n_p = ti.field(ti.i32, shape=())
    n_b = ti.field(ti.i32, shape=())
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=MAX_P)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))'''

frag(((1, 3), INIT_SIM))

# --- pure numpy: lay out the city ----------------------------------------------------

BUILD_STRUCTURES = '''def build_structures():
    """Pure numpy: lay out the buildings as a lattice. Returns (positions, edges).

    Each cell links to its right, up, and both diagonal neighbours — the diagonals are
    the braces that keep a wall from folding flat (a square of four distance bonds is a
    hinge; add a diagonal and it is rigid)."""
    positions = []
    edges = []
    for (bx, w, h) in BUILDINGS:
        local = {}
        for gy in range(h):
            for gx in range(w):
                local[(gx, gy)] = len(positions)
                positions.append([bx + gx * SPACING, FLOOR + RADIUS + gy * SPACING])
        for gy in range(h):
            for gx in range(w):
                a = local[(gx, gy)]
                for (nx, ny) in ((gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1), (gx + 1, gy - 1)):
                    if (nx, ny) in local:
                        edges.append((a, local[(nx, ny)]))
    return np.array(positions, dtype=np.float32), np.array(edges, dtype=np.int32)'''

frag(((1, 4), BUILD_STRUCTURES))

BOND_REST = '''def bond_rest_lengths(positions, edges):
    """Pure numpy: the rest length of each bond is the initial neighbour distance."""
    return np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).astype(np.float32)'''

frag(((1, 4), BOND_REST))

APPLY_SEED = '''def apply_seed():
    """Build the city and upload it: every bond intact, everything at rest."""
    positions, edges = build_structures()
    rests = bond_rest_lengths(positions, edges)
    npart, nbond = len(positions), len(edges)
    pos.from_numpy(np.pad(positions, ((0, MAX_P - npart), (0, 0))))
    prev.from_numpy(np.pad(positions, ((0, MAX_P - npart), (0, 0))))
    b_a.from_numpy(np.pad(edges[:, 0], (0, MAX_B - nbond)))
    b_b.from_numpy(np.pad(edges[:, 1], (0, MAX_B - nbond)))
    b_rest.from_numpy(np.pad(rests, (0, MAX_B - nbond)))
    b_broken.fill(0)
    n_p[None] = npart
    n_b[None] = nbond'''

frag(((1, 4), APPLY_SEED))

# --- predict (Verlet) ----------------------------------------------------------------

PREDICT = '''@ti.kernel
def predict():
    """Verlet: velocity is where you were minus where you are, damped; then fall."""
    for i in range(n_p[None]):
        v = (pos[i] - prev[i]) * DAMP
        prev[i] = pos[i]
        pos[i] = pos[i] + v + ti.Vector([0.0, -GRAVITY]) * DT * DT'''

frag(((1, 5), PREDICT))

# --- spatial hash (project 06), for rubble collision — chapter 2 ----------------------

FLAT_CELL = '''@ti.func
def flat_cell(i):
    ci = ti.min(ti.max(ti.cast(pos[i][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj'''

frag(((2, 2), FLAT_CELL))
frag(((2, 2), "@ti.kernel\ndef count_cells():\n    for c in cell_count:\n        cell_count[c] = 0\n    for i in range(n_p[None]):\n        cell_count[flat_cell(i)] += 1"))
frag(((2, 2), "@ti.kernel\ndef prefix_sum():\n    for _ in range(1):\n        a = 0\n        for c in range(NCELLS):\n            cell_start[c] = a\n            a += cell_count[c]"))
frag(((2, 2), "@ti.kernel\ndef scatter():\n    for c in cell_cursor:\n        cell_cursor[c] = cell_start[c]\n    for i in range(n_p[None]):\n        slot = ti.atomic_add(cell_cursor[flat_cell(i)], 1)\n        sorted_idx[slot] = i"))
frag(((2, 2), "def build_grid():\n    count_cells()\n    prefix_sum()\n    scatter()"))

# --- fracture: bonds snap (chapter 2) ------------------------------------------------

BREAK_BONDS = '''@ti.kernel
def break_bonds():
    """A bond that has been stretched past its breaking strain is gone for good.
    Judged here, on the freshly predicted positions, before the solver heals the stretch."""
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            if (pos[b_b[k]] - pos[b_a[k]]).norm() > b_rest[k] * BREAK_STRAIN:
                b_broken[k] = 1'''

frag(((2, 1), BREAK_BONDS))

# --- the PBD solve: bonds pull to rest, rubble pushes apart --------------------------

CLEAR_DELTA = '''@ti.kernel
def clear_delta():
    for i in range(n_p[None]):
        delta[i] = ti.Vector([0.0, 0.0])
        dn[i] = 0.0'''

frag(((1, 5), CLEAR_DELTA))

SOLVE_BONDS = '''@ti.kernel
def solve_bonds():
    """Each surviving bond nudges its two ends back toward rest length (half the error each)."""
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            a, b = b_a[k], b_b[k]
            d = pos[b] - pos[a]
            dist = d.norm() + 1e-9
            corr = 0.5 * (dist - b_rest[k]) / dist * d
            delta[a] += corr
            dn[a] += 1.0
            delta[b] -= corr
            dn[b] += 1.0'''

frag(((1, 5), SOLVE_BONDS))

SOLVE_COLLISIONS = '''@ti.kernel
def solve_collisions():
    """Rubble does not interpenetrate: any two particles closer than 2R push apart.
    Neighbours are found through the spatial-hash grid, so this stays O(N)."""
    for i in range(n_p[None]):
        ci = ti.min(ti.max(ti.cast(pos[i][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GRID - 1)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                nidx = ni * GRID + nj
                for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                    j = sorted_idx[s]
                    if j > i:
                        d = pos[j] - pos[i]
                        dist = d.norm() + 1e-9
                        mind = 2.0 * RADIUS
                        if dist < mind:
                            corr = 0.5 * (dist - mind) / dist * d
                            delta[i] += corr
                            dn[i] += 1.0
                            delta[j] -= corr
                            dn[j] += 1.0'''

frag(((2, 2), SOLVE_COLLISIONS))

APPLY_DELTA = '''@ti.kernel
def apply_delta():
    """Average every correction landing on a particle, over-relax, and move it once."""
    for i in range(n_p[None]):
        if dn[i] > 0:
            pos[i] += OMEGA * delta[i] / dn[i]'''

frag(((1, 5), APPLY_DELTA))

FLOOR_CONSTRAINT = '''@ti.kernel
def floor_constraint():
    for i in range(n_p[None]):
        if pos[i][1] < FLOOR + RADIUS:
            pos[i][1] = FLOOR + RADIUS
        pos[i][0] = ti.min(ti.max(pos[i][0], RADIUS), 1.0 - RADIUS)'''

frag(((1, 5), FLOOR_CONSTRAINT))

COMPUTE_STRESS = '''@ti.kernel
def compute_stress():
    """Per-particle load: sum of the strain on its surviving bonds (drives the colour)."""
    for i in range(n_p[None]):
        stress[i] = 0.0
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            strain = ti.abs((pos[b_b[k]] - pos[b_a[k]]).norm() / b_rest[k] - 1.0)
            stress[b_a[k]] += strain
            stress[b_b[k]] += strain'''

frag(((2, 3), COMPUTE_STRESS))

# --- the two disasters ---------------------------------------------------------------

EXPLODE = '''@ti.kernel
def explode(mx: ti.f32, my: ti.f32, power: ti.f32, radius: ti.f32):
    """A radial shove, strongest at the blast centre, fading to nothing at its edge."""
    c = ti.Vector([mx, my])
    for i in range(n_p[None]):
        d = pos[i] - c
        r = d.norm() + 1e-6
        if r < radius:
            pos[i] += d / r * power * (1.0 - r / radius) * DT'''

frag(((2, 1), EXPLODE))

QUAKE = '''@ti.kernel
def quake(t: ti.f32, amp: ti.f32):
    """Shear the ground: particles near the floor are dragged sideways in an oscillation,
    the drag fading with height, so the base whips out from under the mass above it."""
    shift = amp * ti.sin(t * QUAKE_FREQ)
    for i in range(n_p[None]):
        zone = (pos[i][1] - FLOOR) / 0.08
        if zone < 1.0:
            pos[i][0] += shift * (1.0 - zone)'''

frag(((3, 1), QUAKE))

# --- the tick: two versions ----------------------------------------------------------

STEP_V1 = """def step():
    predict()
    for _ in range(ITERS):
        clear_delta()
        solve_bonds()
        apply_delta()
        floor_constraint()"""

STEP_V2 = """def step():
    predict()
    break_bonds()
    build_grid()
    for _ in range(ITERS):
        clear_delta()
        solve_bonds()
        solve_collisions()
        apply_delta()
        floor_constraint()
    compute_stress()"""

frag(((1, 6), STEP_V1), ((2, 3), STEP_V2))

BROKEN_FRACTION = '''def broken_fraction():
    """Pure numpy: share of bonds that have snapped — how ruined the city is."""
    nb = n_b[None]
    return float(b_broken.to_numpy()[:nb].sum()) / max(nb, 1)'''

frag(((3, 2), BROKEN_FRACTION))

# --- render: two versions ------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.06, 0.07, 0.10])
        if j < FLOOR * RES:
            pixels[i, j] = ti.Vector([0.15, 0.13, 0.10])
    for i in range(n_p[None]):
        xi = ti.cast(pos[i][0] * RES, ti.i32)
        yi = ti.cast(pos[i][1] * RES, ti.i32)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = ti.Vector([0.6, 0.65, 0.7])"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.06, 0.07, 0.10])
        if j < FLOOR * RES:
            pixels[i, j] = ti.Vector([0.15, 0.13, 0.10])
    for i in range(n_p[None]):
        xi = ti.cast(pos[i][0] * RES, ti.i32)
        yi = ti.cast(pos[i][1] * RES, ti.i32)
        s = ti.min(stress[i] * 3.0, 1.0)
        col = ti.Vector([0.6, 0.65, 0.7]) * (1.0 - s) + ti.Vector([1.0, 0.3, 0.15]) * s
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = col"""

frag(((1, 6), RENDER_V1), ((2, 3), RENDER_V2))

# --- main (sub-fragments; the interactive loop grows across chapters) ----------------

MAIN_HEAD = '''def main():
    init_sim()
    apply_seed()
    for _ in range(60):  # let the city settle onto the ground before the player can touch it
        step()
    gui = ti.GUI("Destruction Engine — taichi-academy", res=RES, background_color=0x10121A)
    frame = 0
    while gui.running:
        frame += 1'''

frag(((1, 6), MAIN_HEAD))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = gui.get_cursor_pos()
                explode(mx, my, EXPLODE_POWER, EXPLODE_RADIUS)"""

EVENTS_V3 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = gui.get_cursor_pos()
                explode(mx, my, EXPLODE_POWER, EXPLODE_RADIUS)
            elif e.key == "r":
                apply_seed()
                for _ in range(60):
                    step()"""

frag(((1, 6), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 3), EVENTS_V3))

QUAKE_WIRE = '''        if gui.is_pressed("q", ti.GUI.SPACE):
            quake(frame * DT, QUAKE_AMP)'''

frag(((3, 3), QUAKE_WIRE))

frag(((1, 6), "        step()"))
frag(((1, 6), "        render()"))
frag(((1, 6), "        gui.set_image(pixels)"))

HUD = '''        gui.text(f"ruined {broken_fraction() * 100:.0f}%", (0.02, 0.98), color=0xFFFFFF)
        gui.text("click: explosion   hold [q]/space: earthquake   [r] rebuild", (0.02, 0.94), color=0xAAAAAA)'''

frag(((3, 3), HUD))

frag(((1, 6), "        gui.show()"))
frag(((1, 6), 'if __name__ == "__main__":\n    main()'))

"""Code SOT for project 24 — artificial life.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 24-artificial-life`.

Evolutions: chapter 1 builds the spatial hash (reused from project 06) and a
plain neighbor COUNT, with straight-line drift — a featureless gas colored by
local density. Chapter 2 replaces drift with turn_and_move (the left/right split
+ the one turn rule), and cells crystallize. count_neighbors (the ch1 counter)
and drift (the ch1 mover) are both superseded by turn_and_move but LEFT IN the
file — the reference keeps them, so the final render equals reference. Chapter 3
adds the stir tool, reseed, and the anatomy legend.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="24-artificial-life",
    default_file="artificial_life.py",
    reference={"artificial_life.py": PROJECT_DIR / "reference" / "artificial_life.py"},
    chapter_steps={1: 3, 2: 2, 3: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Artificial Life: one turn rule on 40,000 particles — and cells self-assemble from soup."""'))
frag(((1, 1), "import math"))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 640"))
frag(((1, 2), "WORLD = 1.0"))
frag(((1, 2), "N = 40000"))

frag(((1, 2), "R = 0.011"))
frag(((2, 1), "ALPHA = math.radians(180.0)"))
frag(((2, 1), "BETA = math.radians(17.0)"))
frag(((1, 2), "V = 0.0015"))

frag(((1, 2), "GRID = 88"))
frag(((1, 2), "CELL = WORLD / GRID"))
frag(((1, 2), "NCELLS = GRID * GRID"))

frag(((3, 1), "STIR_RADIUS = 0.05"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "heading = None"))
frag(((1, 2), "neighbors = None"))
frag(((1, 2), "cell_count = None"))
frag(((1, 2), "cell_start = None"))
frag(((1, 2), "cell_cursor = None"))
frag(((1, 2), "sorted_idx = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, neighbors, cell_count, cell_start, cell_cursor, sorted_idx, pixels"))
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
frag(((1, 2), "    heading = ti.field(ti.f32, shape=N)"))
frag(((1, 2), "    neighbors = ti.field(ti.i32, shape=N)"))
frag(((1, 2), "    cell_count = ti.field(ti.i32, shape=NCELLS)"))
frag(((1, 2), "    cell_start = ti.field(ti.i32, shape=NCELLS)"))
frag(((1, 2), "    cell_cursor = ti.field(ti.i32, shape=NCELLS)"))
frag(((1, 2), "    sorted_idx = ti.field(ti.i32, shape=N)"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

frag(
    (
        (1, 2),
        "def apply_seed(rng_seed=0):\n"
        "    rng = np.random.default_rng(rng_seed)\n"
        "    pos.from_numpy(rng.uniform(0, WORLD, (N, 2)).astype(np.float32))\n"
        "    heading.from_numpy(rng.uniform(0, 2 * np.pi, N).astype(np.float32))",
    )
)

# --- the spatial hash (project 06) ----------------------------------------------------

FLAT_CELL = """@ti.func
def flat_cell(p):
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj"""

frag(((1, 3), FLAT_CELL))

COUNT_CELLS = """@ti.kernel
def count_cells():
    for c in cell_count:
        cell_count[c] = 0
    for p in pos:
        cell_count[flat_cell(p)] += 1"""

frag(((1, 3), COUNT_CELLS))

PREFIX = """@ti.kernel
def prefix_sum():
    for _ in range(1):
        acc = 0
        for c in range(NCELLS):
            cell_start[c] = acc
            acc += cell_count[c]"""

frag(((1, 3), PREFIX))

SCATTER = """@ti.kernel
def scatter():
    for c in cell_cursor:
        cell_cursor[c] = cell_start[c]
    for p in pos:
        idx = flat_cell(p)
        slot = ti.atomic_add(cell_cursor[idx], 1)
        sorted_idx[slot] = p"""

frag(((1, 3), SCATTER))

frag(((1, 3), "def build_grid():\n    count_cells()\n    prefix_sum()\n    scatter()"))

WRAPD = """@ti.func
def wrapd(a, b):
    d = a - b
    if d > 0.5 * WORLD:
        d -= WORLD
    if d < -0.5 * WORLD:
        d += WORLD
    return d"""

frag(((1, 3), WRAPD))

# --- chapter 1: count + drift ---------------------------------------------------------

COUNT_NEIGHBORS = """@ti.kernel
def count_neighbors():
    for p in pos:
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        n = 0
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni = (ci + di) % GRID
            nj = (cj + dj) % GRID
            nidx = ni * GRID + nj
            for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                q = sorted_idx[s]
                if q != p:
                    dx = wrapd(pos[q][0], pos[p][0])
                    dy = wrapd(pos[q][1], pos[p][1])
                    if dx * dx + dy * dy < R * R:
                        n += 1
        neighbors[p] = n"""

frag(((1, 3), COUNT_NEIGHBORS))

DRIFT = """@ti.kernel
def drift():
    for p in pos:
        newp = pos[p] + V * ti.Vector([ti.cos(heading[p]), ti.sin(heading[p])])
        pos[p] = ti.Vector([newp[0] % WORLD, newp[1] % WORLD])"""

frag(((1, 3), DRIFT))

# --- chapter 2: the turn rule ---------------------------------------------------------

TURN_AND_MOVE = """@ti.kernel
def turn_and_move():
    for p in pos:
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        left = 0
        right = 0
        h = heading[p]
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni = (ci + di) % GRID
            nj = (cj + dj) % GRID
            nidx = ni * GRID + nj
            for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                q = sorted_idx[s]
                if q != p:
                    dx = wrapd(pos[q][0], pos[p][0])
                    dy = wrapd(pos[q][1], pos[p][1])
                    if dx * dx + dy * dy < R * R:
                        if ti.sin(ti.atan2(dy, dx) - h) > 0:
                            left += 1
                        else:
                            right += 1
        n = left + right
        neighbors[p] = n
        dphi = ALPHA + BETA * n * (1.0 if right > left else -1.0)
        heading[p] = h + dphi
        newp = pos[p] + V * ti.Vector([ti.cos(heading[p]), ti.sin(heading[p])])
        pos[p] = ti.Vector([newp[0] % WORLD, newp[1] % WORLD])"""

frag(((2, 1), TURN_AND_MOVE))

# --- chapter 3: stir -----------------------------------------------------------------

STIR = """@ti.kernel
def stir(mx: ti.f32, my: ti.f32):
    for p in pos:
        dx = wrapd(pos[p][0], mx)
        dy = wrapd(pos[p][1], my)
        if dx * dx + dy * dy < STIR_RADIUS * STIR_RADIUS:
            heading[p] = ti.atan2(dy, dx)"""

frag(((3, 1), STIR))

# --- render: two versions -------------------------------------------------------------

ANATOMY = """@ti.func
def anatomy_color(n):
    col = ti.Vector([0.2, 0.8, 0.3])
    if n > 12:
        col = ti.Vector([0.95, 0.85, 0.3])
    if n > 26:
        col = ti.Vector([0.85, 0.2, 0.6])
    elif n > 18:
        col = ti.Vector([0.6, 0.4, 0.2])
    return col"""

frag(((1, 3), ANATOMY))

RENDER = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.02, 0.02, 0.04])
    for p in pos:
        xi = ti.cast(pos[p][0] * RES, ti.i32)
        yi = ti.cast(pos[p][1] * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] = anatomy_color(neighbors[p])"""

frag(((1, 3), RENDER))

# --- the tick: two versions -----------------------------------------------------------

frag(
    ((1, 3), "def step():\n    build_grid()\n    count_neighbors()\n    drift()"),
    ((2, 2), "def step():\n    build_grid()\n    turn_and_move()"),
)

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Artificial Life — taichi-academy", res=RES, background_color=0x050508)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((3, 2), EVENTS_V2))

STIR_WIRE = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            stir(mx, my)"""

frag(((3, 1), STIR_WIRE))

frag(((1, 3), "        step()"))
frag(((1, 3), "        render()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((3, 2), '        gui.text("green: free  yellow: membrane  magenta: nucleus", (0.02, 0.98), color=0xFFFFFF)'))
frag(((3, 2), '        gui.text("drag to disturb  [r] new soup", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

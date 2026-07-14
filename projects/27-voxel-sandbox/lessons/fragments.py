"""Code SOT for project 27 — voxel sandbox.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 27-voxel-sandbox`.

Arc: chapter 1 is pure gravity — a grid of materials and the serial per-column fall that
drops a solid column with no striping, plus the paint UI: a working sand + wall painter.
Chapter 2 makes it FLOW — the parallel propose/resolve spread (atomic_min conflict
resolution) gives sand its angle of repose and water its level, and a density swap lets
heavy sink through light. Chapter 3 adds CHEMISTRY — wood, fire, lava, stone, smoke, the
reaction pass and rising smoke — the finished reference.

Helper ti.funcs are registered in reference document order (is_faller with the fall pass in
ch1; is_riser/is_mover/empty_at with the spread in ch2), so their keys are intentionally
non-monotonic relative to file position.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="27-voxel-sandbox",
    default_file="voxel_sandbox.py",
    reference={"voxel_sandbox.py": PROJECT_DIR / "reference" / "voxel_sandbox.py"},
    chapter_steps={1: 6, 2: 3, 3: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Voxel Sandbox: a cellular automaton where sand, water, lava, and fire fall, flow, burn, and react."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -----------------------------------------------------------------------

frag((
    (1, 2),
    "W, H = 256, 256\n"
    "NP = W * H\n"
    "BRUSH = 6\n"
    "# material ids\n"
    "EMPTY, WALL, SAND, WATER, WOOD, FIRE, LAVA, STONE, SMOKE = range(9)\n"
    "N_MAT = 9\n"
    "FIRE_LIFE = 55          # frames a fire burns before dying\n"
    "SMOKE_LIFE = 60\n"
    "IGNITE_CHANCE = 0.22    # per-frame chance a flammable cell next to fire catches\n"
    "LAVA_VISCOSITY = 0.6    # chance lava skips its move this frame (flows slowly)",
))

# --- module-level fields -------------------------------------------------------------

for _name in ("mat", "mat2", "life", "life2", "tgt", "src_of", "density", "flammable", "color", "pixels"):
    frag(((1, 2), f"{_name} = None"))

# --- init + lookup tables ------------------------------------------------------------

INIT_SIM = '''def init_sim(arch=None):
    """Start Taichi, allocate every field once (Metal can't free fields), fill the lookup tables."""
    global mat, mat2, life, life2, tgt, src_of, density, flammable, color, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    mat = ti.field(ti.i32, shape=(W, H))
    mat2 = ti.field(ti.i32, shape=(W, H))
    life = ti.field(ti.i32, shape=(W, H))
    life2 = ti.field(ti.i32, shape=(W, H))
    tgt = ti.field(ti.i32, shape=(W, H))
    src_of = ti.field(ti.i32, shape=(W, H))
    density = ti.field(ti.i32, shape=N_MAT)
    flammable = ti.field(ti.i32, shape=N_MAT)
    color = ti.Vector.field(3, ti.f32, shape=N_MAT)
    pixels = ti.Vector.field(3, ti.f32, shape=(W, H))
    _fill_tables()'''

frag(((1, 3), INIT_SIM))

FILL_TABLES = '''def _fill_tables():
    # density: heavier sinks through lighter; only fluids and sand take part (0 = doesn't sink/swap)
    d = np.zeros(N_MAT, np.int32)
    d[SMOKE], d[FIRE], d[WATER], d[LAVA], d[SAND] = 1, 1, 5, 7, 9
    density.from_numpy(d)
    f = np.zeros(N_MAT, np.int32)
    f[WOOD] = 1
    flammable.from_numpy(f)
    c = np.zeros((N_MAT, 3), np.float32)
    c[EMPTY] = (0.05, 0.06, 0.09)
    c[WALL] = (0.30, 0.30, 0.34)
    c[SAND] = (0.85, 0.72, 0.40)
    c[WATER] = (0.20, 0.45, 0.85)
    c[WOOD] = (0.45, 0.30, 0.16)
    c[FIRE] = (1.00, 0.55, 0.15)
    c[LAVA] = (0.95, 0.35, 0.12)
    c[STONE] = (0.38, 0.36, 0.35)
    c[SMOKE] = (0.50, 0.50, 0.52)
    color.from_numpy(c)'''

frag(((1, 3), FILL_TABLES))

# --- the world: walls and a clean box ------------------------------------------------

frag(((1, 4), "@ti.kernel\ndef clear_all():\n    for i, j in mat:\n        mat[i, j] = EMPTY\n        life[i, j] = 0"))
frag(((1, 4), "@ti.kernel\ndef build_walls():\n    for i, j in mat:\n        if j < 3 or i < 2 or i >= W - 2:\n            mat[i, j] = WALL"))

APPLY_SEED = '''def apply_seed():
    """A clean box: solid floor and side walls, empty air inside — ready to be painted."""
    clear_all()
    build_walls()'''

frag(((1, 4), APPLY_SEED))

# --- material predicates -------------------------------------------------------------

frag(((1, 5), "@ti.func\ndef is_faller(m):\n    return m == SAND or m == WATER or m == LAVA"))
frag(((2, 1), "@ti.func\ndef is_riser(m):\n    return m == SMOKE"))
frag(((2, 1), "@ti.func\ndef is_mover(m):\n    return m == SAND or m == WATER or m == LAVA or m == SMOKE"))
frag((
    (2, 1),
    "@ti.func\ndef empty_at(i, j):\n    r = 0\n    if 0 <= i < W and 0 <= j < H:\n"
    "        if mat[i, j] == EMPTY:\n            r = 1\n    return r",
))

# --- gravity: the serial per-column sweep (race-free, no striping) --------------------

FALL_COLUMNS = '''@ti.kernel
def fall_columns():
    """Straight gravity, one serial sweep per column. Columns never touch each other, so this
    is race-free — and sweeping bottom-up drops a whole solid column by one cell, no gaps."""
    for i in range(W):
        for j in range(1, H):
            m = mat[i, j]
            if is_faller(m) and mat[i, j - 1] == EMPTY:
                mat[i, j - 1] = m
                life[i, j - 1] = life[i, j]
                mat[i, j] = EMPTY
                life[i, j] = 0'''

frag(((1, 5), FALL_COLUMNS))

RISE_COLUMNS = '''@ti.kernel
def rise_columns():
    """Smoke floats up: the same serial column sweep, scanned top-down."""
    for i in range(W):
        for jj in range(1, H):
            j = H - 1 - jj
            m = mat[i, j]
            if is_riser(m) and mat[i, j + 1] == EMPTY:
                mat[i, j + 1] = m
                life[i, j + 1] = life[i, j]
                mat[i, j] = EMPTY
                life[i, j] = 0'''

frag(((3, 1), RISE_COLUMNS))

# --- spreading: parallel propose / resolve into empty cells --------------------------

SELECT_TARGETS = '''@ti.kernel
def select_targets():
    """Each mover picks ONE diagonal/sideways target (straight up/down is the column sweep's job,
    so we only spread when the vertical path is already blocked). Target must be EMPTY."""
    for i, j in mat:
        tgt[i, j] = -1
        src_of[i, j] = NP
        m = mat[i, j]
        if is_mover(m):
            d = 0
            rnd = ti.random()
            if m == SAND and not empty_at(i, j - 1):
                if empty_at(i - 1, j - 1) and empty_at(i + 1, j - 1):
                    d = 2 if rnd < 0.5 else 3
                elif empty_at(i - 1, j - 1):
                    d = 2
                elif empty_at(i + 1, j - 1):
                    d = 3
            elif (m == WATER or m == LAVA) and not empty_at(i, j - 1):
                if not (m == LAVA and rnd < LAVA_VISCOSITY):
                    if empty_at(i - 1, j - 1):
                        d = 2
                    elif empty_at(i + 1, j - 1):
                        d = 3
                    elif empty_at(i - 1, j) and empty_at(i + 1, j):
                        d = 4 if rnd < 0.5 else 5
                    elif empty_at(i - 1, j):
                        d = 4
                    elif empty_at(i + 1, j):
                        d = 5
            elif is_riser(m) and not empty_at(i, j + 1):
                if empty_at(i - 1, j + 1):
                    d = 7
                elif empty_at(i + 1, j + 1):
                    d = 8
            ti_, tj_ = i, j
            if d == 2:
                ti_, tj_ = i - 1, j - 1
            elif d == 3:
                ti_, tj_ = i + 1, j - 1
            elif d == 4:
                ti_ = i - 1
            elif d == 5:
                ti_ = i + 1
            elif d == 7:
                ti_, tj_ = i - 1, j + 1
            elif d == 8:
                ti_, tj_ = i + 1, j + 1
            if d != 0:
                tgt[i, j] = ti_ * H + tj_'''

frag(((2, 1), SELECT_TARGETS))

PROPOSE = '''@ti.kernel
def propose():
    """Many movers may want the same empty cell; the lowest flat index wins, deterministically."""
    for i, j in mat:
        t = tgt[i, j]
        if t >= 0:
            ti.atomic_min(src_of[t // H, t % H], i * H + j)'''

frag(((2, 2), PROPOSE))

RESOLVE = '''@ti.kernel
def resolve():
    """Write the next grid. Targets are only ever EMPTY cells, so 'receiving a mover' and
    'keeping my own material' can never collide on the same cell."""
    for i, j in mat:
        m = mat[i, j]
        if m == EMPTY:
            w = src_of[i, j]
            if w < NP:
                mat2[i, j] = mat[w // H, w % H]
                life2[i, j] = life[w // H, w % H]
            else:
                mat2[i, j] = EMPTY
                life2[i, j] = 0
        else:
            moved = 0
            t = tgt[i, j]
            if t >= 0 and src_of[t // H, t % H] == i * H + j:
                moved = 1
            mat2[i, j] = EMPTY if moved == 1 else m
            life2[i, j] = 0 if moved == 1 else life[i, j]'''

frag(((2, 2), RESOLVE))

frag(((2, 2), "@ti.kernel\ndef commit():\n    for i, j in mat:\n        mat[i, j] = mat2[i, j]\n        life[i, j] = life2[i, j]"))

frag(((2, 2), "def spread():\n    select_targets()\n    propose()\n    resolve()\n    commit()"))

DENSITY_SWAP = '''@ti.kernel
def density_swap(parity: ti.i32):
    """Heavier fluids sink through lighter ones. Only rows of one parity act each call, so the
    (j, j-1) pairs never overlap — an in-place swap with no race."""
    for i, j in mat:
        if (j & 1) == parity and j >= 1:
            a = mat[i, j]
            b = mat[i, j - 1]
            if density[a] > 0 and density[b] > 0 and density[a] > density[b]:
                la, lb = life[i, j], life[i, j - 1]
                mat[i, j] = b
                mat[i, j - 1] = a
                life[i, j] = lb
                life[i, j - 1] = la'''

frag(((2, 3), DENSITY_SWAP))

# --- reactions -----------------------------------------------------------------------

TOUCHES = '''@ti.func
def touches(i, j, what) -> ti.i32:
    r = 0
    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        if not (di == 0 and dj == 0):
            ni, nj = i + di, j + dj
            if 0 <= ni < W and 0 <= nj < H:
                if mat[ni, nj] == what:
                    r = 1
    return r'''

frag(((3, 2), TOUCHES))

REACT = '''@ti.kernel
def react():
    """Local chemistry, double-buffered so every cell reads the same old neighbourhood:
    fire ages and eats fuel, lava quenches to stone on water, water flashes to steam on lava."""
    for i, j in mat:
        m = mat[i, j]
        nm = m
        nl = life[i, j]
        if m == FIRE:
            nl = life[i, j] - 1
            if touches(i, j, WATER) == 1:
                nm, nl = EMPTY, 0                    # doused
            elif nl <= 0:
                if ti.random() < 0.35:
                    nm, nl = SMOKE, SMOKE_LIFE       # embers smoke
                else:
                    nm, nl = EMPTY, 0
        elif m == SMOKE:
            nl = life[i, j] - 1
            if nl <= 0:
                nm = EMPTY
        elif flammable[m] == 1:
            if touches(i, j, FIRE) == 1 or touches(i, j, LAVA) == 1:
                if ti.random() < IGNITE_CHANCE:
                    nm, nl = FIRE, FIRE_LIFE
        elif m == LAVA:
            if touches(i, j, WATER) == 1:
                nm = STONE                           # quenched to rock
        elif m == WATER:
            if touches(i, j, LAVA) == 1 and ti.random() < 0.5:
                nm, nl = SMOKE, SMOKE_LIFE           # flashed to steam
        mat2[i, j] = nm
        life2[i, j] = nl
    for i, j in mat:
        mat[i, j] = mat2[i, j]
        life[i, j] = life2[i, j]'''

frag(((3, 2), REACT))

# --- the tick: three versions --------------------------------------------------------

STEP_V1 = """def step(parity=0):
    fall_columns()"""

STEP_V2 = """def step(parity=0):
    fall_columns()
    spread()
    density_swap(parity & 1)
    density_swap(1 - (parity & 1))"""

STEP_V3 = """def step(parity=0):
    react()
    fall_columns()
    rise_columns()
    spread()
    density_swap(parity & 1)
    density_swap(1 - (parity & 1))"""

frag(((1, 6), STEP_V1), ((2, 3), STEP_V2), ((3, 3), STEP_V3))

PAINT = '''@ti.kernel
def paint(cx: ti.i32, cy: ti.i32, m: ti.i32):
    for i, j in mat:
        if (i - cx) ** 2 + (j - cy) ** 2 < BRUSH * BRUSH:
            if mat[i, j] != WALL or m == EMPTY:
                mat[i, j] = m
                life[i, j] = FIRE_LIFE if m == FIRE else 0'''

frag(((1, 6), PAINT))

COUNT = '''def count(m):
    """Pure numpy: how many cells currently hold material m."""
    return int((mat.to_numpy() == m).sum())'''

frag(((1, 6), COUNT))

# --- render: two versions ------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        m = mat[i, j]
        pixels[i, j] = color[m]"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        m = mat[i, j]
        col = color[m]
        if m == FIRE:
            col = color[FIRE] * (0.55 + 0.45 * ti.random())
        elif m == SMOKE:
            col = color[SMOKE] * (0.4 + 0.6 * life[i, j] / SMOKE_LIFE)
        pixels[i, j] = col"""

frag(((1, 6), RENDER_V1), ((3, 3), RENDER_V2))

# --- main ----------------------------------------------------------------------------

frag((
    (1, 6),
    'PALETTE = [(SAND, "sand"), (WATER, "water"), (WOOD, "wood"), (FIRE, "fire"),\n'
    '           (LAVA, "lava"), (STONE, "stone"), (WALL, "wall"), (EMPTY, "erase")]',
))

MAIN = '''def main():
    init_sim()
    apply_seed()
    # a little scene to play with: a wood platform over a sand dune
    for x in range(90, 170):
        paint(x, 120, WOOD)
    for x in range(60, 110):
        paint(x, 20, SAND)
    gui = ti.GUI("Voxel Sandbox — taichi-academy", res=(W, H), background_color=0x0D0F17)
    brush = SAND
    frame = 0
    while gui.running:
        frame += 1
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "12345678":
                brush = PALETTE[int(e.key) - 1][0]
            elif e.key == "c":
                apply_seed()
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            paint(int(mx * W), int(my * H), brush)
        step(frame)
        render()
        gui.set_image(pixels)
        name = dict(PALETTE).get(brush, "?")
        gui.text(f"brush: {name}   [1-8] pick  drag: paint  [c] clear", (0.02, 0.98), color=0xFFFFFF)
        gui.show()'''

frag(((1, 6), MAIN))

frag(((1, 6), 'if __name__ == "__main__":\n    main()'))

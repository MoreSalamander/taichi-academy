"""Code SOT for project 06 — particle life 3D.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 06-particle-life-3d`.

Evolutions: NUM starts at 800 (chapter 1-3, so brute-force pairwise forces stay
real-time) and jumps to 30000 the moment the spatial hash lands in chapter 4 —
that jump IS the chapter's payoff. seed_particles/apply_seed grow a field at a
time (positions -> +velocity -> +species/colors/rules). compute_forces and
step() each get one clean replacement in chapter 4 (brute force -> grid), while
force_law and integrate never change once written — only HOW neighbors are
found changes, not the physics itself. main() is decomposed into many small
sub-fragments so each step's diff stays small and readable.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="06-particle-life-3d",
    default_file="particle_life_3d.py",
    reference={"particle_life_3d.py": PROJECT_DIR / "reference" / "particle_life_3d.py"},
    chapter_steps={1: 4, 2: 3, 3: 5, 4: 4, 5: 4},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Particle Life 3D: species rules + a spatial hash turn simple math into ecology, at scale."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "NUM = 800"), ((4, 1), "NUM = 30000"))
frag(((3, 1), "NSPEC = 6"))
frag(((1, 2), "WORLD = 1.0"))
frag(((3, 1), "R_MAX = 0.08"))
frag(((3, 1), "BETA = 0.3"))
frag(((4, 1), "GRID = 12"))
frag(((4, 1), "CELL = WORLD / GRID"))
frag(((4, 1), "NCELLS = GRID * GRID * GRID"))
frag(((2, 1), "DT = 0.01"))
frag(((5, 1), "SUBSTEPS = 4"))
frag(((2, 1), "FRICTION = 0.82"))
frag(((3, 1), "FORCE_SCALE = 2.0"))
frag(((5, 3), "PARTICLE_RADIUS = 0.0035"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((2, 1), "vel = None"))
frag(((3, 1), "species = None"))
frag(((3, 1), "colors = None"))
frag(((3, 1), "base_colors = None"))
frag(((3, 1), "rules = None"))
frag(((4, 1), "cell_count = None"))
frag(((4, 1), "cell_start = None"))
frag(((4, 1), "cell_cursor = None"))
frag(((4, 1), "sorted_idx = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'

frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global pos, vel"),
    ((3, 1), f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, species, colors, base_colors, rules"),
    (
        (4, 1),
        f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, species, colors, base_colors, rules\n"
        "    global cell_count, cell_start, cell_cursor, sorted_idx",
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
frag(((1, 2), "    pos = ti.Vector.field(3, ti.f32, shape=NUM)"))
frag(((2, 1), "    vel = ti.Vector.field(3, ti.f32, shape=NUM)"))
frag(((3, 1), "    species = ti.field(ti.i32, shape=NUM)"))
frag(((3, 1), "    colors = ti.Vector.field(3, ti.f32, shape=NUM)"))
frag(((3, 1), "    base_colors = ti.Vector.field(3, ti.f32, shape=NUM)"))
frag(((3, 1), "    rules = ti.field(ti.f32, shape=(NSPEC, NSPEC))"))
frag(((4, 1), "    cell_count = ti.field(ti.i32, shape=NCELLS)"))
frag(((4, 1), "    cell_start = ti.field(ti.i32, shape=NCELLS)"))
frag(((4, 1), "    cell_cursor = ti.field(ti.i32, shape=NCELLS)"))
frag(((4, 1), "    sorted_idx = ti.field(ti.i32, shape=NUM)"))

# --- palette -----------------------------------------------------------------------

PALETTE = '''def species_palette(n):
    """Pure numpy: n distinct hues spaced around the color wheel, as RGB."""
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    h6 = hues * 6.0
    k = h6.astype(np.int32) % 6
    f = h6 - np.floor(h6)
    v, p, q, t = 1.0, 0.15, 1.0 - f, 0.15 + f * 0.85
    table = np.stack(
        [
            np.stack([v + 0 * f, t, p * np.ones_like(f)], axis=1),
            np.stack([q, v + 0 * f, p * np.ones_like(f)], axis=1),
            np.stack([p * np.ones_like(f), v + 0 * f, t], axis=1),
            np.stack([p * np.ones_like(f), q, v + 0 * f], axis=1),
            np.stack([t, p * np.ones_like(f), v + 0 * f], axis=1),
            np.stack([v + 0 * f, p * np.ones_like(f), q], axis=1),
        ],
        axis=0,
    )
    return table[k, np.arange(n)].astype(np.float32)'''

frag(((3, 1), PALETTE))

# --- seeding -------------------------------------------------------------------------

SEED_V1 = '''def seed_particles(n, rng_seed=0):
    """Pure numpy: random positions inside the cube."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)'''

SEED_V2 = '''def seed_particles(n, rng_seed=0):
    """Pure numpy: random positions and velocities inside the cube."""
    rng = np.random.default_rng(rng_seed)
    pos0 = rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)
    vel0 = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    return pos0, vel0'''

SEED_V3 = '''def seed_particles(n, nspec, rng_seed=0):
    """Pure numpy: random positions/velocities/species inside the cube, plus their colors."""
    rng = np.random.default_rng(rng_seed)
    pos0 = rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)
    vel0 = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    spec0 = rng.integers(0, nspec, size=n).astype(np.int32)
    palette = species_palette(nspec)
    col0 = palette[spec0]
    return pos0, vel0, spec0, col0'''

frag(((1, 3), SEED_V1), ((2, 1), SEED_V2), ((3, 2), SEED_V3))

# --- rules ---------------------------------------------------------------------------

RULE_MATRIX = '''def rule_matrix(nspec, rng_seed=0):
    """Pure numpy: random attraction(+)/repulsion(-) coefficient per species pair, in [-1, 1]."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(-1.0, 1.0, size=(nspec, nspec)).astype(np.float32)'''

frag(((3, 2), RULE_MATRIX))

# --- apply seed ------------------------------------------------------------------------

APPLY_V1 = "def apply_seed(pos0):\n    pos.from_numpy(pos0)"

APPLY_V2 = "def apply_seed(seed):\n    pos0, vel0 = seed\n    pos.from_numpy(pos0)\n    vel.from_numpy(vel0)"

APPLY_V3 = (
    "def apply_seed(seed, rules_np):\n"
    "    pos0, vel0, spec0, col0 = seed\n"
    "    pos.from_numpy(pos0)\n"
    "    vel.from_numpy(vel0)\n"
    "    species.from_numpy(spec0)\n"
    "    base_colors.from_numpy(col0)\n"
    "    colors.from_numpy(col0)\n"
    "    rules.from_numpy(rules_np)"
)

frag(((1, 3), APPLY_V1), ((2, 1), APPLY_V2), ((3, 2), APPLY_V3))

# --- spatial hash --------------------------------------------------------------------

FLAT_CELL = """@ti.func
def flat_cell(p) -> ti.i32:
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    ck = ti.min(ti.max(ti.cast(pos[p][2] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID * GRID + cj * GRID + ck"""

frag(((4, 2), FLAT_CELL))

COUNT_CELLS = """@ti.kernel
def count_cells():
    for p in pos:
        cell_count[flat_cell(p)] += 1"""

frag(((4, 2), COUNT_CELLS))

PREFIX_SUM = """@ti.kernel
def prefix_sum():
    for _ in range(1):
        acc = 0
        for c in range(NCELLS):
            cell_start[c] = acc
            acc += cell_count[c]"""

frag(((4, 2), PREFIX_SUM))

SCATTER = """@ti.kernel
def scatter():
    for p in pos:
        idx = flat_cell(p)
        slot = ti.atomic_add(cell_cursor[idx], 1)
        sorted_idx[slot] = p"""

frag(((4, 3), SCATTER))

BUILD_GRID = """def build_grid():
    cell_count.fill(0)
    count_cells()
    prefix_sum()
    cell_cursor.copy_from(cell_start)
    scatter()"""

frag(((4, 3), BUILD_GRID))

# --- the force law -----------------------------------------------------------------

FORCE_LAW = """@ti.func
def force_law(r_norm, a) -> ti.f32:
    f = 0.0
    if r_norm < BETA:
        f = r_norm / BETA - 1.0
    elif r_norm < 1.0:
        f = a * (1.0 - ti.abs(2.0 * r_norm - 1.0 - BETA) / (1.0 - BETA))
    return f"""

frag(((3, 3), FORCE_LAW))

# --- forces: brute force, then grid-accelerated ---------------------------------------

FORCES_BRUTE = """@ti.kernel
def compute_forces():
    for p in pos:
        acc = ti.Vector([0.0, 0.0, 0.0])
        for q in range(NUM):
            if q != p:
                d = pos[q] - pos[p]
                dist = d.norm()
                if 1e-6 < dist < R_MAX:
                    a = rules[species[p], species[q]]
                    f = force_law(dist / R_MAX, a)
                    acc += (d / dist) * f
        vel[p] += acc * FORCE_SCALE * DT"""

FORCES_GRID = """@ti.kernel
def compute_forces():
    for p in pos:
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        ck = ti.min(ti.max(ti.cast(pos[p][2] / CELL, ti.i32), 0), GRID - 1)
        acc = ti.Vector([0.0, 0.0, 0.0])
        for di, dj, dk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if 0 <= ni < GRID and 0 <= nj < GRID and 0 <= nk < GRID:
                nidx = ni * GRID * GRID + nj * GRID + nk
                for slot in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                    q = sorted_idx[slot]
                    if q != p:
                        d = pos[q] - pos[p]
                        dist = d.norm()
                        if 1e-6 < dist < R_MAX:
                            a = rules[species[p], species[q]]
                            f = force_law(dist / R_MAX, a)
                            acc += (d / dist) * f
        vel[p] += acc * FORCE_SCALE * DT"""

frag(((3, 4), FORCES_BRUTE), ((4, 4), FORCES_GRID))

# --- integration -----------------------------------------------------------------------

INTEGRATE = """@ti.kernel
def integrate():
    for p in pos:
        vel[p] *= FRICTION
        newp = pos[p] + vel[p] * DT
        for a in ti.static(range(3)):
            if newp[a] < 0.0:
                newp[a] = -newp[a]
                vel[p][a] = -vel[p][a]
            elif newp[a] > WORLD:
                newp[a] = 2.0 * WORLD - newp[a]
                vel[p][a] = -vel[p][a]
        pos[p] = newp"""

frag(((2, 2), INTEGRATE))

# --- the tick ----------------------------------------------------------------------

frag(
    ((2, 2), "def step():\n    integrate()"),
    ((3, 4), "def step():\n    compute_forces()\n    integrate()"),
    ((4, 4), "def step():\n    build_grid()\n    compute_forces()\n    integrate()"),
)

# --- speed glow ----------------------------------------------------------------------

UPDATE_COLORS = """@ti.kernel
def update_colors():
    for p in pos:
        speed = vel[p].norm()
        glow = ti.math.clamp(speed * 6.0, 0.0, 1.0)
        colors[p] = base_colors[p] * (1.0 - 0.5 * glow) + ti.Vector([1.0, 1.0, 1.0]) * (0.5 * glow)"""

frag(((5, 2), UPDATE_COLORS))

# --- main (many small sub-fragments so each step's diff stays readable) ----------------

frag(
    ((1, 4), "def main():\n    init_sim()\n    apply_seed(seed_particles(NUM))"),
    ((3, 5), "def main():\n    init_sim()\n    apply_seed(seed_particles(NUM, NSPEC), rule_matrix(NSPEC))"),
)
frag(((1, 4), '    window = ti.ui.Window("Particle Life 3D — taichi-academy", (900, 600))'))
frag(((1, 4), "    canvas = window.get_canvas()\n    scene = window.get_scene()"))
frag(((1, 4), "    camera = ti.ui.Camera()\n    camera.position(1.8, 1.8, 1.8)\n    camera.lookat(WORLD / 2, WORLD / 2, WORLD / 2)"))
frag(((2, 3), "    running = True"))
frag(((1, 4), "    while window.running:"))

EVENTS_V1 = """        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False"""

EVENTS_V2 = '''        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "r":
                apply_seed(seed_particles(NUM, rng_seed=np.random.randint(1_000_000)))
            elif e.key == ti.ui.SPACE:
                running = not running'''

EVENTS_V3 = '''        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "r":
                apply_seed(
                    seed_particles(NUM, NSPEC, rng_seed=np.random.randint(1_000_000)),
                    rule_matrix(NSPEC, rng_seed=np.random.randint(1_000_000)),
                )
            elif e.key == ti.ui.SPACE:
                running = not running'''

frag(((1, 4), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 5), EVENTS_V3))

frag(((5, 3), "        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)"))
frag(
    (
        (1, 4),
        "        scene.set_camera(camera)\n"
        "        scene.point_light(pos=(2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))\n"
        "        scene.ambient_light((0.25, 0.25, 0.25))",
    )
)
frag(
    ((2, 3), "        if running:\n            step()"),
    ((5, 1), "        if running:\n            for _ in range(SUBSTEPS):\n                step()"),
)
frag(((5, 2), "        update_colors()"))

PARTICLES_V1 = "        scene.particles(pos, radius=0.006, color=(0.6, 0.8, 1.0))"
PARTICLES_V2 = "        scene.particles(pos, radius=0.006, per_vertex_color=colors)"
PARTICLES_V3 = "        scene.particles(pos, radius=PARTICLE_RADIUS, per_vertex_color=colors)"

frag(((1, 4), PARTICLES_V1), ((3, 5), PARTICLES_V2), ((5, 3), PARTICLES_V3))

frag(((1, 4), "        canvas.scene(scene)"))

HUD = '''        with window.GUI.sub_window("Particle Life 3D", 0.02, 0.02, 0.3, 0.12) as gui:
            gui.text(f"{NUM} particles, {NSPEC} species — {'running' if running else 'paused'}")
            gui.text("[space] pause  [r] reroll ecology  [RMB] orbit  [esc] quit")'''

frag(((5, 4), HUD))
frag(((1, 4), "        window.show()"))
frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

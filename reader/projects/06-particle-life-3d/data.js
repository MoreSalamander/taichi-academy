// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["06-particle-life-3d"] = {
  project: "06-particle-life-3d",
  title: "Particle Life 3D",
  pitch: "Give particles a species and an opinion of each other, then build the spatial hash that lets thousands of them live at once.",
  tier: "medium",
  language: "Python",
  file: "particle_life_3d.py",
  chapters: [
    {
      id: 1, title: "A cube full of dots",
      build: "a 3D GGUI window, a field of random points in a unit cube, and your first orbiting camera.",
      beat: "800 motionless dots floating in a glass cube.",
      steps: [
        { title: "Leave the flat world", adding: "the docstring and both imports.",
          code: `"""Particle Life 3D: species rules + a spatial hash turn simple math into ecology, at scale."""
import numpy as np
import taichi as ti`,
          does: "Arc 2 opens: everything through project 05 lived on a fixed 2D grid, one value per cell. This project's state is a LIST of independent bodies with positions in 3D space — a completely different shape of problem, and it needs a different toolkit (GGUI, Taichi's 3D renderer, instead of the flat ti.GUI canvas).",
          why: "Grids and particles are the two fundamental ways GPU sims represent the world, and you'll use both for the rest of the curriculum. Grids: every cell always has a neighbor at a fixed offset. Particles: neighbors could be ANYWHERE, and finding them is the whole game — that's this project's real subject.",
          see: "Runs clean.",
          checkpoint: "python3 particle_life_3d.py returns silently.",
          recovery: ["Fresh project, fresh venv — same ritual as every project before it."] },
        { title: "800 points in a box", adding: "the population and world-size dials, a position field, and init_sim.",
          code: `NUM = 800
WORLD = 1.0
pos = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(3, ti.f32, shape=NUM)`,
          does: "pos is a field of 3-component vectors — one XYZ position per particle, living in a cube from (0,0,0) to (1,1,1). NUM starts small on purpose: 800.",
          why: "That NUM is a promise you're about to break — on purpose. Chapters 1 through 3 keep it tiny so a brute-force, everybody-checks-everybody approach stays fast enough to feel instant. Chapter 4 introduces the one trick (a spatial hash) that lets NUM jump to 30,000 without the frame rate collapsing. Watch that jump; it's the whole point of this project.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["ti.Vector.field(3, ...) — the 3 is the vector width (x, y, z), separate from shape=NUM, the particle count."] },
        { title: "Scatter them", adding: "a pure-numpy random placer and the upload bridge.",
          code: `def seed_particles(n, rng_seed=0):
    """Pure numpy: random positions inside the cube."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)
def apply_seed(pos0):
    pos.from_numpy(pos0)`,
          does: "n rows of 3 uniform-random numbers in [0, WORLD) — every particle lands somewhere inside the cube, no two the same (astronomically unlikely, anyway).",
          why: "Same split you've used since project 01: numpy decides WHAT the initial state is (easy to reason about, easy to unit test), the GPU field just receives it. That division of labor doesn't change just because the data is now a particle list instead of a grid.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["from_numpy expects shape (n, 3) — one row per particle, matching the field's vector width."] },
        { title: "Open a window into 3D", adding: "the GGUI window, camera, and a first static render.",
          code: `def main():
    init_sim()
    apply_seed(seed_particles(NUM))
    window = ti.ui.Window("Particle Life 3D — taichi-academy", (900, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.8, 1.8, 1.8)
    camera.lookat(WORLD / 2, WORLD / 2, WORLD / 2)
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
        scene.set_camera(camera)
        scene.point_light(pos=(2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.25, 0.25, 0.25))
        scene.particles(pos, radius=0.006, color=(0.6, 0.8, 1.0))
        canvas.scene(scene)
        window.show()
if __name__ == "__main__":
    main()`,
          does: "ti.ui.Window is GGUI — Taichi's real-time 3D renderer, a different animal from the flat ti.GUI canvas every prior project used. A Scene collects what to draw THIS frame (camera, lights, particles); you rebuild that list every single frame, which is why set_camera/point_light/particles/canvas.scene all live inside the while loop. Camera sits at a corner, looking at the cube's center.",
          why: "Notice what's absent: no pixels field, no per-cell kernel painting an image. scene.particles(pos, ...) reads your position field directly and draws a sphere per particle — for particle systems, GGUI IS the renderer. Your only rendering job from here on is choosing where things are and what color they are.",
          see: "A softly lit cube of 800 pale-blue dots, frozen in space, viewed from a corner.",
          checkpoint: "800 motionless dots in 3D. Beat 1.",
          recovery: ["scene.set_camera/point_light/particles must be called EVERY frame, inside the loop — GGUI's scene is rebuilt each time, nothing persists.", "Nothing on screen — check camera.lookat points at the cube's center (WORLD/2), not the corner."] }
      ]
    },
    {
      id: 2, title: "Bounded motion",
      build: "velocity, friction, and walls that bounce instead of leak.",
      beat: "Dots drift and bounce softly inside the cube, no opinions yet.",
      steps: [
        { title: "Give them momentum", adding: "two motion dials, the velocity field, and both growing (adding: velocity to init_sim, seed_particles, apply_seed).",
          code: `DT = 0.01
FRICTION = 0.82
vel = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel
    vel = ti.Vector.field(3, ti.f32, shape=NUM)
def seed_particles(n, rng_seed=0):
    """Pure numpy: random positions and velocities inside the cube."""
    rng = np.random.default_rng(rng_seed)
    pos0 = rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)
    vel0 = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    return pos0, vel0
def apply_seed(seed):
    pos0, vel0 = seed
    pos.from_numpy(pos0)
    vel.from_numpy(vel0)`,
          does: "vel is one more 3-vector per particle: a velocity, seeded to a small random drift. DT is the size of a physics tick; FRICTION (< 1) is a drag coefficient — each tick, velocity gets multiplied by it, bleeding off a little speed forever.",
          why: "seed_particles returning a tuple now, and apply_seed unpacking it, is the same growth pattern as every field this project adds: numpy generates ALL the initial arrays together, one apply_seed call uploads them all together. Notice main()'s call site, apply_seed(seed_particles(NUM)), doesn't need to change at all — Python doesn't care that the return value quietly became a tuple.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["global pos, vel — both names, even though this step only ALLOCATES vel (pos was already handled last chapter)."] },
        { title: "Walls that bounce", adding: "the integrator — friction, then move, then reflect off any wall you'd cross — and the tick function.",
          code: `@ti.kernel
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
        pos[p] = newp
def step():
    integrate()`,
          does: "Each particle only ever reads and writes ITS OWN slot — no neighbors involved yet, so no race to worry about. Damp the velocity, propose a new position, then check each axis independently: cross a wall, and you mirror the overshoot back inside AND flip that axis's velocity sign — a physical bounce, not a teleport.",
          why: "for a in ti.static(range(3)) unrolls into three checks (x, y, z) at compile time, one universal rule applied per axis instead of three hand-written blocks — the same ti.static trick project 05 used for its 4 neighbor directions, here doing double duty as both a loop AND a vector-component selector.",
          see: "Runs clean; nothing visible yet — step() isn't wired into main() until the next beat.",
          checkpoint: "No red text.",
          recovery: ["The mirror formula is 2*WORLD - newp, not -newp, once you're past the wall — you're reflecting the OVERSHOOT, not just negating position.", "Flip vel[p][a]'s sign, not the whole vector — only the axis that hit a wall bounces."] },
        { title: "Set them loose", adding: "the pause toggle, a simple reseed, and the guarded tick call in the loop.",
          code: `    running = True
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "r":
                apply_seed(seed_particles(NUM, rng_seed=np.random.randint(1_000_000)))
            elif e.key == ti.ui.SPACE:
                running = not running
        if running:
            step()`,
          does: "R throws a fresh random seed at the same generator you already wrote; space flips a flag that gates the physics call. Three homes for this fragment: running=True joins main's setup, the two new elif branches join the event loop, and the guarded step() call joins the render loop.",
          why: "Pause-and-inspect is worth building early and cheaply — you'll want it constantly once forces arrive next chapter and things get chaotic. A flag and an if is the whole cost.",
          see: "800 dots drift on their own momentum, damping slowly, bouncing softly off every wall of the cube like a jar of fireflies with nowhere to land. Tap R for a new scatter, space to freeze it.",
          checkpoint: "Drifting, bouncing dots. Beat 2.",
          recovery: ["step() only runs when running is True — check the if sits inside the while loop, after the event loop.", "Nothing moving — vel0's random range is tiny (±0.05) by design; watch for a few seconds."] }
      ]
    },
    {
      id: 3, title: "Give them opinions",
      build: "species, a random attraction/repulsion matrix, and the force law that turns math into ecology.",
      beat: "Colored clumps of like-minded (and not-so-like-minded) particles, checked the slow way.",
      steps: [
        { title: "Six tribes and a rulebook", adding: "four new dials, four new fields, and their lines in init_sim.",
          code: `NSPEC = 6
R_MAX = 0.08
BETA = 0.3
FORCE_SCALE = 2.0
species = None
colors = None
base_colors = None
rules = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, species, colors, base_colors, rules
    species = ti.field(ti.i32, shape=NUM)
    colors = ti.Vector.field(3, ti.f32, shape=NUM)
    base_colors = ti.Vector.field(3, ti.f32, shape=NUM)
    rules = ti.field(ti.f32, shape=(NSPEC, NSPEC))
def species_palette(n):
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
    return table[k, np.arange(n)].astype(np.float32)`,
          does: "species is one small integer per particle (which of 6 tribes). rules is a 6x6 table: rules[a, b] says how species a feels about species b — positive attracts, negative repels, and it need NOT be symmetric (a can love b while b hates a). R_MAX caps how far any of this reaches; BETA marks the boundary of a no-go core zone you'll meet in force_law. species_palette hand-rolls HSV-to-RGB in pure numpy to hand each tribe a distinct, evenly-spaced hue.",
          why: "A rules matrix is the entire 'genome' of an ecology sim — six numbers per pair decide whether you get orbiting pairs, fleeing chains, or a static clump, and you haven't written one line of per-species special-casing to get there. The asymmetry (a loves b, b hates a) is exactly what produces chase-and-flee behavior later, for free.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["rules is 2D: ti.field(ti.f32, shape=(NSPEC, NSPEC)) — a table, not a vector.", "Don't worry about species_palette's HSV math being opaque — it's a black box that produces N distinct colors; what matters is the shape it returns, (n, 3)."] },
        { title: "Cast the tribes, write the rulebook", adding: "species/color assignment in seed_particles, the matrix generator, and every field's line in apply_seed.",
          code: `def seed_particles(n, nspec, rng_seed=0):
    """Pure numpy: random positions/velocities/species inside the cube, plus their colors."""
    rng = np.random.default_rng(rng_seed)
    pos0 = rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)
    vel0 = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    spec0 = rng.integers(0, nspec, size=n).astype(np.int32)
    palette = species_palette(nspec)
    col0 = palette[spec0]
    return pos0, vel0, spec0, col0
def rule_matrix(nspec, rng_seed=0):
    """Pure numpy: random attraction(+)/repulsion(-) coefficient per species pair, in [-1, 1]."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(-1.0, 1.0, size=(nspec, nspec)).astype(np.float32)
def apply_seed(seed, rules_np):
    pos0, vel0, spec0, col0 = seed
    pos.from_numpy(pos0)
    vel.from_numpy(vel0)
    species.from_numpy(spec0)
    base_colors.from_numpy(col0)
    colors.from_numpy(col0)
    rules.from_numpy(rules_np)`,
          does: "Every particle rolls a random tribe (0 to nspec-1); palette[spec0] fancy-indexes straight from 'which tribe' to 'which color' in one line — no loop. rule_matrix is a full 6x6 grid of independent uniform(-1, 1) draws: total chaos, no hand-tuning, and it's exactly this that makes every reseed a different ecosystem.",
          why: "apply_seed now takes TWO arguments — the seed tuple AND the rules matrix — because they're independently regenerable: chapter 5's reroll key will draw a fresh scatter AND a fresh rulebook together, but conceptually they're separate pieces of randomness (where things start vs. how they'll behave).",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["palette[spec0] is numpy fancy indexing — spec0 is an array of indices, and this pulls one row of palette per particle in a single vectorized step, no Python loop.", "Both colors AND base_colors get col0 — colors is what's drawn each frame, base_colors is the untouched original your future speed-glow effect will blend against."] },
        { title: "The clusters formula", adding: "the force law every particle-life sim is built on.",
          code: `@ti.func
def force_law(r_norm, a) -> ti.f32:
    f = 0.0
    if r_norm < BETA:
        f = r_norm / BETA - 1.0
    elif r_norm < 1.0:
        f = a * (1.0 - ti.abs(2.0 * r_norm - 1.0 - BETA) / (1.0 - BETA))
    return f`,
          does: "r_norm is distance divided by R_MAX, squashed to [0, 1]. Below BETA: a universal repulsion, strongest at zero distance, fading to nothing at BETA — nobody is allowed to collapse to a point, REGARDLESS of species (a is never used in this branch). Beyond BETA: a triangular bump, scaled by a — this is the 'social' zone, where being near a species you like PULLS, and one you dislike PUSHES.",
          why: "Two zones, two personalities: physical (everyone repels at point-blank range, or clusters would implode) and social (rule-dependent, or nothing interesting would ever happen). That split — a universal hard core plus a tunable social shell — is the one-paragraph idea behind every particle-life demo you've ever seen, expressed in five lines.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The core-repulsion branch (r_norm < BETA) ignores a completely — that's what makes it universal.", "The triangular bump peaks at r_norm = (1 + BETA) / 2 (the midpoint of the social zone) and is zero at both ends — that's what abs(2*r_norm - 1 - BETA) is computing."] },
        { title: "Check every pair", adding: "the brute-force force kernel and its slot in step (replaces step).",
          code: `@ti.kernel
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
        vel[p] += acc * FORCE_SCALE * DT
def step():
    compute_forces()
    integrate()`,
          does: "Every particle p asks every OTHER particle q: are you within R_MAX? If so, look up how p's species feels about q's species, weigh it by the force law, and accumulate a push/pull direction. d / dist is the unit vector from p toward q — force with direction. Sum it all, scale, add to velocity.",
          why: "This is the honest, obvious algorithm — and it's O(NUM²): 800 particles means 640,000 checks EVERY tick, for every one of the (up to) 4 substeps you'll add later. It works fine at NUM=800. It will NOT work at 30,000 — 900 million checks a tick would crawl. You're about to feel that wall in the next chapter, and then break through it.",
          see: "Runs clean; not wired into the main loop's render yet.",
          checkpoint: "No red text.",
          recovery: ["1e-6 < dist guards against a particle finding itself at dist == 0 — a's own species-pair lookup could otherwise divide by zero in d / dist.", "acc accumulates ALL neighbor contributions before the single vel[p] += at the end — one write per particle, no race."] },
        { title: "See the ecology", adding: "the two-argument seed call, the grown reseed handler, and per-species color in the render.",
          code: `def main():
    init_sim()
    apply_seed(seed_particles(NUM, NSPEC), rule_matrix(NSPEC))
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "r":
                apply_seed(
                    seed_particles(NUM, NSPEC, rng_seed=np.random.randint(1_000_000)),
                    rule_matrix(NSPEC, rng_seed=np.random.randint(1_000_000)),
                )
            elif e.key == ti.ui.SPACE:
                running = not running
        scene.particles(pos, radius=0.006, per_vertex_color=colors)`,
          does: "per_vertex_color=colors replaces the flat color=(...) from chapter 1 — GGUI now paints each particle its OWN species color instead of one shared tint. The reseed handler grows to regenerate BOTH the scatter and the rulebook together, each with its own independent random seed.",
          why: "step() was already wired to call compute_forces() then integrate() last step — nothing left to connect. This is the reveal: same 800 dots, same walls, but now they've been fitted with an interaction matrix.",
          see: "Watch colored dots find their own kind: clumps bloom, chase, and scatter depending on what the random rulebook rolled — sometimes calm clusters, sometimes a slow-motion chase. Tap R for a wholly different ecosystem. It's a little sluggish — that's next chapter's problem to solve.",
          checkpoint: "Species-colored clustering, brute force. Beat 3.",
          recovery: ["per_vertex_color takes a Taichi field directly, not a Python tuple — no parentheses-wrapped color needed anymore.", "All-gray dots — check colors.from_numpy(col0) actually ran; that's in apply_seed, called from both init and the R handler."] }
      ]
    },
    {
      id: 4, title: "Particles at scale",
      build: "a uniform spatial hash — bucket every particle by cell, then only check the 27 cells around you.",
      beat: "The same ecology, spatial-hash-accelerated, and the population leaps from 800 to 30,000.",
      steps: [
        { title: "Thirty thousand, and the bins to hold them", adding: "the grid dials, NUM's real value, four bookkeeping fields, and their lines in init_sim.",
          code: `NUM = 30000
GRID = 12
CELL = WORLD / GRID
NCELLS = GRID * GRID * GRID
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, species, colors, base_colors, rules
    global cell_count, cell_start, cell_cursor, sorted_idx
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=NUM)`,
          does: "NUM jumps 800 to 30,000 — the payoff of this whole chapter, spent on line 1. GRID slices the cube into 12 bins per axis (1,728 cells total); CELL is chosen so each cell is at least as wide as R_MAX — that's the one rule that makes what's coming correct. Four new fields are pure bookkeeping: how many particles per cell, where each cell's slice starts, a scratch cursor, and the sorted particle list itself.",
          why: "The insight behind every spatial hash: if every cell is at least R_MAX wide, then ANY particle within R_MAX of you is guaranteed to be in your cell or one of its 26 neighbors — never further. Checking 27 cells' worth of particles instead of all 30,000 is the entire speedup, and it's exact, not approximate.",
          see: "Runs clean (though brute-force compute_forces at NUM=30,000 would now be painfully slow if you ran it — you're about to replace it).",
          checkpoint: "No red text.",
          recovery: ["CELL = WORLD / GRID, and GRID was chosen (12) specifically so CELL (≈0.083) is just over R_MAX (0.08) — shrink GRID if you ever raise R_MAX, or the 27-cell search stops being exact."] },
        { title: "Which cell, and how many per cell", adding: "the cell-index function and the counting pass.",
          code: `@ti.func
def flat_cell(p) -> ti.i32:
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    ck = ti.min(ti.max(ti.cast(pos[p][2] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID * GRID + cj * GRID + ck
@ti.kernel
def count_cells():
    for p in pos:
        cell_count[flat_cell(p)] += 1
@ti.kernel
def prefix_sum():
    for _ in range(1):
        acc = 0
        for c in range(NCELLS):
            cell_start[c] = acc
            acc += cell_count[c]`,
          does: "flat_cell divides a position by CELL to get integer (x,y,z) cell coordinates (clamped, just in case), then flattens 3 numbers into 1 array index — the same trick as flattening a 2D grid to a 1D array, one dimension further. count_cells has every particle atomically bump its cell's counter. prefix_sum then walks all 1,728 cells IN ORDER, turning per-cell counts into per-cell START OFFSETS — cell c's particles will live in sorted_idx[cell_start[c] : cell_start[c] + cell_count[c]].",
          why: "for _ in range(1): wraps a loop of exactly one — a trick to force a SERIAL inner loop deliberately. This isn't a mistake: a running total (acc) only makes sense computed in order, one cell at a time, so this one kernel deliberately gives up parallelism (1,728 cells is nothing to a GPU) in exchange for a correct, simple scan.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["cell_count[flat_cell(p)] += 1 is an implicit atomic — many particles can share a cell and this runs in parallel, so Taichi atomically increments for you.", "The nested for c in range(NCELLS) inside for _ in range(1) is what makes it serial — only the OUTERMOST loop in a kernel parallelizes."] },
        { title: "Sort them into their bins", adding: "the scatter kernel and the four-step conductor.",
          code: `@ti.kernel
def scatter():
    for p in pos:
        idx = flat_cell(p)
        slot = ti.atomic_add(cell_cursor[idx], 1)
        sorted_idx[slot] = p
def build_grid():
    cell_count.fill(0)
    count_cells()
    prefix_sum()
    cell_cursor.copy_from(cell_start)
    scatter()`,
          does: "cell_cursor starts as a COPY of cell_start (each cell's write pointer begins at its own slice's start). Every particle, in parallel, atomically claims the next free slot in its cell's slice (atomic_add returns the OLD value before incrementing — exactly the slot this particle should use) and writes its own index there. build_grid runs the whole four-act play in order: clear, count, scan, sort.",
          why: "This is a full GPU counting sort, the exact technique real particle engines use to build neighbor grids every single frame. Four kernels, strict order — you can't scan before you've counted, can't scatter before you know where each cell's slice starts. Same 'phases of a tick' discipline project 05 taught for its water pipeline, now bucketing particles instead of routing flux.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["cell_cursor.copy_from(cell_start) MUST happen after prefix_sum and before scatter — scatter consumes and mutates cell_cursor as its write pointer.", "cell_count.fill(0) at the very top — forgetting it means counts pile up across frames instead of resetting."] },
        { title: "27 cells instead of 30,000 particles", adding: "the grid-accelerated force kernel, replacing the brute-force one, and its slot in step.",
          code: `@ti.kernel
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
        vel[p] += acc * FORCE_SCALE * DT
def step():
    build_grid()
    compute_forces()
    integrate()`,
          does: "Find your own cell, then visit its 3x3x3 neighborhood (ti.static(ti.ndrange(...)) unrolls to exactly 27 offsets at compile time). For each in-bounds neighbor cell, walk ONLY the particles bucketed there via cell_start/cell_count, exactly as build_grid arranged them. Everything past that point — the distance check, the rules lookup, force_law, the accumulate — is IDENTICAL to the brute-force version, unchanged, copy-pasted.",
          why: "That's the whole lesson in one observation: the PHYSICS never changed. force_law, integrate, the rules matrix — none of it was touched. Only the neighbor-FINDING strategy changed, from 'ask everyone' to 'ask your neighborhood.' That's the real shape of a performance optimization: isolate the expensive search, leave the correct math alone. (You proved they give the same answer, too — this project's test suite checks the grid version against a brute-force numpy reference, particle for particle.)",
          see: "30,000 particles, and it's SMOOTHER than 800 was with brute force. Watch six colorful tribes bloom into drifting clusters, chase, scatter, and settle — a real ecology, at the scale the pitch promised.",
          checkpoint: "30,000 particles, spatial-hash-accelerated. Beat 4 — the technique that names this arc.",
          recovery: ["Bounds-check EVERY axis before computing nidx — 0 <= ni < GRID and the same for nj, nk — a cell on the cube's edge has fewer than 27 real neighbors.", "The inner for slot in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]) is a half-open range — count, not a second index, sets the end."] }
      ]
    },
    {
      id: 5, title: "A living exhibit",
      build: "smoother motion via substeps, a speed-glow shader, free camera orbit, and a HUD.",
      beat: "Camera you can fly around a glowing, breathing ecology — reroll it anytime.",
      steps: [
        { title: "Smaller ticks, smoother motion", adding: "a substep count and its loop in the render.",
          code: `SUBSTEPS = 4
        if running:
            for _ in range(SUBSTEPS):
                step()`,
          does: "Instead of one big DT-sized physics tick per rendered frame, run 4 smaller ticks. The math is identical each time — step() doesn't know or care how many times it's called.",
          why: "The same trick as project 01's Gray-Scott SUBSTEPS: a single large step per frame can overshoot and oscillate (particles ping-ponging past each other), where several smaller steps settle into stable orbits. Free smoothness, for the cost of calling four kernels instead of one.",
          see: "The ecology looks calmer and more fluid — same rules, steadier motion.",
          checkpoint: "No red text.",
          recovery: ["The for loop belongs INSIDE the if running: block — pausing should stop all four substeps, not just skip the outer call."] },
        { title: "Speed makes them glow", adding: "the color kernel and its call.",
          code: `@ti.kernel
def update_colors():
    for p in pos:
        speed = vel[p].norm()
        glow = ti.math.clamp(speed * 6.0, 0.0, 1.0)
        colors[p] = base_colors[p] * (1.0 - 0.5 * glow) + ti.Vector([1.0, 1.0, 1.0]) * (0.5 * glow)
        update_colors()`,
          does: "Every frame, blend each particle's true species color toward white by up to 50%, scaled by how fast it's currently moving. base_colors (untouched since chapter 3) is the ground truth; colors (what's actually drawn) is the one that gets tinted.",
          why: "This is exactly why chapter 3 kept two color fields instead of one — base_colors is the reference you blend FROM every frame, so the glow never accumulates or drifts; it's always computed fresh from the original hue.",
          see: "Particles caught mid-chase or freshly bounced off a wall flash brighter, almost white at top speed; settled, slow clusters stay in their true saturated color. Motion becomes visible at a glance.",
          checkpoint: "Fast particles glow. No red text.",
          recovery: ["update_colors() must be called EVERY frame, after the physics substeps and before scene.particles — it's reading the current vel, not a cached one."] },
        { title: "Fly around it", adding: "a smaller particle radius, free-look camera controls, and the render call using it.",
          code: `PARTICLE_RADIUS = 0.0035
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        scene.particles(pos, radius=PARTICLE_RADIUS, per_vertex_color=colors)`,
          does: "30,000 dots at chapter 1's radius (0.006) would crowd into a solid haze, so PARTICLE_RADIUS shrinks them. track_user_inputs is a GGUI convenience: hold the right mouse button and the camera flies, WASD-style, wherever you look.",
          why: "One line buys an entire fly camera — GGUI ships this because 'orbit around a 3D scene' is needed by nearly every 3D project from here to the capstones. You'll reach for track_user_inputs again and again for the rest of the curriculum.",
          see: "Hold right-click and fly through the swarm — clusters that looked like distant blobs resolve into hundreds of individual particles orbiting and chasing each other up close.",
          checkpoint: "Free camera orbit. No red text.",
          recovery: ["hold_key=ti.ui.RMB — without holding the right button, a bare mouse move would spin the camera constantly, which feels broken."] },
        { title: "The readout", adding: "the HUD sub-window.",
          code: `        with window.GUI.sub_window("Particle Life 3D", 0.02, 0.02, 0.3, 0.12) as gui:
            gui.text(f"{NUM} particles, {NSPEC} species — {'running' if running else 'paused'}")
            gui.text("[space] pause  [r] reroll ecology  [RMB] orbit  [esc] quit")`,
          does: "GGUI's immediate-mode sub_window draws a floating panel over the 3D scene — population, species count, run state, and the control legend, redrawn fresh every frame just like the rest of the scene.",
          why: "That's Arc 2 opened: grids (Arc 1) traded for particles, and the one idea that makes particles viable at scale — bucket by cell, search only your neighborhood — is now yours for every project ahead that needs it (soft bodies, cloth, snow, ant colonies all lean on this same spatial hash).",
          see: "Reroll a few ecosystems with R: sometimes six mutually hostile tribes shatter into a fine mist, sometimes they braid into slow orbiting knots, sometimes one tribe swallows the rest. The same six numbers-per-pair, wildly different worlds.",
          checkpoint: "HUD reads out live state. Final beat — project 06 complete.",
          recovery: ["window.GUI.sub_window is a context manager (with ... as gui) — the two gui.text calls must be indented inside it."] }
      ]
    }
  ]
};

// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["26-destruction"] = {
  project: "26-destruction",
  title: "Destruction Engine",
  pitch: "Build a city out of nothing but breakable distance bonds — then blow holes in it and shake it to rubble. Structure that holds, fracture that spreads, debris that piles: all from one snapping constraint.",
  tier: "hard",
  language: "Python",
  file: "destruction.py",
  chapters: [
    {
      id: 1, title: "A city that stands",
      build: "a braced lattice of distance bonds, solved with over-relaxed Jacobi PBD under gravity — buildings that hold their shape.",
      beat: "Three blocky buildings sit on the ground, dead still, bearing their own weight.",
      steps: [
        { title: "The floor beneath everything", adding: "the docstring and imports.",
          code: `"""Destruction Engine: buildings are lattices of breakable bonds — explosions and quakes fracture them."""
import numpy as np
import taichi as ti`,
          does: "This is game-tech, not just physics: the goal is destruction that FEELS right — a building that stands solid until you hit it, then fractures where you hit it and collapses under its own weight. The whole engine is built from one idea, the breakable distance bond, so we only need numpy (to lay out the city on the CPU) and Taichi (to solve thousands of bonds every frame on the GPU).",
          why: "Every previous project simulated a material that was already there — a fluid, a cloth, a box of atoms. Destruction is different: it is about a material that CHANGES — solid becoming rubble — and the moment of change (the crack, the collapse) is the whole point. That means the interesting state isn't the physics, it's the topology: which bonds still exist.",
          see: "Runs clean.",
          checkpoint: "python3 destruction.py returns silently.",
          recovery: ["Usual venv setup: source .venv/bin/activate, then run from the project folder."] },
        { title: "The dials and the fields", adding: "every constant and field the engine will need.",
          code: `RES = 640
GRAVITY = 0.5
DT = 1.0 / 60
ITERS = 24
OMEGA = 1.7             # over-relaxation: pushes the Jacobi solver toward rigid faster
FLOOR = 0.04
RADIUS = 0.006          # particle radius, for rubble self-collision
SPACING = 0.014         # rest gap between lattice neighbours
BREAK_STRAIN = 1.4      # a bond snaps once stretched past 1.4x its rest length
DAMP = 0.99
GRID = 64
CELL = 1.0 / GRID
NCELLS = GRID * GRID
MAX_P = 2000
MAX_B = 8000
# buildings: (base_x, width_cols, height_rows) — blocky walls stand; thin spires pancake
BUILDINGS = [(0.08, 22, 18), (0.45, 16, 24), (0.72, 20, 14)]
pos = None
prev = None
delta = None
dn = None
stress = None
b_a = None
b_b = None
b_rest = None
b_broken = None
n_p = None
n_b = None
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None
pixels = None`,
          does: "A building is a grid of particles (pos), and the structure is a list of bonds — b_a and b_b are the two endpoints of each bond, b_rest its resting length, and b_broken a flag that is the entire story of the simulation: 0 while the bond holds, 1 once it snaps. delta and dn are the Position-Based Dynamics scratch space we'll meet in step 5. The two BREAK/BUILDINGS choices are the tuned heart of the demo: BREAK_STRAIN = 1.4 (a bond tolerates 40% stretch before failing) and blocky building shapes, because thin spires pancake under gravity but wide braced walls stand.",
          why: "We store bonds as a flat edge list (b_a, b_b, b_rest) rather than a per-particle neighbour grid because bonds are the thing that changes: fracture is just flipping b_broken[k] to 1, and a broken bond is simply skipped forever after. That single flag, per bond, is what turns a rigid solid into a cloud of rubble — no remeshing, no reconnection, just a boolean.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["MAX_P / MAX_B are capacities: Metal can't free fields, so we allocate the biggest arrays we'll ever need once and track the live counts (n_p, n_b) separately.", "OMEGA and GRAVITY are the two you'll want to fiddle with later — raise gravity or drop OMEGA and you'll watch a tower slowly sag. That sag is the lesson of step 5."] },
        { title: "Allocate once", adding: "init_sim.",
          code: `def init_sim(arch=None):
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
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "The same allocate-everything-up-front pattern the whole curriculum uses, sized to MAX_P / MAX_B so it never has to grow. The collision-grid fields (cell_*, sorted_idx) are allocated now but not used until chapter 2 — allocating them here keeps init_sim a single, stable function we never have to touch again.",
          why: "arch=None picks the GPU and silently falls back to CPU (so the headless tests can force ti.cpu). Allocating once and reusing is not just a Metal workaround — it means a rebuild of the city (the [r] key, later) is a pure data upload, not a reallocation, so it's instant and never stutters.",
          see: "Runs clean — no fields are touched yet.",
          checkpoint: "No red text.",
          recovery: ["If Taichi complains a field is None later, it means init_sim wasn't called before use — main calls it first thing.", "n_p and n_b are shape=() scalar fields (one int each), the live particle and bond counts."] },
        { title: "Lay out the city", adding: "build_structures, bond_rest_lengths, and apply_seed.",
          code: `def build_structures():
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
    return np.array(positions, dtype=np.float32), np.array(edges, dtype=np.int32)
def bond_rest_lengths(positions, edges):
    """Pure numpy: the rest length of each bond is the initial neighbour distance."""
    return np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).astype(np.float32)
def apply_seed():
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
    n_b[None] = nbond`,
          does: "build_structures walks each building's grid of cells, gives every cell an index, then links it to four neighbours: right, up, and both diagonals. Those diagonals are the whole trick. A square held by four edge bonds is a hinge — it folds flat like a cardboard box with no tape on the corners. Add one diagonal and the square becomes two triangles, and a triangle of fixed-length edges is rigid. bond_rest_lengths records each bond's birth length as its target, and apply_seed uploads it all with prev = pos (everything starts at rest, zero velocity).",
          why: "This is the deep reason skyscrapers and bridges are full of diagonal braces and triangular trusses: distance constraints resist stretching but not shearing, so a grid of only horizontal and vertical bonds collapses sideways the instant you lean on it. The diagonal is the cheapest way to buy shear resistance. We're not modelling steel beams — we're modelling exactly the property that makes steel frames stand.",
          see: "Nothing on screen yet — but 1060 particles and ~3900 bonds are now sitting in the fields, a city waiting for a solver.",
          checkpoint: "No red text. build_structures() returns arrays of shape (1060, 2) and (~3904, 2).",
          recovery: ["np.pad fills the unused tail of each MAX-sized field with zeros; only the first n_p / n_b entries are ever touched by the kernels.", "The diagonal (gx+1, gy-1) reaches DOWN-right, so both diagonals of every interior cell are braced, not just one."] },
        { title: "The solver that holds it up", adding: "predict and the four PBD kernels.",
          code: `@ti.kernel
def predict():
    """Verlet: velocity is where you were minus where you are, damped; then fall."""
    for i in range(n_p[None]):
        v = (pos[i] - prev[i]) * DAMP
        prev[i] = pos[i]
        pos[i] = pos[i] + v + ti.Vector([0.0, -GRAVITY]) * DT * DT
@ti.kernel
def clear_delta():
    for i in range(n_p[None]):
        delta[i] = ti.Vector([0.0, 0.0])
        dn[i] = 0.0
@ti.kernel
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
            dn[b] += 1.0
@ti.kernel
def apply_delta():
    """Average every correction landing on a particle, over-relax, and move it once."""
    for i in range(n_p[None]):
        if dn[i] > 0:
            pos[i] += OMEGA * delta[i] / dn[i]
@ti.kernel
def floor_constraint():
    for i in range(n_p[None]):
        if pos[i][1] < FLOOR + RADIUS:
            pos[i][1] = FLOOR + RADIUS
        pos[i][0] = ti.min(ti.max(pos[i][0], RADIUS), 1.0 - RADIUS)`,
          does: "This is Position-Based Dynamics. predict does Verlet integration — it reads velocity as the gap between where a particle was (prev) and is (pos), then lets gravity pull it down. Then the solver repairs the damage: for every intact bond, solve_bonds computes how far it is from its rest length and pushes both ends halfway back. Crucially it doesn't move the particle directly — it ACCUMULATES the push into delta and counts the pushes in dn — so apply_delta can average all the corrections landing on a shared particle and move it once. floor_constraint just clamps everyone above the ground and inside the walls.",
          why: "Why accumulate-then-average instead of moving particles as we go? Because on a GPU thousands of bonds solve in parallel, and a particle shared by eight bonds would get eight simultaneous conflicting writes — a race. Averaging (the Jacobi method) makes the solve order-independent and safe. The cost is slower convergence, which is exactly why OMEGA = 1.7 is there: over-relaxation overshoots each averaged correction by 70%, dragging the lattice toward rigid in 24 iterations instead of hundreds. Drop OMEGA to 1.0 and the tallest tower visibly sags before it settles — the solver can't keep up with gravity.",
          see: "Still assembling — the tick that runs this is the next step.",
          checkpoint: "No red text. Five kernels compile.",
          recovery: ["The b_broken[k] == 0 guard in solve_bonds is the seam where chapter 2 plugs in: a snapped bond simply stops pulling, forever.", "dn is a float count so apply_delta can divide by it; the if dn[i] > 0 guard skips particles no bond touched this frame."] },
        { title: "Run it — a standing city", adding: "step, a plain render, and the main loop.",
          code: `def step():
    predict()
    for _ in range(ITERS):
        clear_delta()
        solve_bonds()
        apply_delta()
        floor_constraint()
@ti.kernel
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
                pixels[x, y] = ti.Vector([0.6, 0.65, 0.7])
def main():
    init_sim()
    apply_seed()
    for _ in range(60):  # let the city settle onto the ground before the player can touch it
        step()
    gui = ti.GUI("Destruction Engine — taichi-academy", res=RES, background_color=0x10121A)
    frame = 0
    while gui.running:
        frame += 1
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step()
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "step wires it together: predict once, then relax the constraints 24 times per frame. render draws each particle as a small grey square over a dark sky and an earth-toned floor band. main builds the city, runs 60 warm-up steps so it settles onto the ground before you can see it, then loops.",
          why: "One prediction, many constraint iterations — that ratio is the PBD dial for stiffness. Each iteration removes some of the remaining error, so more iterations means a more rigid structure; 24 is enough to hold these buildings against gravity without wasting frames. The warm-up loop matters too: without it, the very first frame shows the city dropping the sub-pixel gap between its start height and true rest, a tiny flinch you'd rather the player never see.",
          see: "Three chunky buildings — a wide low one, a tall middle one, a broad short one — standing dead still on the ground. Lean in and the tall one's base is faintly under load, but nothing moves. It just stands there. (That stillness is harder than it looks: it's 24 solver passes a frame quietly winning against gravity.)",
          checkpoint: "A stable skyline that holds its shape indefinitely. Chapter 1 complete.",
          recovery: ["If a building slowly sinks or leans, your solver isn't stiff enough: check OMEGA = 1.7 and ITERS = 24, and that solve_bonds accumulates into delta rather than writing pos directly.", "If it twitches or explodes, DAMP (0.99) may be missing from predict — undamped Verlet on a stiff lattice rings like a struck bell."] }
      ]
    },
    {
      id: 2, title: "Break it",
      build: "a breaking strain, a radial explosion, spatial-hash self-collision so rubble piles, and a stress-coloured render — click to detonate.",
      beat: "Click near a building and it blows apart — a hole punches through, chunks tumble off, debris arcs up and rains down into a pile.",
      steps: [
        { title: "Fracture and the blast", adding: "the explosion constants, break_bonds, and explode.",
          code: `EXPLODE_POWER = 4.5
EXPLODE_RADIUS = 0.16
@ti.kernel
def break_bonds():
    """A bond that has been stretched past its breaking strain is gone for good.
    Judged here, on the freshly predicted positions, before the solver heals the stretch."""
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            if (pos[b_b[k]] - pos[b_a[k]]).norm() > b_rest[k] * BREAK_STRAIN:
                b_broken[k] = 1
@ti.kernel
def explode(mx: ti.f32, my: ti.f32, power: ti.f32, radius: ti.f32):
    """A radial shove, strongest at the blast centre, fading to nothing at its edge."""
    c = ti.Vector([mx, my])
    for i in range(n_p[None]):
        d = pos[i] - c
        r = d.norm() + 1e-6
        if r < radius:
            pos[i] += d / r * power * (1.0 - r / radius) * DT`,
          does: "break_bonds is the entire fracture model: any intact bond stretched past 1.4x its rest length flips to broken, permanently. explode is the disaster — every particle inside the blast radius gets shoved directly away from the centre, hardest at ground zero and fading linearly to nothing at the rim.",
          why: "The subtle part is TIMING, and the docstring flags it: fracture must be judged on the freshly-predicted positions, BEFORE the constraint solver runs. Here's the trap — the solver is deliberately stiff (chapter 1), so if you check for breakage after it runs, it will already have yanked every over-stretched bond back to near its rest length, and nothing ever registers as broken. You have to catch the stretch in the instant after the blast throws the particles apart and before the solver heals it. Get the order wrong and explosions do nothing — the first bug this engine hands you.",
          see: "Assembling — the tick doesn't call these yet.",
          checkpoint: "No red text. Two kernels compile.",
          recovery: ["explode multiplies by DT because it's phrased as a velocity-like shove that Verlet turns into real momentum next frame; that's why a modest-looking POWER launches debris.", "Once b_broken[k] = 1 it is never reset except by a full rebuild — cracks don't heal."] },
        { title: "Rubble that piles", adding: "the spatial hash and self-collision.",
          code: `@ti.func
def flat_cell(i):
    ci = ti.min(ti.max(ti.cast(pos[i][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj
@ti.kernel
def count_cells():
    for c in cell_count:
        cell_count[c] = 0
    for i in range(n_p[None]):
        cell_count[flat_cell(i)] += 1
@ti.kernel
def prefix_sum():
    for _ in range(1):
        a = 0
        for c in range(NCELLS):
            cell_start[c] = a
            a += cell_count[c]
@ti.kernel
def scatter():
    for c in cell_cursor:
        cell_cursor[c] = cell_start[c]
    for i in range(n_p[None]):
        slot = ti.atomic_add(cell_cursor[flat_cell(i)], 1)
        sorted_idx[slot] = i
def build_grid():
    count_cells()
    prefix_sum()
    scatter()
@ti.kernel
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
                            dn[j] += 1.0`,
          does: "This is project 06's counting-sort spatial hash, back again: count how many particles fall in each grid cell, prefix-sum the counts into start offsets, then scatter every particle into a sorted array bucketed by cell. With that, solve_collisions can find each particle's near neighbours by scanning its own cell plus the 8 around it, and push apart any pair closer than 2R — accumulating into the SAME delta / dn the bonds use, so collision and structure solve together.",
          why: "Without self-collision, a shattered building is a ghost: fragments fall straight through each other and through the pile, and you get no rubble heap, no satisfying settle. Collision is what makes debris behave like debris. And the spatial hash is why it's affordable — checking every particle against every other is O(N-squared) and would crawl; the grid makes it O(N), because a particle can only collide with someone in its own neighbourhood of cells.",
          see: "Still assembling — solve_collisions joins the tick next step.",
          checkpoint: "No red text. The hash and collision kernels compile.",
          recovery: ["The j > i guard resolves each pair exactly once, so the push isn't double-counted.", "GRID = 64 makes each cell ~1.6 particle-diameters wide — big enough that a collider is always within the 3x3 block, small enough that cells stay nearly empty. Reusing the same hash for both structure queries and collisions is why it was worth building."] },
        { title: "Wire it up — click to detonate", adding: "compute_stress, the full tick, the stress render, and the mouse handler.",
          code: `@ti.kernel
def compute_stress():
    """Per-particle load: sum of the strain on its surviving bonds (drives the colour)."""
    for i in range(n_p[None]):
        stress[i] = 0.0
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            strain = ti.abs((pos[b_b[k]] - pos[b_a[k]]).norm() / b_rest[k] - 1.0)
            stress[b_a[k]] += strain
            stress[b_b[k]] += strain
def step():
    predict()
    break_bonds()
    build_grid()
    for _ in range(ITERS):
        clear_delta()
        solve_bonds()
        solve_collisions()
        apply_delta()
        floor_constraint()
    compute_stress()
@ti.kernel
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
                pixels[x, y] = col
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = gui.get_cursor_pos()
                explode(mx, my, EXPLODE_POWER, EXPLODE_RADIUS)`,
          does: "The new step() is the finished physics tick: predict, judge fracture on the stretched positions, rebuild the collision grid, run 24 solver passes that now do BOTH bonds and collisions, and finally measure stress. compute_stress sums each particle's bond strains, and render maps that to colour — grey at rest, glowing red-orange under load. The event handler turns a left-click into an explosion at the cursor.",
          why: "The stress colour isn't decoration — it's an X-ray of the forces, and it tells the truth of chapter 1: even a perfectly still building glows faintly red along its base, because the bottom row carries the weight of everything above. When you detonate, you SEE the fracture propagate as a wave of red racing out from the blast a frame ahead of the bonds actually snapping — stress is the warning, the break is the failure. That's the same reason engineers paint real structures with strain gauges.",
          see: "Click near the middle building's base: a hole blows open, a chunk shears off and topples, and debris arcs up and rains back down to pile against the survivors — every piece flashing red as its bonds overload, then greying out once it's free-falling and stress-free. The far buildings stand untouched.",
          checkpoint: "A destructible city — click anywhere to blow a hole. Chapter 2 complete.",
          recovery: ["If clicks do nothing, check the tick order: break_bonds must run AFTER predict and BEFORE the solver, or the stiff solve heals every stretch before it can register as a break.", "If debris sinks into the floor or through itself, solve_collisions isn't inside the iteration loop, or build_grid isn't rebuilt each frame — a stale grid can't find fast-moving fragments."] }
      ]
    },
    {
      id: 3, title: "Shake it",
      build: "an earthquake that shears the ground, a damage readout, and the finished controls.",
      beat: "Hold the quake key and the ground whips sideways under the city — bases shear, towers lean, and hold long enough and the whole skyline slumps into rubble.",
      steps: [
        { title: "The earthquake", adding: "the quake constants and kernel.",
          code: `QUAKE_AMP = 0.012
QUAKE_FREQ = 38.0
@ti.kernel
def quake(t: ti.f32, amp: ti.f32):
    """Shear the ground: particles near the floor are dragged sideways in an oscillation,
    the drag fading with height, so the base whips out from under the mass above it."""
    shift = amp * ti.sin(t * QUAKE_FREQ)
    for i in range(n_p[None]):
        zone = (pos[i][1] - FLOOR) / 0.08
        if zone < 1.0:
            pos[i][0] += shift * (1.0 - zone)`,
          does: "quake drags the ground-level particles sideways in a sine oscillation. The drag is strongest right at the floor and fades to zero at height 0.08 above it (the zone factor), so only the bottom of each building gets whipped back and forth while the mass above tries to stay put.",
          why: "This is why earthquakes destroy buildings from the base up: the ground accelerates sideways, the foundation is forced to follow, but inertia holds the upper floors still — and the difference is shear stress concentrated exactly at the base, which is where our stress colour was already glowing reddest under gravity. The quake doesn't need to be violent; it just has to out-run the solver's ability to hold the base bonds together. 38 rad/s is a fast whip so the direction reverses before the structure can lean into it, maximising the shear.",
          see: "Assembling — the quake needs wiring into main.",
          checkpoint: "No red text. The quake kernel compiles.",
          recovery: ["QUAKE_AMP = 0.012 is tuned: at 0.02 the whole city launches into the air like confetti; at 0.004 nothing breaks. The base shear is exquisitely sensitive to amplitude.", "quake runs before step()'s predict, so the sideways shove becomes velocity through Verlet, just like the explosion."] },
        { title: "Measure the damage", adding: "broken_fraction.",
          code: `def broken_fraction():
    """Pure numpy: share of bonds that have snapped — how ruined the city is."""
    nb = n_b[None]
    return float(b_broken.to_numpy()[:nb].sum()) / max(nb, 1)`,
          does: "A one-line readout: what fraction of all bonds have snapped. Since b_broken is 1 for broken and 0 for intact, summing it and dividing by the bond count gives the ruin ratio directly.",
          why: "This single number is the honest measure of destruction — not how scattered the pixels look, but how much of the structure's connectivity is actually gone. It's what the headless test asserts on (an intact city reads 0%, a blasted one jumps, a quake pushes it higher), and it's the HUD's damage meter. Topology, again: the state that matters is which bonds still exist, and this counts them.",
          see: "Assembling — the HUD that shows it is the last step.",
          checkpoint: "No red text.",
          recovery: ["max(nb, 1) guards against a divide-by-zero if the city were ever empty.", "The [:nb] slice ignores the unused tail of the MAX_B-sized field — only live bonds count."] },
        { title: "The finished engine", adding: "the quake key, the rebuild key, and the HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = gui.get_cursor_pos()
                explode(mx, my, EXPLODE_POWER, EXPLODE_RADIUS)
            elif e.key == "r":
                apply_seed()
                for _ in range(60):
                    step()
        if gui.is_pressed("q", ti.GUI.SPACE):
            quake(frame * DT, QUAKE_AMP)
        gui.text(f"ruined {broken_fraction() * 100:.0f}%", (0.02, 0.98), color=0xFFFFFF)
        gui.text("click: explosion   hold [q]/space: earthquake   [r] rebuild", (0.02, 0.94), color=0xAAAAAA)`,
          does: "The last controls: [r] rebuilds the city from scratch (upload plus a 60-step settle — instant, because nothing reallocates), holding [q] or space runs the quake each frame, and the HUD shows the running ruin percentage and a control legend.",
          why: "The quake is wired as is_pressed, not a one-shot event, so damage ACCUMULATES the longer you hold: a tap shears a few bases loose, a sustained hold walks the ruin meter up past 40% as building after building loses its footing and folds. That's the whole engine's thesis in one gesture — structure is just intact bonds, destruction is just breaking them, and a boolean per bond is enough to turn a skyline into rubble and (with [r]) back again. That completes Arc 6.",
          see: "Rebuild a fresh city, then hold space: the ground blurs sideways, the tall tower's base shears and it leans, topples, and drags its neighbours down in a cascade of red — the ruin meter climbing 0, 12, 30, 45%. Release, and whatever's left settles into a rubble field. Then click the survivors to finish the job, or hit [r] and start over.",
          checkpoint: "A complete destruction sandbox: explosions, earthquakes, rubble, rebuild. Project 26 and Arc 6's game-tech opener complete.",
          recovery: ["is_pressed(\"q\", ti.GUI.SPACE) accepts either key; get_events (one-shot) is for the click and rebuild, is_pressed (held) is for the quake — matching each control to how it should feel.", "If the quake barely does anything, confirm frame is incrementing and passed as frame * DT so the sine actually advances; a frozen t means a constant sideways offset, not a shake."] }
      ]
    }
  ]
};

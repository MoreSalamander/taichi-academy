// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["27-voxel-sandbox"] = {
  project: "27-voxel-sandbox",
  title: "Voxel Sandbox",
  pitch: "A falling-sand world where every pixel is a material. Sand piles, water levels, lava flows and hardens, fire eats wood and smokes — all of it emerging from a handful of rules each cell applies to its neighbours.",
  tier: "hard",
  language: "Python",
  file: "voxel_sandbox.py",
  chapters: [
    {
      id: 1, title: "Grains that fall",
      build: "a grid of materials, a paint brush, and the serial per-column gravity that drops a solid column without the classic striping bug.",
      beat: "Paint sand and it rains down into stacks; paint walls and it piles against them.",
      steps: [
        { title: "The floor beneath everything", adding: "the docstring and imports.",
          code: `"""Voxel Sandbox: a cellular automaton where sand, water, lava, and fire fall, flow, burn, and react."""
import numpy as np
import taichi as ti`,
          does: "A falling-sand game is a cellular automaton: a grid where every cell holds a material, and every frame each cell looks at its neighbours and follows a few simple rules. Nothing in the code knows what a 'pile' or a 'puddle' is — those are just what sand and water DO when a million cells each obey gravity. numpy lays out and inspects the grid; Taichi runs the per-cell rules across the whole grid in parallel.",
          why: "This is the purest 'more is different' project in the whole curriculum: the rules fit on a napkin, but the behaviour — dunes, floods, fires that spread and die — is endlessly rich. It's the same lesson as the ant colony or the molecular-dynamics box, stripped to its cleanest form: local rules, global surprise.",
          see: "Runs clean.",
          checkpoint: "python3 voxel_sandbox.py returns silently.",
          recovery: ["Usual venv setup: source .venv/bin/activate, then run from the project folder."] },
        { title: "Materials and dials", adding: "the grid dimensions, the material ids, and every field.",
          code: `W, H = 256, 256
NP = W * H
BRUSH = 6
# material ids
EMPTY, WALL, SAND, WATER, WOOD, FIRE, LAVA, STONE, SMOKE = range(9)
N_MAT = 9
FIRE_LIFE = 55          # frames a fire burns before dying
SMOKE_LIFE = 60
IGNITE_CHANCE = 0.22    # per-frame chance a flammable cell next to fire catches
LAVA_VISCOSITY = 0.6    # chance lava skips its move this frame (flows slowly)
mat = None
mat2 = None
life = None
life2 = None
tgt = None
src_of = None
density = None
flammable = None
color = None
pixels = None`,
          does: "Every cell is just an integer — one of nine materials. mat is the world; mat2 is its double buffer (some rules read the old grid while writing the new one). life is a per-cell countdown, used only by fire and smoke. tgt and src_of are the scratch space for chapter 2's movement. density, flammable, and color are tiny lookup tables indexed by material.",
          why: "Storing the world as a plain integer grid is what makes a sandbox cheap and universal: adding a new material is adding a number and a few rules, not a new data structure. The whole engine is 256x256 = 65,536 cells, each holding one small int — trivial memory, and every cell updates in parallel on the GPU.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The order of the material ids is arbitrary EXCEPT that EMPTY = 0, so a freshly-zeroed grid is empty air.", "life is separate from mat because most materials ignore it — only fire's burn-down and smoke's fade need a countdown."] },
        { title: "Allocate and tabulate", adding: "init_sim and the lookup tables.",
          code: `def init_sim(arch=None):
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
    _fill_tables()
def _fill_tables():
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
    color.from_numpy(c)`,
          does: "The usual allocate-once pattern, plus three lookup tables filled from numpy. density ranks how heavy each material is (sand 9 sinks through water 5 sinks through nothing); a 0 means 'doesn't take part in sinking.' flammable marks wood. color is each material's RGB. These tables turn per-material behaviour into a single array lookup inside the hot kernels.",
          why: "Table-driven design is the trick that keeps a many-material sandbox from becoming a wall of if-statements. Want denser lava? Change one number. Want glass to be flammable? Flip one flag. The physics kernels never mention specific materials for these properties — they just read the table — so the sandbox grows by editing data, not code.",
          see: "Runs clean — the tables are filled but nothing is drawn yet.",
          checkpoint: "No red text.",
          recovery: ["density is deliberately sparse: WALL, WOOD, STONE, EMPTY all read 0, meaning they never sink or get sunk through — only the fluids and sand rearrange by weight.", "If a colour looks wrong later, it's almost always a swapped RGB triple in this table."] },
        { title: "A box to play in", adding: "clear_all, build_walls, and apply_seed.",
          code: `@ti.kernel
def clear_all():
    for i, j in mat:
        mat[i, j] = EMPTY
        life[i, j] = 0
@ti.kernel
def build_walls():
    for i, j in mat:
        if j < 3 or i < 2 or i >= W - 2:
            mat[i, j] = WALL
def apply_seed():
    """A clean box: solid floor and side walls, empty air inside — ready to be painted."""
    clear_all()
    build_walls()`,
          does: "apply_seed wipes the grid to empty air and lines the bottom and sides with WALL — a container so sand and water have a floor to land on and edges to pool against.",
          why: "Walls are just another material, but an immovable one: no rule ever moves a WALL cell, so it acts as bedrock. Building the container out of the same grid as everything else (rather than special-casing the boundary) means the physics kernels need no edge tests for the floor — they just see WALL and treat it like any solid.",
          see: "Still assembling — nothing renders until the loop exists.",
          checkpoint: "No red text.",
          recovery: ["The floor is 3 cells thick so fast-falling material can't tunnel through it in one step.", "clear_all also zeroes life, so no stale fire countdowns survive a reset."] },
        { title: "Gravity, without the stripes", adding: "is_faller and the serial column sweep.",
          code: `@ti.func
def is_faller(m):
    return m == SAND or m == WATER or m == LAVA
@ti.kernel
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
                life[i, j] = 0`,
          does: "Gravity, done as one serial sweep per column. The outer loop over columns i runs in parallel (Taichi parallelizes the outermost loop); the inner loop over j runs top-to-bottom WITHIN each column, sequentially. Scanning a column from the bottom up, each faller sitting over an empty cell drops one row — and because the cell it just vacated is the next one the sweep looks under, a whole solid column slides down by exactly one, leaving no gaps.",
          why: "This is the subtle heart of GPU falling-sand, and it's worth slowing down for. The obvious approach — every cell in parallel checks the cell below and moves if it's empty — has a fatal flaw: in a solid column, only the bottom grain sees empty space, so it moves alone; next frame the gap it left lets the one above move, and the column shatters into a striped venetian blind of grain-gap-grain. The fix is to make each COLUMN update in a fixed sequential order so the whole stack shifts together, while keeping different COLUMNS parallel and independent — no two columns ever touch the same cell, so there's no data race. Serial where you must, parallel where you can.",
          see: "Assembling — the tick that runs this comes next.",
          checkpoint: "No red text. is_faller and fall_columns compile.",
          recovery: ["Only SAND, WATER, and LAVA fall here; WOOD/STONE/WALL are static, and SMOKE rises (chapter 3).", "The parallel axis is the column index i, the serial axis is height j — swap them and you'd get the striping bug back, plus a race."] },
        { title: "Paint and run", adding: "the tick, the brush, a plain render, and the main loop.",
          code: `def step(parity=0):
    fall_columns()
@ti.kernel
def paint(cx: ti.i32, cy: ti.i32, m: ti.i32):
    for i, j in mat:
        if (i - cx) ** 2 + (j - cy) ** 2 < BRUSH * BRUSH:
            if mat[i, j] != WALL or m == EMPTY:
                mat[i, j] = m
                life[i, j] = FIRE_LIFE if m == FIRE else 0
def count(m):
    """Pure numpy: how many cells currently hold material m."""
    return int((mat.to_numpy() == m).sum())
@ti.kernel
def render():
    for i, j in pixels:
        m = mat[i, j]
        pixels[i, j] = color[m]
PALETTE = [(SAND, "sand"), (WATER, "water"), (WOOD, "wood"), (FIRE, "fire"),
           (LAVA, "lava"), (STONE, "stone"), (WALL, "wall"), (EMPTY, "erase")]
def main():
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
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "step is just gravity for now. paint stamps a brush-sized disc of the current material (but won't overwrite walls unless you're erasing). render colours each cell by its material via the lookup table. main wires up the number keys 1-8 to pick a material, left-drag to paint, [c] to clear, and drops in a starter scene.",
          why: "The paint-and-watch loop is the whole interface — a sandbox is defined by direct manipulation. Note the render is a pure table lookup: one array read per pixel, no branching, which is why it's free even at 65k cells. And step takes a parity argument it doesn't use yet — it's a seat saved for chapter 2's density pass, so main never has to change.",
          see: "Paint sand with the mouse and it rains down, stacking into flat-topped columns against the floor and walls. Erase, paint walls, pour sand over a ledge and watch it spill off the edge. It falls correctly — but a poured heap stacks straight up in towers instead of slumping into a natural cone. That's the next chapter.",
          checkpoint: "An interactive sand-and-wall painter with correct gravity. Chapter 1 complete.",
          recovery: ["If poured sand STRIPES into gaps as it falls, fall_columns isn't sweeping a column serially — check that the inner loop is over j and the outer (parallel) loop is over i.", "If the brush paints over your walls, the mat[i, j] != WALL guard is missing or inverted."] }
      ]
    },
    {
      id: 2, title: "Piles and puddles",
      build: "the parallel propose/resolve spread that gives sand its angle of repose and water its level, plus a density swap so heavy sinks through light.",
      beat: "Poured sand slumps into a natural cone; water finds its level and floods sideways; sand dropped in water sinks to the bottom.",
      steps: [
        { title: "Where should I move?", adding: "the mover predicates and the target picker.",
          code: `@ti.func
def is_riser(m):
    return m == SMOKE
@ti.func
def is_mover(m):
    return m == SAND or m == WATER or m == LAVA or m == SMOKE
@ti.func
def empty_at(i, j):
    r = 0
    if 0 <= i < W and 0 <= j < H:
        if mat[i, j] == EMPTY:
            r = 1
    return r
@ti.kernel
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
                tgt[i, j] = ti_ * H + tj_`,
          does: "This is the sideways half of movement — what a grain does when it CAN'T fall straight down. Each mover, only if the cell directly below it is blocked, picks one target: sand tries its two down-diagonals (randomly choosing when both are open); water tries down-diagonals then straight sideways; lava does the same but skips most frames (LAVA_VISCOSITY) so it oozes; smoke drifts up-diagonally. The choice is stored as a flat index in tgt, or -1 for 'staying put.'",
          why: "The 'only if straight-down is blocked' guard is essential and easy to get wrong. The serial column sweep already owns vertical falling; if a mover ALSO spread diagonally every frame while in mid-air, it would drift sideways as it fell — sand would rain at 45 degrees. By spreading only when it's already resting on something, diagonal motion becomes what it should be: the angle of repose (sand tumbling off a slope that's too steep) and liquid leveling (water creeping outward to flatten). Two passes, cleanly split by direction.",
          see: "Assembling — this only PICKS targets; the next step resolves the conflicts.",
          checkpoint: "No red text. The predicates and select_targets compile.",
          recovery: ["Randomizing left-vs-right when both diagonals are open keeps sand piles symmetric — always preferring one side builds a lopsided, leaning heap.", "src_of is reset to NP (a sentinel meaning 'no source') here, ready for the next step's contest."] },
        { title: "Who gets the cell?", adding: "propose, resolve, commit, and spread.",
          code: `@ti.kernel
def propose():
    """Many movers may want the same empty cell; the lowest flat index wins, deterministically."""
    for i, j in mat:
        t = tgt[i, j]
        if t >= 0:
            ti.atomic_min(src_of[t // H, t % H], i * H + j)
@ti.kernel
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
            life2[i, j] = 0 if moved == 1 else life[i, j]
@ti.kernel
def commit():
    for i, j in mat:
        mat[i, j] = mat2[i, j]
        life[i, j] = life2[i, j]
def spread():
    select_targets()
    propose()
    resolve()
    commit()`,
          does: "Two grains often want the same empty cell — a race. propose settles it: every mover atomically writes its own flat index into its target's src_of, and ti.atomic_min keeps the smallest, so exactly one deterministic winner claims each cell. resolve then writes the next grid: an empty cell that was claimed becomes its winner's material; a mover that sees its own index still standing in its target's src_of moves out (leaving empty); everyone else stays. commit swaps the buffer in.",
          why: "This is the parallel-conflict pattern the whole curriculum keeps returning to (the spatial hash in 06, the fracture solve in 26): when many threads contend for one slot, don't lock — let them all propose and pick a deterministic winner with an atomic. The deep trick that makes resolve simple is that targets are ONLY ever empty cells. That guarantees a cell can either receive a mover OR keep its own material, never both — the two cases can't collide, so there's no ambiguity about what a cell becomes. Restricting where things may move buys you a race-free write with no locks and no second-guessing.",
          see: "Assembling — spread joins the tick next step.",
          checkpoint: "No red text. The four movement pieces compile.",
          recovery: ["atomic_min gives a DETERMINISTIC winner (lowest index), so the same situation always resolves the same way — important for reproducibility.", "A losing mover simply finds its index isn't the winner and stays put; nothing is destroyed, because it never vacated."] },
        { title: "Heavy sinks through light", adding: "the density swap and the fuller tick.",
          code: `@ti.kernel
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
                life[i, j - 1] = la
def step(parity=0):
    fall_columns()
    spread()
    density_swap(parity & 1)
    density_swap(1 - (parity & 1))`,
          does: "The movement passes only move things into EMPTY, so sand dropped onto water would just sit on top. density_swap fixes that: wherever a heavier material sits directly above a lighter one (using the density table), it swaps them. The parity trick makes it race-free — each call only touches rows of one parity, so the (j, j-1) pairs it swaps never overlap. Calling it twice, once per parity, covers the whole grid. The new step runs gravity, then spreading, then both density passes.",
          why: "Density is what sells the sandbox as physical: sand poured into a pond punches through and settles on the bottom while the water burbles up around it; oil would float, lava sinks under water. The parity split is the same idea as chapter 1's serial columns, in miniature — a checkerboard-in-time scheme that guarantees no two swaps ever fight over a shared cell, so it stays correct in parallel without any locks.",
          see: "Now it comes alive. Pour sand and it slumps into a proper cone at its angle of repose instead of a tower; pour water and it spreads out flat, finding its level and flooding across the floor; drop sand into a pool and it sinks straight through to the bottom while the water levels out on top. It finally behaves like a real falling-sand game.",
          checkpoint: "Sand piles, water levels, and heavy sinks through light. Chapter 2 complete.",
          recovery: ["If sand floats on water instead of sinking, check the density table (SAND must out-rank WATER) and that both density passes are called.", "If a diagonal drift appears while things fall, the 'only spread when straight-down is blocked' guard in select_targets is missing."] }
      ]
    },
    {
      id: 3, title: "Fire and stone",
      build: "rising smoke and the reaction pass — fire that eats wood, lava that quenches to stone on water, water that flashes to steam.",
      beat: "Drop fire on wood and it spreads and burns the structure down in smoke; pour water on lava and it freezes into stone with a hiss of steam.",
      steps: [
        { title: "Smoke rises", adding: "the upward column sweep.",
          code: `@ti.kernel
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
                life[i, j] = 0`,
          does: "Smoke is anti-gravity: the exact same serial per-column sweep as fall_columns, but scanned top-DOWN so a column of smoke rises together instead of striping — the mirror image of chapter 1's fall.",
          why: "Reusing the column-sweep structure for rising shows it wasn't a one-off hack for sand — it's the general race-free way to move a whole column of stuff one cell along a fixed axis. Gravity and buoyancy are the same algorithm with the scan direction flipped.",
          see: "Assembling — smoke has no source until reactions make it.",
          checkpoint: "No red text. rise_columns compiles.",
          recovery: ["Scanning top-down matters for the same reason bottom-up did for falling: it lets a solid column shift as one, no gaps.", "Only SMOKE rises (is_riser); everything else is unaffected by this pass."] },
        { title: "Chemistry between neighbours", adding: "the neighbour test and the reaction pass.",
          code: `@ti.func
def touches(i, j, what) -> ti.i32:
    r = 0
    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        if not (di == 0 and dj == 0):
            ni, nj = i + di, j + dj
            if 0 <= ni < W and 0 <= nj < H:
                if mat[ni, nj] == what:
                    r = 1
    return r
@ti.kernel
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
        life[i, j] = life2[i, j]`,
          does: "touches reports whether any of a cell's eight neighbours is a given material. react is the chemistry: fire ages one tick a frame and, when its life runs out, dies (sometimes leaving smoke) — or goes out instantly if it touches water; flammable wood next to fire or lava catches with some probability; lava touching water freezes to stone; water touching lava flashes to steam. It's written into mat2 and copied back, so every cell reads the SAME old neighbourhood.",
          why: "The double buffer is not optional here — it's the difference between a simulation and a bug. If react wrote straight into mat, a fire could ignite a wood cell, and then that freshly-lit cell could ignite ITS neighbour in the very same pass, and flame would teleport across the grid in one frame. Reading the old grid and writing a new one freezes time: every cell sees the world as it was at the frame's start, so fire spreads at one cell per frame, the way it should. That read-old/write-new discipline is the bedrock rule of cellular automata, going back to Conway's Life.",
          see: "Assembling — the tick needs react wired in, next step.",
          checkpoint: "No red text. touches and react compile.",
          recovery: ["IGNITE_CHANCE below 1 makes fire spread raggedly and randomly, like real flame, instead of as a perfect expanding square.", "Fire checks WATER before aging, so a splash douses it immediately — order inside the if matters."] },
        { title: "The living sandbox", adding: "the full tick and the fire-lit render.",
          code: `def step(parity=0):
    react()
    fall_columns()
    rise_columns()
    spread()
    density_swap(parity & 1)
    density_swap(1 - (parity & 1))
@ti.kernel
def render():
    for i, j in pixels:
        m = mat[i, j]
        col = color[m]
        if m == FIRE:
            col = color[FIRE] * (0.55 + 0.45 * ti.random())
        elif m == SMOKE:
            col = color[SMOKE] * (0.4 + 0.6 * life[i, j] / SMOKE_LIFE)
        pixels[i, j] = col`,
          does: "The finished tick: react (chemistry), then fall (gravity), rise (smoke), spread (piling and leveling), and the two density passes. render gains two flourishes — fire flickers with a dab of randomness each frame, and smoke fades as its life runs down (dimmer the closer it is to vanishing).",
          why: "Order in step is a design choice with visible consequences: reacting first means a cell burns based on where its neighbours were, THEN everything moves — so flame and fuel interact before the frame rearranges them. The flicker and fade cost almost nothing (a multiply per pixel) but are what make fire read as fire and smoke as smoke rather than as flat orange and grey blocks — the last 10% that sells the whole thing. That completes the sandbox, and Arc 6.",
          see: "Build a wooden house, set one corner alight, and watch the fire crawl along the beams, collapsing them into falling embers and rising smoke until the whole thing is ash. Pour lava down a hillside and drip water in its path: where they meet, the lava hisses to grey stone and steam boils up. Bury a lava pool under sand, flood it with water from above — the sandbox runs every material's rules at once, and you just watch what emerges. Project 27 and Arc 6 complete.",
          checkpoint: "A complete falling-sand world: sand, water, lava, fire, smoke, stone — all interacting. Project 27 and Arc 6's game-tech pair complete.",
          recovery: ["If fire spreads instantly across all wood in one frame, react is writing into mat instead of mat2 — the double buffer is what paces it to one cell per frame.", "If smoke never disappears, its life isn't counting down, or the nl <= 0 -> EMPTY branch is missing."] }
      ]
    }
  ]
};

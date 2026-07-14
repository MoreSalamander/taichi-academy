// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["24-artificial-life"] = {
  project: "24-artificial-life",
  title: "Artificial Life",
  pitch: "40,000 particles, one turning rule, no notion of a 'cell' anywhere in the code — and cells assemble themselves, grow, and divide.",
  tier: "hard",
  language: "Python",
  file: "artificial_life.py",
  chapters: [
    {
      id: 1, title: "A featureless soup",
      build: "particles, the spatial hash from project 06, a neighbor count, and straight-line drift.",
      beat: "A uniform gas of drifting particles, tinted by how crowded each one's neighborhood is.",
      steps: [
        { title: "Life from a single rule", adding: "the docstring and imports.",
          code: `"""Artificial Life: one turn rule on 40,000 particles — and cells self-assemble from soup."""
import math
import numpy as np
import taichi as ti`,
          does: "This builds the Primordial Particle System (Schmickl et al., 2016): 40,000 identical particles, each obeying ONE rule about which way to turn based on how many neighbors sit to its left versus its right. There is no 'cell' anywhere in the code — no membrane object, no division logic, no reproduction. And yet cells assemble out of the soup, grow membranes, and split in two. It is the purest demonstration in the curriculum that structure can be a side effect of a local rule.",
          why: "Project 23's creatures reproduced because you wrote a reproduce() function. Here NOTHING reproduces on purpose — 'cells' aren't even a concept the program knows about. Watching self-replicating structures emerge from a rule that says nothing about replication is the deepest 'more is different' moment of the whole arc.",
          see: "Runs clean.",
          checkpoint: "python3 artificial_life.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "Particles and the hash", adding: "world/rule dials, the fields, init, and the seeder.",
          code: `RES = 640
WORLD = 1.0
N = 40000
R = 0.011
V = 0.0015
GRID = 88
CELL = WORLD / GRID
NCELLS = GRID * GRID
pos = None
heading = None
neighbors = None
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, neighbors, cell_count, cell_start, cell_cursor, sorted_idx, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    heading = ti.field(ti.f32, shape=N)
    neighbors = ti.field(ti.i32, shape=N)
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=N)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))
def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(rng.uniform(0, WORLD, (N, 2)).astype(np.float32))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N).astype(np.float32))`,
          does: "Each particle is a position and a heading (project 21/23's creature body, stripped to nothing else). The seven cell_* / sorted_idx fields are the spatial hash from project 06 — the whole point of this chapter is that the rule needs each particle's NEIGHBOR COUNT, and counting neighbors for 40,000 particles naively is 1.6 billion distance checks per tick. neighbors[] will cache each particle's count for coloring.",
          why: "R = 0.011 and N = 40000 aren't arbitrary: together they set the average neighbor count to about 15, and THAT number is everything. Too sparse (mean < 10) and nothing coheres; too dense (mean > 30) and cells merge into worms. This system has a narrow density window where life-like cells appear — the same 'critical density' idea as project 22's traffic, here tuned to the edge where structure crystallizes.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The cell_* fields are project 06's counting-sort hash, field-for-field — if the names look familiar, they are.", "GRID = 88 makes each grid cell about 0.0114 wide, just over R, so a particle's neighbors always lie in its own cell plus the 8 around it."] },
        { title: "Count the crowd", adding: "the hash build, toroidal distance, a neighbor counter, straight drift, the anatomy palette, and the render.",
          code: `@ti.func
def flat_cell(p):
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj
@ti.kernel
def count_cells():
    for c in cell_count:
        cell_count[c] = 0
    for p in pos:
        cell_count[flat_cell(p)] += 1
@ti.kernel
def prefix_sum():
    for _ in range(1):
        acc = 0
        for c in range(NCELLS):
            cell_start[c] = acc
            acc += cell_count[c]
@ti.kernel
def scatter():
    for c in cell_cursor:
        cell_cursor[c] = cell_start[c]
    for p in pos:
        idx = flat_cell(p)
        slot = ti.atomic_add(cell_cursor[idx], 1)
        sorted_idx[slot] = p
def build_grid():
    count_cells()
    prefix_sum()
    scatter()
@ti.func
def wrapd(a, b):
    d = a - b
    if d > 0.5 * WORLD:
        d -= WORLD
    if d < -0.5 * WORLD:
        d += WORLD
    return d
@ti.kernel
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
        neighbors[p] = n
@ti.kernel
def drift():
    for p in pos:
        newp = pos[p] + V * ti.Vector([ti.cos(heading[p]), ti.sin(heading[p])])
        pos[p] = ti.Vector([newp[0] % WORLD, newp[1] % WORLD])
@ti.func
def anatomy_color(n):
    col = ti.Vector([0.2, 0.8, 0.3])
    if n > 12:
        col = ti.Vector([0.95, 0.85, 0.3])
    if n > 26:
        col = ti.Vector([0.85, 0.2, 0.6])
    elif n > 18:
        col = ti.Vector([0.6, 0.4, 0.2])
    return col
@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.02, 0.02, 0.04])
    for p in pos:
        xi = ti.cast(pos[p][0] * RES, ti.i32)
        yi = ti.cast(pos[p][1] * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] = anatomy_color(neighbors[p])
def step():
    build_grid()
    count_neighbors()
    drift()
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Artificial Life — taichi-academy", res=RES, background_color=0x050508)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step()
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "The hash (flat_cell, count_cells, prefix_sum, scatter, build_grid) is project 06's counting sort verbatim, on a toroidal grid. count_neighbors walks each particle's 3x3 cell neighborhood and tallies how many others sit within R (wrapd gives the wraparound distance — this world is a donut). drift just slides each particle along its FIXED heading. The palette maps neighbor count to color: green when alone, warming through yellow and brown to magenta when crowded.",
          why: "Building the whole hash to power a COLOR might feel like overkill — but the neighbor count is the one quantity the entire project runs on, and next chapter the turn rule will read the exact same neighborhood. Wiring it up now, while motion is trivial (straight drift), isolates the hash so you can trust it before the rule makes everything move at once.",
          see: "A uniform haze of 40,000 particles streaking in straight lines and wrapping around the edges, faintly speckled where a few randomly drift close enough to tint yellow. Featureless — a gas, not an organism in sight. That blankness is the 'before.'",
          checkpoint: "A drifting gas, colored by crowding. Beat 1.",
          recovery: ["count_neighbors and turn_and_move (next chapter) scan the identical 3x3 neighborhood — the count here is a dry run of the rule's core.", "If the whole screen is one flat green, R may be too small (no neighbors) — confirm R = 0.011 and GRID = 88."] }
      ]
    },
    {
      id: 2, title: "One rule, and life",
      build: "the left/right neighbor split and the single turn rule — cells crystallize out of the soup.",
      beat: "The gas condenses: membranes close into cells, nuclei glow magenta, free particles wander between them like a living tissue.",
      steps: [
        { title: "The whole of the physics", adding: "the two rule constants and the turn-and-move kernel (not wired into the tick yet).",
          code: `ALPHA = math.radians(180.0)
BETA = math.radians(17.0)
@ti.kernel
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
        pos[p] = ti.Vector([newp[0] % WORLD, newp[1] % WORLD])`,
          does: "The rule, in one line of turning: dphi = ALPHA + BETA * N * sign(right - left). Each particle counts neighbors as before, but now SPLIT by side — the sin(atan2(dy,dx) - heading) test asks 'is this neighbor to my left or my right?' (positive means left). Then every particle turns a base ALPHA (a half-turn), plus BETA per neighbor, biased toward whichever side is MORE crowded — and steps forward. That is the entire organism. There is nothing else. The kernel exists now but the tick still calls the chapter-1 drift; the switch is the next step.",
          why: "Trace why this makes cells. A particle with more neighbors on its right turns further right — TOWARD the crowd — so particles curve toward density and orbit it, forming a rotating ring: a membrane. Particles inside a ring see neighbors on all sides (high N, big turns) and get trapped as a nucleus; particles outside see few and wander free. Rings that grow too big become unstable and pinch into two — the cell 'divides.' Every life-like behavior — membranes, nuclei, growth, division — is a consequence of 'turn toward the denser side,' and NONE of it is written down.",
          see: "Runs clean; still a featureless gas — the rule is written but not yet driving anything, because step() hasn't been told to call it.",
          checkpoint: "The rule, defined but dormant. No red text.",
          recovery: ["sign(right - left) with the '1.0 if right > left else -1.0' — flip it and particles flee density instead of seeking it; the soup stays a gas.", "ALPHA = 180 degrees is essential: the base half-turn is what makes orbits close into rings rather than spiraling away."] },
        { title: "Switch it on — and life", adding: "the turn rule into the tick, replacing drift.",
          code: `def step():
    build_grid()
    turn_and_move()`,
          does: "One line swapped — count_neighbors + drift become turn_and_move — and the gas comes alive. Read the anatomy in the colors: green particles are FREE (few neighbors), yellow are MEMBRANE (a ring's wall), brown is CYTOPLASM just inside, magenta is NUCLEUS (the trapped, densest core). The tests confirm the transformation numerically: a uniform soup has almost no dense clusters, and 400 ticks later the population of high-neighbor nucleus particles has multiplied several-fold — condensed from nothing but this rule.",
          why: "The four colors are a microscope, not physics — the simulation only ever computes one number per particle (its neighbor count), and the anatomy is entirely in your reading of it. That's worth sitting with: the 'cell', the 'membrane', the 'nucleus' are names YOU bring to a system that knows only 'how crowded am I.' Emergence is as much in the observer's parsing as in the rule.",
          see: "Within seconds the featureless gas comes alive: particles swirl into rotating rings, rings close into round CELLS with yellow membranes and magenta nuclei, and a haze of green free particles drifts between them like cytoplasm. Watch a large cell wobble, elongate, and split into two daughters. It looks unmistakably biological — and it is 20 lines of a single turn rule.",
          checkpoint: "Cells, from nothing. Beat 2 — the payoff.",
          recovery: ["If cells never form, the density is off — mean neighbor count must sit near 15; too high merges everything, too low prevents rings from closing.", "step() no longer calls count_neighbors — turn_and_move computes the count AND uses it in one pass."] }
      ]
    },
    {
      id: 3, title: "Play god",
      build: "a stir tool and a reseed — reach into the primordial soup and disturb it.",
      beat: "Drag through the tissue to scatter cells apart and watch them re-form; reseed for a whole new world.",
      steps: [
        { title: "A finger in the soup", adding: "a stir radius and the disturbance kernel, wired to the mouse.",
          code: `STIR_RADIUS = 0.05
@ti.kernel
def stir(mx: ti.f32, my: ti.f32):
    for p in pos:
        dx = wrapd(pos[p][0], mx)
        dy = wrapd(pos[p][1], my)
        if dx * dx + dy * dy < STIR_RADIUS * STIR_RADIUS:
            heading[p] = ti.atan2(dy, dx)
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            stir(mx, my)`,
          does: "Every particle within STIR_RADIUS of the cursor is turned to point directly AWAY from it — a shove outward that rips through membranes and blasts cells apart into free particles.",
          why: "The disturbance is the experiment that proves the structure is DYNAMIC, not static: smash a cell to loose particles and, because the rule never stopped running, the survivors immediately begin curving back toward density and re-forming membranes. The cells aren't drawn objects you damaged — they're a living equilibrium that repairs itself, the way a real cell culture reseeds a scratched dish.",
          see: "Drag the mouse across a field of cells: they burst into green confetti in your wake — and behind your cursor, within a few seconds, the confetti curls back into fresh rings and re-crystallizes into cells. You can carve channels through the tissue and watch them heal.",
          checkpoint: "An interactive petri dish. No red text.",
          recovery: ["stir points particles AWAY (atan2 of the offset FROM the cursor) — flip the sign and you'd suck them in, forming an artificial super-cell, also fun to try."] },
        { title: "A fresh primordial soup", adding: "the reseed key and the anatomy legend.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("green: free  yellow: membrane  magenta: nucleus", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to disturb  [r] new soup", (0.02, 0.94), color=0xAAAAAA)`,
          does: "R deals a fresh random soup, and the HUD labels the anatomy so a newcomer can read the tissue.",
          why: "That's Arc 5's emergence quartet complete: ants coordinating through pheromone (21), jams condensing from noise (22), foraging evolving from death (23), and now cells assembling from a single turn rule (24) — four kinds of order, none of them designed, each a different mechanism (stigmergy, instability, selection, self-organization). Every reroll of THIS one retells the same miracle: uniform chaos in, living cells out, from a rule that never once mentions life. Project 25 closes the arc with the physics floor beneath all of it — atoms.",
          see: "Reroll and watch a new gas condense into a new arrangement of cells — different in every particular, identical in the astonishing fact of it. You can no longer look at the tissue and believe nobody programmed the cells. Nobody did.",
          checkpoint: "An endlessly re-seedable living soup. Final beat — project 24 complete.",
          recovery: ["Same reseed idiom as every project since 01 — apply_seed scatters fresh random positions and headings."] }
      ]
    }
  ]
};

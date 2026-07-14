// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["23-evolution"] = {
  project: "23-evolution",
  title: "Evolution",
  pitch: "Nobody writes the foraging logic. A random neural net, energy, death, and mutation — and brains that hunt food appear on their own.",
  tier: "hard",
  language: "Python",
  file: "evolution.py",
  chapters: [
    {
      id: 1, title: "A body and a brain",
      build: "creatures with sensors and a tiny neural network — random weights, driving motion.",
      beat: "Creatures twitch and drift under random neural control — no purpose, no learning yet.",
      steps: [
        { title: "The brain nobody designs", adding: "the docstring and imports.",
          code: `"""Evolution: nobody designs the brain. Sensors, a tiny neural net, mutation — foraging appears."""
import numpy as np
import taichi as ti`,
          does: "This is the arc's deepest idea and the curriculum's first NEURAL project: each creature carries a tiny feed-forward network — five sensors in, one hidden layer, two motor outputs — and NOBODY writes what it does. The weights start random, creatures that forage well by luck reproduce, offspring inherit with small mutations, and over thousands of generations, brains that seek food EMERGE. You will prove it did (evolved brains measurably out-forage random ones) without ever writing a line of foraging logic.",
          why: "Projects 21 and 22 hand-wrote their agents' rules (follow scent; brake for the car ahead). Here the rules are LEARNED. That's the leap from designed emergence to evolved intelligence — and the whole machine is a neural net, a mutation, and a fitness function that is simply 'did you live long enough to breed.'",
          see: "Runs clean.",
          checkpoint: "python3 evolution.py returns silently.",
          recovery: ["Usual venv setup. No trained model, no dataset — the 'training signal' is death."] },
        { title: "State for a mind and a world", adding: "population/network dials and the fields.",
          code: `RES = 512
N_MAX = 3000
START_POP = 400
FOOD_GRID = 128
FOOD_PATCHES = 8
N_SENSORS = 5   # food-left, food-center, food-right, own-energy, bias
N_HIDDEN = 6
N_OUT = 2       # turn, thrust
N_W = N_SENSORS * N_HIDDEN + N_HIDDEN * N_OUT
SENSE_DIST = 0.06
SENSE_ANGLE = 0.6
MAX_TURN = 0.4
MAX_THRUST = 0.006
PI = 3.14159265
pos = None
heading = None
weights = None
food = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, weights, food, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N_MAX)
    heading = ti.field(ti.f32, shape=N_MAX)
    weights = ti.field(ti.f32, shape=(N_MAX, N_W))
    food = ti.field(ti.f32, shape=(FOOD_GRID, FOOD_GRID))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "The key field is weights: a full row of N_W = 42 numbers PER creature — its entire genome AND its entire brain, one and the same. Five sensors (three food-detectors ahead, its own energy, and a constant bias) feed six hidden neurons feed two motors (turn, thrust). 42 weights is the whole nervous system.",
          why: "Storing every creature's network as a row of a big 2D field is what makes 3,000 independent brains run in parallel on the GPU — the forward pass is just indexed arithmetic into weights[c, :]. There's no PyTorch, no autograd; a neural net is revealed here as what it actually is — a few multiply-adds and a squashing function.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["N_W = inputs*hidden + hidden*outputs = 5*6 + 6*2 = 42 — count it once so the brain's weight indexing later makes sense.", "Genome == brain: mutation edits these same 42 numbers that the forward pass reads."] },
        { title: "Sense, think, move", adding: "food patches, the seeder, the food sampler, the neural forward pass, a random-brain mover, and the render.",
          code: `def food_field(rng_seed=0):
    """Pure numpy: a few gaussian food patches, the world's carrying capacity map."""
    rng = np.random.default_rng(rng_seed)
    ii, jj = np.meshgrid(np.arange(FOOD_GRID), np.arange(FOOD_GRID), indexing="ij")
    cap = np.zeros((FOOD_GRID, FOOD_GRID), dtype=np.float32)
    for _ in range(FOOD_PATCHES):
        cx, cy = rng.uniform(0.15, 0.85, 2) * FOOD_GRID
        r = rng.uniform(8, 16)
        cap += np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (2 * r * r))
    return np.clip(cap, 0, 1).astype(np.float32)
def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    p = np.zeros((N_MAX, 2), dtype=np.float32)
    p[:START_POP] = rng.uniform(0.1, 0.9, (START_POP, 2))
    pos.from_numpy(p)
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_MAX).astype(np.float32))
    weights.from_numpy(rng.normal(0, 1.0, (N_MAX, N_W)).astype(np.float32))
    food.from_numpy(food_field(rng_seed))
@ti.func
def food_at(x, y):
    gi = ti.min(ti.max(ti.cast(x * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
    gj = ti.min(ti.max(ti.cast(y * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
    return food[gi, gj]
@ti.func
def brain(c, s0, s1, s2, s3, s4):
    turn = 0.0
    thrust = 0.0
    for k in ti.static(range(N_HIDDEN)):
        base = k * N_SENSORS
        acc = (weights[c, base] * s0 + weights[c, base + 1] * s1 + weights[c, base + 2] * s2
               + weights[c, base + 3] * s3 + weights[c, base + 4] * s4)
        hk = ti.tanh(acc)
        ob = N_HIDDEN * N_SENSORS
        turn += weights[c, ob + k] * hk
        thrust += weights[c, ob + N_HIDDEN + k] * hk
    return ti.tanh(turn), ti.tanh(thrust)
@ti.kernel
def sense_think_move():
    for c in pos:
        p = pos[c]
        hd = heading[c]
        fl = food_at(p[0] + SENSE_DIST * ti.cos(hd + SENSE_ANGLE), p[1] + SENSE_DIST * ti.sin(hd + SENSE_ANGLE))
        fc = food_at(p[0] + SENSE_DIST * ti.cos(hd), p[1] + SENSE_DIST * ti.sin(hd))
        fr = food_at(p[0] + SENSE_DIST * ti.cos(hd - SENSE_ANGLE), p[1] + SENSE_DIST * ti.sin(hd - SENSE_ANGLE))
        turn, thrust = brain(c, fl, fc, fr, 0.5, 1.0)
        hd += turn * MAX_TURN
        sp = ti.max(thrust, 0.0) * MAX_THRUST
        newp = p + sp * ti.Vector([ti.cos(hd), ti.sin(hd)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                hd = PI - hd if k == 0 else -hd
            if newp[k] > 0.99:
                newp[k] = 0.99
                hd = PI - hd if k == 0 else -hd
        pos[c] = newp
        heading[c] = hd
@ti.kernel
def render():
    for i, j in pixels:
        gi = i * FOOD_GRID // RES
        gj = j * FOOD_GRID // RES
        f = food[gi, gj]
        pixels[i, j] = ti.Vector([0.04, 0.10 + 0.40 * f, 0.08])
    for c in pos:
        xi = ti.cast(pos[c][0] * RES, ti.i32)
        yi = ti.cast(pos[c][1] * RES, ti.i32)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = ti.Vector([0.9, 0.9, 0.8])
def step():
    sense_think_move()
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Evolution — taichi-academy", res=RES, background_color=0x000000)
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
          does: "food_field paints a few gaussian patches — the world's carrying-capacity map (project 15's blobs, project 13's normalization). brain is the forward pass, written out longhand: each hidden neuron sums five weighted sensors through tanh, and the two motors sum the six hidden neurons through tanh — outputs land in [-1, 1], a turn fraction and a thrust fraction. sense_think_move wires it up: three food-sensors ahead-left/ahead/ahead-right (project 21's ant antennae), feed the brain, apply its turn and forward thrust.",
          why: "The three-food-sensor layout is deliberately identical to the ant's — but where the ant had a hand-coded 'turn toward the strongest' rule, THIS creature feeds those same three readings into a network and does whatever 42 random numbers say. Right now that's nonsense: random weights produce twitching, circling, wall-hugging. The architecture is complete; the intelligence is absent. Chapters 2-4 supply the only thing missing — a reason for good weights to outlast bad ones.",
          see: "Creatures scattered over a dark field dotted with glowing green food patches, jittering and drifting aimlessly — each driven by its own random brain, none of them caring about the food they blunder across.",
          checkpoint: "Random-brain wandering. Beat 1.",
          recovery: ["brain's sensor 4 is a constant 1.0 bias — the neural equivalent of a threshold, letting a neuron fire even with zero food in view.", "The energy sensor is a placeholder 0.5 for now (no energy exists yet) — chapter 2 makes it real."] }
      ]
    },
    {
      id: 2, title: "Eat or die",
      build: "energy, eating, and death — creatures now have stakes, and the random-brained ones starve.",
      beat: "The population crashes: random brains can't find food, and one by one they wink out.",
      steps: [
        { title: "The cost of living", adding: "energy dials, the alive/energy fields, the full survival loop, regrowth, and the energy-colored render.",
          code: `MOVE_COST = 0.4
LIVE_COST = 0.15
EAT_BITE = 0.5
EAT_GAIN = 9.0
REPRO_ENERGY = 100.0
START_ENERGY = 45.0
FOOD_REGROW = 0.010
energy = None
alive = None
food_cap = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, energy, alive, weights, food, food_cap, pixels
    energy = ti.field(ti.f32, shape=N_MAX)
    alive = ti.field(ti.i32, shape=N_MAX)
    food_cap = ti.field(ti.f32, shape=(FOOD_GRID, FOOD_GRID))
def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    p = np.zeros((N_MAX, 2), dtype=np.float32)
    p[:START_POP] = rng.uniform(0.1, 0.9, (START_POP, 2))
    pos.from_numpy(p)
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_MAX).astype(np.float32))
    e = np.zeros(N_MAX, dtype=np.float32)
    e[:START_POP] = START_ENERGY
    energy.from_numpy(e)
    a = np.zeros(N_MAX, dtype=np.int32)
    a[:START_POP] = 1
    alive.from_numpy(a)
    weights.from_numpy(rng.normal(0, 1.0, (N_MAX, N_W)).astype(np.float32))
    cap = food_field(rng_seed)
    food_cap.from_numpy(cap)
    food.from_numpy(cap.copy())
@ti.kernel
def sense_think_move():
    for c in pos:
        if alive[c] == 1:
            p = pos[c]
            hd = heading[c]
            fl = food_at(p[0] + SENSE_DIST * ti.cos(hd + SENSE_ANGLE), p[1] + SENSE_DIST * ti.sin(hd + SENSE_ANGLE))
            fc = food_at(p[0] + SENSE_DIST * ti.cos(hd), p[1] + SENSE_DIST * ti.sin(hd))
            fr = food_at(p[0] + SENSE_DIST * ti.cos(hd - SENSE_ANGLE), p[1] + SENSE_DIST * ti.sin(hd - SENSE_ANGLE))
            en = ti.min(energy[c] / REPRO_ENERGY, 1.0)
            turn, thrust = brain(c, fl, fc, fr, en, 1.0)
            hd += turn * MAX_TURN
            sp = ti.max(thrust, 0.0) * MAX_THRUST
            newp = p + sp * ti.Vector([ti.cos(hd), ti.sin(hd)])
            for k in ti.static(range(2)):
                if newp[k] < 0.01:
                    newp[k] = 0.01
                    hd = PI - hd if k == 0 else -hd
                if newp[k] > 0.99:
                    newp[k] = 0.99
                    hd = PI - hd if k == 0 else -hd
            pos[c] = newp
            heading[c] = hd
            energy[c] -= LIVE_COST + MOVE_COST * sp / MAX_THRUST
            gi = ti.min(ti.max(ti.cast(newp[0] * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
            gj = ti.min(ti.max(ti.cast(newp[1] * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
            got = ti.atomic_sub(food[gi, gj], EAT_BITE)
            if got > EAT_BITE:
                energy[c] += EAT_GAIN
            else:
                food[gi, gj] += EAT_BITE
            if energy[c] <= 0.0:
                alive[c] = 0
@ti.kernel
def regrow():
    for i, j in food:
        food[i, j] = ti.min(food[i, j] + FOOD_REGROW, food_cap[i, j])
@ti.kernel
def render():
    for i, j in pixels:
        gi = i * FOOD_GRID // RES
        gj = j * FOOD_GRID // RES
        f = food[gi, gj]
        pixels[i, j] = ti.Vector([0.04, 0.10 + 0.40 * f, 0.08])
    for c in pos:
        if alive[c] == 1:
            xi = ti.cast(pos[c][0] * RES, ti.i32)
            yi = ti.cast(pos[c][1] * RES, ti.i32)
            e = ti.min(energy[c] / REPRO_ENERGY, 1.0)
            col = ti.Vector([1.0, 0.9, 0.3]) * e + ti.Vector([0.7, 0.3, 0.9]) * (1 - e)
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                x, y = xi + di, yi + dj
                if 0 <= x < RES and 0 <= y < RES:
                    pixels[x, y] = col
def step():
    sense_think_move()
    regrow()`,
          does: "Now there are stakes. Every creature bleeds LIVE_COST each tick plus MOVE_COST for thrusting; landing on a food cell claims a bite (the atomic-subtract-and-refund from project 21, so food never goes negative) worth EAT_GAIN energy; hit zero energy and alive flips to 0 — permanently. food_cap is the patch map; food regrows toward it. The render colors creatures by energy: gold when fed, purple when starving.",
          why: "This turns wandering into a FITNESS FUNCTION without ever naming one: a creature's weights are 'good' exactly insofar as they keep energy above zero, and 'good' is decided by the world, not a loss function. This is the whole difference between this project and machine learning — there's no gradient, no target output, no teacher. Just death, which quietly removes the weights that don't work.",
          see: "The scatter of creatures flickers gold-to-purple and THINS: random brains that never learned to eat drain to zero and vanish, cell by cell. Within a hundred ticks the field is emptying. Without offspring, it's a mass extinction.",
          checkpoint: "Stakes, and a dying population. Beat 2 — the problem reproduction will solve.",
          recovery: ["The energy sensor (en) is real now — a creature can feel its own hunger, which lets evolved brains later trade off foraging vs resting.", "Every creature-touching line is gated by if alive[c] == 1 — the dead must not move, eat, or draw."] },
        { title: "Count the living", adding: "the population census.",
          code: `n_alive = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, energy, alive, weights, food, food_cap, n_alive, pixels
    n_alive = ti.field(ti.i32, shape=())
@ti.kernel
def count_alive():
    n_alive[None] = 0
    for c in alive:
        if alive[c] == 1:
            ti.atomic_add(n_alive[None], 1)
def population():
    count_alive()
    return int(n_alive[None])`,
          does: "A parallel reduction over the alive flags into one counter — the same atomic-add census as project 15's star count — exposed through a plain python helper for the HUD.",
          why: "Population is the single number that tells the whole story of this project: it will crash now (random brains), stabilize once reproduction arrives (chapter 3), and — the payoff — the SURVIVORS' descendants will be measurably smarter than their random ancestors (chapter 4). One integer, tracking evolution.",
          see: "The number in your HUD (wired next) will read a few hundred and falling — a population in free fall toward zero.",
          checkpoint: "A live census. No red text.",
          recovery: ["count_alive resets n_alive to 0 inside the kernel before the reduction — forgetting the reset accumulates across calls."] },
        { title: "Watch them die", adding: "the generation counter and the population HUD.",
          code: `    gen = 0
        gen += 1
        gui.text(f"generation {gen}  population {population()}", (0.02, 0.98), color=0xFFFFFF)`,
          does: "A tick counter labeled 'generation' and the live population readout — the two dials of the experiment.",
          why: "Run it and confront the honest failure: from START_POP the number only drops. Random neural nets are, overwhelmingly, terrible at staying alive — and nothing yet lets a rare good one leave more copies than a bad one. This is exactly the setup evolution needs: variation (random brains) and selection (death) are both present; the missing third ingredient is HEREDITY.",
          see: "generation climbs, population falls: 400… 340… 210… a countdown to extinction, the food patches glowing untouched over the thinning survivors.",
          checkpoint: "A population in free fall. Beat: heredity is missing.",
          recovery: ["If your population somehow HOLDS steady, check that death (alive[c] = 0 on zero energy) is actually firing — with no deaths there's no selection pressure to see."] }
      ]
    },
    {
      id: 3, title: "Descendants",
      build: "reproduction via a parallel free-list — well-fed creatures split, cloning their brain into a dead slot.",
      beat: "The population stops crashing and locks at carrying capacity — but the brains are frozen, no better than their lucky ancestors.",
      steps: [
        { title: "The reproduction machinery", adding: "the free-list fields, its builder, and cloning reproduction (not yet wired into the tick).",
          code: `free_slots = None
n_free = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, energy, alive, weights, food, food_cap, free_slots, n_free, n_alive, pixels
    free_slots = ti.field(ti.i32, shape=N_MAX)
    n_free = ti.field(ti.i32, shape=())
@ti.kernel
def build_free_list():
    n_free[None] = 0
    for c in alive:
        if alive[c] == 0:
            idx = ti.atomic_add(n_free[None], 1)
            free_slots[idx] = c
@ti.kernel
def reproduce():
    for c in pos:
        if alive[c] == 1 and energy[c] >= REPRO_ENERGY:
            idx = ti.atomic_sub(n_free[None], 1) - 1
            if idx >= 0:
                slot = free_slots[idx]
                energy[c] *= 0.5
                energy[slot] = energy[c]
                pos[slot] = pos[c]
                heading[slot] = ti.random() * (2 * PI)
                alive[slot] = 1
                for w in range(N_W):
                    weights[slot, w] = weights[c, w]`,
          does: "A creature that has eaten its way to REPRO_ENERGY splits: it needs a free slot in the fixed pool, so build_free_list first sweeps up every dead index into free_slots (a parallel counting-scatter — project 06's exact pattern), then each parent atomically POPS one (atomic-subtract the counter; the returned value is your reserved index). Parent energy halves and passes to the child; the child is a CLONE — its 42 weights copied byte for byte. Both functions exist now but nothing calls them yet — the tick is unchanged this step.",
          why: "The free-list is the clean, parallel-safe answer to 'where does a new creature go' — atomic-pop guarantees no two parents claim the same slot, and if the list empties (idx < 0) the birth simply doesn't happen, capping the population without a lock. It's the same slot-claiming discipline as project 06's scatter and 15's star births, now managing a living population that grows AND shrinks.",
          see: "Runs clean; still crashing — the machinery is built but not switched on until the next step wires it into the tick.",
          checkpoint: "Reproduction machinery, dormant. No red text.",
          recovery: ["build_free_list and reproduce are defined but UNUSED this step — that's expected; they compile fine sitting idle.", "Cloning means weights[slot, w] = weights[c, w] with NO change — children are exact copies for now, on purpose."] },
        { title: "Switch it on — and freeze", adding: "reproduction and the free-list into the tick.",
          code: `def step():
    sense_think_move()
    build_free_list()
    reproduce()
    regrow()`,
          does: "Two calls added to the tick — collect the free slots, then let well-fed creatures breed into them — and the crash becomes a stable ecosystem. But look closely at what kind of stability: with cloning, the only brains that can ever exist are exact copies of whichever random ancestors got lucky first.",
          why: "Here's the subtle, important failure. The population holds, but it is EVOLUTIONARILY DEAD. Perfect cloning means the gene pool can never contain anything its founders didn't — foraging skill is frozen at 'beginner's luck' and stays there forever, no matter how many generations pass. Variation and selection and heredity are all present, but without MUTATION there is no NEW variation for selection to act on. That missing ingredient is the entire next chapter, and it's a single line.",
          see: "The free-fall stops: survivors breed, refill the empty slots, and the population settles into a stable band at carrying capacity. But it never sharpens — generation 200 forages exactly as clumsily as generation 5. A living fossil.",
          checkpoint: "Stable but frozen. Beat 3 — mutation is the missing piece.",
          recovery: ["build_free_list must run BEFORE reproduce each tick — the pop reserves slots the sweep just collected.", "If the population still crashes rather than stabilizing, the founders may all have been too clumsy to ever breed — reseed (it's luck of the initial draw)."] }
      ]
    },
    {
      id: 4, title: "Mutation and selection",
      build: "one line — mutate the child's weights — and the loop closes: brains genuinely improve.",
      beat: "Over generations, creatures learn to hunt the food patches — provably better foragers than their ancestors, and nobody taught them.",
      steps: [
        { title: "The one line that changes everything", adding: "mutation dials and the mutating birth.",
          code: `MUT_RATE = 0.10
MUT_SCALE = 0.25
@ti.kernel
def reproduce():
    for c in pos:
        if alive[c] == 1 and energy[c] >= REPRO_ENERGY:
            idx = ti.atomic_sub(n_free[None], 1) - 1
            if idx >= 0:
                slot = free_slots[idx]
                energy[c] *= 0.5
                energy[slot] = energy[c]
                pos[slot] = pos[c]
                heading[slot] = ti.random() * (2 * PI)
                alive[slot] = 1
                for w in range(N_W):
                    m = 0.0
                    if ti.random() < MUT_RATE:
                        m = (ti.random() - 0.5) * 2.0 * MUT_SCALE
                    weights[slot, w] = weights[c, w] + m`,
          does: "The clone becomes a near-clone: each of the 42 inherited weights has a 10% chance of a small random nudge. That is the ONLY change from chapter 3 — five lines where there was one — and it completes Darwin's algorithm: variation (mutation), heredity (inheritance), selection (death). Every ingredient is now present, and evolution is inevitable.",
          why: "This is the payoff, and it is genuinely surprising the first time you see it work: with mutation on, a child that happens to steer slightly better toward food eats slightly more, breeds slightly sooner, and floods its improved weights into the pool — while worse mutants starve out. Repeat for thousands of generations and directed foraging behavior CONDENSES out of random noise, with no designer, no gradient, no goal but survival. The project's tests prove it numerically: freeze the evolved survivors' brains, drop them into a fresh world beside random brains, and the evolved ones gather far more food. Nobody wrote the foraging rule — the world did.",
          see: "Give it a minute of real time (thousands of generations): the loose scatter TIGHTENS onto the green food patches. Creatures that once wandered blindly now visibly hunt — turning toward food, swarming the rich cells, abandoning the barrens. The patches teem; the empty spaces clear. Intelligence, precipitated from death.",
          checkpoint: "Foraging, evolved. Beat 4 — the payoff.",
          recovery: ["MUT_RATE too high (say 0.8) and children barely resemble parents — heredity breaks and evolution can't accumulate gains. MUT_RATE = 0.10 keeps most genes intact while still supplying novelty.", "The mutation is ADDED to the inherited weight, not replacing it — offspring explore NEAR the parent, not from scratch."] },
        { title: "A world to reroll", adding: "the reset key and the legend.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
                gen = 0
        gui.text("gold: fed  purple: starving   [r] new world", (0.02, 0.94), color=0xAAAAAA)`,
          does: "R deals a new patch layout and a fresh random founding population — a new evolutionary run from scratch.",
          why: "Every reroll retells the same story with different specifics: mass die-off of the random founders, stabilization as the lucky ones breed, then the slow tightening onto the patches as mutation-plus-selection sharpens the brains. The DESTINATION is reliable (foraging always emerges) but the PATH never repeats — which lineage wins, which patches get colonized first, is contingent every time. That's evolution: convergent in outcome, unrepeatable in detail.",
          see: "Reroll and watch a fresh world climb from chaos to competence — and notice you can no longer tell, from the final swarming behavior, that these brains were never programmed. That's the whole project in one sentence.",
          checkpoint: "An endlessly rerunnable evolution. Final beat — project 23 complete.",
          recovery: ["Same reseed idiom as every project since 01 — apply_seed rebuilds the founders, the food, and resets everyone to random brains."] }
      ]
    }
  ]
};

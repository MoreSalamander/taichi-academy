// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["21-ant-colony"] = {
  project: "21-ant-colony",
  title: "Ant Colony",
  pitch: "No ant knows the map. Three sensors, one evaporating pheromone, and the colony draws highways anyway.",
  tier: "medium",
  language: "Python",
  file: "ant_colony.py",
  chapters: [
    {
      id: 1, title: "Thirty thousand wanderers",
      build: "an ant swarm with headings, a nest, scattered food, and a pure random walk.",
      beat: "A cloud of ants diffuses out of the nest, drifting at random past food it can't yet remember.",
      steps: [
        { title: "Intelligence without anyone in charge", adding: "the docstring and imports.",
          code: `"""Ant Colony: three sensors, one pheromone, thirty thousand ants — highways emerge."""
import numpy as np
import taichi as ti`,
          does: "Arc 5 begins: systems that adapt. This project is the classic demonstration of STIGMERGY — coordination through marks left in the environment rather than any communication or leader. Each ant follows three dumb rules (wander, follow scent, carry food home); the colony as a whole finds food, builds highways to it, and abandons them when the food runs out.",
          why: "The ant brain you'll write fits in one kernel and contains no pathfinding, no memory of the map, no knowledge of other ants. Everything that LOOKS like planning emerges from agents reading and writing a shared field — the same grid-as-communication-medium idea that Arc 1 used for physics, repurposed as a collective memory.",
          see: "Runs clean.",
          checkpoint: "python3 ant_colony.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "The cast", adding: "world dials and the agent/world fields.",
          code: `RES = 512
GRID = 256
N_ANTS = 30000
NEST = (0.5, 0.5)
NEST_R = 0.03
FOOD_BLOBS = 5
FOOD_R = 10
FOOD_AMOUNT = 60.0
SPEED = 0.0022
WANDER = 0.35
PI = 3.14159265
pos = None
heading = None
food = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, food, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N_ANTS)
    heading = ti.field(ti.f32, shape=N_ANTS)
    food = ti.field(ti.f32, shape=(GRID, GRID))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "Each ant is a position plus a HEADING — an angle, not a velocity vector. Ants move at constant SPEED in whatever direction they face; steering means nudging the angle. The world is a food grid (an amount per cell, like project 05's water) plus the nest, which is just a circle test around a constant.",
          why: "Heading-based agents (angle + constant speed) are the standard representation for creature-like movement — project 06's particles could accelerate freely in any direction, but an ant TURNS. That one change of state representation is most of what makes the motion read as 'walking' instead of 'drifting.'",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["heading is shape=N_ANTS scalars — one angle each, converted to a direction with cos/sin only at the moment of stepping."] },
        { title: "Scatter and wander", adding: "food seeding, the colony reset, a random-walk brain, and the render.",
          code: `def seed_food(rng_seed=0):
    """Pure numpy: food blobs scattered at least a fixed distance from the nest."""
    rng = np.random.default_rng(rng_seed)
    f = np.zeros((GRID, GRID), dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    for _ in range(FOOD_BLOBS):
        while True:
            cx, cy = rng.uniform(0.12, 0.88, 2)
            if np.hypot(cx - NEST[0], cy - NEST[1]) > 0.22:
                break
        d2 = (ii - cx * GRID) ** 2 + (jj - cy * GRID) ** 2
        f += np.where(d2 < FOOD_R**2, FOOD_AMOUNT, 0.0)
    return f
def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(np.tile(np.array(NEST, dtype=np.float32), (N_ANTS, 1)))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_ANTS).astype(np.float32))
    food.from_numpy(seed_food(rng_seed))
@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
        h += (ti.random() - 0.5) * WANDER

        newp = p + SPEED * ti.Vector([ti.cos(h), ti.sin(h)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                h = PI - h if k == 0 else -h
            if newp[k] > 0.99:
                newp[k] = 0.99
                h = PI - h if k == 0 else -h
        pos[a] = newp
        heading[a] = h
@ti.kernel
def render():
    for i, j in pixels:
        gi = i * GRID // RES
        gj = j * GRID // RES
        c = ti.Vector([0.0, 0.0, 0.0])
        if food[gi, gj] > 0:
            c = ti.Vector([0.2, 0.75, 0.25])
        d2 = (i / RES - NEST[0]) ** 2 + (j / RES - NEST[1]) ** 2
        if d2 < NEST_R * NEST_R:
            c = ti.Vector([0.8, 0.5, 0.2])
        pixels[i, j] = c

    for a in pos:
        xi = ti.cast(pos[a][0] * RES, ti.i32)
        yi = ti.cast(pos[a][1] * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] = ti.Vector([0.9, 0.9, 0.85])
def step():
    move_ants()
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Ant Colony — taichi-academy", res=RES, background_color=0x000000)
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
          does: "seed_food rejects any blob position within 0.22 of the nest (rejection sampling — keep rolling until valid, project 14's re-roll idea as a loop). Every ant spawns AT the nest facing a random direction. The brain so far is one line: jitter the heading by a random amount, walk forward, bounce off walls (reflecting the ANGLE: a wall in x flips h to PI - h, the mirror).",
          why: "A pure random walk is the honest baseline for every foraging strategy — mathematically, it DOES find all the food eventually, just excruciatingly slowly and with no way to exploit a find. Everything the next three chapters add is measured against this: same ants, same legs, better information.",
          see: "A white puff of thirty thousand ants blooms out of the orange nest and slowly diffuses across the dark field, drifting straight past the green food blobs with no reaction at all.",
          checkpoint: "A diffusing swarm. Beat 1.",
          recovery: ["The wall bounce reflects the angle, not the velocity — h = PI - h mirrors across a vertical wall, -h across a horizontal one.", "np.tile spawns every ant at the exact same point; the random headings do the spreading."] }
      ]
    },
    {
      id: 2, title: "Pick up, carry home",
      build: "a two-state brain — foragers grab food and become returners who steer home by path integration.",
      beat: "Gold ants stream back to the nest in straight lines — and a race condition creates food from nothing.",
      steps: [
        { title: "Two jobs, one brain", adding: "the state field, path-integration homing, and a naive food pickup.",
          code: `TURN = 0.35
FORAGING, RETURNING = 0, 1
state = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, state, food, pixels
    state = ti.field(ti.i32, shape=N_ANTS)
def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(np.tile(np.array(NEST, dtype=np.float32), (N_ANTS, 1)))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_ANTS).astype(np.float32))
    state.fill(FORAGING)
    food.from_numpy(seed_food(rng_seed))
@ti.func
def wrap_angle(dh):
    while dh > PI:
        dh -= 2 * PI
    while dh < -PI:
        dh += 2 * PI
    return dh
@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
        if state[a] == FORAGING:
            h += (ti.random() - 0.5) * WANDER
        else:
            to_nest = ti.Vector([NEST[0], NEST[1]]) - p
            target = ti.atan2(to_nest[1], to_nest[0])
            h += ti.math.clamp(wrap_angle(target - h), -TURN, TURN)
            h += (ti.random() - 0.5) * 0.1

        newp = p + SPEED * ti.Vector([ti.cos(h), ti.sin(h)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                h = PI - h if k == 0 else -h
            if newp[k] > 0.99:
                newp[k] = 0.99
                h = PI - h if k == 0 else -h
        pos[a] = newp
        heading[a] = h

        gi = ti.min(ti.max(ti.cast(newp[0] * GRID, ti.i32), 0), GRID - 1)
        gj = ti.min(ti.max(ti.cast(newp[1] * GRID, ti.i32), 0), GRID - 1)
        if state[a] == FORAGING:
            if food[gi, gj] > 0.0:
                food[gi, gj] -= 1.0
                state[a] = RETURNING
                heading[a] = h + PI
        else:
            d = newp - ti.Vector([NEST[0], NEST[1]])
            if d.norm() < NEST_R:
                state[a] = FORAGING
                heading[a] = h + PI`,
          does: "The brain becomes a two-state machine, like project 15's gas-or-star: FORAGING ants wander; RETURNING ants compute the exact bearing to the nest (atan2 of the offset), and turn toward it at most TURN radians per tick — wrap_angle keeps the turn direction sane when the angles straddle the ±PI seam. Touch food while foraging: take one unit, about-face (h + PI), switch to RETURNING. Touch the nest while returning: deliver, about-face, forage again.",
          why: "Returners steering straight home is real ant biology, not a cheat: desert ants famously track their own position relative to the nest by PATH INTEGRATION (counting steps and integrating turns) and walk a beeline home from wherever they are. Modeling that as 'the bearing is known' is the standard simplification — the emergent part of this project was never the homeward trip; it's the outward one, coming in chapter 4.",
          see: "The first ants to blunder into food snap gold and cut arrow-straight lines back to the nest, deliver, and head out again. The colony now CYCLES — but each forager still finds food only by dumb luck.",
          checkpoint: "Round trips. No red text.",
          recovery: ["wrap_angle matters: without it, an ant at bearing +3 turning to target -3 would swing the long way around through zero instead of crossing the seam.", "Both state flips end with an about-face (h + PI) — leaving food, you head roughly home; leaving home, roughly back out."] },
        { title: "The food duplication bug", adding: "an atomic claim replacing the naive pickup (replace move_ants' pickup block).",
          code: `@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
        if state[a] == FORAGING:
            h += (ti.random() - 0.5) * WANDER
        else:
            to_nest = ti.Vector([NEST[0], NEST[1]]) - p
            target = ti.atan2(to_nest[1], to_nest[0])
            h += ti.math.clamp(wrap_angle(target - h), -TURN, TURN)
            h += (ti.random() - 0.5) * 0.1

        newp = p + SPEED * ti.Vector([ti.cos(h), ti.sin(h)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                h = PI - h if k == 0 else -h
            if newp[k] > 0.99:
                newp[k] = 0.99
                h = PI - h if k == 0 else -h
        pos[a] = newp
        heading[a] = h

        gi = ti.min(ti.max(ti.cast(newp[0] * GRID, ti.i32), 0), GRID - 1)
        gj = ti.min(ti.max(ti.cast(newp[1] * GRID, ti.i32), 0), GRID - 1)
        if state[a] == FORAGING:
            if food[gi, gj] > 0.0:
                old = ti.atomic_sub(food[gi, gj], 1.0)
                if old > 0.0:
                    state[a] = RETURNING
                    heading[a] = h + PI
                else:
                    food[gi, gj] += 1.0
        else:
            d = newp - ti.Vector([NEST[0], NEST[1]])
            if d.norm() < NEST_R:
                state[a] = FORAGING
                heading[a] = h + PI`,
          does: "The naive pickup was check-then-subtract: 'is there food? then take one.' With thousands of parallel ants crowding one cell, MANY pass the check before ANY subtract lands — this project's prototype measured the food counter at NEGATIVE 1,250, meaning over a thousand phantom meals were carried home. The fix claims first and asks questions after: atomically subtract, inspect the value the atomic returns (the count BEFORE your claim), and if you got there too late — the pre-claim value was already zero or less — refund your subtraction and stay a forager.",
          why: "Check-then-act is THE canonical parallel race, and this is its canonical repair: make the act itself the check, using the atomic's return value. It's the same slot-claiming pattern as project 06's scatter cursor and project 15's star slots — third appearance, now protecting a game-logic invariant (matter is conserved) instead of an index.",
          see: "Visually identical to the last step — which is exactly what makes this class of bug dangerous. The difference is the ledger: food totals now sum exactly, and the tests pin the floor at zero.",
          checkpoint: "Conserved food. Beat 2.",
          recovery: ["ti.atomic_sub RETURNS the old value — that return is the whole mechanism; ignoring it re-creates the race with extra steps.", "The refund (food += 1.0) matters too: without it, late claimants would drive the counter negative even though nobody got food."] }
      ]
    },
    {
      id: 3, title: "The scent of success",
      build: "a pheromone field: returners lay trail, and the trail evaporates and diffuses.",
      beat: "Glowing blue trails condense out of the ants' successful journeys — a memory nobody owns.",
      steps: [
        { title: "Lay the trail", adding: "the pheromone fields, deposit-by-returners, and the trail underlay in render.",
          code: `DEPOSIT = 1.2
trail = None
trail_next = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, state, trail, trail_next, food, pixels
    trail = ti.field(ti.f32, shape=(GRID, GRID))
    trail_next = ti.field(ti.f32, shape=(GRID, GRID))
def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(np.tile(np.array(NEST, dtype=np.float32), (N_ANTS, 1)))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_ANTS).astype(np.float32))
    state.fill(FORAGING)
    trail.fill(0.0)
    food.from_numpy(seed_food(rng_seed))
@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
        if state[a] == FORAGING:
            h += (ti.random() - 0.5) * WANDER
        else:
            to_nest = ti.Vector([NEST[0], NEST[1]]) - p
            target = ti.atan2(to_nest[1], to_nest[0])
            h += ti.math.clamp(wrap_angle(target - h), -TURN, TURN)
            h += (ti.random() - 0.5) * 0.1
            gi = ti.min(ti.max(ti.cast(p[0] * GRID, ti.i32), 0), GRID - 1)
            gj = ti.min(ti.max(ti.cast(p[1] * GRID, ti.i32), 0), GRID - 1)
            trail[gi, gj] += DEPOSIT

        newp = p + SPEED * ti.Vector([ti.cos(h), ti.sin(h)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                h = PI - h if k == 0 else -h
            if newp[k] > 0.99:
                newp[k] = 0.99
                h = PI - h if k == 0 else -h
        pos[a] = newp
        heading[a] = h

        gi = ti.min(ti.max(ti.cast(newp[0] * GRID, ti.i32), 0), GRID - 1)
        gj = ti.min(ti.max(ti.cast(newp[1] * GRID, ti.i32), 0), GRID - 1)
        if state[a] == FORAGING:
            if food[gi, gj] > 0.0:
                old = ti.atomic_sub(food[gi, gj], 1.0)
                if old > 0.0:
                    state[a] = RETURNING
                    heading[a] = h + PI
                else:
                    food[gi, gj] += 1.0
        else:
            d = newp - ti.Vector([NEST[0], NEST[1]])
            if d.norm() < NEST_R:
                state[a] = FORAGING
                heading[a] = h + PI
@ti.kernel
def render():
    for i, j in pixels:
        gi = i * GRID // RES
        gj = j * GRID // RES
        t = ti.min(trail[gi, gj] * 0.08, 1.0)
        c = ti.Vector([0.05, 0.25, 0.5]) * t
        if food[gi, gj] > 0:
            c = ti.Vector([0.2, 0.75, 0.25])
        d2 = (i / RES - NEST[0]) ** 2 + (j / RES - NEST[1]) ** 2
        if d2 < NEST_R * NEST_R:
            c = ti.Vector([0.8, 0.5, 0.2])
        pixels[i, j] = c

    for a in pos:
        xi = ti.cast(pos[a][0] * RES, ti.i32)
        yi = ti.cast(pos[a][1] * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            col = ti.Vector([0.9, 0.9, 0.85])
            if state[a] == RETURNING:
                col = ti.Vector([1.0, 0.8, 0.3])
            pixels[xi, yi] = col`,
          does: "One new rule in the brain: ants CARRYING FOOD drip pheromone wherever they walk. That's the entire writing side of stigmergy. The render gains a deep-blue trail underlay (and gold returners), so the chemical field becomes visible.",
          why: "Only successful ants write. That single asymmetry is what makes the trail MEAN something: pheromone density literally encodes 'food was found this way, recently, this often.' No ant decided to build a map — the map is a side effect of carrying groceries.",
          see: "Faint blue traces appear under every homebound gold ant — still just scribbles between food and nest, because nothing reads them yet, and nothing erases them either.",
          checkpoint: "Trails drawn by success. No red text.",
          recovery: ["The deposit happens in the RETURNING branch only, using the ant's pre-move position — the trail marks where the carrier actually walked."] },
        { title: "Forgetting is a feature", adding: "evaporation + diffusion for the trail, wired into the tick.",
          code: `EVAP = 0.985
DIFFUSE = 0.12
@ti.kernel
def evolve_trail():
    for i, j in trail:
        acc = trail[i, j]
        cnt = 1.0
        for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                acc += trail[ni, nj]
                cnt += 1.0
        avg = acc / cnt
        trail_next[i, j] = (trail[i, j] * (1 - DIFFUSE) + avg * DIFFUSE) * EVAP
@ti.kernel
def copy_trail():
    for i, j in trail:
        trail[i, j] = trail_next[i, j]
def step():
    move_ants()
    evolve_trail()
    copy_trail()`,
          does: "Each tick, the trail blurs slightly into its neighbors (diffusion — so a thin scribble becomes a followable band) and loses 1.5% of itself everywhere (evaporation — so unrefreshed trails fade to nothing in a few hundred ticks). Project 01's double-buffer discipline, applied to a memory instead of a chemical.",
          why: "Evaporation is the colony's forgetting, and forgetting is what makes the memory TRUSTWORTHY: a trail only stays strong while ants keep succeeding along it. When chapter 4's food runs out, this is the mechanism that will retire the obsolete highways automatically — adaptation without a single line of 'detect stale trail' logic.",
          see: "The scribbles soften into smooth glowing ribbons and old wanderings melt away — the field now shows a live, decaying record of recent success only.",
          checkpoint: "A living memory. Beat 3.",
          recovery: ["Diffuse THEN evaporate, both in one kernel pass — order inside the expression matters less than doing both to the same snapshot (trail_next holds the result; copy adopts it)."] }
      ]
    },
    {
      id: 4, title: "Follow the scent",
      build: "three trail sensors for foragers — and the emergence: highways, then adaptation when food dies.",
      beat: "Highways condense between nest and every food source; when a source empties, its highway evaporates.",
      steps: [
        { title: "Three sensors", adding: "trail sampling and sensor steering in the forager branch.",
          code: `SENSE_DIST = 8.0
SENSE_ANGLE = 0.5
@ti.func
def sample_trail(x, y):
    gi = ti.min(ti.max(ti.cast(x * GRID, ti.i32), 0), GRID - 1)
    gj = ti.min(ti.max(ti.cast(y * GRID, ti.i32), 0), GRID - 1)
    return trail[gi, gj]
@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
        if state[a] == FORAGING:
            sd = SENSE_DIST / GRID
            vl = sample_trail(p[0] + sd * ti.cos(h + SENSE_ANGLE), p[1] + sd * ti.sin(h + SENSE_ANGLE))
            vc = sample_trail(p[0] + sd * ti.cos(h), p[1] + sd * ti.sin(h))
            vr = sample_trail(p[0] + sd * ti.cos(h - SENSE_ANGLE), p[1] + sd * ti.sin(h - SENSE_ANGLE))
            if vl > vc and vl > vr:
                h += TURN
            elif vr > vc and vr > vl:
                h -= TURN
            h += (ti.random() - 0.5) * WANDER
        else:
            to_nest = ti.Vector([NEST[0], NEST[1]]) - p
            target = ti.atan2(to_nest[1], to_nest[0])
            h += ti.math.clamp(wrap_angle(target - h), -TURN, TURN)
            h += (ti.random() - 0.5) * 0.1
            gi = ti.min(ti.max(ti.cast(p[0] * GRID, ti.i32), 0), GRID - 1)
            gj = ti.min(ti.max(ti.cast(p[1] * GRID, ti.i32), 0), GRID - 1)
            trail[gi, gj] += DEPOSIT

        newp = p + SPEED * ti.Vector([ti.cos(h), ti.sin(h)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                h = PI - h if k == 0 else -h
            if newp[k] > 0.99:
                newp[k] = 0.99
                h = PI - h if k == 0 else -h
        pos[a] = newp
        heading[a] = h

        gi = ti.min(ti.max(ti.cast(newp[0] * GRID, ti.i32), 0), GRID - 1)
        gj = ti.min(ti.max(ti.cast(newp[1] * GRID, ti.i32), 0), GRID - 1)
        if state[a] == FORAGING:
            if food[gi, gj] > 0.0:
                old = ti.atomic_sub(food[gi, gj], 1.0)
                if old > 0.0:
                    state[a] = RETURNING
                    heading[a] = h + PI
                else:
                    food[gi, gj] += 1.0
        else:
            d = newp - ti.Vector([NEST[0], NEST[1]])
            if d.norm() < NEST_R:
                state[a] = FORAGING
                heading[a] = h + PI`,
          does: "Each forager samples the trail at three points ahead of it — left-forward, dead ahead, right-forward, each SENSE_DIST cells out — and turns toward whichever smells strongest (ties or all-zero: no turn, keep wandering). Reading plus the existing writing closes the stigmergy loop: success writes, searchers read, following leads to success, success writes again.",
          why: "This is the whole 'AI' of the project: nine lines, no memory, no map. And it contains its own amplifier — the more ants a trail recruits, the more of them find food and reinforce it, so strong trails get stronger while evaporation eats the weak ones. Positive feedback plus decay is the engine of every stigmergic system, from real ants to internet recommendation loops.",
          see: "The transformation is startling: within a minute, the diffuse cloud CONDENSES into bright highways connecting nest to every food blob, each a two-way stream — white outbound, gold inbound. Keep watching: when a food blob runs dry, its gold traffic stops, its highway fades, and the colony's wanderers rediscover the remaining sources. Adaptation, uncommanded.",
          checkpoint: "Highways, then adaptation. Beat 4 — the payoff.",
          recovery: ["SENSE_DIST is in GRID CELLS (divided down to world units inline) — a real bug bit this project's own test suite when a hand-planted trail stripe was placed one cell beyond the sensors' actual reach and every sensor read zero.", "Sensors read the trail; the pickup still reads food — two different fields, easy to cross up."] },
        { title: "The colony, complete", adding: "the reseed key and the HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("white: foraging  gold: carrying food", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new food layout", (0.02, 0.94), color=0xAAAAAA)`,
          does: "R deals a fresh food layout and resets the swarm, trails and all — a new foraging problem for the same three-rule brain to solve.",
          why: "Watch a few layouts end to end and notice what you never wrote: no shortest-path algorithm (yet nearer food gets brighter highways — shorter round trips mean more deposits per minute), no task allocation (yet traffic splits across sources), no cleanup logic (yet dead trails vanish). Three rules and a decaying field did all of it. That's the thesis of this whole arc.",
          see: "Every reroll: diffuse → discover → condense → harvest → adapt → exhaust. The colony's story, on demand.",
          checkpoint: "A complete adaptive colony. Final beat — project 21 complete.",
          recovery: ["Same reseed idiom as every project since 01 — apply_seed already clears trail, states, and positions together."] }
      ]
    }
  ]
};

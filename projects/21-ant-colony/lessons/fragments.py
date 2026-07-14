"""Code SOT for project 21 — ant colony.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 21-ant-colony`.

Evolutions: move_ants is the project's spine, growing through five versions —
pure random walk (ch1), +states with a NAIVE food pickup (ch2 s1), the atomic
claim fix (ch2 s2 — the naive version's negative-food race is the lesson),
+trail deposit (ch3), +the three sensors (ch4, the payoff). init_sim/apply_seed
gain the state and trail fields when their chapters need them; render gains the
trail underlay in ch3; step gains the evaporate/diffuse pass in ch3.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="21-ant-colony",
    default_file="ant_colony.py",
    reference={"ant_colony.py": PROJECT_DIR / "reference" / "ant_colony.py"},
    chapter_steps={1: 3, 2: 2, 3: 2, 4: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Ant Colony: three sensors, one pheromone, thirty thousand ants — highways emerge."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "GRID = 256"))
frag(((1, 2), "N_ANTS = 30000"))

frag(((1, 2), "NEST = (0.5, 0.5)"))
frag(((1, 2), "NEST_R = 0.03"))
frag(((1, 2), "FOOD_BLOBS = 5"))
frag(((1, 2), "FOOD_R = 10"))
frag(((1, 2), "FOOD_AMOUNT = 60.0"))

frag(((1, 2), "SPEED = 0.0022"))
frag(((1, 2), "WANDER = 0.35"))
frag(((4, 1), "SENSE_DIST = 8.0"))
frag(((4, 1), "SENSE_ANGLE = 0.5"))
frag(((2, 1), "TURN = 0.35"))
frag(((3, 2), "EVAP = 0.985"))
frag(((3, 2), "DIFFUSE = 0.12"))
frag(((3, 1), "DEPOSIT = 1.2"))
frag(((1, 2), "PI = 3.14159265"))

frag(((2, 1), "FORAGING, RETURNING = 0, 1"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "heading = None"))
frag(((2, 1), "state = None"))
frag(((3, 1), "trail = None"))
frag(((3, 1), "trail_next = None"))
frag(((1, 2), "food = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, food, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, state, food, pixels"),
    ((3, 1), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, state, trail, trail_next, food, pixels"),
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
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=N_ANTS)"))
frag(((1, 2), "    heading = ti.field(ti.f32, shape=N_ANTS)"))
frag(((2, 1), "    state = ti.field(ti.i32, shape=N_ANTS)"))
frag(((3, 1), "    trail = ti.field(ti.f32, shape=(GRID, GRID))"))
frag(((3, 1), "    trail_next = ti.field(ti.f32, shape=(GRID, GRID))"))
frag(((1, 2), "    food = ti.field(ti.f32, shape=(GRID, GRID))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- pure numpy generation ---------------------------------------------------------

SEED_FOOD = '''def seed_food(rng_seed=0):
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
    return f'''

frag(((1, 3), SEED_FOOD))

APPLY_V1 = """def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(np.tile(np.array(NEST, dtype=np.float32), (N_ANTS, 1)))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_ANTS).astype(np.float32))
    food.from_numpy(seed_food(rng_seed))"""

APPLY_V2 = """def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(np.tile(np.array(NEST, dtype=np.float32), (N_ANTS, 1)))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_ANTS).astype(np.float32))
    state.fill(FORAGING)
    food.from_numpy(seed_food(rng_seed))"""

APPLY_V3 = """def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pos.from_numpy(np.tile(np.array(NEST, dtype=np.float32), (N_ANTS, 1)))
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_ANTS).astype(np.float32))
    state.fill(FORAGING)
    trail.fill(0.0)
    food.from_numpy(seed_food(rng_seed))"""

frag(((1, 3), APPLY_V1), ((2, 1), APPLY_V2), ((3, 1), APPLY_V3))

# --- helpers ---------------------------------------------------------------------------

SAMPLE_TRAIL = """@ti.func
def sample_trail(x, y):
    gi = ti.min(ti.max(ti.cast(x * GRID, ti.i32), 0), GRID - 1)
    gj = ti.min(ti.max(ti.cast(y * GRID, ti.i32), 0), GRID - 1)
    return trail[gi, gj]"""

frag(((4, 1), SAMPLE_TRAIL))

WRAP_ANGLE = """@ti.func
def wrap_angle(dh):
    while dh > PI:
        dh -= 2 * PI
    while dh < -PI:
        dh += 2 * PI
    return dh"""

frag(((2, 1), WRAP_ANGLE))

# --- the ant brain: five versions --------------------------------------------------------

WALK_TAIL = """        newp = p + SPEED * ti.Vector([ti.cos(h), ti.sin(h)])
        for k in ti.static(range(2)):
            if newp[k] < 0.01:
                newp[k] = 0.01
                h = PI - h if k == 0 else -h
            if newp[k] > 0.99:
                newp[k] = 0.99
                h = PI - h if k == 0 else -h
        pos[a] = newp
        heading[a] = h"""

MOVE_V1 = (
    """@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
        h += (ti.random() - 0.5) * WANDER

"""
    + WALK_TAIL
)

RETURN_STEER = """        else:
            to_nest = ti.Vector([NEST[0], NEST[1]]) - p
            target = ti.atan2(to_nest[1], to_nest[0])
            h += ti.math.clamp(wrap_angle(target - h), -TURN, TURN)
            h += (ti.random() - 0.5) * 0.1"""

RETURN_STEER_DEPOSIT = RETURN_STEER + """
            gi = ti.min(ti.max(ti.cast(p[0] * GRID, ti.i32), 0), GRID - 1)
            gj = ti.min(ti.max(ti.cast(p[1] * GRID, ti.i32), 0), GRID - 1)
            trail[gi, gj] += DEPOSIT"""

PICKUP_NAIVE = """        gi = ti.min(ti.max(ti.cast(newp[0] * GRID, ti.i32), 0), GRID - 1)
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
                heading[a] = h + PI"""

PICKUP_ATOMIC = """        gi = ti.min(ti.max(ti.cast(newp[0] * GRID, ti.i32), 0), GRID - 1)
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
                heading[a] = h + PI"""

FORAGE_WANDER = """        if state[a] == FORAGING:
            h += (ti.random() - 0.5) * WANDER"""

FORAGE_SENSORS = """        if state[a] == FORAGING:
            sd = SENSE_DIST / GRID
            vl = sample_trail(p[0] + sd * ti.cos(h + SENSE_ANGLE), p[1] + sd * ti.sin(h + SENSE_ANGLE))
            vc = sample_trail(p[0] + sd * ti.cos(h), p[1] + sd * ti.sin(h))
            vr = sample_trail(p[0] + sd * ti.cos(h - SENSE_ANGLE), p[1] + sd * ti.sin(h - SENSE_ANGLE))
            if vl > vc and vl > vr:
                h += TURN
            elif vr > vc and vr > vl:
                h -= TURN
            h += (ti.random() - 0.5) * WANDER"""

MOVE_HEAD = """@ti.kernel
def move_ants():
    for a in pos:
        p = pos[a]
        h = heading[a]
"""

MOVE_V2 = MOVE_HEAD + FORAGE_WANDER + "\n" + RETURN_STEER + "\n\n" + WALK_TAIL + "\n\n" + PICKUP_NAIVE
MOVE_V3 = MOVE_HEAD + FORAGE_WANDER + "\n" + RETURN_STEER + "\n\n" + WALK_TAIL + "\n\n" + PICKUP_ATOMIC
MOVE_V4 = MOVE_HEAD + FORAGE_WANDER + "\n" + RETURN_STEER_DEPOSIT + "\n\n" + WALK_TAIL + "\n\n" + PICKUP_ATOMIC
MOVE_V5 = MOVE_HEAD + FORAGE_SENSORS + "\n" + RETURN_STEER_DEPOSIT + "\n\n" + WALK_TAIL + "\n\n" + PICKUP_ATOMIC

frag(((1, 3), MOVE_V1), ((2, 1), MOVE_V2), ((2, 2), MOVE_V3), ((3, 1), MOVE_V4), ((4, 1), MOVE_V5))

# --- trail dynamics -------------------------------------------------------------------

EVOLVE = """@ti.kernel
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
        trail_next[i, j] = (trail[i, j] * (1 - DIFFUSE) + avg * DIFFUSE) * EVAP"""

frag(((3, 2), EVOLVE))

frag(((3, 2), "@ti.kernel\ndef copy_trail():\n    for i, j in trail:\n        trail[i, j] = trail_next[i, j]"))

# --- render: two versions -------------------------------------------------------------

RENDER_V1 = """@ti.kernel
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
            pixels[xi, yi] = ti.Vector([0.9, 0.9, 0.85])"""

RENDER_V2 = """@ti.kernel
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
            pixels[xi, yi] = col"""

frag(((1, 3), RENDER_V1), ((3, 1), RENDER_V2))

frag(
    ((1, 3), "def step():\n    move_ants()"),
    ((3, 2), "def step():\n    move_ants()\n    evolve_trail()\n    copy_trail()"),
)

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Ant Colony — taichi-academy", res=RES, background_color=0x000000)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((4, 2), EVENTS_V2))

frag(((1, 3), "        step()"))
frag(((1, 3), "        render()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((4, 2), '        gui.text("white: foraging  gold: carrying food", (0.02, 0.98), color=0xFFFFFF)'))
frag(((4, 2), '        gui.text("[r] new food layout", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

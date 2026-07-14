"""Code SOT for project 23 — evolution.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 23-evolution`.

Evolutions: the survival loop is built in stages. sense_think_move gains the
eat/energy/death block in chapter 2 (chapter 1 is pure brain-driven wandering).
reproduce arrives in chapter 3 as CLONING (no mutation — population stabilizes
but is genetically frozen), then gains its mutation loop in chapter 4 (the
payoff: brains actually improve). step() accretes the free-list + reproduce +
regrow calls across chapters 2-3. init_sim/apply_seed grow the food_cap and
free-list fields as their chapters need them.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="23-evolution",
    default_file="evolution.py",
    reference={"evolution.py": PROJECT_DIR / "reference" / "evolution.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Evolution: nobody designs the brain. Sensors, a tiny neural net, mutation — foraging appears."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "N_MAX = 3000"))
frag(((1, 2), "START_POP = 400"))
frag(((1, 2), "FOOD_GRID = 128"))
frag(((1, 2), "FOOD_PATCHES = 8"))

frag(((1, 2), "N_SENSORS = 5   # food-left, food-center, food-right, own-energy, bias"))
frag(((1, 2), "N_HIDDEN = 6"))
frag(((1, 2), "N_OUT = 2       # turn, thrust"))
frag(((1, 2), "N_W = N_SENSORS * N_HIDDEN + N_HIDDEN * N_OUT"))

frag(((1, 2), "SENSE_DIST = 0.06"))
frag(((1, 2), "SENSE_ANGLE = 0.6"))
frag(((1, 2), "MAX_TURN = 0.4"))
frag(((1, 2), "MAX_THRUST = 0.006"))
frag(((2, 1), "MOVE_COST = 0.4"))
frag(((2, 1), "LIVE_COST = 0.15"))
frag(((2, 1), "EAT_BITE = 0.5"))
frag(((2, 1), "EAT_GAIN = 9.0"))
frag(((2, 1), "REPRO_ENERGY = 100.0"))
frag(((2, 1), "START_ENERGY = 45.0"))
frag(((4, 1), "MUT_RATE = 0.10"))
frag(((4, 1), "MUT_SCALE = 0.25"))
frag(((2, 1), "FOOD_REGROW = 0.010"))
frag(((1, 2), "PI = 3.14159265"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "heading = None"))
frag(((2, 1), "energy = None"))
frag(((2, 1), "alive = None"))
frag(((1, 2), "weights = None"))
frag(((1, 2), "food = None"))
frag(((2, 1), "food_cap = None"))
frag(((3, 1), "free_slots = None"))
frag(((3, 1), "n_free = None"))
frag(((2, 2), "n_alive = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, weights, food, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, energy, alive, weights, food, food_cap, pixels"),
    ((2, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, heading, energy, alive, weights, food, food_cap, n_alive, pixels"),
    (
        (3, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, heading, energy, alive, weights, food, food_cap, free_slots, n_free, n_alive, pixels",
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
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=N_MAX)"))
frag(((1, 2), "    heading = ti.field(ti.f32, shape=N_MAX)"))
frag(((2, 1), "    energy = ti.field(ti.f32, shape=N_MAX)"))
frag(((2, 1), "    alive = ti.field(ti.i32, shape=N_MAX)"))
frag(((1, 2), "    weights = ti.field(ti.f32, shape=(N_MAX, N_W))"))
frag(((1, 2), "    food = ti.field(ti.f32, shape=(FOOD_GRID, FOOD_GRID))"))
frag(((2, 1), "    food_cap = ti.field(ti.f32, shape=(FOOD_GRID, FOOD_GRID))"))
frag(((3, 1), "    free_slots = ti.field(ti.i32, shape=N_MAX)"))
frag(((3, 1), "    n_free = ti.field(ti.i32, shape=())"))
frag(((2, 2), "    n_alive = ti.field(ti.i32, shape=())"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- pure numpy generation ---------------------------------------------------------

FOOD_FIELD = '''def food_field(rng_seed=0):
    """Pure numpy: a few gaussian food patches, the world's carrying capacity map."""
    rng = np.random.default_rng(rng_seed)
    ii, jj = np.meshgrid(np.arange(FOOD_GRID), np.arange(FOOD_GRID), indexing="ij")
    cap = np.zeros((FOOD_GRID, FOOD_GRID), dtype=np.float32)
    for _ in range(FOOD_PATCHES):
        cx, cy = rng.uniform(0.15, 0.85, 2) * FOOD_GRID
        r = rng.uniform(8, 16)
        cap += np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (2 * r * r))
    return np.clip(cap, 0, 1).astype(np.float32)'''

frag(((1, 3), FOOD_FIELD))

APPLY_V1 = """def apply_seed(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    p = np.zeros((N_MAX, 2), dtype=np.float32)
    p[:START_POP] = rng.uniform(0.1, 0.9, (START_POP, 2))
    pos.from_numpy(p)
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N_MAX).astype(np.float32))
    weights.from_numpy(rng.normal(0, 1.0, (N_MAX, N_W)).astype(np.float32))
    food.from_numpy(food_field(rng_seed))"""

APPLY_V2 = """def apply_seed(rng_seed=0):
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
    food.from_numpy(cap.copy())"""

frag(((1, 3), APPLY_V1), ((2, 1), APPLY_V2))

# --- the brain -------------------------------------------------------------------------

FOOD_AT = """@ti.func
def food_at(x, y):
    gi = ti.min(ti.max(ti.cast(x * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
    gj = ti.min(ti.max(ti.cast(y * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
    return food[gi, gj]"""

frag(((1, 3), FOOD_AT))

BRAIN = """@ti.func
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
    return ti.tanh(turn), ti.tanh(thrust)"""

frag(((1, 3), BRAIN))

# --- sense_think_move: two versions ----------------------------------------------------

STM_HEAD = """@ti.kernel
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
            heading[c] = hd"""

# chapter 1: everyone is "alive" (a plain 1 field standing in) and there is no energy yet.
# To keep chapter 1 runnable without the energy/alive machinery, chapter 1 uses a
# simplified head that omits the `en` sensor's energy read and the alive gate.
STM_V1 = """@ti.kernel
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
        heading[c] = hd"""

EAT_BLOCK = """
            energy[c] -= LIVE_COST + MOVE_COST * sp / MAX_THRUST
            gi = ti.min(ti.max(ti.cast(newp[0] * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
            gj = ti.min(ti.max(ti.cast(newp[1] * FOOD_GRID, ti.i32), 0), FOOD_GRID - 1)
            got = ti.atomic_sub(food[gi, gj], EAT_BITE)
            if got > EAT_BITE:
                energy[c] += EAT_GAIN
            else:
                food[gi, gj] += EAT_BITE
            if energy[c] <= 0.0:
                alive[c] = 0"""

STM_V2 = STM_HEAD + EAT_BLOCK

frag(((1, 3), STM_V1), ((2, 1), STM_V2))

# --- reproduction ----------------------------------------------------------------------

FREE_LIST = """@ti.kernel
def build_free_list():
    n_free[None] = 0
    for c in alive:
        if alive[c] == 0:
            idx = ti.atomic_add(n_free[None], 1)
            free_slots[idx] = c"""

frag(((3, 1), FREE_LIST))

REPRO_CLONE = """@ti.kernel
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
                    weights[slot, w] = weights[c, w]"""

REPRO_MUTATE = """@ti.kernel
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
                    weights[slot, w] = weights[c, w] + m"""

frag(((3, 1), REPRO_CLONE), ((4, 1), REPRO_MUTATE))

REGROW = """@ti.kernel
def regrow():
    for i, j in food:
        food[i, j] = ti.min(food[i, j] + FOOD_REGROW, food_cap[i, j])"""

frag(((2, 1), REGROW))

COUNT_ALIVE = """@ti.kernel
def count_alive():
    n_alive[None] = 0
    for c in alive:
        if alive[c] == 1:
            ti.atomic_add(n_alive[None], 1)"""

frag(((2, 2), COUNT_ALIVE))

# --- render: two versions -------------------------------------------------------------

RENDER_V1 = """@ti.kernel
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
                pixels[x, y] = ti.Vector([0.9, 0.9, 0.8])"""

RENDER_V2 = """@ti.kernel
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
                    pixels[x, y] = col"""

frag(((1, 3), RENDER_V1), ((2, 1), RENDER_V2))

# --- the tick ----------------------------------------------------------------------

STEP_V1 = "def step():\n    sense_think_move()"
STEP_V2 = "def step():\n    sense_think_move()\n    regrow()"
STEP_V3 = "def step():\n    sense_think_move()\n    build_free_list()\n    reproduce()\n    regrow()"

frag(((1, 3), STEP_V1), ((2, 1), STEP_V2), ((3, 2), STEP_V3))

POPULATION = """def population():
    count_alive()
    return int(n_alive[None])"""

frag(((2, 2), POPULATION))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Evolution — taichi-academy", res=RES, background_color=0x000000)'))
frag(((2, 3), "    gen = 0"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
                gen = 0'''

frag(((1, 3), EVENTS_V1), ((4, 2), EVENTS_V2))

frag(((1, 3), "        step()"))
frag(((2, 3), "        gen += 1"))
frag(((1, 3), "        render()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((2, 3), '        gui.text(f"generation {gen}  population {population()}", (0.02, 0.98), color=0xFFFFFF)'))
frag(((4, 2), '        gui.text("gold: fed  purple: starving   [r] new world", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

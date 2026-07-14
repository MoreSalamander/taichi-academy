"""Evolution: nobody designs the brain. Sensors, a tiny neural net, mutation — foraging appears."""

import numpy as np
import taichi as ti

RES = 512
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
MOVE_COST = 0.4
LIVE_COST = 0.15
EAT_BITE = 0.5
EAT_GAIN = 9.0
REPRO_ENERGY = 100.0
START_ENERGY = 45.0
MUT_RATE = 0.10
MUT_SCALE = 0.25
FOOD_REGROW = 0.010
PI = 3.14159265

pos = None
heading = None
energy = None
alive = None
weights = None
food = None
food_cap = None
free_slots = None
n_free = None
n_alive = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, energy, alive, weights, food, food_cap, free_slots, n_free, n_alive, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N_MAX)
    heading = ti.field(ti.f32, shape=N_MAX)
    energy = ti.field(ti.f32, shape=N_MAX)
    alive = ti.field(ti.i32, shape=N_MAX)
    weights = ti.field(ti.f32, shape=(N_MAX, N_W))
    food = ti.field(ti.f32, shape=(FOOD_GRID, FOOD_GRID))
    food_cap = ti.field(ti.f32, shape=(FOOD_GRID, FOOD_GRID))
    free_slots = ti.field(ti.i32, shape=N_MAX)
    n_free = ti.field(ti.i32, shape=())
    n_alive = ti.field(ti.i32, shape=())
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def food_field(rng_seed=0):
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
                    m = 0.0
                    if ti.random() < MUT_RATE:
                        m = (ti.random() - 0.5) * 2.0 * MUT_SCALE
                    weights[slot, w] = weights[c, w] + m


@ti.kernel
def regrow():
    for i, j in food:
        food[i, j] = ti.min(food[i, j] + FOOD_REGROW, food_cap[i, j])


@ti.kernel
def count_alive():
    n_alive[None] = 0
    for c in alive:
        if alive[c] == 1:
            ti.atomic_add(n_alive[None], 1)


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
    build_free_list()
    reproduce()
    regrow()


def population():
    count_alive()
    return int(n_alive[None])


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Evolution — taichi-academy", res=RES, background_color=0x000000)
    gen = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
                gen = 0
        step()
        gen += 1
        render()
        gui.set_image(pixels)
        gui.text(f"generation {gen}  population {population()}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("gold: fed  purple: starving   [r] new world", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

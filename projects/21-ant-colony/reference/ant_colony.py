"""Ant Colony: three sensors, one pheromone, thirty thousand ants — highways emerge."""

import numpy as np
import taichi as ti

RES = 512
GRID = 256
N_ANTS = 30000

NEST = (0.5, 0.5)
NEST_R = 0.03
FOOD_BLOBS = 5
FOOD_R = 10
FOOD_AMOUNT = 60.0

SPEED = 0.0022
WANDER = 0.35
SENSE_DIST = 8.0
SENSE_ANGLE = 0.5
TURN = 0.35
EVAP = 0.985
DIFFUSE = 0.12
DEPOSIT = 1.2
PI = 3.14159265

FORAGING, RETURNING = 0, 1

pos = None
heading = None
state = None
trail = None
trail_next = None
food = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, heading, state, trail, trail_next, food, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N_ANTS)
    heading = ti.field(ti.f32, shape=N_ANTS)
    state = ti.field(ti.i32, shape=N_ANTS)
    trail = ti.field(ti.f32, shape=(GRID, GRID))
    trail_next = ti.field(ti.f32, shape=(GRID, GRID))
    food = ti.field(ti.f32, shape=(GRID, GRID))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def seed_food(rng_seed=0):
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
    state.fill(FORAGING)
    trail.fill(0.0)
    food.from_numpy(seed_food(rng_seed))


@ti.func
def sample_trail(x, y):
    gi = ti.min(ti.max(ti.cast(x * GRID, ti.i32), 0), GRID - 1)
    gj = ti.min(ti.max(ti.cast(y * GRID, ti.i32), 0), GRID - 1)
    return trail[gi, gj]


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
                heading[a] = h + PI


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
            pixels[xi, yi] = col


def step():
    move_ants()
    evolve_trail()
    copy_trail()


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Ant Colony — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        step()
        render()
        gui.set_image(pixels)
        gui.text("white: foraging  gold: carrying food", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new food layout", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

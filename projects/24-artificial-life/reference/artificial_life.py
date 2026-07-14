"""Artificial Life: one turn rule on 40,000 particles — and cells self-assemble from soup."""

import math

import numpy as np
import taichi as ti

RES = 640
WORLD = 1.0
N = 40000

R = 0.011
ALPHA = math.radians(180.0)
BETA = math.radians(17.0)
V = 0.0015

GRID = 88
CELL = WORLD / GRID
NCELLS = GRID * GRID

STIR_RADIUS = 0.05

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
    heading.from_numpy(rng.uniform(0, 2 * np.pi, N).astype(np.float32))


@ti.func
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
        pos[p] = ti.Vector([newp[0] % WORLD, newp[1] % WORLD])


@ti.kernel
def stir(mx: ti.f32, my: ti.f32):
    for p in pos:
        dx = wrapd(pos[p][0], mx)
        dy = wrapd(pos[p][1], my)
        if dx * dx + dy * dy < STIR_RADIUS * STIR_RADIUS:
            heading[p] = ti.atan2(dy, dx)


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
    turn_and_move()


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Artificial Life — taichi-academy", res=RES, background_color=0x050508)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            stir(mx, my)
        step()
        render()
        gui.set_image(pixels)
        gui.text("green: free  yellow: membrane  magenta: nucleus", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to disturb  [r] new soup", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

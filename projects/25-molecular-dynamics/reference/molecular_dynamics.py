"""Molecular Dynamics: one pair force + Verlet, and matter melts, freezes, and crystallizes."""

import math

import numpy as np
import taichi as ti

RES = 600
L = 40.0
N = 1400
SIGMA = 1.0
EPS = 1.0
RCUT = 2.5
COORD_SHELL = 1.3
DT = 0.004
GRID = 16
CELL = L / GRID
NCELLS = GRID * GRID

THERMO_RATE = 0.05
TEMP_STEP = 0.1
TEMP_MIN = 0.02
TEMP_MAX = 4.0
HEAT_RADIUS = 4.0
HEAT_BOOST = 1.4

pos = None
vel = None
acc = None
coord = None
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None
temp_target = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, acc, coord, cell_count, cell_start, cell_cursor, sorted_idx, temp_target, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    acc = ti.Vector.field(2, ti.f32, shape=N)
    coord = ti.field(ti.i32, shape=N)
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=N)
    temp_target = ti.field(ti.f32, shape=())
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def lattice_positions(n, box):
    """Pure numpy: n atoms on a square grid filling the box, jittered off perfection."""
    side = int(math.ceil(math.sqrt(n)))
    spacing = box / side
    xy = np.array([[(i + 0.5) * spacing, (j + 0.5) * spacing] for i in range(side) for j in range(side)])
    return xy[:n].astype(np.float32)


def maxwell_velocities(n, temperature, rng):
    """Pure numpy: gaussian velocities at a temperature, with net momentum removed."""
    v = rng.normal(0.0, math.sqrt(temperature), (n, 2)).astype(np.float32)
    v -= v.mean(axis=0)
    return v


def apply_seed(rng_seed=0, temperature=1.0):
    rng = np.random.default_rng(rng_seed)
    xy = lattice_positions(N, L) + rng.normal(0, 0.05, (N, 2)).astype(np.float32)
    pos.from_numpy(xy % L)
    vel.from_numpy(maxwell_velocities(N, temperature, rng))
    acc.fill(0.0)
    temp_target[None] = temperature
    build_grid()
    compute_forces()


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
        a = 0
        for c in range(NCELLS):
            cell_start[c] = a
            a += cell_count[c]


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
    if d > 0.5 * L:
        d -= L
    if d < -0.5 * L:
        d += L
    return d


@ti.kernel
def compute_forces():
    for p in pos:
        f = ti.Vector([0.0, 0.0])
        nc = 0
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni = (ci + di) % GRID
            nj = (cj + dj) % GRID
            nidx = ni * GRID + nj
            for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                q = sorted_idx[s]
                if q != p:
                    dx = wrapd(pos[p][0], pos[q][0])
                    dy = wrapd(pos[p][1], pos[q][1])
                    r2 = dx * dx + dy * dy
                    if 1e-4 < r2 < RCUT * RCUT:
                        inv2 = SIGMA * SIGMA / r2
                        inv6 = inv2 * inv2 * inv2
                        fmag = 24.0 * EPS * (2.0 * inv6 * inv6 - inv6) / r2
                        f += fmag * ti.Vector([dx, dy])
                        if r2 < COORD_SHELL * COORD_SHELL:
                            nc += 1
        acc[p] = f
        coord[p] = nc


@ti.kernel
def half_kick():
    for p in vel:
        vel[p] += 0.5 * DT * acc[p]


@ti.kernel
def drift():
    for p in pos:
        newp = pos[p] + DT * vel[p]
        pos[p] = ti.Vector([newp[0] % L, newp[1] % L])


@ti.kernel
def measure_temp() -> ti.f32:
    ke = 0.0
    for p in vel:
        ke += 0.5 * vel[p].dot(vel[p])
    return ke / N


@ti.kernel
def thermostat(cur_temp: ti.f32):
    scale = 1.0
    if cur_temp > 1e-6:
        scale = ti.sqrt(temp_target[None] / cur_temp)
    s = 1.0 + THERMO_RATE * (scale - 1.0)
    for p in vel:
        vel[p] *= s


@ti.kernel
def heat(mx: ti.f32, my: ti.f32):
    for p in pos:
        dx = wrapd(pos[p][0], mx * L)
        dy = wrapd(pos[p][1], my * L)
        if dx * dx + dy * dy < HEAT_RADIUS * HEAT_RADIUS:
            vel[p] *= HEAT_BOOST


def step(thermo=True):
    half_kick()
    drift()
    build_grid()
    compute_forces()
    half_kick()
    if thermo:
        thermostat(measure_temp())


def crystalline_fraction():
    """Pure numpy: fraction of atoms with a full hexagonal shell of 6 close neighbors."""
    return float((coord.to_numpy() == 6).mean())


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.03, 0.03, 0.05])
    for p in pos:
        xi = ti.cast(pos[p][0] / L * RES, ti.i32)
        yi = ti.cast(pos[p][1] / L * RES, ti.i32)
        c = coord[p]
        col = ti.Vector([0.3, 0.5, 1.0])
        if c >= 6:
            col = ti.Vector([1.0, 0.85, 0.3])
        elif c >= 4:
            col = ti.Vector([0.5, 0.9, 0.5])
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = col


def main():
    init_sim()
    apply_seed(temperature=1.0)
    gui = ti.GUI("Molecular Dynamics — taichi-academy", res=RES, background_color=0x08080F)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                temp_target[None] = min(temp_target[None] + TEMP_STEP, TEMP_MAX)
            elif e.key == ti.GUI.DOWN:
                temp_target[None] = max(temp_target[None] - TEMP_STEP, TEMP_MIN)
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000), temperature=1.0)
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            heat(mx, my)
        step()
        render()
        gui.set_image(pixels)
        gui.text(
            f"target T {temp_target[None]:.2f}   crystalline {crystalline_fraction() * 100:.0f}%",
            (0.02, 0.98), color=0xFFFFFF,
        )
        gui.text("[up/down] heat/cool  drag: heat gun  [r] remelt   gold: crystal, blue: fluid", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

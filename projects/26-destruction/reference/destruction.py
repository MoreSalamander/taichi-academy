"""Destruction Engine: buildings are lattices of breakable bonds — explosions and quakes fracture them."""

import numpy as np
import taichi as ti

RES = 640
GRAVITY = 0.5
DT = 1.0 / 60
ITERS = 24
OMEGA = 1.7             # over-relaxation: pushes the Jacobi solver toward rigid faster
FLOOR = 0.04
RADIUS = 0.006          # particle radius, for rubble self-collision
SPACING = 0.014         # rest gap between lattice neighbours
BREAK_STRAIN = 1.4      # a bond snaps once stretched past 1.4x its rest length
DAMP = 0.99

GRID = 64
CELL = 1.0 / GRID
NCELLS = GRID * GRID

MAX_P = 2000
MAX_B = 8000

# buildings: (base_x, width_cols, height_rows) — blocky walls stand; thin spires pancake
BUILDINGS = [(0.08, 22, 18), (0.45, 16, 24), (0.72, 20, 14)]

EXPLODE_POWER = 4.5
EXPLODE_RADIUS = 0.16
QUAKE_AMP = 0.012
QUAKE_FREQ = 38.0

pos = None
prev = None
delta = None
dn = None
stress = None
b_a = None
b_b = None
b_rest = None
b_broken = None
n_p = None
n_b = None
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, prev, delta, dn, stress, b_a, b_b, b_rest, b_broken, n_p, n_b
    global cell_count, cell_start, cell_cursor, sorted_idx, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=MAX_P)
    prev = ti.Vector.field(2, ti.f32, shape=MAX_P)
    delta = ti.Vector.field(2, ti.f32, shape=MAX_P)
    dn = ti.field(ti.f32, shape=MAX_P)
    stress = ti.field(ti.f32, shape=MAX_P)
    b_a = ti.field(ti.i32, shape=MAX_B)
    b_b = ti.field(ti.i32, shape=MAX_B)
    b_rest = ti.field(ti.f32, shape=MAX_B)
    b_broken = ti.field(ti.i32, shape=MAX_B)
    n_p = ti.field(ti.i32, shape=())
    n_b = ti.field(ti.i32, shape=())
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=MAX_P)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def build_structures():
    """Pure numpy: lay out the buildings as a lattice. Returns (positions, edges).

    Each cell links to its right, up, and both diagonal neighbours — the diagonals are
    the braces that keep a wall from folding flat (a square of four distance bonds is a
    hinge; add a diagonal and it is rigid)."""
    positions = []
    edges = []
    for (bx, w, h) in BUILDINGS:
        local = {}
        for gy in range(h):
            for gx in range(w):
                local[(gx, gy)] = len(positions)
                positions.append([bx + gx * SPACING, FLOOR + RADIUS + gy * SPACING])
        for gy in range(h):
            for gx in range(w):
                a = local[(gx, gy)]
                for (nx, ny) in ((gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1), (gx + 1, gy - 1)):
                    if (nx, ny) in local:
                        edges.append((a, local[(nx, ny)]))
    return np.array(positions, dtype=np.float32), np.array(edges, dtype=np.int32)


def bond_rest_lengths(positions, edges):
    """Pure numpy: the rest length of each bond is the initial neighbour distance."""
    return np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).astype(np.float32)


def apply_seed():
    """Build the city and upload it: every bond intact, everything at rest."""
    positions, edges = build_structures()
    rests = bond_rest_lengths(positions, edges)
    npart, nbond = len(positions), len(edges)
    pos.from_numpy(np.pad(positions, ((0, MAX_P - npart), (0, 0))))
    prev.from_numpy(np.pad(positions, ((0, MAX_P - npart), (0, 0))))
    b_a.from_numpy(np.pad(edges[:, 0], (0, MAX_B - nbond)))
    b_b.from_numpy(np.pad(edges[:, 1], (0, MAX_B - nbond)))
    b_rest.from_numpy(np.pad(rests, (0, MAX_B - nbond)))
    b_broken.fill(0)
    n_p[None] = npart
    n_b[None] = nbond


@ti.kernel
def predict():
    """Verlet: velocity is where you were minus where you are, damped; then fall."""
    for i in range(n_p[None]):
        v = (pos[i] - prev[i]) * DAMP
        prev[i] = pos[i]
        pos[i] = pos[i] + v + ti.Vector([0.0, -GRAVITY]) * DT * DT


@ti.func
def flat_cell(i):
    ci = ti.min(ti.max(ti.cast(pos[i][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj


@ti.kernel
def count_cells():
    for c in cell_count:
        cell_count[c] = 0
    for i in range(n_p[None]):
        cell_count[flat_cell(i)] += 1


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
    for i in range(n_p[None]):
        slot = ti.atomic_add(cell_cursor[flat_cell(i)], 1)
        sorted_idx[slot] = i


def build_grid():
    count_cells()
    prefix_sum()
    scatter()


@ti.kernel
def break_bonds():
    """A bond that has been stretched past its breaking strain is gone for good.
    Judged here, on the freshly predicted positions, before the solver heals the stretch."""
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            if (pos[b_b[k]] - pos[b_a[k]]).norm() > b_rest[k] * BREAK_STRAIN:
                b_broken[k] = 1


@ti.kernel
def clear_delta():
    for i in range(n_p[None]):
        delta[i] = ti.Vector([0.0, 0.0])
        dn[i] = 0.0


@ti.kernel
def solve_bonds():
    """Each surviving bond nudges its two ends back toward rest length (half the error each)."""
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            a, b = b_a[k], b_b[k]
            d = pos[b] - pos[a]
            dist = d.norm() + 1e-9
            corr = 0.5 * (dist - b_rest[k]) / dist * d
            delta[a] += corr
            dn[a] += 1.0
            delta[b] -= corr
            dn[b] += 1.0


@ti.kernel
def solve_collisions():
    """Rubble does not interpenetrate: any two particles closer than 2R push apart.
    Neighbours are found through the spatial-hash grid, so this stays O(N)."""
    for i in range(n_p[None]):
        ci = ti.min(ti.max(ti.cast(pos[i][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[i][1] / CELL, ti.i32), 0), GRID - 1)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                nidx = ni * GRID + nj
                for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                    j = sorted_idx[s]
                    if j > i:
                        d = pos[j] - pos[i]
                        dist = d.norm() + 1e-9
                        mind = 2.0 * RADIUS
                        if dist < mind:
                            corr = 0.5 * (dist - mind) / dist * d
                            delta[i] += corr
                            dn[i] += 1.0
                            delta[j] -= corr
                            dn[j] += 1.0


@ti.kernel
def apply_delta():
    """Average every correction landing on a particle, over-relax, and move it once."""
    for i in range(n_p[None]):
        if dn[i] > 0:
            pos[i] += OMEGA * delta[i] / dn[i]


@ti.kernel
def floor_constraint():
    for i in range(n_p[None]):
        if pos[i][1] < FLOOR + RADIUS:
            pos[i][1] = FLOOR + RADIUS
        pos[i][0] = ti.min(ti.max(pos[i][0], RADIUS), 1.0 - RADIUS)


@ti.kernel
def compute_stress():
    """Per-particle load: sum of the strain on its surviving bonds (drives the colour)."""
    for i in range(n_p[None]):
        stress[i] = 0.0
    for k in range(n_b[None]):
        if b_broken[k] == 0:
            strain = ti.abs((pos[b_b[k]] - pos[b_a[k]]).norm() / b_rest[k] - 1.0)
            stress[b_a[k]] += strain
            stress[b_b[k]] += strain


@ti.kernel
def explode(mx: ti.f32, my: ti.f32, power: ti.f32, radius: ti.f32):
    """A radial shove, strongest at the blast centre, fading to nothing at its edge."""
    c = ti.Vector([mx, my])
    for i in range(n_p[None]):
        d = pos[i] - c
        r = d.norm() + 1e-6
        if r < radius:
            pos[i] += d / r * power * (1.0 - r / radius) * DT


@ti.kernel
def quake(t: ti.f32, amp: ti.f32):
    """Shear the ground: particles near the floor are dragged sideways in an oscillation,
    the drag fading with height, so the base whips out from under the mass above it."""
    shift = amp * ti.sin(t * QUAKE_FREQ)
    for i in range(n_p[None]):
        zone = (pos[i][1] - FLOOR) / 0.08
        if zone < 1.0:
            pos[i][0] += shift * (1.0 - zone)


def step():
    predict()
    break_bonds()
    build_grid()
    for _ in range(ITERS):
        clear_delta()
        solve_bonds()
        solve_collisions()
        apply_delta()
        floor_constraint()
    compute_stress()


def broken_fraction():
    """Pure numpy: share of bonds that have snapped — how ruined the city is."""
    nb = n_b[None]
    return float(b_broken.to_numpy()[:nb].sum()) / max(nb, 1)


@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.06, 0.07, 0.10])
        if j < FLOOR * RES:
            pixels[i, j] = ti.Vector([0.15, 0.13, 0.10])
    for i in range(n_p[None]):
        xi = ti.cast(pos[i][0] * RES, ti.i32)
        yi = ti.cast(pos[i][1] * RES, ti.i32)
        s = ti.min(stress[i] * 3.0, 1.0)
        col = ti.Vector([0.6, 0.65, 0.7]) * (1.0 - s) + ti.Vector([1.0, 0.3, 0.15]) * s
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = col


def main():
    init_sim()
    apply_seed()
    for _ in range(60):  # let the city settle onto the ground before the player can touch it
        step()
    gui = ti.GUI("Destruction Engine — taichi-academy", res=RES, background_color=0x10121A)
    frame = 0
    while gui.running:
        frame += 1
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = gui.get_cursor_pos()
                explode(mx, my, EXPLODE_POWER, EXPLODE_RADIUS)
            elif e.key == "r":
                apply_seed()
                for _ in range(60):
                    step()
        if gui.is_pressed("q", ti.GUI.SPACE):
            quake(frame * DT, QUAKE_AMP)
        step()
        render()
        gui.set_image(pixels)
        gui.text(f"ruined {broken_fraction() * 100:.0f}%", (0.02, 0.98), color=0xFFFFFF)
        gui.text("click: explosion   hold [q]/space: earthquake   [r] rebuild", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

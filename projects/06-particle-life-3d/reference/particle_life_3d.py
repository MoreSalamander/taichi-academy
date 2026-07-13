"""Particle Life 3D: species rules + a spatial hash turn simple math into ecology, at scale."""

import numpy as np
import taichi as ti

NUM = 30000
NSPEC = 6
WORLD = 1.0
R_MAX = 0.08
BETA = 0.3
GRID = 12
CELL = WORLD / GRID
NCELLS = GRID * GRID * GRID
DT = 0.01
SUBSTEPS = 4
FRICTION = 0.82
FORCE_SCALE = 2.0
PARTICLE_RADIUS = 0.0035

pos = None
vel = None
species = None
colors = None
base_colors = None
rules = None
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, species, colors, base_colors, rules
    global cell_count, cell_start, cell_cursor, sorted_idx
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(3, ti.f32, shape=NUM)
    vel = ti.Vector.field(3, ti.f32, shape=NUM)
    species = ti.field(ti.i32, shape=NUM)
    colors = ti.Vector.field(3, ti.f32, shape=NUM)
    base_colors = ti.Vector.field(3, ti.f32, shape=NUM)
    rules = ti.field(ti.f32, shape=(NSPEC, NSPEC))
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=NUM)


def species_palette(n):
    """Pure numpy: n distinct hues spaced around the color wheel, as RGB."""
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    h6 = hues * 6.0
    k = h6.astype(np.int32) % 6
    f = h6 - np.floor(h6)
    v, p, q, t = 1.0, 0.15, 1.0 - f, 0.15 + f * 0.85
    table = np.stack(
        [
            np.stack([v + 0 * f, t, p * np.ones_like(f)], axis=1),
            np.stack([q, v + 0 * f, p * np.ones_like(f)], axis=1),
            np.stack([p * np.ones_like(f), v + 0 * f, t], axis=1),
            np.stack([p * np.ones_like(f), q, v + 0 * f], axis=1),
            np.stack([t, p * np.ones_like(f), v + 0 * f], axis=1),
            np.stack([v + 0 * f, p * np.ones_like(f), q], axis=1),
        ],
        axis=0,
    )
    return table[k, np.arange(n)].astype(np.float32)


def seed_particles(n, nspec, rng_seed=0):
    """Pure numpy: random positions/velocities/species inside the cube, plus their colors."""
    rng = np.random.default_rng(rng_seed)
    pos0 = rng.uniform(0.0, WORLD, size=(n, 3)).astype(np.float32)
    vel0 = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    spec0 = rng.integers(0, nspec, size=n).astype(np.int32)
    palette = species_palette(nspec)
    col0 = palette[spec0]
    return pos0, vel0, spec0, col0


def rule_matrix(nspec, rng_seed=0):
    """Pure numpy: random attraction(+)/repulsion(-) coefficient per species pair, in [-1, 1]."""
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(-1.0, 1.0, size=(nspec, nspec)).astype(np.float32)


def apply_seed(seed, rules_np):
    pos0, vel0, spec0, col0 = seed
    pos.from_numpy(pos0)
    vel.from_numpy(vel0)
    species.from_numpy(spec0)
    base_colors.from_numpy(col0)
    colors.from_numpy(col0)
    rules.from_numpy(rules_np)


@ti.func
def flat_cell(p) -> ti.i32:
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    ck = ti.min(ti.max(ti.cast(pos[p][2] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID * GRID + cj * GRID + ck


@ti.kernel
def count_cells():
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
    for p in pos:
        idx = flat_cell(p)
        slot = ti.atomic_add(cell_cursor[idx], 1)
        sorted_idx[slot] = p


def build_grid():
    cell_count.fill(0)
    count_cells()
    prefix_sum()
    cell_cursor.copy_from(cell_start)
    scatter()


@ti.func
def force_law(r_norm, a) -> ti.f32:
    f = 0.0
    if r_norm < BETA:
        f = r_norm / BETA - 1.0
    elif r_norm < 1.0:
        f = a * (1.0 - ti.abs(2.0 * r_norm - 1.0 - BETA) / (1.0 - BETA))
    return f


@ti.kernel
def compute_forces():
    for p in pos:
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        ck = ti.min(ti.max(ti.cast(pos[p][2] / CELL, ti.i32), 0), GRID - 1)
        acc = ti.Vector([0.0, 0.0, 0.0])
        for di, dj, dk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if 0 <= ni < GRID and 0 <= nj < GRID and 0 <= nk < GRID:
                nidx = ni * GRID * GRID + nj * GRID + nk
                for slot in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                    q = sorted_idx[slot]
                    if q != p:
                        d = pos[q] - pos[p]
                        dist = d.norm()
                        if 1e-6 < dist < R_MAX:
                            a = rules[species[p], species[q]]
                            f = force_law(dist / R_MAX, a)
                            acc += (d / dist) * f
        vel[p] += acc * FORCE_SCALE * DT


@ti.kernel
def integrate():
    for p in pos:
        vel[p] *= FRICTION
        newp = pos[p] + vel[p] * DT
        for a in ti.static(range(3)):
            if newp[a] < 0.0:
                newp[a] = -newp[a]
                vel[p][a] = -vel[p][a]
            elif newp[a] > WORLD:
                newp[a] = 2.0 * WORLD - newp[a]
                vel[p][a] = -vel[p][a]
        pos[p] = newp


def step():
    build_grid()
    compute_forces()
    integrate()


@ti.kernel
def update_colors():
    for p in pos:
        speed = vel[p].norm()
        glow = ti.math.clamp(speed * 6.0, 0.0, 1.0)
        colors[p] = base_colors[p] * (1.0 - 0.5 * glow) + ti.Vector([1.0, 1.0, 1.0]) * (0.5 * glow)


def main():
    init_sim()
    apply_seed(seed_particles(NUM, NSPEC), rule_matrix(NSPEC))
    window = ti.ui.Window("Particle Life 3D — taichi-academy", (900, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.8, 1.8, 1.8)
    camera.lookat(WORLD / 2, WORLD / 2, WORLD / 2)
    running = True
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "r":
                apply_seed(
                    seed_particles(NUM, NSPEC, rng_seed=np.random.randint(1_000_000)),
                    rule_matrix(NSPEC, rng_seed=np.random.randint(1_000_000)),
                )
            elif e.key == ti.ui.SPACE:
                running = not running
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.25, 0.25, 0.25))
        if running:
            for _ in range(SUBSTEPS):
                step()
        update_colors()
        scene.particles(pos, radius=PARTICLE_RADIUS, per_vertex_color=colors)
        canvas.scene(scene)
        with window.GUI.sub_window("Particle Life 3D", 0.02, 0.02, 0.3, 0.12) as gui:
            gui.text(f"{NUM} particles, {NSPEC} species — {'running' if running else 'paused'}")
            gui.text("[space] pause  [r] reroll ecology  [RMB] orbit  [esc] quit")
        window.show()


if __name__ == "__main__":
    main()

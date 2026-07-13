import numpy as np
import taichi as ti

import particle_life_3d as pl


def seed(s=0):
    data = pl.seed_particles(pl.NUM, pl.NSPEC, rng_seed=s)
    rules = pl.rule_matrix(pl.NSPEC, rng_seed=s)
    pl.apply_seed(data, rules)
    return data, rules


def run(n):
    for _ in range(n):
        pl.step()


class resized:
    """Reallocate every field at a different particle count for the block, then restore."""

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.orig = pl.NUM
        pl.NUM = self.n
        pl.init_sim(arch=ti.cpu)
        return pl

    def __exit__(self, *exc):
        pl.NUM = self.orig
        pl.init_sim(arch=ti.cpu)


# --- pure numpy generation ---------------------------------------------------------


def test_species_palette_distinct_and_in_range():
    pal = pl.species_palette(pl.NSPEC)
    assert pal.shape == (pl.NSPEC, 3)
    assert np.all(pal >= 0.0) and np.all(pal <= 1.0)
    for i in range(pl.NSPEC):
        for j in range(i + 1, pl.NSPEC):
            assert np.abs(pal[i] - pal[j]).sum() > 0.1, "hues must be visibly distinct"


def test_seed_particles_bounds_and_determinism():
    a = pl.seed_particles(500, pl.NSPEC, rng_seed=5)
    b = pl.seed_particles(500, pl.NSPEC, rng_seed=5)
    c = pl.seed_particles(500, pl.NSPEC, rng_seed=6)
    pos0, vel0, spec0, col0 = a
    assert pos0.shape == (500, 3) and vel0.shape == (500, 3)
    assert np.all(pos0 >= 0.0) and np.all(pos0 <= pl.WORLD)
    assert np.all(spec0 >= 0) and np.all(spec0 < pl.NSPEC)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])
    assert not np.array_equal(a[0], c[0])


def test_rule_matrix_range_and_determinism():
    a = pl.rule_matrix(pl.NSPEC, rng_seed=9)
    b = pl.rule_matrix(pl.NSPEC, rng_seed=9)
    c = pl.rule_matrix(pl.NSPEC, rng_seed=10)
    assert a.shape == (pl.NSPEC, pl.NSPEC)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- spatial hash correctness --------------------------------------------------------


def test_grid_buckets_every_particle_exactly_once():
    seed(s=1)
    pl.build_grid()
    counts = pl.cell_count.to_numpy()
    starts = pl.cell_start.to_numpy()
    assert counts.sum() == pl.NUM
    assert np.array_equal(starts, np.cumsum(counts) - counts), "prefix sum must be exclusive"
    assert sorted(pl.sorted_idx.to_numpy().tolist()) == list(range(pl.NUM))


def test_grid_forces_match_brute_force():
    """The whole point of the spatial hash: same physics, less work. Prove it."""
    with resized(300):
        data, rules = seed(s=2)
        pl.vel.fill(0.0)
        pl.build_grid()
        pl.compute_forces()
        grid_vel = pl.vel.to_numpy()

    pos0, _vel0, spec0, _col0 = data
    acc = np.zeros_like(pos0)
    for i in range(300):
        for j in range(300):
            if i == j:
                continue
            d = pos0[j] - pos0[i]
            dist = np.linalg.norm(d)
            if 1e-6 < dist < pl.R_MAX:
                r = dist / pl.R_MAX
                a = rules[spec0[i], spec0[j]]
                if r < pl.BETA:
                    f = r / pl.BETA - 1.0
                else:
                    f = a * (1.0 - abs(2 * r - 1 - pl.BETA) / (1 - pl.BETA))
                acc[i] += (d / dist) * f
    brute_vel = acc * pl.FORCE_SCALE * pl.DT
    assert np.allclose(grid_vel, brute_vel, atol=1e-5)


# --- the force law itself -----------------------------------------------------------


def test_close_range_repulsion_is_universal():
    """Inside beta, everyone repels everyone — no rule matrix needed."""
    with resized(2):
        pl.species.from_numpy(np.array([0, 1], dtype=np.int32))
        pl.rules.from_numpy(np.zeros((pl.NSPEC, pl.NSPEC), dtype=np.float32))
        gap = 0.5 * pl.BETA * pl.R_MAX
        p = np.array([[0.5 - gap / 2, 0.5, 0.5], [0.5 + gap / 2, 0.5, 0.5]], dtype=np.float32)
        pl.pos.from_numpy(p)
        pl.vel.fill(0.0)
        pl.build_grid()
        pl.compute_forces()
        v = pl.vel.to_numpy()
        assert v[0][0] < 0.0, "left particle pushed further left"
        assert v[1][0] > 0.0, "right particle pushed further right"


def test_species_attraction_pulls_particles_closer():
    with resized(2):
        pl.species.from_numpy(np.array([0, 1], dtype=np.int32))
        rules = np.zeros((pl.NSPEC, pl.NSPEC), dtype=np.float32)
        rules[0, 1] = rules[1, 0] = 1.0
        pl.rules.from_numpy(rules)
        gap = 0.5 * (pl.BETA + 1.0) * pl.R_MAX
        p = np.array([[0.5 - gap / 2, 0.5, 0.5], [0.5 + gap / 2, 0.5, 0.5]], dtype=np.float32)
        pl.pos.from_numpy(p)
        pl.vel.fill(0.0)
        before = np.linalg.norm(p[0] - p[1])
        pl.build_grid()
        pl.compute_forces()
        pl.integrate()
        after = np.linalg.norm(pl.pos.to_numpy()[0] - pl.pos.to_numpy()[1])
        assert after < before


def test_species_repulsion_pushes_particles_apart():
    with resized(2):
        pl.species.from_numpy(np.array([0, 1], dtype=np.int32))
        rules = np.zeros((pl.NSPEC, pl.NSPEC), dtype=np.float32)
        rules[0, 1] = rules[1, 0] = -1.0
        pl.rules.from_numpy(rules)
        gap = 0.5 * (pl.BETA + 1.0) * pl.R_MAX
        p = np.array([[0.5 - gap / 2, 0.5, 0.5], [0.5 + gap / 2, 0.5, 0.5]], dtype=np.float32)
        pl.pos.from_numpy(p)
        pl.vel.fill(0.0)
        before = np.linalg.norm(p[0] - p[1])
        pl.build_grid()
        pl.compute_forces()
        pl.integrate()
        after = np.linalg.norm(pl.pos.to_numpy()[0] - pl.pos.to_numpy()[1])
        assert after > before


# --- whole-sim behavior ---------------------------------------------------------------


def test_determinism():
    """Same seed replays the same ecology — not bit-for-bit (scatter uses atomics,
    so summation order can vary between runs), but matching to float precision."""
    seed(s=4)
    run(30)
    a = pl.pos.to_numpy()
    seed(s=4)
    run(30)
    b = pl.pos.to_numpy()
    assert np.allclose(a, b, atol=1e-3)


def test_particles_stay_in_bounds_and_finite():
    seed(s=5)
    run(80)
    p = pl.pos.to_numpy()
    v = pl.vel.to_numpy()
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(v))
    assert np.all(p >= 0.0) and np.all(p <= pl.WORLD)


def test_update_colors_bounded():
    seed(s=6)
    run(20)
    pl.update_colors()
    c = pl.colors.to_numpy()
    assert np.all(np.isfinite(c))
    assert c.min() >= -1e-4 and c.max() <= 1.0 + 1e-4


def test_ecology_clusters_over_time():
    """Species affinities should clump particles far more than random chance would."""

    def occupancy_variance(p, bins=8):
        occ = np.zeros((bins, bins, bins))
        idx = np.clip((p / pl.WORLD * bins).astype(int), 0, bins - 1)
        for a, b, c in idx:
            occ[a, b, c] += 1
        return occ.var()

    seed(s=8)
    p0 = pl.pos.to_numpy()
    run(250)
    p1 = pl.pos.to_numpy()
    assert occupancy_variance(p1) > occupancy_variance(p0) * 3

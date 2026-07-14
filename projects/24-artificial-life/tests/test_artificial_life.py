import numpy as np
import taichi as ti

import artificial_life as al


def run(n):
    for _ in range(n):
        al.step()


# --- seeding + spatial hash -----------------------------------------------------------


def test_apply_seed_bounds_and_determinism():
    al.apply_seed(rng_seed=3)
    a = al.pos.to_numpy()
    al.apply_seed(rng_seed=3)
    b = al.pos.to_numpy()
    al.apply_seed(rng_seed=4)
    c = al.pos.to_numpy()
    assert a.min() >= 0.0 and a.max() <= al.WORLD
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_grid_buckets_every_particle_once():
    al.build_grid()
    counts = al.cell_count.to_numpy()
    assert counts.sum() == al.N
    assert sorted(al.sorted_idx.to_numpy().tolist()) == list(range(al.N))


def test_wrapd_is_toroidal():
    @ti.kernel
    def probe(a: ti.f32, b: ti.f32) -> ti.f32:
        return al.wrapd(a, b)

    assert abs(probe(0.02, 0.98) - 0.04) < 1e-5, "0.02 and 0.98 are 0.04 apart across the seam"
    assert abs(probe(0.5, 0.4) - 0.1) < 1e-5, "interior distances are unchanged"


# --- neighbor counting ---------------------------------------------------------------


def test_neighbor_count_matches_brute_force():
    """The spatial hash must give the same neighbor count as an all-pairs scan."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, al.WORLD, (al.N, 2)).astype(np.float32)
    al.pos.from_numpy(p)
    al.build_grid()
    al.count_neighbors()
    n_hash = al.neighbors.to_numpy()

    # brute-force check on a random sample of particles (toroidal distance)
    sample = rng.choice(al.N, 40, replace=False)
    for i in sample:
        dx = np.abs(p[:, 0] - p[i, 0])
        dy = np.abs(p[:, 1] - p[i, 1])
        dx = np.minimum(dx, al.WORLD - dx)
        dy = np.minimum(dy, al.WORLD - dy)
        d2 = dx * dx + dy * dy
        brute = int((d2 < al.R * al.R).sum()) - 1  # minus self
        assert n_hash[i] == brute, f"particle {i}: hash {n_hash[i]} vs brute {brute}"


def test_left_right_split_orients_by_heading():
    """A neighbor to the left of a particle's heading should count as 'left'."""
    # place particle 0 at center facing +x; particle 1 directly north (its left)
    p = np.full((al.N, 2), 0.5, dtype=np.float32)
    p[1] = [0.5, 0.5 + al.R * 0.5]  # north of particle 0
    p[2:] = 10.0  # everyone else far away (wraps out of range... push off-grid)
    # keep them in-world but clustered elsewhere
    rng = np.random.default_rng(0)
    p[2:] = rng.uniform(0, al.WORLD, (al.N - 2, 2)) * 0.0 + 0.05
    al.pos.from_numpy(p)
    h = np.zeros(al.N, dtype=np.float32)  # face +x
    al.heading.from_numpy(h)
    al.build_grid()
    al.turn_and_move()
    # facing +x (east), the north neighbor is on the LEFT -> right<left -> turn -ALPHA-ish
    # we can't read left/right directly, but the heading change encodes sign(right-left):
    # with 1 left neighbor and 0 right, right<left, so dphi = ALPHA - BETA*1
    dphi = (al.heading.to_numpy()[0] - 0.0)
    assert abs(dphi - (al.ALPHA - al.BETA)) < 1e-3, "a left neighbor turns by ALPHA - BETA*N"


def test_turn_rule_sign_flips_with_crowd_side():
    """Mirror the neighbor to the right: the turn should flip to ALPHA + BETA."""
    p = np.full((al.N, 2), 0.5, dtype=np.float32)
    p[1] = [0.5, 0.5 - al.R * 0.5]  # SOUTH of particle 0 (its right, facing east)
    p[2:] = 0.05
    al.pos.from_numpy(p)
    al.heading.from_numpy(np.zeros(al.N, dtype=np.float32))
    al.build_grid()
    al.turn_and_move()
    dphi = al.heading.to_numpy()[0]
    assert abs(dphi - (al.ALPHA + al.BETA)) < 1e-3, "a right neighbor turns by ALPHA + BETA*N"


# --- movement / conservation ---------------------------------------------------------


def test_particles_stay_in_the_torus():
    run(200)
    p = al.pos.to_numpy()
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0 and p.max() < al.WORLD


def test_no_particle_is_lost():
    run(50)
    assert al.pos.to_numpy().shape[0] == al.N


# --- the headline: cells self-organize -------------------------------------------------


def test_cells_emerge_from_uniform_soup():
    """The whole point: a uniform random soup has almost no dense clusters, but the one
    turn rule condenses it into cells — a heavy tail of high-neighbor 'nucleus' particles
    that were not there at the start."""
    al.apply_seed(rng_seed=1)
    al.build_grid()
    al.count_neighbors()
    nucleus_before = int((al.neighbors.to_numpy() > 26).sum())

    run(400)
    nucleus_after = int((al.neighbors.to_numpy() > 26).sum())

    assert nucleus_after > nucleus_before * 4, (
        f"cells should condense: nucleus particles {nucleus_before} -> {nucleus_after}"
    )


def test_soup_stays_near_target_density():
    run(400)
    n = al.neighbors.to_numpy()
    assert 8 < n.mean() < 25, f"mean neighbor count should stay in the cell-forming band, got {n.mean():.1f}"


# --- interaction / render ------------------------------------------------------------


def test_stir_points_particles_outward():
    al.apply_seed(rng_seed=1)
    al.stir(0.5, 0.5)
    p = al.pos.to_numpy()
    h = al.heading.to_numpy()
    near = (np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5) < al.STIR_RADIUS)
    assert near.sum() > 0
    # a stirred particle should face away from the stir centre
    idx = np.where(near)[0][0]
    expected = np.arctan2(p[idx, 1] - 0.5, p[idx, 0] - 0.5)
    assert abs(np.angle(np.exp(1j * (h[idx] - expected)))) < 1e-3


def test_render_is_finite_and_bounded():
    run(100)
    al.render()
    px = al.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0

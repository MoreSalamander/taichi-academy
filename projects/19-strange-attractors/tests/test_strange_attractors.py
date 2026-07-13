import numpy as np

import strange_attractors as sa


# --- pure numpy generation ---------------------------------------------------------


def test_seed_points_scale_and_determinism():
    a = sa.seed_points(1000, 8.0, rng_seed=3)
    b = sa.seed_points(1000, 8.0, rng_seed=3)
    c = sa.seed_points(1000, 8.0, rng_seed=4)
    assert a.shape == (1000, 3)
    assert np.abs(a).max() <= 8.0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- the dynamics ---------------------------------------------------------------------


def test_every_attractor_stays_finite_and_bounded():
    for kind in (sa.LORENZ, sa.THOMAS, sa.AIZAWA, sa.CLIFFORD):
        _s, _cx, _cy, _cz, dt, _g, seed_scale, _sp = sa.FRAME[kind]
        sa.pos.from_numpy(sa.seed_points(sa.N_PTS, seed_scale, rng_seed=1))
        for _ in range(3000):
            sa.step_attractor(kind, dt)
        p = sa.pos.to_numpy()
        assert np.all(np.isfinite(p)), f"{sa.NAMES[kind]} diverged"
        assert np.abs(p).max() < 100.0, f"{sa.NAMES[kind]} escaped to {np.abs(p).max():.1f}"


def test_attraction_contracts_the_cloud_onto_a_set():
    """The defining property: wildly different starting clouds end up on the SAME set.
    Compare occupancy grids of two runs from different seeds — they should overlap."""
    kind = sa.LORENZ
    _s, _cx, _cy, _cz, dt, _g, seed_scale, _sp = sa.FRAME[kind]

    def occupancy(rng_seed):
        sa.pos.from_numpy(sa.seed_points(sa.N_PTS, seed_scale, rng_seed))
        for _ in range(4000):
            sa.step_attractor(kind, dt)
        p = sa.pos.to_numpy()
        H, _, _ = np.histogram2d(p[:, 0], p[:, 2], bins=32, range=[[-25, 25], [0, 50]])
        return H > 0

    a = occupancy(1)
    b = occupancy(99)
    overlap = (a & b).sum() / max((a | b).sum(), 1)
    assert overlap > 0.75, f"two different clouds should trace the same attractor, overlap={overlap:.2f}"


def test_lorenz_visits_both_wings():
    _s, _cx, _cy, _cz, dt, _g, seed_scale, _sp = sa.FRAME[sa.LORENZ]
    sa.pos.from_numpy(sa.seed_points(sa.N_PTS, seed_scale, rng_seed=1))
    for _ in range(3000):
        sa.step_attractor(sa.LORENZ, dt)
    x = sa.pos.to_numpy()[:, 0]
    assert (x > 2).any() and (x < -2).any(), "points should populate both butterfly wings"


def test_clifford_is_a_map_not_a_flow():
    """One step of Clifford lands INSIDE its bounded square no matter how far out you
    start — a discrete map teleports, it doesn't integrate."""
    p = np.full((sa.N_PTS, 3), 500.0, dtype=np.float32)
    sa.pos.from_numpy(p)
    sa.step_attractor(sa.CLIFFORD, 1.0)
    out = sa.pos.to_numpy()
    assert np.abs(out[:, :2]).max() <= 1.0 + 1.0 + 1e-4, "sin+cos bounds every clifford step"
    assert np.all(out[:, 2] == 0.0), "clifford is 2D — z stays zero"


def test_thomas_is_cyclically_symmetric():
    """Thomas' equations map (x,y,z) -> (y,z,x) symmetrically: statistics match per axis."""
    _s, _cx, _cy, _cz, dt, _g, seed_scale, _sp = sa.FRAME[sa.THOMAS]
    sa.pos.from_numpy(sa.seed_points(sa.N_PTS, seed_scale, rng_seed=2))
    for _ in range(4000):
        sa.step_attractor(sa.THOMAS, dt)
    p = sa.pos.to_numpy()
    stds = p.std(axis=0)
    assert np.abs(stds - stds.mean()).max() < 0.1, f"axis spreads should match: {stds}"


# --- rendering -----------------------------------------------------------------------


def test_apply_seed_settles_and_clears():
    sa.apply_seed(sa.AIZAWA, rng_seed=1)
    p = sa.pos.to_numpy()
    assert np.all(np.isfinite(p))
    assert np.abs(p).max() < 3.0, "after settling, points live on the (small) aizawa set"
    assert sa.pixels.to_numpy().sum() == 0.0


def test_step_renders_finite_bounded_pixels():
    sa.apply_seed(sa.THOMAS, rng_seed=1)
    for f in range(30):
        sa.step(sa.THOMAS, f * 0.01)
    px = sa.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0
    assert px.sum() > 0


def test_rotation_changes_the_view_but_not_the_state():
    sa.apply_seed(sa.LORENZ, rng_seed=1)
    p_before = sa.pos.to_numpy().copy()
    sa.fade()
    scale, cx, cy, cz, _dt, gain, _seed, spd = sa.FRAME[sa.LORENZ]
    sa.splat(sa.LORENZ, 0.0, scale, cx, cy, cz, gain, spd)
    a = sa.pixels.to_numpy().copy()
    sa.pixels.fill(0.0)
    sa.splat(sa.LORENZ, 1.5, scale, cx, cy, cz, gain, spd)
    b = sa.pixels.to_numpy()
    assert not np.allclose(a, b), "a different angle draws a different projection"
    assert np.array_equal(p_before, sa.pos.to_numpy()), "splat never touches the points"

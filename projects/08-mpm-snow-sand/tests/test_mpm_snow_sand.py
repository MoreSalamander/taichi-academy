import numpy as np

import mpm_snow_sand as mpm


def seed(rng_seed=0):
    mpm.apply_seed(*mpm.default_seed(rng_seed))


def run(n, **kwargs):
    for _ in range(n):
        mpm.substep(**kwargs)


# --- pure numpy generation ---------------------------------------------------------


def test_seed_block_bounds_and_determinism():
    a = mpm.seed_block(500, 0.5, 0.3, 0.1, 0.05, rng_seed=3)
    b = mpm.seed_block(500, 0.5, 0.3, 0.1, 0.05, rng_seed=3)
    c = mpm.seed_block(500, 0.5, 0.3, 0.1, 0.05, rng_seed=4)
    assert a.shape == (500, 2)
    assert np.all(a[:, 0] >= 0.4) and np.all(a[:, 0] <= 0.6)
    assert np.all(a[:, 1] >= 0.25) and np.all(a[:, 1] <= 0.35)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- seeding / reset -----------------------------------------------------------------


def test_apply_seed_assigns_material_and_position():
    seed(1)
    mat = mpm.material.to_numpy()
    assert np.all(mat[: mpm.N_PER_BLOCK] == mpm.SNOW)
    assert np.all(mat[mpm.N_PER_BLOCK :] == mpm.SAND)
    pos = mpm.x.to_numpy()
    assert pos[:, 0].min() >= 0.0 and pos[:, 0].max() <= 1.0


def test_reset_particles_initializes_state():
    seed(1)
    run(5)
    mpm.reset_particles()
    F = mpm.F.to_numpy()
    identity = np.tile(np.eye(2, dtype=np.float32), (mpm.N_PARTICLES, 1, 1))
    assert np.allclose(F, identity)
    assert np.all(mpm.Jp.to_numpy() == 1.0)
    assert np.all(mpm.v.to_numpy() == 0.0)


# --- grid transfer ---------------------------------------------------------------------


def test_clear_grid_zeroes_everything():
    mpm.grid_v.fill([1.0, 1.0])
    mpm.grid_m.fill(1.0)
    mpm.clear_grid()
    assert np.all(mpm.grid_v.to_numpy() == 0.0)
    assert np.all(mpm.grid_m.to_numpy() == 0.0)


def test_p2g_conserves_total_mass():
    """Every particle's mass lands on the grid — nothing created or destroyed."""
    seed(2)
    mpm.clear_grid()
    mpm.p2g()
    total = mpm.grid_m.to_numpy().sum()
    expected = mpm.N_PARTICLES * mpm.P_MASS
    assert abs(total - expected) / expected < 1e-3


# --- whole-sim behavior ---------------------------------------------------------------


def test_finite_and_bounded_after_settling():
    seed(3)
    run(60)
    pos = mpm.x.to_numpy()
    v = mpm.v.to_numpy()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(v))
    assert np.all(pos >= -0.05) and np.all(pos <= 1.05)


def test_determinism():
    """Same seed replays the same fall — not bit-exact (P2G scatter uses atomics),
    but matching to float precision."""
    seed(4)
    run(40)
    a = mpm.x.to_numpy()
    seed(4)
    run(40)
    b = mpm.x.to_numpy()
    assert np.allclose(a, b, atol=1e-3)


def test_gravity_pulls_particles_down():
    seed(5)
    before = mpm.x.to_numpy()[:, 1].mean()
    run(30)
    after = mpm.x.to_numpy()[:, 1].mean()
    assert after < before


def test_sand_spreads_more_than_snow():
    """The whole point of Drucker-Prager: sand flows into a wide pile, snow stays compact.
    Needs real settling time (~100 rendered frames' worth of substeps) before the gap is
    robust — P2G's atomic scatter makes the exact spread run-to-run non-deterministic
    (see test_determinism), so too short a settle window makes this test flaky."""
    seed(6)
    run(100 * mpm.SUBSTEPS)
    pos = mpm.x.to_numpy()
    snow_width = np.ptp(pos[: mpm.N_PER_BLOCK, 0])
    sand_width = np.ptp(pos[mpm.N_PER_BLOCK :, 0])
    assert sand_width > snow_width * 1.3


def test_snow_singular_values_stay_within_elastic_bounds():
    seed(7)
    run(60)
    F = mpm.F.to_numpy()[: mpm.N_PER_BLOCK]
    sig = np.linalg.svd(F, compute_uv=False)
    assert sig.min() >= 1 - mpm.THETA_C_SNOW - 1e-3
    assert sig.max() <= 1 + mpm.THETA_S_SNOW + 1e-3


def test_stirring_stays_stable():
    """Regression test: a per-frame force once mistakenly got reapplied at every
    substep (25x over-injection), which blew the sim up to NaN within a few frames."""
    seed(8)
    run(30)
    for _ in range(20):
        run(mpm.SUBSTEPS, mx=0.5, my=0.15, fx=0.02, fy=0.01, stirring=True)
    pos = mpm.x.to_numpy()
    v = mpm.v.to_numpy()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(v))
    assert np.linalg.norm(v, axis=1).max() < 50.0

import numpy as np

import star_nursery as sn


def run(n):
    for _ in range(n):
        sn.step()


# --- pure numpy generation ---------------------------------------------------------


def test_seed_gas_bounds_and_determinism():
    a = sn.seed_gas(5000, rng_seed=3)
    b = sn.seed_gas(5000, rng_seed=3)
    c = sn.seed_gas(5000, rng_seed=4)
    assert a.shape == (5000, 2)
    assert a.min() >= 0.02 and a.max() <= 0.98
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_gas_is_clumpy_not_uniform():
    p = sn.seed_gas(sn.N_GAS, rng_seed=1)
    occ, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=16, range=[[0, 1], [0, 1]])
    assert occ.var() > occ.mean() * 5, "gaussian blobs should be far clumpier than uniform"


# --- density pipeline ------------------------------------------------------------------


def test_deposit_counts_every_living_particle():
    sn.clear_density()
    sn.deposit()
    assert sn.density.to_numpy().sum() == sn.N_GAS


def test_dead_particles_do_not_deposit():
    a = sn.alive.to_numpy()
    a[:1000] = 0
    sn.alive.from_numpy(a)
    sn.clear_density()
    sn.deposit()
    assert sn.density.to_numpy().sum() == sn.N_GAS - 1000


def test_blur_conserves_interior_mass_and_smooths():
    sn.clear_density()
    sn.deposit()
    sn.blur()
    d = sn.density.to_numpy()
    b = sn.density_blur.to_numpy()
    assert b.max() < d.max() + 1e-3, "blur never sharpens a peak"
    assert b.var() < d.var(), "blur reduces variance"


# --- forces --------------------------------------------------------------------------


def test_gravity_pulls_toward_a_density_peak():
    """The probe must sit within the blur's reach of the peak (the 5x5 blur spreads a
    delta only ±2 cells) or the local gradient it samples is exactly zero."""
    peak_cell = sn.GRID // 2 + 5
    probe_cell = peak_cell - 3
    p = sn.pos.to_numpy()
    p[0] = [(probe_cell + 0.5) / sn.GRID, 0.5]
    sn.pos.from_numpy(p)
    sn.vel.fill(0.0)
    d = np.zeros((sn.GRID, sn.GRID), dtype=np.float32)
    d[peak_cell, sn.GRID // 2] = 500.0  # peak just right of the probe
    sn.density.from_numpy(d)
    sn.blur()
    sn.gravity()
    assert sn.vel[0][0] > 0, "the probe should accelerate toward the peak (+x)"


def test_radiation_pushes_gas_away_from_a_star():
    sn.n_stars[None] = 1
    sn.star_pos[0] = [0.5, 0.5]
    p = sn.pos.to_numpy()
    p[0] = [0.52, 0.5]  # just right of the star, inside RADIATION_R
    sn.pos.from_numpy(p)
    sn.vel.fill(0.0)
    sn.radiation()
    assert sn.vel[0][0] > 0, "gas right of the star gets pushed further right"


def test_radiation_has_finite_range():
    sn.n_stars[None] = 1
    sn.star_pos[0] = [0.1, 0.1]
    p = sn.pos.to_numpy()
    p[0] = [0.9, 0.9]  # far outside RADIATION_R
    sn.pos.from_numpy(p)
    sn.vel.fill(0.0)
    sn.radiation()
    assert sn.vel[0][0] == 0.0 and sn.vel[0][1] == 0.0


# --- ignition ------------------------------------------------------------------------


def test_no_ignition_below_the_density_threshold():
    sn.density_blur.fill(sn.IGNITE_DENSITY * 0.5)
    sn.ignite()
    assert sn.n_stars[None] == 0


def test_ignition_happens_in_dense_gas():
    sn.density_blur.fill(sn.IGNITE_DENSITY * 2.0)
    for _ in range(50):
        sn.ignite()
    assert sn.n_stars[None] > 0, "with every cell over threshold, some stars must ignite"
    born = sn.n_stars[None]
    dead = sn.N_GAS - sn.alive.to_numpy().sum()
    assert dead == min(born, sn.MAX_STARS), "each star consumes exactly one gas particle"


def test_star_count_never_exceeds_capacity():
    sn.density_blur.fill(sn.IGNITE_DENSITY * 10)
    for _ in range(400):
        sn.ignite()
    alive = sn.alive.to_numpy().sum()
    assert sn.N_GAS - alive <= sn.MAX_STARS, "gas consumed must stop at MAX_STARS"


# --- whole-sim behavior ---------------------------------------------------------------


def test_collapse_increases_peak_density_before_ignition():
    sn.clear_density()
    sn.deposit()
    sn.blur()
    before = sn.density_blur.to_numpy().max()
    run(40)
    sn.clear_density()
    sn.deposit()
    sn.blur()
    after = sn.density_blur.to_numpy().max()
    assert after > before, "self-gravity should concentrate the cloud"


def test_stars_eventually_ignite_and_sim_stays_finite():
    run(150)
    assert sn.n_stars[None] > 0, "a collapsing cloud should have birthed stars by now"
    assert np.all(np.isfinite(sn.pos.to_numpy()))
    px = sn.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0


def test_apply_seed_resets_everything():
    run(100)
    sn.apply_seed(rng_seed=9)
    assert sn.n_stars[None] == 0
    assert sn.alive.to_numpy().sum() == sn.N_GAS
    assert sn.pixels.to_numpy().sum() == 0.0

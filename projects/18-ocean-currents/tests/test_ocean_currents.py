import numpy as np

import ocean_currents as oc


def run(n):
    for _ in range(n):
        oc.step()


# --- pure numpy generation ---------------------------------------------------------


def test_seed_continents_ocean_fraction_and_determinism():
    a = oc.seed_continents(oc.N, rng_seed=3)
    b = oc.seed_continents(oc.N, rng_seed=3)
    c = oc.seed_continents(oc.N, rng_seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    ocean = (a == 0).mean()
    assert abs(ocean - oc.OCEAN_FRACTION) < 0.02, f"ocean fraction {ocean:.3f}"


def test_seed_temperature_warm_equator_cold_poles():
    t = oc.seed_temperature(oc.N)
    assert t[:, oc.N // 2].mean() > 0.95, "the equator row is hot"
    assert t[:, 0].mean() < 0.05 and t[:, -1].mean() < 0.05, "both pole rows are cold"


# --- forcing -------------------------------------------------------------------------


def test_wind_bands_alternate_direction_with_latitude():
    """Sample the equator (easterly trades) and the westerlies PEAK at lat ~0.31 —
    an earlier version sampled lat 0.5, which is exactly a band's zero crossing."""
    oc.land.fill(0)
    oc.vel.fill(0.0)
    oc.wind_forcing()
    v = oc.vel.to_numpy()
    equator = v[10, oc.N // 2, 0]
    westerlies = v[10, int(oc.N * 0.66), 0]
    assert equator < 0, "trades blow east-to-west at the equator"
    assert westerlies > 0, "westerlies blow west-to-east at mid latitudes"


def test_wind_skips_land():
    mask = np.zeros((oc.N, oc.N), dtype=np.int32)
    mask[50, 50] = 1
    oc.land.from_numpy(mask)
    oc.vel.fill(0.0)
    oc.wind_forcing()
    v = oc.vel.to_numpy()
    assert v[50, 50, 0] == 0.0
    assert v[51, 50, 0] != 0.0


def test_coriolis_deflects_opposite_ways_per_hemisphere():
    oc.land.fill(0)
    v = np.zeros((oc.N, oc.N, 2), dtype=np.float32)
    v[:, :, 0] = 1.0  # everything moving +x
    oc.vel.from_numpy(v)
    oc.coriolis()
    out = oc.vel.to_numpy()
    north = out[10, int(oc.N * 0.8), 1]
    south = out[10, int(oc.N * 0.2), 1]
    assert north * south < 0, "the same eastward flow deflects opposite ways in the two hemispheres"


def test_storm_spins_opposite_ways_per_hemisphere():
    oc.land.fill(0)
    oc.vel.fill(0.0)
    oc.storm(0.5, 0.8)  # northern hemisphere
    curl_n = _curl_at(int(oc.N * 0.5), int(oc.N * 0.8))
    oc.vel.fill(0.0)
    oc.storm(0.5, 0.2)  # southern hemisphere
    curl_s = _curl_at(int(oc.N * 0.5), int(oc.N * 0.2))
    assert curl_n * curl_s < 0, "cyclones spin opposite directions across the equator"


def _curl_at(ci, cj):
    v = oc.vel.to_numpy()
    return (
        v[ci + 1, cj, 1] - v[ci - 1, cj, 1] - v[ci, cj + 1, 0] + v[ci, cj - 1, 0]
    )


# --- land / boundaries -----------------------------------------------------------------


def test_enforce_land_zeroes_velocity_on_land():
    oc.vel.fill([1.0, 1.0])
    oc.enforce_land()
    v = oc.vel.to_numpy()
    mask = oc.land.to_numpy()
    assert np.all(v[mask == 1] == 0.0)
    assert np.any(v[mask == 0] != 0.0)


def test_relax_temp_pulls_toward_latitude_profile():
    oc.temp.fill(0.5)
    oc.relax_temp()
    t = oc.temp.to_numpy()
    assert t[10, oc.N // 2] > 0.5, "equator warms toward its target"
    assert t[10, 2] < 0.5, "pole cools toward its target"


# --- whole-sim behavior ---------------------------------------------------------------


def test_long_run_stays_finite_and_bounded():
    run(200)
    v = oc.vel.to_numpy()
    t = oc.temp.to_numpy()
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(t))
    assert np.linalg.norm(v, axis=2).max() < 50.0
    oc.render()
    px = oc.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0


def test_currents_transport_heat_poleward():
    """The point of the whole simulation: relative to the sun's pure latitude profile,
    the moving ocean EXPORTS heat from the equator (equatorial seas run cooler than
    the target) and IMPORTS it at the poles (polar seas run warmer)."""
    run(400)
    t = oc.temp.to_numpy()
    mask = oc.land.to_numpy()
    target = oc.seed_temperature(oc.N)
    diff = t - target
    jj = np.arange(oc.N)
    lat = np.abs(jj - oc.N / 2) / (oc.N / 2)
    eq_band = lat < 0.2
    polar_band = lat >= 0.8
    eq_sea = mask[:, eq_band] == 0
    polar_sea = mask[:, polar_band] == 0
    assert diff[:, eq_band][eq_sea].mean() < -0.1, "equatorial seas export their heat"
    assert diff[:, polar_band][polar_sea].mean() > 0.0, "polar seas import it"


def test_determinism_is_bit_exact():
    oc.apply_seed(rng_seed=5)
    run(50)
    a = oc.temp.to_numpy()
    oc.apply_seed(rng_seed=5)
    run(50)
    b = oc.temp.to_numpy()
    assert np.array_equal(a, b)


def test_apply_seed_resets_flow():
    run(50)
    oc.apply_seed(rng_seed=2)
    assert np.all(oc.vel.to_numpy() == 0.0)

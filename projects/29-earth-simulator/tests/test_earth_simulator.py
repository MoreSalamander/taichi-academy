import numpy as np

import earth_simulator as es


def run(n, a_olr=es.A_OLR):
    for _ in range(n):
        es.step(a_olr)


# --- geography -----------------------------------------------------------------------


def test_make_land_is_a_binary_map_with_polar_oceans():
    m = es.make_land(3)
    assert set(np.unique(m)).issubset({0, 1})
    assert (m[:, :6] == 0).all() and (m[:, -6:] == 0).all(), "poles are open ocean"
    assert 0.1 < m.mean() < 0.7, "a sensible mix of land and sea"


def test_seed_is_warm_equator_cold_poles():
    eq = es.band_temp(es.H // 2 - 4, es.H // 2 + 4)
    pole = es.band_temp(0, 8)
    assert eq > 20.0 and pole < 0.0, f"seed climate should slope from warm to cold: {eq:.0f} vs {pole:.0f}"
    assert es.clock[None] == 0.0


# --- energy balance ------------------------------------------------------------------


def test_climate_is_stable_and_banded():
    """The headline: run a few simulated years and the planet settles into a stable climate —
    finite everywhere, a warm equator, cold poles, and ice caps that are neither absent nor
    a total snowball."""
    run(400)
    T = es.T.to_numpy()
    assert np.all(np.isfinite(T)), "no blow-up"
    eq = es.band_temp(es.H // 2 - 4, es.H // 2 + 4)
    pole = es.band_temp(0, 8)
    assert eq > pole + 30.0, f"a real equator-to-pole gradient survives: {eq:.0f} vs {pole:.0f}"
    ice = es.ice_fraction()
    assert 0.15 < ice < 0.85, f"ice caps present but not a snowball: {ice:.2f}"


def test_more_greenhouse_means_less_ice():
    """The ice-albedo / greenhouse lever: a warmer atmosphere (lower A_OLR) shrinks the ice."""
    run(250, a_olr=205.0)   # weak greenhouse -> cold
    cold_ice = es.ice_fraction()
    es.apply_seed(3)
    run(250, a_olr=182.0)   # strong greenhouse -> warm
    warm_ice = es.ice_fraction()
    assert warm_ice < cold_ice - 0.05, f"more greenhouse should melt ice: {cold_ice:.2f} -> {warm_ice:.2f}"


# --- seasons -------------------------------------------------------------------------


def test_seasons_shift_the_warm_band():
    """A quarter-orbit past the equinox the sub-solar point has climbed north, so the northern
    mid-latitudes outwarm the southern ones — seasons, from axial tilt alone."""
    run(int(es.YEAR / 4) + 20)   # into northern summer
    north = es.band_temp(int(es.H * 0.70), int(es.H * 0.85))
    south = es.band_temp(int(es.H * 0.15), int(es.H * 0.30))
    assert north > south, f"northern summer should be warmer than the south: {north:.0f} vs {south:.0f}"


# --- the water cycle -----------------------------------------------------------------


def test_vegetation_grows_on_temperate_land():
    run(400)
    v = es.veg.to_numpy()
    ln = es.land.to_numpy()
    assert v[ln == 1].max() > 0.1, "rain and warmth should green some land"
    assert (v[ln == 0] == 0.0).all(), "nothing grows on the ocean"


# --- render --------------------------------------------------------------------------


def test_render_is_finite_and_bounded():
    run(60)
    es.render()
    px = es.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

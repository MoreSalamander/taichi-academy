import numpy as np

import galaxy as g


def run(n):
    for _ in range(n):
        g.step()


# --- pure numpy generation ---------------------------------------------------------


def test_disk_radii_bounds_and_no_edge_pileup():
    """Regression test: clipping exponential radii used to pile thousands of stars at
    exactly r_max, drawing a bright artificial circle at the galaxy's rim — re-rolling
    the out-of-range ones scatters them instead."""
    rng = np.random.default_rng(1)
    r = g.disk_radii(50000, rng)
    assert r.min() >= 0.01 and r.max() <= 0.85
    at_edge = (r > 0.849).sum()
    assert at_edge < 100, f"{at_edge} stars piled at the rim"


def test_seed_spiral_shape_and_determinism():
    a = g.seed_spiral(g.N_STARS, rng_seed=3)
    b = g.seed_spiral(g.N_STARS, rng_seed=3)
    c = g.seed_spiral(g.N_STARS, rng_seed=4)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(arr_a, arr_b)
    assert not np.array_equal(a[1], c[1])
    assert a[0].shape == (g.N_STARS,) and a[2].shape == (g.N_STARS, 3)


def test_spiral_arms_correlate_angle_with_radius():
    """A log spiral means theta grows with log(r) — check the correlation on one arm."""
    r, theta, _col = g.seed_spiral(g.N_STARS, rng_seed=5, arms=1, twist=3.5)
    corr = np.corrcoef(np.log(r), theta)[0, 1]
    assert corr > 0.7, f"arm angle should track log radius, corr={corr:.2f}"


def test_elliptical_has_no_arm_structure():
    r, theta, _col = g.seed_elliptical(g.N_STARS, rng_seed=5)
    corr = abs(np.corrcoef(np.log(r), theta)[0, 1])
    assert corr < 0.1, f"elliptical should have no radius-angle correlation, corr={corr:.2f}"


def test_ring_concentrates_radius_in_a_band():
    r, _theta, _col = g.seed_ring(g.N_STARS, rng_seed=5)
    ring_stars = ((r > 0.4) & (r < 0.7)).mean()
    assert ring_stars > 0.6, "most stars live in the ring band"


def test_star_colors_bounded_and_core_is_warmer():
    rng = np.random.default_rng(0)
    r = np.array([0.02, 0.8], dtype=np.float32)
    col = g.star_colors(r, rng)
    assert col.min() >= 0.0 and col.max() <= 1.0
    core, arm = col[0], col[1]
    assert core[0] / (core[2] + 1e-6) > arm[0] / (arm[2] + 1e-6), "core skews red/yellow vs arm blue"


# --- kernels -------------------------------------------------------------------------


def test_rotate_is_differential():
    """Inner stars sweep more angle than outer stars — the winding that shapes arms."""
    r = np.full(g.N_STARS, 0.7, dtype=np.float32)
    r[0] = 0.1
    theta = np.zeros(g.N_STARS, dtype=np.float32)
    g.radius_f.from_numpy(r)
    g.angle_f.from_numpy(theta)
    g.rotate(0.1)
    a = g.angle_f.to_numpy()
    assert a[0] > a[1] * 2, "the inner star should sweep much faster"


def test_fade_dims_the_canvas():
    g.pixels.fill(1.0)
    g.fade()
    assert np.allclose(g.pixels.to_numpy(), g.FADE, atol=1e-5)


def test_splat_deposits_light_inside_the_frame():
    g.pixels.fill(0.0)
    g.splat()
    assert g.pixels.to_numpy().sum() > 0


def test_step_stays_finite_and_bounded():
    run(150)
    px = g.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0


def test_apply_seed_clears_the_canvas():
    run(30)
    g.apply_seed(g.seed_galaxy(g.RING, rng_seed=1))
    assert g.pixels.to_numpy().sum() == 0.0


def test_determinism():
    """rotate/fade/clamp write only their own index; splat scatters with atomic +=,
    but every star adds the same value regardless of order — float rounding can still
    differ, so allclose rather than bit-equal."""
    g.apply_seed(g.seed_galaxy(g.SPIRAL, rng_seed=7))
    run(40)
    a = g.pixels.to_numpy()
    g.apply_seed(g.seed_galaxy(g.SPIRAL, rng_seed=7))
    run(40)
    b = g.pixels.to_numpy()
    assert np.allclose(a, b, atol=1e-3)

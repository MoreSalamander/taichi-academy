import numpy as np

import fire


def center_of_heat():
    t = fire.temp.to_numpy()
    total = t.sum()
    jj = np.arange(fire.N)
    return float((t.sum(axis=0) * jj).sum() / total)


def run(n_steps, source=True, curl=None):
    strength = fire.CURL_STRENGTH if curl is None else curl
    for k in range(n_steps):
        if source:
            fire.burn_source(float(k))
        fire.step(strength)


# --- seed: pure numpy -----------------------------------------------------------


def test_seed_pattern_shape_range_determinism():
    a = fire.seed_pattern(fire.N, rng_seed=4)
    b = fire.seed_pattern(fire.N, rng_seed=4)
    c = fire.seed_pattern(fire.N, rng_seed=5)
    assert a.shape == (fire.N, fire.N)
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- physics invariants -----------------------------------------------------------


def test_buoyancy_lifts_hot_cells():
    fire.temp.from_numpy(np.ones((fire.N, fire.N), np.float32))
    fire.apply_buoyancy()
    v = fire.vel.to_numpy()
    assert np.all(v[:, :, 1] > 0.0), "heat must push upward"
    assert np.allclose(v[:, :, 0], 0.0), "buoyancy is vertical only"


def test_fire_rises():
    """The center of heat must climb as the seed blob convects upward."""
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    before = center_of_heat()
    run(100, source=False)
    after = center_of_heat()
    assert after > before + 5.0, f"blob did not rise: {before:.1f} -> {after:.1f}"


def test_cooling_fades_heat():
    fire.temp.from_numpy(np.ones((fire.N, fire.N), np.float32))
    for _ in range(200):
        fire.cool()
    assert fire.temp.to_numpy().max() < 0.1


def test_burn_source_bounded_and_local():
    fire.clear_fields()
    for k in range(300):
        fire.burn_source(float(k))
    t = fire.temp.to_numpy()
    assert t.max() <= 1.5 + 1e-5, "source must saturate, not accumulate"
    assert t[fire.N // 2, 12] > 1.0, "hot at the hearth"
    assert t[fire.N // 2, fire.N - 20] < 1e-3, "cold far above (no advection yet)"


def test_advect_zero_velocity_is_identity():
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    before = fire.temp.to_numpy()
    fire.advect(fire.temp, fire.temp_next)
    assert np.allclose(before, fire.temp_next.to_numpy(), atol=1e-6)


def test_projection_reduces_divergence():
    rng = np.random.default_rng(0)
    fire.vel.from_numpy(rng.uniform(-1, 1, size=(fire.N, fire.N, 2)).astype(np.float32))
    fire.compute_divergence()
    before = np.abs(fire.divergence.to_numpy()).mean()
    fire.project()
    fire.compute_divergence()
    after = np.abs(fire.divergence.to_numpy()).mean()
    assert after < before * 0.2


# --- full runs ----------------------------------------------------------------------


def test_determinism():
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    run(60)
    first_t, first_v = fire.temp.to_numpy(), fire.vel.to_numpy()
    fire.clear_fields()
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    run(60)
    assert np.array_equal(fire.temp.to_numpy(), first_t)
    assert np.array_equal(fire.vel.to_numpy(), first_v)


def test_finite_and_bounded_after_300_steps():
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    run(300)
    for arr in (fire.temp.to_numpy(), fire.smoke.to_numpy(), fire.vel.to_numpy()):
        assert np.all(np.isfinite(arr))
    assert fire.temp.to_numpy().max() <= 1.5 + 1e-5
    assert np.abs(fire.vel.to_numpy()).max() < 50.0


def test_render_and_clear():
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    run(30)
    fire.render()
    px = fire.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    fire.clear_fields()
    assert fire.temp.to_numpy().max() == 0.0
    assert np.abs(fire.vel.to_numpy()).max() == 0.0

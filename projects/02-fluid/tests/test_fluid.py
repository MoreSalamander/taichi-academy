import numpy as np

import fluid


def rigid_vortex(n, omega=0.01):
    """Divergence-free rigid rotation about the grid center (pure numpy)."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.zeros((n, n, 2), dtype=np.float32)
    v[:, :, 0] = -(jj - n / 2) * omega
    v[:, :, 1] = (ii - n / 2) * omega
    return v


# --- seed_pattern: pure numpy ---------------------------------------------------


def test_seed_pattern_shape_range_determinism():
    a = fluid.seed_pattern(fluid.N, rng_seed=5)
    b = fluid.seed_pattern(fluid.N, rng_seed=5)
    c = fluid.seed_pattern(fluid.N, rng_seed=6)
    assert a.shape == (fluid.N, fluid.N, 3)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert a.max() > 0.5, "blobs should be visible"
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- advection -------------------------------------------------------------------


def test_advect_zero_velocity_is_identity():
    """With no velocity, backtrace lands on the grid point — bilerp must return it exactly."""
    fluid.apply_seed(fluid.seed_pattern(fluid.N, rng_seed=3))
    before = fluid.dye.to_numpy()
    fluid.advect(fluid.dye, fluid.dye_next)
    after = fluid.dye_next.to_numpy()
    assert np.allclose(before, after, atol=1e-6)


def test_advect_conserves_range():
    """Bilerp is an average — it can never exceed the field's existing bounds."""
    fluid.apply_seed(fluid.seed_pattern(fluid.N, rng_seed=3))
    fluid.vel.from_numpy(rigid_vortex(fluid.N))
    for _ in range(20):
        fluid.advect(fluid.dye, fluid.dye_next)
        fluid.advect(fluid.vel, fluid.vel_next)
        fluid.copy_back()
    d = fluid.dye.to_numpy()
    assert d.min() >= -1e-4 and d.max() <= 1.0 + 1e-4


# --- incompressibility ------------------------------------------------------------


def test_rigid_vortex_is_divergence_free():
    fluid.vel.from_numpy(rigid_vortex(fluid.N))
    fluid.compute_divergence()
    div = fluid.divergence.to_numpy()
    interior = div[2:-2, 2:-2]  # wrap seam at the edges is expected for a non-periodic vortex
    assert np.abs(interior).max() < 1e-4


def test_projection_reduces_divergence():
    rng = np.random.default_rng(0)
    noisy = rng.uniform(-1, 1, size=(fluid.N, fluid.N, 2)).astype(np.float32)
    fluid.vel.from_numpy(noisy)
    fluid.compute_divergence()
    before = np.abs(fluid.divergence.to_numpy()).mean()
    fluid.project()
    fluid.compute_divergence()
    after = np.abs(fluid.divergence.to_numpy()).mean()
    assert after < before * 0.2, f"projection too weak: {before:.4f} -> {after:.4f}"


# --- interaction ------------------------------------------------------------------


def test_splat_adds_dye_and_velocity():
    fluid.clear_fields()
    fluid.splat(0.5, 0.5, 3.0, 0.0, 1.0, 0.2, 0.1)
    c = fluid.N // 2
    d = fluid.dye.to_numpy()
    v = fluid.vel.to_numpy()
    assert d[c, c, 0] > 0.5
    assert v[c, c, 0] > 1.0
    assert d[10, 10].sum() < 1e-3, "far corner untouched"


def test_clear_fields_zeroes_everything():
    fluid.apply_seed(fluid.seed_pattern(fluid.N, rng_seed=3))
    fluid.splat(0.5, 0.5, 3.0, 0.0, 1.0, 1.0, 1.0)
    fluid.clear_fields()
    assert np.abs(fluid.dye.to_numpy()).max() == 0.0
    assert np.abs(fluid.vel.to_numpy()).max() == 0.0


# --- full steps --------------------------------------------------------------------


def run_with_splats(n_steps, curl_strength):
    fluid.apply_seed(fluid.seed_pattern(fluid.N, rng_seed=3))
    for k in range(n_steps):
        if k % 10 == 0:
            fluid.splat(0.3 + 0.001 * k, 0.5, 4.0, 1.0, 1.0, 0.4, 0.1)
        fluid.step(curl_strength)


def test_determinism():
    run_with_splats(60, fluid.CURL_STRENGTH)
    first_d, first_v = fluid.dye.to_numpy(), fluid.vel.to_numpy()
    fluid.clear_fields()
    run_with_splats(60, fluid.CURL_STRENGTH)
    assert np.array_equal(fluid.dye.to_numpy(), first_d)
    assert np.array_equal(fluid.vel.to_numpy(), first_v)


def test_finite_and_bounded_after_200_steps():
    run_with_splats(200, fluid.CURL_STRENGTH)
    for arr in (fluid.dye.to_numpy(), fluid.vel.to_numpy(), fluid.pressure.to_numpy()):
        assert np.all(np.isfinite(arr))
    assert np.abs(fluid.vel.to_numpy()).max() < 50.0, "velocity should stay tame (decay + projection)"


def test_render_finite_unit_range():
    run_with_splats(30, fluid.CURL_STRENGTH)
    fluid.render()
    px = fluid.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0

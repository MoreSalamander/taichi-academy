import numpy as np

import gray_scott as gs

CORAL_F, CORAL_K = gs.PRESETS[0][1], gs.PRESETS[0][2]


def run_steps(n, feed=CORAL_F, kill=CORAL_K):
    for _ in range(n):
        gs.step(feed, kill)


# --- seed_pattern: pure numpy, no Taichi needed -------------------------------


def test_seed_pattern_shapes_and_center():
    u0, v0 = gs.seed_pattern(gs.N, rng_seed=0)
    assert u0.shape == v0.shape == (gs.N, gs.N)
    c = gs.N // 2
    assert v0[c, c] == 1.0
    assert u0[c, c] == 0.5
    assert v0[0, 0] == 0.0
    assert u0[0, 0] == 1.0


def test_seed_pattern_deterministic():
    a_u, a_v = gs.seed_pattern(gs.N, rng_seed=7)
    b_u, b_v = gs.seed_pattern(gs.N, rng_seed=7)
    assert np.array_equal(a_u, b_u)
    assert np.array_equal(a_v, b_v)
    c_u, c_v = gs.seed_pattern(gs.N, rng_seed=8)
    assert not np.array_equal(a_v, c_v)


# --- kernels ------------------------------------------------------------------


def test_flat_field_is_fixed_point_without_feed():
    """A uniform field has zero laplacian; with F=k=0 and no V, nothing moves."""
    gs.u.from_numpy(np.full((gs.N, gs.N), 0.7, dtype=np.float32))
    gs.v.from_numpy(np.zeros((gs.N, gs.N), dtype=np.float32))
    run_steps(5, feed=0.0, kill=0.0)
    assert np.allclose(gs.u.to_numpy(), 0.7, atol=1e-6)
    assert np.allclose(gs.v.to_numpy(), 0.0, atol=1e-6)


def test_determinism():
    gs.apply_seed(gs.seed_pattern(gs.N, rng_seed=3))
    run_steps(200)
    first_u, first_v = gs.u.to_numpy(), gs.v.to_numpy()
    gs.apply_seed(gs.seed_pattern(gs.N, rng_seed=3))
    run_steps(200)
    assert np.array_equal(gs.u.to_numpy(), first_u)
    assert np.array_equal(gs.v.to_numpy(), first_v)


def test_finite_and_bounded_after_500_steps():
    gs.apply_seed(gs.seed_pattern(gs.N, rng_seed=3))
    run_steps(500)
    for arr in (gs.u.to_numpy(), gs.v.to_numpy()):
        assert np.all(np.isfinite(arr))
        assert arr.min() >= -0.05
        assert arr.max() <= 1.5


def test_pattern_grows_and_stays_alive():
    """The V pattern must spread beyond the seed and not die out."""
    seed_u, seed_v = gs.seed_pattern(gs.N, rng_seed=3)
    gs.apply_seed((seed_u, seed_v))
    seed_area = int((seed_v > 0.1).sum())
    run_steps(400)
    v_now = gs.v.to_numpy()
    alive_area = int((v_now > 0.1).sum())
    assert v_now.max() > 0.1, "pattern died"
    assert alive_area > seed_area * 2, "pattern did not spread"


def test_splat_paints_v():
    gs.apply_seed((np.ones((gs.N, gs.N), np.float32), np.zeros((gs.N, gs.N), np.float32)))
    gs.splat(0.5, 0.5, 8.0)
    v_now = gs.v.to_numpy()
    c = gs.N // 2
    assert v_now[c, c] == 1.0
    assert v_now[10, 10] == 0.0


def test_render_finite_all_palettes():
    gs.apply_seed(gs.seed_pattern(gs.N, rng_seed=3))
    run_steps(50)
    for pal in range(len(gs.PALETTES)):
        gs.render(pal)
        px = gs.pixels.to_numpy()
        assert np.all(np.isfinite(px))
        assert px.min() >= 0.0
        assert px.max() <= 1.0

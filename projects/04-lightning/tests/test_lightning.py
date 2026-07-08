import numpy as np

import lightning as lt


# --- pure numpy: segments and bolts -----------------------------------------------


def test_deposit_segment_stamps_a_line():
    f = np.zeros((lt.N, lt.N), np.float32)
    lt.deposit_segment(f, np.array([100.0, 50.0]), np.array([100.0, 400.0]), 1.0)
    col = f[100, 50:400]
    assert col.min() == 1.0, "the segment column should be fully lit"
    assert f[300, 300] == 0.0, "far cells untouched"


def test_deposit_segment_keeps_max_brightness():
    f = np.zeros((lt.N, lt.N), np.float32)
    lt.deposit_segment(f, np.array([10.0, 10.0]), np.array([10.0, 100.0]), 1.0)
    lt.deposit_segment(f, np.array([10.0, 10.0]), np.array([10.0, 100.0]), 0.3)
    assert f[10, 50] == 1.0, "dimmer re-stamp must not darken"


def test_generate_bolt_deterministic_and_spans():
    a = lt.generate_bolt(lt.N, 0.5, rng_seed=7)
    b = lt.generate_bolt(lt.N, 0.5, rng_seed=7)
    c = lt.generate_bolt(lt.N, 0.5, rng_seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.max() <= 1.0 and a.min() >= 0.0
    assert a[:, lt.N - 5 :].max() == 1.0, "bolt starts at the top"
    assert a[:, :5].max() == 1.0, "bolt reaches the ground"


def test_generate_bolt_branches_add_dim_strands():
    """Branches deposit at reduced brightness — expect lit cells between 0 and full."""
    a = lt.generate_bolt(lt.N, 0.5, rng_seed=7)
    lit = a[a > 0]
    assert (lit < 0.9).any(), "expected dimmer branch pixels"
    assert (lit > 0.9).any(), "expected a bright trunk"


# --- GPU pipeline ------------------------------------------------------------------


def test_strike_absorbs_into_bolt_and_glow():
    lt.strike(0.5, rng_seed=7)
    b = lt.bolt.to_numpy()
    g = lt.glow.to_numpy()
    assert b.max() == 1.0
    assert g.max() > 0.0
    assert np.array_equal(b > 0, lt.generate_bolt(lt.N, 0.5, rng_seed=7) > 0)


def test_fade_decays_both_layers():
    lt.strike(0.5, rng_seed=7)
    for _ in range(60):
        lt.fade()
    assert lt.bolt.to_numpy().max() < 0.01
    assert lt.glow.to_numpy().max() < 0.2


def test_diffuse_spreads_and_conserves():
    lt.strike(0.5, rng_seed=7)
    before = lt.glow.to_numpy()
    for _ in range(10):
        lt.diffuse_glow()
        lt.copy_glow()
    after = lt.glow.to_numpy()
    assert after.max() < before.max(), "peak should soften"
    assert (after > 0.01).sum() > (before > 0.01).sum(), "halo should widen"
    assert abs(after.sum() - before.sum()) / before.sum() < 1e-3, "diffusion conserves total glow"


def test_full_loop_deterministic():
    for k in range(5):
        lt.strike(0.1 + 0.2 * k, rng_seed=k)
        for _ in range(10):
            lt.step()
    first_b, first_g = lt.bolt.to_numpy(), lt.glow.to_numpy()
    lt.clear_fields()
    for k in range(5):
        lt.strike(0.1 + 0.2 * k, rng_seed=k)
        for _ in range(10):
            lt.step()
    assert np.array_equal(lt.bolt.to_numpy(), first_b)
    assert np.array_equal(lt.glow.to_numpy(), first_g)


def test_finite_and_render_bounds():
    for k in range(8):
        lt.strike(np.random.default_rng(k).random(), rng_seed=k)
        for _ in range(15):
            lt.step()
    for arr in (lt.bolt.to_numpy(), lt.glow.to_numpy()):
        assert np.all(np.isfinite(arr))
    lt.render(0.7)
    px = lt.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0


def test_clear_fields():
    lt.strike(0.5, rng_seed=7)
    lt.clear_fields()
    assert lt.bolt.to_numpy().max() == 0.0
    assert lt.glow.to_numpy().max() == 0.0

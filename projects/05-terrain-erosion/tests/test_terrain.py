import numpy as np

import terrain as tr


def seed():
    tr.apply_seed(tr.fbm_terrain(tr.N, rng_seed=3))


def run(n):
    for _ in range(n):
        tr.step()


# --- pure numpy generation ---------------------------------------------------------


def test_resize_bilinear_endpoints_and_shape():
    a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    big = tr.resize_bilinear(a, 64)
    assert big.shape == (64, 64)
    assert abs(big[0, 0] - 0.0) < 1e-6 and abs(big[-1, -1] - 3.0) < 1e-6
    assert abs(big[0, -1] - 1.0) < 1e-6 and abs(big[-1, 0] - 2.0) < 1e-6


def test_fbm_terrain_range_and_determinism():
    a = tr.fbm_terrain(tr.N, rng_seed=5)
    b = tr.fbm_terrain(tr.N, rng_seed=5)
    c = tr.fbm_terrain(tr.N, rng_seed=6)
    assert a.shape == (tr.N, tr.N)
    assert a.min() == 0.0 and abs(a.max() - 1.0) < 1e-6
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fbm_has_multiscale_detail():
    a = tr.fbm_terrain(tr.N, rng_seed=5)
    coarse = np.abs(np.diff(a[::64, ::64], axis=0)).mean()
    fine = np.abs(np.diff(a, axis=0)).mean()
    assert coarse > 0.01, "large-scale relief exists"
    assert fine < coarse, "fine detail is fainter than coarse relief"


# --- conservation invariants ---------------------------------------------------------


def test_water_flow_conserves_water():
    """Flux moving (no rain/evap) must neither create nor destroy water."""
    seed()
    rng = np.random.default_rng(0)
    tr.w.from_numpy(rng.uniform(0.0, 0.01, size=(tr.N, tr.N)).astype(np.float32))
    before = tr.w.to_numpy().sum()
    for _ in range(20):
        tr.compute_flux()
        tr.move_water()
        # undo evaporation to isolate transport
        tr.w.from_numpy(tr.w_next.to_numpy() / (1.0 - tr.EVAP))
    after = tr.w.to_numpy().sum()
    assert abs(after - before) / before < 1e-3


def test_erosion_conserves_height_plus_sediment():
    seed()
    rng = np.random.default_rng(0)
    tr.w.from_numpy(rng.uniform(0.0, 0.01, size=(tr.N, tr.N)).astype(np.float32))
    total_before = tr.h.to_numpy().sum() + tr.s.to_numpy().sum()
    for _ in range(20):
        tr.compute_flux()
        tr.erode_deposit()
    total_after = tr.h.to_numpy().sum() + tr.s.to_numpy().sum()
    assert abs(total_after - total_before) / total_before < 1e-4


def test_thermal_conserves_and_relaxes():
    seed()
    before_sum = tr.h.to_numpy().sum()
    before_rough = np.abs(np.diff(tr.h.to_numpy(), axis=0)).mean()
    for _ in range(50):
        tr.thermal()
        tr.copy_height()
    after = tr.h.to_numpy()
    assert abs(after.sum() - before_sum) / before_sum < 1e-4, "talus slippage conserves rock"
    assert np.abs(np.diff(after, axis=0)).mean() < before_rough, "slopes relax"


# --- behavior ---------------------------------------------------------------------------


def test_water_pools_downhill():
    """After raining a while, low cells hold more water than high cells."""
    seed()
    run(150)
    hh = tr.h.to_numpy()
    ww = tr.w.to_numpy()
    low = ww[hh < np.quantile(hh, 0.2)].mean()
    high = ww[hh > np.quantile(hh, 0.8)].mean()
    assert low > high * 1.5, f"water should pool low: low={low:.5f} high={high:.5f}"


def test_erosion_carves():
    seed()
    before = tr.h.to_numpy()
    run(300)
    after = tr.h.to_numpy()
    assert not np.allclose(before, after, atol=1e-4), "terrain must change"
    assert np.all(after >= -1e-4), "height never goes negative"


def test_determinism():
    seed()
    run(60)
    first = tr.h.to_numpy()
    seed()
    tr.w.fill(0.0)
    tr.s.fill(0.0)
    run(60)
    assert np.array_equal(tr.h.to_numpy(), first)


def test_finite_and_render_bounds():
    seed()
    run(200)
    for arr in (tr.h.to_numpy(), tr.w.to_numpy(), tr.s.to_numpy()):
        assert np.all(np.isfinite(arr))
    tr.render()
    px = tr.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

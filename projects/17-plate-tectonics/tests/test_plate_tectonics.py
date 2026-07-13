import numpy as np

import plate_tectonics as pt


def run(n):
    for f in range(n):
        pt.step(f)


# --- pure numpy generation ---------------------------------------------------------


def test_voronoi_covers_every_cell_with_a_valid_plate():
    seeds = np.array([[10.0, 10.0], [200.0, 60.0], [100.0, 220.0]], dtype=np.float32)
    pid = pt.voronoi_plates(pt.N, seeds)
    assert pid.shape == (pt.N, pt.N)
    assert pid.min() >= 0 and pid.max() < 3
    assert len(np.unique(pid)) == 3, "every seed should own at least one cell"


def test_voronoi_is_toroidal():
    """A cell at the far right edge should belong to a seed at the far LEFT edge —
    wraparound distance, not euclidean."""
    seeds = np.array([[2.0, 128.0], [128.0, 128.0]], dtype=np.float32)
    pid = pt.voronoi_plates(pt.N, seeds)
    assert pid[pt.N - 2, 128] == 0, "the right edge wraps around to the left-edge seed"


def test_seed_world_determinism_and_layout():
    a_pid, a_h, a_v = pt.seed_world(rng_seed=3)
    b_pid, _b_h, _b_v = pt.seed_world(rng_seed=3)
    c_pid, _c_h, _c_v = pt.seed_world(rng_seed=4)
    assert np.array_equal(a_pid, b_pid)
    assert not np.array_equal(a_pid, c_pid)
    assert a_h.shape == (pt.N, pt.N)
    speeds = np.linalg.norm(a_v, axis=1)
    assert np.allclose(speeds, 1.0, atol=1e-5), "plate velocities are unit vectors"


# --- boundary physics -----------------------------------------------------------------


def test_convergent_boundary_uplifts():
    """Two plates aimed straight at each other: the boundary must rise."""
    pid = np.zeros((pt.N, pt.N), dtype=np.int32)
    pid[pt.N // 2 :, :] = 1
    pt.plate_id.from_numpy(pid)
    v = np.zeros((pt.N_PLATES, 2), dtype=np.float32)
    v[0] = [1.0, 0.0]   # plate 0 moves +x, toward plate 1
    v[1] = [-1.0, 0.0]  # plate 1 moves -x, toward plate 0
    pt.plate_vel.from_numpy(v)
    pt.height.fill(0.4)
    before = pt.height.to_numpy()[pt.N // 2 - 1, 50]
    for f in range(50):
        pt.boundary_forces()
        pt.erode()
    after = pt.height.to_numpy()[pt.N // 2 - 1, 50]
    assert after > before + 0.05, f"convergent boundary should uplift ({before:.3f} -> {after:.3f})"


def test_divergent_boundary_rifts():
    pid = np.zeros((pt.N, pt.N), dtype=np.int32)
    pid[pt.N // 2 :, :] = 1
    pt.plate_id.from_numpy(pid)
    v = np.zeros((pt.N_PLATES, 2), dtype=np.float32)
    v[0] = [-1.0, 0.0]  # plate 0 pulls away
    v[1] = [1.0, 0.0]   # plate 1 pulls away
    pt.plate_vel.from_numpy(v)
    pt.height.fill(0.4)
    before = pt.height.to_numpy()[pt.N // 2 - 1, 50]
    for f in range(50):
        pt.boundary_forces()
        pt.erode()
    after = pt.height.to_numpy()[pt.N // 2 - 1, 50]
    assert after < before - 0.05, f"divergent boundary should rift ({before:.3f} -> {after:.3f})"


def test_plate_interiors_stay_flat_without_drift():
    """Boundary forces only (no drift), on a UNIFORM height field so that boundary
    uplift/rift is the only possible source of change — far interiors must stay calm.
    (The real seed's continent/ocean height steps would also diffuse inward under
    erosion, and with drift enabled interiors legitimately change; both would
    confound what this test isolates.)"""
    pt.height.fill(0.4)
    pid = pt.plate_id.to_numpy()
    for _ in range(80):
        pt.boundary_forces()
        pt.erode()
    h1 = pt.height.to_numpy()
    interior = np.ones_like(pid, dtype=bool)
    for shift in range(1, 6):  # stay 5+ cells from any boundary (erosion spreads a little)
        for di, dj in ((shift, 0), (-shift, 0), (0, shift), (0, -shift)):
            interior &= np.roll(pid, (di, dj), axis=(0, 1)) == pid
    assert np.abs(h1[interior] - 0.4).max() < 0.05, "far interiors stay calm"
    assert np.abs(h1 - 0.4).max() > 0.1, "…while the boundaries themselves moved"


def test_height_stays_clamped():
    run(300)
    h = pt.height.to_numpy()
    assert np.all(np.isfinite(h))
    assert h.min() >= 0.0 and h.max() <= 1.0


# --- drift ---------------------------------------------------------------------------


def test_drift_moves_the_plate_map():
    before = pt.plate_id.to_numpy().copy()
    pt.drift()
    pt.copy_drift()
    after = pt.plate_id.to_numpy()
    assert not np.array_equal(before, after), "unit-speed plates must shift the map by one cell"
    changed = (before != after).mean()
    assert changed < 0.5, "a one-cell shift should change only cells near boundaries"


def test_drift_conserves_plate_populations_roughly():
    before = np.bincount(pt.plate_id.to_numpy().ravel(), minlength=pt.N_PLATES)
    for _ in range(10):
        pt.drift()
        pt.copy_drift()
    after = np.bincount(pt.plate_id.to_numpy().ravel(), minlength=pt.N_PLATES)
    # gather-style drift lets fast plates grow at trailing edges, but not wildly
    assert np.abs(after - before).max() < pt.N * pt.N * 0.2


# --- activity / whole-sim ---------------------------------------------------------------


def test_quakes_flash_and_decay():
    pt.activity.fill(1.0)
    pt.decay_activity()
    assert np.allclose(pt.activity.to_numpy(), pt.ACTIVITY_DECAY, atol=1e-5)


def test_render_is_finite_and_bounded():
    run(50)
    pt.render()
    px = pt.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0


def test_apply_seed_resets_activity():
    run(100)
    pt.apply_seed(rng_seed=8)
    assert pt.activity.to_numpy().sum() == 0.0

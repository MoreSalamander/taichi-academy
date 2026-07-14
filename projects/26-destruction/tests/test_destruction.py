import numpy as np

import destruction as dz


def settle(n=60):
    for _ in range(n):
        dz.step()


# --- pure-numpy geometry -------------------------------------------------------------


def test_structures_fit_in_the_box():
    positions, edges = dz.build_structures()
    assert positions.shape[1] == 2
    assert positions.min() >= 0.0 and positions.max() <= 1.0
    assert positions[:, 1].min() >= dz.FLOOR
    # every listed building contributes its full block of cells
    expected = sum(w * h for (_, w, h) in dz.BUILDINGS)
    assert len(positions) == expected


def test_bonds_reference_real_particles_and_rest_on_the_lattice():
    positions, edges = dz.build_structures()
    assert edges.min() >= 0 and edges.max() < len(positions)
    rests = dz.bond_rest_lengths(positions, edges)
    # neighbours are one step (SPACING) or one diagonal (sqrt2 * SPACING) apart
    diag = np.sqrt(2) * dz.SPACING
    assert np.all((np.isclose(rests, dz.SPACING, atol=1e-4)) | (np.isclose(rests, diag, atol=1e-4)))


def test_seed_uploads_every_bond_intact():
    assert dz.n_b[None] > 0
    assert dz.b_broken.to_numpy()[: dz.n_b[None]].sum() == 0


# --- the solver keeps a building standing ---------------------------------------------


def test_buildings_stand_under_gravity():
    """The braced lattice barely sags: after settling, the tallest particle is still near
    the top of the tallest building, not collapsed into a pile."""
    top0 = dz.pos.to_numpy()[: dz.n_p[None], 1].max()
    settle(120)
    top1 = dz.pos.to_numpy()[: dz.n_p[None], 1].max()
    assert top1 > 0.9 * top0, f"structure pancaked: {top0:.3f} -> {top1:.3f}"
    assert dz.broken_fraction() == 0.0, "nothing should break just from standing"


def test_everything_stays_above_the_floor():
    settle(120)
    p = dz.pos.to_numpy()[: dz.n_p[None]]
    assert np.all(np.isfinite(p))
    assert p[:, 1].min() >= dz.FLOOR
    assert p[:, 0].min() >= 0.0 and p[:, 0].max() <= 1.0


# --- spatial hash & self-collision ----------------------------------------------------


def test_grid_buckets_every_particle_once():
    dz.build_grid()
    assert dz.cell_count.to_numpy().sum() == dz.n_p[None]
    assert sorted(dz.sorted_idx.to_numpy()[: dz.n_p[None]].tolist()) == list(range(dz.n_p[None]))


def test_self_collision_separates_overlapping_particles():
    """Two free particles dropped almost on top of each other are pushed to >= 2R apart."""
    p = dz.pos.to_numpy()
    p[0] = [0.5, 0.5]
    p[1] = [0.5 + 0.5 * dz.RADIUS, 0.5]  # deep overlap
    dz.pos.from_numpy(p)
    dz.prev.from_numpy(p)
    dz.b_broken.fill(1)  # ignore bonds; isolate collision response
    for _ in range(20):
        dz.step()
    q = dz.pos.to_numpy()
    assert np.linalg.norm(q[1] - q[0]) > 1.9 * dz.RADIUS


# --- fracture --------------------------------------------------------------------------


def test_bond_snaps_past_the_breaking_strain():
    """Yank one bond's endpoints apart beyond BREAK_STRAIN and it registers as broken."""
    a, b = int(dz.b_a[0]), int(dz.b_b[0])
    rest = float(dz.b_rest[0])
    p = dz.pos.to_numpy()
    p[b] = p[a] + np.array([rest * (dz.BREAK_STRAIN + 0.3), 0.0], dtype=np.float32)
    dz.pos.from_numpy(p)
    dz.prev.from_numpy(p)
    dz.break_bonds()
    assert dz.b_broken.to_numpy()[0] == 1


def test_bond_survives_below_the_breaking_strain():
    a, b = int(dz.b_a[0]), int(dz.b_b[0])
    rest = float(dz.b_rest[0])
    p = dz.pos.to_numpy()
    p[b] = p[a] + np.array([rest * (dz.BREAK_STRAIN - 0.2), 0.0], dtype=np.float32)
    dz.pos.from_numpy(p)
    dz.prev.from_numpy(p)
    dz.break_bonds()
    assert dz.b_broken.to_numpy()[0] == 0


def test_broken_bonds_stay_broken():
    dz.b_broken.fill(0)
    b = dz.b_broken.to_numpy()
    b[3] = 1
    dz.b_broken.from_numpy(b)
    dz.break_bonds()  # bond 3 is not stretched, but must not "heal"
    assert dz.b_broken.to_numpy()[3] == 1


# --- the headline: explosion & quake fracture the city --------------------------------


def test_explosion_snaps_bonds_and_launches_debris():
    settle(60)
    before = dz.broken_fraction()
    ytop_before = dz.pos.to_numpy()[: dz.n_p[None], 1].max()
    dz.explode(0.53, 0.12, dz.EXPLODE_POWER, dz.EXPLODE_RADIUS)
    for _ in range(30):
        dz.step()
    after = dz.broken_fraction()
    ytop_after = dz.pos.to_numpy()[: dz.n_p[None], 1].max()
    assert after > before + 0.05, f"the blast should snap many bonds: {before:.2f} -> {after:.2f}"
    assert ytop_after > ytop_before, "debris is thrown up above the original skyline"
    assert np.all(np.isfinite(dz.pos.to_numpy()[: dz.n_p[None]]))


def test_explosion_only_breaks_bonds_near_the_blast():
    settle(60)
    # locality by original geometry: a bond's home is where it sat before the blast
    p0 = dz.pos.to_numpy()
    a = dz.b_a.to_numpy()[: dz.n_b[None]]
    b = dz.b_b.to_numpy()[: dz.n_b[None]]
    mids = 0.5 * (p0[a] + p0[b])
    far = np.linalg.norm(mids - np.array([0.53, 0.12]), axis=1) > 0.35
    dz.explode(0.53, 0.12, dz.EXPLODE_POWER, dz.EXPLODE_RADIUS)
    dz.step()  # the blast impulse becomes the only motion this step
    broken = dz.b_broken.to_numpy()[: dz.n_b[None]]
    assert broken[far].sum() == 0, "bonds far from the blast should be untouched"


def test_quake_shears_the_city_apart():
    settle(60)
    before = dz.broken_fraction()
    for f in range(240):
        dz.quake(f * dz.DT, dz.QUAKE_AMP)
        dz.step()
    after = dz.broken_fraction()
    assert after > before + 0.02, f"the quake should shear bonds loose: {before:.2f} -> {after:.2f}"
    assert np.all(np.isfinite(dz.pos.to_numpy()[: dz.n_p[None]]))


# --- render ---------------------------------------------------------------------------


def test_render_is_finite_and_bounded():
    settle(30)
    dz.render()
    px = dz.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0

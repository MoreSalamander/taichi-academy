import numpy as np

import tornado as tn


def run(n):
    for _ in range(n):
        tn.step()


# --- pure numpy generation ---------------------------------------------------------


def test_seed_debris_ring_bounds_and_determinism():
    a = tn.seed_debris(rng_seed=2)
    b = tn.seed_debris(rng_seed=2)
    c = tn.seed_debris(rng_seed=3)
    assert a.shape == (tn.N_DEBRIS, 2)
    r = np.linalg.norm(a - [tn.CX, tn.CY], axis=1)
    assert np.all(r >= tn.CORE_R * 1.2 - 1e-3)
    assert np.all(r <= tn.N * 0.45 + 1e-3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- seeding -------------------------------------------------------------------------


def test_apply_seed_clears_fields_and_places_debris():
    tn.vel.fill([1.0, 1.0])
    tn.dye.fill(1.0)
    tn.apply_seed(rng_seed=4)
    assert np.all(tn.vel.to_numpy() == 0.0)
    assert np.all(tn.dye.to_numpy() == 0.0)
    assert np.all(tn.dvel.to_numpy() == 0.0)
    r = np.linalg.norm(tn.dpos.to_numpy() - [tn.CX, tn.CY], axis=1)
    assert np.all(r > 0)


# --- sampling / interpolation --------------------------------------------------------


def test_bilerp_of_a_constant_field_is_that_constant():
    tn.dye.fill(0.5)
    tn.dye[10, 10] = [0.5, 0.5, 0.5]  # keep it a genuinely constant field
    import taichi as ti

    @ti.kernel
    def probe(x: ti.f32, y: ti.f32) -> ti.math.vec3:
        return tn.bilerp(tn.dye, x, y)

    val = probe(123.4, 77.8)
    assert np.allclose(np.array(val), 0.5)


# --- physics -----------------------------------------------------------------------


def test_vortex_forcing_spins_the_fluid_tangentially():
    tn.vortex_forcing()
    v = tn.vel.to_numpy()
    i, j = int(tn.CX + tn.CORE_R), int(tn.CY)
    rx, ry = i - tn.CX, j - tn.CY
    radial = np.array([rx, ry]) / np.linalg.norm([rx, ry])
    vel_here = v[i, j]
    radial_component = np.dot(vel_here, radial)
    assert abs(radial_component) < np.linalg.norm(vel_here), "mostly tangential, not radial, at the core edge"
    assert np.linalg.norm(vel_here) > 0


def test_project_makes_the_flow_nearly_divergence_free():
    for _ in range(10):
        tn.step()
    tn.compute_divergence()
    div = tn.divergence.to_numpy()
    assert np.abs(div).mean() < 0.1, "the Jacobi pressure solve should nearly cancel divergence"


def test_decay_shrinks_dye_and_velocity():
    tn.dye.fill(1.0)
    tn.vel.fill([2.0, 2.0])
    tn.decay()
    assert np.allclose(tn.dye.to_numpy(), tn.DYE_DECAY, atol=1e-5)
    assert np.allclose(tn.vel.to_numpy(), 2.0 * tn.VEL_DECAY, atol=1e-5)


def test_stir_adds_force_near_the_cursor_only():
    before = tn.vel.to_numpy().copy()
    tn.stir(0.5, 0.5, 50.0, -30.0)
    after = tn.vel.to_numpy()
    near = after[int(tn.N * 0.5), int(tn.N * 0.5)]
    far = after[5, 5]
    assert np.linalg.norm(near - before[int(tn.N * 0.5), int(tn.N * 0.5)]) > 1.0
    assert np.allclose(far, before[5, 5], atol=1e-3)


# --- debris --------------------------------------------------------------------------


def test_debris_drag_pulls_toward_local_fluid_velocity():
    """Place debris well inside DEBRIS_HOME_R so the containment pull (tested
    separately below) can't confound this — isolate the drag term alone."""
    p = tn.dpos.to_numpy()
    p[:] = [tn.CX, tn.CY]
    tn.dpos.from_numpy(p)
    tn.vel.fill([3.0, 0.0])
    tn.dvel.fill(0.0)
    tn.advect_debris()
    dv = tn.dvel.to_numpy()
    assert np.all(dv[:, 0] > 0), "drag should pull debris velocity toward the fluid's"


def test_debris_beyond_home_radius_gets_pulled_back():
    p = tn.dpos.to_numpy()
    p[0] = [tn.CX + tn.DEBRIS_HOME_R + 50.0, tn.CY]
    tn.dpos.from_numpy(p)
    tn.dvel.fill(0.0)
    tn.vel.fill(0.0)
    tn.advect_debris()
    dv = tn.dvel[0]
    assert dv[0] < 0, "too far out on the +x side, the home pull should push back toward -x"


# --- whole-sim behavior ---------------------------------------------------------------


def test_finite_and_bounded_after_many_frames():
    run(150)
    v = tn.vel.to_numpy()
    d = tn.dye.to_numpy()
    p = tn.dpos.to_numpy()
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(d)) and np.all(np.isfinite(p))
    assert np.all(p >= -1e-3) and np.all(p <= tn.N + 1e-3)
    assert d.min() >= 0.0 and d.max() <= 1.0 + 1e-3


def test_determinism_is_bit_exact():
    """Every kernel here writes only to its own loop index (gather-style neighbor
    reads via sample(), never a scatter into another cell or particle's slot) —
    no atomics anywhere, so this is bit-exact, unlike projects 06/08/09."""
    tn.apply_seed(7)
    run(40)
    a = tn.vel.to_numpy()
    ad = tn.dpos.to_numpy()
    tn.apply_seed(7)
    run(40)
    b = tn.vel.to_numpy()
    bd = tn.dpos.to_numpy()
    assert np.array_equal(a, b)
    assert np.array_equal(ad, bd)


def test_debris_stay_roughly_ring_shaped_around_the_core():
    run(150)
    p = tn.dpos.to_numpy()
    r = np.linalg.norm(p - [tn.CX, tn.CY], axis=1)
    assert r.std() < tn.DEBRIS_HOME_R, "debris should cluster near a characteristic radius, not scatter everywhere"

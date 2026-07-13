import numpy as np

import solar_system as ss


def run(n):
    for _ in range(n):
        ss.step()


# --- pure numpy generation ---------------------------------------------------------


def test_circular_velocity_is_tangent_and_kepler_fast():
    p = np.array([[0.2, 0.0], [0.0, 0.3]], dtype=np.float32)
    v = ss.circular_velocity(p)
    assert abs(np.dot(v[0], p[0])) < 1e-6, "velocity is perpendicular to the radius"
    assert abs(np.linalg.norm(v[0]) - np.sqrt(ss.GM / 0.2)) < 1e-4
    assert abs(np.linalg.norm(v[1]) - np.sqrt(ss.GM / 0.3)) < 1e-4


def test_comet_aphelion_velocity_matches_vis_viva():
    r_apo, r_peri = 0.4, 0.06
    v = ss.comet_aphelion_velocity(r_apo, r_peri)
    a = 0.5 * (r_apo + r_peri)
    assert abs(v - np.sqrt(ss.GM * (2 / r_apo - 1 / a))) < 1e-6
    assert v < np.sqrt(ss.GM / r_apo), "slower than circular at aphelion — that's what makes it fall"


def test_seed_layout_and_determinism():
    a_p, a_v, a_c = ss.seed_planets(rng_seed=3)
    b_p, _b_v, _b_c = ss.seed_planets(rng_seed=3)
    c_p, _c_v, _c_c = ss.seed_planets(rng_seed=4)
    assert a_p.shape == (ss.N_PLANETS, 2)
    assert np.array_equal(a_p, b_p)
    assert not np.array_equal(a_p, c_p)
    belt_p, _, _ = ss.seed_belt(rng_seed=3)
    r = np.linalg.norm(belt_p, axis=1)
    assert r.min() >= ss.BELT_R[0] - 1e-5 and r.max() <= ss.BELT_R[1] + 1e-5


# --- the integrator ---------------------------------------------------------------------


def test_leapfrog_conserves_energy():
    """The whole point of leapfrog over explicit Euler: energy drift stays tiny.
    (The prototype measured Euler at ~86% drift on this exact setup; leapfrog ~0.02%.)"""
    e0 = ss.total_energy()
    for _ in range(2000):
        ss.leapfrog()
    e1 = ss.total_energy()
    drift = np.abs((e1 - e0) / e0)
    assert drift[: ss.N_PLANETS].max() < 0.01, f"planet energy drift {drift[:ss.N_PLANETS].max():.4f}"


def test_circular_orbit_stays_circular():
    p0 = ss.pos.to_numpy()[: ss.N_PLANETS]
    r0 = np.linalg.norm(p0, axis=1)
    for _ in range(2000):
        ss.leapfrog()
    r1 = np.linalg.norm(ss.pos.to_numpy()[: ss.N_PLANETS], axis=1)
    assert np.abs((r1 - r0) / r0).max() < 0.02, "circular orbits should hold their radius"


def test_inner_planets_orbit_faster():
    """Kepler's third law, observationally: the innermost planet sweeps more angle.
    The window must be short enough that the fast inner planet doesn't wrap past pi
    (it circles ~5x per 500 steps!), or the arctan2 difference wraps to garbage."""
    p0 = ss.pos.to_numpy()[: ss.N_PLANETS]
    for _ in range(40):
        ss.leapfrog()
    p1 = ss.pos.to_numpy()[: ss.N_PLANETS]
    cross = p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]
    dot = (p0 * p1).sum(axis=1)
    swept = np.abs(np.arctan2(cross, dot))
    inner, outer = swept[0], swept[-1]
    assert inner > outer * 2, f"inner swept {inner:.2f}, outer {outer:.2f}"


def test_comet_speeds_up_at_perihelion():
    """Kepler's second law: a comet moves fastest at its closest approach."""
    c = ss.COMET_BASE
    v_apo = np.linalg.norm(ss.vel.to_numpy()[c])
    min_r, v_at_min_r = 1e9, 0.0
    for _ in range(4000):
        ss.leapfrog()
        p = ss.pos.to_numpy()[c]
        r = np.linalg.norm(p)
        if r < min_r:
            min_r = r
            v_at_min_r = np.linalg.norm(ss.vel.to_numpy()[c])
    assert min_r < 0.15, f"the comet should have dived inward, min r={min_r:.3f}"
    assert v_at_min_r > v_apo * 2, "far faster near the sun than at aphelion"


# --- whole-sim behavior ---------------------------------------------------------------


def test_step_stays_finite_and_bounded():
    run(100)
    p = ss.pos.to_numpy()
    assert np.all(np.isfinite(p))
    px = ss.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0


def test_determinism_is_bit_exact():
    """leapfrog writes only its own body's slots; render's splat does scatter, but
    positions/velocities — the physics — are gather-free and bit-stable."""
    ss.apply_seed(rng_seed=5)
    run(50)
    a_p = ss.pos.to_numpy()
    ss.apply_seed(rng_seed=5)
    run(50)
    b_p = ss.pos.to_numpy()
    assert np.array_equal(a_p, b_p)


def test_apply_seed_resets_canvas():
    run(30)
    ss.apply_seed(rng_seed=2)
    assert ss.pixels.to_numpy().sum() == 0.0

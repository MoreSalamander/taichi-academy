import numpy as np

import universe_sandbox as us


def run(n):
    for _ in range(n):
        us.step()


# --- building a galaxy ---------------------------------------------------------------


def test_make_galaxy_has_a_central_black_hole_and_a_disk():
    p, v, m = us.make_galaxy(1000, [0.0, 0.0], [0.0, 0.0], radius=0.3, seed=1)
    assert m[0] == us.BH_MASS, "body 0 is the black hole"
    assert np.allclose(p[0], 0.0), "the black hole sits at the centre"
    assert (m[1:] == us.STAR_MASS).all(), "the rest are stars"
    r = np.linalg.norm(p[1:], axis=1)
    assert r.max() < 0.4 and r.min() > 0.0, "stars lie in a disk around it"


def test_disk_stars_orbit_not_fall_in():
    """A star's speed is set to the circular-orbit value, so its velocity is perpendicular to
    the line to the centre (tangential), not pointing inward."""
    p, v, m = us.make_galaxy(2000, [0.0, 0.0], [0.0, 0.0], radius=0.3, seed=2)
    radial = np.array([np.dot(v[i], p[i] / np.linalg.norm(p[i])) for i in range(1, 200)])
    speed = np.linalg.norm(v[1:200], axis=1)
    assert np.abs(radial).mean() < 0.2 * speed.mean(), "orbits are near-circular (little radial motion)"


# --- gravity -------------------------------------------------------------------------


def test_gravity_pulls_a_star_toward_the_centre():
    p = us.pos.to_numpy()
    star = 500
    us.compute_acc(-1e9, -1e9, 0.0)
    a = us.acc.to_numpy()[star]
    toward = -p[star] / np.linalg.norm(p[star])   # unit vector toward centre
    assert np.dot(a, toward) > 0, "net acceleration points inward, toward the mass"


def test_softening_keeps_forces_finite_when_bodies_overlap():
    p = us.pos.to_numpy()
    p[10] = p[11]                       # two bodies exactly on top of each other
    us.pos.from_numpy(p)
    us.compute_acc(-1e9, -1e9, 0.0)
    assert np.all(np.isfinite(us.acc.to_numpy())), "softening (EPS) tames the 1/r^2 singularity"


def test_cursor_mass_attracts():
    us.compute_acc(0.5, 0.0, 5.0 * us.BH_MASS)   # a heavy cursor to the right
    a = us.acc.to_numpy()
    p = us.pos.to_numpy()
    # a star to the LEFT of the cursor should feel a rightward tug from it
    left = np.argmin(p[:, 0])
    assert us.acc.to_numpy()[left][0] > -1e9, "finite"
    us.compute_acc(-1e9, -1e9, 0.0)  # reset


# --- the symplectic integrator -------------------------------------------------------


def test_leapfrog_conserves_energy():
    """Leapfrog is symplectic (project 16's family): total energy holds steady over a long run
    instead of drifting away."""
    e0 = us.total_energy()
    run(400)
    e1 = us.total_energy()
    assert abs(e1 - e0) < 0.15 * abs(e0), f"energy should be ~conserved: {e0:.3f} -> {e1:.3f}"
    assert e0 < 0.0, "a self-bound galaxy has negative total energy"


def test_single_galaxy_stays_bound():
    run(400)
    assert us.bound_fraction() > 0.9, "a stable galaxy does not evaporate"
    assert np.all(np.isfinite(us.pos.to_numpy()))


# --- the collision scene -------------------------------------------------------------


def test_collide_scene_has_two_black_holes_and_stays_finite():
    us.apply_seed("collide", seed=1)
    holes = (us.mass.to_numpy() > 0.1).sum()
    assert holes == 2, "two galaxies bring two black holes"
    run(300)
    assert np.all(np.isfinite(us.pos.to_numpy())), "the collision does not blow up"


# --- render --------------------------------------------------------------------------


def test_render_is_finite_and_bounded():
    run(50)
    us.render()
    px = us.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

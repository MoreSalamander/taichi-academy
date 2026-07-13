import numpy as np

import soft_body as sb


def run(n):
    for _ in range(n):
        sb.substep()


def shoelace_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def body(pos, b):
    return pos[b * sb.N_RING : (b + 1) * sb.N_RING]


# --- pure numpy generation ---------------------------------------------------------


def test_seed_ring_shape_and_radius():
    ring = sb.seed_ring(0.5, 0.5, 0.1, 20)
    assert ring.shape == (20, 2)
    dist = np.linalg.norm(ring - np.array([0.5, 0.5]), axis=1)
    assert np.allclose(dist, 0.1, atol=1e-5)


def test_rest_lengths_match_perimeter_segments():
    ring = sb.seed_ring(0.5, 0.5, 0.1, 12)
    rl = sb.rest_lengths(ring)
    expected = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    assert np.allclose(rl, expected)
    assert np.allclose(rl, rl[0], atol=1e-5), "a regular polygon has equal-length edges"


# --- seeding -----------------------------------------------------------------------


def test_apply_seed_assigns_bodies_and_materials():
    bid = sb.body_id.to_numpy()
    assert np.all(bid[: sb.N_RING] == 0)
    assert np.all(bid[sb.N_RING : 2 * sb.N_RING] == 1)
    assert np.array_equal(sb.stiffness.to_numpy(), sb.STIFFNESS_NP)
    assert sb.grabbed[None] == -1


# --- forces --------------------------------------------------------------------------


def test_gravity_scales_with_mass():
    sb.compute_forces()
    f = sb.force.to_numpy()
    m = sb.mass.to_numpy()
    # a particle at rest length, zero velocity: spring/pressure roughly cancel at t=0
    # for a regular polygon, but gravity's contribution is exactly -GRAVITY*mass
    assert f[0, 1] < 0, "gravity pulls every particle down"
    assert abs(f[0, 1] - f[sb.N_RING, 1]) < 100, "same mass -> comparable gravity term"
    assert m[0] == sb.MASS_NP[0]


def test_stretched_spring_pulls_together():
    p = sb.pos.to_numpy()
    p[1] += [0.05, 0.0]  # stretch the spring between particle 0 and particle 1
    sb.pos.from_numpy(p)
    sb.vel.fill(0.0)
    sb.compute_forces()
    f = sb.force.to_numpy()
    # particle 0 should be pulled toward particle 1 (positive x), and vice versa
    assert f[0, 0] > 0
    assert f[1, 0] < 0


def test_pressure_pushes_outward_when_compressed():
    """Shrink a ring toward its centroid; pressure should push each vertex back out."""
    p = sb.pos.to_numpy()
    base = 2 * sb.N_RING  # balloon body: has real gas pressure
    ring = p[base : base + sb.N_RING]
    centroid = ring.mean(axis=0)
    p[base : base + sb.N_RING] = centroid + (ring - centroid) * 0.5
    sb.pos.from_numpy(p)
    sb.vel.fill(0.0)
    sb.compute_forces()
    f = sb.force.to_numpy()[base : base + sb.N_RING]
    outward = ring - centroid
    # average alignment between force and the (pre-shrink) outward direction
    align = (f * outward).sum(axis=1)
    assert align.mean() > 0, "pressure should push a compressed ring back outward"


def test_floor_bounce_and_friction():
    """Isolate integrate()'s wall/floor logic from compute_forces() entirely — moving
    a single ring particle away from its neighbors would otherwise create a huge,
    unrelated spring force from the rest of its (untouched) ring."""
    p = sb.pos.to_numpy()
    p[0] = [0.5, 0.001]
    sb.pos.from_numpy(p)
    v = sb.vel.to_numpy()
    v[0] = [1.0, -5.0]
    sb.vel.from_numpy(v)
    sb.force.fill(0.0)
    sb.integrate()
    assert sb.pos[0][1] == 0.0
    assert sb.vel[0][1] > 0.0, "downward velocity flips sign on floor contact"
    assert abs(sb.vel[0][0] - 0.7) < 1e-5, "horizontal velocity damped by floor friction"


# --- whole-sim behavior ---------------------------------------------------------------


def test_finite_and_bounded_after_falling():
    run(150)
    p = sb.pos.to_numpy()
    v = sb.vel.to_numpy()
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(v))
    assert np.all(p[:, 1] >= -1e-3) and np.all(p[:, 0] >= -1e-3) and np.all(p[:, 0] <= sb.WORLD + 1e-3)


def test_determinism():
    """Same seed replays the same fall — not bit-exact (the spring pass scatters
    force via += on shared indices, so summation order can vary), but matching
    to float precision."""
    sb.apply_seed()
    run(60)
    a = sb.pos.to_numpy()
    sb.apply_seed()
    run(60)
    b = sb.pos.to_numpy()
    assert np.allclose(a, b, atol=1e-3)


def test_bodies_keep_a_real_shape_not_a_collapsed_line():
    """Regression check: a ring with only perimeter springs and no pressure force
    can fold flat under gravity without stretching any single spring much — this
    is exactly why every body needs SOME internal pressure, not just jelly/rubber."""
    run(300)
    pos = sb.pos.to_numpy()
    for b in range(sb.N_BODIES):
        area = shoelace_area(body(pos, b))
        assert area > 0.3 * (np.pi * sb.BODY_RADIUS**2), f"body {b} collapsed nearly flat"


def test_balloon_ends_up_puffier_than_jelly_and_rubber():
    run(300)
    pos = sb.pos.to_numpy()
    areas = [shoelace_area(body(pos, b)) for b in range(sb.N_BODIES)]
    jelly, rubber, balloon = areas
    assert balloon > jelly > rubber, f"expected balloon > jelly > rubber, got {areas}"


def test_grab_pulls_the_nearest_particle_toward_the_target():
    run(100)
    pos = sb.pos.to_numpy()
    target_particle = body(pos, 0)[0]  # click ON the ring, not its empty centroid
    sb.grab_at(float(target_particle[0]), float(target_particle[1]))
    assert sb.grabbed[None] >= 0
    sb.grab_target[None] = [0.9, 0.9]
    for _ in range(80):
        sb.substep()
    p = sb.pos.to_numpy()
    v = sb.vel.to_numpy()
    grabbed_pos = p[sb.grabbed[None]]
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(v))
    dist_to_target = np.linalg.norm(grabbed_pos - np.array([0.9, 0.9]))
    assert dist_to_target < 0.3, "the grabbed particle should be pulled toward the target"


def test_grab_at_misses_returns_no_particle():
    sb.grab_at(0.5, 0.99)  # nothing seeded way up there
    assert sb.grabbed[None] == -1

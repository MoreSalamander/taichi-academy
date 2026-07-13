import numpy as np

import cloth_rope as cr


def run(n, wind=0.0):
    for i in range(n):
        cr.substep(i * cr.DT, wind=wind)


# --- pure numpy generation ---------------------------------------------------------


def test_build_rope_shape_and_link_length():
    pts, edges = cr.build_rope()
    assert pts.shape == (cr.ROPE_N, 2)
    assert len(edges) == cr.ROPE_N - 1
    link = edges[0][2]
    assert link > 0
    for a, b, rest in edges:
        assert abs(rest - link) < 1e-6, "every rope link starts at the same rest length"


def test_build_cloth_shape_and_edge_count():
    pts, edges = cr.build_cloth()
    assert pts.shape == (cr.CLOTH_N, 2)
    structural = (cr.CLOTH_W - 1) * cr.CLOTH_H + cr.CLOTH_W * (cr.CLOTH_H - 1)
    shear = 2 * (cr.CLOTH_W - 1) * (cr.CLOTH_H - 1)
    assert len(edges) == structural + shear
    rests = sorted({round(e[2], 6) for e in edges})
    assert len(rests) == 2, "exactly two distinct rest lengths: structural spacing and its diagonal"


# --- seeding -------------------------------------------------------------------------


def test_apply_seed_pins_the_right_particles():
    im = cr.inv_mass.to_numpy()
    assert im[cr.ROPE_BASE] == 0.0
    assert np.all(im[cr.ROPE_BASE + 1 : cr.ROPE_N] == 1.0)
    for j in range(cr.CLOTH_H):
        assert im[cr.idx_cloth(0, j)] == 0.0
    for j in range(cr.CLOTH_H):
        assert im[cr.idx_cloth(1, j)] == 1.0
    assert cr.n_constraints[None] == len(cr.build_rope()[1]) + len(cr.build_cloth()[1])
    assert cr.grabbed[None] == -1


# --- forces / constraints -----------------------------------------------------------


def test_predict_moves_free_particles_down_and_leaves_pinned_alone():
    before = cr.pos.to_numpy()
    cr.predict(0.0, 0.0)
    after = cr.pos.to_numpy()
    assert after[cr.ROPE_BASE, 1] == before[cr.ROPE_BASE, 1], "pinned rope root never moves"
    assert after[cr.ROPE_BASE + 5, 1] < before[cr.ROPE_BASE + 5, 1], "gravity pulls a free link down"


def test_solve_constraints_pulls_a_stretched_link_together():
    p = cr.pos.to_numpy()
    p[cr.ROPE_BASE + 6] = p[cr.ROPE_BASE + 5] + [0.05, 0.0]  # stretch one rope link
    cr.pos.from_numpy(p)
    before_dist = np.linalg.norm(p[cr.ROPE_BASE + 6] - p[cr.ROPE_BASE + 5])
    cr.solve_constraints()
    after = cr.pos.to_numpy()
    after_dist = np.linalg.norm(after[cr.ROPE_BASE + 6] - after[cr.ROPE_BASE + 5])
    assert after_dist < before_dist


def test_pinned_particles_never_move_via_constraints():
    p = cr.pos.to_numpy()
    p[cr.ROPE_BASE + 1] += [0.2, 0.2]  # yank the pinned root's neighbor far away
    cr.pos.from_numpy(p)
    before_root = cr.pos.to_numpy()[cr.ROPE_BASE]
    for _ in range(cr.ITERS):
        cr.solve_constraints()
    after_root = cr.pos.to_numpy()[cr.ROPE_BASE]
    assert np.array_equal(before_root, after_root), "inv_mass=0 means the constraint can never move it"


def test_apply_bounds_clamps_floor_and_walls():
    p = cr.pos.to_numpy()
    p[cr.ROPE_BASE + 1] = [-0.5, -0.5]
    cr.pos.from_numpy(p)
    cr.apply_bounds()
    clamped = cr.pos[cr.ROPE_BASE + 1]
    assert clamped[0] == 0.0
    assert clamped[1] == 0.0


# --- whole-sim behavior ---------------------------------------------------------------


def test_finite_and_bounded_after_many_frames_with_wind():
    """Regression test: a parallel `pos[a] += ...` constraint solve (rather than the
    serial one this project uses) raced on GPU — multiple constraints touching the
    same particle corrupted its position, exploding to NaN within a couple of frames."""
    run(120, wind=cr.WIND)
    p = cr.pos.to_numpy()
    assert np.all(np.isfinite(p))
    assert np.all(p >= -1e-3) and np.all(p[:, 0] <= cr.WORLD + 1e-3)


def test_determinism_is_bit_exact():
    """Unlike projects 06/08/09, this solver never scatters into a shared field from
    a parallel loop — it's a deliberately SERIAL Gauss-Seidel pass — so there's no
    atomic-summation-order nondeterminism to allclose away. Same seed, same bits."""
    cr.apply_seed()
    run(60, wind=cr.WIND)
    a = cr.pos.to_numpy()
    cr.apply_seed()
    run(60, wind=cr.WIND)
    b = cr.pos.to_numpy()
    assert np.array_equal(a, b)


def test_grab_moves_particle_to_target_and_release_frees_it():
    cr.grab_at(*cr.pos.to_numpy()[cr.ROPE_BASE + 3])
    assert cr.grabbed[None] == cr.ROPE_BASE + 3
    cr.grab_target[None] = [0.8, 0.8]
    run(30)
    grabbed_idx = cr.grabbed[None]
    assert np.allclose(cr.pos.to_numpy()[grabbed_idx], [0.8, 0.8], atol=1e-4)
    cr.release()
    assert cr.grabbed[None] == -1
    run(10)
    p = cr.pos.to_numpy()
    assert np.all(np.isfinite(p))


def test_grab_at_misses_returns_no_particle():
    cr.grab_at(0.99, 0.01)
    assert cr.grabbed[None] == -1

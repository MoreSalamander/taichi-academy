import numpy as np
import taichi as ti

import volumetric_clouds as vc


# --- pure numpy generation ---------------------------------------------------------


def test_resize_trilinear_shape_and_corners():
    a = np.zeros((2, 2, 2), dtype=np.float32)
    a[1, 1, 1] = 1.0
    big = vc.resize_trilinear(a, 16)
    assert big.shape == (16, 16, 16)
    assert abs(big[0, 0, 0] - 0.0) < 1e-6
    assert abs(big[-1, -1, -1] - 1.0) < 1e-6


def test_fbm3d_range_and_determinism():
    a = vc.fbm3d(vc.VOL_N, rng_seed=3)
    b = vc.fbm3d(vc.VOL_N, rng_seed=3)
    c = vc.fbm3d(vc.VOL_N, rng_seed=4)
    assert a.shape == (vc.VOL_N, vc.VOL_N, vc.VOL_N)
    assert a.min() == 0.0 and abs(a.max() - 1.0) < 1e-6
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_density_bounded_and_shaped_into_a_layer():
    d = vc.seed_density(vc.VOL_N, rng_seed=2)
    assert d.shape == (vc.VOL_N, vc.VOL_N, vc.VOL_N)
    assert d.min() >= 0.0 and d.max() <= 1.0
    top_and_bottom = np.concatenate([d[:, 0, :].ravel(), d[:, -1, :].ravel()])
    middle = d[:, vc.VOL_N // 2, :].ravel()
    assert top_and_bottom.mean() < middle.mean(), "the vertical profile should thin out at top/bottom"


# --- sampling / geometry --------------------------------------------------------------


def test_sample_density_returns_constant_field_value():
    vc.density.fill(0.42)

    @ti.kernel
    def probe(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.f32:
        return vc.sample_density(ti.Vector([x, y, z]))

    assert abs(probe(0.5, 0.5, 0.5) - 0.42) < 1e-5


def test_sample_density_outside_the_box_is_zero():
    vc.density.fill(1.0)

    @ti.kernel
    def probe(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.f32:
        return vc.sample_density(ti.Vector([x, y, z]))

    assert probe(1.5, 0.5, 0.5) == 0.0
    assert probe(0.5, -0.5, 0.5) == 0.0


def test_ray_box_hits_the_unit_cube_head_on():
    @ti.kernel
    def probe() -> ti.math.vec2:
        t_enter, t_exit = vc.ray_box(ti.Vector([0.5, 0.5, -1.0]), ti.Vector([0.0, 0.0, 1.0]))
        return ti.Vector([t_enter, t_exit])

    t_enter, t_exit = probe()
    assert abs(t_enter - 1.0) < 1e-4
    assert abs(t_exit - 2.0) < 1e-4


def test_ray_box_misses_a_ray_that_never_crosses_it():
    """A ray from x=2, moving only in y, never enters the box's x in [0, 1] range."""

    @ti.kernel
    def probe() -> ti.math.vec2:
        t_enter, t_exit = vc.ray_box(ti.Vector([2.0, 0.5, 0.5]), ti.Vector([0.0, 1.0, 0.0]))
        return ti.Vector([t_enter, t_exit])

    t_enter, t_exit = probe()
    assert t_exit < t_enter, "no valid entry/exit interval for a ray that can't reach the box"


def test_march_light_is_full_through_empty_space():
    vc.density.fill(0.0)

    @ti.kernel
    def probe() -> ti.f32:
        return vc.march_light(ti.Vector([0.5, 0.5, 0.5]), ti.Vector([0.0, 1.0, 0.0]))

    assert abs(probe() - 1.0) < 1e-5


def test_march_light_is_dimmer_through_dense_cloud():
    vc.density.fill(1.0)

    @ti.kernel
    def probe() -> ti.f32:
        return vc.march_light(ti.Vector([0.5, 0.5, 0.5]), ti.Vector([0.0, 1.0, 0.0]))

    assert probe() < 0.5, "marching toward the sun through dense cloud should shadow it substantially"


# --- camera --------------------------------------------------------------------------


def test_camera_position_orbits_at_a_fixed_radius():
    import math

    for theta in (0.0, 1.0, 3.7):
        cx, cy, cz = vc.camera_position(theta)
        dx = cx - vc.CENTER[0]
        dz = cz - vc.CENTER[2]
        assert abs(math.hypot(dx, dz) - vc.CAM_RADIUS) < 1e-5
        assert abs(cy - (vc.CENTER[1] + vc.CAM_HEIGHT)) < 1e-6


# --- whole-render behavior --------------------------------------------------------------


def test_render_is_finite_and_bounded():
    cx, cy, cz = vc.camera_position(0.5)
    vc.render(cx, cy, cz, 0.5, 0.5, 0.2)
    px = vc.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0


def test_sky_only_pixel_matches_the_sky_gradient():
    """Looking straight up at the top-left corner (above the cloud layer and the box)
    should show pure sky — no cloud contribution at all."""
    vc.density.fill(0.0)
    cx, cy, cz = vc.camera_position(0.0)
    vc.render(cx, cy, cz, 0.5, 0.5, 0.2)
    px = vc.pixels.to_numpy()
    corner = px[2, 2]
    assert corner[2] > corner[0], "the sky gradient is blue-dominant (more blue than red)"


def test_reseed_changes_the_cloud_shape():
    a = vc.density.to_numpy().copy()
    vc.apply_seed(rng_seed=99)
    b = vc.density.to_numpy()
    assert not np.array_equal(a, b)


def test_stable_across_a_full_orbit():
    for i in range(12):
        theta = i * (2 * np.pi / 12)
        cx, cy, cz = vc.camera_position(theta)
        vc.render(cx, cy, cz, 0.5, 0.5, 0.2)
        px = vc.pixels.to_numpy()
        assert np.all(np.isfinite(px))

import numpy as np
import taichi as ti

import planet as pl


# --- pure numpy generation ---------------------------------------------------------


def test_fbm3d_range_and_determinism():
    a = pl.fbm3d(pl.VOL_N, rng_seed=3)
    b = pl.fbm3d(pl.VOL_N, rng_seed=3)
    c = pl.fbm3d(pl.VOL_N, rng_seed=4)
    assert a.shape == (pl.VOL_N, pl.VOL_N, pl.VOL_N)
    assert a.min() == 0.0 and abs(a.max() - 1.0) < 1e-6
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_terrain_anchors_the_ocean_fraction():
    """The whole point of the quantile re-anchor: EVERY seed gets the same ocean share."""
    for s in (0, 5, 11):
        e = pl.seed_terrain(pl.VOL_N, rng_seed=s)
        frac = (e < pl.SEA_LEVEL).mean()
        assert abs(frac - pl.OCEAN_FRACTION) < 0.02, f"seed {s}: ocean fraction {frac:.3f}"


def test_seed_terrain_bounded():
    e = pl.seed_terrain(pl.VOL_N, rng_seed=2)
    assert e.min() >= 0.0 and e.max() <= 1.0


# --- sampling / geometry --------------------------------------------------------------


def test_sample_elevation_of_constant_field():
    pl.elevation.fill(0.7)

    @ti.kernel
    def probe(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.f32:
        return pl.sample_elevation(ti.Vector([x, y, z]))

    assert abs(probe(0.0, 1.0, 0.0) - 0.7) < 1e-5, "the north pole of the unit sphere"
    assert abs(probe(0.7, 0.0, -0.7) - 0.7) < 1e-5


def test_ray_sphere_hits_head_on():
    @ti.kernel
    def probe() -> ti.f32:
        return pl.ray_sphere(ti.Vector([0.0, 0.0, -3.0]), ti.Vector([0.0, 0.0, 1.0]))

    t = probe()
    assert abs(t - (3.0 - pl.PLANET_R)) < 1e-4


def test_ray_sphere_misses():
    @ti.kernel
    def probe() -> ti.f32:
        return pl.ray_sphere(ti.Vector([0.0, 0.0, -3.0]), ti.Vector([0.0, 1.0, 0.0]))

    assert probe() < 0.0


# --- coloring ------------------------------------------------------------------------


def test_surface_color_ocean_is_blue_land_is_not():
    @ti.kernel
    def probe(h: ti.f32, lat: ti.f32) -> ti.math.vec3:
        return pl.surface_color(h, lat)

    ocean = np.array(probe(pl.SEA_LEVEL * 0.5, 0.0))
    land = np.array(probe(pl.SEA_LEVEL + 0.2, 0.0))
    assert ocean[2] > ocean[0], "ocean: blue beats red"
    assert land[1] > land[2], "low land: green beats blue"


def test_surface_color_poles_are_icy_white():
    @ti.kernel
    def probe(h: ti.f32, lat: ti.f32) -> ti.math.vec3:
        return pl.surface_color(h, lat)

    pole = np.array(probe(pl.SEA_LEVEL * 0.5, 1.0))
    assert pole.min() > 0.9, "at the exact pole, even ocean is fully iced over"


# --- whole-render behavior --------------------------------------------------------------


def test_render_is_finite_and_bounded():
    cx, cy, cz = pl.camera_position(0.6)
    pl.render(cx, cy, cz, *pl.SUN)
    px = pl.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0


def test_planet_silhouette_is_round():
    """Pixels far from the image center should be space; the middle should be planet."""
    cx, cy, cz = pl.camera_position(0.0)
    pl.render(cx, cy, cz, *pl.SUN)
    px = pl.pixels.to_numpy()
    center = px[pl.RES // 2, pl.RES // 2]
    corner = px[2, 2]
    assert center.sum() > corner.sum(), "planet at center is brighter than empty space at the corner"
    assert corner.max() < 0.1, "the corner is dark space"


def test_camera_position_orbits_at_a_fixed_radius():
    import math

    for theta in (0.0, 1.3, 4.0):
        cx, cy, cz = pl.camera_position(theta)
        assert abs(math.hypot(cx, cz) - pl.CAM_RADIUS) < 1e-5
        assert cy == pl.CAM_HEIGHT


def test_reseed_changes_the_world():
    a = pl.elevation.to_numpy().copy()
    pl.apply_seed(rng_seed=42)
    b = pl.elevation.to_numpy()
    assert not np.array_equal(a, b)


def test_stable_across_a_full_orbit():
    for i in range(8):
        theta = i * (2 * np.pi / 8)
        cx, cy, cz = pl.camera_position(theta)
        pl.render(cx, cy, cz, *pl.SUN)
        assert np.all(np.isfinite(pl.pixels.to_numpy()))

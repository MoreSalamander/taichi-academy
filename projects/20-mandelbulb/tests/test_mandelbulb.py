import numpy as np
import taichi as ti

import mandelbulb as mb


def probe_de(x, y, z):
    @ti.kernel
    def k() -> ti.f32:
        return mb.bulb_de(ti.Vector([x, y, z]))

    return k()


# --- the distance estimator ------------------------------------------------------------


def test_de_is_positive_outside_the_bulb():
    assert probe_de(2.0, 0.0, 0.0) > 0.1
    assert probe_de(0.0, 0.0, 3.0) > 0.5


def test_de_shrinks_approaching_the_surface():
    """The estimator's defining property: closer to the set, smaller the bound."""
    far = probe_de(2.5, 0.0, 0.0)
    mid = probe_de(1.8, 0.0, 0.0)
    near = probe_de(1.35, 0.0, 0.0)
    assert far > mid > near > 0.0


def test_de_never_overshoots_along_a_ray():
    """Sphere tracing is only correct if stepping by the DE never crosses the surface:
    after stepping d = DE(p), the new point's DE must still be non-negative."""
    p = np.array([2.5, 0.3, 0.4], dtype=np.float32)
    rd = -p / np.linalg.norm(p)
    for _ in range(60):
        d = probe_de(*map(float, p))
        if d < 1e-4:
            break
        p = p + rd * d
        assert probe_de(*map(float, p)) > -1e-4, "a DE step must never tunnel through the set"


def test_normal_points_outward_on_the_positive_axis():
    @ti.kernel
    def k() -> ti.math.vec3:
        return mb.normal_at(ti.Vector([1.4, 0.0, 0.0]), 0.001)

    n = np.array(k())
    assert n[0] > 0.7, f"the outward normal near +x should point mostly +x, got {n}"


# --- camera / render --------------------------------------------------------------------


def test_camera_position_orbits_at_radius():
    import math

    for theta, radius in ((0.0, 2.0), (1.3, 1.5)):
        cx, cy, cz = mb.camera_position(theta, radius)
        assert abs(math.hypot(cx, cy) - radius) < 1e-5
        assert abs(cz - radius * mb.CAM_HEIGHT_RATIO) < 1e-6


def test_render_hits_the_bulb_and_stays_bounded():
    cx, cy, cz = mb.camera_position(0.7, 2.2)
    mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * 2.2)
    px = mb.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0
    hit = (px.sum(axis=2) > 0.1).mean()
    assert 0.2 < hit < 0.9, f"the bulb should fill a sensible fraction of the frame, got {hit:.2f}"


def test_zoomed_view_still_finds_surface():
    """Regression test: an early zoom floor of 0.15 put the camera INSIDE the bulb —
    every ray 'hit' at t=0 and the frame rendered as a featureless flat wall."""
    cx, cy, cz = mb.camera_position(0.7, mb.RADIUS_MIN)
    mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * mb.RADIUS_MIN)
    px = mb.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    lit = px.sum(axis=2)
    assert lit.std() > 0.1, "a close-up must show structure, not a uniform wall"


def test_different_angles_show_different_views():
    cx, cy, cz = mb.camera_position(0.0, 2.2)
    mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * 2.2)
    a = mb.pixels.to_numpy().copy()
    cx, cy, cz = mb.camera_position(1.2, 2.2)
    mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * 2.2)
    b = mb.pixels.to_numpy()
    assert not np.allclose(a, b)


def test_render_is_bit_exact_deterministic():
    """No seeds, no randomness, no atomics — pure math should render identically."""
    cx, cy, cz = mb.camera_position(0.5, 2.0)
    mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * 2.0)
    a = mb.pixels.to_numpy().copy()
    mb.pixels.fill(0.0)
    mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * 2.0)
    b = mb.pixels.to_numpy()
    assert np.array_equal(a, b)

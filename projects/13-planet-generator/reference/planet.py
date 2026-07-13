"""Planet Generator: a whole world from one seed — 3D noise on a sphere, no map seams."""

import math

import numpy as np
import taichi as ti

RES = 400
VOL_N = 64

OCEAN_FRACTION = 0.62
SEA_LEVEL = 0.5
ELEV_GAIN = 1.5
ICE_LAT = 0.72

PLANET_R = 1.0
CAM_RADIUS = 2.6
CAM_HEIGHT = 0.5
ORBIT_SPEED = 0.004
ZOOM = 1.8

SUN = (0.6, 0.3, -0.6)

elevation = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global elevation, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    elevation = ti.field(ti.f32, shape=(VOL_N, VOL_N, VOL_N))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


def resize_trilinear(a, n):
    """Pure numpy: trilinear resize of a small 3D array up to n x n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1 - f)[:, None, None] + a[i1] * f[:, None, None]
    a = a[:, i0] * (1 - f)[None, :, None] + a[:, i1] * f[None, :, None]
    a = a[:, :, i0] * (1 - f)[None, None, :] + a[:, :, i1] * f[None, None, :]
    return a


def fbm3d(n, rng_seed=0, octaves=5, roughness=0.55):
    """Pure numpy: fractal 3D noise — octaves of noise, each finer and fainter."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n, n, n), dtype=np.float32)
    amp = 1.0
    res = 4
    for _ in range(octaves):
        layer = rng.uniform(0.0, 1.0, size=(res, res, res)).astype(np.float32)
        out += amp * resize_trilinear(layer, n)
        amp *= roughness
        res *= 2
    out -= out.min()
    out /= out.max()
    return out.astype(np.float32)


def seed_terrain(n, rng_seed=0):
    """Pure numpy: fbm noise, re-anchored so a fixed fraction of the world is ocean."""
    noise = fbm3d(n, rng_seed)
    sea = np.quantile(noise, OCEAN_FRACTION)
    return np.clip(SEA_LEVEL + (noise - sea) * ELEV_GAIN, 0.0, 1.0).astype(np.float32)


def apply_seed(rng_seed=0):
    elevation.from_numpy(seed_terrain(VOL_N, rng_seed))


@ti.func
def sample_elevation(p):
    q = ti.math.clamp(p * 0.5 + 0.5, 0.0, 1.0)
    x = q * (VOL_N - 1)
    x0 = ti.cast(ti.floor(x), ti.i32)
    x1 = ti.min(x0 + 1, VOL_N - 1)
    f = x - x0
    c00 = elevation[x0[0], x0[1], x0[2]] * (1 - f[0]) + elevation[x1[0], x0[1], x0[2]] * f[0]
    c10 = elevation[x0[0], x1[1], x0[2]] * (1 - f[0]) + elevation[x1[0], x1[1], x0[2]] * f[0]
    c01 = elevation[x0[0], x0[1], x1[2]] * (1 - f[0]) + elevation[x1[0], x0[1], x1[2]] * f[0]
    c11 = elevation[x0[0], x1[1], x1[2]] * (1 - f[0]) + elevation[x1[0], x1[1], x1[2]] * f[0]
    c0 = c00 * (1 - f[1]) + c10 * f[1]
    c1 = c01 * (1 - f[1]) + c11 * f[1]
    return c0 * (1 - f[2]) + c1 * f[2]


@ti.func
def ray_sphere(origin, rd):
    b = origin.dot(rd)
    c = origin.dot(origin) - PLANET_R * PLANET_R
    disc = b * b - c
    t = -1.0
    if disc > 0:
        t = -b - ti.sqrt(disc)
    return t


@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t


@ti.func
def surface_color(h, lat):
    c = ti.Vector([0.05, 0.15, 0.45])
    if h > SEA_LEVEL:
        land = (h - SEA_LEVEL) / (1.0 - SEA_LEVEL)
        c = band(ti.Vector([0.75, 0.7, 0.45]), ti.Vector([0.2, 0.5, 0.2]), land, 0.0, 0.35)
        c = band(c, ti.Vector([0.45, 0.4, 0.35]), land, 0.45, 0.75)
        c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.8, 0.95)
    else:
        shallow = h / SEA_LEVEL
        c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), shallow, 0.4, 1.0)
    if lat > ICE_LAT:
        ice = (lat - ICE_LAT) / (1.0 - ICE_LAT)
        c = c * (1.0 - ice) + ti.Vector([0.95, 0.97, 1.0]) * ice
    return c


@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32, sunx: ti.f32, suny: ti.f32, sunz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        col = ti.Vector([0.02, 0.02, 0.05])
        t = ray_sphere(cam, rd)
        if t > 0:
            p = cam + rd * t
            n = p.normalized()
            h = sample_elevation(n)
            c = surface_color(h, ti.abs(n[1]))

            diffuse = ti.max(n.dot(sun), 0.0)
            spec = 0.0
            if h <= SEA_LEVEL:
                refl = 2.0 * n.dot(sun) * n - sun
                spec = ti.pow(ti.max(refl.dot(-rd), 0.0), 32) * 0.6
            shade = 0.06 + 0.94 * diffuse
            col = c * shade + ti.Vector([1.0, 0.95, 0.85]) * spec

            rim = ti.pow(1.0 - ti.max(n.dot(-rd), 0.0), 3)
            col += ti.Vector([0.3, 0.5, 1.0]) * rim * 0.8 * (0.3 + 0.7 * diffuse)

        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)


def camera_position(theta):
    return (
        CAM_RADIUS * math.sin(theta),
        CAM_HEIGHT,
        -CAM_RADIUS * math.cos(theta),
    )


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Planet Generator — taichi-academy", res=RES, background_color=0x0A0A12)
    theta = 0.0
    pmx = None
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED
        cx, cy, cz = camera_position(theta)
        render(cx, cy, cz, *SUN)
        gui.set_image(pixels)
        gui.text("drag to orbit  [r] new planet", (0.02, 0.98), color=0xFFFFFF)
        gui.show()


if __name__ == "__main__":
    main()

"""Volumetric Clouds: 3D fractal noise, ray marching, and a light march for real shading."""

import math

import numpy as np
import taichi as ti

RES = 400
VOL_N = 48

STEPS = 96
LIGHT_STEPS = 6
LIGHT_STEP_SIZE = 0.06
ABSORPTION = 18.0
AMBIENT = 0.25
SUN_INTENSITY = 1.3

CENTER = (0.5, 0.35, 0.5)
CAM_RADIUS = 1.1
CAM_HEIGHT = 0.35
ORBIT_SPEED = 0.01

density = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global density, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    density = ti.field(ti.f32, shape=(VOL_N, VOL_N, VOL_N))
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


def fbm3d(n, rng_seed=0, octaves=4, roughness=0.55):
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


def seed_density(n, rng_seed=0):
    """Pure numpy: fbm noise shaped into a cloud layer by a vertical profile."""
    noise = fbm3d(n, rng_seed)
    yv = np.linspace(0.0, 1.0, n)
    profile = np.clip(1.0 - np.abs(yv - 0.45) / 0.35, 0.0, 1.0) ** 1.2
    shaped = noise * profile[None, :, None]
    return np.clip((shaped - 0.22) / 0.78, 0.0, 1.0).astype(np.float32)


def apply_seed(rng_seed=0):
    density.from_numpy(seed_density(VOL_N, rng_seed))


@ti.func
def sample_density(p):
    v = 0.0
    if 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 and 0.0 <= p[2] <= 1.0:
        x = p * (VOL_N - 1)
        x0 = ti.cast(ti.floor(x), ti.i32)
        x1 = ti.min(x0 + 1, VOL_N - 1)
        f = x - x0
        c00 = density[x0[0], x0[1], x0[2]] * (1 - f[0]) + density[x1[0], x0[1], x0[2]] * f[0]
        c10 = density[x0[0], x1[1], x0[2]] * (1 - f[0]) + density[x1[0], x1[1], x0[2]] * f[0]
        c01 = density[x0[0], x0[1], x1[2]] * (1 - f[0]) + density[x1[0], x0[1], x1[2]] * f[0]
        c11 = density[x0[0], x1[1], x1[2]] * (1 - f[0]) + density[x1[0], x1[1], x1[2]] * f[0]
        c0 = c00 * (1 - f[1]) + c10 * f[1]
        c1 = c01 * (1 - f[1]) + c11 * f[1]
        v = c0 * (1 - f[2]) + c1 * f[2]
    return v


@ti.func
def ray_box(origin, rd):
    inv = 1.0 / rd
    t0 = (ti.Vector([0.0, 0.0, 0.0]) - origin) * inv
    t1 = (ti.Vector([1.0, 1.0, 1.0]) - origin) * inv
    tmin = ti.min(t0, t1)
    tmax = ti.max(t0, t1)
    t_enter = ti.max(ti.max(tmin[0], tmin[1]), tmin[2])
    t_exit = ti.min(ti.min(tmax[0], tmax[1]), tmax[2])
    return t_enter, t_exit


@ti.func
def march_light(p, sun):
    lt = 1.0
    for k in range(LIGHT_STEPS):
        lp = p + sun * LIGHT_STEP_SIZE * (k + 1)
        ld = sample_density(lp)
        lt *= ti.exp(-ld * LIGHT_STEP_SIZE * ABSORPTION)
    return lt


@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32, sunx: ti.f32, suny: ti.f32, sunz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz])
    center = ti.Vector([CENTER[0], CENTER[1], CENTER[2]])
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (center - cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward + u * right + v * up).normalized()

        t_enter, t_exit = ray_box(cam, rd)
        t_enter = ti.max(t_enter, 0.0)

        transmittance = 1.0
        color = ti.Vector([0.0, 0.0, 0.0])
        if t_exit > t_enter:
            step_size = (t_exit - t_enter) / STEPS
            t = t_enter
            for _s in range(STEPS):
                p = cam + rd * t
                d = sample_density(p)
                if d > 0.001:
                    lt = march_light(p, sun)
                    luminance = AMBIENT + lt * SUN_INTENSITY
                    alpha = 1.0 - ti.exp(-d * step_size * ABSORPTION)
                    color += transmittance * alpha * luminance * ti.Vector([1.0, 0.97, 0.92])
                    transmittance *= 1.0 - alpha
                    if transmittance < 0.01:
                        break
                t += step_size

        sky_t = ti.max(rd[1], 0.0)
        sky = ti.Vector([0.5, 0.65, 0.9]) * (1.0 - sky_t) + ti.Vector([0.2, 0.35, 0.7]) * sky_t
        pixels[i, j] = ti.math.clamp(color + transmittance * sky, 0.0, 1.0)


def camera_position(theta):
    return (
        CENTER[0] + CAM_RADIUS * math.sin(theta),
        CENTER[1] + CAM_HEIGHT,
        CENTER[2] - CAM_RADIUS * math.cos(theta),
    )


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Volumetric Clouds — taichi-academy", res=RES, background_color=0x0A0A12)
    theta = 0.0
    sun = (0.5, 0.5, 0.2)
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
        render(cx, cy, cz, *sun)
        gui.set_image(pixels)
        gui.text("drag to orbit  [r] new clouds", (0.02, 0.98), color=0xFFFFFF)
        gui.show()


if __name__ == "__main__":
    main()

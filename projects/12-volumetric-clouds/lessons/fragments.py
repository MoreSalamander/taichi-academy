"""Code SOT for project 12 — volumetric clouds.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 12-volumetric-clouds`.

Evolutions: render() goes through three full-body versions — a cheap top-down
max-intensity projection (chapter 1, no camera needed, just prove the noise is
cloud-shaped), a perspective ray marcher with flat ambient-only shading
(chapter 2, real 3D but no shadows), and the final version adding a light
march for real volumetric shading (chapter 3). main()'s camera update grows
from a plain auto-rotate into mouse-drag orbit control in chapter 4.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="12-volumetric-clouds",
    default_file="volumetric_clouds.py",
    reference={"volumetric_clouds.py": PROJECT_DIR / "reference" / "volumetric_clouds.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Volumetric Clouds: 3D fractal noise, ray marching, and a light march for real shading."""'))
frag(((1, 1), "import math"))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 400"))
frag(((1, 2), "VOL_N = 48"))

frag(((2, 1), "STEPS = 96"))
frag(((3, 1), "LIGHT_STEPS = 6"))
frag(((3, 1), "LIGHT_STEP_SIZE = 0.06"))
frag(((2, 1), "ABSORPTION = 18.0"))
frag(((2, 1), "AMBIENT = 0.25"))
frag(((3, 1), "SUN_INTENSITY = 1.3"))

frag(((2, 3), "CENTER = (0.5, 0.35, 0.5)"))
frag(((2, 3), "CAM_RADIUS = 1.1"))
frag(((2, 3), "CAM_HEIGHT = 0.35"))
frag(((2, 3), "ORBIT_SPEED = 0.01"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "density = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global density, pixels"))
frag(
    (
        (1, 2),
        "    if arch is None:\n"
        "        try:\n"
        "            ti.init(arch=ti.gpu)\n"
        "        except Exception:\n"
        "            ti.init(arch=ti.cpu)\n"
        "    else:\n"
        "        ti.init(arch=arch)",
    )
)
frag(((1, 2), "    density = ti.field(ti.f32, shape=(VOL_N, VOL_N, VOL_N))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- pure numpy generation ---------------------------------------------------------

RESIZE = '''def resize_trilinear(a, n):
    """Pure numpy: trilinear resize of a small 3D array up to n x n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1 - f)[:, None, None] + a[i1] * f[:, None, None]
    a = a[:, i0] * (1 - f)[None, :, None] + a[:, i1] * f[None, :, None]
    a = a[:, :, i0] * (1 - f)[None, None, :] + a[:, :, i1] * f[None, None, :]
    return a'''

frag(((1, 3), RESIZE))

FBM3D = '''def fbm3d(n, rng_seed=0, octaves=4, roughness=0.55):
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
    return out.astype(np.float32)'''

frag(((1, 3), FBM3D))

SEED_DENSITY = '''def seed_density(n, rng_seed=0):
    """Pure numpy: fbm noise shaped into a cloud layer by a vertical profile."""
    noise = fbm3d(n, rng_seed)
    yv = np.linspace(0.0, 1.0, n)
    profile = np.clip(1.0 - np.abs(yv - 0.45) / 0.35, 0.0, 1.0) ** 1.2
    shaped = noise * profile[None, :, None]
    return np.clip((shaped - 0.22) / 0.78, 0.0, 1.0).astype(np.float32)'''

frag(((1, 3), SEED_DENSITY))

frag(((1, 3), "def apply_seed(rng_seed=0):\n    density.from_numpy(seed_density(VOL_N, rng_seed))"))

# --- sampling / geometry --------------------------------------------------------------

SAMPLE_DENSITY = """@ti.func
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
    return v"""

frag(((2, 1), SAMPLE_DENSITY))

RAY_BOX = """@ti.func
def ray_box(origin, rd):
    inv = 1.0 / rd
    t0 = (ti.Vector([0.0, 0.0, 0.0]) - origin) * inv
    t1 = (ti.Vector([1.0, 1.0, 1.0]) - origin) * inv
    tmin = ti.min(t0, t1)
    tmax = ti.max(t0, t1)
    t_enter = ti.max(ti.max(tmin[0], tmin[1]), tmin[2])
    t_exit = ti.min(ti.min(tmax[0], tmax[1]), tmax[2])
    return t_enter, t_exit"""

frag(((2, 1), RAY_BOX))

MARCH_LIGHT = """@ti.func
def march_light(p, sun):
    lt = 1.0
    for k in range(LIGHT_STEPS):
        lp = p + sun * LIGHT_STEP_SIZE * (k + 1)
        ld = sample_density(lp)
        lt *= ti.exp(-ld * LIGHT_STEP_SIZE * ABSORPTION)
    return lt"""

frag(((3, 1), MARCH_LIGHT))

# --- the renderer: three full versions -------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        di = ti.min(ti.cast(i / RES * VOL_N, ti.i32), VOL_N - 1)
        dj = ti.min(ti.cast(j / RES * VOL_N, ti.i32), VOL_N - 1)
        m = 0.0
        for k in range(VOL_N):
            m = ti.max(m, density[di, k, dj])
        pixels[i, j] = ti.Vector([m, m, m])"""

RENDER_V2 = """@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
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
                    alpha = 1.0 - ti.exp(-d * step_size * ABSORPTION)
                    color += transmittance * alpha * AMBIENT * ti.Vector([1.0, 0.97, 0.92])
                    transmittance *= 1.0 - alpha
                    if transmittance < 0.01:
                        break
                t += step_size

        sky_t = ti.max(rd[1], 0.0)
        sky = ti.Vector([0.5, 0.65, 0.9]) * (1.0 - sky_t) + ti.Vector([0.2, 0.35, 0.7]) * sky_t
        pixels[i, j] = ti.math.clamp(color + transmittance * sky, 0.0, 1.0)"""

RENDER_V3 = """@ti.kernel
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
        pixels[i, j] = ti.math.clamp(color + transmittance * sky, 0.0, 1.0)"""

frag(((1, 3), RENDER_V1), ((2, 2), RENDER_V2), ((3, 2), RENDER_V3))

CAMERA_POSITION = """def camera_position(theta):
    return (
        CENTER[0] + CAM_RADIUS * math.sin(theta),
        CENTER[1] + CAM_HEIGHT,
        CENTER[2] - CAM_RADIUS * math.cos(theta),
    )"""

frag(((2, 3), CAMERA_POSITION))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Volumetric Clouds — taichi-academy", res=RES, background_color=0x0A0A12)'))
frag(((2, 3), "    theta = 0.0"))
frag(((3, 2), "    sun = (0.5, 0.5, 0.2)"))
frag(((4, 1), "    pmx = None"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((4, 2), EVENTS_V2))

THETA_UPDATE_V1 = "        theta += ORBIT_SPEED"

THETA_UPDATE_V2 = """        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED"""

frag(((2, 3), THETA_UPDATE_V1), ((4, 1), THETA_UPDATE_V2))

frag(((2, 3), "        cx, cy, cz = camera_position(theta)"))

RENDER_CALL_V1 = "        render()"
RENDER_CALL_V2 = "        render(cx, cy, cz)"
RENDER_CALL_V3 = "        render(cx, cy, cz, *sun)"

frag(((1, 3), RENDER_CALL_V1), ((2, 3), RENDER_CALL_V2), ((3, 2), RENDER_CALL_V3))

frag(((1, 3), "        gui.set_image(pixels)"))
frag(((4, 3), '        gui.text("drag to orbit  [r] new clouds", (0.02, 0.98), color=0xFFFFFF)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

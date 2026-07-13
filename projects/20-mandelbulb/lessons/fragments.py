"""Code SOT for project 20 — mandelbulb.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 20-mandelbulb`.

Evolutions: the renderer is one fragment with four versions — a flat 2D slice
of the distance field itself (chapter 1: SEE the oracle before using it), a
sphere-tracer with flat white hits (chapter 2), +diffuse normals, +AO and the
normal palette (final). main()'s camera grows auto-orbit (ch2), zoom keys
(ch3), then drag (ch3). This project has NO random seed anywhere — the
mandelbulb is one fixed object; only the camera changes.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="20-mandelbulb",
    default_file="mandelbulb.py",
    reference={"mandelbulb.py": PROJECT_DIR / "reference" / "mandelbulb.py"},
    chapter_steps={1: 3, 2: 3, 3: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Mandelbulb: a 3D fractal you can\'t mesh — ray-marched by asking \'how far is it, at least?\'"""'))
frag(((1, 1), "import math"))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 400"))
frag(((1, 2), "POWER = 8.0"))
frag(((1, 2), "DE_ITERS = 12"))
frag(((1, 2), "BAILOUT = 2.0"))
frag(((2, 1), "MAX_STEPS = 128"))
frag(((2, 1), "EPS_BASE = 0.0004"))
frag(((2, 1), "MAX_DIST = 4.0"))
frag(((2, 1), "ZOOM = 1.6"))

frag(((2, 1), "CAM_HEIGHT_RATIO = 0.36"))
frag(((2, 1), "ORBIT_SPEED = 0.004"))
frag(((3, 1), "RADIUS_MIN = 1.35"))
frag(((3, 1), "RADIUS_MAX = 4.0"))

frag(((2, 1), "SUN = (0.5, -0.4, 0.75)"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pixels"))
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
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- the distance estimator ------------------------------------------------------------

BULB_DE = """@ti.func
def bulb_de(p):
    z = p
    dr = 1.0
    r = ti.max(z.norm(), 1e-9)
    for _ in range(DE_ITERS):
        if r < BAILOUT:
            theta = ti.acos(z[2] / r) * POWER
            phi = ti.atan2(z[1], z[0]) * POWER
            zr = ti.pow(r, POWER)
            dr = ti.pow(r, POWER - 1.0) * POWER * dr + 1.0
            z = zr * ti.Vector([ti.sin(theta) * ti.cos(phi), ti.sin(theta) * ti.sin(phi), ti.cos(theta)]) + p
            r = ti.max(z.norm(), 1e-9)
    return 0.5 * ti.log(r) * r / dr"""

frag(((1, 3), BULB_DE))

NORMAL_AT = """@ti.func
def normal_at(p, e):
    dx = bulb_de(p + ti.Vector([e, 0.0, 0.0])) - bulb_de(p - ti.Vector([e, 0.0, 0.0]))
    dy = bulb_de(p + ti.Vector([0.0, e, 0.0])) - bulb_de(p - ti.Vector([0.0, e, 0.0]))
    dz = bulb_de(p + ti.Vector([0.0, 0.0, e])) - bulb_de(p - ti.Vector([0.0, 0.0, e]))
    return ti.Vector([dx, dy, dz]).normalized()"""

frag(((2, 2), NORMAL_AT))

# --- the renderer: four versions --------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render_slice():
    for i, j in pixels:
        x = (i / RES - 0.5) * 3.0
        y = (j / RES - 0.5) * 3.0
        d = bulb_de(ti.Vector([x, y, 0.0]))
        v = ti.math.clamp(d * 0.8, 0.0, 1.0)
        c = ti.Vector([v, v, v])
        if d < 0.01:
            c = ti.Vector([0.9, 0.5, 1.0])
        pixels[i, j] = c"""

RENDER_V2 = """@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            col = ti.Vector([0.9, 0.9, 0.9])
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)"""

RENDER_V3 = """@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            p = cam + rd * t
            n = normal_at(p, eps)
            diffuse = ti.max(n.dot(sun), 0.0)
            col = ti.Vector([0.85, 0.8, 0.9]) * (0.15 + 0.85 * diffuse)
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)"""

RENDER_V4 = """@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            p = cam + rd * t
            n = normal_at(p, eps)
            diffuse = ti.max(n.dot(sun), 0.0)
            ao = 1.0 - steps / float(MAX_STEPS)
            base = ti.Vector([0.85, 0.6, 0.95]) * 0.5 + 0.5 * ti.Vector([ti.abs(n[0]), ti.abs(n[1]), ti.abs(n[2])])
            col = base * (0.15 + 0.85 * diffuse) * ao
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)"""

frag(((1, 3), RENDER_V1), ((2, 1), RENDER_V2), ((2, 2), RENDER_V3), ((2, 3), RENDER_V4))

CAMERA_POSITION = """def camera_position(theta, radius):
    return (
        radius * math.sin(theta),
        -radius * math.cos(theta),
        radius * CAM_HEIGHT_RATIO,
    )"""

frag(((2, 1), CAMERA_POSITION))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()"))
frag(((1, 3), '    gui = ti.GUI("Mandelbulb — taichi-academy", res=RES, background_color=0x000000)'))
frag(((2, 1), "    theta = 0.7\n    radius = 2.2"))
frag(((3, 2), "    pmx = None"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in (ti.GUI.UP, "w"):
                radius = max(radius * 0.94, RADIUS_MIN)
            elif e.key in (ti.GUI.DOWN, "s"):
                radius = min(radius / 0.94, RADIUS_MAX)'''

frag(((1, 3), EVENTS_V1), ((3, 1), EVENTS_V2))

ANGLE_V1 = "        theta += ORBIT_SPEED"

ANGLE_V2 = """        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED"""

frag(((2, 1), ANGLE_V1), ((3, 2), ANGLE_V2))

RENDER_CALL_V1 = "        render_slice()"

RENDER_CALL_V2 = """        cx, cy, cz = camera_position(theta, radius)
        eps = EPS_BASE * radius
        render(cx, cy, cz, *SUN, eps)"""

frag(((1, 3), RENDER_CALL_V1), ((2, 1), RENDER_CALL_V2))

frag(((1, 3), "        gui.set_image(pixels)"))
frag(((3, 2), '        gui.text(f"radius {radius:.2f}", (0.02, 0.98), color=0xFFFFFF)'))
frag(((3, 2), '        gui.text("drag to orbit  [w/up] zoom in  [s/down] zoom out", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

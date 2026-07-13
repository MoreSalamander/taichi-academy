"""Code SOT for project 19 — strange attractors.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 19-strange-attractors`.

Evolutions: the attractor-id line, FRAME table, step_attractor, and splat all
grow one attractor at a time through chapter 3 (full-body replacements — the
checker compares normalized text). Chapter 1 deliberately ships WITHOUT the
settle loop so the learner watches the random cloud collapse onto the Lorenz
butterfly live; chapter 2 adds rotation+speed color, then settle+reseed.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="19-strange-attractors",
    default_file="strange_attractors.py",
    reference={"strange_attractors.py": PROJECT_DIR / "reference" / "strange_attractors.py"},
    chapter_steps={1: 3, 2: 2, 3: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Strange Attractors: 200,000 points fall onto the same impossible shape, every time."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "N_PTS = 200000"))
frag(((1, 2), "FADE = 0.90"))
frag(((2, 1), "ROT_SPEED = 0.01"))
frag(((2, 2), "SETTLE_STEPS = 2000"))

frag(
    ((1, 2), "LORENZ = 0"),
    ((3, 1), "LORENZ, THOMAS = 0, 1"),
    ((3, 2), "LORENZ, THOMAS, AIZAWA = 0, 1, 2"),
    ((3, 3), "LORENZ, THOMAS, AIZAWA, CLIFFORD = 0, 1, 2, 3"),
)
frag(((3, 3), 'NAMES = {LORENZ: "lorenz", THOMAS: "thomas", AIZAWA: "aizawa", CLIFFORD: "clifford"}'))

FRAME_V1 = """# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0),
}"""

FRAME_V2 = """# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
}"""

FRAME_V3 = """# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
}"""

FRAME_V4 = """# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
    AIZAWA: (0.28, 0.0, 0.0, 0.0, 0.01, 0.015, 0.4, 0.35),
}"""

FRAME_V5 = """# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
    AIZAWA: (0.28, 0.0, 0.0, 0.0, 0.01, 0.015, 0.4, 0.35),
    CLIFFORD: (0.20, 0.0, 0.0, 0.0, 1.0, 0.04, 1.0, 0.0),
}"""

frag(((1, 2), FRAME_V1), ((2, 1), FRAME_V2), ((3, 1), FRAME_V3), ((3, 2), FRAME_V4), ((3, 3), FRAME_V5))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, pixels"))
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
frag(((1, 2), "    pos = ti.Vector.field(3, ti.f32, shape=N_PTS)"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- seeding + dynamics ---------------------------------------------------------------

SEED_POINTS = '''def seed_points(n, seed_scale, rng_seed=0):
    """Pure numpy: a random cloud sized to land inside the attractor's basin."""
    rng = np.random.default_rng(rng_seed)
    return (rng.uniform(-1.0, 1.0, size=(n, 3)) * seed_scale).astype(np.float32)'''

frag(((1, 3), SEED_POINTS))

DERIV_LORENZ = """@ti.func
def deriv_lorenz(p):
    return ti.Vector([10.0 * (p[1] - p[0]), p[0] * (28.0 - p[2]) - p[1], p[0] * p[1] - 8.0 / 3.0 * p[2]])"""

frag(((1, 3), DERIV_LORENZ))

DERIV_THOMAS = """@ti.func
def deriv_thomas(p):
    b = 0.19
    return ti.Vector([ti.sin(p[1]) - b * p[0], ti.sin(p[2]) - b * p[1], ti.sin(p[0]) - b * p[2]])"""

frag(((3, 1), DERIV_THOMAS))

DERIV_AIZAWA = """@ti.func
def deriv_aizawa(p):
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
    x, y, z = p[0], p[1], p[2]
    return ti.Vector([
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z**3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x**3,
    ])"""

frag(((3, 2), DERIV_AIZAWA))

MAP_CLIFFORD = """@ti.func
def map_clifford(p):
    a, b, c, d = -1.4, 1.6, 1.0, 0.7
    return ti.Vector([ti.sin(a * p[1]) + c * ti.cos(a * p[0]), ti.sin(b * p[0]) + d * ti.cos(b * p[1]), 0.0])"""

frag(((3, 3), MAP_CLIFFORD))

STEP_ATTR_V1 = """@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        pos[i] = p + dt * deriv_lorenz(p)"""

STEP_ATTR_V2 = """@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        else:
            pos[i] = p + dt * deriv_thomas(p)"""

STEP_ATTR_V3 = """@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        elif kind == THOMAS:
            pos[i] = p + dt * deriv_thomas(p)
        else:
            pos[i] = p + dt * deriv_aizawa(p)"""

STEP_ATTR_V4 = """@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        elif kind == THOMAS:
            pos[i] = p + dt * deriv_thomas(p)
        elif kind == AIZAWA:
            pos[i] = p + dt * deriv_aizawa(p)
        else:
            pos[i] = map_clifford(p)"""

frag(((1, 3), STEP_ATTR_V1), ((3, 1), STEP_ATTR_V2), ((3, 2), STEP_ATTR_V3), ((3, 3), STEP_ATTR_V4))

APPLY_V1 = """def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    pixels.fill(0.0)"""

APPLY_V2 = """def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale, _spd = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    pixels.fill(0.0)"""

APPLY_V3 = """def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale, _spd = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    for _ in range(SETTLE_STEPS):
        step_attractor(kind, dt)
    pixels.fill(0.0)"""

frag(((1, 3), APPLY_V1), ((2, 1), APPLY_V2), ((2, 2), APPLY_V3))

# --- rendering -----------------------------------------------------------------------

frag(((1, 3), "@ti.kernel\ndef fade():\n    for i, j in pixels:\n        pixels[i, j] *= FADE"))

SPLAT_V1 = """@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32):
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0]
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] += ti.Vector([0.9, 0.9, 1.0]) * gain"""

SPLAT_V2 = """@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = ti.math.clamp(deriv_lorenz(pos[i]).norm() * speed_scale, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain"""

SPLAT_V3 = """@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            else:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain"""

SPLAT_V4 = """@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            elif kind == THOMAS:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            else:
                spd = deriv_aizawa(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain"""

SPLAT_V5 = """@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        if kind == CLIFFORD:
            rx = p[0]
            ry = p[1]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            elif kind == THOMAS:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            elif kind == AIZAWA:
                spd = deriv_aizawa(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain"""

frag(((1, 3), SPLAT_V1), ((2, 1), SPLAT_V2), ((3, 1), SPLAT_V3), ((3, 2), SPLAT_V4), ((3, 3), SPLAT_V5))

frag(((1, 3), "@ti.kernel\ndef clamp_pixels():\n    for i, j in pixels:\n        pixels[i, j] = ti.min(pixels[i, j], 1.0)"))

STEP_V1 = """def step(kind, angle):
    scale, cx, cy, cz, dt, gain, _seed = FRAME[kind]
    step_attractor(kind, dt)
    fade()
    splat(kind, angle, scale, cx, cy, cz, gain)
    clamp_pixels()"""

STEP_V2 = """def step(kind, angle):
    scale, cx, cy, cz, dt, gain, _seed, speed_scale = FRAME[kind]
    step_attractor(kind, dt)
    fade()
    splat(kind, angle, scale, cx, cy, cz, gain, speed_scale)
    clamp_pixels()"""

frag(((1, 3), STEP_V1), ((2, 1), STEP_V2))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    kind = LORENZ\n    apply_seed(kind)"))
frag(((1, 3), '    gui = ti.GUI("Strange Attractors — taichi-academy", res=RES, background_color=0x000000)'))
frag(((2, 1), "    angle = 0.0"))
frag(((3, 3), "    pmx = None"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "12":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))'''

EVENTS_V4 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "123":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))'''

EVENTS_V5 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "1234":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((2, 2), EVENTS_V2), ((3, 1), EVENTS_V3), ((3, 2), EVENTS_V4), ((3, 3), EVENTS_V5))

ANGLE_V1 = "        angle += ROT_SPEED"

ANGLE_V2 = """        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                angle -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            angle += ROT_SPEED"""

frag(((2, 1), ANGLE_V1), ((3, 3), ANGLE_V2))

frag(((1, 3), "        step(kind, 0.0)"), ((2, 1), "        step(kind, angle)"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((3, 3), '        gui.text(f"attractor: {NAMES[kind]}", (0.02, 0.98), color=0xFFFFFF)'))
frag(((3, 3), '        gui.text("[1] lorenz  [2] thomas  [3] aizawa  [4] clifford  drag to spin  [r] rescatter", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

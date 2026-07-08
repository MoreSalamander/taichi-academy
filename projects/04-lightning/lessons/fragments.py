"""Code SOT for project 04 — branching lightning.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 04-lightning`.

Evolutions: main opens with a straight-line preview in ch1 that ch2 deletes
(strike() takes over); generate_bolt gains its branch block in ch3; absorb and
fade gain glow lines in ch4; render grows sky→halo→flash across ch1/ch4/ch5;
the event block accretes a branch per feature.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="04-lightning",
    default_file="lightning.py",
    reference={"lightning.py": PROJECT_DIR / "reference" / "lightning.py"},
    chapter_steps={1: 4, 2: 3, 3: 2, 4: 4, 5: 4},
)
frag = SPEC.frag

# --- module head -----------------------------------------------------------------

frag(((1, 1), '"""Branching lightning: recursive bolts, blue afterglow, storm flashes."""'))
frag(((1, 3), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants ---------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((3, 2), "FADE = 0.90"))
frag(((4, 2), "GLOW_FADE = 0.96"))
frag(((4, 3), "GLOW_SPREAD = 0.2"))
frag(((5, 1), "FLASH_FADE = 0.85"))
frag(((3, 1), "BRANCH_CHANCE = 0.35"))
frag(((5, 2), "STORM_PERIOD = 90"))

# --- module-level fields --------------------------------------------------------------

frag(
    ((1, 2), "bolt = None\npixels = None"),
    ((2, 2), "bolt = None\ndeposit = None\npixels = None"),
    ((4, 1), "bolt = None\ndeposit = None\nglow = None\nglow_next = None\npixels = None"),
)

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global bolt, pixels"),
    ((2, 2), f"def init_sim(arch=None):\n{DOC}\n    global bolt, deposit, pixels"),
    ((4, 1), f"def init_sim(arch=None):\n{DOC}\n    global bolt, deposit, glow, glow_next, pixels"),
)
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
frag(((1, 2), "    bolt = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 2), "    deposit = ti.field(ti.f32, shape=(N, N))"))
frag(((4, 1), "    glow = ti.field(ti.f32, shape=(N, N))\n    glow_next = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- pure numpy: geometry --------------------------------------------------------------

SEGMENT = '''def deposit_segment(field, p0, p1, bright):
    """Pure numpy: stamp a straight bright segment into a (n, n) array."""
    n = field.shape[0]
    length = float(np.hypot(*(p1 - p0)))
    steps = max(2, int(length * 2))
    ts = np.linspace(0.0, 1.0, steps)
    xs = np.clip(p0[0] + (p1[0] - p0[0]) * ts, 0, n - 1).astype(np.int32)
    ys = np.clip(p0[1] + (p1[1] - p0[1]) * ts, 0, n - 1).astype(np.int32)
    field[xs, ys] = np.maximum(field[xs, ys], bright)'''

frag(((1, 3), SEGMENT))

BOLT_V1 = '''def generate_bolt(n, x_frac, rng_seed=0):
    """Pure numpy + recursion: a jagged, branching bolt as a (n, n) brightness array."""
    rng = np.random.default_rng(rng_seed)
    field = np.zeros((n, n), dtype=np.float32)
    def jag(p0, p1, bright, depth):
        d = p1 - p0
        length = float(np.hypot(*d))
        if length < 8.0 or depth > 10:
            deposit_segment(field, p0, p1, bright)
            return
        mid = (p0 + p1) / 2
        perp = np.array([-d[1], d[0]]) / (length + 1e-9)
        mid = mid + perp * rng.uniform(-0.25, 0.25) * length
        jag(p0, mid, bright, depth + 1)
        jag(mid, p1, bright, depth + 1)
    start = np.array([x_frac * n, n - 1.0])
    end = np.array([x_frac * n + rng.uniform(-0.15, 0.15) * n, 0.0])
    jag(start, end, 1.0, 0)
    return field'''

BOLT_V2 = '''def generate_bolt(n, x_frac, rng_seed=0):
    """Pure numpy + recursion: a jagged, branching bolt as a (n, n) brightness array."""
    rng = np.random.default_rng(rng_seed)
    field = np.zeros((n, n), dtype=np.float32)
    def jag(p0, p1, bright, depth):
        d = p1 - p0
        length = float(np.hypot(*d))
        if length < 8.0 or depth > 10:
            deposit_segment(field, p0, p1, bright)
            return
        mid = (p0 + p1) / 2
        perp = np.array([-d[1], d[0]]) / (length + 1e-9)
        mid = mid + perp * rng.uniform(-0.25, 0.25) * length
        jag(p0, mid, bright, depth + 1)
        jag(mid, p1, bright, depth + 1)
        if depth <= 4 and rng.random() < BRANCH_CHANCE:
            dirv = mid - p0
            ang = rng.uniform(-0.7, 0.7)
            ca, sa = np.cos(ang), np.sin(ang)
            rot = np.array([dirv[0] * ca - dirv[1] * sa, dirv[0] * sa + dirv[1] * ca])
            jag(mid, mid + rot * 0.7, bright * 0.45, depth + 1)
    start = np.array([x_frac * n, n - 1.0])
    end = np.array([x_frac * n + rng.uniform(-0.15, 0.15) * n, 0.0])
    jag(start, end, 1.0, 0)
    return field'''

frag(((2, 1), BOLT_V1), ((3, 1), BOLT_V2))

# --- GPU pipeline -----------------------------------------------------------------------

frag(
    (
        (2, 2),
        "@ti.kernel\ndef absorb():\n    for i, j in bolt:\n"
        "        bolt[i, j] = ti.max(bolt[i, j], deposit[i, j])",
    ),
    (
        (4, 2),
        "@ti.kernel\ndef absorb():\n    for i, j in bolt:\n"
        "        bolt[i, j] = ti.max(bolt[i, j], deposit[i, j])\n"
        "        glow[i, j] += deposit[i, j]",
    ),
)

frag(((2, 2), "def strike(x_frac, rng_seed=0):\n    deposit.from_numpy(generate_bolt(N, x_frac, rng_seed))\n    absorb()"))

frag(
    ((3, 2), "@ti.kernel\ndef fade():\n    for i, j in bolt:\n        bolt[i, j] *= FADE"),
    (
        (4, 2),
        "@ti.kernel\ndef fade():\n    for i, j in bolt:\n"
        "        bolt[i, j] *= FADE\n        glow[i, j] *= GLOW_FADE",
    ),
)

DIFFUSE = """@ti.kernel
def diffuse_glow():
    for i, j in glow:
        lap = (
            glow[(i + 1) % N, j]
            + glow[(i - 1) % N, j]
            + glow[i, (j + 1) % N]
            + glow[i, (j - 1) % N]
            - 4.0 * glow[i, j]
        )
        glow_next[i, j] = glow[i, j] + GLOW_SPREAD * lap"""

frag(((4, 3), DIFFUSE))
frag(((4, 3), "@ti.kernel\ndef copy_glow():\n    for i, j in glow:\n        glow[i, j] = glow_next[i, j]"))

frag(((5, 3), "@ti.kernel\ndef clear_fields():\n    for i, j in bolt:\n        bolt[i, j] = 0.0\n        glow[i, j] = 0.0"))

frag(
    ((3, 2), "def step():\n    fade()"),
    ((4, 3), "def step():\n    fade()\n    diffuse_glow()\n    copy_glow()"),
)

# --- rendering -----------------------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        pixels[i, j] = ti.math.clamp(sky + core, 0.0, 1.0)"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        g = ti.min(glow[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        halo = g * ti.Vector([0.25, 0.40, 0.95])
        pixels[i, j] = ti.math.clamp(sky + halo + core, 0.0, 1.0)"""

RENDER_V3 = """@ti.kernel
def render(flash: ti.f32):
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        g = ti.min(glow[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04]) + flash * ti.Vector([0.06, 0.08, 0.16])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        halo = g * ti.Vector([0.25, 0.40, 0.95])
        pixels[i, j] = ti.math.clamp(sky + halo + core, 0.0, 1.0)"""

frag(((1, 4), RENDER_V1), ((4, 4), RENDER_V2), ((5, 1), RENDER_V3))

# --- main loop (ordered sub-fragments) --------------------------------------------------------

MAIN_V1 = """def main():
    init_sim()
    preview = np.zeros((N, N), dtype=np.float32)
    deposit_segment(preview, np.array([N / 2, N - 1.0]), np.array([N / 2, 0.0]), 1.0)
    bolt.from_numpy(preview)"""

frag(((1, 4), MAIN_V1), ((2, 3), "def main():\n    init_sim()"))
frag(((1, 4), '    gui = ti.GUI("Lightning — taichi-academy", res=(N, N))'))
frag(((5, 2), "    storm_on = True"))
frag(((5, 1), "    flash = 0.0"))
frag(((5, 2), "    frame = 0"))
frag(((1, 4), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))"""

EVENTS_V3 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0"""

EVENTS_V4 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                storm_on = not storm_on
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0"""

EVENTS_V5 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
            elif e.key == ti.GUI.SPACE:
                storm_on = not storm_on
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0'''

frag(((1, 4), EVENTS_V1), ((2, 3), EVENTS_V2), ((5, 1), EVENTS_V3), ((5, 2), EVENTS_V4), ((5, 3), EVENTS_V5))

frag(
    (
        (5, 2),
        "        if storm_on and frame % STORM_PERIOD == 0:\n"
        "            strike(np.random.random(), np.random.randint(1_000_000))\n"
        "            flash = 1.0",
    )
)
frag(((3, 2), "        step()"))
frag(((5, 2), "        frame += 1"))

SHOW_V1 = """        render()
        gui.set_image(pixels)
        gui.show()"""

SHOW_V2 = """        render(flash)
        flash *= FLASH_FADE
        gui.set_image(pixels)
        gui.show()"""

SHOW_V3 = '''        render(flash)
        flash *= FLASH_FADE
        gui.set_image(pixels)
        storm = "on" if storm_on else "off"
        gui.text(f"storm: {storm}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("click to strike  [space] storm  [r] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()'''

frag(((1, 4), SHOW_V1), ((5, 1), SHOW_V2), ((5, 4), SHOW_V3))

frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

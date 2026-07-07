"""Code SOT for project 01 — Gray-Scott reaction-diffusion.

Fragments are listed in FINAL document order; each version is keyed by
(chapter, step). Later versions REPLACE earlier ones (the learner edits the
existing block). Run `python tools/build_fulls.py --project 01-reaction-diffusion`
to verify every step compiles and the final assembly equals the reference.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="01-reaction-diffusion",
    default_file="gray_scott.py",
    reference={"gray_scott.py": PROJECT_DIR / "reference" / "gray_scott.py"},
    chapter_steps={1: 4, 2: 5, 3: 4, 4: 4, 5: 4, 6: 2, 7: 5},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------

frag(((1, 1), '"""Gray-Scott reaction-diffusion: two chemicals paint living patterns."""'))
frag(((2, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((3, 1), "DU = 1.0\nDV = 0.5\nDT = 1.0"))
frag(((4, 3), "SUBSTEPS = 12"))
frag(((2, 3), "SEED_SIZE = 24"))
frag(((6, 1), "BRUSH_RADIUS = 8.0"))
frag(
    ((4, 3), "FEED = 0.0545\nKILL = 0.0620"),
    (
        (5, 1),
        'PRESETS = [\n'
        '    ("coral", 0.0545, 0.0620),\n'
        '    ("mitosis", 0.0367, 0.0649),\n'
        '    ("worms", 0.0780, 0.0610),\n'
        '    ("waves", 0.0140, 0.0450),\n'
        '    ("solitons", 0.0300, 0.0600),\n'
        ']',
    ),
)
frag(
    (
        (7, 1),
        "N_STOPS = 5\n"
        "PALETTES = np.array(\n"
        "    [\n"
        "        [[0.00, 0.00, 0.05], [0.10, 0.00, 0.30], [0.80, 0.20, 0.10], [1.00, 0.70, 0.10], [1.00, 1.00, 0.90]],\n"
        "        [[0.00, 0.02, 0.08], [0.00, 0.20, 0.45], [0.00, 0.60, 0.70], [0.40, 0.90, 0.85], [0.95, 1.00, 1.00]],\n"
        "        [[0.02, 0.00, 0.05], [0.25, 0.00, 0.40], [0.10, 0.55, 0.20], [0.70, 0.95, 0.20], [1.00, 1.00, 0.75]],\n"
        "    ],\n"
        "    dtype=np.float32,\n"
        ")",
    )
)

# --- module-level fields (allocated in init_sim) ---------------------------------

frag(
    ((1, 2), "pixels = None"),
    ((2, 1), "u = None\nv = None\nu_next = None\nv_next = None\npixels = None"),
    ((7, 1), "u = None\nv = None\nu_next = None\nv_next = None\npixels = None\npal_stops = None"),
)

INIT_V1 = '''def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))'''

INIT_V2 = '''def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global u, v, u_next, v_next, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    u = ti.field(ti.f32, shape=(N, N))
    v = ti.field(ti.f32, shape=(N, N))
    u_next = ti.field(ti.f32, shape=(N, N))
    v_next = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))'''

INIT_V3 = '''def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global u, v, u_next, v_next, pixels, pal_stops
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    u = ti.field(ti.f32, shape=(N, N))
    v = ti.field(ti.f32, shape=(N, N))
    u_next = ti.field(ti.f32, shape=(N, N))
    v_next = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))
    pal_stops = ti.Vector.field(3, ti.f32, shape=(len(PALETTES), N_STOPS))
    pal_stops.from_numpy(PALETTES)'''

frag(((1, 2), INIT_V1), ((2, 2), INIT_V2), ((7, 2), INIT_V3))

# --- seeding ----------------------------------------------------------------------

SEED_V1 = '''def seed_pattern(n, size=SEED_SIZE):
    """Pure numpy: U everywhere, a square of V dropped in the center."""
    u0 = np.ones((n, n), dtype=np.float32)
    v0 = np.zeros((n, n), dtype=np.float32)
    half = size // 2
    c = n // 2
    v0[c - half : c + half, c - half : c + half] = 1.0
    u0[c - half : c + half, c - half : c + half] = 0.5
    return u0, v0'''

SEED_V2 = '''def seed_pattern(n, size=SEED_SIZE, rng_seed=0, extra_spots=4):
    """Pure numpy: U everywhere, plus a center square of V and a few random spots."""
    u0 = np.ones((n, n), dtype=np.float32)
    v0 = np.zeros((n, n), dtype=np.float32)
    half = size // 2
    c = n // 2
    v0[c - half : c + half, c - half : c + half] = 1.0
    u0[c - half : c + half, c - half : c + half] = 0.5
    rng = np.random.default_rng(rng_seed)
    for _ in range(extra_spots):
        x, y = rng.integers(half, n - half, size=2)
        v0[x - half : x + half, y - half : y + half] = 1.0
        u0[x - half : x + half, y - half : y + half] = 0.5
    return u0, v0'''

frag(((2, 3), SEED_V1), ((5, 3), SEED_V2))

frag(((2, 4), "def apply_seed(seed):\n    u0, v0 = seed\n    u.from_numpy(u0)\n    v.from_numpy(v0)"))

# --- physics kernels ---------------------------------------------------------------

LAPLACIAN = """@ti.func
def laplacian(f: ti.template(), i, j):
    side = f[(i + 1) % N, j] + f[(i - 1) % N, j] + f[i, (j + 1) % N] + f[i, (j - 1) % N]
    corner = (
        f[(i + 1) % N, (j + 1) % N]
        + f[(i + 1) % N, (j - 1) % N]
        + f[(i - 1) % N, (j + 1) % N]
        + f[(i - 1) % N, (j - 1) % N]
    )
    return 0.2 * side + 0.05 * corner - f[i, j]"""

frag(((3, 1), LAPLACIAN))

UPDATE_V1 = """@ti.kernel
def update():
    for i, j in u:
        u_next[i, j] = u[i, j] + DT * DU * laplacian(u, i, j)
        v_next[i, j] = v[i, j] + DT * DV * laplacian(v, i, j)"""

UPDATE_V2 = """@ti.kernel
def update(feed: ti.f32, kill: ti.f32):
    for i, j in u:
        reaction = u[i, j] * v[i, j] * v[i, j]
        u_next[i, j] = u[i, j] + DT * (DU * laplacian(u, i, j) - reaction + feed * (1.0 - u[i, j]))
        v_next[i, j] = v[i, j] + DT * (DV * laplacian(v, i, j) + reaction - (feed + kill) * v[i, j])"""

frag(((3, 2), UPDATE_V1), ((4, 1), UPDATE_V2))

frag(((3, 3), "@ti.kernel\ndef copy_back():\n    for i, j in u:\n        u[i, j] = u_next[i, j]\n        v[i, j] = v_next[i, j]"))

frag(
    ((3, 3), "def step():\n    update()\n    copy_back()"),
    ((4, 2), "def step(feed, kill):\n    update(feed, kill)\n    copy_back()"),
)

SPLAT = """@ti.kernel
def splat(x: ti.f32, y: ti.f32, radius: ti.f32):
    for i, j in v:
        dx = i - x * N
        dy = j - y * N
        if dx * dx + dy * dy < radius * radius:
            v[i, j] = 1.0
            u[i, j] = 0.5"""

frag(((6, 1), SPLAT))

# --- rendering: gradient -> grayscale U -> palette-mapped V -------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([i / N, j / N, 0.3])"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([u[i, j], u[i, j], u[i, j]])"""

RENDER_V3 = """@ti.kernel
def render(pal: ti.i32):
    for i, j in pixels:
        t = ti.math.clamp(v[i, j] / 0.4, 0.0, 1.0)
        x = t * (N_STOPS - 1)
        s = ti.min(int(x), N_STOPS - 2)
        f = x - s
        pixels[i, j] = pal_stops[pal, s] * (1.0 - f) + pal_stops[pal, s + 1] * f"""

frag(((1, 3), RENDER_V1), ((2, 5), RENDER_V2), ((7, 3), RENDER_V3))

# --- main loop (decomposed into ordered sub-fragments) -------------------------------

frag(
    ((1, 4), "def main():\n    init_sim()"),
    ((2, 5), "def main():\n    init_sim()\n    apply_seed(seed_pattern(N))"),
)
frag(((1, 4), '    gui = ti.GUI("Gray-Scott — taichi-academy", res=(N, N))'))
frag(((5, 1), "    preset = 0"), ((7, 4), "    preset = 0\n    pal = 0"))
frag(((1, 4), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key in "12345":
                preset = int(e.key) - 1"""

EVENTS_V3 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "p":
                pal = (pal + 1) % len(PALETTES)
            elif e.key in "12345":
                preset = int(e.key) - 1"""

frag(((1, 4), EVENTS_V1), ((5, 4), EVENTS_V2), ((7, 4), EVENTS_V3))

frag(((6, 2), "        if gui.is_pressed(ti.GUI.LMB):\n            mx, my = gui.get_cursor_pos()\n            splat(mx, my, BRUSH_RADIUS)"))

frag(
    ((3, 4), "        step()"),
    ((4, 4), "        for _ in range(SUBSTEPS):\n            step(FEED, KILL)"),
    ((5, 2), "        name, feed, kill = PRESETS[preset]\n        for _ in range(SUBSTEPS):\n            step(feed, kill)"),
)

SHOW_V1 = """        render()
        gui.set_image(pixels)
        gui.show()"""

SHOW_V2 = '''        render(pal)
        gui.set_image(pixels)
        gui.text(f"{name}  F={feed:.4f} k={kill:.4f}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1-5] preset  [r] reseed  [p] palette  paint with mouse", (0.02, 0.94), color=0xAAAAAA)
        gui.show()'''

frag(((1, 4), SHOW_V1), ((7, 5), SHOW_V2))

frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

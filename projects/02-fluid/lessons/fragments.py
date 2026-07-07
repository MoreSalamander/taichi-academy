"""Code SOT for project 02 — stable-fluids ink box.

Fragments in FINAL document order; versions keyed by (chapter, step); later
versions REPLACE earlier ones. Verify with
`python tools/build_fulls.py --project 02-fluid`.

Notable evolutions: init_sim is decomposed (header + init block + one fragment
per field line) so chapters add fields without retyping the function; the
fill_vortex training-wheels kernel from ch2 EVOLVES into clear_fields in ch3;
main's header gains fill_vortex() in ch2 and loses it again in ch3.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="02-fluid",
    default_file="fluid.py",
    reference={"fluid.py": PROJECT_DIR / "reference" / "fluid.py"},
    chapter_steps={1: 4, 2: 5, 3: 5, 4: 6, 5: 4, 6: 3},
)
frag = SPEC.frag

# --- module head -----------------------------------------------------------------

frag(((1, 1), '"""Stable fluids: stir a box of incompressible ink with your mouse."""'))
frag(((1, 3), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants ---------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((2, 4), "DT = 1.0"))
frag(((4, 5), "JACOBI_ITERS = 40"))
frag(((3, 4), "DYE_DECAY = 0.995\nVEL_DECAY = 0.999"))
frag(((3, 1), "BRUSH_RADIUS = 14.0\nFORCE_SCALE = 300.0"))
frag(((5, 3), "CURL_STRENGTH = 2.0"))
frag(
    (
        (6, 1),
        'DYE_COLORS = [\n'
        '    ("ember", 1.00, 0.35, 0.10),\n'
        '    ("sky", 0.15, 0.55, 1.00),\n'
        '    ("mint", 0.20, 1.00, 0.45),\n'
        '    ("violet", 0.70, 0.30, 1.00),\n'
        '    ("gold", 1.00, 0.85, 0.25),\n'
        ']',
    )
)

# --- module-level fields -------------------------------------------------------------

frag(
    ((1, 2), "dye = None\npixels = None"),
    ((2, 1), "vel = None\ndye = None\ndye_next = None\npixels = None"),
    ((3, 3), "vel = None\nvel_next = None\ndye = None\ndye_next = None\npixels = None"),
    (
        (4, 1),
        "vel = None\nvel_next = None\ndye = None\ndye_next = None\n"
        "pressure = None\npressure_next = None\ndivergence = None\npixels = None",
    ),
    (
        (5, 1),
        "vel = None\nvel_next = None\ndye = None\ndye_next = None\n"
        "pressure = None\npressure_next = None\ndivergence = None\ncurl = None\npixels = None",
    ),
)

# init_sim: header (global line grows) + init block + one fragment per field line
DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global dye, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global vel, dye, dye_next, pixels"),
    ((3, 3), f"def init_sim(arch=None):\n{DOC}\n    global vel, vel_next, dye, dye_next, pixels"),
    ((4, 1), f"def init_sim(arch=None):\n{DOC}\n    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, pixels"),
    ((5, 1), f"def init_sim(arch=None):\n{DOC}\n    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, curl, pixels"),
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
frag(((2, 1), "    vel = ti.Vector.field(2, ti.f32, shape=(N, N))"))
frag(((3, 3), "    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))"))
frag(((1, 2), "    dye = ti.Vector.field(3, ti.f32, shape=(N, N))"))
frag(((2, 1), "    dye_next = ti.Vector.field(3, ti.f32, shape=(N, N))"))
frag(
    (
        (4, 1),
        "    pressure = ti.field(ti.f32, shape=(N, N))\n"
        "    pressure_next = ti.field(ti.f32, shape=(N, N))\n"
        "    divergence = ti.field(ti.f32, shape=(N, N))",
    )
)
frag(((5, 1), "    curl = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- seeding -----------------------------------------------------------------------

SEED = '''def seed_pattern(n, rng_seed=0, blobs=3):
    """Pure numpy: a few soft ink blobs to start with."""
    dye0 = np.zeros((n, n, 3), dtype=np.float32)
    colors = [(1.0, 0.35, 0.1), (0.15, 0.55, 1.0), (0.2, 1.0, 0.45)]
    rng = np.random.default_rng(rng_seed)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sigma = n / 14.0
    for k in range(blobs):
        cx, cy = rng.integers(n // 4, 3 * n // 4, size=2)
        w = np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (sigma * sigma))
        for ch in range(3):
            dye0[:, :, ch] += w * colors[k % 3][ch]
    return dye0.clip(0.0, 1.0).astype(np.float32)'''

frag(((1, 3), SEED))
frag(((1, 3), "def apply_seed(dye0):\n    dye.from_numpy(dye0)"))

# --- sampling ----------------------------------------------------------------------

frag(
    (
        (2, 2),
        "@ti.func\n"
        "def sample(f: ti.template(), i, j):\n"
        "    return f[((i % N) + N) % N, ((j % N) + N) % N]",
    )
)

BILERP = """@ti.func
def bilerp(f: ti.template(), x, y):
    x0 = int(ti.floor(x))
    y0 = int(ti.floor(y))
    fx = x - x0
    fy = y - y0
    a = sample(f, x0, y0)
    b = sample(f, x0 + 1, y0)
    c = sample(f, x0, y0 + 1)
    d = sample(f, x0 + 1, y0 + 1)
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy"""

frag(((2, 2), BILERP))

# --- advection ----------------------------------------------------------------------

ADVECT = """@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)"""

frag(((2, 4), ADVECT))

frag(
    ((2, 4), "@ti.kernel\ndef copy_back():\n    for i, j in dye:\n        dye[i, j] = dye_next[i, j]"),
    ((3, 3), "@ti.kernel\ndef copy_back():\n    for i, j in dye:\n        dye[i, j] = dye_next[i, j]\n        vel[i, j] = vel_next[i, j]"),
)

# fill_vortex (ch2 training wheels) EVOLVES into clear_fields (ch3 reset tool)
frag(
    (
        (2, 3),
        "@ti.kernel\n"
        "def fill_vortex():\n"
        "    for i, j in vel:\n"
        "        vel[i, j] = ti.Vector([-(j - N / 2), (i - N / 2)]) * 0.01",
    ),
    (
        (3, 5),
        "@ti.kernel\n"
        "def clear_fields():\n"
        "    for i, j in dye:\n"
        "        dye[i, j] = ti.Vector([0.0, 0.0, 0.0])\n"
        "        vel[i, j] = ti.Vector([0.0, 0.0])\n"
        "        pressure[i, j] = 0.0",
    ),
)

# --- forces --------------------------------------------------------------------------

SPLAT = """@ti.kernel
def splat(x: ti.f32, y: ti.f32, fx: ti.f32, fy: ti.f32, r: ti.f32, g: ti.f32, b: ti.f32):
    for i, j in dye:
        dx = i - x * N
        dy = j - y * N
        w = ti.exp(-(dx * dx + dy * dy) / (BRUSH_RADIUS * BRUSH_RADIUS))
        dye[i, j] += w * ti.Vector([r, g, b])
        vel[i, j] += w * ti.Vector([fx, fy])"""

frag(((3, 1), SPLAT))

frag(
    (
        (3, 4),
        "@ti.kernel\ndef decay():\n    for i, j in dye:\n        dye[i, j] *= DYE_DECAY\n        vel[i, j] *= VEL_DECAY",
    )
)

# --- incompressibility ----------------------------------------------------------------

DIVERGENCE = """@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0]
            - vel[i, j][0]
            + sample(vel, i, j + 1)[1]
            - vel[i, j][1]
        )"""

frag(((4, 2), DIVERGENCE))

JACOBI = """@ti.kernel
def pressure_jacobi():
    for i, j in pressure:
        pressure_next[i, j] = (
            sample(pressure, i + 1, j)
            + sample(pressure, i - 1, j)
            + sample(pressure, i, j + 1)
            + sample(pressure, i, j - 1)
            - divergence[i, j]
        ) * 0.25"""

frag(((4, 3), JACOBI))
frag(((4, 3), "@ti.kernel\ndef copy_pressure():\n    for i, j in pressure:\n        pressure[i, j] = pressure_next[i, j]"))

GRADIENT = """@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector([
            pressure[i, j] - sample(pressure, i - 1, j),
            pressure[i, j] - sample(pressure, i, j - 1),
        ])
        vel[i, j] -= grad"""

frag(((4, 4), GRADIENT))

frag(
    (
        (4, 5),
        "def project():\n"
        "    compute_divergence()\n"
        "    for _ in range(JACOBI_ITERS):\n"
        "        pressure_jacobi()\n"
        "        copy_pressure()\n"
        "    subtract_gradient()",
    )
)

# --- vorticity confinement --------------------------------------------------------------

CURL = """@ti.kernel
def compute_curl():
    for i, j in vel:
        curl[i, j] = (
            sample(vel, i + 1, j)[1]
            - sample(vel, i - 1, j)[1]
            - sample(vel, i, j + 1)[0]
            + sample(vel, i, j - 1)[0]
        ) * 0.5"""

VORTICITY = """@ti.kernel
def apply_vorticity(strength: ti.f32):
    for i, j in vel:
        grad = ti.Vector([
            ti.abs(sample(curl, i + 1, j)) - ti.abs(sample(curl, i - 1, j)),
            ti.abs(sample(curl, i, j + 1)) - ti.abs(sample(curl, i, j - 1)),
        ]) * 0.5
        n = grad / (grad.norm() + 1e-5)
        vel[i, j] += DT * strength * curl[i, j] * ti.Vector([n[1], -n[0]])"""

frag(((5, 2), CURL))
frag(((5, 2), VORTICITY))

# --- the tick -----------------------------------------------------------------------------

frag(
    ((2, 4), "def step():\n    advect(dye, dye_next)\n    copy_back()"),
    ((3, 3), "def step():\n    advect(dye, dye_next)\n    advect(vel, vel_next)\n    copy_back()"),
    ((3, 4), "def step():\n    advect(dye, dye_next)\n    advect(vel, vel_next)\n    copy_back()\n    decay()"),
    ((4, 6), "def step():\n    advect(dye, dye_next)\n    advect(vel, vel_next)\n    copy_back()\n    project()\n    decay()"),
    (
        (5, 3),
        "def step(curl_strength):\n"
        "    advect(dye, dye_next)\n"
        "    advect(vel, vel_next)\n"
        "    copy_back()\n"
        "    if curl_strength > 0.0:\n"
        "        compute_curl()\n"
        "        apply_vorticity(curl_strength)\n"
        "    project()\n"
        "    decay()",
    ),
)

# --- rendering -----------------------------------------------------------------------------

frag(((1, 4), "@ti.kernel\ndef render():\n    for i, j in pixels:\n        pixels[i, j] = ti.math.clamp(dye[i, j], 0.0, 1.0)"))

# --- main loop (ordered sub-fragments) --------------------------------------------------------

frag(
    ((1, 4), "def main():\n    init_sim()\n    apply_seed(seed_pattern(N))"),
    ((2, 3), "def main():\n    init_sim()\n    apply_seed(seed_pattern(N))\n    fill_vortex()"),
    ((3, 5), "def main():\n    init_sim()\n    apply_seed(seed_pattern(N))"),
)
frag(((1, 4), '    gui = ti.GUI("Stable Fluids — taichi-academy", res=(N, N))'))
frag(((6, 1), "    color_idx = 0"))
frag(((5, 3), "    curls_on = True"))
frag(((3, 2), "    pmx, pmy = 0.0, 0.0\n    dragging = False"))
frag(((1, 4), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "v":
                curls_on = not curls_on"""

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "c":
                color_idx = (color_idx + 1) % len(DYE_COLORS)
            elif e.key == "v":
                curls_on = not curls_on'''

frag(((1, 4), EVENTS_V1), ((5, 4), EVENTS_V2), ((6, 2), EVENTS_V3))

MOUSE_V1 = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                splat(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE, 1.0, 0.35, 0.1)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False"""

MOUSE_V2 = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                name, r, g, b = DYE_COLORS[color_idx]
                splat(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE, r, g, b)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False"""

frag(((3, 2), MOUSE_V1), ((6, 1), MOUSE_V2))

frag(
    ((2, 5), "        step()"),
    ((5, 3), "        step(CURL_STRENGTH if curls_on else 0.0)"),
)

SHOW_V1 = """        render()
        gui.set_image(pixels)
        gui.show()"""

SHOW_V2 = '''        render()
        gui.set_image(pixels)
        name = DYE_COLORS[color_idx][0]
        curls = "on" if curls_on else "off"
        gui.text(f"dye: {name}  curls: {curls}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to stir  [c] color  [v] curls  [r] reset", (0.02, 0.94), color=0xAAAAAA)
        gui.show()'''

frag(((1, 4), SHOW_V1), ((6, 3), SHOW_V2))

frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

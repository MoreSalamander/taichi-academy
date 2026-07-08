"""Code SOT for project 03 — fire & smoke.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 03-fire`.

Evolutions: render gains the smoke shroud in ch4; burn_source/cool/copy_back
each gain their smoke lines in ch4; init_sim grows per-field like project 02
(its global statement ends as TWO lines in ch5); step accretes the whole
algorithm one call at a time across ch2-ch5.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="03-fire",
    default_file="fire.py",
    reference={"fire.py": PROJECT_DIR / "reference" / "fire.py"},
    chapter_steps={1: 4, 2: 5, 3: 3, 4: 3, 5: 6, 6: 3},
)
frag = SPEC.frag

# --- module head -----------------------------------------------------------------

frag(((1, 1), '"""Fire and smoke: heat rises, flames lick, smoke shrouds the glow."""'))
frag(((1, 3), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants ---------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((2, 3), "DT = 1.0"))
frag(((2, 4), "BUOYANCY = 0.05"))
frag(((3, 3), "COOLING = 0.985"))
frag(((4, 1), "SMOKE_DECAY = 0.992"))
frag(((3, 3), "VEL_DECAY = 0.99"))
frag(((3, 1), "SOURCE_RADIUS = 40.0"))
frag(((6, 1), "TORCH_RADIUS = 10.0\nFORCE_SCALE = 300.0"))
frag(((5, 3), "JACOBI_ITERS = 40"))
frag(((5, 4), "CURL_STRENGTH = 2.0"))

# --- module-level fields --------------------------------------------------------------

frag(
    ((1, 2), "temp = None\npixels = None"),
    ((2, 1), "vel = None\nvel_next = None\ntemp = None\ntemp_next = None\npixels = None"),
    (
        (4, 1),
        "vel = None\nvel_next = None\ntemp = None\ntemp_next = None\n"
        "smoke = None\nsmoke_next = None\npixels = None",
    ),
    (
        (5, 1),
        "vel = None\nvel_next = None\ntemp = None\ntemp_next = None\n"
        "smoke = None\nsmoke_next = None\n"
        "pressure = None\npressure_next = None\ndivergence = None\ncurl = None\npixels = None",
    ),
)

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global temp, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global vel, vel_next, temp, temp_next, pixels"),
    ((4, 1), f"def init_sim(arch=None):\n{DOC}\n    global vel, vel_next, temp, temp_next, smoke, smoke_next, pixels"),
    (
        (5, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global vel, vel_next, temp, temp_next, smoke, smoke_next\n"
        "    global pressure, pressure_next, divergence, curl, pixels",
    ),
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
frag(((2, 1), "    vel = ti.Vector.field(2, ti.f32, shape=(N, N))\n    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))"))
frag(((1, 2), "    temp = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 1), "    temp_next = ti.field(ti.f32, shape=(N, N))"))
frag(((4, 1), "    smoke = ti.field(ti.f32, shape=(N, N))\n    smoke_next = ti.field(ti.f32, shape=(N, N))"))
frag(
    (
        (5, 1),
        "    pressure = ti.field(ti.f32, shape=(N, N))\n"
        "    pressure_next = ti.field(ti.f32, shape=(N, N))\n"
        "    divergence = ti.field(ti.f32, shape=(N, N))\n"
        "    curl = ti.field(ti.f32, shape=(N, N))",
    )
)
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- seeding -----------------------------------------------------------------------

SEED = '''def seed_pattern(n, rng_seed=0):
    """Pure numpy: one hot ember blob low in the box."""
    rng = np.random.default_rng(rng_seed)
    cx = n // 2 + int(rng.integers(-n // 8, n // 8))
    cy = n // 4
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sigma = n / 16.0
    t0 = np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (sigma * sigma))
    return t0.astype(np.float32)'''

frag(((1, 3), SEED))
frag(((1, 3), "def apply_seed(t0):\n    temp.from_numpy(t0)"))

# --- sampling (retyped from project 02 — reinforcement) ------------------------------

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

ADVECT = """@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)"""

frag(((2, 3), ADVECT))

frag(
    (
        (2, 3),
        "@ti.kernel\ndef copy_back():\n    for i, j in temp:\n"
        "        temp[i, j] = temp_next[i, j]\n        vel[i, j] = vel_next[i, j]",
    ),
    (
        (4, 2),
        "@ti.kernel\ndef copy_back():\n    for i, j in temp:\n"
        "        temp[i, j] = temp_next[i, j]\n        smoke[i, j] = smoke_next[i, j]\n"
        "        vel[i, j] = vel_next[i, j]",
    ),
)

frag(
    (
        (6, 3),
        "@ti.kernel\ndef clear_fields():\n    for i, j in temp:\n"
        "        temp[i, j] = 0.0\n        smoke[i, j] = 0.0\n"
        "        vel[i, j] = ti.Vector([0.0, 0.0])\n        pressure[i, j] = 0.0",
    )
)

# --- fire physics -------------------------------------------------------------------

frag(
    (
        (2, 4),
        "@ti.kernel\ndef apply_buoyancy():\n    for i, j in vel:\n"
        "        vel[i, j][1] += DT * BUOYANCY * temp[i, j]",
    )
)

BURN_V1 = """@ti.kernel
def burn_source(t: ti.f32):
    for i, j in temp:
        dx = i - N / 2
        dy = j - 12.0
        flick = 1.0 + 0.35 * ti.sin(0.31 * t + 0.05 * i)
        w = ti.exp(-(dx * dx + dy * dy) / (SOURCE_RADIUS * SOURCE_RADIUS)) * flick
        temp[i, j] = ti.min(temp[i, j] + 0.8 * w, 1.5)"""

BURN_V2 = """@ti.kernel
def burn_source(t: ti.f32):
    for i, j in temp:
        dx = i - N / 2
        dy = j - 12.0
        flick = 1.0 + 0.35 * ti.sin(0.31 * t + 0.05 * i)
        w = ti.exp(-(dx * dx + dy * dy) / (SOURCE_RADIUS * SOURCE_RADIUS)) * flick
        temp[i, j] = ti.min(temp[i, j] + 0.8 * w, 1.5)
        smoke[i, j] = ti.min(smoke[i, j] + 0.03 * w, 1.0)"""

frag(((3, 1), BURN_V1), ((4, 2), BURN_V2))

TORCH = """@ti.kernel
def torch(x: ti.f32, y: ti.f32, fx: ti.f32, fy: ti.f32):
    for i, j in temp:
        dx = i - x * N
        dy = j - y * N
        w = ti.exp(-(dx * dx + dy * dy) / (TORCH_RADIUS * TORCH_RADIUS))
        temp[i, j] = ti.min(temp[i, j] + 0.9 * w, 1.5)
        smoke[i, j] = ti.min(smoke[i, j] + 0.05 * w, 1.0)
        vel[i, j] += w * ti.Vector([fx, fy])"""

frag(((6, 1), TORCH))

frag(
    (
        (3, 3),
        "@ti.kernel\ndef cool():\n    for i, j in temp:\n"
        "        temp[i, j] *= COOLING\n        vel[i, j] *= VEL_DECAY",
    ),
    (
        (4, 1),
        "@ti.kernel\ndef cool():\n    for i, j in temp:\n"
        "        temp[i, j] *= COOLING\n        smoke[i, j] *= SMOKE_DECAY\n"
        "        vel[i, j] *= VEL_DECAY",
    ),
)

# --- incompressibility + confinement (retyped from project 02) -------------------------

DIVERGENCE = """@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0]
            - vel[i, j][0]
            + sample(vel, i, j + 1)[1]
            - vel[i, j][1]
        )"""

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

GRADIENT = """@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector([
            pressure[i, j] - sample(pressure, i - 1, j),
            pressure[i, j] - sample(pressure, i, j - 1),
        ])
        vel[i, j] -= grad"""

frag(((5, 2), DIVERGENCE))
frag(((5, 2), JACOBI))
frag(((5, 2), "@ti.kernel\ndef copy_pressure():\n    for i, j in pressure:\n        pressure[i, j] = pressure_next[i, j]"))
frag(((5, 3), GRADIENT))
frag(
    (
        (5, 3),
        "def project():\n"
        "    compute_divergence()\n"
        "    for _ in range(JACOBI_ITERS):\n"
        "        pressure_jacobi()\n"
        "        copy_pressure()\n"
        "    subtract_gradient()",
    )
)

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

frag(((5, 4), CURL))
frag(((5, 4), VORTICITY))

# --- the tick -----------------------------------------------------------------------------

frag(
    ((2, 3), "def step():\n    advect(temp, temp_next)\n    advect(vel, vel_next)\n    copy_back()"),
    ((2, 4), "def step():\n    advect(temp, temp_next)\n    advect(vel, vel_next)\n    copy_back()\n    apply_buoyancy()"),
    ((3, 3), "def step():\n    advect(temp, temp_next)\n    advect(vel, vel_next)\n    copy_back()\n    apply_buoyancy()\n    cool()"),
    (
        (4, 2),
        "def step():\n    advect(temp, temp_next)\n    advect(smoke, smoke_next)\n"
        "    advect(vel, vel_next)\n    copy_back()\n    apply_buoyancy()\n    cool()",
    ),
    (
        (5, 5),
        "def step(curl_strength):\n    advect(temp, temp_next)\n    advect(smoke, smoke_next)\n"
        "    advect(vel, vel_next)\n    copy_back()\n    apply_buoyancy()\n"
        "    if curl_strength > 0.0:\n        compute_curl()\n        apply_vorticity(curl_strength)\n"
        "    project()\n    cool()",
    ),
)

# --- rendering -----------------------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        t = ti.math.clamp(temp[i, j], 0.0, 1.0)
        pixels[i, j] = ti.math.clamp(ti.Vector([1.6 * t, 1.2 * t * t, t * t * t]), 0.0, 1.0)"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        t = ti.math.clamp(temp[i, j], 0.0, 1.0)
        fire = ti.Vector([1.6 * t, 1.2 * t * t, t * t * t])
        s = smoke[i, j] * 0.25
        pixels[i, j] = ti.math.clamp(fire + ti.Vector([s, s, s]), 0.0, 1.0)"""

frag(((1, 4), RENDER_V1), ((4, 3), RENDER_V2))

# --- main loop (ordered sub-fragments) --------------------------------------------------------

frag(((1, 4), "def main():\n    init_sim()\n    apply_seed(seed_pattern(N))"))
frag(((1, 4), '    gui = ti.GUI("Fire & Smoke — taichi-academy", res=(N, N))'))
frag(((3, 2), "    fire_on = True"))
frag(((5, 5), "    curls_on = True"))
frag(((3, 2), "    frame = 0"))
frag(((6, 2), "    pmx, pmy = 0.0, 0.0\n    dragging = False"))
frag(((1, 4), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on"""

EVENTS_V3 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on
            elif e.key == "v":
                curls_on = not curls_on"""

EVENTS_V4 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on
            elif e.key == "v":
                curls_on = not curls_on'''

frag(((1, 4), EVENTS_V1), ((3, 2), EVENTS_V2), ((5, 6), EVENTS_V3), ((6, 3), EVENTS_V4))

TORCH_BLOCK = """        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                torch(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE)
            else:
                torch(mx, my, 0.0, 0.0)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False"""

frag(((6, 2), TORCH_BLOCK))

frag(((3, 2), "        if fire_on:\n            burn_source(float(frame))"))
frag(
    ((2, 5), "        step()"),
    ((5, 5), "        step(CURL_STRENGTH if curls_on else 0.0)"),
)
frag(((3, 2), "        frame += 1"))

SHOW_V1 = """        render()
        gui.set_image(pixels)
        gui.show()"""

SHOW_V2 = '''        render()
        gui.set_image(pixels)
        bonfire = "lit" if fire_on else "out"
        curls = "on" if curls_on else "off"
        gui.text(f"bonfire: {bonfire}  curls: {curls}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to torch  [space] bonfire  [v] curls  [r] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()'''

frag(((1, 4), SHOW_V1), ((6, 3), SHOW_V2))

frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

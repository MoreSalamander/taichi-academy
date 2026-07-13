"""Code SOT for project 14 — galaxy creator.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 14-galaxy-creator`.

Evolutions: the render pipeline grows call by call in step() — splat only
(chapter 1: a static scatter of stars), +rotate (chapter 2: motion but the
canvas smears to solid white), +fade/clamp (chapter 2's fix). seed_galaxy's
dispatch and the two extra galaxy types arrive together in chapter 3, and
main()'s event loop grows the 1/2/3 keys with them.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="14-galaxy-creator",
    default_file="galaxy.py",
    reference={"galaxy.py": PROJECT_DIR / "reference" / "galaxy.py"},
    chapter_steps={1: 4, 2: 3, 3: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Galaxy Creator: star particles on spiral arms, differential rotation, additive light."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "N_STARS = 60000"))

frag(
    ((1, 2), "SPIRAL = 0"),
    ((3, 1), "SPIRAL, ELLIPTICAL, RING = 0, 1, 2"),
)
frag(((3, 3), 'NAMES = {SPIRAL: "spiral", ELLIPTICAL: "elliptical", RING: "ring"}'))

frag(((2, 2), "FADE = 0.88"))
frag(((1, 4), "SPLAT_GAIN = 0.35"))
frag(((1, 4), "DISK_SCALE = 0.55"))
frag(((2, 1), "ROT_SPEED = 0.35"))
frag(((2, 1), "ROT_SOFTEN = 0.05"))
frag(((2, 1), "DT = 0.016"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "radius_f = None"))
frag(((1, 2), "angle_f = None"))
frag(((1, 2), "color_f = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global radius_f, angle_f, color_f, pixels"))
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
frag(((1, 2), "    radius_f = ti.field(ti.f32, shape=N_STARS)"))
frag(((1, 2), "    angle_f = ti.field(ti.f32, shape=N_STARS)"))
frag(((1, 2), "    color_f = ti.Vector.field(3, ti.f32, shape=N_STARS)"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- pure numpy generation ---------------------------------------------------------

STAR_COLORS = '''def star_colors(r, rng, core_scale=0.12, core_col=(0.7, 0.6, 0.4), arm_col=(0.5, 0.6, 1.0)):
    """Pure numpy: blend core color to arm color by radius, dimmed per-star at random."""
    n = len(r)
    core = np.exp(-r / core_scale)
    col = np.zeros((n, 3), dtype=np.float32)
    for ch in range(3):
        col[:, ch] = core_col[ch] * core + arm_col[ch] * (1 - core)
    brightness = rng.uniform(0.3, 1.0, n) ** 2
    return (col * brightness[:, None]).astype(np.float32)'''

frag(((1, 3), STAR_COLORS))

DISK_RADII = '''def disk_radii(n, rng, scale=0.18, r_min=0.01, r_max=0.85):
    """Pure numpy: exponential-falloff radii, re-rolling any that land outside the disk."""
    r = rng.exponential(scale, n)
    bad = (r < r_min) | (r > r_max)
    r[bad] = rng.uniform(r_min, r_max, bad.sum())
    return r.astype(np.float32)'''

frag(((1, 3), DISK_RADII))

SEED_SPIRAL = '''def seed_spiral(n, rng_seed=0, arms=2, twist=3.5):
    """Pure numpy: stars scattered along logarithmic spiral arms."""
    rng = np.random.default_rng(rng_seed)
    r = disk_radii(n, rng)
    arm = rng.integers(0, arms, n)
    theta = arm * (2 * np.pi / arms) + twist * np.log(r / 0.01)
    theta = theta + rng.normal(0, 0.25, n) * (0.3 + r)
    return r, theta.astype(np.float32), star_colors(r, rng)'''

frag(((1, 3), SEED_SPIRAL))

SEED_ELLIPTICAL = '''def seed_elliptical(n, rng_seed=0):
    """Pure numpy: a smooth, armless, golden-old-star blob."""
    rng = np.random.default_rng(rng_seed)
    r = disk_radii(n, rng, scale=0.22)
    theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
    col = star_colors(r, rng, core_scale=0.3, core_col=(0.9, 0.75, 0.5), arm_col=(0.8, 0.6, 0.4))
    return r, theta, col'''

frag(((3, 1), SEED_ELLIPTICAL))

SEED_RING = '''def seed_ring(n, rng_seed=0):
    """Pure numpy: a thin ring of hot blue stars with a sparse old core."""
    rng = np.random.default_rng(rng_seed)
    n_core = n // 5
    n_ring = n - n_core
    r_ring = rng.normal(0.55, 0.045, n_ring)
    r_core = rng.exponential(0.06, n_core)
    r = np.clip(np.concatenate([r_ring, r_core]), 0.01, 0.85).astype(np.float32)
    theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
    col = star_colors(r, rng, core_scale=0.1, core_col=(0.9, 0.8, 0.6), arm_col=(0.4, 0.65, 1.0))
    return r, theta, col'''

frag(((3, 2), SEED_RING))

SEED_GALAXY_V1 = """def seed_galaxy(kind, rng_seed=0):
    return seed_spiral(N_STARS, rng_seed)"""

SEED_GALAXY_V2 = """def seed_galaxy(kind, rng_seed=0):
    if kind == SPIRAL:
        return seed_spiral(N_STARS, rng_seed)
    return seed_elliptical(N_STARS, rng_seed)"""

SEED_GALAXY_V3 = """def seed_galaxy(kind, rng_seed=0):
    if kind == SPIRAL:
        return seed_spiral(N_STARS, rng_seed)
    if kind == ELLIPTICAL:
        return seed_elliptical(N_STARS, rng_seed)
    return seed_ring(N_STARS, rng_seed)"""

frag(((1, 3), SEED_GALAXY_V1), ((3, 1), SEED_GALAXY_V2), ((3, 2), SEED_GALAXY_V3))

APPLY_SEED = """def apply_seed(seed):
    r, theta, col = seed
    radius_f.from_numpy(r)
    angle_f.from_numpy(theta)
    color_f.from_numpy(col)
    pixels.fill(0.0)"""

frag(((1, 3), APPLY_SEED))

# --- kernels -------------------------------------------------------------------------

ROTATE = """@ti.kernel
def rotate(dt: ti.f32):
    for s in radius_f:
        omega = ROT_SPEED / (radius_f[s] + ROT_SOFTEN)
        angle_f[s] += omega * dt"""

frag(((2, 1), ROTATE))

FADE_K = """@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE"""

frag(((2, 2), FADE_K))

SPLAT = """@ti.kernel
def splat():
    for s in radius_f:
        r = radius_f[s]
        a = angle_f[s]
        x = 0.5 + r * ti.cos(a) * DISK_SCALE
        y = 0.5 + r * ti.sin(a) * DISK_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] += color_f[s] * SPLAT_GAIN"""

frag(((1, 4), SPLAT))

CLAMP = """@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)"""

frag(((2, 2), CLAMP))

frag(
    ((1, 4), "def step(dt=0.016):\n    splat()"),
    ((2, 1), "def step(dt=DT):\n    rotate(dt)\n    splat()"),
    ((2, 2), "def step(dt=DT):\n    rotate(dt)\n    fade()\n    splat()\n    clamp_pixels()"),
)

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 4), "def main():\n    init_sim()\n    kind = SPIRAL\n    apply_seed(seed_galaxy(kind))"))
frag(((1, 4), '    gui = ti.GUI("Galaxy Creator — taichi-academy", res=RES, background_color=0x000000)'))
frag(((1, 4), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "123":
                kind = int(e.key) - 1
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "r":
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))'''

frag(((1, 4), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 3), EVENTS_V3))

frag(((1, 4), "        step()"))
frag(((1, 4), "        gui.set_image(pixels)"))

HUD_V1 = ""
frag(
    ((3, 3), '        gui.text(f"galaxy: {NAMES[kind]}", (0.02, 0.98), color=0xFFFFFF)'),
)
frag(
    ((3, 3), '        gui.text("[1] spiral  [2] elliptical  [3] ring  [r] reroll", (0.02, 0.94), color=0xAAAAAA)'),
)
frag(((1, 4), "        gui.show()"))
frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

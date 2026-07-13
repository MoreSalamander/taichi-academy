"""Code SOT for project 15 — star nursery.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 15-star-nursery`.

Evolutions: step() accretes the lifecycle one stage at a time — density+gravity
+integrate (chapter 2: pure collapse), +ignite/age (chapter 3: stars are born
but exert nothing), +radiation (chapter 4: the feedback loop closes). render()
has two versions: gas-only (chapter 2) and gas+star-glow (chapter 3). The HUD
star counter arrives with chapter 4's polish.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="15-star-nursery",
    default_file="star_nursery.py",
    reference={"star_nursery.py": PROJECT_DIR / "reference" / "star_nursery.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Star Nursery: a molecular cloud collapses under its own gravity and ignites into stars."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "GRID = 128"))
frag(((1, 2), "N_GAS = 40000"))
frag(((1, 2), "MAX_STARS = 400"))

frag(((2, 2), "GRAVITY_PULL = 120.0"))
frag(((2, 3), "DAMPING = 0.96"))
frag(((3, 1), "IGNITE_DENSITY = 22.0"))
frag(((3, 1), "IGNITE_PROB = 0.001"))
frag(((4, 1), "RADIATION = 250.0"))
frag(((4, 1), "RADIATION_R = 0.05"))
frag(((2, 2), "DT = 0.004"))

frag(((1, 3), "GAS_COLOR = (0.10, 0.08, 0.20)"))
frag(((3, 2), "STAR_COLOR = (1.0, 0.9, 0.7)"))
frag(((1, 3), "CANVAS_FADE = 0.85"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "vel = None"))
frag(((1, 2), "alive = None"))
frag(((1, 2), "star_pos = None"))
frag(((1, 2), "star_age = None"))
frag(((1, 2), "n_stars = None"))
frag(((2, 1), "density = None"))
frag(((2, 1), "density_blur = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    (
        (1, 2),
        f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, alive, star_pos, star_age, n_stars, pixels",
    ),
    (
        (2, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, vel, alive, star_pos, star_age, n_stars, density, density_blur, pixels",
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
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=N_GAS)"))
frag(((1, 2), "    vel = ti.Vector.field(2, ti.f32, shape=N_GAS)"))
frag(((1, 2), "    alive = ti.field(ti.i32, shape=N_GAS)"))
frag(((1, 2), "    star_pos = ti.Vector.field(2, ti.f32, shape=MAX_STARS)"))
frag(((1, 2), "    star_age = ti.field(ti.f32, shape=MAX_STARS)"))
frag(((1, 2), "    n_stars = ti.field(ti.i32, shape=())"))
frag(((2, 1), "    density = ti.field(ti.f32, shape=(GRID, GRID))"))
frag(((2, 1), "    density_blur = ti.field(ti.f32, shape=(GRID, GRID))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- seeding -------------------------------------------------------------------------

SEED_GAS = '''def seed_gas(n, rng_seed=0, blobs=4):
    """Pure numpy: a few overlapping gaussian gas clouds."""
    rng = np.random.default_rng(rng_seed)
    centers = rng.uniform(0.25, 0.75, size=(blobs, 2))
    which = rng.integers(0, blobs, n)
    p = centers[which] + rng.normal(0, 0.09, size=(n, 2))
    return np.clip(p, 0.02, 0.98).astype(np.float32)'''

frag(((1, 3), SEED_GAS))

APPLY_SEED = """def apply_seed(rng_seed=0):
    pos.from_numpy(seed_gas(N_GAS, rng_seed))
    vel.fill(0.0)
    alive.fill(1)
    n_stars[None] = 0
    pixels.fill(0.0)"""

frag(((1, 3), APPLY_SEED))

# --- density pipeline ------------------------------------------------------------------

frag(((2, 1), "@ti.kernel\ndef clear_density():\n    for i, j in density:\n        density[i, j] = 0.0"))

DEPOSIT = """@ti.kernel
def deposit():
    for p in pos:
        if alive[p] == 1:
            gi = ti.cast(pos[p][0] * GRID, ti.i32)
            gj = ti.cast(pos[p][1] * GRID, ti.i32)
            if 0 <= gi < GRID and 0 <= gj < GRID:
                density[gi, gj] += 1.0"""

frag(((2, 1), DEPOSIT))

BLUR = """@ti.kernel
def blur():
    for i, j in density_blur:
        acc = 0.0
        cnt = 0.0
        for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                acc += density[ni, nj]
                cnt += 1.0
        density_blur[i, j] = acc / cnt"""

frag(((2, 1), BLUR))

GRAVITY = """@ti.kernel
def gravity():
    for p in pos:
        if alive[p] == 1:
            gi = ti.min(ti.max(ti.cast(pos[p][0] * GRID, ti.i32), 1), GRID - 2)
            gj = ti.min(ti.max(ti.cast(pos[p][1] * GRID, ti.i32), 1), GRID - 2)
            gx = (density_blur[gi + 1, gj] - density_blur[gi - 1, gj]) * 0.5
            gy = (density_blur[gi, gj + 1] - density_blur[gi, gj - 1]) * 0.5
            vel[p] += DT * GRAVITY_PULL * ti.Vector([gx, gy]) / GRID"""

frag(((2, 2), GRAVITY))

RADIATION_K = """@ti.kernel
def radiation():
    for p in pos:
        if alive[p] == 1:
            f = ti.Vector([0.0, 0.0])
            for s in range(n_stars[None]):
                d = pos[p] - star_pos[s]
                r2 = d.dot(d)
                if r2 < RADIATION_R * RADIATION_R:
                    r = ti.sqrt(r2) + 1e-4
                    f += RADIATION * (1.0 - r / RADIATION_R) * d / r
            vel[p] += DT * f"""

frag(((4, 1), RADIATION_K))

INTEGRATE = """@ti.kernel
def integrate():
    for p in pos:
        if alive[p] == 1:
            vel[p] *= DAMPING
            newp = pos[p] + DT * vel[p]
            for a in ti.static(range(2)):
                if newp[a] < 0.01:
                    newp[a] = 0.01
                    vel[p][a] *= -0.5
                if newp[a] > 0.99:
                    newp[a] = 0.99
                    vel[p][a] *= -0.5
            pos[p] = newp"""

frag(((2, 3), INTEGRATE))

IGNITE = """@ti.kernel
def ignite():
    for p in pos:
        if alive[p] == 1 and n_stars[None] < MAX_STARS:
            gi = ti.min(ti.max(ti.cast(pos[p][0] * GRID, ti.i32), 0), GRID - 1)
            gj = ti.min(ti.max(ti.cast(pos[p][1] * GRID, ti.i32), 0), GRID - 1)
            if density_blur[gi, gj] > IGNITE_DENSITY:
                if ti.random() < IGNITE_PROB:
                    s = ti.atomic_add(n_stars[None], 1)
                    if s < MAX_STARS:
                        star_pos[s] = pos[p]
                        star_age[s] = 0.0
                        alive[p] = 0"""

frag(((3, 1), IGNITE))

frag(((3, 1), "@ti.kernel\ndef age_stars(dt: ti.f32):\n    for s in range(n_stars[None]):\n        star_age[s] += dt"))

# --- render: two versions -------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for p in pos:
        if alive[p] == 1:
            xi = ti.cast(pos[p][0] * RES, ti.i32)
            yi = ti.cast(pos[p][1] * RES, ti.i32)
            if 0 <= xi < RES and 0 <= yi < RES:
                pixels[xi, yi] += ti.Vector([GAS_COLOR[0], GAS_COLOR[1], GAS_COLOR[2]])"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for p in pos:
        if alive[p] == 1:
            xi = ti.cast(pos[p][0] * RES, ti.i32)
            yi = ti.cast(pos[p][1] * RES, ti.i32)
            if 0 <= xi < RES and 0 <= yi < RES:
                pixels[xi, yi] += ti.Vector([GAS_COLOR[0], GAS_COLOR[1], GAS_COLOR[2]])

    for s in range(n_stars[None]):
        cx = star_pos[s][0] * RES
        cy = star_pos[s][1] * RES
        glow = ti.min(star_age[s] * 2.0, 1.0)
        for di, dj in ti.ndrange((-3, 4), (-3, 4)):
            xi = ti.cast(cx, ti.i32) + di
            yi = ti.cast(cy, ti.i32) + dj
            if 0 <= xi < RES and 0 <= yi < RES:
                w = ti.exp(-(di * di + dj * dj) / 4.0)
                pixels[xi, yi] += glow * w * ti.Vector([STAR_COLOR[0], STAR_COLOR[1], STAR_COLOR[2]])"""

frag(((1, 3), RENDER_V1), ((3, 2), RENDER_V2))

CLAMP = """@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)"""

frag(((1, 3), CLAMP))

# --- the tick ----------------------------------------------------------------------

STEP_V1 = """def step():
    render()
    clamp_pixels()"""

STEP_V2 = """def step():
    clear_density()
    deposit()
    blur()
    gravity()
    integrate()
    render()
    clamp_pixels()"""

STEP_V3 = """def step():
    clear_density()
    deposit()
    blur()
    gravity()
    integrate()
    ignite()
    age_stars(DT)
    render()
    clamp_pixels()"""

STEP_V4 = """def step():
    clear_density()
    deposit()
    blur()
    gravity()
    radiation()
    integrate()
    ignite()
    age_stars(DT)
    render()
    clamp_pixels()"""

frag(((1, 3), STEP_V1), ((2, 3), STEP_V2), ((3, 2), STEP_V3), ((4, 1), STEP_V4))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Star Nursery — taichi-academy", res=RES, background_color=0x000000)'))
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

frag(((1, 3), "        step()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((4, 2), '        gui.text(f"stars born: {n_stars[None]}", (0.02, 0.98), color=0xFFFFFF)'))
frag(((4, 2), '        gui.text("[r] new cloud", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

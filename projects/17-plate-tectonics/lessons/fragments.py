"""Code SOT for project 17 — plate tectonics.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 17-plate-tectonics`.

Evolutions: boundary_forces has two versions — pure uplift/rift (chapter 2)
and +quake flashes (chapter 4). step() accretes boundary physics (ch2), drift
(ch3, gaining its frame parameter), then activity decay (ch4). init_sim,
apply_seed, and render each gain their activity-related lines in chapter 4.
The hillshade+bands render arrives fully formed in chapter 1 — it's project
05's renderer, reused as vocabulary rather than re-derived.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="17-plate-tectonics",
    default_file="plate_tectonics.py",
    reference={"plate_tectonics.py": PROJECT_DIR / "reference" / "plate_tectonics.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Plate Tectonics: voronoi plates drift, collide into mountains, tear open rifts."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "N = 256"))
frag(((1, 2), "N_PLATES = 7"))
frag(((2, 1), "DT = 0.02"))
frag(((3, 1), "DRIFT_STEP = 1.0"))
frag(((3, 1), "DRIFT_EVERY = 6"))
frag(((2, 1), "UPLIFT = 0.22"))
frag(((2, 1), "RIFT = 0.15"))
frag(((2, 2), "EROSION = 0.10"))
frag(((1, 3), "SEA = 0.48"))
frag(((3, 1), "NEW_CRUST = 0.25"))
frag(((1, 3), "RELIEF = 40.0"))

frag(((4, 1), "QUAKE_CONV = 0.8"))
frag(((4, 1), "QUAKE_PROB = 0.002"))
frag(((4, 1), "ACTIVITY_DECAY = 0.92"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "plate_id = None"))
frag(((3, 1), "plate_id_next = None"))
frag(((1, 2), "height = None"))
frag(((2, 1), "height_next = None"))
frag(((1, 2), "plate_vel = None"))
frag(((4, 1), "activity = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global plate_id, height, plate_vel, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global plate_id, height, height_next, plate_vel, pixels"),
    (
        (3, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global plate_id, plate_id_next, height, height_next, plate_vel, pixels",
    ),
    (
        (4, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global plate_id, plate_id_next, height, height_next, plate_vel, activity, pixels",
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
frag(((1, 2), "    plate_id = ti.field(ti.i32, shape=(N, N))"))
frag(((3, 1), "    plate_id_next = ti.field(ti.i32, shape=(N, N))"))
frag(((1, 2), "    height = ti.field(ti.f32, shape=(N, N))"))
frag(((2, 1), "    height_next = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    plate_vel = ti.Vector.field(2, ti.f32, shape=N_PLATES)"))
frag(((4, 1), "    activity = ti.field(ti.f32, shape=(N, N))"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- pure numpy generation ---------------------------------------------------------

VORONOI = '''def voronoi_plates(n, seeds):
    """Pure numpy: nearest-seed labeling with toroidal (wraparound) distance."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    best = np.full((n, n), 1e18)
    pid = np.zeros((n, n), dtype=np.int32)
    for k in range(len(seeds)):
        dx = np.abs(ii - seeds[k, 0])
        dy = np.abs(jj - seeds[k, 1])
        dx = np.minimum(dx, n - dx)
        dy = np.minimum(dy, n - dy)
        d = dx * dx + dy * dy
        m = d < best
        best[m] = d[m]
        pid[m] = k
    return pid'''

frag(((1, 3), VORONOI))

SEED_WORLD = '''def seed_world(rng_seed=0):
    """Pure numpy: plates from voronoi, continents vs oceans, one drift vector per plate."""
    rng = np.random.default_rng(rng_seed)
    seeds = rng.uniform(0, N, size=(N_PLATES, 2)).astype(np.float32)
    pid = voronoi_plates(N, seeds)
    is_continent = rng.random(N_PLATES) < 0.4
    h = np.where(is_continent[pid], 0.58, 0.30).astype(np.float32)
    h += rng.normal(0, 0.02, (N, N)).astype(np.float32)
    v = rng.uniform(-1, 1, size=(N_PLATES, 2)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return pid, h, v'''

frag(((1, 3), SEED_WORLD))

APPLY_V1 = """def apply_seed(rng_seed=0):
    pid, h, v = seed_world(rng_seed)
    plate_id.from_numpy(pid)
    height.from_numpy(h)
    plate_vel.from_numpy(v)"""

APPLY_V2 = APPLY_V1 + "\n    activity.fill(0.0)"

frag(((1, 3), APPLY_V1), ((4, 1), APPLY_V2))

frag(((1, 3), "@ti.func\ndef wrap(i):\n    return ((i % N) + N) % N"))

# --- boundary physics: two versions -----------------------------------------------------

FORCES_V1 = """@ti.kernel
def boundary_forces():
    for i, j in height:
        me = plate_id[i, j]
        delta = 0.0
        for k in ti.static(range(4)):
            di = (1, -1, 0, 0)[k]
            dj = (0, 0, 1, -1)[k]
            ni, nj = wrap(i + di), wrap(j + dj)
            other = plate_id[ni, nj]
            if other != me:
                rel = plate_vel[me] - plate_vel[other]
                conv = rel[0] * di + rel[1] * dj
                if conv > 0:
                    delta += UPLIFT * conv * DT * (1.0 - height[i, j])
                else:
                    delta += RIFT * conv * DT * height[i, j]
        height_next[i, j] = ti.math.clamp(height[i, j] + delta, 0.0, 1.0)"""

FORCES_V2 = """@ti.kernel
def boundary_forces():
    for i, j in height:
        me = plate_id[i, j]
        delta = 0.0
        for k in ti.static(range(4)):
            di = (1, -1, 0, 0)[k]
            dj = (0, 0, 1, -1)[k]
            ni, nj = wrap(i + di), wrap(j + dj)
            other = plate_id[ni, nj]
            if other != me:
                rel = plate_vel[me] - plate_vel[other]
                conv = rel[0] * di + rel[1] * dj
                if conv > 0:
                    delta += UPLIFT * conv * DT * (1.0 - height[i, j])
                else:
                    delta += RIFT * conv * DT * height[i, j]
                if ti.abs(conv) > QUAKE_CONV and ti.random() < QUAKE_PROB:
                    activity[i, j] = 1.0
        height_next[i, j] = ti.math.clamp(height[i, j] + delta, 0.0, 1.0)"""

frag(((2, 1), FORCES_V1), ((4, 1), FORCES_V2))

ERODE = """@ti.kernel
def erode():
    for i, j in height:
        avg = 0.25 * (
            height_next[wrap(i + 1), j]
            + height_next[wrap(i - 1), j]
            + height_next[i, wrap(j + 1)]
            + height_next[i, wrap(j - 1)]
        )
        height[i, j] = height_next[i, j] * (1 - EROSION) + avg * EROSION"""

frag(((2, 2), ERODE))

# --- drift ---------------------------------------------------------------------------

DRIFT = """@ti.kernel
def drift():
    for i, j in plate_id:
        best_h = -1.0
        best_id = -1
        for k in range(N_PLATES):
            si = wrap(i - ti.cast(ti.round(plate_vel[k][0] * DRIFT_STEP), ti.i32))
            sj = wrap(j - ti.cast(ti.round(plate_vel[k][1] * DRIFT_STEP), ti.i32))
            if plate_id[si, sj] == k:
                if height[si, sj] > best_h:
                    best_h = height[si, sj]
                    best_id = k
        if best_id >= 0:
            plate_id_next[i, j] = best_id
            height_next[i, j] = best_h
        else:
            plate_id_next[i, j] = plate_id[i, j]
            height_next[i, j] = NEW_CRUST"""

frag(((3, 1), DRIFT))

COPY_DRIFT = """@ti.kernel
def copy_drift():
    for i, j in plate_id:
        plate_id[i, j] = plate_id_next[i, j]
        height[i, j] = height_next[i, j]"""

frag(((3, 1), COPY_DRIFT))

SMOOTH = """@ti.kernel
def smooth_after_drift():
    for i, j in height:
        height_next[i, j] = height[i, j]
    for i, j in height:
        avg = 0.25 * (
            height_next[wrap(i + 1), j]
            + height_next[wrap(i - 1), j]
            + height_next[i, wrap(j + 1)]
            + height_next[i, wrap(j - 1)]
        )
        height[i, j] = height_next[i, j] * 0.5 + avg * 0.5"""

frag(((3, 2), SMOOTH))

frag(((4, 2), "@ti.kernel\ndef decay_activity():\n    for i, j in activity:\n        activity[i, j] *= ACTIVITY_DECAY"))

# --- render -------------------------------------------------------------------------

BAND = """@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t"""

frag(((1, 3), BAND))

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        hh = height[i, j]
        c = ti.Vector([0.05, 0.15, 0.4])
        if hh > SEA:
            land = (hh - SEA) / (1.0 - SEA)
            c = band(ti.Vector([0.55, 0.6, 0.3]), ti.Vector([0.45, 0.4, 0.35]), land, 0.1, 0.5)
            c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.55, 0.8)
        else:
            c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), hh / SEA, 0.3, 1.0)
        dhdx = (height[wrap(i + 1), j] - hh) * RELIEF
        dhdy = (height[i, wrap(j + 1)] - hh) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.4 + 0.6 * normal.dot(light), 0.0, 1.0)
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        hh = height[i, j]
        c = ti.Vector([0.05, 0.15, 0.4])
        if hh > SEA:
            land = (hh - SEA) / (1.0 - SEA)
            c = band(ti.Vector([0.55, 0.6, 0.3]), ti.Vector([0.45, 0.4, 0.35]), land, 0.1, 0.5)
            c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.55, 0.8)
        else:
            c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), hh / SEA, 0.3, 1.0)
        dhdx = (height[wrap(i + 1), j] - hh) * RELIEF
        dhdy = (height[i, wrap(j + 1)] - hh) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.4 + 0.6 * normal.dot(light), 0.0, 1.0)
        c = c * shade
        c += activity[i, j] * ti.Vector([1.0, 0.35, 0.05])
        pixels[i, j] = ti.math.clamp(c, 0.0, 1.0)"""

frag(((1, 3), RENDER_V1), ((4, 2), RENDER_V2))

# --- the tick ----------------------------------------------------------------------

STEP_V1 = """def step(frame):
    boundary_forces()
    erode()"""

STEP_V2 = """def step(frame):
    boundary_forces()
    erode()
    if frame % DRIFT_EVERY == 0:
        drift()
        copy_drift()
        smooth_after_drift()"""

STEP_V3 = """def step(frame):
    boundary_forces()
    erode()
    if frame % DRIFT_EVERY == 0:
        drift()
        copy_drift()
        smooth_after_drift()
    decay_activity()"""

frag(((2, 3), STEP_V1), ((3, 2), STEP_V2), ((4, 2), STEP_V3))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Plate Tectonics — taichi-academy", res=N, background_color=0x0A0A12)'))
frag(((2, 3), "    frame = 0"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))'''

frag(((1, 3), EVENTS_V1), ((4, 3), EVENTS_V2))

frag(((2, 3), "        step(frame)"))
frag(((2, 3), "        frame += 1"))
frag(((1, 3), "        render()"))
frag(((1, 3), "        gui.set_image(pixels)"))
frag(((4, 3), '        gui.text("mountains rise where plates meet", (0.02, 0.98), color=0xFFFFFF)'))
frag(((4, 3), '        gui.text("[r] new world", (0.02, 0.94), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

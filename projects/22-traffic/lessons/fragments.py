"""Code SOT for project 22 — traffic.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 22-traffic`.

Evolutions: update_velocity accretes the NaSch rules one at a time —
accelerate only (ch1: cars sail through each other), +gap braking (ch2 s1:
safe deterministic platoons), +random slowdown (ch2 s2: phantom jams are
born). build_occupancy, render_ring, step, and main's event block each grow
lights/controls versions in ch3. The space-time diagram and its [v] toggle
land at ch2 s3 — the classic visualization of the jam waves.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="22-traffic",
    default_file="traffic.py",
    reference={"traffic.py": PROJECT_DIR / "reference" / "traffic.py"},
    chapter_steps={1: 3, 2: 3, 3: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Traffic: four rules per driver, and jams appear out of nowhere — then drive backward."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "RES = 512"))
frag(((1, 2), "ROAD_LEN = 1000"))
frag(((1, 2), "MAX_CARS = 600"))
frag(((1, 2), "START_CARS = 220"))
frag(((3, 2), "CAR_STEP = 20"))
frag(((1, 2), "VMAX = 5"))
frag(((2, 2), "P_SLOW = 0.25"))

frag(((3, 1), "N_LIGHTS = 3"))
frag(((3, 1), "LIGHT_PERIOD = 120"))
frag(((3, 1), "LIGHT_GREEN = 70"))

frag(((1, 2), "RING_R = 0.4"))
frag(((1, 2), "PI2 = 6.28318530"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "car_pos = None"))
frag(((1, 2), "car_v = None"))
frag(((1, 2), "active = None"))
frag(((2, 1), "occupancy = None"))
frag(((3, 1), "light_pos = None"))
frag(((1, 2), "pixels = None"))
frag(((2, 3), "spacetime = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global car_pos, car_v, active, pixels"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global car_pos, car_v, active, occupancy, pixels"),
    ((2, 3), f"def init_sim(arch=None):\n{DOC}\n    global car_pos, car_v, active, occupancy, pixels, spacetime"),
    (
        (3, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global car_pos, car_v, active, occupancy, light_pos, pixels, spacetime",
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
frag(((1, 2), "    car_pos = ti.field(ti.i32, shape=MAX_CARS)"))
frag(((1, 2), "    car_v = ti.field(ti.i32, shape=MAX_CARS)"))
frag(((1, 2), "    active = ti.field(ti.i32, shape=MAX_CARS)"))
frag(((2, 1), "    occupancy = ti.field(ti.i32, shape=ROAD_LEN)"))
frag(((3, 1), "    light_pos = ti.field(ti.i32, shape=N_LIGHTS)"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))
frag(((2, 3), "    spacetime = ti.Vector.field(3, ti.f32, shape=(RES, RES))"))

# --- seeding -------------------------------------------------------------------------

SEED_V1 = '''def seed_road(n, rng_seed=0):
    """Pure numpy: n cars in distinct random cells, everyone stopped."""
    rng = np.random.default_rng(rng_seed)
    positions = np.sort(rng.choice(ROAD_LEN, size=n, replace=False)).astype(np.int32)
    pos_arr = np.full(MAX_CARS, -1, dtype=np.int32)
    pos_arr[:n] = positions
    car_pos.from_numpy(pos_arr)
    car_v.from_numpy(np.zeros(MAX_CARS, dtype=np.int32))
    act = np.zeros(MAX_CARS, dtype=np.int32)
    act[:n] = 1
    active.from_numpy(act)'''

SEED_V2 = SEED_V1.replace(
    '"""Pure numpy: n cars in distinct random cells, everyone stopped."""',
    '"""Pure numpy: n cars in distinct random cells, everyone stopped."""',
) + "\n    spacetime.fill(0.02)"

SEED_V3 = '''def seed_road(n, rng_seed=0):
    """Pure numpy: n cars in distinct random cells, everyone stopped, lights evenly spaced."""
    rng = np.random.default_rng(rng_seed)
    positions = np.sort(rng.choice(ROAD_LEN, size=n, replace=False)).astype(np.int32)
    pos_arr = np.full(MAX_CARS, -1, dtype=np.int32)
    pos_arr[:n] = positions
    car_pos.from_numpy(pos_arr)
    car_v.from_numpy(np.zeros(MAX_CARS, dtype=np.int32))
    act = np.zeros(MAX_CARS, dtype=np.int32)
    act[:n] = 1
    active.from_numpy(act)
    light_pos.from_numpy(np.linspace(0, ROAD_LEN, N_LIGHTS, endpoint=False).astype(np.int32))
    spacetime.fill(0.02)'''

frag(((1, 3), SEED_V1), ((2, 3), SEED_V2), ((3, 1), SEED_V3))

SET_CAR_COUNT = '''def set_car_count(n_target, rng_seed=0):
    """Pure numpy + field surgery: activate/deactivate cars to hit a target count."""
    rng = np.random.default_rng(rng_seed)
    act = active.to_numpy()
    pos_arr = car_pos.to_numpy()
    n_now = int(act.sum())
    if n_target < n_now:
        live = np.where(act == 1)[0]
        act[live[n_target:]] = 0
    elif n_target > n_now:
        occupied = set(pos_arr[act == 1].tolist())
        empty = np.array([c for c in range(ROAD_LEN) if c not in occupied])
        need = min(n_target, MAX_CARS) - n_now
        chosen = rng.choice(empty, size=need, replace=False)
        dead = np.where(act == 0)[0][:need]
        pos_arr[dead] = chosen
        act[dead] = 1
    car_pos.from_numpy(pos_arr)
    active.from_numpy(act)
    return int(act.sum())'''

frag(((3, 2), SET_CAR_COUNT))

# --- the four rules --------------------------------------------------------------------

OCC_V1 = """@ti.kernel
def build_occupancy():
    for c in occupancy:
        occupancy[c] = 0
    for i in car_pos:
        if active[i] == 1:
            occupancy[car_pos[i]] = 1"""

OCC_V2 = """@ti.kernel
def build_occupancy(t: ti.i32, lights_on: ti.i32):
    for c in occupancy:
        occupancy[c] = 0
    for i in car_pos:
        if active[i] == 1:
            occupancy[car_pos[i]] = 1
    if lights_on == 1:
        for k in range(N_LIGHTS):
            phase = (t + k * LIGHT_PERIOD // N_LIGHTS) % LIGHT_PERIOD
            if phase >= LIGHT_GREEN:
                occupancy[light_pos[k]] = 1"""

frag(((2, 1), OCC_V1), ((3, 1), OCC_V2))

UPD_V1 = """@ti.kernel
def update_velocity():
    for i in car_pos:
        if active[i] == 1:
            car_v[i] = ti.min(car_v[i] + 1, VMAX)"""

UPD_V2 = """@ti.kernel
def update_velocity():
    for i in car_pos:
        if active[i] == 1:
            v = ti.min(car_v[i] + 1, VMAX)
            gap = VMAX + 1
            for d in range(1, VMAX + 2):
                cell = (car_pos[i] + d) % ROAD_LEN
                if occupancy[cell] == 1 and gap == VMAX + 1:
                    gap = d
            v = ti.min(v, gap - 1)
            car_v[i] = v"""

UPD_V3 = """@ti.kernel
def update_velocity():
    for i in car_pos:
        if active[i] == 1:
            v = ti.min(car_v[i] + 1, VMAX)
            gap = VMAX + 1
            for d in range(1, VMAX + 2):
                cell = (car_pos[i] + d) % ROAD_LEN
                if occupancy[cell] == 1 and gap == VMAX + 1:
                    gap = d
            v = ti.min(v, gap - 1)
            if ti.random() < P_SLOW:
                v = ti.max(v - 1, 0)
            car_v[i] = v"""

frag(((1, 3), UPD_V1), ((2, 1), UPD_V2), ((2, 2), UPD_V3))

MOVE = """@ti.kernel
def move_cars():
    for i in car_pos:
        if active[i] == 1:
            car_pos[i] = (car_pos[i] + car_v[i]) % ROAD_LEN"""

frag(((1, 3), MOVE))

INCIDENT = """@ti.kernel
def incident(cell: ti.i32):
    for i in car_pos:
        if active[i] == 1:
            d = ti.abs(car_pos[i] - cell)
            d = ti.min(d, ROAD_LEN - d)
            if d < 15:
                car_v[i] = 0"""

frag(((3, 3), INCIDENT))

STEP_V1 = """def step(t):
    update_velocity()
    move_cars()"""

STEP_V2 = """def step(t):
    build_occupancy()
    update_velocity()
    move_cars()"""

STEP_V3 = """def step(t, lights_on=True):
    build_occupancy(t, 1 if lights_on else 0)
    update_velocity()
    move_cars()"""

frag(((1, 3), STEP_V1), ((2, 1), STEP_V2), ((3, 1), STEP_V3))

# --- rendering -----------------------------------------------------------------------

SPEED_COLOR = """@ti.func
def speed_color(v):
    s = v / VMAX
    return ti.Vector([1.0, 0.2, 0.15]) * (1 - s) + ti.Vector([0.3, 1.0, 0.4]) * s"""

frag(((1, 3), SPEED_COLOR))

RING_V1 = """@ti.kernel
def render_ring():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.02, 0.02, 0.04])
    for c in range(ROAD_LEN):
        ang = c / ROAD_LEN * PI2
        xi = ti.cast((0.5 + RING_R * ti.cos(ang)) * RES, ti.i32)
        yi = ti.cast((0.5 + RING_R * ti.sin(ang)) * RES, ti.i32)
        pixels[xi, yi] = ti.Vector([0.15, 0.15, 0.18])
    for i in car_pos:
        if active[i] == 1:
            ang = car_pos[i] / ROAD_LEN * PI2
            x = 0.5 + RING_R * ti.cos(ang)
            y = 0.5 + RING_R * ti.sin(ang)
            col = speed_color(car_v[i])
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                xi = ti.cast(x * RES, ti.i32) + di
                yi = ti.cast(y * RES, ti.i32) + dj
                if 0 <= xi < RES and 0 <= yi < RES:
                    pixels[xi, yi] = col"""

RING_V2 = """@ti.kernel
def render_ring(t: ti.i32, lights_on: ti.i32):
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.02, 0.02, 0.04])
    for c in range(ROAD_LEN):
        ang = c / ROAD_LEN * PI2
        xi = ti.cast((0.5 + RING_R * ti.cos(ang)) * RES, ti.i32)
        yi = ti.cast((0.5 + RING_R * ti.sin(ang)) * RES, ti.i32)
        pixels[xi, yi] = ti.Vector([0.15, 0.15, 0.18])
    if lights_on == 1:
        for k in range(N_LIGHTS):
            ang = light_pos[k] / ROAD_LEN * PI2
            x = 0.5 + (RING_R + 0.04) * ti.cos(ang)
            y = 0.5 + (RING_R + 0.04) * ti.sin(ang)
            phase = (t + k * LIGHT_PERIOD // N_LIGHTS) % LIGHT_PERIOD
            col = ti.Vector([0.1, 0.9, 0.2])
            if phase >= LIGHT_GREEN:
                col = ti.Vector([0.95, 0.15, 0.1])
            for di, dj in ti.ndrange((-2, 3), (-2, 3)):
                xi = ti.cast(x * RES, ti.i32) + di
                yi = ti.cast(y * RES, ti.i32) + dj
                if 0 <= xi < RES and 0 <= yi < RES:
                    pixels[xi, yi] = col
    for i in car_pos:
        if active[i] == 1:
            ang = car_pos[i] / ROAD_LEN * PI2
            x = 0.5 + RING_R * ti.cos(ang)
            y = 0.5 + RING_R * ti.sin(ang)
            col = speed_color(car_v[i])
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                xi = ti.cast(x * RES, ti.i32) + di
                yi = ti.cast(y * RES, ti.i32) + dj
                if 0 <= xi < RES and 0 <= yi < RES:
                    pixels[xi, yi] = col"""

frag(((1, 3), RING_V1), ((3, 1), RING_V2))

SPACETIME = """@ti.kernel
def render_spacetime():
    for i, j in spacetime:
        if j < RES - 1:
            spacetime[i, j] = spacetime[i, j + 1]
    for i in range(RES):
        spacetime[i, RES - 1] = ti.Vector([0.02, 0.02, 0.04])
    for i in car_pos:
        if active[i] == 1:
            xi = car_pos[i] * RES // ROAD_LEN
            spacetime[xi, RES - 1] = speed_color(car_v[i])"""

frag(((2, 3), SPACETIME))

MEAN_SPEED = '''def mean_speed():
    """Pure numpy: the live flow reading for the HUD."""
    act = active.to_numpy() == 1
    if act.sum() == 0:
        return 0.0
    return float(car_v.to_numpy()[act].mean())'''

frag(((3, 2), MEAN_SPEED))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    seed_road(START_CARS)"))
frag(((3, 2), "    n = START_CARS"))
frag(((1, 3), "    t = 0"))
frag(((3, 1), "    lights_on = True"))
frag(((2, 3), "    show_ring = True"))
frag(((1, 3), '    gui = ti.GUI("Traffic — taichi-academy", res=RES, background_color=0x000000)'))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "v":
                show_ring = not show_ring'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                lights_on = not lights_on
            elif e.key == "v":
                show_ring = not show_ring'''

EVENTS_V4 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                n = set_car_count(min(n + CAR_STEP, MAX_CARS), rng_seed=np.random.randint(1_000_000))
            elif e.key == ti.GUI.DOWN:
                n = set_car_count(max(n - CAR_STEP, CAR_STEP))
            elif e.key == ti.GUI.SPACE:
                lights_on = not lights_on
            elif e.key == "v":
                show_ring = not show_ring'''

EVENTS_V5 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                n = set_car_count(min(n + CAR_STEP, MAX_CARS), rng_seed=np.random.randint(1_000_000))
            elif e.key == ti.GUI.DOWN:
                n = set_car_count(max(n - CAR_STEP, CAR_STEP))
            elif e.key == ti.GUI.SPACE:
                lights_on = not lights_on
            elif e.key == "v":
                show_ring = not show_ring
            elif e.key == "r":
                seed_road(n, rng_seed=np.random.randint(1_000_000))
            elif e.key == ti.GUI.LMB:
                mx, my = gui.get_cursor_pos()
                ang = np.arctan2(my - 0.5, mx - 0.5) % PI2
                incident(int(ang / PI2 * ROAD_LEN))'''

frag(((1, 3), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 1), EVENTS_V3), ((3, 2), EVENTS_V4), ((3, 3), EVENTS_V5))

frag(((1, 3), "        step(t)"), ((3, 1), "        step(t, lights_on)"))
frag(((2, 3), "        render_spacetime()"))

DRAW_V1 = """        render_ring()
        gui.set_image(pixels)"""

DRAW_V2 = """        if show_ring:
            render_ring()
            gui.set_image(pixels)
        else:
            gui.set_image(spacetime)"""

DRAW_V3 = """        if show_ring:
            render_ring(t, 1 if lights_on else 0)
            gui.set_image(pixels)
        else:
            gui.set_image(spacetime)"""

frag(((1, 3), DRAW_V1), ((2, 3), DRAW_V2), ((3, 1), DRAW_V3))

frag(((1, 3), "        t += 1"))
frag(((3, 2), "        density = n / ROAD_LEN"))
frag(
    ((3, 2), '        gui.text(f"cars {n}  density {density:.2f}  mean speed {mean_speed():.2f}", (0.02, 0.98), color=0xFFFFFF)'),
)
frag(
    ((3, 3), '        gui.text("[up/down] density  [space] lights  [v] view  [r] reset  click: incident", (0.02, 0.94), color=0xAAAAAA)'),
)
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

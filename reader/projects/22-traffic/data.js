// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["22-traffic"] = {
  project: "22-traffic",
  title: "Traffic Simulator",
  pitch: "Four rules per driver on a ring road — and traffic jams appear out of NOTHING, then drive backward through the cars.",
  tier: "medium",
  language: "Python",
  file: "traffic.py",
  chapters: [
    {
      id: 1, title: "A road of cells",
      build: "the Nagel-Schreckenberg lattice — cars as integers on a ring — with only the accelerate rule.",
      beat: "Cars circulate the ring at full speed… and sail straight through each other.",
      steps: [
        { title: "The physics of rush hour", adding: "the docstring and imports.",
          code: `"""Traffic: four rules per driver, and jams appear out of nowhere — then drive backward."""
import numpy as np
import taichi as ti`,
          does: "This project builds the Nagel-Schreckenberg model — the 1992 cellular automaton that made traffic a physics problem. A road is a ring of cells; a car is an integer velocity on a cell; a driver is FOUR rules (speed up, don't hit the car ahead, sometimes dawdle, move). From those four rules, real traffic phenomena emerge: free flow, a critical density, and the famous PHANTOM JAM — a wave of stopped cars with no cause, drifting backward against traffic.",
          why: "You have personally been inside this simulation: every stop-and-go wave on a highway with no visible accident is a NaSch phantom jam. It's the arc's second lesson in emergence — project 21's order emerged from communication (pheromones); this one's CHAOS emerges from nothing but one dawdle rule and too many cars.",
          see: "Runs clean.",
          checkpoint: "python3 traffic.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "Integer traffic", adding: "road dials and the car fields.",
          code: `RES = 512
ROAD_LEN = 1000
MAX_CARS = 600
START_CARS = 220
VMAX = 5
RING_R = 0.4
PI2 = 6.28318530
car_pos = None
car_v = None
active = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global car_pos, car_v, active, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    car_pos = ti.field(ti.i32, shape=MAX_CARS)
    car_v = ti.field(ti.i32, shape=MAX_CARS)
    active = ti.field(ti.i32, shape=MAX_CARS)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "Everything is an INTEGER: position is a cell index on a 1,000-cell ring, velocity is 0 to 5 cells per tick. No floats, no forces, no dt — a pure cellular automaton, the first since project 01's grids and the first ever on particles. active is the familiar alive-flag so the car count can change at runtime within a fixed pool.",
          why: "Discreteness here is a feature, not an approximation: NaSch's famous results (the critical density, the backward wave speed) are properties of the INTEGER model. It's a reminder that not every simulation wants more precision — some phenomena live in the coarse structure.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["VMAX = 5 cells/tick is the standard NaSch calibration — with ~7.5m cells and 1s ticks it corresponds to roughly 135 km/h."] },
        { title: "Rule one, alone", adding: "seeding, the accelerate-only driver, movement, and the ring render.",
          code: `def seed_road(n, rng_seed=0):
    """Pure numpy: n cars in distinct random cells, everyone stopped."""
    rng = np.random.default_rng(rng_seed)
    positions = np.sort(rng.choice(ROAD_LEN, size=n, replace=False)).astype(np.int32)
    pos_arr = np.full(MAX_CARS, -1, dtype=np.int32)
    pos_arr[:n] = positions
    car_pos.from_numpy(pos_arr)
    car_v.from_numpy(np.zeros(MAX_CARS, dtype=np.int32))
    act = np.zeros(MAX_CARS, dtype=np.int32)
    act[:n] = 1
    active.from_numpy(act)
@ti.kernel
def update_velocity():
    for i in car_pos:
        if active[i] == 1:
            car_v[i] = ti.min(car_v[i] + 1, VMAX)
@ti.kernel
def move_cars():
    for i in car_pos:
        if active[i] == 1:
            car_pos[i] = (car_pos[i] + car_v[i]) % ROAD_LEN
def step(t):
    update_velocity()
    move_cars()
@ti.func
def speed_color(v):
    s = v / VMAX
    return ti.Vector([1.0, 0.2, 0.15]) * (1 - s) + ti.Vector([0.3, 1.0, 0.4]) * s
@ti.kernel
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
                    pixels[xi, yi] = col
def main():
    init_sim()
    seed_road(START_CARS)
    t = 0
    gui = ti.GUI("Traffic — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step(t)
        render_ring()
        gui.set_image(pixels)
        t += 1
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "seed_road places cars via np.random.choice with replace=False — distinct cells guaranteed by construction. The driver so far knows one rule: accelerate (+1 per tick, capped at VMAX). The ring render maps cell index to angle and colors each car by speed — red stopped, green cruising — the fundamental readout for everything ahead.",
          why: "One rule is deliberately broken physics: with no concept of the car ahead, everyone accelerates to VMAX and stays there — and cars drive straight THROUGH each other (positions collide freely; nothing checks). The uniform green flow you see is a lie that only a collision check would expose, and building that check — rule two — is the entire next chapter's opening move.",
          see: "A ring of dots that all turn green within five ticks and circulate at full speed forever, ghosting through one another.",
          checkpoint: "Full-speed ghosts. Beat 1.",
          recovery: ["The % ROAD_LEN in move_cars is the ring: drive off cell 999 and you're back at cell 0.", "speed_color's red-to-green ramp is the project's one piece of visual vocabulary — every later view reuses it."] }
      ]
    },
    {
      id: 2, title: "Four rules, and the phantom",
      build: "the gap rule (an occupancy grid), the dawdle rule, and the space-time diagram that reveals backward-drifting jams.",
      beat: "Jams condense out of pure randomness and crawl backward — visible as diagonal red waves in space-time.",
      steps: [
        { title: "Rule two: don't hit the car ahead", adding: "the occupancy grid and gap-limited braking.",
          code: `occupancy = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global car_pos, car_v, active, occupancy, pixels
    occupancy = ti.field(ti.i32, shape=ROAD_LEN)
@ti.kernel
def build_occupancy():
    for c in occupancy:
        occupancy[c] = 0
    for i in car_pos:
        if active[i] == 1:
            occupancy[car_pos[i]] = 1
@ti.kernel
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
            car_v[i] = v
def step(t):
    build_occupancy()
    update_velocity()
    move_cars()`,
          does: "The occupancy grid is rebuilt fresh each tick (cars mark their cells — project 15's deposit, boolean edition), and every driver scans at most VMAX+1 cells ahead for the nearest mark. The braking rule v = min(v, gap - 1) means: never plan to ENTER the cell the car ahead occupies. That single inequality is the no-collision guarantee — not a collision check after the fact, but an invariant the update can't violate.",
          why: "This is also why the PARALLEL update is safe with no atomics at all: each car moves at most gap-1 cells into space that was provably empty, and since every car only moves FORWARD, no two claims can cross. Structural safety beats defensive checking — compare project 21's food race, which needed an atomic precisely because that structure was absent.",
          see: "Run it: full-speed flow where the road is open, smooth deceleration into any slow patch — and, at this density, mostly uniform motion. Deterministic drivers are flawless drivers. Suspiciously flawless.",
          checkpoint: "Safe following. No red text.",
          recovery: ["The gap scan keeps the FIRST hit via the 'and gap == VMAX + 1' guard — a loop-carried min without a break, the same break-free idiom project 20's DE taught.", "gap - 1, not gap: distance to the car minus the cell it stands on."] },
        { title: "Rule three: sometimes, dawdle", adding: "the random slowdown.",
          code: `P_SLOW = 0.25
@ti.kernel
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
            car_v[i] = v`,
          does: "One new rule: with probability 25%, a driver drops one unit of speed for no reason at all — a glance at the radio, a moment's hesitation. That's the entire change.",
          why: "This is THE ingredient. Without it, NaSch traffic is a clockwork that never jams below full packing. With it, a random hesitation forces the follower to brake harder than the leader slowed, the follower's follower harder still — and the amplifying wave freezes into a jam that no one caused. Nagel and Schreckenberg's insight was that human imperfection isn't noise ON the traffic system; at high enough density, it IS the traffic system.",
          see: "Same ring, new weather: clusters of red condense out of the smooth flow, persist, and — watch one carefully — drift slowly BACKWARD around the ring while every car in it moves only forward. A wave made of stopping.",
          checkpoint: "Phantom jams. Beat 2 — the model's famous result.",
          recovery: ["The dawdle applies AFTER the safety rule, so it can only ever slow a car further — randomness never causes a collision."] },
        { title: "The space-time X-ray", adding: "the scrolling diagram and the [v] view toggle.",
          code: `spacetime = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global car_pos, car_v, active, occupancy, pixels, spacetime
    spacetime = ti.Vector.field(3, ti.f32, shape=(RES, RES))
def seed_road(n, rng_seed=0):
    """Pure numpy: n cars in distinct random cells, everyone stopped."""
    rng = np.random.default_rng(rng_seed)
    positions = np.sort(rng.choice(ROAD_LEN, size=n, replace=False)).astype(np.int32)
    pos_arr = np.full(MAX_CARS, -1, dtype=np.int32)
    pos_arr[:n] = positions
    car_pos.from_numpy(pos_arr)
    car_v.from_numpy(np.zeros(MAX_CARS, dtype=np.int32))
    act = np.zeros(MAX_CARS, dtype=np.int32)
    act[:n] = 1
    active.from_numpy(act)
    spacetime.fill(0.02)
@ti.kernel
def render_spacetime():
    for i, j in spacetime:
        if j < RES - 1:
            spacetime[i, j] = spacetime[i, j + 1]
    for i in range(RES):
        spacetime[i, RES - 1] = ti.Vector([0.02, 0.02, 0.04])
    for i in car_pos:
        if active[i] == 1:
            xi = car_pos[i] * RES // ROAD_LEN
            spacetime[xi, RES - 1] = speed_color(car_v[i])
    show_ring = True
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "v":
                show_ring = not show_ring
        render_spacetime()
        if show_ring:
            render_ring()
            gui.set_image(pixels)
        else:
            gui.set_image(spacetime)`,
          does: "Every tick, the diagram scrolls up one row and paints the road's current state along the bottom — position across, time upward, color by speed. It's the standard scientific visualization for 1D traffic, built from a shift-and-stamp kernel.",
          why: "The diagram makes the phantom's strangest property undeniable: every individual car draws a rightward-leaning green streak (forward motion), yet the red jam bands lean LEFT — the jam travels backward at a fixed speed (about -1 cell per tick in standard NaSch) while consisting entirely of forward-moving cars. A wave is not the stuff it's made of; this picture is that sentence, drawn.",
          see: "Press V: diagonal green rain crossed by thick red bands sloping the other way. Trace one band: cars enter it from the right, sit, and escape left, while the band itself glides steadily backward.",
          checkpoint: "Backward waves, visible. Beat 3.",
          recovery: ["The scroll copies row j+1 into j in one parallel pass — safe because every write reads only the row above it, never a row another thread writes."] }
      ]
    },
    {
      id: 3, title: "Lights, density, disaster",
      build: "traffic lights as blocked cells, live density control with a flow meter, and click-to-crash.",
      beat: "A live traffic lab: find the critical density, time the lights, cause a pileup, watch the wave.",
      steps: [
        { title: "A red light is a parked car", adding: "light state, phased red/green cycles, and their occupancy trick.",
          code: `N_LIGHTS = 3
LIGHT_PERIOD = 120
LIGHT_GREEN = 70
light_pos = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global car_pos, car_v, active, occupancy, light_pos, pixels, spacetime
    light_pos = ti.field(ti.i32, shape=N_LIGHTS)
def seed_road(n, rng_seed=0):
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
    spacetime.fill(0.02)
@ti.kernel
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
                occupancy[light_pos[k]] = 1
def step(t, lights_on=True):
    build_occupancy(t, 1 if lights_on else 0)
    update_velocity()
    move_cars()
@ti.kernel
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
                    pixels[xi, yi] = col
    lights_on = True
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                lights_on = not lights_on
            elif e.key == "v":
                show_ring = not show_ring
        step(t, lights_on)
        if show_ring:
            render_ring(t, 1 if lights_on else 0)
            gui.set_image(pixels)
        else:
            gui.set_image(spacetime)`,
          does: "A red light needs zero new driver logic: during its red phase, the light's cell is simply marked OCCUPIED in the grid, and the existing gap rule makes every approaching car queue behind it exactly as if a car were parked there. The three lights run offset phases (a poor man's green wave), and space toggles the whole system off.",
          why: "Representing a light as a phantom parked car is the elegant move of this chapter: infrastructure enters the simulation through the same one-bit channel the cars use, so the driver model stays untouched. When a mechanism is expressed in the model's native vocabulary, features cost almost nothing.",
          see: "Queues build at each red, release in green pulses, and those pulses ripple around the ring. In space-time view the lights draw vertical dashed red columns with jam wedges growing behind them.",
          checkpoint: "Working lights. No red text.",
          recovery: ["The phase offset (k * LIGHT_PERIOD // N_LIGHTS) staggers the lights — set it to 0 and all three block simultaneously, a fun worst-case to try."] },
        { title: "The density dial", adding: "live car-count surgery and the flow meter.",
          code: `CAR_STEP = 20
def set_car_count(n_target, rng_seed=0):
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
    return int(act.sum())
def mean_speed():
    """Pure numpy: the live flow reading for the HUD."""
    act = active.to_numpy() == 1
    if act.sum() == 0:
        return 0.0
    return float(car_v.to_numpy()[act].mean())
    n = START_CARS
        for e in gui.get_events(ti.GUI.PRESS):
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
        density = n / ROAD_LEN
        gui.text(f"cars {n}  density {density:.2f}  mean speed {mean_speed():.2f}", (0.02, 0.98), color=0xFFFFFF)`,
          does: "Up/down arrows add or remove twenty cars live. New cars must appear in EMPTY cells — set_car_count pulls both fields to numpy, computes the free cells, and does the surgery there (a rare, human-triggered operation, so the CPU round-trip costs nothing that matters). The HUD becomes an instrument: density and mean speed, live.",
          why: "This turns the project into the experiment traffic physicists actually run: sweep density and watch mean speed. Below roughly 0.08, adding cars barely hurts; past the critical region the flow COLLAPSES — the famous fundamental diagram of traffic, discovered with your arrow keys. Turn lights off (space) for the clean version of the experiment.",
          see: "Tap up-arrow with lights off and watch the meter: 4.7… 4.6… 4.2… then somewhere past 200 cars the phantom jams take hold and it tumbles toward 1. Tap down and the road heals.",
          checkpoint: "A live fundamental-diagram lab. No red text.",
          recovery: ["New cars spawn at v=0 in random empty cells — expect a momentary dip in mean speed as they merge, just like real on-ramps."] },
        { title: "Cause a pileup", adding: "the incident kernel, click wiring, reset, and the control legend.",
          code: `@ti.kernel
def incident(cell: ti.i32):
    for i in car_pos:
        if active[i] == 1:
            d = ti.abs(car_pos[i] - cell)
            d = ti.min(d, ROAD_LEN - d)
            if d < 15:
                car_v[i] = 0
        for e in gui.get_events(ti.GUI.PRESS):
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
                incident(int(ang / PI2 * ROAD_LEN))
        gui.text("[up/down] density  [space] lights  [v] view  [r] reset  click: incident", (0.02, 0.94), color=0xAAAAAA)`,
          does: "Click anywhere near the ring: the click's angle converts back to a road cell (the render mapping, inverted with atan2), and every car within 15 cells slams to a stop — a fender-bender. Then the model takes over: the stopped knot becomes a jam wave and begins its backward crawl.",
          why: "The incident is the controlled experiment the phantom jam denies you: HERE, you know exactly where and when the disturbance happened, so you can watch its wave propagate, measure its backward speed against the space-time diagram, and see how long the road takes to forgive one moment of chaos — at low density, seconds; near critical density, never. Four rules produced all of it.",
          see: "Click at moderate density and flip to space-time: a red wedge blooms at your click and its trailing edge glides backward at the model's fixed wave speed while cars thread through it. You have made weather, and now you can forecast it.",
          checkpoint: "Click-to-crash, full control legend. Final beat — project 22 complete.",
          recovery: ["The distance check wraps the ring (min of d and ROAD_LEN - d) — an incident at cell 0 must also stop cars at cell 995.", "atan2's result is remapped with % PI2 because it returns negatives on the lower half — forgetting that maps bottom clicks to the wrong side."] }
      ]
    }
  ]
};

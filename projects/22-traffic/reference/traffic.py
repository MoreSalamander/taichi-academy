"""Traffic: four rules per driver, and jams appear out of nowhere — then drive backward."""

import numpy as np
import taichi as ti

RES = 512
ROAD_LEN = 1000
MAX_CARS = 600
START_CARS = 220
CAR_STEP = 20
VMAX = 5
P_SLOW = 0.25

N_LIGHTS = 3
LIGHT_PERIOD = 120
LIGHT_GREEN = 70

RING_R = 0.4
PI2 = 6.28318530

car_pos = None
car_v = None
active = None
occupancy = None
light_pos = None
pixels = None
spacetime = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global car_pos, car_v, active, occupancy, light_pos, pixels, spacetime
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
    occupancy = ti.field(ti.i32, shape=ROAD_LEN)
    light_pos = ti.field(ti.i32, shape=N_LIGHTS)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))
    spacetime = ti.Vector.field(3, ti.f32, shape=(RES, RES))


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
            car_v[i] = v


@ti.kernel
def move_cars():
    for i in car_pos:
        if active[i] == 1:
            car_pos[i] = (car_pos[i] + car_v[i]) % ROAD_LEN


@ti.kernel
def incident(cell: ti.i32):
    for i in car_pos:
        if active[i] == 1:
            d = ti.abs(car_pos[i] - cell)
            d = ti.min(d, ROAD_LEN - d)
            if d < 15:
                car_v[i] = 0


def step(t, lights_on=True):
    build_occupancy(t, 1 if lights_on else 0)
    update_velocity()
    move_cars()


@ti.func
def speed_color(v):
    s = v / VMAX
    return ti.Vector([1.0, 0.2, 0.15]) * (1 - s) + ti.Vector([0.3, 1.0, 0.4]) * s


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


def mean_speed():
    """Pure numpy: the live flow reading for the HUD."""
    act = active.to_numpy() == 1
    if act.sum() == 0:
        return 0.0
    return float(car_v.to_numpy()[act].mean())


def main():
    init_sim()
    seed_road(START_CARS)
    n = START_CARS
    t = 0
    lights_on = True
    show_ring = True
    gui = ti.GUI("Traffic — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
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
        step(t, lights_on)
        render_spacetime()
        if show_ring:
            render_ring(t, 1 if lights_on else 0)
            gui.set_image(pixels)
        else:
            gui.set_image(spacetime)
        t += 1
        density = n / ROAD_LEN
        gui.text(f"cars {n}  density {density:.2f}  mean speed {mean_speed():.2f}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[up/down] density  [space] lights  [v] view  [r] reset  click: incident", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

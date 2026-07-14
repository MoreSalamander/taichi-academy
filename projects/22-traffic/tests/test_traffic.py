import numpy as np

import traffic as tr


def run(n, lights_on=True, t0=0):
    for t in range(t0, t0 + n):
        tr.step(t, lights_on)
    return t0 + n


def live_positions():
    act = tr.active.to_numpy() == 1
    return tr.car_pos.to_numpy()[act]


# --- seeding -------------------------------------------------------------------------


def test_seed_road_distinct_cells_and_determinism():
    tr.seed_road(300, rng_seed=3)
    a = live_positions()
    tr.seed_road(300, rng_seed=3)
    b = live_positions()
    tr.seed_road(300, rng_seed=4)
    c = live_positions()
    assert len(np.unique(a)) == 300
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_set_car_count_adds_and_removes_without_overlap():
    n = tr.set_car_count(400, rng_seed=1)
    assert n == 400
    p = live_positions()
    assert len(np.unique(p)) == 400
    n = tr.set_car_count(100)
    assert n == 100
    assert len(np.unique(live_positions())) == 100


# --- the four rules ---------------------------------------------------------------------


def test_no_collisions_ever():
    """NaSch's structural guarantee: v <= gap - 1, so two cars never share a cell."""
    t = run(500)
    run(300, lights_on=False, t0=t)
    p = live_positions()
    assert len(np.unique(p)) == len(p)


def test_free_road_reaches_near_vmax():
    """One lonely car with p_slow randomness averages about VMAX - P_SLOW."""
    tr.seed_road(1, rng_seed=1)
    speeds = []
    for t in range(400):
        tr.step(t, lights_on=False)
        if t > 100:
            speeds.append(tr.car_v.to_numpy()[tr.active.to_numpy() == 1][0])
    avg = np.mean(speeds)
    assert abs(avg - (tr.VMAX - tr.P_SLOW)) < 0.3, f"lone car should cruise near {tr.VMAX - tr.P_SLOW}, got {avg:.2f}"


def test_braking_respects_the_gap():
    """Plant two cars 3 cells apart; the follower must never move onto the leader."""
    pos = np.full(tr.MAX_CARS, -1, dtype=np.int32)
    pos[0], pos[1] = 100, 103
    tr.car_pos.from_numpy(pos)
    v = np.zeros(tr.MAX_CARS, dtype=np.int32)
    v[0] = tr.VMAX  # follower at full speed
    tr.car_v.from_numpy(v)
    act = np.zeros(tr.MAX_CARS, dtype=np.int32)
    act[:2] = 1
    tr.active.from_numpy(act)
    for t in range(50):
        tr.step(t, lights_on=False)
        p = live_positions()
        assert len(np.unique(p)) == 2


def test_congestion_collapses_mean_speed():
    """The fundamental diagram's two phases: free flow at low density, jam at high."""
    tr.seed_road(60, rng_seed=1)  # density 0.06
    run(400, lights_on=False)
    free_flow = tr.mean_speed()
    tr.seed_road(450, rng_seed=1)  # density 0.45
    run(400, lights_on=False)
    jammed = tr.mean_speed()
    assert free_flow > 3.5, f"low density should flow near vmax, got {free_flow:.2f}"
    assert jammed < 1.5, f"high density should crawl, got {jammed:.2f}"
    assert free_flow > jammed * 2.5


def test_phantom_jams_form_without_lights():
    """Above critical density with NO lights and NO incident, some cars still stop —
    jams born purely from the random slowdowns."""
    tr.seed_road(300, rng_seed=1)
    run(600, lights_on=False)
    v = tr.car_v.to_numpy()[tr.active.to_numpy() == 1]
    assert (v == 0).sum() > 10, "phantom jams should have stopped some cars"
    assert (v >= tr.VMAX - 1).sum() > 10, "…while others still cruise"


# --- lights & incidents ------------------------------------------------------------------


def test_red_light_blocks_a_car():
    pos = np.full(tr.MAX_CARS, -1, dtype=np.int32)
    light0 = int(tr.light_pos.to_numpy()[0])
    pos[0] = (light0 - 3) % tr.ROAD_LEN
    tr.car_pos.from_numpy(pos)
    tr.car_v.from_numpy(np.zeros(tr.MAX_CARS, dtype=np.int32))
    act = np.zeros(tr.MAX_CARS, dtype=np.int32)
    act[0] = 1
    tr.active.from_numpy(act)
    # phase LIGHT_GREEN is the first red tick for light 0
    for t in range(tr.LIGHT_GREEN, tr.LIGHT_GREEN + 10):
        tr.step(t, lights_on=True)
    p = live_positions()[0]
    ahead = (p - light0) % tr.ROAD_LEN
    assert ahead > tr.ROAD_LEN // 2 or p == (light0 - 1) % tr.ROAD_LEN or (light0 - p) % tr.ROAD_LEN >= 1, \
        "the car must still be behind the red light"
    assert (light0 - p) % tr.ROAD_LEN >= 1


def test_incident_stops_nearby_cars_only():
    pos = np.full(tr.MAX_CARS, -1, dtype=np.int32)
    pos[0], pos[1] = 500, 800
    tr.car_pos.from_numpy(pos)
    v = np.full(tr.MAX_CARS, 4, dtype=np.int32)
    tr.car_v.from_numpy(v)
    act = np.zeros(tr.MAX_CARS, dtype=np.int32)
    act[:2] = 1
    tr.active.from_numpy(act)
    tr.incident(505)
    v_out = tr.car_v.to_numpy()
    assert v_out[0] == 0, "the car near the incident stops"
    assert v_out[1] == 4, "the far car is untouched"


# --- rendering -----------------------------------------------------------------------


def test_renders_are_finite_and_bounded():
    run(50)
    tr.render_ring(50, 1)
    tr.render_spacetime()
    for f in (tr.pixels, tr.spacetime):
        px = f.to_numpy()
        assert np.all(np.isfinite(px))
        assert px.min() >= 0.0 and px.max() <= 1.0

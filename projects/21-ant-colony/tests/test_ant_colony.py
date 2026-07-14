import numpy as np

import ant_colony as ac


def run(n):
    for _ in range(n):
        ac.step()


# --- pure numpy generation ---------------------------------------------------------


def test_seed_food_layout_and_determinism():
    a = ac.seed_food(rng_seed=3)
    b = ac.seed_food(rng_seed=3)
    c = ac.seed_food(rng_seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.sum() > 0
    # no food inside the nest exclusion zone
    ii, jj = np.meshgrid(np.arange(ac.GRID), np.arange(ac.GRID), indexing="ij")
    near_nest = np.hypot(ii / ac.GRID - ac.NEST[0], jj / ac.GRID - ac.NEST[1]) < 0.15
    assert a[near_nest].sum() == 0.0


def test_apply_seed_starts_everyone_at_the_nest_foraging():
    p = ac.pos.to_numpy()
    assert np.allclose(p, ac.NEST)
    assert (ac.state.to_numpy() == ac.FORAGING).all()
    assert ac.trail.to_numpy().sum() == 0.0


# --- steering ------------------------------------------------------------------------


def test_foragers_turn_toward_stronger_trail():
    """Plant a trail stripe to one side of a single ant and check it turns that way."""
    p = np.tile(np.array([0.5, 0.5], dtype=np.float32), (ac.N_ANTS, 1))
    ac.pos.from_numpy(p)
    h = np.zeros(ac.N_ANTS, dtype=np.float32)  # everyone faces +x
    ac.heading.from_numpy(h)
    ac.state.fill(ac.FORAGING)
    ac.food.fill(0.0)
    # the left sensor sits at y = 0.5 + (SENSE_DIST/GRID)*sin(SENSE_ANGLE) — row ~131 of 256;
    # the stripe must start at or below that row or all three sensors read zero
    t = np.zeros((ac.GRID, ac.GRID), dtype=np.float32)
    t[:, ac.GRID // 2 + 2 :] = 50.0  # strong trail above (in +y, the 'left' sensor side)
    ac.trail.from_numpy(t)
    ac.move_ants()
    dh = ac.heading.to_numpy() - h
    # wander is random ±WANDER/2; the deterministic TURN dominates on average
    assert dh.mean() > ac.TURN * 0.5, f"ants should steer toward the trail, mean turn {dh.mean():.3f}"


def test_returners_steer_toward_the_nest():
    p = np.tile(np.array([0.8, 0.5], dtype=np.float32), (ac.N_ANTS, 1))
    ac.pos.from_numpy(p)
    ac.heading.fill(0.0)  # facing AWAY from the nest (+x)
    ac.state.fill(ac.RETURNING)
    ac.food.fill(0.0)
    for _ in range(30):
        ac.move_ants()
    p1 = ac.pos.to_numpy()
    d0 = np.hypot(0.8 - ac.NEST[0], 0.5 - ac.NEST[1])
    d1 = np.hypot(p1[:, 0] - ac.NEST[0], p1[:, 1] - ac.NEST[1])
    assert (d1 < d0).mean() > 0.95, "nearly all returners should be closing on the nest"


def test_returners_lay_trail():
    p = np.tile(np.array([0.7, 0.7], dtype=np.float32), (ac.N_ANTS, 1))
    ac.pos.from_numpy(p)
    ac.state.fill(ac.RETURNING)
    ac.food.fill(0.0)
    ac.trail.fill(0.0)
    ac.move_ants()
    assert ac.trail.to_numpy().sum() > 0.0


def test_foragers_do_not_lay_trail():
    ac.food.fill(0.0)
    ac.trail.fill(0.0)
    ac.move_ants()
    assert ac.trail.to_numpy().sum() == 0.0


# --- food pickup -------------------------------------------------------------------


def test_food_pickup_flips_state_and_consumes():
    total0 = ac.food.to_numpy().sum()
    run(400)
    picked = (ac.state.to_numpy() == ac.RETURNING).sum()
    total1 = ac.food.to_numpy().sum()
    assert picked > 0, "someone should have found food by now"
    assert total1 < total0


def test_food_never_goes_negative():
    """Regression test: the pickup used to be check-then-subtract, and thousands of
    ants hitting one cell in the same step drove it far negative. The atomic claim
    (subtract first, refund if the claim came up empty) keeps the floor at zero."""
    run(600)
    assert ac.food.to_numpy().min() >= 0.0


def test_reaching_the_nest_flips_back_to_foraging():
    p = np.tile(np.array([ac.NEST[0] + 0.001, ac.NEST[1]], dtype=np.float32), (ac.N_ANTS, 1))
    ac.pos.from_numpy(p)
    ac.state.fill(ac.RETURNING)
    ac.food.fill(0.0)
    ac.move_ants()
    assert (ac.state.to_numpy() == ac.FORAGING).all()


# --- trail dynamics -------------------------------------------------------------------


def test_trail_evaporates():
    ac.trail.fill(10.0)
    ac.evolve_trail()
    ac.copy_trail()
    t = ac.trail.to_numpy()
    interior = t[2:-2, 2:-2]
    assert np.allclose(interior, 10.0 * ac.EVAP, atol=1e-4)


def test_trail_diffuses():
    t = np.zeros((ac.GRID, ac.GRID), dtype=np.float32)
    t[100, 100] = 100.0
    ac.trail.from_numpy(t)
    ac.evolve_trail()
    ac.copy_trail()
    out = ac.trail.to_numpy()
    assert out[101, 100] > 0.0, "the spike should bleed into neighbors"
    assert out[100, 100] < 100.0


# --- emergence ------------------------------------------------------------------------


def test_trails_form_and_sim_stays_finite():
    run(800)
    assert np.all(np.isfinite(ac.pos.to_numpy()))
    assert ac.trail.to_numpy().max() > 1.0, "returners should have built visible trails"
    ac.render()
    px = ac.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

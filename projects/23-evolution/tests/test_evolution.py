import numpy as np
import taichi as ti

import evolution as ev


def run(n):
    for _ in range(n):
        ev.step()


# --- pure numpy generation ---------------------------------------------------------


def test_food_field_patches_and_determinism():
    a = ev.food_field(rng_seed=3)
    b = ev.food_field(rng_seed=3)
    c = ev.food_field(rng_seed=4)
    assert a.shape == (ev.FOOD_GRID, ev.FOOD_GRID)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # patchy: rich peaks over sparse barrens (most of the map is well below the peak)
    assert a.max() > 0.9 and (a < 0.1).mean() > 0.3


def test_apply_seed_starts_a_living_population():
    assert ev.population() == ev.START_POP
    e = ev.energy.to_numpy()[ev.alive.to_numpy() == 1]
    assert np.allclose(e, ev.START_ENERGY)
    assert np.allclose(ev.food.to_numpy(), ev.food_cap.to_numpy())


# --- the brain -----------------------------------------------------------------------


def test_brain_output_is_bounded_and_deterministic():
    @ti.kernel
    def probe(s0: ti.f32, s1: ti.f32, s2: ti.f32, s3: ti.f32) -> ti.math.vec2:
        turn, thrust = ev.brain(0, s0, s1, s2, s3, 1.0)
        return ti.Vector([turn, thrust])

    a = np.array(probe(1.0, 0.0, 0.0, 0.5))
    b = np.array(probe(1.0, 0.0, 0.0, 0.5))
    assert np.array_equal(a, b), "same weights + inputs -> same output"
    assert np.all(np.abs(a) <= 1.0), "tanh keeps motor outputs in [-1, 1]"


def test_brain_reads_its_own_weights():
    """Two creatures with different weights should compute different outputs."""
    rng = np.random.default_rng(1)
    w = ev.weights.to_numpy()
    w[0] = rng.normal(0, 1, ev.N_W)
    w[1] = rng.normal(0, 1, ev.N_W)
    ev.weights.from_numpy(w)

    @ti.kernel
    def probe(c: ti.i32) -> ti.math.vec2:
        turn, thrust = ev.brain(c, 0.7, 0.3, 0.1, 0.5, 1.0)
        return ti.Vector([turn, thrust])

    assert not np.allclose(np.array(probe(0)), np.array(probe(1)))


# --- survival loop -------------------------------------------------------------------


def test_eating_gains_energy_and_depletes_food():
    p = ev.pos.to_numpy()
    p[0] = [0.5, 0.5]
    ev.pos.from_numpy(p)
    ev.food.fill(1.0)
    ev.food_cap.fill(1.0)
    # a brain that always thrusts forward so it lands on a food cell
    w = ev.weights.to_numpy()
    w[0] = 0.0
    ev.weights.from_numpy(w)
    e0 = ev.energy.to_numpy()[0]
    ev.sense_think_move()
    e1 = ev.energy.to_numpy()[0]
    # net energy change = eat gain (if it bit food) minus costs; on full food it should net positive
    assert e1 > e0, "sitting on abundant food should net energy"
    assert ev.food.to_numpy().min() < 1.0, "some food was consumed"


def test_food_never_goes_negative():
    """Regression: parallel eaters on one cell used to drive it negative — the
    atomic bite (subtract, refund if empty) keeps food non-negative."""
    run(400)
    assert ev.food.to_numpy().min() >= -1e-6


def test_starvation_kills():
    e = ev.energy.to_numpy()
    e[0] = ev.LIVE_COST * 0.5  # not enough to survive one tick
    ev.energy.from_numpy(e)
    # stop it eating: empty the world
    ev.food.fill(0.0)
    ev.food_cap.fill(0.0)
    ev.sense_think_move()
    assert ev.alive.to_numpy()[0] == 0


# --- reproduction (free list) ---------------------------------------------------------


def test_build_free_list_lists_every_dead_slot():
    a = ev.alive.to_numpy()
    a[:] = 0
    a[10] = 1  # one survivor
    ev.alive.from_numpy(a)
    ev.build_free_list()
    assert ev.n_free[None] == ev.N_MAX - 1
    listed = set(ev.free_slots.to_numpy()[: ev.n_free[None]].tolist())
    assert 10 not in listed and len(listed) == ev.N_MAX - 1


def test_reproduction_fills_a_slot_and_halves_parent():
    e = ev.energy.to_numpy()
    e[0] = ev.REPRO_ENERGY + 10
    ev.energy.from_numpy(e)
    n0 = ev.population()
    ev.build_free_list()
    ev.reproduce()
    n1 = ev.population()
    assert n1 == n0 + 1, "one birth"
    assert abs(ev.energy.to_numpy()[0] - (ev.REPRO_ENERGY + 10) / 2) < 1e-3, "parent energy halved"


def test_reproduction_respects_the_cap():
    a = np.ones(ev.N_MAX, dtype=np.int32)
    ev.alive.from_numpy(a)  # world full
    e = np.full(ev.N_MAX, ev.REPRO_ENERGY + 10, dtype=np.float32)
    ev.energy.from_numpy(e)
    ev.build_free_list()
    assert ev.n_free[None] == 0
    ev.reproduce()
    assert ev.population() == ev.N_MAX, "no births when the world is full"


def test_children_inherit_with_occasional_mutation():
    w = ev.weights.to_numpy()
    w[0] = 0.5  # a distinctive parent genome
    ev.weights.from_numpy(w)
    e = ev.energy.to_numpy()
    e[0] = ev.REPRO_ENERGY + 10
    ev.energy.from_numpy(e)
    # force the child into a known slot by leaving exactly one free slot
    a = np.ones(ev.N_MAX, dtype=np.int32)
    a[0] = 1
    a[2000] = 0
    ev.alive.from_numpy(a)
    e2 = ev.energy.to_numpy()
    e2[0] = ev.REPRO_ENERGY + 10
    e2[1:] = 1.0  # nobody else reproduces
    ev.energy.from_numpy(e2)
    ev.build_free_list()
    ev.reproduce()
    child = ev.weights.to_numpy()[2000]
    parent = ev.weights.to_numpy()[0]
    diff = np.abs(child - parent)
    assert diff.max() < 2 * ev.MUT_SCALE + 1e-4, "mutations are bounded steps off the parent"
    assert (diff < 1e-6).mean() > 0.5, "most genes copy exactly (mutation rate is low)"


# --- the headline: evolution actually improves the brains ------------------------------


def _frozen_trial(weight_pool, world_seed, steps=300):
    """Drop a gene pool into a fresh world and run it with NO reproduction (genes
    frozen) — pure behavior. Returns mean energy of the survivors."""
    reps = int(np.ceil(ev.N_MAX / len(weight_pool)))
    w_full = np.tile(weight_pool, (reps, 1))[: ev.N_MAX].astype(np.float32)
    ev.apply_seed(world_seed)
    ev.weights.from_numpy(w_full)
    for _ in range(steps):
        ev.sense_think_move()
        ev.regrow()
    a = ev.alive.to_numpy() == 1
    return ev.energy.to_numpy()[a].mean() if a.sum() else 0.0


def test_evolved_brains_beat_random_brains():
    """The whole point of the project, as a measurement: evolve a population, then
    race its survivors' genes against fresh random genes in an IDENTICAL world with
    reproduction switched off. The evolved brains forage better — no designer told
    them how."""
    ev.apply_seed(rng_seed=1)
    for _ in range(2000):
        ev.step()
    survivors = ev.weights.to_numpy()[ev.alive.to_numpy() == 1]
    assert len(survivors) > 50

    rng = np.random.default_rng(7)
    random_pool = rng.normal(0, 1.0, (400, ev.N_W)).astype(np.float32)

    evolved_fitness = _frozen_trial(survivors, world_seed=42)
    random_fitness = _frozen_trial(random_pool, world_seed=42)
    assert evolved_fitness > random_fitness * 1.2, (
        f"evolved brains should forage better: evolved={evolved_fitness:.0f} random={random_fitness:.0f}"
    )


# --- stability -----------------------------------------------------------------------


def test_population_persists_and_stays_finite():
    run(1500)
    assert ev.population() > 50, "the colony should not go extinct"
    assert np.all(np.isfinite(ev.pos.to_numpy()))
    assert np.all(np.isfinite(ev.weights.to_numpy()))
    ev.render()
    px = ev.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

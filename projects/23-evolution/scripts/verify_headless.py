"""CI-safe sanity check: no window, CPU backend, a population evolves and foraging improves."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import evolution as ev


def frozen_fitness(weight_pool, world_seed, steps=300):
    reps = int(np.ceil(ev.N_MAX / len(weight_pool)))
    w_full = np.tile(weight_pool, (reps, 1))[: ev.N_MAX].astype(np.float32)
    ev.apply_seed(world_seed)
    ev.weights.from_numpy(w_full)
    for _ in range(steps):
        ev.sense_think_move()
        ev.regrow()
    a = ev.alive.to_numpy() == 1
    return ev.energy.to_numpy()[a].mean() if a.sum() else 0.0


def main():
    ev.init_sim(arch=ti.cpu)
    ev.apply_seed(rng_seed=1)
    for _ in range(2000):
        ev.step()
    pop = ev.population()
    assert pop > 50, "colony went extinct"
    survivors = ev.weights.to_numpy()[ev.alive.to_numpy() == 1]
    rng = np.random.default_rng(7)
    random_pool = rng.normal(0, 1.0, (400, ev.N_W)).astype(np.float32)
    evolved = frozen_fitness(survivors, 42)
    random = frozen_fitness(random_pool, 42)
    assert evolved > random, "evolution should have improved foraging"
    print(f"OK — evolved to pop {pop}; frozen fitness evolved={evolved:.0f} vs random={random:.0f}")


if __name__ == "__main__":
    main()

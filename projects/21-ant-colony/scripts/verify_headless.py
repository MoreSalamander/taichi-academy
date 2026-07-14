"""CI-safe sanity check: no window, CPU backend, ants find food and build trails."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ant_colony as ac


def main():
    ac.init_sim(arch=ti.cpu)
    ac.apply_seed(rng_seed=1)
    food0 = ac.food.to_numpy().sum()
    for _ in range(800):
        ac.step()
    food1 = ac.food.to_numpy().sum()
    assert food1 < food0, "the colony should have harvested something"
    assert ac.food.to_numpy().min() >= 0.0
    assert np.all(np.isfinite(ac.pos.to_numpy()))
    assert ac.trail.to_numpy().max() > 1.0
    ac.render()
    px = ac.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — 800 steps, {food0 - food1:.0f} food harvested, trails formed")


if __name__ == "__main__":
    main()

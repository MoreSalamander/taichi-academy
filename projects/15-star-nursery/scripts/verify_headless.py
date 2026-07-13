"""CI-safe sanity check: no window, CPU backend, a cloud collapses and births stars."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import star_nursery as sn


def main():
    sn.init_sim(arch=ti.cpu)
    sn.apply_seed(rng_seed=1)
    for _ in range(150):
        sn.step()
    assert sn.n_stars[None] > 0
    assert np.all(np.isfinite(sn.pos.to_numpy()))
    px = sn.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — 150 steps, {sn.n_stars[None]} stars born, {sn.alive.to_numpy().sum()} gas particles remain")


if __name__ == "__main__":
    main()

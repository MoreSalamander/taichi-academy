"""CI-safe sanity check: no window, CPU backend, cells self-organize from soup."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import artificial_life as al


def main():
    al.init_sim(arch=ti.cpu)
    al.apply_seed(rng_seed=1)
    al.build_grid()
    al.count_neighbors()
    before = int((al.neighbors.to_numpy() > 26).sum())
    for _ in range(400):
        al.step()
    after = int((al.neighbors.to_numpy() > 26).sum())
    p = al.pos.to_numpy()
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() < al.WORLD
    assert after > before * 4, "cells failed to condense"
    al.render()
    px = al.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — cells self-organized: nucleus particles {before} -> {after}")


if __name__ == "__main__":
    main()

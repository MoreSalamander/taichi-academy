"""CI-safe sanity check: no window, CPU backend, all three galaxy types spin cleanly."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import galaxy as g


def main():
    g.init_sim(arch=ti.cpu)
    for kind in (g.SPIRAL, g.ELLIPTICAL, g.RING):
        g.apply_seed(g.seed_galaxy(kind, rng_seed=1))
        for _ in range(60):
            g.step()
        px = g.pixels.to_numpy()
        assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
        assert px.sum() > 0
    print(f"OK — 3 galaxy types x 60 frames, {g.N_STARS} stars each")


if __name__ == "__main__":
    main()

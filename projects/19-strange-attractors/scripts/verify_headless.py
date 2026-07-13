"""CI-safe sanity check: no window, CPU backend, all four attractors settle and render."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import strange_attractors as sa


def main():
    sa.init_sim(arch=ti.cpu)
    for kind in (sa.LORENZ, sa.THOMAS, sa.AIZAWA, sa.CLIFFORD):
        sa.apply_seed(kind, rng_seed=1)
        for f in range(30):
            sa.step(kind, f * 0.01)
        p = sa.pos.to_numpy()
        px = sa.pixels.to_numpy()
        assert np.all(np.isfinite(p)), sa.NAMES[kind]
        assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
        assert px.sum() > 0
    print(f"OK — 4 attractors x 30 frames, {sa.N_PTS} points each")


if __name__ == "__main__":
    main()

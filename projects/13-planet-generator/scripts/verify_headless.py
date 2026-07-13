"""CI-safe sanity check: no window, CPU backend, an orbit + reseed renders cleanly."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import planet as pl


def main():
    pl.init_sim(arch=ti.cpu)
    pl.apply_seed(rng_seed=1)
    for i in range(8):
        theta = i * (2 * np.pi / 8)
        cx, cy, cz = pl.camera_position(theta)
        pl.render(cx, cy, cz, *pl.SUN)
    px = pl.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    e = pl.seed_terrain(pl.VOL_N, rng_seed=2)
    frac = (e < pl.SEA_LEVEL).mean()
    assert abs(frac - pl.OCEAN_FRACTION) < 0.02
    print(f"OK — full orbit rendered, ocean fraction {frac:.3f} (target {pl.OCEAN_FRACTION})")


if __name__ == "__main__":
    main()

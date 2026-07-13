"""CI-safe sanity check: no window, CPU backend, a painted stroke of each material behaves."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import particle_painting as pp


def main():
    pp.init_sim(arch=ti.cpu)
    pp.clear()
    for material in (pp.FIRE, pp.SMOKE, pp.SPARKS, pp.WATER):
        for i in range(40):
            pp.step(0.3 + 0.4 * (i / 40), 0.5, material, True)
    for _ in range(20):
        pp.step(0.0, 0.0, pp.FIRE, False)
    px = pp.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    alive = (pp.life.to_numpy() > 0).sum()
    print(f"OK — 4 strokes painted, {alive} particles alive, pixels in [{px.min():.3f},{px.max():.3f}]")


if __name__ == "__main__":
    main()

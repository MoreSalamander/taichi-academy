"""CI-safe sanity check: no window, CPU backend, orbits hold and energy behaves."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import solar_system as ss


def main():
    ss.init_sim(arch=ti.cpu)
    ss.apply_seed(rng_seed=1)
    e0 = ss.total_energy()
    for _ in range(120):
        ss.step()
    e1 = ss.total_energy()
    drift = np.abs((e1 - e0) / e0)[: ss.N_PLANETS].max()
    assert drift < 0.01
    assert np.all(np.isfinite(ss.pos.to_numpy()))
    px = ss.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — 120 frames, planet energy drift {drift:.5f}")


if __name__ == "__main__":
    main()

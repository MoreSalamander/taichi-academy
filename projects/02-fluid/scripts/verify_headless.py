"""CI-safe sanity check: no window, CPU backend, 100 stirred steps stay finite."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import fluid


def main():
    fluid.init_sim(arch=ti.cpu)
    fluid.apply_seed(fluid.seed_pattern(fluid.N, rng_seed=3))
    for k in range(100):
        if k % 10 == 0:
            fluid.splat(0.4, 0.5, 5.0, 2.0, 1.0, 0.4, 0.1)
        fluid.step(fluid.CURL_STRENGTH)
    v = fluid.vel.to_numpy()
    d = fluid.dye.to_numpy()
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(d)), "NaN/inf"
    fluid.compute_divergence()
    div = np.abs(fluid.divergence.to_numpy()).mean()
    fluid.render()
    assert np.all(np.isfinite(fluid.pixels.to_numpy()))
    print(f"OK — 100 stirred steps, |vel|max={np.abs(v).max():.2f}, mean|div|={div:.4f}")


if __name__ == "__main__":
    main()

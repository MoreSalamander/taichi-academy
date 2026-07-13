"""CI-safe sanity check: no window, CPU backend, mountains rise and the map drifts."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import plate_tectonics as pt


def main():
    pt.init_sim(arch=ti.cpu)
    pt.apply_seed(rng_seed=1)
    pid0 = pt.plate_id.to_numpy().copy()
    h0 = pt.height.to_numpy().copy()
    for f in range(200):
        pt.step(f)
    h1 = pt.height.to_numpy()
    assert np.all(np.isfinite(h1)) and h1.min() >= 0.0 and h1.max() <= 1.0
    assert not np.array_equal(pid0, pt.plate_id.to_numpy()), "the plates should have drifted"
    assert np.abs(h1 - h0).max() > 0.05, "boundaries should have built relief"
    pt.render()
    px = pt.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — 200 steps, relief built ({np.abs(h1-h0).max():.2f} max change), plates drifted")


if __name__ == "__main__":
    main()

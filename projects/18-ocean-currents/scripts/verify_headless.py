"""CI-safe sanity check: no window, CPU backend, currents flow and carry heat."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ocean_currents as oc


def main():
    oc.init_sim(arch=ti.cpu)
    oc.apply_seed(rng_seed=1)
    for _ in range(150):
        oc.step()
    oc.storm(0.4, 0.7)
    for _ in range(30):
        oc.step()
    v = oc.vel.to_numpy()
    t = oc.temp.to_numpy()
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(t))
    speed = np.linalg.norm(v, axis=2)
    assert speed.max() > 0.1, "the wind should have set the ocean moving"
    oc.render()
    px = oc.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — 180 steps + a storm, max current speed {speed.max():.2f}")


if __name__ == "__main__":
    main()

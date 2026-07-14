"""CI-safe sanity check: no window, CPU backend. A galaxy holds together and conserves energy
under the leapfrog integrator, and a two-galaxy collision runs without blowing up."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import universe_sandbox as us


def main():
    us.init_sim(arch=ti.cpu)
    us.apply_seed("single", seed=1)
    e0 = us.total_energy()
    for _ in range(400):
        us.step()
    e1 = us.total_energy()
    drift = abs(e1 - e0) / abs(e0)
    assert e0 < 0.0, "a bound galaxy has negative total energy"
    assert drift < 0.15, f"leapfrog should conserve energy, drift {drift:.3f}"
    assert us.bound_fraction() > 0.9, "the galaxy should stay bound"

    us.apply_seed("collide", seed=1)
    assert (us.mass.to_numpy() > 0.1).sum() == 2, "two black holes"
    for _ in range(400):
        us.step()
    assert np.all(np.isfinite(us.pos.to_numpy())), "the collision blew up"

    us.render()
    px = us.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — galaxy bound with {drift*100:.1f}% energy drift over 400 steps; collision stayed finite")


if __name__ == "__main__":
    main()

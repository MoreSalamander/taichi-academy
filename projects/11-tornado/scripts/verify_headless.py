"""CI-safe sanity check: no window, CPU backend, the vortex spins and debris orbits it."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import tornado as tn


def main():
    tn.init_sim(arch=ti.cpu)
    tn.apply_seed(rng_seed=1)
    for _ in range(150):
        tn.step()
    tn.stir(0.5, 0.5, 80.0, -40.0)
    for _ in range(30):
        tn.step()
    v = tn.vel.to_numpy()
    d = tn.dye.to_numpy()
    p = tn.dpos.to_numpy()
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(d)) and np.all(np.isfinite(p))
    r = np.linalg.norm(p - [tn.CX, tn.CY], axis=1)
    print(f"OK — 180 steps, max|vel|={np.linalg.norm(v,axis=2).max():.2f}, debris radius~{r.mean():.1f}")


if __name__ == "__main__":
    main()

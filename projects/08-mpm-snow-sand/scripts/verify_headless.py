"""CI-safe sanity check: no window, CPU backend, snow+sand fall, settle, and get stirred."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import mpm_snow_sand as mpm


def main():
    mpm.init_sim(arch=ti.cpu)
    mpm.apply_seed(*mpm.default_seed(rng_seed=1))
    for _ in range(60 * mpm.SUBSTEPS):
        mpm.substep()
    for _ in range(20):
        mpm.substep(0.5, 0.15, 0.02, 0.01, stirring=True)
    pos = mpm.x.to_numpy()
    v = mpm.v.to_numpy()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(v))
    snow_w = np.ptp(pos[: mpm.N_PER_BLOCK, 0])
    sand_w = np.ptp(pos[mpm.N_PER_BLOCK :, 0])
    n = 60 * mpm.SUBSTEPS + 20
    print(f"OK — {n} substeps, snow width={snow_w:.3f}, sand width={sand_w:.3f}, pos in [{pos.min():.3f},{pos.max():.3f}]")


if __name__ == "__main__":
    main()

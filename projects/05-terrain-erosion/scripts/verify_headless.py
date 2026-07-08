"""CI-safe sanity check: no window, CPU backend, 200 weathering steps behave."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import terrain as tr


def main():
    tr.init_sim(arch=ti.cpu)
    tr.apply_seed(tr.fbm_terrain(tr.N, rng_seed=3))
    for _ in range(200):
        tr.step()
    hh, ww = tr.h.to_numpy(), tr.w.to_numpy()
    assert np.all(np.isfinite(hh)) and np.all(np.isfinite(ww))
    tr.render()
    assert np.all(np.isfinite(tr.pixels.to_numpy()))
    print(f"OK — 200 steps, h in [{hh.min():.3f},{hh.max():.3f}], total water={ww.sum():.1f}")


if __name__ == "__main__":
    main()

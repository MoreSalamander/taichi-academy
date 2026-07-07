"""CI-safe sanity check: no window, CPU backend, 200 steps must stay finite and alive."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import gray_scott as gs


def main():
    gs.init_sim(arch=ti.cpu)
    gs.apply_seed(gs.seed_pattern(gs.N, rng_seed=3))
    name, feed, kill = gs.PRESETS[0]
    for _ in range(200):
        gs.step(feed, kill)
    v = gs.v.to_numpy()
    assert np.all(np.isfinite(v)), "NaN/inf in V"
    assert v.max() > 0.1, "pattern died"
    gs.render(0)
    px = gs.pixels.to_numpy()
    assert np.all(np.isfinite(px)), "NaN/inf in pixels"
    print(f"OK — 200 steps of '{name}', v in [{v.min():.3f}, {v.max():.3f}], pattern alive")


if __name__ == "__main__":
    main()

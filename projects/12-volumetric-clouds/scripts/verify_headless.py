"""CI-safe sanity check: no window, CPU backend, a full orbit renders and reseeds cleanly."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import volumetric_clouds as vc


def main():
    vc.init_sim(arch=ti.cpu)
    vc.apply_seed(rng_seed=1)
    for i in range(12):
        theta = i * (2 * np.pi / 12)
        cx, cy, cz = vc.camera_position(theta)
        vc.render(cx, cy, cz, 0.5, 0.5, 0.2)
    px = vc.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    vc.apply_seed(rng_seed=2)
    cx, cy, cz = vc.camera_position(0.0)
    vc.render(cx, cy, cz, 0.5, 0.5, 0.2)
    px2 = vc.pixels.to_numpy()
    assert np.all(np.isfinite(px2))
    print(f"OK — full orbit + reseed rendered, pixels in [{px2.min():.3f},{px2.max():.3f}]")


if __name__ == "__main__":
    main()

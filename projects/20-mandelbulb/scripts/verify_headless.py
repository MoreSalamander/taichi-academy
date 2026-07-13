"""CI-safe sanity check: no window, CPU backend, the bulb renders far and near."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import mandelbulb as mb


def main():
    mb.init_sim(arch=ti.cpu)
    for radius in (2.2, mb.RADIUS_MIN):
        cx, cy, cz = mb.camera_position(0.7, radius)
        mb.render(cx, cy, cz, *mb.SUN, mb.EPS_BASE * radius)
        px = mb.pixels.to_numpy()
        assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
        assert px.sum() > 100.0
    print("OK — far and zoomed views both render structure")


if __name__ == "__main__":
    main()

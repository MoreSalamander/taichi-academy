"""CI-safe sanity check: no window, CPU backend, traffic flows, jams, and never collides."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import traffic as tr


def main():
    tr.init_sim(arch=ti.cpu)
    tr.seed_road(tr.START_CARS, rng_seed=1)
    for t in range(600):
        tr.step(t)
        tr.render_spacetime()
    act = tr.active.to_numpy() == 1
    p = tr.car_pos.to_numpy()[act]
    assert len(np.unique(p)) == act.sum(), "no two cars may share a cell"
    ms = tr.mean_speed()
    assert 0.0 < ms < tr.VMAX
    tr.render_ring(600, 1)
    for f in (tr.pixels, tr.spacetime):
        px = f.to_numpy()
        assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — 600 steps, {act.sum()} cars, mean speed {ms:.2f}, zero collisions")


if __name__ == "__main__":
    main()

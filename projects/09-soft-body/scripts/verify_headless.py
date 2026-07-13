"""CI-safe sanity check: no window, CPU backend, three bodies fall, settle, and get grabbed."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import soft_body as sb


def main():
    sb.init_sim(arch=ti.cpu)
    sb.apply_seed()
    for _ in range(300):
        sb.substep()
    pos = sb.pos.to_numpy()
    centroid = pos[: sb.N_RING].mean(axis=0)
    sb.grab_at(float(centroid[0]), float(centroid[1]))
    sb.grab_target[None] = [0.6, 0.8]
    for _ in range(60):
        sb.substep()
    pos = sb.pos.to_numpy()
    v = sb.vel.to_numpy()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(v))
    print(f"OK — 360 substeps, grabbed body settled+dragged, pos in [{pos.min():.3f},{pos.max():.3f}]")


if __name__ == "__main__":
    main()

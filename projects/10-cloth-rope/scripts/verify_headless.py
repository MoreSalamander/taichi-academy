"""CI-safe sanity check: no window, CPU backend, rope+cloth blow in the wind and get grabbed."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import cloth_rope as cr


def main():
    cr.init_sim(arch=ti.cpu)
    cr.apply_seed()
    for i in range(150):
        cr.substep(i * cr.DT, wind=cr.WIND)
    cr.grab_at(*cr.pos.to_numpy()[cr.ROPE_N + 5])
    cr.grab_target[None] = [0.7, 0.7]
    for i in range(40):
        cr.substep(i * cr.DT, wind=cr.WIND)
    p = cr.pos.to_numpy()
    assert np.all(np.isfinite(p))
    print(f"OK — 190 substeps, {cr.n_constraints[None]} constraints solved, pos in [{p.min():.3f},{p.max():.3f}]")


if __name__ == "__main__":
    main()

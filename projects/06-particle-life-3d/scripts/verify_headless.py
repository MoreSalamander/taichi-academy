"""CI-safe sanity check: no window, CPU backend, 300 steps of ecology behave."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import particle_life_3d as pl


def main():
    pl.init_sim(arch=ti.cpu)
    pl.apply_seed(pl.seed_particles(pl.NUM, pl.NSPEC, rng_seed=3), pl.rule_matrix(pl.NSPEC, rng_seed=3))
    for _ in range(300):
        pl.step()
    p, v = pl.pos.to_numpy(), pl.vel.to_numpy()
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(v))
    assert np.all(p >= 0.0) and np.all(p <= pl.WORLD)
    pl.update_colors()
    assert np.all(np.isfinite(pl.colors.to_numpy()))
    speed = np.linalg.norm(v, axis=1)
    print(f"OK — 300 steps, {pl.NUM} particles, mean speed={speed.mean():.3f}, pos in [{p.min():.3f},{p.max():.3f}]")


if __name__ == "__main__":
    main()

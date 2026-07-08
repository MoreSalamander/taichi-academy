"""CI-safe sanity check: no window, CPU backend, a small storm stays finite."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import lightning as lt


def main():
    lt.init_sim(arch=ti.cpu)
    for k in range(6):
        lt.strike(0.15 * (k + 1), rng_seed=k)
        for _ in range(30):
            lt.step()
    b, g = lt.bolt.to_numpy(), lt.glow.to_numpy()
    assert np.all(np.isfinite(b)) and np.all(np.isfinite(g))
    lt.render(0.5)
    assert np.all(np.isfinite(lt.pixels.to_numpy()))
    print(f"OK — 6 strikes, bolt_max={b.max():.3f}, glow_max={g.max():.3f}")


if __name__ == "__main__":
    main()

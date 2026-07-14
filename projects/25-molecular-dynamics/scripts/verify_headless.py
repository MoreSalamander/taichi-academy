"""CI-safe sanity check: no window, CPU backend, a liquid melts, then cools to a crystal."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import molecular_dynamics as md


def main():
    md.init_sim(arch=ti.cpu)
    md.apply_seed(rng_seed=2, temperature=1.0)
    md.temp_target[None] = 3.0
    for _ in range(200):
        md.step()
    hot = md.crystalline_fraction()
    for target in np.linspace(2.5, 0.1, 12):
        md.temp_target[None] = target
        for _ in range(50):
            md.step()
    cold = md.crystalline_fraction()
    p = md.pos.to_numpy()
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() < md.L
    assert cold > hot * 1.8, f"cooling failed to crystallize: {hot:.2f} -> {cold:.2f}"
    md.render()
    px = md.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — melted (crystalline {hot*100:.0f}%) then froze to a crystal ({cold*100:.0f}%)")


if __name__ == "__main__":
    main()

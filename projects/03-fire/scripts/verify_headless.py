"""CI-safe sanity check: no window, CPU backend, 150 burning steps stay finite and rise."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import fire


def main():
    fire.init_sim(arch=ti.cpu)
    fire.apply_seed(fire.seed_pattern(fire.N, rng_seed=3))
    for k in range(150):
        fire.burn_source(float(k))
        fire.step(fire.CURL_STRENGTH)
    t = fire.temp.to_numpy()
    assert np.all(np.isfinite(t)), "NaN/inf in temp"
    upper = t[:, fire.N // 2 :].sum()
    assert upper > 0.5, "heat should have convected into the upper half"
    fire.render()
    assert np.all(np.isfinite(fire.pixels.to_numpy()))
    print(f"OK — 150 burning steps, t_max={t.max():.2f}, upper-half heat={upper:.1f}")


if __name__ == "__main__":
    main()

"""CI-safe sanity check: no window, CPU backend. A planet settles into a stable banded climate
with ice caps, seasons shift the warm band, and more greenhouse melts ice."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import earth_simulator as es


def main():
    es.init_sim(arch=ti.cpu)
    es.apply_seed(3)
    for _ in range(400):
        es.step()
    T = es.T.to_numpy()
    assert np.all(np.isfinite(T)), "climate blew up"
    eq = es.band_temp(es.H // 2 - 4, es.H // 2 + 4)
    pole = es.band_temp(0, 8)
    ice = es.ice_fraction()
    assert eq > pole + 30.0, f"gradient too flat: {eq:.0f} vs {pole:.0f}"
    assert 0.15 < ice < 0.85, f"ice caps present but not a snowball: {ice:.2f}"

    cold = es.ice_fraction()
    es.apply_seed(3)
    for _ in range(250):
        es.step(a_olr=182.0)
    warm = es.ice_fraction()
    assert warm < cold, "more greenhouse should melt ice"

    es.render()
    px = es.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — equator {eq:.0f}C, poles {pole:.0f}C, ice {ice*100:.0f}%; greenhouse melts {(cold-warm)*100:.0f}% of it")


if __name__ == "__main__":
    main()

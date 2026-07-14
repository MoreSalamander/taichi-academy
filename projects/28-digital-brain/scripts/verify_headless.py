"""CI-safe sanity check: no window, CPU backend. A pacemaker lights the sheet into sustained
travelling waves, and the driven site learns (its weights outclimb a distant corner)."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import digital_brain as db


def main():
    db.init_sim(arch=ti.cpu)
    db.apply_seed(0)
    c = db.N // 2
    for _ in range(150):
        db.step(c, c, 5.0, 0.7)

    rate = db.firing_rate()
    assert 0.02 < rate < 0.6, f"the sheet should hum with sustained waves, got {rate:.3f}"

    center = db.mean_weight(c - 12, c + 12, c - 12, c + 12)
    corner = db.mean_weight(0, 20, 0, 20)
    assert center > corner + 0.1, f"the pacemaker should learn: {center:.3f} vs corner {corner:.3f}"

    assert np.all(np.isfinite(db.p.to_numpy()))
    db.render()
    px = db.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — {rate*100:.1f}% firing in steady waves; pacemaker weight {center:.2f} vs corner {corner:.2f}")


if __name__ == "__main__":
    main()

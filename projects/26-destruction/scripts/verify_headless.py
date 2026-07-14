"""CI-safe sanity check: no window, CPU backend. A city stands, an explosion tears a hole,
then an earthquake shears what is left into rubble."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import destruction as dz


def main():
    dz.init_sim(arch=ti.cpu)
    dz.apply_seed()
    for _ in range(120):
        dz.step()
    standing = dz.broken_fraction()
    assert standing == 0.0, f"the city should not self-destruct while standing (broke {standing:.2%})"

    dz.explode(0.53, 0.12, dz.EXPLODE_POWER, dz.EXPLODE_RADIUS)
    for _ in range(60):
        dz.step()
    blasted = dz.broken_fraction()
    assert blasted > 0.05, f"the explosion should tear a real hole, only broke {blasted:.2%}"

    for f in range(240):
        dz.quake(f * dz.DT, dz.QUAKE_AMP)
        dz.step()
    quaked = dz.broken_fraction()
    assert quaked > blasted, "the quake should keep shearing bonds loose"

    p = dz.pos.to_numpy()[: dz.n_p[None]]
    assert np.all(np.isfinite(p)) and p[:, 1].min() >= dz.FLOOR
    dz.render()
    px = dz.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — stood intact, explosion ruined {blasted*100:.0f}%, quake pushed it to {quaked*100:.0f}%")


if __name__ == "__main__":
    main()

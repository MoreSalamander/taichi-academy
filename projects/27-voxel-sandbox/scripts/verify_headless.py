"""CI-safe sanity check: no window, CPU backend. Sand piles, water levels, lava quenches to
stone on water, and fire eats a wood block — all from local cellular-automaton rules."""

import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import voxel_sandbox as vs


def put(a, cells, m, lifev=0):
    la = vs.life.to_numpy()
    for (i, j) in cells:
        a[i, j] = m
        la[i, j] = lifev
    vs.mat.from_numpy(a)
    vs.life.from_numpy(la)


def main():
    vs.init_sim(arch=ti.cpu)
    vs.apply_seed()
    a = vs.mat.to_numpy()
    put(a, [(i, 3) for i in range(40, 120)], vs.WALL)              # a shelf
    put(a, [(i, 4) for i in range(50, 70)], vs.LAVA)
    put(a, [(i, 4) for i in range(70, 90)], vs.WATER)
    wood = [(i, j) for i in range(95, 115) for j in range(4, 24)]
    put(a, wood, vs.WOOD)
    put(a, [(105, 13)], vs.FIRE, lifev=vs.FIRE_LIFE)
    # a sand dune dropped from a point
    put(vs.mat.to_numpy(), [(150, j) for j in range(4, 60)], vs.SAND)

    sand0, wood0, lava0 = vs.count(vs.SAND), vs.count(vs.WOOD), vs.count(vs.LAVA)
    for f in range(400):
        vs.step(f)

    assert vs.count(vs.SAND) == sand0, "sand is conserved (no duplication or loss)"
    assert vs.count(vs.STONE) > 0, "lava quenched by water became stone"
    assert vs.count(vs.LAVA) < lava0, "lava was consumed"
    assert vs.count(vs.WOOD) < wood0 * 0.5, "fire ate most of the wood"

    floor_sand = int((vs.mat.to_numpy()[:, 4] == vs.SAND).sum())
    assert floor_sand > 3, "the sand column slumped into a wider pile"

    m = vs.mat.to_numpy()
    assert np.all((m >= 0) & (m < vs.N_MAT)), "every cell holds a valid material"
    vs.render()
    px = vs.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0
    print(f"OK — sand pile {floor_sand} wide, {vs.count(vs.STONE)} stone cast, "
          f"wood {wood0} -> {vs.count(vs.WOOD)}")


if __name__ == "__main__":
    main()

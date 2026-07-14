import numpy as np

import voxel_sandbox as vs


def blank():
    """A grid of pure EMPTY (no walls), so a test can isolate one behaviour."""
    m = np.full((vs.W, vs.H), vs.EMPTY, dtype=np.int32)
    vs.mat.from_numpy(m)
    vs.life.from_numpy(np.zeros((vs.W, vs.H), np.int32))


def put(cells, m, lifev=0):
    a = vs.mat.to_numpy()
    la = vs.life.to_numpy()
    for (i, j) in cells:
        a[i, j] = m
        la[i, j] = lifev
    vs.mat.from_numpy(a)
    vs.life.from_numpy(la)


def run(n):
    for f in range(n):
        vs.step(f)


# --- the seed ------------------------------------------------------------------------


def test_seed_builds_a_box():
    m = vs.mat.to_numpy()
    assert (m[:, 0] == vs.WALL).all() and (m[:, 2] == vs.WALL).all(), "solid floor"
    assert (m[0, :] == vs.WALL).all() and (m[-1, :] == vs.WALL).all(), "side walls"
    assert (m[10:-10, 3:] == vs.EMPTY).all(), "empty air inside"


# --- gravity: the anti-striping guarantee --------------------------------------------


def test_a_solid_column_falls_as_one_block():
    """The whole point of the serial column sweep: a solid column of sand drops by exactly one
    cell per step with NO gaps — the classic parallel-CA striping bug does not happen."""
    blank()
    col = [(50, j) for j in range(100, 120)]
    put(col, vs.SAND)
    vs.fall_columns()  # isolate straight gravity (full step() would also spread it sideways)
    js = np.where(vs.mat.to_numpy()[50] == vs.SAND)[0]
    assert len(js) == 20, "no grains created or destroyed"
    assert js.min() == 99 and js.max() == 118, "column shifted down exactly one"
    assert (np.diff(js) == 1).all(), "still contiguous — no venetian-blind gaps"


def test_sand_is_conserved_over_time():
    blank()
    put([(i, j) for i in range(40, 60) for j in range(150, 170)], vs.SAND)
    before = vs.count(vs.SAND)
    run(120)
    assert vs.count(vs.SAND) == before, "movement neither duplicates nor deletes sand"


def test_sand_settles_into_a_pile():
    """Dropped as a tall thin column onto the floor, sand spreads into a wider pile — the
    diagonal spread pass giving it an angle of repose."""
    put([(128, j) for j in range(3, 60)], vs.SAND)
    run(200)
    rows = vs.mat.to_numpy()
    floor = rows[:, 3]
    width = int((floor == vs.SAND).sum())
    assert width > 3, f"a 1-wide column should slump into a wider base, got {width}"


# --- water flows and levels ----------------------------------------------------------


def test_water_spreads_sideways():
    put([(128, j) for j in range(3, 40)], vs.WATER)
    before = vs.mat.to_numpy()
    cols_before = np.unique(np.where(before == vs.WATER)[0])
    run(200)
    after = vs.mat.to_numpy()
    cols_after = np.unique(np.where(after == vs.WATER)[0])
    assert len(cols_after) > len(cols_before) + 5, "water should puddle out across many columns"


# --- walls & static materials --------------------------------------------------------


def test_walls_never_move():
    m0 = vs.mat.to_numpy().copy()
    run(60)
    assert (vs.mat.to_numpy()[m0 == vs.WALL] == vs.WALL).all(), "walls are immovable"


def test_stone_is_static():
    blank()
    put([(100, 100)], vs.STONE)
    run(30)
    assert vs.mat.to_numpy()[100, 100] == vs.STONE, "stone does not fall or flow"


# --- density: heavy sinks through light ----------------------------------------------


def test_sand_sinks_through_water():
    blank()
    water = [(i, j) for i in range(40, 60) for j in range(20, 40)]
    sand = [(i, j) for i in range(45, 55) for j in range(40, 45)]  # sand sitting on top
    put(water, vs.WATER)
    put(sand, vs.SAND)
    run(200)
    m = vs.mat.to_numpy()
    sand_rows = np.where(m == vs.SAND)[1]
    water_rows = np.where(m == vs.WATER)[1]
    assert sand_rows.mean() < water_rows.mean(), "denser sand ends up below the water"


# --- reactions -----------------------------------------------------------------------


def test_lava_and_water_make_stone():
    blank()
    put([(i, 3) for i in range(40, 80)], vs.WALL)          # a shelf to hold them
    put([(i, 4) for i in range(40, 60)], vs.LAVA)
    put([(i, 4) for i in range(60, 80)], vs.WATER)
    lava0, water0 = vs.count(vs.LAVA), vs.count(vs.WATER)
    run(80)
    assert vs.count(vs.STONE) > 0, "lava quenched by water turns to stone"
    assert vs.count(vs.LAVA) < lava0, "lava is consumed"


def test_fire_burns_wood_down():
    blank()
    put([(i, 3) for i in range(40, 80)], vs.WALL)
    wood = [(i, j) for i in range(50, 70) for j in range(4, 24)]
    put(wood, vs.WOOD)
    put([(60, 13)], vs.FIRE, lifev=vs.FIRE_LIFE)
    wood0 = vs.count(vs.WOOD)
    run(300)
    assert vs.count(vs.WOOD) < wood0 * 0.5, f"fire should eat most of the wood, {wood0} -> {vs.count(vs.WOOD)}"


def test_wood_alone_does_not_burn():
    blank()
    put([(i, j) for i in range(50, 60) for j in range(50, 60)], vs.WOOD)
    wood0 = vs.count(vs.WOOD)
    run(60)
    assert vs.count(vs.WOOD) == wood0, "wood with no flame is inert"


def test_water_douses_fire():
    blank()
    put([(50, 50)], vs.FIRE, lifev=vs.FIRE_LIFE)
    put([(51, 50), (49, 50), (50, 51), (50, 49)], vs.WATER)
    vs.react()
    assert vs.mat.to_numpy()[50, 50] != vs.FIRE, "a flame surrounded by water goes out"


# --- render --------------------------------------------------------------------------


def test_render_is_finite_and_bounded():
    put([(128, j) for j in range(3, 40)], vs.WATER)
    run(30)
    vs.render()
    px = vs.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

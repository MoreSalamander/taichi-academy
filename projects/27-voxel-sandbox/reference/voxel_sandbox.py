"""Voxel Sandbox: a cellular automaton where sand, water, lava, and fire fall, flow, burn, and react."""

import numpy as np
import taichi as ti

W, H = 256, 256
NP = W * H
BRUSH = 6

# material ids
EMPTY, WALL, SAND, WATER, WOOD, FIRE, LAVA, STONE, SMOKE = range(9)
N_MAT = 9

FIRE_LIFE = 55          # frames a fire burns before dying
SMOKE_LIFE = 60
IGNITE_CHANCE = 0.22    # per-frame chance a flammable cell next to fire catches
LAVA_VISCOSITY = 0.6    # chance lava skips its move this frame (flows slowly)

mat = None
mat2 = None
life = None
life2 = None
tgt = None
src_of = None
density = None
flammable = None
color = None
pixels = None


def init_sim(arch=None):
    """Start Taichi, allocate every field once (Metal can't free fields), fill the lookup tables."""
    global mat, mat2, life, life2, tgt, src_of, density, flammable, color, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    mat = ti.field(ti.i32, shape=(W, H))
    mat2 = ti.field(ti.i32, shape=(W, H))
    life = ti.field(ti.i32, shape=(W, H))
    life2 = ti.field(ti.i32, shape=(W, H))
    tgt = ti.field(ti.i32, shape=(W, H))
    src_of = ti.field(ti.i32, shape=(W, H))
    density = ti.field(ti.i32, shape=N_MAT)
    flammable = ti.field(ti.i32, shape=N_MAT)
    color = ti.Vector.field(3, ti.f32, shape=N_MAT)
    pixels = ti.Vector.field(3, ti.f32, shape=(W, H))
    _fill_tables()


def _fill_tables():
    # density: heavier sinks through lighter; only fluids and sand take part (0 = doesn't sink/swap)
    d = np.zeros(N_MAT, np.int32)
    d[SMOKE], d[FIRE], d[WATER], d[LAVA], d[SAND] = 1, 1, 5, 7, 9
    density.from_numpy(d)
    f = np.zeros(N_MAT, np.int32)
    f[WOOD] = 1
    flammable.from_numpy(f)
    c = np.zeros((N_MAT, 3), np.float32)
    c[EMPTY] = (0.05, 0.06, 0.09)
    c[WALL] = (0.30, 0.30, 0.34)
    c[SAND] = (0.85, 0.72, 0.40)
    c[WATER] = (0.20, 0.45, 0.85)
    c[WOOD] = (0.45, 0.30, 0.16)
    c[FIRE] = (1.00, 0.55, 0.15)
    c[LAVA] = (0.95, 0.35, 0.12)
    c[STONE] = (0.38, 0.36, 0.35)
    c[SMOKE] = (0.50, 0.50, 0.52)
    color.from_numpy(c)


@ti.kernel
def clear_all():
    for i, j in mat:
        mat[i, j] = EMPTY
        life[i, j] = 0


@ti.kernel
def build_walls():
    for i, j in mat:
        if j < 3 or i < 2 or i >= W - 2:
            mat[i, j] = WALL


def apply_seed():
    """A clean box: solid floor and side walls, empty air inside — ready to be painted."""
    clear_all()
    build_walls()


# --- movement rules ------------------------------------------------------------------


@ti.func
def is_faller(m):
    return m == SAND or m == WATER or m == LAVA


@ti.func
def is_riser(m):
    return m == SMOKE


@ti.func
def is_mover(m):
    return m == SAND or m == WATER or m == LAVA or m == SMOKE


@ti.func
def empty_at(i, j):
    r = 0
    if 0 <= i < W and 0 <= j < H:
        if mat[i, j] == EMPTY:
            r = 1
    return r


@ti.kernel
def fall_columns():
    """Straight gravity, one serial sweep per column. Columns never touch each other, so this
    is race-free — and sweeping bottom-up drops a whole solid column by one cell, no gaps."""
    for i in range(W):
        for j in range(1, H):
            m = mat[i, j]
            if is_faller(m) and mat[i, j - 1] == EMPTY:
                mat[i, j - 1] = m
                life[i, j - 1] = life[i, j]
                mat[i, j] = EMPTY
                life[i, j] = 0


@ti.kernel
def rise_columns():
    """Smoke floats up: the same serial column sweep, scanned top-down."""
    for i in range(W):
        for jj in range(1, H):
            j = H - 1 - jj
            m = mat[i, j]
            if is_riser(m) and mat[i, j + 1] == EMPTY:
                mat[i, j + 1] = m
                life[i, j + 1] = life[i, j]
                mat[i, j] = EMPTY
                life[i, j] = 0


@ti.kernel
def select_targets():
    """Each mover picks ONE diagonal/sideways target (straight up/down is the column sweep's job,
    so we only spread when the vertical path is already blocked). Target must be EMPTY."""
    for i, j in mat:
        tgt[i, j] = -1
        src_of[i, j] = NP
        m = mat[i, j]
        if is_mover(m):
            d = 0
            rnd = ti.random()
            if m == SAND and not empty_at(i, j - 1):
                if empty_at(i - 1, j - 1) and empty_at(i + 1, j - 1):
                    d = 2 if rnd < 0.5 else 3
                elif empty_at(i - 1, j - 1):
                    d = 2
                elif empty_at(i + 1, j - 1):
                    d = 3
            elif (m == WATER or m == LAVA) and not empty_at(i, j - 1):
                if not (m == LAVA and rnd < LAVA_VISCOSITY):
                    if empty_at(i - 1, j - 1):
                        d = 2
                    elif empty_at(i + 1, j - 1):
                        d = 3
                    elif empty_at(i - 1, j) and empty_at(i + 1, j):
                        d = 4 if rnd < 0.5 else 5
                    elif empty_at(i - 1, j):
                        d = 4
                    elif empty_at(i + 1, j):
                        d = 5
            elif is_riser(m) and not empty_at(i, j + 1):
                if empty_at(i - 1, j + 1):
                    d = 7
                elif empty_at(i + 1, j + 1):
                    d = 8
            ti_, tj_ = i, j
            if d == 2:
                ti_, tj_ = i - 1, j - 1
            elif d == 3:
                ti_, tj_ = i + 1, j - 1
            elif d == 4:
                ti_ = i - 1
            elif d == 5:
                ti_ = i + 1
            elif d == 7:
                ti_, tj_ = i - 1, j + 1
            elif d == 8:
                ti_, tj_ = i + 1, j + 1
            if d != 0:
                tgt[i, j] = ti_ * H + tj_


@ti.kernel
def propose():
    """Many movers may want the same empty cell; the lowest flat index wins, deterministically."""
    for i, j in mat:
        t = tgt[i, j]
        if t >= 0:
            ti.atomic_min(src_of[t // H, t % H], i * H + j)


@ti.kernel
def resolve():
    """Write the next grid. Targets are only ever EMPTY cells, so 'receiving a mover' and
    'keeping my own material' can never collide on the same cell."""
    for i, j in mat:
        m = mat[i, j]
        if m == EMPTY:
            w = src_of[i, j]
            if w < NP:
                mat2[i, j] = mat[w // H, w % H]
                life2[i, j] = life[w // H, w % H]
            else:
                mat2[i, j] = EMPTY
                life2[i, j] = 0
        else:
            moved = 0
            t = tgt[i, j]
            if t >= 0 and src_of[t // H, t % H] == i * H + j:
                moved = 1
            mat2[i, j] = EMPTY if moved == 1 else m
            life2[i, j] = 0 if moved == 1 else life[i, j]


@ti.kernel
def commit():
    for i, j in mat:
        mat[i, j] = mat2[i, j]
        life[i, j] = life2[i, j]


def spread():
    select_targets()
    propose()
    resolve()
    commit()


@ti.kernel
def density_swap(parity: ti.i32):
    """Heavier fluids sink through lighter ones. Only rows of one parity act each call, so the
    (j, j-1) pairs never overlap — an in-place swap with no race."""
    for i, j in mat:
        if (j & 1) == parity and j >= 1:
            a = mat[i, j]
            b = mat[i, j - 1]
            if density[a] > 0 and density[b] > 0 and density[a] > density[b]:
                la, lb = life[i, j], life[i, j - 1]
                mat[i, j] = b
                mat[i, j - 1] = a
                life[i, j] = lb
                life[i, j - 1] = la


# --- reactions -----------------------------------------------------------------------


@ti.func
def touches(i, j, what) -> ti.i32:
    r = 0
    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        if not (di == 0 and dj == 0):
            ni, nj = i + di, j + dj
            if 0 <= ni < W and 0 <= nj < H:
                if mat[ni, nj] == what:
                    r = 1
    return r


@ti.kernel
def react():
    """Local chemistry, double-buffered so every cell reads the same old neighbourhood:
    fire ages and eats fuel, lava quenches to stone on water, water flashes to steam on lava."""
    for i, j in mat:
        m = mat[i, j]
        nm = m
        nl = life[i, j]
        if m == FIRE:
            nl = life[i, j] - 1
            if touches(i, j, WATER) == 1:
                nm, nl = EMPTY, 0                    # doused
            elif nl <= 0:
                if ti.random() < 0.35:
                    nm, nl = SMOKE, SMOKE_LIFE       # embers smoke
                else:
                    nm, nl = EMPTY, 0
        elif m == SMOKE:
            nl = life[i, j] - 1
            if nl <= 0:
                nm = EMPTY
        elif flammable[m] == 1:
            if touches(i, j, FIRE) == 1 or touches(i, j, LAVA) == 1:
                if ti.random() < IGNITE_CHANCE:
                    nm, nl = FIRE, FIRE_LIFE
        elif m == LAVA:
            if touches(i, j, WATER) == 1:
                nm = STONE                           # quenched to rock
        elif m == WATER:
            if touches(i, j, LAVA) == 1 and ti.random() < 0.5:
                nm, nl = SMOKE, SMOKE_LIFE           # flashed to steam
        mat2[i, j] = nm
        life2[i, j] = nl
    for i, j in mat:
        mat[i, j] = mat2[i, j]
        life[i, j] = life2[i, j]


def step(parity=0):
    react()
    fall_columns()
    rise_columns()
    spread()
    density_swap(parity & 1)
    density_swap(1 - (parity & 1))


@ti.kernel
def paint(cx: ti.i32, cy: ti.i32, m: ti.i32):
    for i, j in mat:
        if (i - cx) ** 2 + (j - cy) ** 2 < BRUSH * BRUSH:
            if mat[i, j] != WALL or m == EMPTY:
                mat[i, j] = m
                life[i, j] = FIRE_LIFE if m == FIRE else 0


def count(m):
    """Pure numpy: how many cells currently hold material m."""
    return int((mat.to_numpy() == m).sum())


@ti.kernel
def render():
    for i, j in pixels:
        m = mat[i, j]
        col = color[m]
        if m == FIRE:
            col = color[FIRE] * (0.55 + 0.45 * ti.random())
        elif m == SMOKE:
            col = color[SMOKE] * (0.4 + 0.6 * life[i, j] / SMOKE_LIFE)
        pixels[i, j] = col


PALETTE = [(SAND, "sand"), (WATER, "water"), (WOOD, "wood"), (FIRE, "fire"),
           (LAVA, "lava"), (STONE, "stone"), (WALL, "wall"), (EMPTY, "erase")]


def main():
    init_sim()
    apply_seed()
    # a little scene to play with: a wood platform over a sand dune
    for x in range(90, 170):
        paint(x, 120, WOOD)
    for x in range(60, 110):
        paint(x, 20, SAND)
    gui = ti.GUI("Voxel Sandbox — taichi-academy", res=(W, H), background_color=0x0D0F17)
    brush = SAND
    frame = 0
    while gui.running:
        frame += 1
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "12345678":
                brush = PALETTE[int(e.key) - 1][0]
            elif e.key == "c":
                apply_seed()
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            paint(int(mx * W), int(my * H), brush)
        step(frame)
        render()
        gui.set_image(pixels)
        name = dict(PALETTE).get(brush, "?")
        gui.text(f"brush: {name}   [1-8] pick  drag: paint  [c] clear", (0.02, 0.98), color=0xFFFFFF)
        gui.show()


if __name__ == "__main__":
    main()

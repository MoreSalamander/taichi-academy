"""Cloth & Rope: Position-Based Dynamics — Verlet integration plus distance constraints."""

import numpy as np
import taichi as ti

ROPE_N = 40
CLOTH_W, CLOTH_H = 24, 16
CLOTH_N = CLOTH_W * CLOTH_H
N = ROPE_N + CLOTH_N
ROPE_BASE = 0
CLOTH_BASE = ROPE_N

MAX_CONSTRAINTS = 4000

GRAVITY = 9.8
DT = 1.0 / 60
ITERS = 6
DAMPING = 0.995
WORLD = 1.0
WIND = 6.0

SPACING = 0.02
CLOTH_ORIGIN = (0.4, 0.92)
ROPE_ORIGIN = (0.15, 0.9)

GRAB_RADIUS = 0.05

pos = None
prev_pos = None
inv_mass = None
c_a = None
c_b = None
c_len = None
n_constraints = None
grabbed = None
grab_target = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, prev_pos, inv_mass, c_a, c_b, c_len, n_constraints
    global grabbed, grab_target
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    prev_pos = ti.Vector.field(2, ti.f32, shape=N)
    inv_mass = ti.field(ti.f32, shape=N)
    c_a = ti.field(ti.i32, shape=MAX_CONSTRAINTS)
    c_b = ti.field(ti.i32, shape=MAX_CONSTRAINTS)
    c_len = ti.field(ti.f32, shape=MAX_CONSTRAINTS)
    n_constraints = ti.field(ti.i32, shape=())
    grabbed = ti.field(ti.i32, shape=())
    grab_target = ti.Vector.field(2, ti.f32, shape=())


def idx_cloth(i, j):
    return CLOTH_BASE + j * CLOTH_W + i


def build_rope():
    """Pure numpy: a diagonal chain of points, and the edges between consecutive links."""
    ox, oy = ROPE_ORIGIN
    pts = np.array(
        [[ox + 0.006 * i, oy - 0.012 * i] for i in range(ROPE_N)], dtype=np.float32
    )
    link = float(np.linalg.norm(pts[1] - pts[0]))
    edges = [(ROPE_BASE + i, ROPE_BASE + i + 1, link) for i in range(ROPE_N - 1)]
    return pts, edges


def build_cloth():
    """Pure numpy: a WxH grid of points, plus structural and shear edges."""
    ox, oy = CLOTH_ORIGIN
    pts = np.zeros((CLOTH_N, 2), dtype=np.float32)
    for j in range(CLOTH_H):
        for i in range(CLOTH_W):
            pts[j * CLOTH_W + i] = [ox + i * SPACING, oy - j * SPACING]
    edges = []
    for j in range(CLOTH_H):
        for i in range(CLOTH_W):
            if i + 1 < CLOTH_W:
                edges.append((idx_cloth(i, j), idx_cloth(i + 1, j), SPACING))
            if j + 1 < CLOTH_H:
                edges.append((idx_cloth(i, j), idx_cloth(i, j + 1), SPACING))
            if i + 1 < CLOTH_W and j + 1 < CLOTH_H:
                diag = SPACING * 2**0.5
                edges.append((idx_cloth(i, j), idx_cloth(i + 1, j + 1), diag))
                edges.append((idx_cloth(i + 1, j), idx_cloth(i, j + 1), diag))
    return pts, edges


def apply_seed():
    rope_pts, rope_edges = build_rope()
    cloth_pts, cloth_edges = build_cloth()
    positions = np.concatenate([rope_pts, cloth_pts], axis=0)

    im = np.ones(N, dtype=np.float32)
    im[ROPE_BASE] = 0.0
    for j in range(CLOTH_H):
        im[idx_cloth(0, j)] = 0.0

    edges = rope_edges + cloth_edges
    ea = np.array([e[0] for e in edges], dtype=np.int32)
    eb = np.array([e[1] for e in edges], dtype=np.int32)
    el = np.array([e[2] for e in edges], dtype=np.float32)

    pos.from_numpy(positions)
    prev_pos.from_numpy(positions)
    inv_mass.from_numpy(im)
    c_a.from_numpy(np.pad(ea, (0, MAX_CONSTRAINTS - len(ea))))
    c_b.from_numpy(np.pad(eb, (0, MAX_CONSTRAINTS - len(eb))))
    c_len.from_numpy(np.pad(el, (0, MAX_CONSTRAINTS - len(el))))
    n_constraints[None] = len(edges)
    grabbed[None] = -1


@ti.kernel
def predict(t: ti.f32, wind: ti.f32):
    for p in pos:
        if p == grabbed[None]:
            prev_pos[p] = pos[p]
            pos[p] = grab_target[None]
        elif inv_mass[p] > 0:
            vel = (pos[p] - prev_pos[p]) * DAMPING
            prev_pos[p] = pos[p]
            g = ti.Vector([wind * ti.sin(t * 3.0 + p * 0.15), -GRAVITY])
            pos[p] = pos[p] + vel + g * DT * DT


@ti.kernel
def solve_constraints():
    for _ in range(1):
        for c in range(n_constraints[None]):
            a, b, rest = c_a[c], c_b[c], c_len[c]
            d = pos[b] - pos[a]
            dist = d.norm() + 1e-6
            wa, wb = inv_mass[a], inv_mass[b]
            if a == grabbed[None]:
                wa = 0.0
            if b == grabbed[None]:
                wb = 0.0
            wsum = wa + wb
            if wsum > 0:
                corr = (dist - rest) / dist * d / wsum
                pos[a] += wa * corr
                pos[b] -= wb * corr


@ti.kernel
def apply_bounds():
    for p in pos:
        if inv_mass[p] > 0 and p != grabbed[None]:
            if pos[p][0] < 0.0:
                pos[p][0] = 0.0
            if pos[p][0] > WORLD:
                pos[p][0] = WORLD
            if pos[p][1] < 0.0:
                pos[p][1] = 0.0


def substep(t=0.0, wind=WIND):
    predict(t, wind)
    for _ in range(ITERS):
        solve_constraints()
    apply_bounds()


def grab_at(mx, my):
    p = pos.to_numpy()
    im = inv_mass.to_numpy()
    d2 = (p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2
    d2[im == 0.0] = 1e9
    i = int(np.argmin(d2))
    if d2[i] < GRAB_RADIUS**2:
        grabbed[None] = i
    else:
        grabbed[None] = -1


def release():
    grabbed[None] = -1


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Cloth & Rope — taichi-academy", res=512, background_color=0x0A0A12)
    frame = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()
            elif e.key == ti.GUI.LMB:
                grab_at(*gui.get_cursor_pos())
        for e in gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                release()
        if gui.is_pressed(ti.GUI.LMB) and grabbed[None] >= 0:
            grab_target[None] = gui.get_cursor_pos()
        substep(frame * DT)
        frame += 1
        p = pos.to_numpy()
        gui.circles(p[:ROPE_N], radius=2, color=0xE8544A)
        gui.circles(p[ROPE_N:], radius=1.5, color=0x8EC9FF)
        gui.text("drag to grab  [r] reset", (0.02, 0.98), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

"""Code SOT for project 10 — cloth & rope.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 10-cloth-rope`.

Evolutions: predict/substep each gain a wind term (chapter 3) and then grab
support (chapter 4); solve_constraints and apply_bounds gain grab support in
chapter 4 only. Unlike projects 06/08/09, solve_constraints is a deliberately
SERIAL pass from the moment it's written — the reference implementation's
development hit a real GPU race condition from a parallel version, so this
project never teaches the broken form at all, only the fixed one, with the
reasoning explained up front.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="10-cloth-rope",
    default_file="cloth_rope.py",
    reference={"cloth_rope.py": PROJECT_DIR / "reference" / "cloth_rope.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Cloth & Rope: Position-Based Dynamics — Verlet integration plus distance constraints."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "ROPE_N = 40"))
frag(((1, 2), "CLOTH_W, CLOTH_H = 24, 16"))
frag(((1, 2), "CLOTH_N = CLOTH_W * CLOTH_H"))
frag(((1, 2), "N = ROPE_N + CLOTH_N"))
frag(((1, 2), "ROPE_BASE = 0"))
frag(((1, 2), "CLOTH_BASE = ROPE_N"))

frag(((1, 2), "MAX_CONSTRAINTS = 4000"))

frag(((2, 1), "GRAVITY = 9.8"))
frag(((2, 1), "DT = 1.0 / 60"))
frag(((2, 2), "ITERS = 6"))
frag(((2, 1), "DAMPING = 0.995"))
frag(((2, 2), "WORLD = 1.0"))
frag(((3, 1), "WIND = 6.0"))

frag(((1, 2), "SPACING = 0.02"))
frag(((1, 2), 'CLOTH_ORIGIN = (0.4, 0.92)'))
frag(((1, 2), 'ROPE_ORIGIN = (0.15, 0.9)'))

frag(((4, 1), "GRAB_RADIUS = 0.05"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "prev_pos = None"))
frag(((1, 2), "inv_mass = None"))
frag(((1, 2), "c_a = None"))
frag(((1, 2), "c_b = None"))
frag(((1, 2), "c_len = None"))
frag(((1, 2), "n_constraints = None"))
frag(((4, 1), "grabbed = None"))
frag(((4, 1), "grab_target = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    (
        (1, 2),
        f"def init_sim(arch=None):\n{DOC}\n    global pos, prev_pos, inv_mass, c_a, c_b, c_len, n_constraints",
    ),
    (
        (4, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, prev_pos, inv_mass, c_a, c_b, c_len, n_constraints\n"
        "    global grabbed, grab_target",
    ),
)
frag(
    (
        (1, 2),
        "    if arch is None:\n"
        "        try:\n"
        "            ti.init(arch=ti.gpu)\n"
        "        except Exception:\n"
        "            ti.init(arch=ti.cpu)\n"
        "    else:\n"
        "        ti.init(arch=arch)",
    )
)
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    prev_pos = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    inv_mass = ti.field(ti.f32, shape=N)"))
frag(((1, 2), "    c_a = ti.field(ti.i32, shape=MAX_CONSTRAINTS)"))
frag(((1, 2), "    c_b = ti.field(ti.i32, shape=MAX_CONSTRAINTS)"))
frag(((1, 2), "    c_len = ti.field(ti.f32, shape=MAX_CONSTRAINTS)"))
frag(((1, 2), "    n_constraints = ti.field(ti.i32, shape=())"))
frag(((4, 1), "    grabbed = ti.field(ti.i32, shape=())"))
frag(((4, 1), "    grab_target = ti.Vector.field(2, ti.f32, shape=())"))

# --- topology ----------------------------------------------------------------------

frag(((1, 3), "def idx_cloth(i, j):\n    return CLOTH_BASE + j * CLOTH_W + i"))

BUILD_ROPE = '''def build_rope():
    """Pure numpy: a diagonal chain of points, and the edges between consecutive links."""
    ox, oy = ROPE_ORIGIN
    pts = np.array(
        [[ox + 0.006 * i, oy - 0.012 * i] for i in range(ROPE_N)], dtype=np.float32
    )
    link = float(np.linalg.norm(pts[1] - pts[0]))
    edges = [(ROPE_BASE + i, ROPE_BASE + i + 1, link) for i in range(ROPE_N - 1)]
    return pts, edges'''

frag(((1, 3), BUILD_ROPE))

BUILD_CLOTH = '''def build_cloth():
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
    return pts, edges'''

frag(((1, 3), BUILD_CLOTH))

APPLY_V1 = """def apply_seed():
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
    n_constraints[None] = len(edges)"""

APPLY_V2 = APPLY_V1 + "\n    grabbed[None] = -1"

frag(((1, 3), APPLY_V1), ((4, 1), APPLY_V2))

# --- predict (Verlet) ----------------------------------------------------------------

PREDICT_V1 = """@ti.kernel
def predict():
    for p in pos:
        if inv_mass[p] > 0:
            vel = (pos[p] - prev_pos[p]) * DAMPING
            prev_pos[p] = pos[p]
            pos[p] = pos[p] + vel + ti.Vector([0.0, -GRAVITY]) * DT * DT"""

PREDICT_V2 = """@ti.kernel
def predict(t: ti.f32, wind: ti.f32):
    for p in pos:
        if inv_mass[p] > 0:
            vel = (pos[p] - prev_pos[p]) * DAMPING
            prev_pos[p] = pos[p]
            g = ti.Vector([wind * ti.sin(t * 3.0 + p * 0.15), -GRAVITY])
            pos[p] = pos[p] + vel + g * DT * DT"""

PREDICT_V3 = """@ti.kernel
def predict(t: ti.f32, wind: ti.f32):
    for p in pos:
        if p == grabbed[None]:
            prev_pos[p] = pos[p]
            pos[p] = grab_target[None]
        elif inv_mass[p] > 0:
            vel = (pos[p] - prev_pos[p]) * DAMPING
            prev_pos[p] = pos[p]
            g = ti.Vector([wind * ti.sin(t * 3.0 + p * 0.15), -GRAVITY])
            pos[p] = pos[p] + vel + g * DT * DT"""

frag(((2, 1), PREDICT_V1), ((3, 1), PREDICT_V2), ((4, 1), PREDICT_V3))

# --- solve (Gauss-Seidel, deliberately serial) ----------------------------------------

SOLVE_V1 = """@ti.kernel
def solve_constraints():
    for _ in range(1):
        for c in range(n_constraints[None]):
            a, b, rest = c_a[c], c_b[c], c_len[c]
            d = pos[b] - pos[a]
            dist = d.norm() + 1e-6
            wa, wb = inv_mass[a], inv_mass[b]
            wsum = wa + wb
            if wsum > 0:
                corr = (dist - rest) / dist * d / wsum
                pos[a] += wa * corr
                pos[b] -= wb * corr"""

SOLVE_V2 = """@ti.kernel
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
                pos[b] -= wb * corr"""

frag(((2, 2), SOLVE_V1), ((4, 1), SOLVE_V2))

BOUNDS_V1 = """@ti.kernel
def apply_bounds():
    for p in pos:
        if inv_mass[p] > 0:
            if pos[p][0] < 0.0:
                pos[p][0] = 0.0
            if pos[p][0] > WORLD:
                pos[p][0] = WORLD
            if pos[p][1] < 0.0:
                pos[p][1] = 0.0"""

BOUNDS_V2 = """@ti.kernel
def apply_bounds():
    for p in pos:
        if inv_mass[p] > 0 and p != grabbed[None]:
            if pos[p][0] < 0.0:
                pos[p][0] = 0.0
            if pos[p][0] > WORLD:
                pos[p][0] = WORLD
            if pos[p][1] < 0.0:
                pos[p][1] = 0.0"""

frag(((2, 2), BOUNDS_V1), ((4, 1), BOUNDS_V2))

frag(
    ((2, 3), "def substep():\n    predict()\n    for _ in range(ITERS):\n        solve_constraints()\n    apply_bounds()"),
    (
        (3, 2),
        "def substep(t=0.0, wind=WIND):\n"
        "    predict(t, wind)\n"
        "    for _ in range(ITERS):\n"
        "        solve_constraints()\n"
        "    apply_bounds()",
    ),
)

GRAB_AT = """def grab_at(mx, my):
    p = pos.to_numpy()
    im = inv_mass.to_numpy()
    d2 = (p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2
    d2[im == 0.0] = 1e9
    i = int(np.argmin(d2))
    if d2[i] < GRAB_RADIUS**2:
        grabbed[None] = i
    else:
        grabbed[None] = -1"""

frag(((4, 1), GRAB_AT))
frag(((4, 1), "def release():\n    grabbed[None] = -1"))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Cloth & Rope — taichi-academy", res=512, background_color=0x0A0A12)'))
frag(((3, 2), "    frame = 0"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()
            elif e.key == ti.GUI.LMB:
                grab_at(*gui.get_cursor_pos())'''

frag(((1, 3), EVENTS_V1), ((2, 3), EVENTS_V2), ((4, 2), EVENTS_V3))

RELEASE_EVENTS = """        for e in gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                release()"""

frag(((4, 2), RELEASE_EVENTS))

DRAG_UPDATE = """        if gui.is_pressed(ti.GUI.LMB) and grabbed[None] >= 0:
            grab_target[None] = gui.get_cursor_pos()"""

frag(((4, 3), DRAG_UPDATE))

frag(((2, 3), "        substep()"), ((3, 2), "        substep(frame * DT)"))
frag(((3, 2), "        frame += 1"))
frag(((1, 3), "        p = pos.to_numpy()"))
frag(((1, 3), "        gui.circles(p[:ROPE_N], radius=2, color=0xE8544A)"))
frag(((1, 3), "        gui.circles(p[ROPE_N:], radius=1.5, color=0x8EC9FF)"))
frag(((4, 3), '        gui.text("drag to grab  [r] reset", (0.02, 0.98), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

"""Code SOT for project 09 — soft body.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 09-soft-body`.

Evolutions: compute_forces grows in three passes — gravity+springs (chapter 2,
where bodies can visibly collapse flat), +internal pressure (chapter 3, the
fix), +an optional grab pull (chapter 4). apply_seed and init_sim grow the
same way, one field at a time, as each new piece of state is introduced.
integrate() and substep(), once written, never change again.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="09-soft-body",
    default_file="soft_body.py",
    reference={"soft_body.py": PROJECT_DIR / "reference" / "soft_body.py"},
    chapter_steps={1: 3, 2: 3, 3: 2, 4: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Soft Body: spring rings plus internal pressure — jelly, rubber, and a balloon, one engine."""'))
frag(((1, 1), "import math"))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "N_RING = 28"))
frag(((1, 2), "N_BODIES = 3"))
frag(((1, 2), "N = N_RING * N_BODIES"))
frag(((2, 1), "DT = 5e-4"))
frag(((2, 3), "SUBSTEPS = 30"))
frag(((2, 1), "GRAVITY = 9.8"))
frag(((2, 2), "WORLD = 1.0"))
frag(((1, 2), "BODY_RADIUS = 0.08"))
frag(((1, 2), "CENTERS = [(0.25, 0.6), (0.5, 0.6), (0.75, 0.6)]"))

frag(((1, 2), "STIFFNESS_NP = np.array([800.0, 4000.0, 300.0], dtype=np.float32)"))
frag(((1, 2), "DAMPING_NP = np.array([6.0, 2.0, 3.0], dtype=np.float32)"))
frag(((3, 1), "GAS_NP = np.array([1.5, 1.0, 2.5], dtype=np.float32)"))
frag(((1, 2), "MASS_NP = np.array([0.02, 0.02, 0.02], dtype=np.float32)"))

frag(((4, 1), "GRAB_K = 400.0"))
frag(((4, 1), "GRAB_DAMP = 4.0"))
frag(((4, 1), "GRAB_RADIUS = 0.1"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "vel = None"))
frag(((2, 1), "force = None"))
frag(((1, 2), "rest_len = None"))
frag(((1, 2), "body_id = None"))
frag(((1, 2), "stiffness = None"))
frag(((1, 2), "damping = None"))
frag(((3, 1), "gas = None"))
frag(((1, 2), "mass = None"))
frag(((4, 1), "grabbed = None"))
frag(((4, 1), "grab_target = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, rest_len, body_id, stiffness, damping, mass"),
    (
        (2, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, vel, force, rest_len, body_id, stiffness, damping, mass",
    ),
    (
        (3, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, vel, force, rest_len, body_id, stiffness, damping, gas, mass",
    ),
    (
        (4, 1),
        f"def init_sim(arch=None):\n{DOC}\n"
        "    global pos, vel, force, rest_len, body_id, stiffness, damping, gas, mass\n"
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
frag(((1, 2), "    vel = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((2, 1), "    force = ti.Vector.field(2, ti.f32, shape=N)"))
frag(((1, 2), "    rest_len = ti.field(ti.f32, shape=N)"))
frag(((1, 2), "    body_id = ti.field(ti.i32, shape=N)"))
frag(((1, 2), "    stiffness = ti.field(ti.f32, shape=N_BODIES)"))
frag(((1, 2), "    damping = ti.field(ti.f32, shape=N_BODIES)"))
frag(((3, 1), "    gas = ti.field(ti.f32, shape=N_BODIES)"))
frag(((1, 2), "    mass = ti.field(ti.f32, shape=N_BODIES)"))
frag(((4, 1), "    grabbed = ti.field(ti.i32, shape=())"))
frag(((4, 1), "    grab_target = ti.Vector.field(2, ti.f32, shape=())"))

# --- seeding -------------------------------------------------------------------------

SEED_RING = '''def seed_ring(cx, cy, radius, n):
    """Pure numpy: n points evenly spaced around a circle."""
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1).astype(np.float32)'''

frag(((1, 3), SEED_RING))

REST_LENGTHS = """def rest_lengths(ring):
    nxt = np.roll(ring, -1, axis=0)
    return np.linalg.norm(nxt - ring, axis=1).astype(np.float32)"""

frag(((1, 3), REST_LENGTHS))

APPLY_V1 = """def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    mass.from_numpy(MASS_NP)"""

APPLY_V2 = """def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    gas.from_numpy(GAS_NP)
    mass.from_numpy(MASS_NP)"""

APPLY_V3 = """def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    gas.from_numpy(GAS_NP)
    mass.from_numpy(MASS_NP)
    grabbed[None] = -1"""

frag(((1, 3), APPLY_V1), ((3, 1), APPLY_V2), ((4, 1), APPLY_V3))

# --- forces ----------------------------------------------------------------------------

FORCES_V1 = """@ti.kernel
def compute_forces():
    for p in pos:
        b = body_id[p]
        force[p] = ti.Vector([0.0, -GRAVITY * mass[b]])

    for p in pos:
        b = body_id[p]
        q = (p // N_RING) * N_RING + (p + 1) % N_RING
        d = pos[q] - pos[p]
        dist = d.norm() + 1e-6
        dirn = d / dist
        stretch = dist - rest_len[p]
        rel_v = (vel[q] - vel[p]).dot(dirn)
        f = (stiffness[b] * stretch + damping[b] * rel_v) * dirn
        force[p] += f
        force[q] -= f"""

PRESSURE_BLOCK = """
    for b in range(N_BODIES):
        area = 0.0
        base = b * N_RING
        for i in range(N_RING):
            p0, p1 = pos[base + i], pos[base + (i + 1) % N_RING]
            area += p0[0] * p1[1] - p1[0] * p0[1]
        area = ti.abs(area) * 0.5 + 1e-6
        pressure = gas[b] / area
        for i in range(N_RING):
            p0, p1 = pos[base + i], pos[base + (i + 1) % N_RING]
            edge = p1 - p0
            normal = ti.Vector([edge[1], -edge[0]])
            f = pressure * normal * 0.5
            force[base + i] += f
            force[base + (i + 1) % N_RING] += f"""

FORCES_V2 = FORCES_V1 + "\n" + PRESSURE_BLOCK.strip("\n")

GRAB_BLOCK = """

    if grabbed[None] >= 0:
        g = grabbed[None]
        pull = GRAB_K * (grab_target[None] - pos[g]) - GRAB_DAMP * vel[g]
        force[g] += pull"""

FORCES_V3 = FORCES_V2 + GRAB_BLOCK

frag(((2, 1), FORCES_V1), ((3, 2), FORCES_V2), ((4, 1), FORCES_V3))

INTEGRATE = """@ti.kernel
def integrate():
    for p in pos:
        b = body_id[p]
        vel[p] += DT * force[p] / mass[b]
        newp = pos[p] + DT * vel[p]
        if newp[1] < 0.0:
            newp[1] = 0.0
            vel[p][1] *= -0.3
            vel[p][0] *= 0.7
        if newp[0] < 0.0:
            newp[0] = 0.0
            vel[p][0] *= -0.3
        elif newp[0] > WORLD:
            newp[0] = WORLD
            vel[p][0] *= -0.3
        pos[p] = newp"""

frag(((2, 2), INTEGRATE))

frag(((2, 3), "def substep():\n    compute_forces()\n    integrate()"))

GRAB_AT = """def grab_at(mx, my):
    p = pos.to_numpy()
    d2 = (p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2
    i = int(np.argmin(d2))
    if d2[i] < GRAB_RADIUS**2:
        grabbed[None] = i
    else:
        grabbed[None] = -1"""

frag(((4, 1), GRAB_AT))
frag(((4, 1), "def release():\n    grabbed[None] = -1"))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed()"))
frag(((1, 3), '    gui = ti.GUI("Soft Body — taichi-academy", res=512, background_color=0x0A0A12)'))
frag(
    (
        (1, 3),
        "    colors = np.zeros(N, dtype=np.uint32)\n"
        "    colors[0:N_RING] = 0x8EC9FF\n"
        "    colors[N_RING : 2 * N_RING] = 0xE8544A\n"
        "    colors[2 * N_RING :] = 0xFFD35C",
    )
)
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

frag(((2, 3), "        for _ in range(SUBSTEPS):\n            substep()"))
frag(((1, 3), "        gui.circles(pos.to_numpy(), radius=2.5, color=colors)"))
frag(((4, 3), '        gui.text("drag a body to grab it  [r] reset", (0.02, 0.98), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

"""Code SOT for project 30 — universe sandbox.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 30-universe-sandbox`.

Arc: chapter 1 is Newton's law and a symplectic leapfrog — a single galaxy of stars orbiting a
central black hole, rock-stable. Chapter 2 loads two galaxies onto a collision course and adds
energy/survival diagnostics. Chapter 3 lets the player drag a heavy black hole through the scene
(the cursor term in the force sum) — the finished sandbox.

compute_acc and main both grow across chapters (the cursor force, the collide scene, the mouse),
so a couple of pieces are keyed later than where they sit in the file.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="30-universe-sandbox",
    default_file="universe_sandbox.py",
    reference={"universe_sandbox.py": PROJECT_DIR / "reference" / "universe_sandbox.py"},
    chapter_steps={1: 6, 2: 2, 3: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Universe Sandbox: direct N-body gravity with softening and a symplectic leapfrog — rotating\ngalaxies around black holes that orbit, collide, and fling out tidal streams of stars."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants + fields --------------------------------------------------------------

frag((
    (1, 2),
    "N = 4000                 # bodies (stars + a couple of black holes)\n"
    "G = 1.0\n"
    "EPS = 0.02               # softening length — smooths the 1/r^2 singularity at close range\n"
    "DT = 0.008\n"
    "RES = 512\n"
    "VIEW = 1.8               # half-width of the viewport in world units\n"
    "BH_MASS = 0.15           # a galaxy's central black hole dominates its potential\n"
    "STAR_MASS = 1e-5         # stars are near-massless tracers of the gravitational field",
))

for _name in ("pos", "vel", "acc", "mass", "pixels"):
    frag(((1, 2), f"{_name} = None"))

# --- init ----------------------------------------------------------------------------

INIT_SIM = '''def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, acc, mass, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    acc = ti.Vector.field(2, ti.f32, shape=N)
    mass = ti.field(ti.f32, shape=N)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))'''

frag(((1, 3), INIT_SIM))

# --- building a galaxy ---------------------------------------------------------------

MAKE_GALAXY = '''def make_galaxy(n, center, bulk_vel, spin=1.0, radius=0.22, seed=0):
    """Pure numpy: a black hole (index 0) ringed by a disk of stars on circular orbits, drifting
    at bulk_vel. Circular speed uses the SOFTENED central force, so inner orbits are not too fast."""
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 2), np.float32)
    v = np.zeros((n, 2), np.float32)
    m = np.full(n, STAR_MASS, np.float32)
    p[0], v[0], m[0] = center, bulk_vel, BH_MASS
    for i in range(1, n):
        r = radius * np.sqrt(rng.random()) + 0.03
        th = rng.random() * 2 * np.pi
        p[i] = [center[0] + r * np.cos(th), center[1] + r * np.sin(th)]
        vc = np.sqrt(G * BH_MASS * r * r / (r * r + EPS * EPS) ** 1.5)
        tang = np.array([-np.sin(th), np.cos(th)]) * spin
        v[i] = np.array(bulk_vel) + tang * vc
    return p, v, m'''

frag(((1, 4), MAKE_GALAXY))

# --- scenes (single in ch1, collide added in ch2) ------------------------------------

APPLY_SEED_V1 = '''def apply_seed(scene="single", seed=1):
    """Load a scene: one grand galaxy of stars orbiting a central black hole."""
    p, v, m = make_galaxy(N, [0.0, 0.0], [0.0, 0.0], spin=1.0, radius=0.35, seed=seed)
    pos.from_numpy(p)
    vel.from_numpy(v)
    mass.from_numpy(m)
    compute_acc(-1e9, -1e9, 0.0)'''

APPLY_SEED_V2 = '''def apply_seed(scene="collide", seed=1):
    """Load a scene: 'single' is one grand galaxy; 'collide' sends two onto a collision course."""
    if scene == "single":
        p, v, m = make_galaxy(N, [0.0, 0.0], [0.0, 0.0], spin=1.0, radius=0.35, seed=seed)
    else:
        h = N // 2
        p1, v1, m1 = make_galaxy(h, [-0.55, 0.14], [0.28, 0.0], spin=1.0, radius=0.22, seed=seed)
        p2, v2, m2 = make_galaxy(N - h, [0.55, -0.14], [-0.28, 0.0], spin=1.0, radius=0.22, seed=seed + 1)
        p, v, m = np.vstack([p1, p2]), np.vstack([v1, v2]), np.concatenate([m1, m2])
    pos.from_numpy(p)
    vel.from_numpy(v)
    mass.from_numpy(m)
    compute_acc(-1e9, -1e9, 0.0)'''

frag(((1, 5), APPLY_SEED_V1), ((2, 1), APPLY_SEED_V2))

# --- the force law (cursor term added in ch3) ----------------------------------------

ACC_V1 = '''@ti.kernel
def compute_acc(cx: ti.f32, cy: ti.f32, cm: ti.f32):
    """Every body feels the softened pull of every other — direct O(N^2) summation."""
    for i in range(N):
        a = ti.Vector([0.0, 0.0])
        pi = pos[i]
        for j in range(N):
            d = pos[j] - pi
            r2 = d.dot(d) + EPS * EPS
            a += G * mass[j] * d / (r2 * ti.sqrt(r2))
        acc[i] = a'''

ACC_V2 = '''@ti.kernel
def compute_acc(cx: ti.f32, cy: ti.f32, cm: ti.f32):
    """Every body feels the softened pull of every other (direct O(N^2) summation), plus an
    optional cursor mass the player drags through the scene."""
    cursor = ti.Vector([cx, cy])
    for i in range(N):
        a = ti.Vector([0.0, 0.0])
        pi = pos[i]
        for j in range(N):
            d = pos[j] - pi
            r2 = d.dot(d) + EPS * EPS
            a += G * mass[j] * d / (r2 * ti.sqrt(r2))
        d = cursor - pi
        r2 = d.dot(d) + EPS * EPS
        a += G * cm * d / (r2 * ti.sqrt(r2))
        acc[i] = a'''

frag(((1, 5), ACC_V1), ((3, 1), ACC_V2))

# --- the symplectic leapfrog ---------------------------------------------------------

frag(((1, 5), "@ti.kernel\ndef kick(h: ti.f32):\n    for i in range(N):\n        vel[i] += h * acc[i]"))
frag(((1, 5), "@ti.kernel\ndef drift(h: ti.f32):\n    for i in range(N):\n        pos[i] += h * vel[i]"))

STEP = '''def step(cx=-1e9, cy=-1e9, cm=0.0):
    """One leapfrog (kick-drift-kick) tick — symplectic, so orbits stay stable for ages."""
    kick(0.5 * DT)
    drift(DT)
    compute_acc(cx, cy, cm)
    kick(0.5 * DT)'''

frag(((1, 5), STEP))

# --- diagnostics ---------------------------------------------------------------------

TOTAL_ENERGY = '''def total_energy():
    """Pure numpy: kinetic + gravitational potential energy of the whole system."""
    p = pos.to_numpy()
    v = vel.to_numpy()
    m = mass.to_numpy()
    ke = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
    d = p[:, None, :] - p[None, :, :]
    r = np.sqrt(np.sum(d * d, axis=2) + EPS * EPS)
    iu = np.triu_indices(len(p), k=1)
    pe = -G * float(np.sum(m[iu[0]] * m[iu[1]] / r[iu]))
    return ke + pe'''

frag(((1, 6), TOTAL_ENERGY))

BOUND_FRACTION = '''def bound_fraction(cx=0.0, cy=0.0, extent=VIEW):
    """Pure numpy: fraction of bodies still within `extent` of a center — how much survived."""
    p = pos.to_numpy()
    return float((np.abs(p - np.array([cx, cy])).max(axis=1) < extent).mean())'''

frag(((1, 6), BOUND_FRACTION))

# --- render --------------------------------------------------------------------------

RENDER = '''@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.02, 0.02, 0.04])
    for i in range(N):
        x = ti.cast((pos[i][0] / (2.0 * VIEW) + 0.5) * RES, ti.i32)
        y = ti.cast((pos[i][1] / (2.0 * VIEW) + 0.5) * RES, ti.i32)
        if 0 <= x < RES and 0 <= y < RES:
            f = ti.min(vel[i].norm() / 1.5, 1.0)
            col = ti.Vector([0.5, 0.7, 1.0]) * f + ti.Vector([1.0, 0.55, 0.2]) * (1.0 - f)
            pixels[x, y] += col * 0.6                      # additive glow: dense cores blaze
            if mass[i] > 0.1:                              # a black hole: bright and fat
                for dx, dy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < RES and 0 <= yy < RES:
                        pixels[xx, yy] = ti.Vector([1.0, 0.95, 0.8])
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)           # dense cores saturate to white'''

frag(((1, 6), RENDER))

# --- main (grows across chapters) ----------------------------------------------------

MAIN_OPEN_V1 = '''def main():
    init_sim()
    apply_seed("single")'''

MAIN_OPEN_V2 = '''def main():
    init_sim()
    apply_seed("collide")'''

frag(((1, 6), MAIN_OPEN_V1), ((2, 2), MAIN_OPEN_V2))

frag((
    (1, 6),
    '    gui = ti.GUI("Universe Sandbox — taichi-academy", res=RES, background_color=0x05050A)\n'
    "    while gui.running:\n"
    "        cx, cy, cm = -1e9, -1e9, 0.0",
))

EVENTS_V1 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "s":
                apply_seed("single", np.random.randint(1_000_000))'''

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "c":
                apply_seed("collide", np.random.randint(1_000_000))
            elif e.key == "s":
                apply_seed("single", np.random.randint(1_000_000))'''

frag(((1, 6), EVENTS_V1), ((2, 2), EVENTS_V2))

CURSOR = '''        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            cx, cy = (mx - 0.5) * 2.0 * VIEW, (my - 0.5) * 2.0 * VIEW
            cm = 3.0 * BH_MASS                             # a heavy black hole under the cursor'''

frag(((3, 2), CURSOR))

MAIN_TAIL = '''        step(cx, cy, cm)
        render()
        gui.set_image(pixels)
        gui.text("drag: black-hole cursor   [c] collide   [s] single galaxy", (0.02, 0.98), color=0xFFFFFF)
        gui.show()'''

frag(((1, 6), MAIN_TAIL))

frag(((1, 6), 'if __name__ == "__main__":\n    main()'))

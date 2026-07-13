"""Code SOT for project 08 — MPM snow & sand.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 08-mpm-snow-sand`.

Evolutions: p2g goes through three versions — dust (mass+momentum only, no
stress, chapter 2), snow-elastic (chapter 3, sand particles still behave as
dust since sand_stress doesn't exist yet), and the final version wiring both
materials (chapter 4). grid_forces and substep each gain their stirring
parameters only in chapter 5. g2p and clear_grid, once written, never change —
same "the physics stays, only the material law changes" shape as project 06.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="08-mpm-snow-sand",
    default_file="mpm_snow_sand.py",
    reference={"mpm_snow_sand.py": PROJECT_DIR / "reference" / "mpm_snow_sand.py"},
    chapter_steps={1: 3, 2: 3, 3: 3, 4: 3, 5: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""MPM Snow & Sand: one elastoplastic engine, particles AND a grid working together."""'))
frag(((1, 1), "import math"))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((2, 1), "N_GRID = 128"))
frag(((2, 1), "DX = 1.0 / N_GRID"))
frag(((2, 1), "INV_DX = float(N_GRID)"))
frag(((2, 1), "DT = 1e-4"))
frag(((2, 1), "SUBSTEPS = 25"))
frag(((3, 3), "P_VOL = (DX * 0.5) ** 2"))
frag(((2, 1), "P_MASS = P_VOL * 1.0"))
frag(((2, 1), "GRAVITY = 9.8"))

frag(((1, 2), "SNOW, SAND = 0, 1"))
frag(((1, 2), "N_PER_BLOCK = 6000"))
frag(((1, 2), "N_PARTICLES = N_PER_BLOCK * 2"))

frag(((3, 1), "E_SNOW, NU_SNOW = 1.4e4, 0.2"))
frag(((3, 1), "THETA_C_SNOW, THETA_S_SNOW = 2.5e-2, 4.5e-3"))
frag(((3, 1), "HARDEN_SNOW = 10.0"))

frag(((4, 1), "E_SAND, NU_SAND = 3.5e3, 0.3"))
frag(((4, 1), "FRICTION_DEG = 35.0"))
SAND_ALPHA = (
    "SAND_ALPHA = math.sqrt(2.0 / 3.0) * (2.0 * math.sin(math.radians(FRICTION_DEG))) / (\n"
    "    3.0 - math.sin(math.radians(FRICTION_DEG))\n"
    ")"
)
frag(((4, 1), SAND_ALPHA))

frag(((5, 1), "STIR_RADIUS = 0.002"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "x = None"))
frag(((1, 2), "v = None"))
frag(((1, 2), "C = None"))
frag(((1, 2), "F = None"))
frag(((1, 2), "Jp = None"))
frag(((1, 2), "material = None"))
frag(((2, 1), "grid_v = None"))
frag(((2, 1), "grid_m = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'
frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global x, v, C, F, Jp, material"),
    ((2, 1), f"def init_sim(arch=None):\n{DOC}\n    global x, v, C, F, Jp, material, grid_v, grid_m"),
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
frag(((1, 2), "    x = ti.Vector.field(2, ti.f32, shape=N_PARTICLES)"))
frag(((1, 2), "    v = ti.Vector.field(2, ti.f32, shape=N_PARTICLES)"))
frag(((1, 2), "    C = ti.Matrix.field(2, 2, ti.f32, shape=N_PARTICLES)"))
frag(((1, 2), "    F = ti.Matrix.field(2, 2, ti.f32, shape=N_PARTICLES)"))
frag(((1, 2), "    Jp = ti.field(ti.f32, shape=N_PARTICLES)"))
frag(((1, 2), "    material = ti.field(ti.i32, shape=N_PARTICLES)"))
frag(((2, 1), "    grid_v = ti.Vector.field(2, ti.f32, shape=(N_GRID, N_GRID))"))
frag(((2, 1), "    grid_m = ti.field(ti.f32, shape=(N_GRID, N_GRID))"))

# --- seeding -------------------------------------------------------------------------

SEED_BLOCK = '''def seed_block(n, cx, cy, hx, hy, rng_seed):
    """Pure numpy: n random points inside a cx,cy-centered rectangle."""
    rng = np.random.default_rng(rng_seed)
    off = (rng.random((n, 2)).astype(np.float32) * 2 - 1) * np.array([hx, hy], dtype=np.float32)
    return off + np.array([cx, cy], dtype=np.float32)'''

frag(((1, 3), SEED_BLOCK))

RESET = """@ti.kernel
def reset_particles():
    for p in x:
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        Jp[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)"""

frag(((1, 3), RESET))

APPLY_SEED = """def apply_seed(pos_snow, pos_sand):
    pos = np.concatenate([pos_snow, pos_sand], axis=0)
    mat = np.concatenate(
        [np.full(len(pos_snow), SNOW, dtype=np.int32), np.full(len(pos_sand), SAND, dtype=np.int32)]
    )
    x.from_numpy(pos)
    material.from_numpy(mat)
    reset_particles()"""

frag(((1, 3), APPLY_SEED))

# --- grid bookkeeping ------------------------------------------------------------------

CLEAR_GRID = """@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0"""

frag(((2, 2), CLEAR_GRID))

# --- material laws -------------------------------------------------------------------

SNOW_STRESS = """@ti.func
def snow_stress(p, U, sig, V):
    mu_0 = E_SNOW / (2 * (1 + NU_SNOW))
    lambda_0 = E_SNOW * NU_SNOW / ((1 + NU_SNOW) * (1 - 2 * NU_SNOW))
    h = ti.exp(HARDEN_SNOW * (1.0 - Jp[p]))
    mu, la = mu_0 * h, lambda_0 * h
    J = 1.0
    sig_c = sig
    for d in ti.static(range(2)):
        new_sig = ti.min(ti.max(sig_c[d, d], 1 - THETA_C_SNOW), 1 + THETA_S_SNOW)
        Jp[p] *= sig_c[d, d] / new_sig
        sig_c[d, d] = new_sig
        J *= new_sig
    F[p] = U @ sig_c @ V.transpose()
    return mu, la, J"""

frag(((3, 2), SNOW_STRESS))

SAND_STRESS = """@ti.func
def sand_stress(p, U, sig, V):
    mu_0 = E_SAND / (2 * (1 + NU_SAND))
    lambda_0 = E_SAND * NU_SAND / ((1 + NU_SAND) * (1 - 2 * NU_SAND))
    e0 = ti.log(ti.max(sig[0, 0], 1e-4))
    e1 = ti.log(ti.max(sig[1, 1], 1e-4))
    eps = ti.Vector([e0, e1])
    eps_trace = eps[0] + eps[1]
    eps_hat = eps - (eps_trace / 2.0) * ti.Vector([1.0, 1.0])
    eps_hat_norm = eps_hat.norm() + 1e-20
    new_eps = eps
    if eps_trace > 0.0:
        new_eps = ti.Vector([0.0, 0.0])
        Jp[p] *= ti.exp(eps_trace)
    else:
        delta_gamma = eps_hat_norm + (2.0 * lambda_0 + 2.0 * mu_0) / (2.0 * mu_0) * eps_trace * SAND_ALPHA
        if delta_gamma > 0.0:
            new_eps = eps - (delta_gamma / eps_hat_norm) * eps_hat
    new_sig0, new_sig1 = ti.exp(new_eps[0]), ti.exp(new_eps[1])
    F[p] = U @ ti.Matrix([[new_sig0, 0.0], [0.0, new_sig1]]) @ V.transpose()
    return mu_0, lambda_0, new_sig0 * new_sig1"""

frag(((4, 2), SAND_STRESS))

# --- particle -> grid ------------------------------------------------------------------

P2G_HEAD = """@ti.kernel
def p2g():
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]"""

P2G_DUST = "        affine = P_MASS * C[p]"

P2G_SNOW_ONLY = """        F[p] = (ti.Matrix.identity(ti.f32, 2) + DT * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        mu, la, J = 0.0, 0.0, 1.0
        if material[p] == SNOW:
            mu, la, J = snow_stress(p, U, sig, V)

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
        stress += ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-DT * P_VOL * 4 * INV_DX * INV_DX) * stress
        affine = stress + P_MASS * C[p]"""

P2G_BOTH = """        F[p] = (ti.Matrix.identity(ti.f32, 2) + DT * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        mu, la, J = 0.0, 0.0, 1.0
        if material[p] == SNOW:
            mu, la, J = snow_stress(p, U, sig, V)
        else:
            mu, la, J = sand_stress(p, U, sig, V)

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
        stress += ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-DT * P_VOL * 4 * INV_DX * INV_DX) * stress
        affine = stress + P_MASS * C[p]"""

frag(((2, 2), P2G_HEAD))
frag(((2, 2), P2G_DUST), ((3, 3), P2G_SNOW_ONLY), ((4, 3), P2G_BOTH))

P2G_TAIL = """        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * DX
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (P_MASS * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * P_MASS"""

frag(((2, 2), P2G_TAIL))

# --- grid forces -------------------------------------------------------------------

GRID_FORCES_V1 = """@ti.kernel
def grid_forces():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1.0 / grid_m[i, j]) * grid_v[i, j]
        grid_v[i, j][1] -= DT * GRAVITY
        if i < 3 and grid_v[i, j][0] < 0:
            grid_v[i, j][0] = 0
        if i > N_GRID - 3 and grid_v[i, j][0] > 0:
            grid_v[i, j][0] = 0
        if j < 3 and grid_v[i, j][1] < 0:
            grid_v[i, j][1] = 0
        if j > N_GRID - 3 and grid_v[i, j][1] > 0:
            grid_v[i, j][1] = 0"""

GRID_FORCES_V2 = """@ti.kernel
def grid_forces(mx: ti.f32, my: ti.f32, fx_: ti.f32, fy_: ti.f32, stirring: ti.i32):
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1.0 / grid_m[i, j]) * grid_v[i, j]
        grid_v[i, j][1] -= DT * GRAVITY
        if stirring == 1:
            gx, gy = i * DX, j * DX
            d2 = (gx - mx) ** 2 + (gy - my) ** 2
            grid_v[i, j] += ti.exp(-d2 / STIR_RADIUS) * ti.Vector([fx_, fy_])
        if i < 3 and grid_v[i, j][0] < 0:
            grid_v[i, j][0] = 0
        if i > N_GRID - 3 and grid_v[i, j][0] > 0:
            grid_v[i, j][0] = 0
        if j < 3 and grid_v[i, j][1] < 0:
            grid_v[i, j][1] = 0
        if j > N_GRID - 3 and grid_v[i, j][1] > 0:
            grid_v[i, j][1] = 0"""

frag(((2, 2), GRID_FORCES_V1), ((5, 1), GRID_FORCES_V2))

# --- grid -> particle ------------------------------------------------------------------

G2P = """@ti.kernel
def g2p():
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * INV_DX * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += DT * v[p]"""

frag(((2, 2), G2P))

frag(
    ((2, 2), "def substep():\n    clear_grid()\n    p2g()\n    grid_forces()\n    g2p()"),
    (
        (5, 1),
        "def substep(mx=0.0, my=0.0, fx=0.0, fy=0.0, stirring=False):\n"
        "    clear_grid()\n"
        "    p2g()\n"
        "    grid_forces(mx, my, fx, fy, 1 if stirring else 0)\n"
        "    g2p()",
    ),
)

DEFAULT_SEED = """def default_seed(rng_seed=0):
    pos_snow = seed_block(N_PER_BLOCK, 0.28, 0.12, 0.12, 0.05, rng_seed)
    pos_sand = seed_block(N_PER_BLOCK, 0.72, 0.12, 0.12, 0.05, rng_seed + 1)
    return pos_snow, pos_sand"""

frag(((1, 3), DEFAULT_SEED))

# --- main (small sub-fragments) -------------------------------------------------------

frag(((1, 3), "def main():\n    init_sim()\n    apply_seed(*default_seed())"))
frag(((1, 3), '    gui = ti.GUI("MPM Snow & Sand — taichi-academy", res=512, background_color=0x0A0A12)'))
frag(
    (
        (1, 3),
        "    colors = np.zeros(N_PARTICLES, dtype=np.uint32)\n"
        "    colors[:N_PER_BLOCK] = 0xE8F3FF\n"
        "    colors[N_PER_BLOCK:] = 0xD8A650",
    )
)
frag(((5, 2), "    pmx, pmy = 0.0, 0.0\n    dragging = False"))
frag(((1, 3), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(*default_seed(rng_seed=np.random.randint(1_000_000)))'''

frag(((1, 3), EVENTS_V1), ((5, 2), EVENTS_V2))

STEP_CALL_V1 = "        for _ in range(SUBSTEPS):\n            substep()"

STIR_BLOCK = '''        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            fx, fy = 0.0, 0.0
            if dragging:
                fx, fy = (mx - pmx) * 40.0 / SUBSTEPS, (my - pmy) * 40.0 / SUBSTEPS
            for _ in range(SUBSTEPS):
                substep(mx, my, fx, fy, stirring=True)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False
            for _ in range(SUBSTEPS):
                substep()'''

frag(((2, 3), STEP_CALL_V1), ((5, 3), STIR_BLOCK))

frag(((1, 3), "        gui.circles(x.to_numpy(), radius=1.5, color=colors)"))
frag(((5, 3), '        gui.text("drag to stir  [r] reset", (0.02, 0.98), color=0xAAAAAA)'))
frag(((1, 3), "        gui.show()"))
frag(((1, 3), 'if __name__ == "__main__":\n    main()'))

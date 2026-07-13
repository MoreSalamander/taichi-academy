// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["08-mpm-snow-sand"] = {
  project: "08-mpm-snow-sand",
  title: "MPM Snow & Sand",
  pitch: "One elastoplastic engine, two materials — snow that compacts and sand that finds its own angle of repose.",
  tier: "hard",
  language: "Python",
  file: "mpm_snow_sand.py",
  chapters: [
    {
      id: 1, title: "Two piles of dust",
      build: "particle state for two materials and a first static render — no physics yet.",
      beat: "Two motionless piles of colored dust.",
      steps: [
        { title: "A new kind of project", adding: "the docstring and imports.",
          code: `"""MPM Snow & Sand: one elastoplastic engine, particles AND a grid working together."""
import math
import numpy as np
import taichi as ti`,
          does: "MPM stands for Material Point Method: particles carry the material's identity and history (this snow, this sand), but a background GRID does the actual force computation. Every step, information flows particle-to-grid, gets resolved there, then flows grid-to-particle again. It's the most powerful technique in the curriculum so far — a genuine hybrid of everything project 01-07 taught separately.",
          why: "Grids alone (Arc 1) can't represent a solid object that keeps its shape while moving. Particles alone (Arc 2 so far) can't easily compute internal forces between neighbors — project 06 needed a whole spatial hash just to find nearby particles. MPM sidesteps that: the grid IS the neighbor-finder, particles just visit it.",
          see: "Runs clean.",
          checkpoint: "python3 mpm_snow_sand.py returns silently.",
          recovery: ["Same ritual as every project — venv first."] },
        { title: "State for two materials", adding: "material IDs, particle counts, six particle fields, and init_sim.",
          code: `SNOW, SAND = 0, 1
N_PER_BLOCK = 6000
N_PARTICLES = N_PER_BLOCK * 2
x = None
v = None
C = None
F = None
Jp = None
material = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global x, v, C, F, Jp, material
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    x = ti.Vector.field(2, ti.f32, shape=N_PARTICLES)
    v = ti.Vector.field(2, ti.f32, shape=N_PARTICLES)
    C = ti.Matrix.field(2, 2, ti.f32, shape=N_PARTICLES)
    F = ti.Matrix.field(2, 2, ti.f32, shape=N_PARTICLES)
    Jp = ti.field(ti.f32, shape=N_PARTICLES)
    material = ti.field(ti.i32, shape=N_PARTICLES)`,
          does: "x and v are the familiar position/velocity pair. C is new: an affine velocity matrix (2x2) — not just HOW FAST a particle moves but how the velocity FIELD varies right around it, which is what lets particles carry rotation and shear, not just translation. F is the deformation gradient — a 2x2 matrix tracking how much the material around this particle has stretched, squashed, or sheared since it was born. Jp tracks accumulated plastic (permanent) volume change. material picks snow or sand.",
          why: "F is the single idea that makes MPM able to simulate SOLIDS: a particle with F == identity is undeformed; F far from identity means it's been stretched or squashed, and that deformation is exactly what internal elastic stress reads from. You've never needed anything like it for grids or free particles — it only exists because these particles are meant to hold a SHAPE.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Six fields, six global names, six allocation lines — same accounting discipline every project has used since project 01."] },
        { title: "See the piles", adding: "the pure-numpy block seeder, particle reset, seed application, and a static render.",
          code: `def seed_block(n, cx, cy, hx, hy, rng_seed):
    """Pure numpy: n random points inside a cx,cy-centered rectangle."""
    rng = np.random.default_rng(rng_seed)
    off = (rng.random((n, 2)).astype(np.float32) * 2 - 1) * np.array([hx, hy], dtype=np.float32)
    return off + np.array([cx, cy], dtype=np.float32)
@ti.kernel
def reset_particles():
    for p in x:
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        Jp[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)
def apply_seed(pos_snow, pos_sand):
    pos = np.concatenate([pos_snow, pos_sand], axis=0)
    mat = np.concatenate(
        [np.full(len(pos_snow), SNOW, dtype=np.int32), np.full(len(pos_sand), SAND, dtype=np.int32)]
    )
    x.from_numpy(pos)
    material.from_numpy(mat)
    reset_particles()
def default_seed(rng_seed=0):
    pos_snow = seed_block(N_PER_BLOCK, 0.28, 0.12, 0.12, 0.05, rng_seed)
    pos_sand = seed_block(N_PER_BLOCK, 0.72, 0.12, 0.12, 0.05, rng_seed + 1)
    return pos_snow, pos_sand
def main():
    init_sim()
    apply_seed(*default_seed())
    gui = ti.GUI("MPM Snow & Sand — taichi-academy", res=512, background_color=0x0A0A12)
    colors = np.zeros(N_PARTICLES, dtype=np.uint32)
    colors[:N_PER_BLOCK] = 0xE8F3FF
    colors[N_PER_BLOCK:] = 0xD8A650
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        gui.circles(x.to_numpy(), radius=1.5, color=colors)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "seed_block scatters n points inside a rectangle — one call for a snow pile, one for a sand pile, at different screen positions. reset_particles gives every particle a clean slate: identity F (undeformed), Jp=1 (no plastic history yet), zero velocity and affine matrix. gui.circles is ti.GUI's built-in scatter-plot renderer — the standard way MPM demos draw particles, no pixel field needed.",
          why: "F starting at IDENTITY, not zero, matters: identity means 'this material currently occupies exactly the space it was born into, undeformed' — the natural rest state. Starting it at zero would mean 'infinitely collapsed,' which is nonsense. That one design choice is why every material law you write from here reads F relative to identity.",
          see: "Two rectangles of dust — pale blue snow on the left, warm gold sand on the right — sitting frozen above an invisible floor.",
          checkpoint: "Two static piles. Beat 1.",
          recovery: ["apply_seed takes the RESULTS of two seed_block calls, unpacked with * from default_seed's tuple.", "colors is a plain numpy uint32 array — gui.circles wants packed 0xRRGGBB ints, not float triples like GGUI did."] }
      ]
    },
    {
      id: 2, title: "The grid joins the particles",
      build: "the background grid, particle-to-grid transfer, gravity, walls, and grid-to-particle transfer — the full MPM loop, no material stiffness yet.",
      beat: "Formless dust falls and puddles on the floor — no material has opinions about its own shape yet.",
      steps: [
        { title: "A grid underneath everything", adding: "grid resolution/timestep dials and the velocity/mass grid fields.",
          code: `N_GRID = 128
DX = 1.0 / N_GRID
INV_DX = float(N_GRID)
DT = 1e-4
SUBSTEPS = 25
P_MASS = P_VOL * 1.0
GRAVITY = 9.8
grid_v = None
grid_m = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global x, v, C, F, Jp, material, grid_v, grid_m
    grid_v = ti.Vector.field(2, ti.f32, shape=(N_GRID, N_GRID))
    grid_m = ti.field(ti.f32, shape=(N_GRID, N_GRID))`,
          does: "grid_v and grid_m are ordinary 2D fields, same shape idea as every grid project since 01 — but here they're SCRATCH SPACE, rebuilt from scratch every physics tick, never carrying state between ticks the way h or u did. DT is tiny (1e-4) because MPM's explicit stress calculation is only stable at small steps; SUBSTEPS runs many of them per rendered frame, exactly like project 01's Gray-Scott.",
          why: "The grid being disposable, recomputed-every-tick scratch space (rather than persistent state) is the key mental shift from Arc 1: there, the grid WAS the world. Here, the PARTICLES are the world, and the grid is just where the math happens to be convenient — a borrowed calculator, wiped clean and reused every single tick.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["P_VOL isn't defined until chapter 3, but P_MASS = P_VOL * 1.0 is written now anyway — Python only evaluates that line when init happens, and P_VOL will exist by then. Order in the FILE, not just chapter number, is what matters."] },
        { title: "Deposit, resolve, gather", adding: "grid clearing, the particle-to-grid pass (dust only — no stress), gravity and walls, and grid-to-particle.",
          code: `@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0
@ti.kernel
def p2g():
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = P_MASS * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * DX
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (P_MASS * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * P_MASS
@ti.kernel
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
            grid_v[i, j][1] = 0
@ti.kernel
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
        x[p] += DT * v[p]
def substep():
    clear_grid()
    p2g()
    grid_forces()
    g2p()`,
          does: "base/fx/w locate each particle among its 9 nearest grid nodes (a quadratic B-spline — smoother than project 06's nearest-cell lookup) and weight how much it contributes to each. p2g scatters mass and momentum outward — for now affine is JUST P_MASS * C[p] (no stress at all: this is 'dust,' with no internal stiffness). grid_forces turns momentum into velocity (divide by mass), adds gravity, and reflects velocity at the domain edges — the exact wall-bounce idea from project 06's integrate(), just applied to the grid instead of particles directly. g2p is the mirror of p2g: gather grid velocity AND the local velocity gradient (into new_C) back onto each particle, then advect its position.",
          why: "Four phases, strict order — clear before deposit, deposit before resolve, resolve before gather — the same 'phases of a tick' discipline as every multi-kernel project since 05. What's different: information doesn't just move particle-to-particle (project 06) or cell-to-cell (project 01-05), it crosses BETWEEN representations twice a tick. That round trip is MPM's entire trick.",
          see: "Runs clean; nothing animates yet — nothing calls substep() from the render loop.",
          checkpoint: "No red text.",
          recovery: ["w is a quadratic spline (three weights, not two like project 06's linear falloff) — that's why the 3x3 neighborhood loop is unconditional here (project 06 bounds-checked because its grid had edges the same size as the world; here DX is deliberately tiny so particles are always safely mid-grid).", "g2p's new_C uses outer_product(dpos) — this is where a particle learns whether the space around it is spinning or shearing, not just translating."] },
        { title: "Let it fall", adding: "the physics call in the render loop.",
          code: `        for _ in range(SUBSTEPS):
            substep()`,
          does: "25 tiny ticks, every rendered frame, before the next gui.circles.",
          why: "Free particles, no stiffness: exactly what 'dust' means physically. Watch what happens WITHOUT F, Jp, or a material law — this is the honest baseline every material law from here improves on.",
          see: "Both piles fall, hit the floor, and spread — thin, formless, indistinguishable puddles. Snow and sand look and behave identically: neither has been told to hold a shape yet.",
          checkpoint: "Two puddles, same shape. Beat 2.",
          recovery: ["The loop goes right after the event loop, before gui.circles — physics, then paint."] }
      ]
    },
    {
      id: 3, title: "Snow remembers its shape",
      build: "the deformation gradient's first real job: an elastoplastic stress law that lets snow compact without losing its cohesion.",
      beat: "Snow stops puddling and settles into a mound that holds together; sand (not yet given a law) still puddles.",
      steps: [
        { title: "Snow's dials", adding: "stiffness, compression/stretch limits, and a hardening rate.",
          code: `E_SNOW, NU_SNOW = 1.4e4, 0.2
THETA_C_SNOW, THETA_S_SNOW = 2.5e-2, 4.5e-3
HARDEN_SNOW = 10.0`,
          does: "E (Young's modulus) and nu (Poisson's ratio) are the two numbers that describe any elastic material's stiffness and how much it bulges when squashed — standard continuum-mechanics dials, not MPM-specific. THETA_C/THETA_S cap how much a particle's local stretch can compress (2.5%) or expand (0.45%) before yielding PERMANENTLY — snow crunches more easily than it stretches. HARDEN controls a feedback loop: compacted snow gets STIFFER.",
          why: "These exact values (1.4e4, 0.2, 2.5e-2, 4.5e-3, 10) are the ones from the material-point-method snow research this technique comes from — real numbers, not arbitrary. You'll retune completely different ones for sand in chapter 4, and the CONTRAST between the two tunings is the whole lesson.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Nothing visible changes yet — these are just numbers waiting for a law to use them."] },
        { title: "The snow law", adding: "the elastoplastic stress function.",
          code: `@ti.func
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
    return mu, la, J`,
          does: "mu_0/lambda_0 (the Lame parameters) are E and nu converted into the two numbers the actual stress formula wants. U, sig, V come from an SVD of F — sig's diagonal is exactly how much the material is stretched along its two principal axes RIGHT NOW. Each axis gets clamped into the elastic band; whatever falls outside becomes a PERMANENT correction (absorbed into Jp, never recovered), and Jp<1 (compacted) feeds back through h to stiffen the material.",
          why: "SVD turns a messy 2x2 deformation into two independent numbers you can reason about and clamp directly — this is THE technique that makes elastoplastic materials tractable at all, and it only works because sig's diagonal entries are always non-negative stretch factors, never signed or rotational (that's what U and V absorbed). You met SVD nowhere else in this curriculum; MPM is where it earns its keep.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["sig_c = sig makes a LOCAL copy before mutating it — sig itself came from ti.svd and shouldn't be edited in place.", "Jp *= sig_c[d,d] / new_sig BEFORE overwriting sig_c[d,d] — you need the ORIGINAL stretch to compute how much was clamped away."] },
        { title: "Snow only, for now", adding: "particle volume and the real stress computation in p2g (sand still falls as dust).",
          code: `P_VOL = (DX * 0.5) ** 2
        F[p] = (ti.Matrix.identity(ti.f32, 2) + DT * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        mu, la, J = 0.0, 0.0, 1.0
        if material[p] == SNOW:
            mu, la, J = snow_stress(p, U, sig, V)

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
        stress += ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-DT * P_VOL * 4 * INV_DX * INV_DX) * stress
        affine = stress + P_MASS * C[p]`,
          does: "F[p] first absorbs this tick's velocity gradient (I + dt*C) — deformation accumulates continuously, tick after tick. Every particle gets an SVD, but only SNOW calls snow_stress; sand's mu/la/J stay at their neutral defaults (0, 0, 1), so its stress works out to exactly zero — sand is UNCHANGED, still pure dust. The stress formula itself (2*mu*(F - UV^T)@F^T + ... ) is a corotational elasticity law: it measures deformation relative to the CLOSEST pure rotation, so spinning a particle costs nothing, only actual stretching does.",
          why: "This step's whole point is the contrast: one branch, one material given a law, the other left exactly as chapter 2 built it. Watching snow suddenly hold its shape while sand keeps puddling right next to it is the clearest possible proof that the material LAW — not the particle, not the grid — is what makes something behave like snow.",
          see: "Snow stops spreading flat — it settles into a rounded, cohesive mound, maybe with a visible crack line as it compacts under its own weight (real elastoplastic materials do that too). Sand, still lawless, keeps puddling exactly as before.",
          checkpoint: "Snow holds a mound; sand still a puddle. Beat 3.",
          recovery: ["P_VOL is declared UP among the constants in the file, but introduced here — it isn't needed until this exact stress formula.", "if material[p] == SNOW with NO else yet — that asymmetry is intentional, not a bug."] }
      ]
    },
    {
      id: 4, title: "Sand finds its angle of repose",
      build: "a Drucker-Prager granular law — sand doesn't just resist deformation, it YIELDS under shear and flows into a pile.",
      beat: "Sand spreads into a real heap with a natural slope; snow stays compact beside it.",
      steps: [
        { title: "Sand's dials", adding: "sand's own stiffness and a friction angle.",
          code: `E_SAND, NU_SAND = 3.5e3, 0.3
FRICTION_DEG = 35.0
SAND_ALPHA = math.sqrt(2.0 / 3.0) * (2.0 * math.sin(math.radians(FRICTION_DEG))) / (
    3.0 - math.sin(math.radians(FRICTION_DEG))
)`,
          does: "Sand is softer than snow (E is 4x smaller) — the real physical difference is the LAW you're about to write, not just stiffness. FRICTION_DEG (35 degrees, a realistic dry-sand value) is an angle real granular materials have — the steepest slope a pile can hold before grains slide. SAND_ALPHA converts that human-meaningful angle into the one number the Drucker-Prager yield formula actually needs.",
          why: "Friction angle is THE defining number for granular materials — it's why sandcastles have a maximum slope and sugar piles look different from flour piles. No such angle exists for snow's elastoplastic model; that absence is itself the story of why the two materials need genuinely different physics, not just different constants plugged into the same formula.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["math.radians converts degrees to radians before any trig — FRICTION_DEG is written in the units a human thinks in; math needs the other kind."] },
        { title: "The sand law", adding: "Drucker-Prager plasticity, in log-strain space.",
          code: `@ti.func
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
    return mu_0, lambda_0, new_sig0 * new_sig1`,
          does: "eps = log(sig) converts stretch into LOG-strain, where volume change (eps_trace, compression/expansion) and shape change (eps_hat, shear) separate cleanly by addition instead of multiplication. If a grain is being pulled apart (eps_trace > 0) sand has NO tensile strength — it fully yields, resetting to undeformed. Otherwise, delta_gamma tests whether the SHEAR magnitude, scaled by the friction cone (SAND_ALPHA), exceeds what the current compression can support — if so, the strain gets projected back onto the yield surface, permanently.",
          why: "Snow's law (chapter 3) clamped each axis independently — a purely VOLUMETRIC test. Sand's law compares shear against compression directly — that's the actual physics of a friction cone, and it's the only way to get a material that resists being squashed but readily slides sideways once it's steep enough. Same SVD ingredients as snow, an entirely different yield SURFACE.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["ti.max(sig[d,d], 1e-4) before ti.log — log(0) is undefined, and a fresh particle's sig can be exactly 1.0, safely away from zero, but this guard matters once things start compacting hard.", "eps_hat_norm has a tiny +1e-20 — dividing by a norm that could be exactly zero for an undeformed particle would be a silent NaN factory."] },
        { title: "Both materials, real physics", adding: "the sand branch in p2g.",
          code: `        F[p] = (ti.Matrix.identity(ti.f32, 2) + DT * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        mu, la, J = 0.0, 0.0, 1.0
        if material[p] == SNOW:
            mu, la, J = snow_stress(p, U, sig, V)
        else:
            mu, la, J = sand_stress(p, U, sig, V)

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
        stress += ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-DT * P_VOL * 4 * INV_DX * INV_DX) * stress
        affine = stress + P_MASS * C[p]`,
          does: "One new elif, filling in the else this project has carried since chapter 3. Everything downstream — the stress formula, the grid transfer — is UNCHANGED and shared by both materials; only the branch that turns 'how deformed is this particle' into 'how stiff should it resist' differs.",
          why: "This is the payoff of writing the stress math generically back in chapter 3: adding an entirely different constitutive model (isotropic clamp vs. friction cone) cost exactly one function and one elif. That's what a well-factored material law buys you — and it's the same lesson project 06 taught when swapping brute-force neighbors for a spatial hash without touching force_law at all.",
          see: "Let both piles run for a while: sand spreads outward from its own weight into a real heap with a natural, consistent slope — an angle of repose, exactly like the FRICTION_DEG dial promised — while snow beside it stays a compact, cohesive mound. Two truly different materials from one engine.",
          checkpoint: "Sand piles at its angle of repose; snow stays compact. Beat 4.",
          recovery: ["The order matters: if SNOW / else — sand is now everything that ISN'T snow, cleanly.", "If sand explodes instead of piling, the likely culprit is SAND_ALPHA or E_SAND drifting too far from the given values — Drucker-Prager sand is notoriously sensitive to tuning."] }
      ]
    },
    {
      id: 5, title: "Stir the sandbox",
      build: "mouse-driven stirring, a reset key, and a HUD — MPM becomes a toy you can play with.",
      beat: "Drag through snow and sand and feel them respond differently; reroll both piles anytime.",
      steps: [
        { title: "A hand in the sandbox", adding: "a stir radius dial and stirring support in grid_forces and substep.",
          code: `STIR_RADIUS = 0.002
@ti.kernel
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
            grid_v[i, j][1] = 0
def substep(mx=0.0, my=0.0, fx=0.0, fy=0.0, stirring=False):
    clear_grid()
    p2g()
    grid_forces(mx, my, fx, fy, 1 if stirring else 0)
    g2p()`,
          does: "A gaussian bump (ti.exp(-d2/STIR_RADIUS)) around the cursor adds a push directly to nearby grid velocities — the same 'torch' idea projects 02 and 03 used to stir fluid and ignite fire, applied to an MPM grid instead. substep grows default arguments so every existing call site (chapter 2's plain substep()) keeps working unchanged.",
          why: "Pushing the GRID, not particles directly, means the stir force automatically respects everything the grid already enforces — walls, mass-weighting, all of it — instead of needing its own special-case collision logic.",
          see: "Runs clean; nothing calls this with stirring=True yet.",
          checkpoint: "No red text.",
          recovery: ["stirring: ti.i32, not a Python bool — kernels can't take bool arguments directly, so main() will pass 1 or 0."] },
        { title: "The reset key", adding: "drag-tracking state and the 'r' handler.",
          code: `    pmx, pmy = 0.0, 0.0
    dragging = False
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(*default_seed(rng_seed=np.random.randint(1_000_000)))`,
          does: "pmx/pmy remember last frame's cursor position, so a drag's DIRECTION can be computed as this frame's position minus last frame's. R reseeds both piles at their original spots with a fresh random scatter.",
          why: "The familiar reseed idiom, one more time — but notice apply_seed() itself does the real reset work (including calling reset_particles(), which zeroes F back to identity and Jp back to 1). Forgetting that call would leave OLD deformation history baked into freshly-repositioned particles — the same 'reset must clear ALL state that belongs to the old world' lesson project 05 taught with its water and sediment fields.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["dragging starts False — the very first frame of a click shouldn't compute a direction from garbage previous-frame data."] },
        { title: "Drag to stir", adding: "the mouse-drag stirring branch and the HUD.",
          code: `        if gui.is_pressed(ti.GUI.LMB):
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
                substep()
        gui.text("drag to stir  [r] reset", (0.02, 0.98), color=0xAAAAAA)`,
          does: "fx, fy is the cursor's velocity THIS FRAME, converted into a stir force — and notice the / SUBSTEPS. Reference implementations sometimes skip that division and reapply the SAME full-strength force at every one of the 25 substeps in a frame — a real bug this project's own reference code hit during development: a 25x over-injection of velocity that blew the whole simulation to NaN within a couple of frames. Dividing first means the TOTAL impulse across one frame's substeps matches what the mouse actually did, no matter how many substeps compose it.",
          why: "This is a genuinely easy trap: SUBSTEPS exists so the physics stays numerically stable at a tiny DT, but any force computed 'per frame' (mouse deltas, torches, anything user-driven) has to be explicitly SPREAD across those substeps, or it gets replayed in full every single one. The project's test suite (test_stirring_stays_stable) exists specifically because this bug was real, not hypothetical.",
          see: "Drag through the sand and watch it scatter and re-settle, fluid and responsive. Drag through the snow and feel it push back stiffer, holding its shape more before it finally yields. Same hand, two different materials answering differently — the whole point of the last two chapters, now something you can FEEL.",
          checkpoint: "Interactive stirring on both materials. Final beat — project 08 complete.",
          recovery: ["If dragging ever reintroduces instability, the first thing to check is always that /SUBSTEPS division — it's the single highest-leverage number in this whole file."] }
      ]
    }
  ]
};

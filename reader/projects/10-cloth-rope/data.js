// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["10-cloth-rope"] = {
  project: "10-cloth-rope",
  title: "Cloth & Rope",
  pitch: "Not springs this time — positions that simply refuse to disobey their constraints. One solver, a rope and a flag.",
  tier: "medium",
  language: "Python",
  file: "cloth_rope.py",
  chapters: [
    {
      id: 1, title: "A chain and a grid",
      build: "particle state for a rope and a cloth grid, sharing one constraint list, and a first static render.",
      beat: "A motionless rope and a motionless flag, side by side.",
      steps: [
        { title: "A third way to be soft", adding: "the docstring and imports.",
          code: `"""Cloth & Rope: Position-Based Dynamics — Verlet integration plus distance constraints."""
import numpy as np
import taichi as ti`,
          does: "Project 09 held a shape together with SPRING FORCES (Hooke's law, tunable stiffness, can ring or explode if you push it too hard). This project holds shape a different way: Position-Based Dynamics — after every tick, directly nudge positions until distances are correct again, no force or stiffness constant involved at all.",
          why: "Force-based springs and position-based constraints solve the same problem — 'keep these two points a fixed distance apart' — with different tradeoffs. Springs are physically intuitive but can go unstable at high stiffness. Constraints are less 'physical' but nearly impossible to blow up, which is exactly why PBD is the standard technique behind most game cloth and rope you've ever seen.",
          see: "Runs clean.",
          checkpoint: "python3 cloth_rope.py returns silently.",
          recovery: ["Usual venv ritual."] },
        { title: "One particle pool, two structures", adding: "sizing/layout dials and the shared particle + constraint fields.",
          code: `ROPE_N = 40
CLOTH_W, CLOTH_H = 24, 16
CLOTH_N = CLOTH_W * CLOTH_H
N = ROPE_N + CLOTH_N
ROPE_BASE = 0
CLOTH_BASE = ROPE_N
MAX_CONSTRAINTS = 4000
SPACING = 0.02
CLOTH_ORIGIN = (0.4, 0.92)
ROPE_ORIGIN = (0.15, 0.9)
pos = None
prev_pos = None
inv_mass = None
c_a = None
c_b = None
c_len = None
n_constraints = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, prev_pos, inv_mass, c_a, c_b, c_len, n_constraints
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
    n_constraints = ti.field(ti.i32, shape=())`,
          does: "No velocity field this time — prev_pos is Verlet integration's whole trick: velocity is implied by how far a point moved last tick, never stored explicitly. inv_mass (1/mass) replaces a plain mass field for a reason you'll feel in chapter 2: setting it to exactly 0 is how a particle becomes UNMOVABLE, cheaply, with no special-case code anywhere else. c_a/c_b/c_len store an EXPLICIT edge list — pairs of particle indices plus a rest length — general enough to describe a rope's chain OR a cloth's grid with the exact same three arrays.",
          why: "Project 09's ring topology was implicit (particle p's spring partner was always p+1, computed on the fly). A cloth grid's connections are richer — each interior point touches four or more neighbors — so this project stores the edge list explicitly instead, once, up front. Same underlying idea (a topology description separate from physics), a more general shape to fit a more general problem.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["MAX_CONSTRAINTS is a fixed upper bound (4000) for the constraint arrays — apply_seed will pad up to it, the same over-allocate-once pattern every project has used for GPU fields."] },
        { title: "See them both", adding: "grid indexing, both topology builders, seed application, and a static render.",
          code: `def idx_cloth(i, j):
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
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Cloth & Rope — taichi-academy", res=512, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        p = pos.to_numpy()
        gui.circles(p[:ROPE_N], radius=2, color=0xE8544A)
        gui.circles(p[ROPE_N:], radius=1.5, color=0x8EC9FF)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "build_cloth adds THREE kinds of edges per interior cell: structural (left-right and up-down neighbors, holding the grid's basic shape) and shear (both diagonals of each cell, resisting the grid folding or skewing). im[ROPE_BASE] = 0.0 pins the rope's first link; the whole left COLUMN of the cloth gets pinned too — like a flag on a pole, not a curtain on a rod.",
          why: "The shear (diagonal) constraints matter more than they look: a grid with ONLY structural edges has a real weakness — you can fold it flat, particle over particle, without stretching a single structural edge much, the exact same collapse project 09 hit with a springless ring. Diagonals resist that fold directly, the way real woven fabric resists shearing.",
          see: "A rope hanging diagonally, and a rectangular grid of dots beside it — both frozen, both waiting for physics.",
          checkpoint: "Two static structures. Beat 1.",
          recovery: ["Pinning a whole COLUMN (for j in range(CLOTH_H): im[idx_cloth(0, j)] = 0.0), not a row — that's what makes it hang like a flag off a pole rather than a curtain off a rail."] }
      ]
    },
    {
      id: 2, title: "Predict, then fix",
      build: "Verlet integration, a deliberately serial constraint solver, and floor/wall bounds.",
      beat: "Rope and flag fall, drape, and settle — no wind yet, nothing collapses.",
      steps: [
        { title: "Move first, ask forgiveness later", adding: "motion dials and Verlet prediction.",
          code: `GRAVITY = 9.8
DT = 1.0 / 60
DAMPING = 0.995
@ti.kernel
def predict():
    for p in pos:
        if inv_mass[p] > 0:
            vel = (pos[p] - prev_pos[p]) * DAMPING
            prev_pos[p] = pos[p]
            pos[p] = pos[p] + vel + ti.Vector([0.0, -GRAVITY]) * DT * DT`,
          does: "vel isn't stored — it's RECONSTRUCTED every tick from how far a particle moved since last time (pos - prev_pos), damped slightly. Then prev_pos is saved (this tick's starting point, for NEXT tick's velocity estimate) before pos moves ahead under gravity.",
          why: "This is Verlet integration, and it's why there's no vel field anywhere in this project. It gives you the same 'implicit velocity' as project 09's Verlet-flavored spring code would, but slightly simpler to reason about, and — bonus — automatically consistent with position-based constraints later, since both operate purely on POSITIONS.",
          see: "Runs clean; nothing calls this yet.",
          checkpoint: "No red text.",
          recovery: ["Pinned particles (inv_mass == 0) skip this ENTIRELY — no gravity, no velocity, they simply don't move here. That's the whole value of inv_mass=0: one guard clause, zero special cases downstream."] },
        { title: "The constraint solver, and why it's serial", adding: "solver dials, the position-correction pass, and floor/wall bounds.",
          code: `ITERS = 6
WORLD = 1.0
@ti.kernel
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
                pos[b] -= wb * corr
@ti.kernel
def apply_bounds():
    for p in pos:
        if inv_mass[p] > 0:
            if pos[p][0] < 0.0:
                pos[p][0] = 0.0
            if pos[p][0] > WORLD:
                pos[p][0] = WORLD
            if pos[p][1] < 0.0:
                pos[p][1] = 0.0`,
          does: "For every edge, corr computes exactly how far apart the two ends need to move to restore rest to the current distance, weighted by each end's inv_mass (a pinned end with inv_mass=0 contributes nothing and absorbs nothing — the WHOLE correction goes to its free partner). ITERS runs this whole pass several times a tick, because fixing one constraint can un-fix a neighboring one — repetition is what makes a chain of many constraints converge toward all being satisfied at once.",
          why: "Notice for _ in range(1): wrapping the constraint loop — the same deliberate-serialization trick project 08 used for its prefix sum. Here it's not just a convenience: the reference implementation for this project ORIGINALLY used a plain parallel for c in range(n_constraints[None]) loop, and it blew up to NaN within a couple of frames on GPU — two constraints sharing a particle wrote to the same pos[] slot at the same instant, and Taichi's automatic atomics didn't save it. Solving one constraint fully before the next starts is slower per-iteration but perfectly safe, and for a constraint count this size (under 2000), still comfortably real-time.",
          see: "Runs clean; not wired into the loop yet.",
          checkpoint: "No red text.",
          recovery: ["apply_bounds only clamps the FLOOR (y < 0) and the side walls — no ceiling. A flung particle can fly arbitrarily high; that's intentional, matching every prior project's wall convention."] },
        { title: "Let go", adding: "the tick conductor, the reset key, and the physics call.",
          code: `def substep():
    predict()
    for _ in range(ITERS):
        solve_constraints()
    apply_bounds()
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()
        substep()`,
          does: "One substep per rendered frame this time (not many small ones) — PBD's constraint projection is stable regardless of step size, so it doesn't need project 08's tiny-DT-plus-many-substeps treatment.",
          why: "That stability claim is worth testing yourself: try DAMPING = 1.0 (no energy loss at all) and watch the rope swing essentially forever without exploding — a force-based spring at equivalent stiffness would have long since diverged. That robustness is PBD's whole sales pitch, made visible.",
          see: "The rope swings down and hangs; the flag drapes down under gravity like a heavy curtain — both settle, neither collapses flat, neither explodes.",
          checkpoint: "Rope and flag fall and settle. Beat 2.",
          recovery: ["substep() takes no arguments yet — chapter 3 changes that when wind needs a time value to oscillate against."] }
      ]
    },
    {
      id: 3, title: "Catch the wind",
      build: "a time-varying wind force baked into prediction.",
      beat: "The flag stops hanging like a curtain and starts flying like a flag.",
      steps: [
        { title: "A breeze with a phase", adding: "a wind strength dial and the wind term in predict.",
          code: `WIND = 6.0
@ti.kernel
def predict(t: ti.f32, wind: ti.f32):
    for p in pos:
        if inv_mass[p] > 0:
            vel = (pos[p] - prev_pos[p]) * DAMPING
            prev_pos[p] = pos[p]
            g = ti.Vector([wind * ti.sin(t * 3.0 + p * 0.15), -GRAVITY])
            pos[p] = pos[p] + vel + g * DT * DT`,
          does: "A sideways sine force, but with a PHASE OFFSET (p * 0.15) that differs per particle — at any given instant, particles at different indices are at different points in the sine wave, not all pushed the same direction at once.",
          why: "That per-particle phase offset is the entire trick behind believable cloth flutter — without it, 'wind' would just be the whole flag sliding sideways in lockstep, rigid and unconvincing. With it, a ripple visibly travels across the fabric, because neighboring particles (close in index, close in phase) are almost, but not quite, in sync.",
          see: "Runs clean; substep still calls the old predict() signature, so this isn't wired in yet.",
          checkpoint: "No red text.",
          recovery: ["predict now needs (t, wind) — every call site needs updating, which is exactly what the next step does."] },
        { title: "Wire in the clock", adding: "substep's new signature, a frame counter, and the updated call site.",
          code: `def substep(t=0.0, wind=WIND):
    predict(t, wind)
    for _ in range(ITERS):
        solve_constraints()
    apply_bounds()
    frame = 0
        substep(frame * DT)
        frame += 1`,
          does: "frame counts rendered frames since launch; t = frame * DT turns that into seconds, which is what feeds the wind's sine wave. substep's new defaults (t=0.0, wind=WIND) mean old-style calls like substep() (if you had any left) would still work, just with no motion in the wind term.",
          why: "wind=WIND as a DEFAULT argument, not a hardcoded constant inside the function, is what lets a future 'no wind' mode or a variable gust strength plug in later without touching predict or substep again — the same 'push configuration up to the caller' instinct that's shown up in every project since 02's torch(fx, fy).",
          see: "The flag comes alive: a wave visibly ripples along the fabric, corners flutter, the whole thing leans and billows instead of hanging dead still. The rope, unaffected by wind but still swinging under its own momentum, sways beside it.",
          checkpoint: "The flag flutters. Beat 3.",
          recovery: ["frame = 0 belongs in main(), right after the gui = ti.GUI(...) line — it needs to exist before the while loop starts using it.", "frame += 1 comes AFTER the substep call, not before — frame 0 should use t=0."] }
      ]
    },
    {
      id: 4, title: "Grab and let go",
      build: "click-and-drag grabbing that temporarily overrides Verlet prediction, plus a HUD.",
      beat: "Reach in, grab either structure, fling it around, and let go.",
      steps: [
        { title: "A third way to be immovable", adding: "grab state, and grab-awareness in predict/solve/bounds.",
          code: `GRAB_RADIUS = 0.05
grabbed = None
grab_target = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, prev_pos, inv_mass, c_a, c_b, c_len, n_constraints
    global grabbed, grab_target
    grabbed = ti.field(ti.i32, shape=())
    grab_target = ti.Vector.field(2, ti.f32, shape=())
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
    grabbed[None] = -1`,
          does: "A grabbed particle takes a THIRD path through predict — not gravity-and-wind, not the permanent-pin's do-nothing — it gets teleported straight to grab_target every tick, no physics involved. solve_constraints and apply_bounds each grow one small check so a grabbed particle behaves like a temporary pin (inv_mass effectively zero) everywhere else in the pipeline, without ever touching its real inv_mass value. grab_at excludes already-pinned particles (d2[im==0.0]=1e9) — grabbing the rope's fixed root or the flag's pole edge wouldn't do anything anyway.",
          why: "This is a THIRD way to be immovable, distinct from the other two: permanently pinned (inv_mass=0, forever) and grabbed (still inv_mass=1, but overridden every tick from outside). Keeping them separate — rather than just setting inv_mass=0 while grabbing — means release is trivial: grabbed[None]=-1, and the particle instantly resumes normal physics with whatever velocity its recent motion implied, no cleanup needed.",
          see: "Runs clean; no mouse events call grab_at yet.",
          checkpoint: "No red text.",
          recovery: ["Every kernel that checks grabbed[None] does so by comparing an INDEX, not a boolean — grabbed[None]=-1 means nothing, any p>=0 means that specific particle."] },
        { title: "Click to grab, let go to release", adding: "the LMB press and release handlers.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()
            elif e.key == ti.GUI.LMB:
                grab_at(*gui.get_cursor_pos())
        for e in gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                release()`,
          does: "Same press/release pairing project 09 used for its own grab feature — a PRESS event fires the search, a RELEASE event clears it.",
          why: "Two structures sharing one grabbed index means you can only ever hold ONE point on ONE structure at a time — grabbing the flag lets go of the rope automatically, and vice versa, for free, just by grab_at always overwriting the single shared field.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["grab_at's own d2 < GRAB_RADIUS**2 threshold means clicking on empty canvas correctly grabs nothing — grabbed[None] gets set to -1 in that case, not left stale."] },
        { title: "Fling it around", adding: "the live drag update and the HUD.",
          code: `        if gui.is_pressed(ti.GUI.LMB) and grabbed[None] >= 0:
            grab_target[None] = gui.get_cursor_pos()
        gui.text("drag to grab  [r] reset", (0.02, 0.98), color=0xAAAAAA)`,
          does: "While held, the target chases the cursor every frame, and predict teleports the grabbed particle there each tick — the rest of the structure gets DRAGGED ALONG purely through the constraint solver propagating that motion outward, tick by tick, exactly the way a real rope or flag would transmit a tug through its own fabric.",
          why: "That's the project: one general-purpose edge-list solver, described once, describing a chain and a woven grid equally well — and now something you can physically grab and throw. Yank the rope's free end and watch the whip travel down its length; grab a flag corner mid-flutter and watch the fold you create get absorbed back into the wind's rippling as you let go.",
          see: "Grab the rope's loose end and swing it — feel the whip travel down the chain. Grab a corner of the flag and pull; release and watch it snap back into the wind's flutter.",
          checkpoint: "Fully interactive cloth and rope. Final beat — project 10 complete.",
          recovery: ["The is_pressed check runs every frame the button is down, not just once on click — that's what makes it a continuous drag rather than a single teleport."] }
      ]
    }
  ]
};

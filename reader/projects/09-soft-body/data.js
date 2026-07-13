// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["09-soft-body"] = {
  project: "09-soft-body",
  title: "Soft Body",
  pitch: "Springs alone let a body collapse flat — internal pressure is what turns a spring loop into jelly, rubber, or a balloon.",
  tier: "medium",
  language: "Python",
  file: "soft_body.py",
  chapters: [
    {
      id: 1, title: "Rings of dots",
      build: "particle state for three ring-shaped bodies and a first static render.",
      beat: "Three motionless rings, already tuned for three different materials.",
      steps: [
        { title: "A breather after MPM", adding: "the docstring and imports.",
          code: `"""Soft Body: spring rings plus internal pressure — jelly, rubber, and a balloon, one engine."""
import math
import numpy as np
import taichi as ti`,
          does: "Project 08 simulated solids through a continuum field (deformation gradients, a grid, SVD). This project simulates a solid a much older, simpler way: a ring of point masses connected by springs. No grid, no SVD — just Hooke's law and an idea about air pressure.",
          why: "Different problems call for different tools. MPM shines at large deformable masses with genuinely different internal physics (snow crunching, sand flowing). A single bouncy blob doesn't need that machinery — a spring mesh is simpler, faster, and just as expressive for THIS shape of problem. Knowing when NOT to reach for the heavy tool is as valuable as knowing the tool.",
          see: "Runs clean.",
          checkpoint: "python3 soft_body.py returns silently.",
          recovery: ["Same venv ritual as always."] },
        { title: "Three bodies, three personalities", adding: "ring/body dials, per-material constant arrays, particle fields, and init_sim.",
          code: `N_RING = 28
N_BODIES = 3
N = N_RING * N_BODIES
BODY_RADIUS = 0.08
CENTERS = [(0.25, 0.6), (0.5, 0.6), (0.75, 0.6)]
STIFFNESS_NP = np.array([800.0, 4000.0, 300.0], dtype=np.float32)
DAMPING_NP = np.array([6.0, 2.0, 3.0], dtype=np.float32)
MASS_NP = np.array([0.02, 0.02, 0.02], dtype=np.float32)
pos = None
vel = None
rest_len = None
body_id = None
stiffness = None
damping = None
mass = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, rest_len, body_id, stiffness, damping, mass
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    rest_len = ti.field(ti.f32, shape=N)
    body_id = ti.field(ti.i32, shape=N)
    stiffness = ti.field(ti.f32, shape=N_BODIES)
    damping = ti.field(ti.f32, shape=N_BODIES)
    mass = ti.field(ti.f32, shape=N_BODIES)`,
          does: "N_RING particles per body, N_BODIES side by side. STIFFNESS_NP/DAMPING_NP/MASS_NP are three numbers each — one per body — already written with three DIFFERENT materials in mind, even though nothing reads them yet. rest_len will hold each particle's spring rest-length to its ring neighbor; body_id says which of the three rings a particle belongs to.",
          why: "Notice these arrays are tuned for three DIFFERENT bodies from the very first line that mentions them — unlike project 08, which discovered its second material (sand) in chapter 4, this project decides up front that variety is the point, and spends the next three chapters giving that variety somewhere to show up.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Per-body fields (stiffness, damping, mass) are shape=N_BODIES — three slots, not one per particle. Every particle looks its value up via body_id[p]."] },
        { title: "See the rings", adding: "the ring seeder, rest-length calculator, seed application, and a static render.",
          code: `def seed_ring(cx, cy, radius, n):
    """Pure numpy: n points evenly spaced around a circle."""
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1).astype(np.float32)
def rest_lengths(ring):
    nxt = np.roll(ring, -1, axis=0)
    return np.linalg.norm(nxt - ring, axis=1).astype(np.float32)
def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    mass.from_numpy(MASS_NP)
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Soft Body — taichi-academy", res=512, background_color=0x0A0A12)
    colors = np.zeros(N, dtype=np.uint32)
    colors[0:N_RING] = 0x8EC9FF
    colors[N_RING : 2 * N_RING] = 0xE8544A
    colors[2 * N_RING :] = 0xFFD35C
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        gui.circles(pos.to_numpy(), radius=2.5, color=colors)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "np.roll shifts every point one slot over, so nxt - ring gives every particle's vector to its ring-neighbor in one vectorized shot — rest_lengths never writes a Python loop over 28 points. apply_seed builds three separate rings, concatenates them into one flat array, and uploads everything the fields need, including their PERMANENT rest lengths (a regular polygon's edges are already equal, so this just records that shared value).",
          why: "Recording rest_len explicitly (rather than assuming it) is what makes the SAME spring code work for a body that's since been stretched or squashed — the spring always remembers what length it's trying to return to, no matter how far the current distance has drifted.",
          see: "Three motionless circles side by side — pale blue, red, gold — already differently sized-feeling even though nothing has moved.",
          checkpoint: "Three static rings. Beat 1.",
          recovery: ["np.concatenate(rings, axis=0) stacks the three (N_RING, 2) arrays into one (N, 2) array — axis=0 stacks rows, not columns."] }
      ]
    },
    {
      id: 2, title: "Springs and a floor",
      build: "gravity, structural springs between ring neighbors, and floor/wall collision — the full loop, no pressure yet.",
      beat: "Bodies fall — and may collapse flat. Springs alone can't stop it.",
      steps: [
        { title: "Weight and a rubber band", adding: "motion dials and the spring+gravity force kernel.",
          code: `DT = 5e-4
GRAVITY = 9.8
force = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, force, rest_len, body_id, stiffness, damping, mass
    force = ti.Vector.field(2, ti.f32, shape=N)
@ti.kernel
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
        force[q] -= f`,
          does: "Every particle starts with just its own weight. q is the NEXT particle around the same ring (the // and % arithmetic wraps within a body without ever crossing into the next one — the same trick project 06 used for its 3D cell index). stretch is how far past (or short of) rest length the spring currently is; rel_v is how fast the two ends are approaching or separating, PROJECTED onto the spring's direction — that's a damper, bleeding energy so the spring doesn't ring forever. Hooke's law (stiffness * stretch) plus that damping term give one combined force, applied equal-and-opposite to both ends.",
          why: "force[p] += f then force[q] -= f is Newton's third law, written directly: whatever p pulls on q, q pulls back on p exactly as hard, opposite direction. Get this backwards (a common mistake) and springs would inject energy instead of storing it — the whole system would fly apart instead of wobbling and settling.",
          see: "Runs clean; nothing calls this yet.",
          checkpoint: "No red text.",
          recovery: ["q's index arithmetic mirrors project 06's flat_cell — (p // N_RING) picks the body, (p + 1) % N_RING wraps to the next particle WITHIN it.", "dirn = d / dist, then stretch and rel_v both use dirn — the spring force only ever points along the current spring direction, never off-axis."] },
        { title: "A floor to land on", adding: "a world-size dial and the integrator.",
          code: `WORLD = 1.0
@ti.kernel
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
        pos[p] = newp`,
          does: "Standard semi-implicit Euler: accelerate, then move. The floor check does two things at once on contact — a bounce (vel.y flips and shrinks, 0.3x restitution) and friction (vel.x shrinks too, 0.7x, so a body doesn't just skid forever). Side walls bounce the same way, no friction needed there.",
          why: "Two DIFFERENT damping factors on the SAME collision (0.3 vertical, 0.7 horizontal) is a small detail with a big visual effect: pure elastic bounce (1.0) never settles; pure inelastic (0.0) glues to the floor instantly. Tuned restitution just under 1 is what makes a body feel like it's actually landing, not just stopping.",
          see: "Runs clean; still not wired into the render loop.",
          checkpoint: "No red text.",
          recovery: ["Two separate if blocks for the y-floor and the two x-walls — not elif — a particle could in principle need both corrections in the same tick (a corner case, literally)."] },
        { title: "Let go", adding: "the reset key, the physics call, and the tick conductor.",
          code: `SUBSTEPS = 30
def substep():
    compute_forces()
    integrate()
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()
        for _ in range(SUBSTEPS):
            substep()`,
          does: "30 substeps a frame — DT is tiny (5e-4) because explicit spring integration, like MPM's explicit stress update, is only stable at small steps; stiffer springs need smaller steps or they'll overshoot and blow up. R re-drops all three bodies from their start positions.",
          why: "Watch closely: depending on how stiff a body's springs are relative to its (nonexistent, so far) internal support, it may not just squash on landing — it may fold flat, particles sliding past each other into a degenerate line, and NEVER spring back. That's not a bug in your typing. A ring of springs with nothing resisting BENDING has a real physical weakness: you can flatten a loop of inextensible string into a straight line without stretching a single segment much. The next chapter's whole job is fixing exactly this.",
          see: "Three bodies drop and hit the floor. At least one — maybe all three, depending on how hard they land — may crumple into a flat smear instead of staying round.",
          checkpoint: "Bodies fall; shape may collapse. A real problem, on purpose. Beat 2.",
          recovery: ["If everything looks fine and none collapse, that's not wrong either — it depends on impact energy. Try dropping them from higher (raise CENTERS's y values) to see it more clearly.", "The substep loop goes right after the event loop, before gui.circles — same placement as every project since 01."] }
      ]
    },
    {
      id: 3, title: "Give it air",
      build: "an internal pressure force — the fix for chapter 2's collapse, and the reveal of three genuinely different materials.",
      beat: "No more collapsing — and jelly, rubber, and a balloon now visibly differ.",
      steps: [
        { title: "A dial nothing reads yet", adding: "a pressure constant per body and its field.",
          code: `GAS_NP = np.array([1.5, 1.0, 2.5], dtype=np.float32)
gas = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, force, rest_len, body_id, stiffness, damping, gas, mass
    gas = ti.field(ti.f32, shape=N_BODIES)
def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    gas.from_numpy(GAS_NP)
    mass.from_numpy(MASS_NP)`,
          does: "One more per-body number: how much 'gas' (in the loosest, ideal-gas-law sense) each body holds inside it. Notice the balloon (index 2) already has the most.",
          why: "PV = constant is the one piece of physics borrowed here: pressure is inversely proportional to the space it's confined to. Squeeze a body's enclosed area smaller and, if gas stays constant, pressure must rise — an automatic restoring force with no spring involved. That's the missing ingredient from chapter 2.",
          see: "Runs clean; not applied to any force yet.",
          checkpoint: "No red text.",
          recovery: ["gas is per-BODY (shape=N_BODIES), same pattern as stiffness/damping/mass — one value shared by every particle in a ring."] },
        { title: "The fix", adding: "the pressure force, computed from each body's own enclosed area.",
          code: `@ti.kernel
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
        force[q] -= f
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
            force[base + (i + 1) % N_RING] += f`,
          does: "The area sum (p0.x*p1.y - p1.x*p0.y, accumulated all the way around) is the shoelace formula — the exact enclosed area of any polygon, computed with no square roots and no trig. pressure = gas / area is literally PV = const, rearranged. Then every EDGE (not vertex) gets pushed along its own outward normal ([edge.y, -edge.x], a 90-degree rotation), split evenly onto its two endpoints — a real, physically-motivated 'inflate from the inside' force, not a shape-memory hack.",
          why: "This is why gas / area, not a constant push, matters: as a body gets squashed, its area shrinks, pressure rises automatically, and the restoring force gets STRONGER exactly when it's needed most — a genuine feedback loop, the same one that makes a real balloon push back harder the more you squeeze it.",
          see: "Drop all three again: nothing collapses anymore. And now they clearly differ — jelly compacts down a bit under its own weight, rubber barely bows at all (its stiffness was doing more work than you could feel before), and the balloon visibly PUFFS UP past its original size, exactly like a body with three times the internal gas should.",
          checkpoint: "Three distinct materials, no more collapsing. Beat 3.",
          recovery: ["The outward-normal formula assumes the ring is wound counter-clockwise (as seed_ring's increasing angle does) — reverse the winding and this force would push INWARD, accelerating the exact collapse it's meant to prevent.", "area gets a tiny +1e-6 floor — a body squashed to a sliver shouldn't ever divide by exactly zero."] }
      ]
    },
    {
      id: 4, title: "Grab and play",
      build: "click-and-drag grabbing, a release, and a HUD — turn the sandbox into a toy.",
      beat: "Reach in and pull a body around; let go and watch it spring back.",
      steps: [
        { title: "A hand that pulls, not teleports", adding: "grab dials, grab state, the pull force, and the finder.",
          code: `GRAB_K = 400.0
GRAB_DAMP = 4.0
GRAB_RADIUS = 0.1
grabbed = None
grab_target = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, force, rest_len, body_id, stiffness, damping, gas, mass
    global grabbed, grab_target
    grabbed = ti.field(ti.i32, shape=())
    grab_target = ti.Vector.field(2, ti.f32, shape=())
def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    gas.from_numpy(GAS_NP)
    mass.from_numpy(MASS_NP)
    grabbed[None] = -1
@ti.kernel
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
        force[q] -= f
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
            force[base + (i + 1) % N_RING] += f

    if grabbed[None] >= 0:
        g = grabbed[None]
        pull = GRAB_K * (grab_target[None] - pos[g]) - GRAB_DAMP * vel[g]
        force[g] += pull
def grab_at(mx, my):
    p = pos.to_numpy()
    d2 = (p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2
    i = int(np.argmin(d2))
    if d2[i] < GRAB_RADIUS**2:
        grabbed[None] = i
    else:
        grabbed[None] = -1
def release():
    grabbed[None] = -1`,
          does: "grabbed holds ONE particle index, or -1 for nothing. grab_at (plain Python, using numpy — no kernel needed for a one-off nearest-neighbor search over a few dozen points) finds whichever particle is closest to a click, as long as it's within GRAB_RADIUS. Once something's grabbed, compute_forces adds one more term: a spring-and-damper pull toward wherever the cursor currently is — the exact same stiffness*stretch + damping*rel_v shape as the ring springs, just pointed at grab_target instead of a ring neighbor.",
          why: "A SPRING pull toward the cursor, not teleporting the particle's position directly, matters more than it looks: project 08's stir bug came from injecting raw velocity every substep without accounting for how many substeps compose a frame. A force-based pull is self-limiting by construction — however fast you yank the mouse, the force is always proportional to CURRENT distance, so it can overshoot and wobble but can never inject unbounded energy the way a hard position-set could.",
          see: "Runs clean; not wired into mouse events yet.",
          checkpoint: "No red text.",
          recovery: ["grabbed[None] = -1 belongs in apply_seed too — pressing R while dragging should let go, not leave a stale index pointing at a freshly reset particle."] },
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
          does: "gui.get_events(ti.GUI.RELEASE) is new — every prior project only ever asked about PRESS events. It fires once, the instant a button comes back up.",
          why: "Press-and-release is a matched pair, same as project 08's dragging flag — but this time each half gets its own explicit event stream instead of an is_pressed poll, because 'the moment you let go' needs to be caught exactly once, not sampled every frame.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Two separate event loops — PRESS and RELEASE are different streams; you can't catch a release inside the PRESS loop."] },
        { title: "Feel it respond", adding: "the live drag-target update and the HUD.",
          code: `        if gui.is_pressed(ti.GUI.LMB) and grabbed[None] >= 0:
            grab_target[None] = gui.get_cursor_pos()
        gui.text("drag a body to grab it  [r] reset", (0.02, 0.98), color=0xAAAAAA)`,
          does: "While the button stays down AND something is grabbed, the target chases the live cursor every frame — that's what makes the pull feel like dragging instead of a single tug.",
          why: "That's the project: three different materials, one small spring-and-pressure engine, and now a hand in the middle of it. Reach into the balloon and it resists, puffs back; grab jelly and it lags, wobbles, settles slow; grab rubber and it snaps back almost instantly. Same code path, three different feelings — because three numbers per body said so.",
          see: "Click and drag any body around the canvas — feel the balloon push back hardest, jelly wobble loosest, rubber snap back fastest. Let go and watch each one spring toward its rest shape in its own characteristic way.",
          checkpoint: "Fully interactive soft bodies. Final beat — project 09 complete.",
          recovery: ["The is_pressed check happens EVERY frame (not just on click) — that's what makes it a continuous drag instead of a one-time pull."] }
      ]
    }
  ]
};

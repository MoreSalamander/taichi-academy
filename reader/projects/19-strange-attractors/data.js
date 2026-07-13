// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["19-strange-attractors"] = {
  project: "19-strange-attractors",
  title: "Strange Attractors",
  pitch: "Three equations, no randomness — and 200,000 points that never repeat and never escape. Chaos, drawn in light.",
  tier: "easy",
  language: "Python",
  file: "strange_attractors.py",
  chapters: [
    {
      id: 1, title: "The butterfly catches everything",
      build: "200,000 points, the Lorenz equations, and a splat — then watch a random cloud collapse onto the attractor, live.",
      beat: "A shapeless cloud of dots falls, in seconds, onto the famous butterfly.",
      steps: [
        { title: "Art from pure math", adding: "the docstring and imports.",
          code: `"""Strange Attractors: 200,000 points fall onto the same impossible shape, every time."""
import numpy as np
import taichi as ti`,
          does: "Arc 4 drops physics entirely: no forces, no materials, no conservation laws — just iterated equations and what they look like. A strange attractor is the shape a chaotic system's trajectories settle onto: never repeating, never escaping, tracing the same ghostly form forever. This project renders four famous ones with the particle-cloud + additive-light machinery you already own.",
          why: "Everything here is reuse: project 14's fade-splat-clamp canvas, project 12's rotating projection idea, project 07's particle pool. The only genuinely new content is a handful of differential equations — which is the point. The pipeline is now vocabulary; the equations are the interchangeable part.",
          see: "Runs clean.",
          checkpoint: "python3 strange_attractors.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "A cloud and a canvas", adding: "count dials, the framing table, and both fields.",
          code: `RES = 512
N_PTS = 200000
FADE = 0.90
LORENZ = 0
# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0),
}
pos = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(3, ti.f32, shape=N_PTS)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "200,000 free points in 3D, and a FRAME table holding each attractor's presentation numbers: where it lives in space (center), how big it draws (scale), how fast its clock ticks (dt), how bright each point splats (gain), and how wide to scatter the starting cloud (seed_scale). One entry for now.",
          why: "Every attractor lives at its own scale — Lorenz sprawls across ±20 with its center hovering at z=25; Aizawa fits inside ±1.5. Framing data (a python dict, read outside kernels, passed in as arguments) keeps those presentation decisions OUT of the math kernels, which will stay pure equations.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["FRAME values are passed to kernels as plain float arguments — a python dict can't be read inside a Taichi kernel."] },
        { title: "Watch the collapse", adding: "the seeder, the Lorenz equations, an Euler step, the splat, and the loop.",
          code: `def seed_points(n, seed_scale, rng_seed=0):
    """Pure numpy: a random cloud sized to land inside the attractor's basin."""
    rng = np.random.default_rng(rng_seed)
    return (rng.uniform(-1.0, 1.0, size=(n, 3)) * seed_scale).astype(np.float32)
@ti.func
def deriv_lorenz(p):
    return ti.Vector([10.0 * (p[1] - p[0]), p[0] * (28.0 - p[2]) - p[1], p[0] * p[1] - 8.0 / 3.0 * p[2]])
@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        pos[i] = p + dt * deriv_lorenz(p)
def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    pixels.fill(0.0)
@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE
@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32):
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0]
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] += ti.Vector([0.9, 0.9, 1.0]) * gain
@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)
def step(kind, angle):
    scale, cx, cy, cz, dt, gain, _seed = FRAME[kind]
    step_attractor(kind, dt)
    fade()
    splat(kind, angle, scale, cx, cy, cz, gain)
    clamp_pixels()
def main():
    init_sim()
    kind = LORENZ
    apply_seed(kind)
    gui = ti.GUI("Strange Attractors — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step(kind, 0.0)
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "deriv_lorenz is the whole Lorenz system — three coupled equations Edward Lorenz distilled from weather convection in 1963, with the classic constants (10, 28, 8/3) baked in. step_attractor is a plain Euler step: every point moves a little along its own local flow direction, all 200,000 in parallel. The splat projects x horizontally, z vertically (a fixed side-on view), and deposits faint white light.",
          why: "No settle phase yet, deliberately: the first thing you see is a fat random blob of 200,000 points DISSOLVING — streaming along invisible rails, funneling into two lobes, and within seconds tracing the butterfly. Every point takes a different path; every point ends up on the same shape. That collapse IS what 'attractor' means, and no still image teaches it like watching it happen.",
          see: "A square blur of dots comes apart like smoke in wind and re-forms as the Lorenz butterfly — two swirling wings joined at a waist, drawn in accumulating light.",
          checkpoint: "The cloud collapses onto the butterfly. Beat 1.",
          recovery: ["splat already takes kind and angle arguments it doesn't use yet — the signature is future-proofed so later chapters change bodies, not call sites.", "If the screen saturates white, FADE or gain is off — the canvas needs to forget faster than the points deposit."] }
      ]
    },
    {
      id: 2, title: "Spin it, color it, tame it",
      build: "a rotating 3D projection, velocity-mapped color, then a settle phase and a reseed key.",
      beat: "The butterfly turns in space, colored by speed — blue where it glides, gold where it whips.",
      steps: [
        { title: "Rotation and heat", adding: "a spin dial, the rotating projection, and speed-mapped color (replace FRAME, apply_seed, splat, step).",
          code: `ROT_SPEED = 0.01
# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
}
def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale, _spd = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    pixels.fill(0.0)
@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = ti.math.clamp(deriv_lorenz(pos[i]).norm() * speed_scale, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain
def step(kind, angle):
    scale, cx, cy, cz, dt, gain, _seed, speed_scale = FRAME[kind]
    step_attractor(kind, dt)
    fade()
    splat(kind, angle, scale, cx, cy, cz, gain, speed_scale)
    clamp_pixels()
    angle = 0.0
        angle += ROT_SPEED
        step(kind, angle)`,
          does: "rx = x·cos + y·sin rotates the horizontal axes before projecting — the camera slowly circles the attractor while z stays vertical, exactly project 12/13's orbit idea collapsed to one line of trigonometry. And each point now colors itself by its own SPEED: the norm of its derivative, mapped from deep blue (gliding) to hot gold (whipping through a tight turn).",
          why: "Speed coloring isn't decoration — it's reading the mathematics off the screen. The derivative IS the system's velocity, so the gold zones show where the flow accelerates (the tight turnaround under each Lorenz wing) and the blue where it lingers. You're visualizing the vector field through the particles that ride it.",
          see: "The butterfly turns slowly in space, revealing it's not flat — two tilted discs in 3D — with golden fringes where trajectories whip around the wing edges and cool blue sheets where they cruise.",
          checkpoint: "A rotating, speed-colored butterfly. No red text.",
          recovery: ["step's FRAME unpack grows to 8 values — every unpack site must match the new tuple length or Python raises ValueError at the first frame."] },
        { title: "Skip the transient", adding: "a settle loop in apply_seed and the reseed key.",
          code: `SETTLE_STEPS = 2000
def apply_seed(kind, rng_seed=0):
    _scale, _cx, _cy, _cz, dt, _gain, seed_scale, _spd = FRAME[kind]
    pos.from_numpy(seed_points(N_PTS, seed_scale, rng_seed))
    for _ in range(SETTLE_STEPS):
        step_attractor(kind, dt)
    pixels.fill(0.0)
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))`,
          does: "apply_seed now runs 2,000 silent steps before showing anything — the cloud lands on the attractor off-screen, so a reseed (or a later attractor switch) starts with the shape already fully formed instead of replaying the collapse every time.",
          why: "Dynamicists call the settling journey the TRANSIENT, and discarding it is standard practice: the attractor is the long-run behavior; the transient is an artifact of where you happened to start. Chapter 1 showed the transient on purpose (it teaches what attraction means); from here on it's noise, and 2,000 cheap steps make it vanish. This was also a real tuning discovery: an early build settled only 200 steps, and Lorenz drew as a shapeless bright blob — points still bunched mid-transit, nowhere near covering the attractor.",
          see: "Tap R: the fully-formed butterfly reappears instantly from a completely different starting cloud — same shape, always. That instant sameness is the attractor's whole claim.",
          checkpoint: "Reseeds land pre-settled. Beat 2.",
          recovery: ["The settle loop runs BEFORE pixels.fill(0.0) — no light from the journey survives onto the canvas."] }
      ]
    },
    {
      id: 3, title: "A zoo of chaos",
      build: "three more systems — Thomas, Aizawa, and the Clifford map — behind one keyboard switch.",
      beat: "Four wildly different infinities, one keystroke apart.",
      steps: [
        { title: "Thomas: chaos from sine waves", adding: "the second attractor and the kind plumbing (replace the id line, FRAME, step_attractor, splat, and the event block).",
          code: `LORENZ, THOMAS = 0, 1
# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
}
@ti.func
def deriv_thomas(p):
    b = 0.19
    return ti.Vector([ti.sin(p[1]) - b * p[0], ti.sin(p[2]) - b * p[1], ti.sin(p[0]) - b * p[2]])
@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        else:
            pos[i] = p + dt * deriv_thomas(p)
@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            else:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "12":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))`,
          does: "Thomas' system is almost embarrassingly simple — sin of the NEXT coordinate minus a little damping, cycled through x→y→z symmetrically — and it produces one of the most beautiful attractors known: an endlessly interweaving celtic knot. The kind argument, dormant since chapter 1, finally earns its place: one branch in the stepper, one in the splat's speed lookup, and the whole pipeline serves two systems.",
          why: "Look at what did NOT change: seed_points, fade, clamp, step, apply_seed's structure, the canvas — the entire pipeline. Adding a universe of new behavior cost one @ti.func of equations and a FRAME row. That ratio — new math to new plumbing — is the payoff of every architectural habit this curriculum has drilled.",
          see: "Press 2: the butterfly dissolves and a tangled, perfectly symmetric knot of glowing threads fades in, rotating — noticeably calmer and more even-toned than Lorenz.",
          checkpoint: "Two attractors, switchable. No red text.",
          recovery: ["Thomas' cyclic symmetry (x→y→z) means its cloud spreads identically along all three axes — a property the tests actually verify numerically."] },
        { title: "Aizawa: the sphere with a spike", adding: "the third system (replace the id line, FRAME, step_attractor, splat, events).",
          code: `LORENZ, THOMAS, AIZAWA = 0, 1, 2
# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
    AIZAWA: (0.28, 0.0, 0.0, 0.0, 0.01, 0.015, 0.4, 0.35),
}
@ti.func
def deriv_aizawa(p):
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
    x, y, z = p[0], p[1], p[2]
    return ti.Vector([
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z**3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x**3,
    ])
@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        elif kind == THOMAS:
            pos[i] = p + dt * deriv_thomas(p)
        else:
            pos[i] = p + dt * deriv_aizawa(p)
@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            elif kind == THOMAS:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            else:
                spd = deriv_aizawa(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "123":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))`,
          does: "Aizawa is the temperamental one: six tuned constants, a z-cubed term, and an attractor shaped like a hollow sphere with a glowing column threading its axis. Note its FRAME row: seed_scale is 0.4, twenty times tighter than Lorenz's 8.0.",
          why: "That tiny seed_scale records a real failure from this project's development: reusing Lorenz's wide seeding cloud sent Aizawa's points straight to infinity — its z³ term diverges outside a small basin of attraction, and the whole field went NaN. Attractors only attract from WITHIN their basin; where you're allowed to start is part of each system's identity, which is exactly why seed_scale lives in the per-attractor table and not as a global constant.",
          see: "Press 3: a translucent golden sphere with a bright blue-white column twisting up through its core — utterly unlike the other two.",
          checkpoint: "Three attractors. No red text.",
          recovery: ["If Aizawa shows a black screen, the points have diverged — check seed_scale is 0.4, not something Lorenz-sized."] },
        { title: "Clifford: a map, not a flow", adding: "the fourth system, the name table, drag-to-spin, and the HUD.",
          code: `LORENZ, THOMAS, AIZAWA, CLIFFORD = 0, 1, 2, 3
NAMES = {LORENZ: "lorenz", THOMAS: "thomas", AIZAWA: "aizawa", CLIFFORD: "clifford"}
# per-attractor framing: (scale, cx, cy, cz, dt, gain, seed_scale, speed_scale)
FRAME = {
    LORENZ: (0.018, 0.0, 0.0, 25.0, 0.004, 0.10, 8.0, 0.006),
    THOMAS: (0.11, 0.0, 0.0, 0.0, 0.06, 0.05, 3.0, 0.8),
    AIZAWA: (0.28, 0.0, 0.0, 0.0, 0.01, 0.015, 0.4, 0.35),
    CLIFFORD: (0.20, 0.0, 0.0, 0.0, 1.0, 0.04, 1.0, 0.0),
}
@ti.func
def map_clifford(p):
    a, b, c, d = -1.4, 1.6, 1.0, 0.7
    return ti.Vector([ti.sin(a * p[1]) + c * ti.cos(a * p[0]), ti.sin(b * p[0]) + d * ti.cos(b * p[1]), 0.0])
@ti.kernel
def step_attractor(kind: ti.i32, dt: ti.f32):
    for i in pos:
        p = pos[i]
        if kind == LORENZ:
            pos[i] = p + dt * deriv_lorenz(p)
        elif kind == THOMAS:
            pos[i] = p + dt * deriv_thomas(p)
        elif kind == AIZAWA:
            pos[i] = p + dt * deriv_aizawa(p)
        else:
            pos[i] = map_clifford(p)
@ti.kernel
def splat(kind: ti.i32, angle: ti.f32, scale: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
          gain: ti.f32, speed_scale: ti.f32):
    ca = ti.cos(angle)
    sa = ti.sin(angle)
    for i in pos:
        p = pos[i] - ti.Vector([cx, cy, cz])
        rx = p[0] * ca + p[1] * sa
        ry = p[2]
        if kind == CLIFFORD:
            rx = p[0]
            ry = p[1]
        x = 0.5 + rx * scale
        y = 0.5 + ry * scale
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            spd = 1.0
            if kind == LORENZ:
                spd = deriv_lorenz(pos[i]).norm() * speed_scale
            elif kind == THOMAS:
                spd = deriv_thomas(pos[i]).norm() * speed_scale
            elif kind == AIZAWA:
                spd = deriv_aizawa(pos[i]).norm() * speed_scale
            spd = ti.math.clamp(spd, 0.0, 1.0)
            cool = ti.Vector([0.15, 0.3, 0.9])
            hot = ti.Vector([1.0, 0.85, 0.4])
            pixels[xi, yi] += (cool * (1 - spd) + hot * spd) * gain
    pmx = None
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "1234":
                kind = int(e.key) - 1
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
            elif e.key == "r":
                apply_seed(kind, rng_seed=np.random.randint(1_000_000))
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                angle -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            angle += ROT_SPEED
        gui.text(f"attractor: {NAMES[kind]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1] lorenz  [2] thomas  [3] aizawa  [4] clifford  drag to spin  [r] rescatter", (0.02, 0.94), color=0xAAAAAA)`,
          does: "Clifford is a different SPECIES of dynamical system: a discrete MAP, not a continuous flow. Where the other three integrate (pos += dt * derivative — a small step along a smooth path), Clifford TELEPORTS: pos = f(pos), a whole jump per tick, no dt, no in-between. In the stepper that's the difference between p + dt*deriv(p) and plain map(p). It's also 2D, so the splat projects its x/y directly, skipping the rotation — a flat lace doesn't need an orbit.",
          why: "Flows and maps are the two great families of dynamical systems, and now you've implemented both in one kernel and can SEE the difference: the flows draw smooth luminous threads (each point crawls, its trail connected); Clifford renders as pure pointillism — dust that never moves so much as reappears, because consecutive positions of one point are nowhere near each other. Same attractor concept, entirely different texture.",
          see: "Press 4: a gauzy, veil-like 2D lace materializes — sharp filaments and soft shadows, static in structure yet shimmering as 200,000 points teleport around it. Drag any of the four to spin it under your hand.",
          checkpoint: "Four systems, drag-to-spin, a HUD. Final beat — project 19 complete.",
          recovery: ["Clifford's FRAME dt is 1.0 and speed_scale 0.0 — a map has no meaningful dt or derivative; the numbers are placeholders the branches never use.", "The drag idiom is the same pmx pattern as projects 12/13 — fifth appearance; by now it should type itself."] }
      ]
    }
  ]
};

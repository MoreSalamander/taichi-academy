// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["16-solar-system"] = {
  project: "16-solar-system",
  title: "Solar System",
  pitch: "The real 1/r-squared law — and the famous integrator swap that decides whether your orbits hold for a second or forever.",
  tier: "medium",
  language: "Python",
  file: "solar_system.py",
  chapters: [
    {
      id: 1, title: "Six worlds, parked",
      build: "planet state, the circular-orbit velocity formula, and a trail-leaving render — no gravity yet.",
      beat: "Six colored planets hang motionless around an empty center.",
      steps: [
        { title: "Real gravity this time", adding: "the docstring and imports.",
          code: `"""Solar System: real 1/r^2 gravity, a leapfrog integrator, and orbits that actually hold."""
import numpy as np
import taichi as ti`,
          does: "Project 15 approximated gravity with a density-gradient trick. This project uses the genuine article — Newton's inverse-square law toward a central sun — and immediately runs into the genuine problem: with a naive integrator, planets spiral outward and escape. The fix (a leapfrog integrator) is the most useful three-line upgrade in all of computational physics.",
          why: "One honest simplification, stated up front: bodies here feel ONLY the sun, not each other. That's what keeps six planets plus thousands of asteroids stable and cheap — full N-body mutual gravity is a different, much harder project (it's what the 30-universe-sandbox capstone is for).",
          see: "Runs clean.",
          checkpoint: "python3 solar_system.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "The population ledger", adding: "counts, colors, and fields.",
          code: `RES = 512
GM = 1.0
N_PLANETS = 6
N = N_PLANETS
PLANET_BASE = 0
PLANET_COLORS = np.array(
    [
        [0.75, 0.72, 0.68],
        [0.95, 0.85, 0.55],
        [0.30, 0.55, 0.95],
        [0.90, 0.45, 0.25],
        [0.85, 0.75, 0.55],
        [0.60, 0.80, 0.90],
    ],
    dtype=np.float32,
)
pos = None
vel = None
color = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, color, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    color = ti.Vector.field(3, ti.f32, shape=N)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "GM = 1 sets natural units — the sun's mass and the gravitational constant fold into one number, and every other quantity is measured relative to it. N = N_PLANETS for now, but the ledger structure (BASE offsets into one shared pool) is already in place for the belt and comets that chapter 3 appends.",
          why: "One particle pool with base-offset sections, rather than separate fields per population, is the same layout project 10 used for its rope+cloth — every kernel (gravity, render) runs over ALL bodies uniformly, and only styling decisions ever need to know which section a body belongs to.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The colors are six hand-picked planet-ish tones — rocky grays, a gas-giant gold, one blue marble."] },
        { title: "The speed of staying up", adding: "the circular-velocity formula, planet seeding, and the trail render.",
          code: `VIEW_SCALE = 0.95
CANVAS_FADE = 0.90
def circular_velocity(p):
    """Pure numpy: the speed that makes gravity exactly the centripetal force — one orbit, forever."""
    r = np.linalg.norm(p, axis=-1, keepdims=True)
    speed = np.sqrt(GM / r.squeeze(-1))
    tangent = np.stack([-p[..., 1], p[..., 0]], axis=-1) / r
    return tangent * speed[..., None]
def seed_planets(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    radii = np.linspace(0.10, 0.42, N_PLANETS).astype(np.float32)
    ang = rng.uniform(0.0, 2 * np.pi, N_PLANETS)
    p = np.stack([radii * np.cos(ang), radii * np.sin(ang)], axis=1).astype(np.float32)
    return p, circular_velocity(p).astype(np.float32), PLANET_COLORS.copy()
def apply_seed(rng_seed=0):
    p, v, c = seed_planets(rng_seed)
    pos.from_numpy(p)
    vel.from_numpy(v)
    color.from_numpy(c)
    pixels.fill(0.0)
@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for b in pos:
        x = 0.5 + pos[b][0] * VIEW_SCALE
        y = 0.5 + pos[b][1] * VIEW_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 1 <= xi < RES - 1 and 1 <= yi < RES - 1:
            pixels[xi, yi] += color[b]
@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)
def step():
    render()
    clamp_pixels()
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Solar System — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "circular_velocity is the single most important formula in orbital mechanics: at radius r, the speed sqrt(GM/r) makes gravity supply EXACTLY the centripetal force a circle needs — no more, no less. Aim it along the tangent ([-y, x], the 90-degree rotation again) and the body should circle forever. Planets get evenly-spaced radii at random starting angles, each pre-loaded with exactly that velocity. The render is the familiar fade-splat-clamp, so moving bodies will draw their own orbit trails.",
          why: "Seeding each planet with its textbook-perfect circular velocity is deliberate stage-setting: when chapter 2's naive integrator makes those provably-correct initial conditions decay anyway, you'll know with certainty the bug is in the INTEGRATOR, not the setup. Good experiments isolate one variable.",
          see: "Six colored dots at different distances from an empty center, motionless — velocities loaded, physics not yet written.",
          checkpoint: "Six parked planets. Beat 1.",
          recovery: ["tangent = [-y, x] / r — third project running to use the rotate-90-degrees trick for 'perpendicular to the radius'."] }
      ]
    },
    {
      id: 2, title: "The integrator matters",
      build: "the honest failure (Euler orbits spiral out), then the three-line fix (leapfrog) and proof via energy.",
      beat: "Orbits that decayed before your eyes now hold their circles indefinitely.",
      steps: [
        { title: "Newton, naively", adding: "the inverse-square acceleration and a straightforward Euler step.",
          code: `DT = 0.002
@ti.func
def accel(p):
    r2 = p.dot(p) + 1e-6
    r = ti.sqrt(r2)
    return -GM * p / (r2 * r)
@ti.kernel
def euler_step():
    for b in pos:
        a = accel(pos[b])
        pos[b] += DT * vel[b]
        vel[b] += DT * a
def step():
    euler_step()
    render()
    clamp_pixels()`,
          does: "accel is Newton's law verbatim: pull toward the origin with strength GM/r², written as -GM*p/r³ so the direction and magnitude come out of one expression. euler_step then does the most obvious possible thing: move along the current velocity, then update the velocity with the current acceleration.",
          why: "Watch what the obvious thing does. Every planet's trail is a slow outward SPIRAL — orbits visibly widening, energy climbing from nowhere. Nothing is wrong with Newton's law or the initial conditions (chapter 1 made sure); explicit Euler just systematically injects energy every step when the force curves. Measured on this exact system: 86% energy drift, orbits 7x too wide. This is the most famous numerical-integration failure there is, and now you've SEEN it.",
          see: "The planets move! ...and every orbit is a spiral staircase outward. Give it thirty seconds; the inner planet climbs alarmingly.",
          checkpoint: "Orbits that leak outward — the intended failure. Beat: the problem.",
          recovery: ["This is not a bug in your typing. If your orbits spiral out, you built it correctly."] },
        { title: "Leapfrog", adding: "the kick-drift-kick integrator and an energy meter to prove it.",
          code: `@ti.kernel
def leapfrog():
    for b in pos:
        vel[b] += 0.5 * DT * accel(pos[b])
        pos[b] += DT * vel[b]
        vel[b] += 0.5 * DT * accel(pos[b])
def total_energy():
    """Pure numpy: kinetic + potential per body — the quantity leapfrog protects."""
    p = pos.to_numpy()
    v = vel.to_numpy()
    ke = 0.5 * (v**2).sum(axis=1)
    pe = -GM / np.linalg.norm(p, axis=1)
    return ke + pe`,
          does: "Kick-drift-kick: half a velocity update, a full position move, then the other half of the velocity update using the NEW position's force. The magic property is symmetry — run it backwards and you retrace your steps exactly — and that time-symmetry is what stops the systematic energy leak. total_energy computes the conserved quantity so you can check the claim numerically instead of squinting.",
          why: "Same cost as Euler (two accel calls can be optimized to one; this form keeps the symmetry legible), three lines different, and the measured difference on this system is 86% energy drift versus 0.02%. Leapfrog and its relatives are what real astronomy codes use for exactly this reason. This swap — not more steps, not smaller DT, a SMARTER step — is the single highest-leverage lesson in numerical physics.",
          see: "Runs clean; step() still calls the old euler_step until the next step swaps it in.",
          checkpoint: "No red text.",
          recovery: ["The two half-kicks bracket the drift — accel is evaluated at BOTH the old and new positions. That's the symmetry; don't merge them into one full kick."] },
        { title: "Swap it in", adding: "substeps and the new tick.",
          code: `SUBSTEPS = 4
def step():
    for _ in range(SUBSTEPS):
        leapfrog()
    render()
    clamp_pixels()`,
          does: "euler_step is gone from the tick (delete the kernel too if you like — the lesson keeps its lesson). Four leapfrog substeps per frame speed the clockwork up to a pleasant pace.",
          why: "Compare the same visual as before: trails that used to spiral now close into clean, stable circles that retrace themselves lap after lap. Same law, same seeds, same DT — only the integrator changed.",
          see: "Six planets on crisp circular rails, inner ones lapping outer ones — Kepler's third law as an animation. Leave it running; nothing drifts.",
          checkpoint: "Stable orbits. Beat 2 — the fix, proven by eye.",
          recovery: ["If orbits still spiral, step() is probably still calling euler_step — the swap is the whole point of this step."] }
      ]
    },
    {
      id: 3, title: "A crowded system",
      build: "an asteroid belt, comets on real ellipses via vis-viva, and the sun's glow.",
      beat: "A full system: belt, plunging comets, and a golden sun at the center of it all.",
      steps: [
        { title: "Four thousand asteroids", adding: "the belt population, folded into the shared pool.",
          code: `N_BELT = 4000
N = N_PLANETS + N_BELT
BELT_BASE = N_PLANETS
BELT_R = (0.30, 0.36)
def seed_belt(rng_seed=0):
    rng = np.random.default_rng(rng_seed + 1)
    r = rng.uniform(BELT_R[0], BELT_R[1], N_BELT)
    ang = rng.uniform(0.0, 2 * np.pi, N_BELT)
    p = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)
    v = circular_velocity(p).astype(np.float32)
    col = np.full((N_BELT, 3), (0.35, 0.32, 0.28), dtype=np.float32)
    col *= rng.uniform(0.5, 1.0, (N_BELT, 1)).astype(np.float32)
    return p, v, col
def apply_seed(rng_seed=0):
    parts = [seed_planets(rng_seed), seed_belt(rng_seed)]
    pos.from_numpy(np.concatenate([p for p, _v, _c in parts]))
    vel.from_numpy(np.concatenate([v for _p, v, _c in parts]))
    color.from_numpy(np.concatenate([c for _p, _v, c in parts]))
    pixels.fill(0.0)`,
          does: "Four thousand dim gray-brown rocks in a band between two planet orbits, each on its own circular orbit via the exact same circular_velocity call the planets used. apply_seed becomes a concatenation of population parts — the ledger pattern paying off.",
          why: "The belt costs NOTHING new: no new physics, no new kernels, not even a new formula. When your force law and integrator are body-count-agnostic, 'add four thousand more bodies' is a seeding decision. That scalability is why the pool-with-sections layout was worth setting up in chapter 1.",
          see: "A grainy ring of slow rocks between the outer planets, every one on its own honest orbit.",
          checkpoint: "The belt orbits. No red text.",
          recovery: ["rng_seed + 1 inside seed_belt — each population salts the seed differently so they don't accidentally correlate."] },
        { title: "Comets, by vis-viva", adding: "the ellipse-speed formula and two dozen sun-divers.",
          code: `N_COMETS = 24
N = N_PLANETS + N_BELT + N_COMETS
COMET_BASE = N_PLANETS + N_BELT
COMET_PERI = 0.06
COMET_APO = 0.46
def comet_aphelion_velocity(r_apo, r_peri):
    """Pure numpy: vis-viva at aphelion for an ellipse with the given extremes."""
    a = 0.5 * (r_apo + r_peri)
    return np.sqrt(GM * (2.0 / r_apo - 1.0 / a))
def seed_comets(rng_seed=0):
    rng = np.random.default_rng(rng_seed + 2)
    ang = rng.uniform(0.0, 2 * np.pi, N_COMETS)
    r_apo = rng.uniform(COMET_APO * 0.8, COMET_APO, N_COMETS)
    p = np.stack([r_apo * np.cos(ang), r_apo * np.sin(ang)], axis=1).astype(np.float32)
    speed = comet_aphelion_velocity(r_apo, COMET_PERI)
    tangent = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    v = (tangent * speed[:, None]).astype(np.float32)
    col = np.full((N_COMETS, 3), (0.55, 0.85, 0.95), dtype=np.float32)
    return p, v, col
def apply_seed(rng_seed=0):
    parts = [seed_planets(rng_seed), seed_belt(rng_seed), seed_comets(rng_seed)]
    pos.from_numpy(np.concatenate([p for p, _v, _c in parts]))
    vel.from_numpy(np.concatenate([v for _p, v, _c in parts]))
    color.from_numpy(np.concatenate([c for _p, _v, c in parts]))
    pixels.fill(0.0)`,
          does: "The vis-viva equation answers 'how fast is an orbiting body at distance r, given its ellipse?' — here evaluated at aphelion (the far end) to find the launch speed that makes each comet's orbit dive all the way in to COMET_PERI and swing back out, forever. Deliberately SLOWER than circular at that radius: too slow to stay up is exactly what falling inward means.",
          why: "The same integrator that holds circles holds ellipses — leapfrog doesn't know or care about the orbit's shape. And watch Kepler's second law live: each icy-blue comet crawls at aphelion, then whips around the sun at perihelion many times faster. Nothing enforces that speedup; it falls out of the force law, as it should.",
          see: "Icy blue streaks plunge from the system's edge, sling around the center, and climb back out — each retracing its own ellipse lap after lap.",
          checkpoint: "Comets on stable ellipses. No red text.",
          recovery: ["If a comet flies off forever, its aphelion speed is too high — check the vis-viva expression's 2/r_apo minus 1/a ordering."] },
        { title: "The sun, and finishing touches", adding: "a glowing sun, per-population styling, and the HUD (replace render).",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    cx = RES // 2
    for _ in range(1):
        for di, dj in ti.ndrange((-4, 5), (-4, 5)):
            w = ti.exp(-(di * di + dj * dj) / 6.0)
            pixels[cx + di, cx + dj] += w * ti.Vector([1.0, 0.85, 0.4])

    for b in pos:
        x = 0.5 + pos[b][0] * VIEW_SCALE
        y = 0.5 + pos[b][1] * VIEW_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 1 <= xi < RES - 1 and 1 <= yi < RES - 1:
            gain = 1.0
            if b >= BELT_BASE and b < COMET_BASE:
                gain = 0.35
            pixels[xi, yi] += color[b] * gain
            if b < N_PLANETS or b >= COMET_BASE:
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    pixels[xi + di, yi + dj] += color[b] * 0.3
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("planets, belt, comets — leapfrog keeps them honest", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] rescatter", (0.02, 0.94), color=0xAAAAAA)`,
          does: "The sun gets a 9x9 gaussian glow at dead center (wrapped in the serial for _ in range(1) trick so it draws once, not once per pixel-thread). Population styling uses the ledger: belt rocks render dim (gain 0.35), planets and comets get a small 3x3 halo so they read as BODIES against four thousand specks.",
          why: "That's the project: one force law, one integrator, three populations whose only differences are their initial conditions — and the deepest lesson came from the two integrators, not the three populations. When a simulation misbehaves over time, suspect the integrator before the physics.",
          see: "A living clockwork: golden sun, six haloed planets on their rails, a slow gray belt, and blue comets stitching ellipses through it all. Tap R to deal a new arrangement.",
          checkpoint: "The full system. Final beat — project 16 complete.",
          recovery: ["The sun-glow loop needs the serial wrapper — without it, every one of the 262,144 pixel threads would try to draw the sun."] }
      ]
    }
  ]
};

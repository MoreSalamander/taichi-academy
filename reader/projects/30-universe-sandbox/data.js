// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["30-universe-sandbox"] = {
  project: "30-universe-sandbox",
  title: "Universe Sandbox",
  pitch: "The final project: nothing but Newton's law of gravity, summed over every pair of bodies and stepped with a symplectic integrator. From it — rotating galaxies, orbiting black holes, colliding disks that fling out tidal streams of stars.",
  tier: "epic",
  language: "Python",
  file: "universe_sandbox.py",
  chapters: [
    {
      id: 1, title: "Gravity and orbits",
      build: "direct N-body gravity with softening and a symplectic leapfrog integrator — a single galaxy of stars orbiting a central black hole, stable for ages.",
      beat: "A spiral of thousands of stars wheels steadily around a blazing central black hole.",
      steps: [
        { title: "The floor beneath everything", adding: "the docstring and imports.",
          code: `"""Universe Sandbox: direct N-body gravity with softening and a symplectic leapfrog — rotating
galaxies around black holes that orbit, collide, and fling out tidal streams of stars."""
import numpy as np
import taichi as ti`,
          does: "The thirtieth and final project, and fittingly it runs on the oldest rule in physics: every mass attracts every other with a force that falls off as one-over-distance-squared. That's the entire law. Sum it over thousands of stars, step it forward in time, and out come galaxies, orbits, and collisions — the largest structures in the universe from the simplest possible interaction.",
          why: "It's the perfect closing note for a curriculum about emergence: no simulation could be more purely 'local rule, global pattern.' There is nothing in the code about spirals or galaxies — just pairwise gravity. Everything you'll see is what a swarm of gravitating points DOES. numpy builds the galaxies; Taichi computes the all-pairs force on the GPU every frame.",
          see: "Runs clean.",
          checkpoint: "python3 universe_sandbox.py returns silently.",
          recovery: ["Usual venv setup: source .venv/bin/activate, then run from the project folder."] },
        { title: "Bodies and dials", adding: "the constants and fields.",
          code: `N = 4000                 # bodies (stars + a couple of black holes)
G = 1.0
EPS = 0.02               # softening length — smooths the 1/r^2 singularity at close range
DT = 0.008
RES = 512
VIEW = 1.8               # half-width of the viewport in world units
BH_MASS = 0.15           # a galaxy's central black hole dominates its potential
STAR_MASS = 1e-5         # stars are near-massless tracers of the gravitational field
pos = None
vel = None
acc = None
mass = None
pixels = None`,
          does: "Each of the N bodies carries a position, velocity, acceleration, and mass — that's a full gravitational simulation. The masses tell the whole story: a black hole at BH_MASS = 0.15 utterly dominates the near-massless stars (STAR_MASS = 1e-5), so each galaxy's stars orbit their central hole like planets around a sun. EPS, the softening length, is the single most important number here.",
          why: "Softening (EPS) is the trick that makes N-body gravity survivable. The true force blows up to infinity as two bodies approach (one over distance squared, with distance going to zero), which would send a star to infinite speed in one timestep. Softening replaces the distance with sqrt(distance-squared + EPS-squared), capping the force at close range — as if each body were a little cloud rather than a point. Making the stars near-massless is a classic move too (it goes back to the first galaxy-collision simulations in 1972): the stars trace the gravitational field the black holes create, without the cost and instability of every star pulling on every other.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Turn EPS toward zero and the first close encounter will fling a star to a near-infinite velocity and blow up the run — it is a guardrail, not a decoration.", "G = 1 and these masses set an arbitrary but self-consistent unit system; only ratios matter."] },
        { title: "Allocate once", adding: "init_sim.",
          code: `def init_sim(arch=None):
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
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "The allocate-once pattern one last time. Position, velocity, and acceleration are the three arrays a leapfrog integrator needs; mass is fixed per body.",
          why: "acc is stored rather than recomputed on the fly because the leapfrog integrator (next steps) reuses each step's acceleration across its two half-kicks — the same reason the molecular-dynamics project kept its accelerations around. Four thousand bodies is a comfortable size for direct all-pairs gravity on a GPU; it's 16 million force calculations a frame, which Metal shrugs off.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Everything is 2D (a face-on view of the galactic plane) — simpler to see and cheaper than 3D, and collisions read beautifully from above.", "N is fixed: the two black holes live at known indices inside the same arrays as the stars."] },
        { title: "Build a galaxy", adding: "make_galaxy.",
          code: `def make_galaxy(n, center, bulk_vel, spin=1.0, radius=0.22, seed=0):
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
    return p, v, m`,
          does: "Assembles one galaxy: a black hole at the center (body 0), then a disk of stars scattered at random radii and angles. The crucial part is the velocity — each star gets exactly the CIRCULAR-ORBIT speed for its radius, aimed tangentially (perpendicular to the line to the center), plus the galaxy's overall drift. The sqrt(random) in the radius makes the disk evenly filled rather than bunched at the center.",
          why: "Getting the orbital speed right is the difference between a galaxy and a collapsing (or exploding) mess. Too slow and the stars fall into the black hole; too fast and they fly away. The circular speed is where gravity exactly supplies the centripetal force to hold a circle — and it's computed from the SOFTENED force, not the raw one-over-r-squared, because that's the force the simulation will actually apply. Match the initial velocities to the real force law and the disk spins as a stable, coherent whole; use the textbook unsoftened formula and the inner stars, over-sped for the gentler softened force, would drift outward. Spin lets us make two galaxies rotate the same way or oppositely.",
          see: "Still assembling — a galaxy now exists as numpy arrays, waiting to be loaded.",
          checkpoint: "No red text. make_galaxy returns three arrays of the right shape.",
          recovery: ["The tangential direction (-sin, cos) is the radius vector (cos, sin) rotated 90 degrees — that's what makes the star orbit rather than fall.", "Body 0 is always the black hole; the render and tests rely on it being the heavy one."] },
        { title: "Newton's law and the leapfrog", adding: "the scene loader, the force sum, and the integrator.",
          code: `def apply_seed(scene="single", seed=1):
    """Load a scene: one grand galaxy of stars orbiting a central black hole."""
    p, v, m = make_galaxy(N, [0.0, 0.0], [0.0, 0.0], spin=1.0, radius=0.35, seed=seed)
    pos.from_numpy(p)
    vel.from_numpy(v)
    mass.from_numpy(m)
    compute_acc(-1e9, -1e9, 0.0)
@ti.kernel
def compute_acc(cx: ti.f32, cy: ti.f32, cm: ti.f32):
    """Every body feels the softened pull of every other — direct O(N^2) summation."""
    for i in range(N):
        a = ti.Vector([0.0, 0.0])
        pi = pos[i]
        for j in range(N):
            d = pos[j] - pi
            r2 = d.dot(d) + EPS * EPS
            a += G * mass[j] * d / (r2 * ti.sqrt(r2))
        acc[i] = a
@ti.kernel
def kick(h: ti.f32):
    for i in range(N):
        vel[i] += h * acc[i]
@ti.kernel
def drift(h: ti.f32):
    for i in range(N):
        pos[i] += h * vel[i]
def step(cx=-1e9, cy=-1e9, cm=0.0):
    """One leapfrog (kick-drift-kick) tick — symplectic, so orbits stay stable for ages."""
    kick(0.5 * DT)
    drift(DT)
    compute_acc(cx, cy, cm)
    kick(0.5 * DT)`,
          does: "The engine. compute_acc is Newton's law itself: for every body, sum the softened pull of all N others — a direct O(N-squared) force calculation. Then the leapfrog integrator advances time in three moves: KICK the velocities by a half-step using the current acceleration, DRIFT the positions a full step, recompute the forces, and KICK again by the second half-step. apply_seed loads a single galaxy and primes the accelerations. (compute_acc takes cursor arguments it ignores for now — chapter 3 will use them.)",
          why: "Why this specific kick-drift-kick dance instead of the obvious 'move by velocity, change velocity by force'? Because leapfrog is SYMPLECTIC — it respects the geometric structure of Hamiltonian mechanics, and as a result its energy error doesn't accumulate. A naive integrator would let every orbit slowly spiral in or out as rounding errors pile up, and after a few thousand steps the galaxy would be visibly wrong. Leapfrog's orbits stay put essentially forever. It's the same integrator family as the molecular-dynamics and solar-system projects, and it's what real astrophysicists use to run simulations for billions of years of model time.",
          see: "Assembling — the loop that runs it is next.",
          checkpoint: "No red text. The force sum and integrator compile.",
          recovery: ["The half-kick / full-drift / half-kick split is what makes it symplectic — a single full kick before or after the drift would NOT conserve energy the same way.", "compute_acc reads every pos before writing any acc, so there's no order dependence — a clean parallel all-pairs sum."] },
        { title: "A galaxy, turning", adding: "the diagnostics, the star render, and the main loop.",
          code: `def total_energy():
    """Pure numpy: kinetic + gravitational potential energy of the whole system."""
    p = pos.to_numpy()
    v = vel.to_numpy()
    m = mass.to_numpy()
    ke = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
    d = p[:, None, :] - p[None, :, :]
    r = np.sqrt(np.sum(d * d, axis=2) + EPS * EPS)
    iu = np.triu_indices(len(p), k=1)
    pe = -G * float(np.sum(m[iu[0]] * m[iu[1]] / r[iu]))
    return ke + pe
def bound_fraction(cx=0.0, cy=0.0, extent=VIEW):
    """Pure numpy: fraction of bodies still within \`extent\` of a center — how much survived."""
    p = pos.to_numpy()
    return float((np.abs(p - np.array([cx, cy])).max(axis=1) < extent).mean())
@ti.kernel
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
        pixels[i, j] = ti.min(pixels[i, j], 1.0)           # dense cores saturate to white
def main():
    init_sim()
    apply_seed("single")
    gui = ti.GUI("Universe Sandbox — taichi-academy", res=RES, background_color=0x05050A)
    while gui.running:
        cx, cy, cm = -1e9, -1e9, 0.0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "s":
                apply_seed("single", np.random.randint(1_000_000))
        step(cx, cy, cm)
        render()
        gui.set_image(pixels)
        gui.text("drag: black-hole cursor   [c] collide   [s] single galaxy", (0.02, 0.98), color=0xFFFFFF)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "total_energy sums kinetic plus gravitational potential — the number a symplectic integrator should hold steady. bound_fraction measures how many bodies are still in frame. render draws each star with additive glow (so crowded regions blaze brighter, giving the galaxy a luminous core) and colors it by speed — fast inner stars blue-white, slow outer stars orange — with the black hole a fat bright dot. main runs a single galaxy, with [s] to reseed.",
          why: "The additive rendering is what makes a swarm of dots read as a galaxy: where many stars pile into one pixel, their light sums and the core glows, exactly as a real galaxy's brightness traces its stellar density. And total_energy is more than a readout — it's how you VERIFY the physics: run the galaxy for hundreds of steps and the energy barely budges, proving the leapfrog is doing its job. A galaxy that quietly spins in place, conserving its energy, is a surprisingly hard thing to get right, and it's the foundation the collision is built on.",
          see: "A luminous spiral galaxy: a bright golden-white core where the stars crowd around the black hole, fading to a sparse blue-white disk of stars wheeling around it. Leave it running and it just turns, serene and stable, holding its shape indefinitely. Press [s] for a fresh random galaxy.",
          checkpoint: "A stable, rotating, energy-conserving galaxy. Chapter 1 complete.",
          recovery: ["If the galaxy visibly expands or contracts over time, the integrator isn't conserving energy — check the half-kick / drift / half-kick order in step.", "If stars stream off immediately, the circular velocities are wrong: confirm make_galaxy uses the softened-force speed and that apply_seed primed compute_acc."] }
      ]
    },
    {
      id: 2, title: "Galaxies collide",
      build: "a second scene that launches two galaxies onto a collision course, with the black holes at their hearts leading the dance.",
      beat: "Two galaxies sweep toward each other, their disks tearing into streamers as the black holes swing past and merge.",
      steps: [
        { title: "Two galaxies on a collision course", adding: "the collide scene.",
          code: `def apply_seed(scene="collide", seed=1):
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
    compute_acc(-1e9, -1e9, 0.0)`,
          does: "apply_seed grows a second scene. 'collide' builds two half-size galaxies, offset left and right, each drifting toward the other with a small vertical miss-distance so they pass close rather than dead-center. Their stars and black holes are stacked into the same arrays with numpy, and the force sum treats them as one system of 4000 mutually-gravitating bodies.",
          why: "Nothing about the physics changes — it's the same compute_acc summing over the same N bodies. All that's new is the initial conditions: two spinning disks aimed at each other. That's the beauty of a general N-body engine — 'a galaxy collision' isn't a special mode with special code, it's just a different starting arrangement of the exact same law. The small vertical offset (the impact parameter) matters enormously: a glancing pass raises graceful tidal tails, while a head-on hit just scrambles everything.",
          see: "Still assembling — chapter needs main to load the new scene, next step.",
          checkpoint: "No red text. apply_seed('collide') builds two galaxies.",
          recovery: ["The two galaxies share one set of arrays, so from compute_acc's view there's just one 4000-body system — the 'two galaxies' exist only in how they were arranged.", "Change the offsets and drift speeds to explore flybys versus mergers — it's a genuinely chaotic, sensitive system."] },
        { title: "Loose the collision", adding: "the collide scene in main.",
          code: `def main():
    init_sim()
    apply_seed("collide")
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "c":
                apply_seed("collide", np.random.randint(1_000_000))
            elif e.key == "s":
                apply_seed("single", np.random.randint(1_000_000))`,
          does: "main now opens on the collision scene, and the key handler gains [c] to reroll a fresh collision alongside [s] for a single galaxy.",
          why: "This is the payoff of the whole project. Two galaxies, each perfectly stable on its own, are set on an approach — and their mutual gravity does the rest. As they pass, each galaxy's near side is pulled harder than its far side (a tidal force), stretching the disks into long streamers of stars — the same 'tidal tails' seen in real interacting galaxies like the Antennae and the Mice. The black holes, being the heavy anchors, sink toward each other and swing around in a gravitational dance, sometimes merging. Every bit of that structure emerges from summing one-over-r-squared over pairs of points; nobody drew the tails.",
          see: "Two galaxies wheel toward each other. At closest passage their disks distort, flinging out curving streamers of stars into the dark while the two bright black-hole cores whip past one another — and, depending on the roll, fall back and spiral together. Press [c] to fling a fresh pair; every collision plays out differently.",
          checkpoint: "Colliding galaxies with tidal streamers. Chapter 2 complete.",
          recovery: ["Galaxy collisions are chaotic — some rolls merge, some fly apart, some make dramatic tails; that variety is real physics, not a bug.", "If a collision scatters everything off-screen, the galaxies passed too close and too fast — that's a genuine (if messy) outcome; press [c] for another."] }
      ]
    },
    {
      id: 3, title: "The black-hole cursor",
      build: "an interactive heavy mass you drag through the scene with the mouse — reach in and stir the cosmos by hand.",
      beat: "Drag the cursor through a galaxy and it warps around your black hole, streaming stars into your wake.",
      steps: [
        { title: "A mass in the force sum", adding: "the cursor term in gravity.",
          code: `@ti.kernel
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
        acc[i] = a`,
          does: "compute_acc grows one more term: after summing the pull of all N bodies, each star also feels a phantom mass cm sitting at the cursor position (cx, cy) — softened just like everything else. When cm is zero (no mouse held), the term vanishes and nothing changes.",
          why: "It's the same law applied to one more body — a body that happens to teleport to wherever you point. This is the cleanest possible way to make an N-body simulation interactive: don't script special 'stirring' behaviour, just add a gravitating mass to the sum and let Newton do the rest. Everything the cursor does — bending orbits, tearing off streams, capturing stars into a little entourage — is emergent, because gravity is gravity whether the mass is a black hole or your mouse.",
          see: "Assembling — the mouse needs wiring, one last step.",
          checkpoint: "No red text. compute_acc now accepts a cursor mass.",
          recovery: ["The cursor term reuses the exact softened-force expression, so it's stable at close range like every other body.", "With cm defaulting to 0 in step, existing scenes behave identically until the mouse supplies a real mass."] },
        { title: "Stir the cosmos", adding: "the mouse control.",
          code: `        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            cx, cy = (mx - 0.5) * 2.0 * VIEW, (my - 0.5) * 2.0 * VIEW
            cm = 3.0 * BH_MASS                             # a heavy black hole under the cursor`,
          does: "While the left mouse button is held, the cursor's screen position is converted into world coordinates and a heavy black hole (three times a galaxy's own) is placed there, passed into step and thus into the force sum. Release, and it's gone.",
          why: "This closes the sandbox — and the whole curriculum. You're no longer watching a simulation; you're reaching into a gravitational field with your hand. Drag your black hole through a serene galaxy and watch it warp: stars swing toward you, stretch into a wake, some torn loose entirely, some captured into orbit around the cursor. It's the same tidal physics as the galaxy collision, now under your direct control — the culmination of thirty projects that each turned a handful of local rules into a world. From one reaction-diffusion equation to a universe you can stir by hand, the lesson was always the same: simple rules, patiently iterated, become everything. Project 30, Arc 7, and the whole curriculum — complete.",
          see: "Hold the mouse over a drifting galaxy and drag: the stars bend toward your cursor, streaming into a bright tail behind your motion, a knot of captured stars trailing you like a comet's coma. Sweep through the two-galaxy collision and you can fling the whole scene into new chaos. A universe, in a window, at your fingertips.",
          checkpoint: "An interactive gravitational sandbox — galaxies, black holes, and a cursor you stir them with. Project 30 and the taichi-academy curriculum complete.",
          recovery: ["The (mx - 0.5) * 2 * VIEW conversion maps the 0..1 cursor position into the same world coordinates the render uses, so your black hole appears exactly under the pointer.", "3x BH_MASS is heavy enough to visibly disrupt a galaxy on a pass; lower it for a gentle stir, raise it to shred everything."] }
      ]
    }
  ]
};

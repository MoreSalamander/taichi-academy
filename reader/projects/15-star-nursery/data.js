// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["15-star-nursery"] = {
  project: "15-star-nursery",
  title: "Star Nursery",
  pitch: "Gas that pulls itself into filaments, ignites where it's densest, then blows itself apart — a full stellar lifecycle in one loop.",
  tier: "hard",
  language: "Python",
  file: "star_nursery.py",
  chapters: [
    {
      id: 1, title: "A cold cloud",
      build: "gas particles, star slots that start empty, and a nebula render — nothing moves yet.",
      beat: "Wisps of dark violet gas hang motionless: a molecular cloud before anything happens.",
      steps: [
        { title: "A lifecycle, not just a picture", adding: "the docstring and imports.",
          code: `"""Star Nursery: a molecular cloud collapses under its own gravity and ignites into stars."""
import numpy as np
import taichi as ti`,
          does: "This project simulates a STORY with stages: cold gas drifts, its own gravity gathers it into dense filaments, the densest knots cross a threshold and IGNITE into stars, and those stars' radiation pushes the leftover gas away — carving glowing cavities and quenching further collapse nearby. Every stage feeds the next.",
          why: "It's also this curriculum's first PARTICLE STATE MACHINE: a particle isn't just a position, it has a life stage (gas, or consumed-into-a-star) that changes permanently at runtime. Projects until now moved particles; this one lets them BECOME something else.",
          see: "Runs clean.",
          checkpoint: "python3 star_nursery.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "Gas, and empty cradles", adding: "population dials and all the particle/star fields.",
          code: `RES = 512
GRID = 128
N_GAS = 40000
MAX_STARS = 400
pos = None
vel = None
alive = None
star_pos = None
star_age = None
n_stars = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, alive, star_pos, star_age, n_stars, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N_GAS)
    vel = ti.Vector.field(2, ti.f32, shape=N_GAS)
    alive = ti.field(ti.i32, shape=N_GAS)
    star_pos = ti.Vector.field(2, ti.f32, shape=MAX_STARS)
    star_age = ti.field(ti.f32, shape=MAX_STARS)
    n_stars = ti.field(ti.i32, shape=())
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "Two populations, allocated up front: 40,000 gas particles (pos/vel/alive), and 400 star SLOTS (star_pos/star_age) that all start unused — n_stars counts how many have been claimed, exactly like project 07's ring-buffer cursor. alive is each gas particle's life-stage flag: 1 means gas, 0 means it became a star.",
          why: "Pre-allocating the maximum stars and counting upward is the only shape that works on the GPU — you can't append to a field at runtime (and on Metal you can't even resize one). Every 'dynamic' population in this curriculum is really a fixed pool plus a counter; this project makes that idiom explicit.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["n_stars is a 0-dimensional field (shape=()) — one global integer living on the GPU, read as n_stars[None]."] },
        { title: "See the cloud", adding: "the blob seeder, the reset, a fading nebula render, and the loop.",
          code: `GAS_COLOR = (0.10, 0.08, 0.20)
CANVAS_FADE = 0.85
def seed_gas(n, rng_seed=0, blobs=4):
    """Pure numpy: a few overlapping gaussian gas clouds."""
    rng = np.random.default_rng(rng_seed)
    centers = rng.uniform(0.25, 0.75, size=(blobs, 2))
    which = rng.integers(0, blobs, n)
    p = centers[which] + rng.normal(0, 0.09, size=(n, 2))
    return np.clip(p, 0.02, 0.98).astype(np.float32)
def apply_seed(rng_seed=0):
    pos.from_numpy(seed_gas(N_GAS, rng_seed))
    vel.fill(0.0)
    alive.fill(1)
    n_stars[None] = 0
    pixels.fill(0.0)
@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for p in pos:
        if alive[p] == 1:
            xi = ti.cast(pos[p][0] * RES, ti.i32)
            yi = ti.cast(pos[p][1] * RES, ti.i32)
            if 0 <= xi < RES and 0 <= yi < RES:
                pixels[xi, yi] += ti.Vector([GAS_COLOR[0], GAS_COLOR[1], GAS_COLOR[2]])
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
    gui = ti.GUI("Star Nursery — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "seed_gas drops each particle into one of four randomly-placed gaussian blobs — a lumpy, overlapping cloud complex, not a uniform fog. The render is the fade-splat-clamp trio from projects 07 and 14, tinted a deep violet so accumulating gas reads as nebula.",
          why: "The initial LUMPINESS is load-bearing physics, not decoration: gravity amplifies whatever density contrast already exists, so a perfectly uniform cloud would collapse nowhere (every direction pulls equally). Real star formation needs seeds of unevenness; ours come free with the gaussian blobs.",
          see: "Four soft violet cloud-wisps overlapping in the dark — a molecular cloud, holding still.",
          checkpoint: "A static nebula. Beat 1.",
          recovery: ["The alive check in render already matters — stars will stop being drawn as gas the moment they ignite, chapters from now, with no render change needed."] }
      ]
    },
    {
      id: 2, title: "The cloud falls inward",
      build: "a density grid, a blur, and gravity that follows the density gradient — self-collapse.",
      beat: "The wisps stop holding still: gas streams into knots and filaments, like the cosmic web in miniature.",
      steps: [
        { title: "Where is the mass?", adding: "the density grid, its twin, the deposit pass, and the blur.",
          code: `density = None
density_blur = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, alive, star_pos, star_age, n_stars, density, density_blur, pixels
    density = ti.field(ti.f32, shape=(GRID, GRID))
    density_blur = ti.field(ti.f32, shape=(GRID, GRID))
@ti.kernel
def clear_density():
    for i, j in density:
        density[i, j] = 0.0
@ti.kernel
def deposit():
    for p in pos:
        if alive[p] == 1:
            gi = ti.cast(pos[p][0] * GRID, ti.i32)
            gj = ti.cast(pos[p][1] * GRID, ti.i32)
            if 0 <= gi < GRID and 0 <= gj < GRID:
                density[gi, gj] += 1.0
@ti.kernel
def blur():
    for i, j in density_blur:
        acc = 0.0
        cnt = 0.0
        for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID and 0 <= nj < GRID:
                acc += density[ni, nj]
                cnt += 1.0
        density_blur[i, j] = acc / cnt`,
          does: "deposit is MPM's P2G idea at its simplest: every gas particle bumps its grid cell's counter (an implicit atomic — thousands of particles share cells). blur averages each cell with its 5x5 neighborhood, turning the spiky per-cell counts into a smooth density landscape.",
          why: "The blur isn't cosmetic — it IS the physics approximation. Real gravity reaches across all space; a smoothed density field is a cheap stand-in where each particle feels the mass within a couple of cells' reach. It's the same 'coarse field instead of N-squared pairs' move as project 06's spatial hash, but for a force instead of a neighbor search.",
          see: "Runs clean; nothing consumes density yet.",
          checkpoint: "No red text.",
          recovery: ["Two grids: density (raw, cleared and re-deposited every tick) and density_blur (the smoothed copy everything downstream reads)."] },
        { title: "Fall toward the crowd", adding: "the gravity kernel — climb the density gradient.",
          code: `GRAVITY_PULL = 120.0
DT = 0.004
@ti.kernel
def gravity():
    for p in pos:
        if alive[p] == 1:
            gi = ti.min(ti.max(ti.cast(pos[p][0] * GRID, ti.i32), 1), GRID - 2)
            gj = ti.min(ti.max(ti.cast(pos[p][1] * GRID, ti.i32), 1), GRID - 2)
            gx = (density_blur[gi + 1, gj] - density_blur[gi - 1, gj]) * 0.5
            gy = (density_blur[gi, gj + 1] - density_blur[gi, gj - 1]) * 0.5
            vel[p] += DT * GRAVITY_PULL * ti.Vector([gx, gy]) / GRID`,
          does: "Each particle samples the density SLOPE at its own cell (central differences — project 05's hillshade gradient, repurposed) and accelerates uphill, toward where the mass already is. That's gravity's essence: mass attracts mass.",
          why: "This creates a feedback loop with real teeth: a slightly dense region pulls in gas, which makes it denser, which pulls harder. Astronomers call the runaway version Jeans instability — it's why clouds fragment into filaments and knots rather than contracting into one big ball, and you're about to watch it happen.",
          see: "Runs clean; gravity isn't in the tick yet.",
          checkpoint: "No red text.",
          recovery: ["Indices clamp to [1, GRID-2], not [0, GRID-1] — central differences need both neighbors to exist."] },
        { title: "Watch it collapse", adding: "damped integration and the full collapse tick.",
          code: `DAMPING = 0.96
@ti.kernel
def integrate():
    for p in pos:
        if alive[p] == 1:
            vel[p] *= DAMPING
            newp = pos[p] + DT * vel[p]
            for a in ti.static(range(2)):
                if newp[a] < 0.01:
                    newp[a] = 0.01
                    vel[p][a] *= -0.5
                if newp[a] > 0.99:
                    newp[a] = 0.99
                    vel[p][a] *= -0.5
            pos[p] = newp
def step():
    clear_density()
    deposit()
    blur()
    gravity()
    integrate()
    render()
    clamp_pixels()`,
          does: "Standard damped integration with wall bounces (project 06's integrate, nearly line for line). The tick now runs the full collapse pipeline: measure the mass, smooth it, fall toward it, move.",
          why: "DAMPING stands in for the gas pressure and radiative cooling this model doesn't simulate — without it, particles would slingshot through the dense knots and oscillate forever instead of settling INTO them. One multiply, doing the job of a thermodynamics engine.",
          see: "The cloud comes alive: wisps stream toward each other, pile into bright knots, and stretch into filaments between them — a miniature cosmic web assembling itself from nothing but 'mass attracts mass'.",
          checkpoint: "Visible gravitational collapse. Beat 2.",
          recovery: ["Order in step matters: density must be rebuilt (clear/deposit/blur) BEFORE gravity reads it, every single tick."] }
      ]
    },
    {
      id: 3, title: "Ignition",
      build: "the density-triggered birth of stars, and their glow in the render.",
      beat: "Deep inside the densest knots, the first stars switch on.",
      steps: [
        { title: "The threshold of birth", adding: "ignition dials and the kernel that turns gas into stars.",
          code: `IGNITE_DENSITY = 22.0
IGNITE_PROB = 0.001
@ti.kernel
def ignite():
    for p in pos:
        if alive[p] == 1 and n_stars[None] < MAX_STARS:
            gi = ti.min(ti.max(ti.cast(pos[p][0] * GRID, ti.i32), 0), GRID - 1)
            gj = ti.min(ti.max(ti.cast(pos[p][1] * GRID, ti.i32), 0), GRID - 1)
            if density_blur[gi, gj] > IGNITE_DENSITY:
                if ti.random() < IGNITE_PROB:
                    s = ti.atomic_add(n_stars[None], 1)
                    if s < MAX_STARS:
                        star_pos[s] = pos[p]
                        star_age[s] = 0.0
                        alive[p] = 0
@ti.kernel
def age_stars(dt: ti.f32):
    for s in range(n_stars[None]):
        star_age[s] += dt`,
          does: "A gas particle sitting in dense-enough gas has a small chance per tick of igniting: it claims the next star slot (ti.atomic_add returns the old count — the same slot-claiming trick as project 06's scatter), copies its position in, and flips its own alive flag to 0. It has permanently changed species.",
          why: "IGNITE_DENSITY is the whole story's hinge, and it was mistuned once during this project's development: set near the cloud's STARTING density, stars ignited everywhere instantly and their radiation blasted the cloud apart before any collapse could happen — a wall of stars, no nursery. The threshold has to be reachable only by collapse (roughly 4x the initial blob density), so that gravity must do its work first. Thresholds relative to initial conditions, not absolute feel — a recurring procgen lesson.",
          see: "Runs clean; not in the tick yet.",
          checkpoint: "No red text.",
          recovery: ["Two guards on the slot claim: the cheap n_stars check before rolling the dice, and the strict s < MAX_STARS after — atomics can race a few counts past the limit; the second check is the one that holds."] },
        { title: "Let there be light", adding: "star glow in the render and ignition in the tick.",
          code: `STAR_COLOR = (1.0, 0.9, 0.7)
@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] *= CANVAS_FADE

    for p in pos:
        if alive[p] == 1:
            xi = ti.cast(pos[p][0] * RES, ti.i32)
            yi = ti.cast(pos[p][1] * RES, ti.i32)
            if 0 <= xi < RES and 0 <= yi < RES:
                pixels[xi, yi] += ti.Vector([GAS_COLOR[0], GAS_COLOR[1], GAS_COLOR[2]])

    for s in range(n_stars[None]):
        cx = star_pos[s][0] * RES
        cy = star_pos[s][1] * RES
        glow = ti.min(star_age[s] * 2.0, 1.0)
        for di, dj in ti.ndrange((-3, 4), (-3, 4)):
            xi = ti.cast(cx, ti.i32) + di
            yi = ti.cast(cy, ti.i32) + dj
            if 0 <= xi < RES and 0 <= yi < RES:
                w = ti.exp(-(di * di + dj * dj) / 4.0)
                pixels[xi, yi] += glow * w * ti.Vector([STAR_COLOR[0], STAR_COLOR[1], STAR_COLOR[2]])
def step():
    clear_density()
    deposit()
    blur()
    gravity()
    integrate()
    ignite()
    age_stars(DT)
    render()
    clamp_pixels()`,
          does: "Each star paints a 7x7 gaussian halo of warm white — a real glow, not a single pixel — scaled by ti.min(star_age * 2, 1): newborn stars FADE IN over half a second instead of popping into existence. Note the halo loop is a plain ti.ndrange, not ti.static — 49 unrolled iterations per star would blow up compile time for no run-time win.",
          why: "The age-driven fade-in is one float per star doing narrative work: your eye catches each birth as a soft bloom in the middle of a bright knot, exactly where the density said it should happen. Watching WHERE the stars appear is watching the collapse's map of its own densest places.",
          see: "Within seconds of a knot forming, a warm glow blooms inside it — then another, and another. The nursery is naming its children.",
          checkpoint: "Stars ignite inside dense knots. Beat 3.",
          recovery: ["ignite comes AFTER integrate in the tick (positions are settled) and BEFORE render (newborns glow the same frame)."] }
      ]
    },
    {
      id: 4, title: "The stars push back",
      build: "radiation pressure from every star, closing the feedback loop, plus the HUD.",
      beat: "Newborn stars blow glowing cavities in the cloud that birthed them.",
      steps: [
        { title: "Feedback", adding: "radiation dials and the pushback force, wired into the tick.",
          code: `RADIATION = 250.0
RADIATION_R = 0.05
@ti.kernel
def radiation():
    for p in pos:
        if alive[p] == 1:
            f = ti.Vector([0.0, 0.0])
            for s in range(n_stars[None]):
                d = pos[p] - star_pos[s]
                r2 = d.dot(d)
                if r2 < RADIATION_R * RADIATION_R:
                    r = ti.sqrt(r2) + 1e-4
                    f += RADIATION * (1.0 - r / RADIATION_R) * d / r
            vel[p] += DT * f
def step():
    clear_density()
    deposit()
    blur()
    gravity()
    radiation()
    integrate()
    ignite()
    age_stars(DT)
    render()
    clamp_pixels()`,
          does: "Every gas particle checks every star (a brute-force pair loop — fine here because stars max out at 400, unlike project 06's 30,000-vs-30,000 problem) and takes a push AWAY from each one it's near, strongest at the star and fading to nothing at RADIATION_R.",
          why: "This closes the loop that makes the system self-REGULATING: collapse ignites stars, stars disperse the gas, dispersed gas can't collapse — so star formation chokes off its own fuel supply locally. Real nurseries do exactly this (astronomers call it stellar feedback), and it's why star formation is slow and patchy instead of one giant flash.",
          see: "Each glowing knot starts hollowing out: stars sit in growing dark cavities rimmed by bright compressed gas shells — the classic 'blown bubble' look of real emission nebulae.",
          checkpoint: "Stars carve cavities. No red text.",
          recovery: ["radiation slots between gravity and integrate — both forces accumulate into vel before the single position update."] },
        { title: "Count the children", adding: "the reseed key and the census HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text(f"stars born: {n_stars[None]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new cloud", (0.02, 0.94), color=0xAAAAAA)`,
          does: "The HUD reads the star counter live off the GPU; R rolls a fresh cloud (apply_seed already resets stars, gas, and canvas together).",
          why: "Let one cloud run for a few minutes and watch the whole arc: collapse, ignition burst, cavity-blowing, and finally a quiet field of stars with the gas exhausted or expelled — a nursery aging into a cluster. That END state is real too: open star clusters like the Pleiades are exactly this, a nursery whose gas is gone.",
          see: "The census climbs fast during the first ignition burst, then slows as feedback starves the collapse — an S-curve you can watch in a number.",
          checkpoint: "A full stellar lifecycle, with a birth counter. Final beat — project 15 complete.",
          recovery: ["Same reseed idiom as always — and apply_seed's full reset (chapter 1) is why one keypress cleanly restarts an entire lifecycle."] }
      ]
    }
  ]
};

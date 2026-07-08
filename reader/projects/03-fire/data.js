// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["03-fire"] = {
  project: "03-fire",
  title: "Fire & Smoke",
  pitch: "Heat rises, flames lick, smoke shrouds the glow — your fluid solver becomes a bonfire.",
  tier: "medium",
  language: "Python",
  file: "fire.py",
  chapters: [
    {
      id: 1, title: "A glowing ember",
      build: "a temperature field rendered in true fire colors — using algebra instead of a palette table.",
      beat: "An ember blob glows in real fire colors.",
      steps: [
        { title: "Load the GPU toolkit", adding: "the docstring and the Taichi import.",
          code: `"""Fire and smoke: heat rises, flames lick, smoke shrouds the glow."""
import taichi as ti`,
          does: "Project 03 opens like the other two. This one is a remix: you'll rebuild project 02's fluid core from muscle memory, then teach it thermodynamics.",
          why: "Fire IS a fluid — hot air moving under buoyancy. Everything you typed last project is about to pay a second dividend.",
          see: "Runs clean, nothing visible.",
          checkpoint: "python3 fire.py returns silently.",
          recovery: ["No module named taichi — activate the venv first."] },
        { title: "The temperature field", adding: "the grid size, placeholders, and init_sim allocating temperature and pixels.",
          code: `N = 512
temp = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global temp, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    temp = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "One scalar per cell: how hot. 0 is cold black air; around 1 is roaring flame (we'll let the source push a little past it, capped at 1.5).",
          why: "In project 01 the star field was chemicals; in 02, ink and motion; here the protagonist is heat. Everything else — lift, glow, smoke — derives from this one number.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["temp is a plain ti.field — one number per cell, no Vector."] },
        { title: "One hot ember", adding: "numpy (above import taichi) and a pure-numpy seed: a single Gaussian blob, low in the box.",
          code: `import numpy as np
def seed_pattern(n, rng_seed=0):
    """Pure numpy: one hot ember blob low in the box."""
    rng = np.random.default_rng(rng_seed)
    cx = n // 2 + int(rng.integers(-n // 8, n // 8))
    cy = n // 4
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sigma = n / 16.0
    t0 = np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (sigma * sigma))
    return t0.astype(np.float32)
def apply_seed(t0):
    temp.from_numpy(t0)`,
          does: "The meshgrid-Gaussian recipe from project 02's ink blobs, reused for one hot spot: center-ish horizontally (a seeded random nudge), a quarter of the way up. apply_seed uploads it.",
          why: "Low in the box on purpose — this ember has somewhere to go once heat learns to rise.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["cy = n // 4 puts the blob low — remember the window's origin is bottom-left, so small j is DOWN.", "indexing=\"ij\" again — grid convention."] },
        { title: "Fire from algebra", adding: "the render kernel — no palette table this time — plus the standard main loop.",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        t = ti.math.clamp(temp[i, j], 0.0, 1.0)
        pixels[i, j] = ti.math.clamp(ti.Vector([1.6 * t, 1.2 * t * t, t * t * t]), 0.0, 1.0)
def main():
    init_sim()
    apply_seed(seed_pattern(N))
    gui = ti.GUI("Fire & Smoke — taichi-academy", res=(N, N))
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "The fire ramp is three powers of temperature: red = 1.6t (fires up immediately), green = 1.2t² (joins later), blue = t³ (only at white heat). Low t glows deep red, mid t orange, high t yellow-white — a blackbody curve faked in one line.",
          why: "Project 01 taught palettes as lookup tables; this is the other way — color as a FUNCTION. Powers of t are free, need no memory, and you'll reuse the trick for glow effects all series.",
          see: "A deep-red ember with an orange-yellow heart, low in a black box. Frozen — it doesn't know physics yet.",
          checkpoint: "A glowing ember blob. Beat 1.",
          recovery: ["Two clamps: t first (source can exceed 1), then the color vector (1.6t exceeds 1).", "Green is t*t and blue t*t*t — swap them and fire turns sickly green."] }
      ]
    },
    {
      id: 2, title: "Heat rises",
      build: "advection from muscle memory, plus the one new force that makes fire fire: buoyancy.",
      beat: "The ember lifts off and mushrooms upward.",
      steps: [
        { title: "Motion fields", adding: "velocity and the temperature double-buffer (placeholders, global line, three field lines).",
          code: `vel = None
vel_next = None
temp = None
temp_next = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, pixels
    vel = ti.Vector.field(2, ti.f32, shape=(N, N))
    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))
    temp_next = ti.field(ti.f32, shape=(N, N))`,
          does: "The same cast as project 02: a 2-vector flow field with its twin, and a _next twin for temperature so advection can double-buffer.",
          why: "Fire needs the full fluid machinery — heat is CARRIED by air that moves because heat made it move. That loop starts here.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The three-edit ritual from project 02: placeholders, global line, field lines (vel pair above temp, temp_next after temp)."] },
        { title: "Sampler and bilerp — second time's the charm", adding: "the wrap-safe sampler and bilinear interpolation, retyped from project 02.",
          code: `@ti.func
def sample(f: ti.template(), i, j):
    return f[((i % N) + N) % N, ((j % N) + N) % N]
@ti.func
def bilerp(f: ti.template(), x, y):
    x0 = int(ti.floor(x))
    y0 = int(ti.floor(y))
    fx = x - x0
    fy = y - y0
    a = sample(f, x0, y0)
    b = sample(f, x0 + 1, y0)
    c = sample(f, x0, y0 + 1)
    d = sample(f, x0 + 1, y0 + 1)
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy`,
          does: "Identical to project 02 — wrap any index onto the grid; read between cells by blending the four neighbors.",
          why: "Retyping isn't busywork: this pair is the load-bearing wall of every grid sim you'll build, and the second typing is where it moves from 'I followed it' to 'I know it'. Type it without peeking first, then check.",
          see: "Runs clean.",
          checkpoint: "No red text — and be honest: did you peek?",
          recovery: ["The negative-safe wrap is ((i % N) + N) % N.", "b and c differ only in which coordinate gets +1."] },
        { title: "Advection, verbatim", adding: "the time step, advect, copy-back for temp and vel, and the first step().",
          code: `DT = 1.0
@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)
@ti.kernel
def copy_back():
    for i, j in temp:
        temp[i, j] = temp_next[i, j]
        vel[i, j] = vel_next[i, j]
def step():
    advect(temp, temp_next)
    advect(vel, vel_next)
    copy_back()`,
          does: "Semi-Lagrangian advection again: every cell backtracks along its arrow and asks what was there. Heat and motion both ride the flow; both write into twins; copy_back adopts.",
          why: "In project 02 the passenger was ink; now it's temperature. Same truck, different cargo — that's why advect was written generic.",
          see: "Runs clean; velocity is still all zeros, so nothing moves yet.",
          checkpoint: "No red text.",
          recovery: ["Backward along the arrow: i MINUS the velocity.", "step advects temp first, then vel, then one copy_back."] },
        { title: "Buoyancy — the soul of fire", adding: "the lift constant and the one genuinely new kernel of this chapter, joining step.",
          code: `BUOYANCY = 0.05
@ti.kernel
def apply_buoyancy():
    for i, j in vel:
        vel[i, j][1] += DT * BUOYANCY * temp[i, j]
def step():
    advect(temp, temp_next)
    advect(vel, vel_next)
    copy_back()
    apply_buoyancy()`,
          does: "Every cell's upward velocity grows in proportion to how hot it is. Cold cells feel nothing; the ember's heart gets a steady upward shove every tick.",
          why: "This two-line force is the entire engine of fire, thermals, and lava lamps: hot air is lighter, so it rises — and as it rises it carries its own heat with it (advection), which keeps it rising. Keep BUOYANCY small: it compounds every frame, and the sim has no brakes yet.",
          see: "Runs clean; one line in main and it comes alive.",
          checkpoint: "No red text.",
          recovery: ["Index [1] is the vertical component — [0] would blow the fire sideways.", "It's += — buoyancy accumulates into the existing motion."] },
        { title: "Liftoff", adding: "the tick in main's loop (after the event block, before render()).",
          code: `        step()`,
          does: "Advect, then lift, every frame.",
          why: "Watch the feedback loop you just closed: heat lifts air, air carries heat higher, higher heat lifts more air.",
          see: "The ember rises — and as it climbs it flattens, spreads, and rolls into a mushroom cap eating its own edges. That shape is the advection-buoyancy feedback, live.",
          checkpoint: "The blob rises and mushrooms. Beat 2.",
          recovery: ["step() inside the while loop, before render().", "Blob rockets off instantly — BUOYANCY should be 0.05."] }
      ]
    },
    {
      id: 3, title: "The bonfire",
      build: "a flickering heat source at the hearth, and cooling so the fire has a top.",
      beat: "An everlasting campfire burns at the hearth.",
      steps: [
        { title: "The hearth", adding: "the source size and a kernel that pumps flickering heat in at the bottom (add after apply_buoyancy).",
          code: `SOURCE_RADIUS = 40.0
@ti.kernel
def burn_source(t: ti.f32):
    for i, j in temp:
        dx = i - N / 2
        dy = j - 12.0
        flick = 1.0 + 0.35 * ti.sin(0.31 * t + 0.05 * i)
        w = ti.exp(-(dx * dx + dy * dy) / (SOURCE_RADIUS * SOURCE_RADIUS)) * flick
        temp[i, j] = ti.min(temp[i, j] + 0.8 * w, 1.5)`,
          does: "A wide Gaussian hot spot centered at the bottom (j near 12). The flicker line modulates it with a sine that drifts over time (0.31t) AND varies across space (0.05i) — so the hearth breathes instead of pumping evenly. ti.min caps heat at 1.5: a fire has a hottest.",
          why: "Two lessons hide here. Deterministic flicker: sin(t) looks random enough but replays identically — the tests depend on that, and ti.random() would break them. And saturation: sources must cap, or values climb forever.",
          see: "Runs clean; nothing calls it yet.",
          checkpoint: "No red text.",
          recovery: ["The flicker multiplies w — it modulates the whole source.", "ti.min(..., 1.5), not +=-and-hope — the cap is the point."] },
        { title: "Light it", adding: "two state lines in main (after the gui line), the space-bar toggle (replace the event block), the source call (before step()), and a frame counter tick (after step()).",
          code: `    fire_on = True
    frame = 0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on
        if fire_on:
            burn_source(float(frame))
        frame += 1`,
          does: "A frame counter feeds the flicker's clock, and space toggles the bonfire. The source burns right before the physics step so fresh heat immediately gets advected and lifted.",
          why: "frame is the sim's heartbeat made visible — you'll pass counters into kernels whenever anything needs to change over time deterministically.",
          see: "Launch: a column of flame boils up from the hearth and keeps going — but notice the whole box slowly fills with heat haze. No exit yet; next step fixes that.",
          checkpoint: "A rising flame column; space snuffs and relights it.",
          recovery: ["Four homes: two state lines after gui, the toggle in events, the if fire_on block before step(), frame += 1 after step().", "The kernel takes a float: burn_source(float(frame))."] },
        { title: "Every fire has a top", adding: "two decay dials, the cooling kernel, and its place at the end of step (replace step).",
          code: `COOLING = 0.985
VEL_DECAY = 0.99
@ti.kernel
def cool():
    for i, j in temp:
        temp[i, j] *= COOLING
        vel[i, j] *= VEL_DECAY
def step():
    advect(temp, temp_next)
    advect(vel, vel_next)
    copy_back()
    apply_buoyancy()
    cool()`,
          does: "Heat loses 1.5% per tick, motion 1%. Rising air cools as it climbs, so its buoyancy fades, so it stops rising — flames get a natural height where injection and cooling balance.",
          why: "This is the brake pedal for chapter 2's feedback loop. Without it the earlier build literally runs away (velocities in the hundreds — we measured). Sources need sinks: burn in, cool out, equilibrium between. That pairing is the deepest pattern in this whole project.",
          see: "The column now tapers to a believable flame tip: bright at the hearth, fading through orange to red to black with height. Leave it running — it never degrades.",
          checkpoint: "A stable, flickering campfire. Beat 3.",
          recovery: ["cool() goes LAST in step.", "VEL_DECAY at 0.99 is load-bearing — this sim pumps energy in every frame and needs real drag."] }
      ]
    },
    {
      id: 4, title: "Smoke",
      build: "a second passenger riding the same flow — dark where fire is bright.",
      beat: "Gray smoke curls up and shrouds the glow.",
      steps: [
        { title: "The smoke field", adding: "smoke's decay dial, its two fields (placeholders, global line, field lines), and its line in cool (replace cool).",
          code: `SMOKE_DECAY = 0.992
vel = None
vel_next = None
temp = None
temp_next = None
smoke = None
smoke_next = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, smoke, smoke_next, pixels
    smoke = ti.field(ti.f32, shape=(N, N))
    smoke_next = ti.field(ti.f32, shape=(N, N))
@ti.kernel
def cool():
    for i, j in temp:
        temp[i, j] *= COOLING
        smoke[i, j] *= SMOKE_DECAY
        vel[i, j] *= VEL_DECAY`,
          does: "A scalar density field with the usual twin, decaying slower than heat (0.992 vs 0.985) — smoke outlives the flame that made it.",
          why: "Smoke is a 'passive tracer': it doesn't push anything, it just rides. Fires read as real because of what lingers after the bright part fades.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The smoke line slots between temp and vel inside cool.", "Both new fields are scalars, shaped (N, N)."] },
        { title: "Smoke rides along", adding: "smoke joining copy_back, the source, and step (replace all three).",
          code: `@ti.kernel
def copy_back():
    for i, j in temp:
        temp[i, j] = temp_next[i, j]
        smoke[i, j] = smoke_next[i, j]
        vel[i, j] = vel_next[i, j]
@ti.kernel
def burn_source(t: ti.f32):
    for i, j in temp:
        dx = i - N / 2
        dy = j - 12.0
        flick = 1.0 + 0.35 * ti.sin(0.31 * t + 0.05 * i)
        w = ti.exp(-(dx * dx + dy * dy) / (SOURCE_RADIUS * SOURCE_RADIUS)) * flick
        temp[i, j] = ti.min(temp[i, j] + 0.8 * w, 1.5)
        smoke[i, j] = ti.min(smoke[i, j] + 0.03 * w, 1.0)
def step():
    advect(temp, temp_next)
    advect(smoke, smoke_next)
    advect(vel, vel_next)
    copy_back()
    apply_buoyancy()
    cool()`,
          does: "Three edits with one theme: the burner now emits a little smoke with its heat (0.03 vs 0.8 — fires make far more heat than soot), and smoke advects and copies exactly like temperature.",
          why: "Count the cost of a whole new substance in the sim: one advect call, one copy line, one source line. THIS is why the solver was built generic — passengers are nearly free now.",
          see: "Runs clean — smoke exists and moves, but the renderer doesn't know it yet.",
          checkpoint: "No red text.",
          recovery: ["advect(smoke, smoke_next) goes between the temp and vel advects.", "Source smoke is ti.min-capped at 1.0, same pattern as heat."] },
        { title: "The shroud", adding: "smoke joining the picture (replace render).",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        t = ti.math.clamp(temp[i, j], 0.0, 1.0)
        fire = ti.Vector([1.6 * t, 1.2 * t * t, t * t * t])
        s = smoke[i, j] * 0.25
        pixels[i, j] = ti.math.clamp(fire + ti.Vector([s, s, s]), 0.0, 1.0)`,
          does: "The fire ramp as before, plus a gray veil: equal parts R, G, B scaled by smoke density, added on top and clamped.",
          why: "Additive gray is the one-line version of participating media — real volumetric smoke absorbs and scatters, but 'add a little gray where density lives' gets you 80% of the look for 2% of the code. (Project 12, volumetric clouds, does it properly.)",
          see: "Gray wisps curl up past the flame tip and hang above the fire, slowly thinning. The flame reads hotter by contrast.",
          checkpoint: "Smoke above the flame. Beat 4.",
          recovery: ["s is a plain float — build the gray with ti.Vector([s, s, s]).", "Clamp the SUM — bright flame plus smoke can exceed 1."] }
      ]
    },
    {
      id: 5, title: "Real fire",
      build: "the pressure projection and vorticity confinement, retyped from project 02 — flames that actually roll.",
      beat: "Flames lick, roll, and curl like the real thing.",
      steps: [
        { title: "Four more fields", adding: "pressure, its twin, divergence, and curl — the global statement now needs two lines.",
          code: `vel = None
vel_next = None
temp = None
temp_next = None
smoke = None
smoke_next = None
pressure = None
pressure_next = None
divergence = None
curl = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, smoke, smoke_next
    global pressure, pressure_next, divergence, curl, pixels
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))
    curl = ti.field(ti.f32, shape=(N, N))`,
          does: "The complete incompressibility-and-swirl toolkit from project 02, allocated in one go. Eleven fields total now — and Python is fine with global split across two statements.",
          why: "Right now hot air rises without making room for itself — that's why the plume looks like a lava lamp. These four fields buy back real fluid behavior.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["TWO global lines — one statement can't hold all eleven names readably.", "All four new fields are scalars."] },
        { title: "Measure and relax", adding: "divergence, the Jacobi iteration, and its copy-back — from project 02, from memory if you can.",
          code: `@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0]
            - vel[i, j][0]
            + sample(vel, i, j + 1)[1]
            - vel[i, j][1]
        )
@ti.kernel
def pressure_jacobi():
    for i, j in pressure:
        pressure_next[i, j] = (
            sample(pressure, i + 1, j)
            + sample(pressure, i - 1, j)
            + sample(pressure, i, j + 1)
            + sample(pressure, i, j - 1)
            - divergence[i, j]
        ) * 0.25
@ti.kernel
def copy_pressure():
    for i, j in pressure:
        pressure[i, j] = pressure_next[i, j]`,
          does: "Forward-difference divergence (neighbor-ahead minus self), and the relax-toward-the-answer Jacobi loop, exactly as in project 02.",
          why: "Second retype — and now you know WHY the divergence is forward, not central: its mirror twin arrives next step, and the pair must compose into the operator Jacobi solves.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Forward differences in divergence: sample(i + 1) minus vel[i, j].", "Jacobi subtracts divergence, then * 0.25."] },
        { title: "Push back, and conduct", adding: "the backward-difference gradient, the iteration count, and project() — the mirror-twin pair completed.",
          code: `JACOBI_ITERS = 40
@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector([
            pressure[i, j] - sample(pressure, i - 1, j),
            pressure[i, j] - sample(pressure, i, j - 1),
        ])
        vel[i, j] -= grad
def project():
    compute_divergence()
    for _ in range(JACOBI_ITERS):
        pressure_jacobi()
        copy_pressure()
    subtract_gradient()`,
          does: "Backward gradient (self minus neighbor-behind) mirroring the forward divergence, and the conductor: measure once, relax 40 rounds, push back once.",
          why: "Forward divergence + backward gradient = exactly the neighbor-average Laplacian Jacobi solves. Say it once more, because it's the most common silent bug in hand-built fluid solvers.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Backward: pressure[i, j] MINUS the i-1 / j-1 samples.", "subtract_gradient once, after the loop."] },
        { title: "Confinement returns", adding: "curl measurement and the vorticity force, with fire's strength dial.",
          code: `CURL_STRENGTH = 2.0
@ti.kernel
def compute_curl():
    for i, j in vel:
        curl[i, j] = (
            sample(vel, i + 1, j)[1]
            - sample(vel, i - 1, j)[1]
            - sample(vel, i, j + 1)[0]
            + sample(vel, i, j - 1)[0]
        ) * 0.5
@ti.kernel
def apply_vorticity(strength: ti.f32):
    for i, j in vel:
        grad = ti.Vector([
            ti.abs(sample(curl, i + 1, j)) - ti.abs(sample(curl, i - 1, j)),
            ti.abs(sample(curl, i, j + 1)) - ti.abs(sample(curl, i, j - 1)),
        ]) * 0.5
        n = grad / (grad.norm() + 1e-5)
        vel[i, j] += DT * strength * curl[i, j] * ti.Vector([n[1], -n[0]])`,
          does: "Score each cell's spin, find where spin concentrates, push around it — vorticity confinement, verbatim from project 02.",
          why: "Confinement was invented FOR fire (the original paper's target was movie flames). The licking, tearing tongues of a real flame are small vortices — exactly what the grid smears away and this force feeds back.",
          see: "Runs clean; wired in next step.",
          checkpoint: "No red text.",
          recovery: ["The 90° turn: ti.Vector([n[1], -n[0]]).", "abs() on the curl samples in the gradient."] },
        { title: "Assemble the real fire", adding: "step's final form (replace it), the curls flag in main (after fire_on), and the new tick call (replace step()).",
          code: `def step(curl_strength):
    advect(temp, temp_next)
    advect(smoke, smoke_next)
    advect(vel, vel_next)
    copy_back()
    apply_buoyancy()
    if curl_strength > 0.0:
        compute_curl()
        apply_vorticity(curl_strength)
    project()
    cool()
    curls_on = True
        step(CURL_STRENGTH if curls_on else 0.0)`,
          does: "The full algorithm: move everything, lift by heat, feed the swirls, enforce incompressibility, cool. Confinement before projection, as always — inject energy, then launder it legal.",
          why: "Read step() top to bottom and hear the whole story: advection (02), buoyancy (03), confinement (02), projection (02), cooling (03). Two projects of ideas in ten lines.",
          see: "Launch. The plume no longer just rises — it ROLLS, sheds vortices off its shoulders, and the flame tip tears into licking tongues.",
          checkpoint: "Flames that lick and roll. Beat 5 — the payoff.",
          recovery: ["Three homes: step at top level, curls_on = True after fire_on in main, the call replacing bare step().", "Confinement block before project() inside step."] },
        { title: "The A/B switch", adding: "the V key (replace the event block).",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on
            elif e.key == "v":
                curls_on = not curls_on`,
          does: "Same honest toggle as project 02.",
          why: "Judge the effect with your own eyes: V off, the fire goes soft and lava-lampy; V on, it sharpens into tongues.",
          see: "Toggle V while it burns — the flame's whole personality flips.",
          checkpoint: "V switches soft plume ↔ licking flames.",
          recovery: ["The v branch goes after the space branch."] }
      ]
    },
    {
      id: 6, title: "The torch",
      build: "a fire brush, a clear key, and the HUD — the bonfire becomes an instrument.",
      beat: "Drag fire anywhere — your pocket inferno.",
      steps: [
        { title: "The torch kernel", adding: "two dials and a mouse-sized burner that also pushes (add after burn_source).",
          code: `TORCH_RADIUS = 10.0
FORCE_SCALE = 300.0
@ti.kernel
def torch(x: ti.f32, y: ti.f32, fx: ti.f32, fy: ti.f32):
    for i, j in temp:
        dx = i - x * N
        dy = j - y * N
        w = ti.exp(-(dx * dx + dy * dy) / (TORCH_RADIUS * TORCH_RADIUS))
        temp[i, j] = ti.min(temp[i, j] + 0.9 * w, 1.5)
        smoke[i, j] = ti.min(smoke[i, j] + 0.05 * w, 1.0)
        vel[i, j] += w * ti.Vector([fx, fy])`,
          does: "burn_source's little sibling at the cursor: a tight Gaussian injecting heat, a puff of smoke, and — like project 02's splat — a push in the drag direction.",
          why: "Compare torch with burn_source and splat: all three are 'Gaussian stamp writes fields'. You now own a reusable mental template — emitter kernels — that shows up in every remaining project.",
          see: "Runs clean; wiring next.",
          checkpoint: "No red text.",
          recovery: ["Heat and smoke get ti.min caps; velocity gets += — pushes stack, temperatures saturate."] },
        { title: "Wire the drag", adding: "mouse memory (after frame = 0) and the drag block (after the event block) — with one new wrinkle for single clicks.",
          code: `    pmx, pmy = 0.0, 0.0
    dragging = False
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                torch(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE)
            else:
                torch(mx, my, 0.0, 0.0)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False`,
          does: "Project 02's drag pattern, plus an else: the very first frame of a click torches with zero push instead of doing nothing — so a single tap plants a flame.",
          why: "Small interaction details like tap-vs-drag are what make a toy feel finished. One extra branch, noticeably better hands.",
          see: "Tap: a flame blooms and rises. Drag: a burning stroke that shears in the wind of its own heat.",
          checkpoint: "Tap plants fire; drag paints it.",
          recovery: ["The else branch calls torch with 0.0, 0.0 forces.", "State lines go with the others in main's setup."] },
        { title: "Clear key and HUD", adding: "the reset kernel (after copy_back), the R key (replace the event block), and the final draw block with HUD (replace render/set_image/show).",
          code: `@ti.kernel
def clear_fields():
    for i, j in temp:
        temp[i, j] = 0.0
        smoke[i, j] = 0.0
        vel[i, j] = ti.Vector([0.0, 0.0])
        pressure[i, j] = 0.0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
            elif e.key == ti.GUI.SPACE:
                fire_on = not fire_on
            elif e.key == "v":
                curls_on = not curls_on
        render()
        gui.set_image(pixels)
        bonfire = "lit" if fire_on else "out"
        curls = "on" if curls_on else "off"
        gui.text(f"bonfire: {bonfire}  curls: {curls}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to torch  [space] bonfire  [v] curls  [r] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()`,
          does: "A blank-everything kernel on R (heat, smoke, motion, pressure — pressure too, or the next frame inherits a stale solve), and the two-line HUD in the house style.",
          why: "That's three complete instruments now — chemicals, ink, fire — all sharing one skeleton you could type in your sleep. Project 04 (lightning) finally breaks the mold: no grid advection at all.",
          see: "Snuff the bonfire with space, clear with R, paint torch strokes in the dark, relight. The fire triptych is complete.",
          checkpoint: "R clears, space toggles, HUD reads out. Final beat — project 03 complete.",
          recovery: ["clear_fields zeroes pressure too — stale pressure haunts the next solve.", "Event order: Escape, r, space, v."] }
      ]
    }
  ]
};

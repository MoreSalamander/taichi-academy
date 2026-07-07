// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["02-fluid"] = {
  project: "02-fluid",
  title: "Fluid Simulator",
  pitch: "Stir a box of incompressible ink — semi-Lagrangian advection, a real pressure solve, and swirls that answer your hand.",
  tier: "medium",
  language: "Python",
  file: "fluid.py",
  chapters: [
    {
      id: 1, title: "A box of ink",
      build: "the familiar scaffold — fields, a numpy seed, a window — but holding colored ink instead of chemicals.",
      beat: "Three glowing ink blobs hang in a black box.",
      steps: [
        { title: "Load the GPU toolkit", adding: "the docstring and the Taichi import — the same opening move as project 01.",
          code: `"""Stable fluids: stir a box of incompressible ink with your mouse."""
import taichi as ti`,
          does: "Names the file's mission and brings in Taichi. 'Stable fluids' is Jos Stam's famous 1999 algorithm — the one behind most real-time smoke and ink you've seen in games.",
          why: "Project 01 taught you fields and kernels. This project reuses every one of those muscles and adds two big ideas: moving a field through itself, and forcing a velocity field to behave like real water.",
          see: "Runs clean, nothing visible.",
          checkpoint: "python3 fluid.py returns silently.",
          recovery: ["No module named taichi — activate the venv: source .venv/bin/activate from the repo root."] },
        { title: "The ink field", adding: "the grid size, placeholders for ink and pixels, and init_sim — note the global line only names what exists so far; it will grow.",
          code: `N = 512
dye = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global dye, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    dye = ti.Vector.field(3, ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "Same shape as project 01's opening: wake the GPU (Metal, with CPU fallback), allocate once. dye holds an RGB amount of ink per cell — the thing you'll be pushing around all project.",
          why: "This project grows to NINE fields. We'll add them one chapter at a time, and only ever touch two places: the placeholder block and init_sim.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["dye is a Vector field with 3 channels — ink has color, unlike project 01's scalar chemicals."] },
        { title: "Ink blobs, made in numpy", adding: "the numpy import (above import taichi), and the seed: soft Gaussian blobs plus the upload bridge.",
          code: `import numpy as np
def seed_pattern(n, rng_seed=0, blobs=3):
    """Pure numpy: a few soft ink blobs to start with."""
    dye0 = np.zeros((n, n, 3), dtype=np.float32)
    colors = [(1.0, 0.35, 0.1), (0.15, 0.55, 1.0), (0.2, 1.0, 0.45)]
    rng = np.random.default_rng(rng_seed)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sigma = n / 14.0
    for k in range(blobs):
        cx, cy = rng.integers(n // 4, 3 * n // 4, size=2)
        w = np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (sigma * sigma))
        for ch in range(3):
            dye0[:, :, ch] += w * colors[k % 3][ch]
    return dye0.clip(0.0, 1.0).astype(np.float32)
def apply_seed(dye0):
    dye.from_numpy(dye0)`,
          does: "meshgrid builds two full grids of coordinates so the Gaussian w = exp(-dist²/σ²) is computed for every cell at once — no Python loop over pixels. Three soft blobs in ember, sky-blue, and mint land at seeded-random spots. apply_seed is the same one-way numpy→GPU bridge as before.",
          why: "Gaussians are the series' soft brush — you'll meet exp(-d²/σ²) again in this project's mouse splat. And it's pure numpy again: testable with no GPU.",
          see: "Runs clean; nothing calls it yet.",
          checkpoint: "No red text.",
          recovery: ["indexing=\"ij\" matters — it makes ii vary along the first axis, matching how fields are indexed.", "The clip keeps overlapping blobs from exceeding 1.0 before upload."] },
        { title: "Show the ink", adding: "a render kernel that just clamps ink to the screen, and the main loop (bottom of the file).",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.math.clamp(dye[i, j], 0.0, 1.0)
def main():
    init_sim()
    apply_seed(seed_pattern(N))
    gui = ti.GUI("Stable Fluids — taichi-academy", res=(N, N))
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "dye is already RGB, so rendering is one line: clamp it to legal color range. The main loop is project 01's skeleton verbatim — events, draw, show.",
          why: "Notice how much arrives free the second time: the loop, the guard, the render idea. New projects in this series start fast because the skeleton is muscle memory now.",
          see: "A black window with three soft glowing ink blobs — ember, blue, mint. Frozen, for now.",
          checkpoint: "Three blobs on black. Beat 1.",
          recovery: ["Blank black window — main must call apply_seed(seed_pattern(N)) after init_sim().", "clamp is ti.math.clamp(value, 0.0, 1.0) — three arguments."] }
      ]
    },
    {
      id: 2, title: "The whirlpool",
      build: "a velocity field, and advection — the machinery that carries ink along a flow.",
      beat: "The blobs swirl into spiral arms.",
      steps: [
        { title: "The flow field", adding: "velocity and a next-buffer for ink (grow the placeholder block, the global line, and add two field lines in init_sim after the ti.init block).",
          code: `vel = None
dye = None
dye_next = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, dye, dye_next, pixels
    vel = ti.Vector.field(2, ti.f32, shape=(N, N))
    dye_next = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "vel stores a 2D arrow per cell — which way the fluid moves there, in cells per tick. dye_next is the double-buffer twin you know from project 01: advection reads dye while writing dye_next.",
          why: "A fluid IS its velocity field. Everything from here on — stirring, pressure, swirls — is really about shaping vel; the ink just makes it visible.",
          see: "Runs clean, blobs still frozen.",
          checkpoint: "No red text.",
          recovery: ["Three separate edits: the placeholder block gains vel and dye_next; the global line now reads global vel, dye, dye_next, pixels; and the two new field lines go inside init_sim — vel above dye, dye_next below it.", "vel is 2 channels; dye_next is 3."] },
        { title: "Reading between the cells", adding: "two helper functions: a wrap-safe sampler, and bilinear interpolation (add them after apply_seed).",
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
          does: "sample wraps any index back onto the grid — the double-% handles negative numbers too. bilerp answers a question grids can't normally answer: 'what's the value at x=41.7, y=203.2?' It reads the four surrounding cells and blends them by closeness — lerp across x, then lerp those results across y.",
          why: "This is THE enabling trick for fluids. Advection needs to look 'a fraction of a cell upstream', and bilerp is how a discrete grid pretends to be continuous. You lerped palette colors in project 01 — same blend, now in 2D.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Both are @ti.func — helpers for kernels, like laplacian was.", "The wrap is ((i % N) + N) % N — written twice, once per axis.", "b and c differ only in which axis gets the +1 — easy to cross."] },
        { title: "Training wheels: a fixed whirlpool", adding: "a kernel that fills vel with a rigid rotation, and one call in main (add fill_vortex() right after apply_seed).",
          code: `@ti.kernel
def fill_vortex():
    for i, j in vel:
        vel[i, j] = ti.Vector([-(j - N / 2), (i - N / 2)]) * 0.01
def main():
    init_sim()
    apply_seed(seed_pattern(N))
    fill_vortex()`,
          does: "Each cell's arrow points perpendicular to the line from the center — the classic rigid-rotation recipe (-y, x). The whole grid turns like a record player at 0.01 radians per tick.",
          why: "Before velocity can move itself, we give it a hand-made flow to learn advection against. These training wheels come off next chapter — and this exact kernel will evolve into something else.",
          see: "Runs clean; the vortex exists but nothing uses it yet.",
          checkpoint: "No red text.",
          recovery: ["The minus sign goes on the j term: ti.Vector([-(j - N / 2), (i - N / 2)]).", "fill_vortex() goes inside main, right after apply_seed(seed_pattern(N))."] },
        { title: "Advection — look upstream", adding: "the time step, the advect kernel, the copy-back, and a step function (add after bilerp / after fill_vortex).",
          code: `DT = 1.0
@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)
@ti.kernel
def copy_back():
    for i, j in dye:
        dye[i, j] = dye_next[i, j]
def step():
    advect(dye, dye_next)
    copy_back()`,
          does: "Semi-Lagrangian advection, the heart of stable fluids: each cell asks 'what was HERE a moment ago?' — it steps backward along its own velocity arrow and bilerps the answer from the old field. Writes go to the _next twin; copy_back adopts them.",
          why: "You might expect to push values forward along arrows — but forward-pushing scatters (two cells land in the same place, gaps appear). Pulling backward guarantees every cell gets exactly one answer, and bilerp's averaging is what makes this method famously stable — it cannot blow up.",
          see: "Runs clean; one line to go.",
          checkpoint: "No red text.",
          recovery: ["It's i MINUS the velocity — backward along the arrow, not forward.", "advect is generic: it takes the field pair as template arguments; we'll reuse it on vel itself next chapter."] },
        { title: "Let it flow", adding: "the tick inside main's loop (right after the event block, before render()).",
          code: `        step()`,
          does: "Every frame, all ink takes one step through the whirlpool.",
          why: "Same moment as project 01 chapter 3 — a frozen picture becomes a simulation with one line.",
          see: "The three blobs begin to orbit, and because the record player turns faster near the rim than... actually — rigid rotation turns everything together, but the blobs are off-center, so they shear into long curved streaks and wind into spiral arms. Hypnotic.",
          checkpoint: "Blobs swirl into spirals. Beat 2.",
          recovery: ["Nothing moves — step() must be inside the while loop, before render(), indented to match.", "Everything smears to gray instantly — check DT = 1.0 and the 0.01 in fill_vortex."] }
      ]
    },
    {
      id: 3, title: "Stir it yourself",
      build: "mouse forces, self-advection, and fading — the whirlpool becomes yours.",
      beat: "Ink follows your hand.",
      steps: [
        { title: "The splat brush", adding: "two dials and a kernel that stamps ink AND velocity in one soft Gaussian blot (add after decay's future spot — right after clear... after copy_back's step function).",
          code: `BRUSH_RADIUS = 14.0
FORCE_SCALE = 300.0
@ti.kernel
def splat(x: ti.f32, y: ti.f32, fx: ti.f32, fy: ti.f32, r: ti.f32, g: ti.f32, b: ti.f32):
    for i, j in dye:
        dx = i - x * N
        dy = j - y * N
        w = ti.exp(-(dx * dx + dy * dy) / (BRUSH_RADIUS * BRUSH_RADIUS))
        dye[i, j] += w * ti.Vector([r, g, b])
        vel[i, j] += w * ti.Vector([fx, fy])`,
          does: "Every cell computes its distance to the mouse and weights itself by a Gaussian — full strength at the center, feathering smoothly to nothing. It adds ink (r,g,b) and a push (fx,fy) with that same soft falloff.",
          why: "Compare with project 01's splat: that one was a hard circle (inside/outside); this one is the Gaussian from seed_pattern, reused as a brush. Soft edges are why fluid stirring feels organic instead of stampy.",
          see: "Runs clean; not wired to the mouse yet.",
          checkpoint: "No red text.",
          recovery: ["Seven typed arguments, all ti.f32 — position, force, color.", "It's += on both fields — a brush adds; it doesn't overwrite."] },
        { title: "Wire the drag", adding: "mouse memory before the loop (after the gui line), and the drag block inside it (after the event block).",
          code: `    pmx, pmy = 0.0, 0.0
    dragging = False
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                splat(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE, 1.0, 0.35, 0.1)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False`,
          does: "The force isn't where the mouse IS — it's how the mouse MOVED: this frame's position minus last frame's, scaled up. The dragging flag skips the very first frame of a click (no previous position yet), so ink never jumps.",
          why: "Direction-from-motion is what makes stirring feel physical: flick fast, ink flies; drag slow, it oozes. Remembering a little state between frames (pmx, pmy) is a pattern every interactive sim uses.",
          see: "Drag through a blob — ember ink pours from your cursor and smears through the whirlpool. It still rides the fixed vortex; that's next.",
          checkpoint: "Dragging paints moving ink.",
          recovery: ["pmx, pmy and dragging go BEFORE the while loop; the if-block goes inside it, after the events.", "Ink appears but never moves with the drag — the force arguments are (mx - pmx) and (my - pmy), current minus previous."] },
        { title: "The flow moves itself", adding: "vel's own next-buffer (placeholders, global line, one init line), and vel joining the advection (replace copy_back and step).",
          code: `vel = None
vel_next = None
dye = None
dye_next = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pixels
    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))
@ti.kernel
def copy_back():
    for i, j in dye:
        dye[i, j] = dye_next[i, j]
        vel[i, j] = vel_next[i, j]
def step():
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()`,
          does: "advect(vel, vel_next) is the mind-bender: velocity carried by ITSELF — each arrow looks upstream along the arrows and becomes what was there. This is the actual nonlinearity of fluid motion. Because advect was written generic over template fields, reusing it costs one line.",
          why: "Self-advection is what lets a flick keep traveling after your mouse stops — the push you injected rides the flow it created. Every honest fluid solver has this feedback loop; it's also why fluids are chaotic and beautiful.",
          see: "Now your flicks GLIDE — throw ink and it keeps sailing, curling as it goes. (It also feels a bit syrupy and wrong. Chapter 4 fixes that.)",
          checkpoint: "A flick keeps moving after you let go.",
          recovery: ["Three small edits again: placeholder, global line, one field line — vel_next sits right under vel.", "copy_back now copies both pairs; step advects dye first, then vel."] },
        { title: "Nothing lasts forever", adding: "two decay dials, a fade kernel, and its call at the end of step (replace step).",
          code: `DYE_DECAY = 0.995
VEL_DECAY = 0.999
@ti.kernel
def decay():
    for i, j in dye:
        dye[i, j] *= DYE_DECAY
        vel[i, j] *= VEL_DECAY
def step():
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    decay()`,
          does: "Every tick, ink keeps 99.5% of itself and motion keeps 99.9%. Compounded over hundreds of frames, that's smoke thinning into air and swirls spending their energy.",
          why: "Without decay the box fills with ink and old motion accumulates forever. Two multiplies per cell buy you friction and evaporation — the cheapest physics you'll ever add.",
          see: "Old ink slowly ghosts away; the box stays readable no matter how much you pour in.",
          checkpoint: "Ink fades over time instead of piling up.",
          recovery: ["decay() goes LAST in step — fade what you just moved.", "The dials are that close to 1.0 on purpose; try 0.95 once to see why it's not lower."] },
        { title: "Take off the training wheels", adding: "fill_vortex's retirement: the kernel becomes clear_fields (replace it), and main drops the fill_vortex() call (replace main's first lines).",
          code: `@ti.kernel
def clear_fields():
    for i, j in dye:
        dye[i, j] = ti.Vector([0.0, 0.0, 0.0])
        vel[i, j] = ti.Vector([0.0, 0.0])
        pressure[i, j] = 0.0
def main():
    init_sim()
    apply_seed(seed_pattern(N))`,
          does: "The hand-made whirlpool is gone — from here on, every arrow in vel is one YOU put there with the mouse. In its place, a reset kernel that blanks ink, motion, and pressure (a field arriving next chapter — Python only checks names when a kernel first runs, so this compiles fine today).",
          why: "The scaffold-then-replace move is all over this series: build against a fake (rigid vortex), then swap in the real thing (your stirring). The fake becomes the reset tool — nothing wasted.",
          see: "Launch: blobs sit still until YOU move them. Stir gently, then flick — it's all yours now, if a little mushy. One chapter from greatness.",
          checkpoint: "No motion until you drag. Beat 3.",
          recovery: ["fill_vortex is REPLACED by clear_fields — one kernel in that spot, not two.", "main loses exactly one line: fill_vortex().", "Don't press r yet — nothing calls clear_fields until chapter 6 (and pressure arrives in chapter 4)."] }
      ]
    },
    {
      id: 4, title: "Incompressible",
      build: "the pressure projection — the constraint that turns mush into water.",
      beat: "It stops feeling like paste — it flows like water.",
      steps: [
        { title: "Three fields for one constraint", adding: "pressure, its Jacobi twin, and divergence (placeholder block, global line, three init lines after dye_next).",
          code: `vel = None
vel_next = None
dye = None
dye_next = None
pressure = None
pressure_next = None
divergence = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, pixels
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))`,
          does: "Three scalar fields: divergence will measure the crime (cells creating or destroying fluid), pressure will be computed as the correction, pressure_next is the double-buffer for the iterative solve.",
          why: "Real water can't be squeezed — the same amount flows out of any region as flows in. Your velocity field doesn't know that rule yet; these three fields are the machinery that enforces it every frame.",
          see: "Runs clean.",
          checkpoint: "No red text (and clear_fields' pressure line is now backed by a real field).",
          recovery: ["Same three-edit ritual: placeholders, global line, field lines — the trio sits between dye_next and pixels."] },
        { title: "Measure the crime", adding: "the divergence kernel (add after decay).",
          code: `@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0]
            - vel[i, j][0]
            + sample(vel, i, j + 1)[1]
            - vel[i, j][1]
        )`,
          does: "For each cell: how much more flows out the right side than in from my own x-arrow, plus the same for up — net outflow. Positive means this cell is 'creating' fluid; negative means it's swallowing it. Real water scores zero everywhere.",
          why: "Note the shape: neighbor-ahead minus self — a forward difference. Remember it when the gradient appears two steps from now; the two are mirror twins, and that pairing is what makes the solve actually work.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["[0] on the x-differences, [1] on the y-differences — vel is a 2-vector.", "It's forward: sample(vel, i + 1, j) minus vel[i, j], not the i-1 neighbor."] },
        { title: "Relax toward the answer", adding: "the Jacobi iteration and its copy-back (add after compute_divergence).",
          code: `@ti.kernel
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
          does: "We need a pressure field whose bumps exactly cancel the divergence — that's a Poisson equation, and Jacobi iteration solves it by relaxation: each cell repeatedly becomes the average of its four neighbors, minus the local divergence. Run it enough times and the field settles into the answer.",
          why: "This is your first iterative solver — 'guess, average, repeat' — and it's the same family (with upgrades) that powers serious physics engines. Notice it's project 01's neighbor-average pattern wearing a new hat: diffusion IS relaxation.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Averaging means * 0.25 at the end, after subtracting divergence.", "Double-buffered like everything else: write pressure_next, copy back — Jacobi needs the OLD neighbors."] },
        { title: "Push back", adding: "the gradient subtraction (add after copy_pressure).",
          code: `@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector([
            pressure[i, j] - sample(pressure, i - 1, j),
            pressure[i, j] - sample(pressure, i, j - 1),
        ])
        vel[i, j] -= grad`,
          does: "Pressure pushes from high to low: each cell subtracts the slope of the pressure hill under it. High-pressure spots (where fluid was piling up) shove the flow outward until the piling stops.",
          why: "Here's the mirror twin: divergence looked FORWARD (i+1 minus me), the gradient looks BACKWARD (me minus i-1). Composed, they make exactly the neighbor-average operator Jacobi just solved — the algebra closes, so the correction genuinely zeroes the divergence instead of just shrinking it. Mismatched pairs are the classic silent bug in fluid solvers.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Backward on both axes: pressure[i, j] minus the i-1 / j-1 neighbors.", "It's vel -= grad, subtracting — pressure pushes away from the pile-up."] },
        { title: "The projection", adding: "the iteration count and the plain-Python conductor (add after subtract_gradient).",
          code: `JACOBI_ITERS = 40
def project():
    compute_divergence()
    for _ in range(JACOBI_ITERS):
        pressure_jacobi()
        copy_pressure()
    subtract_gradient()`,
          does: "The full ritual: measure the crime once, relax pressure 40 rounds toward the correction, push back once. Plain Python gluing GPU kernels — like step, but for one sub-task.",
          why: "40 is a dial between speed and stiffness: fewer iterations, squishier fluid; more, crisper but slower. Since pressure is never cleared, each frame starts from last frame's answer — a warm start that makes 40 plenty.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Both kernels inside the loop — relax, adopt, repeat.", "subtract_gradient comes once, AFTER the loop."] },
        { title: "Turn it on", adding: "project() joining the tick (replace step).",
          code: `def step():
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    project()
    decay()`,
          does: "Every frame now ends with the incompressibility police: move things, then immediately cancel any pile-ups the motion created.",
          why: "Order matters: advect creates divergence, project removes it, decay fades what remains. This five-line step() IS the stable-fluids algorithm — you've built the whole thing.",
          see: "Launch and stir. The difference is unmistakable — pushes spread around obstacles of their own making, vortices spin off your strokes, ink folds into itself in sheets. It flows.",
          checkpoint: "Stirring makes eddies and swirls, not mush. Beat 4 — the big one.",
          recovery: ["project() goes after copy_back(), before decay().", "Feels unchanged — make sure step() calls project(), and that you replaced the old step rather than adding a second one."] }
      ]
    },
    {
      id: 5, title: "Crisper swirls",
      build: "vorticity confinement — an artist's knob that feeds the small whirls the grid keeps eating.",
      beat: "Tight curls bloom behind every stroke.",
      steps: [
        { title: "One more field", adding: "the curl field (placeholder, global line, one init line after divergence).",
          code: `vel = None
vel_next = None
dye = None
dye_next = None
pressure = None
pressure_next = None
divergence = None
curl = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, curl, pixels
    curl = ti.field(ti.f32, shape=(N, N))`,
          does: "curl will hold each cell's local spin: positive for counter-clockwise, negative for clockwise. That's the ninth and final field.",
          why: "Bilerp's averaging — the very thing that makes advection stable — slowly blurs small whirlpools away. To fight back we first have to SEE the spin.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The three-edit ritual, one last time — curl slots between divergence and pixels."] },
        { title: "See the spin, feed the spin", adding: "the curl measurement and the confinement force (add after project).",
          code: `@ti.kernel
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
          does: "compute_curl cross-differences the velocity to score each cell's rotation. apply_vorticity then finds which way 'more spin' lies (the gradient of |curl|), turns that arrow 90° — (n[1], -n[0]) — and pushes each cell around its local whirl, harder where spin is stronger.",
          why: "This is vorticity confinement, invented for movie smoke: give back the swirl energy the grid dissipates. The + 1e-5 is a guard you'll type in every project — never divide by a length that might be zero.",
          see: "Runs clean; not in the tick yet.",
          checkpoint: "No red text.",
          recovery: ["The 90° turn is ti.Vector([n[1], -n[0]]) — components swapped, minus on the second.", "abs() on the curl samples in the gradient — we chase spin STRENGTH, ignoring direction."] },
        { title: "A knob, not a hardcode", adding: "the strength dial (with the constants), step's final form (replace it), a state flag in main (after the gui line), and the tick call (replace step()).",
          code: `CURL_STRENGTH = 2.0
def step(curl_strength):
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    if curl_strength > 0.0:
        compute_curl()
        apply_vorticity(curl_strength)
    project()
    decay()
    curls_on = True
        step(CURL_STRENGTH if curls_on else 0.0)`,
          does: "step grows its first argument — how hard to confine — and skips the two kernels entirely at zero. main keeps a curls_on flag and passes the dial or 0.0 each frame.",
          why: "Confinement runs BEFORE project on purpose: it injects a bit of rogue energy, and the projection immediately launders it back into a legal, divergence-free flow. Artist knob, then physics police.",
          see: "Launch and stir — strokes now break into tighter, livelier curls that keep spinning after you stop.",
          checkpoint: "Visibly curlier than chapter 4.",
          recovery: ["Three homes: CURL_STRENGTH with the constants, curls_on inside main after the gui line, and the step(...) call replacing the bare step().", "The conditional call reads: step(CURL_STRENGTH if curls_on else 0.0)."] },
        { title: "Prove it with a key", adding: "the V key (replace the event block).",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "v":
                curls_on = not curls_on`,
          does: "V flips the flag; not is Python's cleanest toggle.",
          why: "An A/B switch is the honest way to judge an effect: stir hard, tap V, stir again. Your eyes will tell you exactly what confinement buys.",
          see: "Curls on: sharp, persistent whirls. Curls off: the same stroke relaxes into soft mush. Toggle mid-swirl and watch the character change.",
          checkpoint: "V audibly— visibly changes the fluid's personality. Beat 5.",
          recovery: ["The v branch goes after the Escape branch, same indentation.", "No visible change — confirm the tick call passes 0.0 when off (step(CURL_STRENGTH if curls_on else 0.0))."] }
      ]
    },
    {
      id: 6, title: "The instrument",
      build: "five inks, a reset, and a HUD — the sim becomes something you play.",
      beat: "Five inks, a reset, a HUD — your fluid instrument.",
      steps: [
        { title: "Five inks", adding: "the ink palette (after CURL_STRENGTH), a selection index in main (after the gui line), and the color lookup in the drag block (replace it).",
          code: `DYE_COLORS = [
    ("ember", 1.00, 0.35, 0.10),
    ("sky", 0.15, 0.55, 1.00),
    ("mint", 0.20, 1.00, 0.45),
    ("violet", 0.70, 0.30, 1.00),
    ("gold", 1.00, 0.85, 0.25),
]
    color_idx = 0
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if dragging:
                name, r, g, b = DYE_COLORS[color_idx]
                splat(mx, my, (mx - pmx) * FORCE_SCALE, (my - pmy) * FORCE_SCALE, r, g, b)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False`,
          does: "A named-color table (name, r, g, b), an index remembering the current ink, and the drag block unpacking the live row instead of hardcoded ember.",
          why: "Same 'dials become data' move as project 01's PRESETS — once colors are rows in a table, cycling them is an index bump.",
          see: "Still paints ember (index 0) — the cycle key lands next step.",
          checkpoint: "Painting works exactly as before.",
          recovery: ["Three homes again: DYE_COLORS at top level, color_idx = 0 inside main, and the lookup line inside the drag block.", "Unpack all four: name, r, g, b = DYE_COLORS[color_idx] — the name feeds the HUD soon."] },
        { title: "Reset and recolor", adding: "the R and C keys (replace the event block).",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "c":
                color_idx = (color_idx + 1) % len(DYE_COLORS)
            elif e.key == "v":
                curls_on = not curls_on`,
          does: "R finally calls the clear_fields kernel you built in chapter 3, then re-seeds fresh random blobs. C is the wrap-with-% cycle you know from project 01's palettes.",
          why: "Every tool you reached for was already on the bench — the reset kernel from the training-wheels swap, the cycle idiom from last project. Late chapters should feel like assembly, not invention.",
          see: "C cycles ember → sky → mint → violet → gold as you paint. R wipes the box and drops three fresh blobs.",
          checkpoint: "All four keys live: R, C, V, Esc.",
          recovery: ["Order in the block: Escape, r, c, v.", "The reseed line ends with three closing parens — seed_pattern(...) inside apply_seed(...)."] },
        { title: "The HUD", adding: "the final draw block with two text overlays (replace render()/set_image/show at the bottom of the loop).",
          code: `        render()
        gui.set_image(pixels)
        name = DYE_COLORS[color_idx][0]
        curls = "on" if curls_on else "off"
        gui.text(f"dye: {name}  curls: {curls}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to stir  [c] color  [v] curls  [r] reset", (0.02, 0.94), color=0xAAAAAA)
        gui.show()`,
          does: "Reads the current ink's name from the table, formats the state line with an f-string, and overlays it plus a dim control legend — same HUD recipe as project 01's finale.",
          why: "Instrument, not demo: the screen now tells you what you're holding and what it can do. That closes project 02 — a real-time incompressible fluid solver, every line typed by you.",
          see: "Stir violet through gold through sky, toggle curls, reset, paint again. Semi-Lagrangian advection, a converging pressure solve, vorticity confinement — words that were noise seven chapters ago, all under your fingers now.",
          checkpoint: "HUD reads out ink and curls. Final beat — project 02 complete.",
          recovery: ["The two gui.text lines go between set_image and show, y at 0.98 and 0.94.", "NameError: name — the lookup line defining it sits right after gui.set_image(pixels)."] }
      ]
    }
  ]
};

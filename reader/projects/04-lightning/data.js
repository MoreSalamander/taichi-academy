// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["04-lightning"] = {
  project: "04-lightning",
  title: "Lightning",
  pitch: "Recursive bolts fork across the sky, leave glowing ghosts, and flash the whole world white.",
  tier: "easy-med",
  language: "Python",
  file: "lightning.py",
  chapters: [
    {
      id: 1, title: "A dark sky",
      build: "a night sky, a way to stamp bright line segments into a field, and a first (boring) bolt.",
      beat: "A straight bolt of light splits the dark.",
      steps: [
        { title: "Load the GPU toolkit", adding: "the docstring and the Taichi import.",
          code: `"""Branching lightning: recursive bolts, blue afterglow, storm flashes."""
import taichi as ti`,
          does: "Project 04 breaks the mold: the first three projects were grids evolving under physics kernels; this one GENERATES shapes with recursive CPU code and uses the GPU for light — deposit, fade, glow.",
          why: "Not everything is a simulation. Procedural generation — building complex shapes from simple recursive rules — is the other half of this series (planets, galaxies, terrain all come from it), and lightning is its perfect first lesson.",
          see: "Runs clean, nothing visible.",
          checkpoint: "python3 lightning.py returns silently.",
          recovery: ["No module named taichi — activate the venv first."] },
        { title: "Two fields", adding: "the grid size, placeholders, and init_sim allocating the bolt layer and the pixels.",
          code: `N = 512
bolt = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global bolt, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    bolt = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "bolt holds brightness — where lightning IS right now, one scalar per cell. The familiar allocate-once opening, at its smallest.",
          why: "Notice how light this project starts: two fields. Layers get added exactly when a chapter needs them.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Same skeleton as every project — if this feels automatic now, that's the curriculum working."] },
        { title: "Stamping lines", adding: "numpy (above import taichi) and the segment stamper — pure numpy, the workhorse of the whole project.",
          code: `import numpy as np
def deposit_segment(field, p0, p1, bright):
    """Pure numpy: stamp a straight bright segment into a (n, n) array."""
    n = field.shape[0]
    length = float(np.hypot(*(p1 - p0)))
    steps = max(2, int(length * 2))
    ts = np.linspace(0.0, 1.0, steps)
    xs = np.clip(p0[0] + (p1[0] - p0[0]) * ts, 0, n - 1).astype(np.int32)
    ys = np.clip(p0[1] + (p1[1] - p0[1]) * ts, 0, n - 1).astype(np.int32)
    field[xs, ys] = np.maximum(field[xs, ys], bright)`,
          does: "To draw a line into a grid: walk from p0 to p1 in small steps (two per pixel of length so no gaps), round each point to a cell, and light it. np.linspace makes all the in-between points in one call; fancy indexing field[xs, ys] lights them all at once. np.maximum instead of = means a dim line can never erase a bright one crossing it.",
          why: "Rasterization — turning geometry into pixels — in eight lines. Every renderer on earth does some version of this walk. The max-write rule matters the moment bolts fork and cross.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["steps scales with length — a fixed count would leave gaps on long segments.", "np.clip keeps endpoints that wander off-grid from crashing the indexing."] },
        { title: "First light", adding: "the render kernel and a main that stamps one straight test bolt (the preview lines are temporary — chapter 2 deletes them).",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        pixels[i, j] = ti.math.clamp(sky + core, 0.0, 1.0)
def main():
    init_sim()
    preview = np.zeros((N, N), dtype=np.float32)
    deposit_segment(preview, np.array([N / 2, N - 1.0]), np.array([N / 2, 0.0]), 1.0)
    bolt.from_numpy(preview)
    gui = ti.GUI("Lightning — taichi-academy", res=(N, N))
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "Render composites two layers additively: a nearly-black night sky plus a blue-white core wherever bolt is lit. Main stamps one dead-straight segment from top (j = N-1) to ground (j = 0) as a test pattern.",
          why: "Additive layer compositing — sky + core, later + halo + flash — is how this project builds its look; each chapter adds one term to the sum. The straight preview proves the stamp-upload-render pipeline before recursion complicates things.",
          see: "A thin, dead-straight white line down the middle of a dark blue-black sky. The world's most boring lightning — chapter 2 fixes that.",
          checkpoint: "One straight vertical line. Beat 1.",
          recovery: ["Line missing — bolt.from_numpy(preview) must come after the deposit_segment call.", "Remember the origin is bottom-left: N-1 is the TOP of the window."] }
      ]
    },
    {
      id: 2, title: "The jagged path",
      build: "the recursive heart: midpoint displacement, and strikes on click.",
      beat: "Click — a crooked bolt strikes where you point.",
      steps: [
        { title: "Midpoint displacement", adding: "the bolt generator — a function with a smaller function inside it, calling itself (add after deposit_segment).",
          code: `def generate_bolt(n, x_frac, rng_seed=0):
    """Pure numpy + recursion: a jagged, branching bolt as a (n, n) brightness array."""
    rng = np.random.default_rng(rng_seed)
    field = np.zeros((n, n), dtype=np.float32)
    def jag(p0, p1, bright, depth):
        d = p1 - p0
        length = float(np.hypot(*d))
        if length < 8.0 or depth > 10:
            deposit_segment(field, p0, p1, bright)
            return
        mid = (p0 + p1) / 2
        perp = np.array([-d[1], d[0]]) / (length + 1e-9)
        mid = mid + perp * rng.uniform(-0.25, 0.25) * length
        jag(p0, mid, bright, depth + 1)
        jag(mid, p1, bright, depth + 1)
    start = np.array([x_frac * n, n - 1.0])
    end = np.array([x_frac * n + rng.uniform(-0.15, 0.15) * n, 0.0])
    jag(start, end, 1.0, 0)
    return field`,
          does: "jag takes a segment and asks: short enough? Stamp it and stop (the base case). Otherwise: find the midpoint, shove it sideways — along the perpendicular (-dy, dx), by a random fraction of the length — and jag each half. Every level of recursion doubles the segments and halves their size; kinks are big at the top and fine at the bottom, exactly like real lightning.",
          why: "This is recursion earning its keep: five lines of rule, unbounded jaggedness. 'Displace the midpoint, recurse on halves' is THE fractal recipe — the same idea builds mountain ridges (project 05 territory) and coastlines. The base case comes FIRST — write the stopping rule before the recursive calls, always.",
          see: "Runs clean; nothing calls it yet.",
          checkpoint: "No red text.",
          recovery: ["The perpendicular is np.array([-d[1], d[0]]) — components swapped, minus on the first.", "Displacement scales with length — a fixed shove makes fuzzy noise, not lightning.", "Both guards matter: length < 8.0 AND depth > 10 (a runaway recursion crashes Python)."] },
        { title: "The strike pipeline", adding: "a deposit field (placeholder, global, field line), the absorb kernel, and strike() gluing CPU to GPU (add after generate_bolt).",
          code: `bolt = None
deposit = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global bolt, deposit, pixels
    deposit = ti.field(ti.f32, shape=(N, N))
@ti.kernel
def absorb():
    for i, j in bolt:
        bolt[i, j] = ti.max(bolt[i, j], deposit[i, j])
def strike(x_frac, rng_seed=0):
    deposit.from_numpy(generate_bolt(N, x_frac, rng_seed))
    absorb()`,
          does: "The CPU builds a bolt in numpy; from_numpy ships it to the deposit field; absorb merges it into the live bolt layer with max — new lightning brightens the sky, never darkens it.",
          why: "This three-stage pipeline — generate on CPU, upload, merge on GPU — is the standard shape for 'procedural content enters a live sim'. You'll use it verbatim when galaxies upload star fields.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The three-edit ritual for deposit: placeholder, global line, field line.", "absorb uses ti.max, mirroring deposit_segment's np.maximum — same rule, both sides of the bridge."] },
        { title: "Strike on click", adding: "main dropping its training-wheels preview (replace main's first lines), and a mouse branch in the events (replace the event block).",
          code: `def main():
    init_sim()
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))`,
          does: "The straight preview is deleted — main is two lines again, and the sky starts dark. A mouse click arrives as an event like a key press (e.key == ti.GUI.LMB); e.pos hands over where, and its x becomes the strike position with a fresh random seed.",
          why: "Same scaffold-then-replace move as project 02's vortex. And note clicks-as-events versus project 02's is_pressed polling: events fire once per click (one bolt), polling fires every frame (continuous paint). Choosing between them is a design decision you now own.",
          see: "Launch: darkness. Click: a crooked bolt tears down from the top at your cursor's x. Every click, a different bolt — they pile up like a long-exposure photo (fading comes next chapter).",
          checkpoint: "Click strikes. Beat 2.",
          recovery: ["Delete all three preview lines — main() is just init_sim() then the gui line.", "e.pos unpacks to mx, my — we use mx and politely ignore my."] }
      ]
    },
    {
      id: 3, title: "Branches and fade",
      build: "forks that make it lightning, and decay that makes it a flash.",
      beat: "Forked lightning flashes and dies.",
      steps: [
        { title: "The fork", adding: "a branch chance (with the constants) and one block inside jag, after the two recursive calls (replace generate_bolt).",
          code: `BRANCH_CHANCE = 0.35
def generate_bolt(n, x_frac, rng_seed=0):
    """Pure numpy + recursion: a jagged, branching bolt as a (n, n) brightness array."""
    rng = np.random.default_rng(rng_seed)
    field = np.zeros((n, n), dtype=np.float32)
    def jag(p0, p1, bright, depth):
        d = p1 - p0
        length = float(np.hypot(*d))
        if length < 8.0 or depth > 10:
            deposit_segment(field, p0, p1, bright)
            return
        mid = (p0 + p1) / 2
        perp = np.array([-d[1], d[0]]) / (length + 1e-9)
        mid = mid + perp * rng.uniform(-0.25, 0.25) * length
        jag(p0, mid, bright, depth + 1)
        jag(mid, p1, bright, depth + 1)
        if depth <= 4 and rng.random() < BRANCH_CHANCE:
            dirv = mid - p0
            ang = rng.uniform(-0.7, 0.7)
            ca, sa = np.cos(ang), np.sin(ang)
            rot = np.array([dirv[0] * ca - dirv[1] * sa, dirv[0] * sa + dirv[1] * ca])
            jag(mid, mid + rot * 0.7, bright * 0.45, depth + 1)
    start = np.array([x_frac * n, n - 1.0])
    end = np.array([x_frac * n + rng.uniform(-0.15, 0.15) * n, 0.0])
    jag(start, end, 1.0, 0)
    return field`,
          does: "Sometimes (35%, and only in the first few recursion levels) the midpoint sprouts a THIRD recursive call: the incoming direction rotated by a random angle (that four-line ca/sa dance is the 2D rotation formula), shorter, and at 45% brightness. Branches jag recursively too, so they fork again themselves.",
          why: "One conditional recursive call turns a crooked line into a lightning TREE. Dimmer branches encode the physics story — less charge takes the side road — and it's what your eye uses to find the main channel.",
          see: "Click: bolts now fork into dimmer side-strands that fork again. Unmistakably lightning.",
          checkpoint: "Forked bolts. Compare a few clicks.",
          recovery: ["The branch block goes AFTER the two jag calls, inside jag.", "depth <= 4 keeps forks near the trunk — branches everywhere reads as a root ball, not lightning.", "Rotation: [x·ca − y·sa, x·sa + y·ca] — minus on the first line."] },
        { title: "Flash and die", adding: "the fade dial, the fade kernel, a step() to conduct per-frame work, and its call in the loop (before render()).",
          code: `FADE = 0.90
@ti.kernel
def fade():
    for i, j in bolt:
        bolt[i, j] *= FADE
def step():
    fade()
        step()`,
          does: "The bolt layer loses 10% per frame — after half a second it's effectively gone. step() looks silly holding one call; it's the skeleton the next chapter fills.",
          why: "10% per frame is fast on purpose: real lightning lives for milliseconds, and the eye reads the sharp decay as violence. Compare the gentle 0.5% fades in the fluid projects — decay RATE is texture.",
          see: "Click: the bolt blazes, dims through blue, and is gone in a blink. The long-exposure pile-up is cured.",
          checkpoint: "Bolts flash and die. Beat 3.",
          recovery: ["step() goes inside the while loop before render(); def step() at top level.", "Bolts vanish instantly — FADE is 0.90, not 0.09."] }
      ]
    },
    {
      id: 4, title: "Afterglow",
      build: "a second light layer that spreads and lingers — diffusion returns as a glow.",
      beat: "Every strike leaves a spreading blue ghost.",
      steps: [
        { title: "The glow layer", adding: "glow and its double-buffer twin (placeholders, global line, field lines).",
          code: `bolt = None
deposit = None
glow = None
glow_next = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global bolt, deposit, glow, glow_next, pixels
    glow = ti.field(ti.f32, shape=(N, N))
    glow_next = ti.field(ti.f32, shape=(N, N))`,
          does: "A second brightness layer with the _next twin you know from three projects of double-buffering.",
          why: "Bolt is the strike; glow will be its memory — charged air still shining. Separating them lets each fade and spread at its own speed, which is the whole trick of layered light.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The ritual: placeholder, global, field lines — glow pair between deposit and pixels."] },
        { title: "Feed and fade the glow", adding: "glow's decay dial, its line in absorb, and its line in fade (replace both kernels).",
          code: `GLOW_FADE = 0.96
@ti.kernel
def absorb():
    for i, j in bolt:
        bolt[i, j] = ti.max(bolt[i, j], deposit[i, j])
        glow[i, j] += deposit[i, j]
@ti.kernel
def fade():
    for i, j in bolt:
        bolt[i, j] *= FADE
        glow[i, j] *= GLOW_FADE`,
          does: "Every strike now pours its brightness into the glow layer too — with +=, so repeated strikes in one place build up charge. And glow fades at 4% a frame against the bolt's 10%: the flash dies fast, the ghost lingers.",
          why: "Two layers, two clocks. Fast decay reads as the event; slow decay reads as its consequence. That pairing (impact + afterglow) is a universal effects recipe — explosions, muzzle flashes, spell hits.",
          see: "Runs clean — glow accumulates but the renderer can't see it yet.",
          checkpoint: "No red text.",
          recovery: ["absorb: max for bolt, += for glow — different rules on purpose.", "GLOW_FADE (0.96) must be closer to 1 than FADE (0.90) or the ghost dies before the flash."] },
        { title: "Diffusion returns", adding: "the spread dial, a diffusion kernel for glow, its copy-back, and both joining step (replace step).",
          code: `GLOW_SPREAD = 0.2
@ti.kernel
def diffuse_glow():
    for i, j in glow:
        lap = (
            glow[(i + 1) % N, j]
            + glow[(i - 1) % N, j]
            + glow[i, (j + 1) % N]
            + glow[i, (j - 1) % N]
            - 4.0 * glow[i, j]
        )
        glow_next[i, j] = glow[i, j] + GLOW_SPREAD * lap
@ti.kernel
def copy_glow():
    for i, j in glow:
        glow[i, j] = glow_next[i, j]
def step():
    fade()
    diffuse_glow()
    copy_glow()`,
          does: "The Laplacian from project 01, back for its second job: each frame, glow flows toward its neighbors' average — thin bright strands melt into wide soft halos. Same neighbors-minus-self stencil, same write-to-twin-then-copy discipline.",
          why: "Full circle: chapter 3 of your first project taught this exact operator as chemistry; here it's light bleeding through fog. Operators aren't tied to their first story — the Laplacian IS spreading, whatever's spreading.",
          see: "Runs clean; one render change to see it.",
          checkpoint: "No red text.",
          recovery: ["GLOW_SPREAD at 0.2 keeps the plain 5-point stencil stable (remember project 01's NaN lesson — 0.25 is the cliff edge).", "step order: fade, diffuse, copy."] },
        { title: "The ghost made visible", adding: "the halo term in render (replace render).",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        g = ti.min(glow[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        halo = g * ti.Vector([0.25, 0.40, 0.95])
        pixels[i, j] = ti.math.clamp(sky + halo + core, 0.0, 1.0)`,
          does: "One new additive term: deep electric blue scaled by glow, layered under the white core (added first = visually beneath).",
          why: "White-hot core inside a saturated halo is how cameras see lightning (the center overexposes to white; the fringe keeps its color). Fake the optics, get the drama.",
          see: "Click: the bolt flashes white, dies, and leaves a soft blue tree-shaped ghost that swells and thins away over a couple of seconds. Click twice in one spot — the ghost stacks brighter.",
          checkpoint: "Blue spreading afterglow. Beat 4.",
          recovery: ["Sum order sky + halo + core — core added last sits visually on top.", "Both b and g get ti.min clamps before use — stacked strikes push glow past 1."] }
      ]
    },
    {
      id: 5, title: "The storm",
      build: "screen flash, self-striking weather, a clear key, and the HUD.",
      beat: "The sky flashes — a storm rages on its own.",
      steps: [
        { title: "The flash", adding: "a fade dial, flash as render's argument (replace render), flash state in main (after the gui line), the flash line in the click branch (replace the event block), and the new draw block (replace render()/set_image/show).",
          code: `FLASH_FADE = 0.85
@ti.kernel
def render(flash: ti.f32):
    for i, j in pixels:
        b = ti.min(bolt[i, j], 1.0)
        g = ti.min(glow[i, j], 1.0)
        sky = ti.Vector([0.01, 0.01, 0.04]) + flash * ti.Vector([0.06, 0.08, 0.16])
        core = b * ti.Vector([0.92, 0.96, 1.00])
        halo = g * ti.Vector([0.25, 0.40, 0.95])
        pixels[i, j] = ti.math.clamp(sky + halo + core, 0.0, 1.0)
    flash = 0.0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0
        render(flash)
        flash *= FLASH_FADE
        gui.set_image(pixels)
        gui.show()`,
          does: "Flash is a single number living in main — no field needed, because it lights every pixel equally. A strike sets it to 1; render brightens the whole sky by it; it decays 15% per frame right in the loop.",
          why: "Cheapest effect in the project, biggest physical read: the whole sky lighting up is what tells your body a strike happened. Note the state lives CPU-side and is passed in — not everything deserves a field.",
          see: "Click: the entire sky blinks blue-white with the strike and settles over half a second.",
          checkpoint: "Clicks flash the whole sky.",
          recovery: ["Four homes: FLASH_FADE with the constants, flash = 0.0 in main's setup, flash = 1.0 in the click branch, and the render(flash)/flash *= FLASH_FADE pair in the draw block.", "flash *= FLASH_FADE goes right after render(flash)."] },
        { title: "Weather", adding: "the storm period, two state lines, the space toggle (replace the event block), the auto-strike block (after the events), and the frame counter (after step()).",
          code: `STORM_PERIOD = 90
    storm_on = True
    frame = 0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                storm_on = not storm_on
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0
        if storm_on and frame % STORM_PERIOD == 0:
            strike(np.random.random(), np.random.randint(1_000_000))
            flash = 1.0
        frame += 1`,
          does: "Every 90 frames (about 1.5 seconds), the storm strikes somewhere random on its own — the frame % PERIOD == 0 idiom from project 03's flicker clock, now scheduling events. Space is the weather switch.",
          why: "The moment a toy acts WITHOUT input, it becomes a world. One modulo and two lines of state is all 'autonomous' costs here.",
          see: "Lean back: bolts tear down at random, the sky flashing with each. Click to join in; space for sudden calm.",
          checkpoint: "It storms by itself. Space stops it.",
          recovery: ["The auto-strike block sits between the event loop and step().", "frame += 1 after step() — count frames, not events."] },
        { title: "Clear air", adding: "the reset kernel (after copy_glow) and the R key (replace the event block).",
          code: `@ti.kernel
def clear_fields():
    for i, j in bolt:
        bolt[i, j] = 0.0
        glow[i, j] = 0.0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                clear_fields()
            elif e.key == ti.GUI.SPACE:
                storm_on = not storm_on
            elif e.key == ti.GUI.LMB:
                mx, my = e.pos
                strike(mx, np.random.randint(1_000_000))
                flash = 1.0`,
          does: "Blank both light layers. (No pressure or velocity to worry about in this project — the reset is honest at two lines.)",
          why: "House style: every instrument gets R. Notice how much smaller this one is than fire's — the reset always tells you exactly how much state a sim carries.",
          see: "Mid-storm, tap R: instant black, then the next scheduled strike relights the sky.",
          checkpoint: "R wipes the sky clean.",
          recovery: ["Event order: Escape, r, space, LMB."] },
        { title: "The HUD", adding: "the two text lines in the draw block (replace it one last time).",
          code: `        render(flash)
        flash *= FLASH_FADE
        gui.set_image(pixels)
        storm = "on" if storm_on else "off"
        gui.text(f"storm: {storm}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("click to strike  [space] storm  [r] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()`,
          does: "The standard two-line HUD: state on top, controls dimmed below.",
          why: "Project 04 complete — and your first generator. Look at the split: recursion and randomness on the CPU where Python shines; light, spread, and fade on the GPU where parallelism shines. Choosing the right side for each job is the real skill this project taught.",
          see: "A storm that strikes itself, forked bolts on demand, ghosts and flashes — four instruments in the case now. Next up: terrain erosion, where midpoint displacement returns to build mountains.",
          checkpoint: "HUD reads out the storm. Final beat — project 04 complete.",
          recovery: ["Text lines between set_image and show, y at 0.98 and 0.94."] }
      ]
    }
  ]
};

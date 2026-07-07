// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["01-reaction-diffusion"] = {
  project: "01-reaction-diffusion",
  title: "Reaction-Diffusion",
  pitch: "Two invisible chemicals spread and react on your GPU — and coral, cells, and worms grow out of the math.",
  tier: "easy",
  language: "Python",
  file: "gray_scott.py",
  chapters: [
    {
      id: 1, title: "A window of pixels",
      build: "a window whose every pixel is computed by your GPU, all at once, every frame.",
      beat: "A window opens filled with a smooth color gradient your GPU painted.",
      steps: [
        { title: "Load the GPU toolkit", adding: "the note-to-self at the top, and the line that gives you Taichi — Python that runs on your graphics card.",
          code: `"""Gray-Scott reaction-diffusion: two chemicals paint living patterns."""
import taichi as ti`,
          does: "The triple-quoted line is a docstring — a note describing what this file is; Python keeps it but doesn't run it. import taichi as ti brings in Taichi, a library that takes functions you write in Python and compiles them to run on the GPU.",
          why: "Everything in this whole series rides on this import. Your Mac's GPU can do thousands of little calculations at the same time — Taichi is how we hand it work.",
          see: "Run python3 gray_scott.py — nothing visible, just your prompt back. We only opened the toolbox.",
          checkpoint: "It ran and returned to the prompt with no red text.",
          recovery: ["No module named taichi — the venv isn't active; run source .venv/bin/activate from the repo root first.", "SyntaxError near the top — the docstring needs three double-quotes on each side."] },
        { title: "A grid of pixels on the GPU", adding: "the grid size, a placeholder for the pixel grid, and the function that starts Taichi and allocates it (add below the import).",
          code: `N = 512
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "ti.init(arch=ti.gpu) wakes the GPU (on your Mac that's Metal); if that ever fails we fall back to the CPU so the program still runs. A field is Taichi's array living in GPU memory — this one is 512x512, and each cell holds a Vector of 3 floats: red, green, blue. One cell per pixel.",
          why: "Fields are THE Taichi idea — big grids of numbers the GPU works on in parallel. And we allocate them once, inside one function, because Metal can't free GPU fields; every project in this series follows this pattern.",
          see: "Runs clean, still nothing visible — the grid exists in GPU memory but nobody has painted or shown it.",
          checkpoint: "python3 gray_scott.py returns silently, no red text.",
          recovery: ["NameError: N — N = 512 must come before init_sim uses it.", "Watch the field line: ti.Vector.field(3, ti.f32, shape=(N, N)) — the 3 is channels (RGB), the shape is the grid.", "IndentationError — everything from global pixels down sits inside init_sim, indented once."] },
        { title: "Your first kernel", adding: "a GPU function that paints every pixel — all 262,144 of them — at the same time.",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([i / N, j / N, 0.3])`,
          does: "@ti.kernel marks render as code Taichi compiles for the GPU. The line for i, j in pixels looks like a normal loop, but it isn't one: the GPU runs every (i, j) cell simultaneously. Each pixel colors itself from its own coordinates — more red as i grows, more green as j grows.",
          why: "This is the mental flip the whole series trains: stop thinking 'visit each pixel one after another' and start thinking 'every pixel computes itself at once'. Every simulation to come is kernels like this.",
          see: "Still nothing on screen — the kernel exists but nothing calls it yet. One more step.",
          checkpoint: "Runs clean, no window yet.",
          recovery: ["Missing @ti.kernel above def render — without it this is plain slow Python.", "Error about Vector — it's ti.Vector([...]) with square brackets inside the parentheses."] },
        { title: "Open the window", adding: "the main function: start the sim, open a window, and show the painted pixels every frame (add at the bottom of the file).",
          code: `def main():
    init_sim()
    gui = ti.GUI("Gray-Scott — taichi-academy", res=(N, N))
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "ti.GUI opens a 512x512 window. The while gui.running loop is the heartbeat: each pass it checks for key presses (Esc quits), runs the render kernel on the GPU, hands the pixels field to the window, and shows it. The __name__ guard at the bottom means main() runs when you launch the file, but not when tests import it.",
          why: "This loop shape — events, compute, draw, show — is the skeleton of every visual project in the series. From here on we only swap what happens in the middle.",
          see: "A window opens filled with a smooth gradient: dark blue-green in one corner flowing to bright yellow-pink at the other. Your GPU is painting a quarter-million pixels per frame. Esc closes it.",
          checkpoint: "The gradient window opens and stays until you press Esc. That's Beat 1.",
          recovery: ["Window opens black — check render() is called inside the loop, before gui.set_image(pixels).", "NameError: main — the last two lines must be at the far left margin, not indented.", "Nothing opens and it exits — while gui.running: and everything under it must be indented inside main()."] }
      ]
    },
    {
      id: 2, title: "Two chemicals",
      build: "the two chemical grids U and V, and a seed of V dropped into a sea of U.",
      beat: "A dark square floats in a pale field — chemical V dropped into a sea of U.",
      steps: [
        { title: "Four more grids", adding: "numpy at the top (below the docstring, above import taichi), and placeholders for the chemical fields (replace the pixels = None line).",
          code: `import numpy as np
u = None
v = None
u_next = None
v_next = None
pixels = None`,
          does: "numpy is Python's CPU array library — we'll use it to build starting patterns before uploading them to the GPU. u and v will hold the two chemical concentrations at every grid cell. u_next and v_next are their 'next frame' twins — you'll see why in a moment.",
          why: "The simulation is literally these two numbers per cell, changing over time. The _next twins exist because every cell updates at once: you can't overwrite u while other cells are still reading it. Compute into _next, then swap — the double-buffer trick, and you'll use it in every grid sim ever.",
          see: "Runs clean; nothing visible changed.",
          checkpoint: "python3 gray_scott.py still shows the gradient.",
          recovery: ["Keep the order: import numpy as np sits above import taichi as ti.", "All five placeholder lines sit together at the left margin, above def init_sim."] },
        { title: "Allocate the chemicals", adding: "the four chemical fields inside init_sim (replace the whole init_sim function — the global line grows, and four field lines join pixels).",
          code: `def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global u, v, u_next, v_next, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    u = ti.field(ti.f32, shape=(N, N))
    v = ti.field(ti.f32, shape=(N, N))
    u_next = ti.field(ti.f32, shape=(N, N))
    v_next = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "Four plain scalar fields — one f32 number per cell this time (concentration), not a 3-vector like pixels. Same 512x512 shape, allocated once, right after Taichi wakes up.",
          why: "ti.field for one number per cell, ti.Vector.field for several — that's the whole vocabulary. You now have five GPU grids: two chemicals, their two futures, and the picture.",
          see: "Runs clean, gradient unchanged.",
          checkpoint: "Still the gradient, no red text.",
          recovery: ["The global line must name all five now: global u, v, u_next, v_next, pixels.", "Scalar fields are ti.field(ti.f32, ...) — no Vector, no channel count."] },
        { title: "Mix the starting chemicals", adding: "the seed size, and a pure-numpy function that builds the starting pattern (add above init_sim... anywhere at top level works; keep it after the constants).",
          code: `SEED_SIZE = 24
def seed_pattern(n, size=SEED_SIZE):
    """Pure numpy: U everywhere, a square of V dropped in the center."""
    u0 = np.ones((n, n), dtype=np.float32)
    v0 = np.zeros((n, n), dtype=np.float32)
    half = size // 2
    c = n // 2
    v0[c - half : c + half, c - half : c + half] = 1.0
    u0[c - half : c + half, c - half : c + half] = 0.5
    return u0, v0`,
          does: "np.ones makes a grid of 1.0s (U everywhere), np.zeros a grid of 0.0s (no V). Then numpy's slice syntax paints a 24x24 square in the center: V goes to 1.0 there, U dips to 0.5. It returns both grids — plain CPU arrays, no Taichi in sight.",
          why: "Deliberate layering: the pattern-making is pure numpy, so it can be tested without a GPU at all. Every project keeps its 'generate' step CPU-pure like this and only uploads to the GPU at the edge.",
          see: "Runs clean — the function exists, nothing calls it yet.",
          checkpoint: "No red text.",
          recovery: ["dtype=np.float32 matters — the GPU fields are f32 and the upload must match.", "The slice is v0[c - half : c + half, c - half : c + half] — same expression twice, rows then columns."] },
        { title: "Upload to the GPU", adding: "the little bridge that copies a seed from numpy into the GPU fields (add right after seed_pattern).",
          code: `def apply_seed(seed):
    u0, v0 = seed
    u.from_numpy(u0)
    v.from_numpy(v0)`,
          does: "from_numpy copies a CPU array into a GPU field, whole grid in one call. That's the entire CPU→GPU bridge.",
          why: "This tiny function is a boundary you'll respect all series long: numpy builds, fields receive. Later, .to_numpy() crosses back the other way for tests.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["from_numpy is called ON the field: u.from_numpy(u0) — field first, array in the parentheses."] },
        { title: "Show the chemicals", adding: "a new render that shows U as grayscale (replace the render kernel), and the seed call in main (replace the first two lines of main).",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([u[i, j], u[i, j], u[i, j]])
def main():
    init_sim()
    apply_seed(seed_pattern(N))`,
          does: "Each pixel now paints itself from the U concentration under it — same number in red, green, and blue makes gray: U=1 is white, U=0.5 is mid-gray. And main now builds a seed and uploads it right after the fields exist.",
          why: "Rendering is just 'turn my numbers into color' — you swapped a gradient for data with one line. This render will evolve once more, into full color, in the final chapter.",
          see: "The gradient is gone. A pale field with a small dark-gray square dead center — that's chemical V's seed, seen through U's dip to 0.5.",
          checkpoint: "White-ish window, small gray square in the middle. Beat 2.",
          recovery: ["Still the gradient — you edited a copy; there must be exactly one render kernel in the file.", "All white, no square — main must call apply_seed(seed_pattern(N)) after init_sim().", "Crash mentioning field not allocated — apply_seed must come AFTER init_sim() in main."] }
      ]
    },
    {
      id: 3, title: "Diffusion",
      build: "the spreading half of the simulation — chemicals bleeding into their neighbors.",
      beat: "The square melts — it blurs outward like ink in still water.",
      steps: [
        { title: "The neighbor question", adding: "three diffusion dials (add with the other constants, under N = 512), and a helper every cell uses to ask 'how do I compare to my neighbors?'.",
          code: `DU = 1.0
DV = 0.5
DT = 1.0
@ti.func
def laplacian(f: ti.template(), i, j):
    side = f[(i + 1) % N, j] + f[(i - 1) % N, j] + f[i, (j + 1) % N] + f[i, (j - 1) % N]
    corner = (
        f[(i + 1) % N, (j + 1) % N]
        + f[(i + 1) % N, (j - 1) % N]
        + f[(i - 1) % N, (j + 1) % N]
        + f[(i - 1) % N, (j - 1) % N]
    )
    return 0.2 * side + 0.05 * corner - f[i, j]`,
          does: "The laplacian measures how a cell differs from its surroundings: weighted neighbors (sides count 0.2, corners 0.05) minus itself. Positive means 'my neighbors have more than me — stuff will flow in'. @ti.func is a helper callable from inside kernels, and ti.template() lets one helper serve any field — we'll aim it at both u and v. The % N wraps the edges so the world has no walls.",
          why: "This one formula IS diffusion — heat spreading, ink dispersing, smoke thinning are all 'move toward your neighbors' average'. The 0.2/0.05 weights keep the spread smooth and stable in all directions; you'll meet this exact stencil again in the fluid project.",
          see: "Runs clean; the square still sits frozen.",
          checkpoint: "No red text.",
          recovery: ["@ti.func, not @ti.kernel — helpers called by kernels use func.", "Every neighbor index needs its % N — miss one and the edges will error or leak.", "DU, DV, DT live at top level with N, not inside a function."] },
        { title: "Spread, into the twin", adding: "the update kernel — every cell computes its next value into the _next twin (replace nothing; add after laplacian).",
          code: `@ti.kernel
def update():
    for i, j in u:
        u_next[i, j] = u[i, j] + DT * DU * laplacian(u, i, j)
        v_next[i, j] = v[i, j] + DT * DV * laplacian(v, i, j)`,
          does: "Each cell takes its current value and adds a step of spreading: time step DT times diffusion speed (DU or DV) times the laplacian. Crucially it writes into u_next/v_next — the current grids stay untouched while every cell reads them.",
          why: "Here's the double-buffer paying off. All 262,144 cells run at once; if they wrote into u while reading u, cells would read half-updated neighbors and the pattern would corrupt. Compute into next, swap after — burn this into your hands.",
          see: "Runs clean; still frozen — we compute the future but never adopt it.",
          checkpoint: "No red text.",
          recovery: ["Writes go to u_next and v_next — writing into u here is the classic bug this chapter exists to teach.", "V spreads at half speed: DU on the u line, DV on the v line."] },
        { title: "Adopt the future", adding: "the copy-back kernel and a step function that runs one full tick (add after update).",
          code: `@ti.kernel
def copy_back():
    for i, j in u:
        u[i, j] = u_next[i, j]
        v[i, j] = v_next[i, j]
def step():
    update()
    copy_back()`,
          does: "copy_back moves the computed future into the live grids — the swap that completes the double-buffer. step() is plain Python glue: one tick = compute, then adopt.",
          why: "update/copy_back/step is the exact heartbeat every grid simulation in this series will have. Different physics inside update; same skeleton.",
          see: "Runs clean; one more line and it moves.",
          checkpoint: "No red text.",
          recovery: ["copy_back is a kernel (@ti.kernel); step is plain Python — no decorator.", "step calls update() first, copy_back() second — adopt after compute."] },
        { title: "Let it run", adding: "one line inside main's loop — tick the simulation every frame (add right after the event-handling block, before render()).",
          code: `        step()`,
          does: "Every frame now advances the chemicals one tick before drawing them.",
          why: "This is the moment a picture becomes a simulation — the loop stops redrawing a frozen state and starts evolving one.",
          see: "The gray square slowly melts at the edges and blurs outward, like ink in still water. Give it half a minute; V spreads slower than U (DV is half of DU).",
          checkpoint: "The square visibly blurs over time. Beat 3.",
          recovery: ["Nothing moves — step() goes inside the while loop, indented to match render().", "It must sit before render() so each frame draws the fresh state."] }
      ]
    },
    {
      id: 4, title: "Reaction",
      build: "the second half of Gray-Scott — U feeds in, V eats U, and patterns come alive.",
      beat: "Coral grows. The blur becomes a living, spreading pattern.",
      steps: [
        { title: "The reaction", adding: "the chemistry inside update (replace the whole update kernel).",
          code: `@ti.kernel
def update(feed: ti.f32, kill: ti.f32):
    for i, j in u:
        reaction = u[i, j] * v[i, j] * v[i, j]
        u_next[i, j] = u[i, j] + DT * (DU * laplacian(u, i, j) - reaction + feed * (1.0 - u[i, j]))
        v_next[i, j] = v[i, j] + DT * (DV * laplacian(v, i, j) + reaction - (feed + kill) * v[i, j])`,
          does: "reaction = u·v·v is the heart: one U meets two Vs and becomes V — so the reaction term leaves u's line with a minus and enters v's line with a plus. feed tops U back up toward 1.0 everywhere; kill drains V away. The kernel now takes feed and kill as typed arguments (ti.f32) so we can change them without recompiling.",
          why: "Diffusion alone always flattens to gray soup. Add this reaction and the two forces fight — spreading versus consuming — and the fight is what draws coral, cells, and worms. Two lines of algebra, endless pattern.",
          see: "Runs clean — but main still calls step() with no arguments, so don't launch expecting patterns yet; the plumbing catches up over the next steps.",
          checkpoint: "python3 -c \"import gray_scott\" style silence: the file parses. (A full run would crash at step() — expected, fix incoming.)",
          recovery: ["Kernel arguments need types: update(feed: ti.f32, kill: ti.f32).", "Signs carry the story: minus reaction on the u line, plus reaction on the v line.", "The kill term is (feed + kill) * v[i, j] — both rates drain V."] },
        { title: "Thread it through", adding: "feed and kill flowing through step (replace the step function).",
          code: `def step(feed, kill):
    update(feed, kill)
    copy_back()`,
          does: "step now carries the two dials down into the kernel.",
          why: "Plain-Python glue can just pass values along — only the kernel boundary needs type labels.",
          see: "Parses clean; main's bare step() is still stale for one more step.",
          checkpoint: "No syntax errors.",
          recovery: ["Both parameters, both places: def step(feed, kill) and update(feed, kill)."] },
        { title: "The two dials", adding: "how many ticks per frame, and the first F/k setting (add with the constants, under DT = 1.0).",
          code: `SUBSTEPS = 12
FEED = 0.0545
KILL = 0.0620`,
          does: "SUBSTEPS: one tick per frame is glacial, so we'll run 12 ticks per drawn frame. FEED and KILL are the famous F and k of Gray-Scott — this pair grows coral.",
          why: "Everything about what grows lives in these two numbers. Nudge the third decimal and coral becomes cells becomes worms — next chapter puts five such pairs on number keys.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Top level, with the other constants — not inside a function."] },
        { title: "Turn the crank", adding: "the substep loop in main (replace the bare step() line).",
          code: `        for _ in range(SUBSTEPS):
            step(FEED, KILL)`,
          does: "Twelve simulation ticks per drawn frame, each fed the coral dials.",
          why: "Simulation speed and frame rate are now independent — a knob you'll tune in every project (the GPU barely notices 12 ticks).",
          see: "Launch it and wait about twenty seconds: the blurred square's rim crinkles, fingers reach out, and coral growth spreads toward the edges. This is the payoff of the whole project — algebra becoming biology.",
          checkpoint: "A growing coral-like pattern fills the window. Beat 4 — the big one.",
          recovery: ["Both lines replace the old step() line, at the same indentation it had.", "Blows up to solid white/black — check chapter 4 step 1's signs and the 0.2/0.05 laplacian weights from chapter 3.", "Nothing grows, just blur — SUBSTEPS loop must call step(FEED, KILL), not step()."] }
      ]
    },
    {
      id: 5, title: "Presets",
      build: "five famous F/k pairs on number keys, and fresh random seeds on demand.",
      beat: "Five different life-forms on number keys, and R plants new seeds.",
      steps: [
        { title: "The pattern zoo", adding: "the preset table (replace the FEED and KILL lines), and a place to remember the current choice (add preset = 0 in main, right after the gui line).",
          code: `PRESETS = [
    ("coral", 0.0545, 0.0620),
    ("mitosis", 0.0367, 0.0649),
    ("worms", 0.0780, 0.0610),
    ("waves", 0.0140, 0.0450),
    ("solitons", 0.0300, 0.0600),
]
    preset = 0`,
          does: "A plain Python list of (name, F, k) triples — five settings from the Gray-Scott map, each growing a different creature. preset = 0 is main's memory of which one is live.",
          why: "The dials become data. Once settings are entries in a table, 'switch pattern' is just 'change an index' — the cheapest UI there is.",
          see: "Parses clean; main still mentions FEED/KILL for one more step, so don't run yet.",
          checkpoint: "No syntax errors.",
          recovery: ["Delete the old FEED = / KILL = lines — the table replaces them.", "preset = 0 lives INSIDE main (indented), after gui = ti.GUI(...); the PRESETS table lives at top level."] },
        { title: "Read the table", adding: "the lookup in main's loop (replace the substep loop).",
          code: `        name, feed, kill = PRESETS[preset]
        for _ in range(SUBSTEPS):
            step(feed, kill)`,
          does: "Each frame unpacks the current preset into name, feed, kill and runs the ticks with those dials — lowercase feed/kill now, values straight from the table.",
          why: "The loop reads whatever preset points at, so switching creatures mid-run is now possible the moment we wire keys to it.",
          see: "Runs again! Same coral as before — preset 0 — but now table-driven.",
          checkpoint: "Coral grows exactly like chapter 4's ending.",
          recovery: ["NameError: FEED — this block replaces the old loop; no capitals remain in main.", "All three lines share the indentation of the old step line's block."] },
        { title: "Seeds with personality", adding: "random extra seed spots (replace the whole seed_pattern function).",
          code: `def seed_pattern(n, size=SEED_SIZE, rng_seed=0, extra_spots=4):
    """Pure numpy: U everywhere, plus a center square of V and a few random spots."""
    u0 = np.ones((n, n), dtype=np.float32)
    v0 = np.zeros((n, n), dtype=np.float32)
    half = size // 2
    c = n // 2
    v0[c - half : c + half, c - half : c + half] = 1.0
    u0[c - half : c + half, c - half : c + half] = 0.5
    rng = np.random.default_rng(rng_seed)
    for _ in range(extra_spots):
        x, y = rng.integers(half, n - half, size=2)
        v0[x - half : x + half, y - half : y + half] = 1.0
        u0[x - half : x + half, y - half : y + half] = 0.5
    return u0, v0
`,
          does: "np.random.default_rng(rng_seed) makes a random generator with a controllable seed; four extra V squares land at random spots. Same rng_seed, same spots — the randomness is reproducible.",
          why: "Seeded randomness is the series' standing rule: tests replay the exact same 'random' world. And multiple seed points make patterns collide, which is where the best structures happen.",
          see: "Launch: coral now grows from five islands at once and the fronts weave together where they meet.",
          checkpoint: "Five growth sites instead of one.",
          recovery: ["Still one square — main calls seed_pattern(N), which now defaults to extra_spots=4; make sure you replaced the old function rather than adding a second one.", "rng.integers(half, n - half, size=2) hands back both x and y in one call."] },
        { title: "Keys: switch and reseed", adding: "number keys and R (replace the event-handling block in main).",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key in "12345":
                preset = int(e.key) - 1`,
          does: "Two new branches: R builds a brand-new seed with a random rng_seed and uploads it; keys 1-5 move preset to a new row of the table (int(e.key) - 1 turns key '3' into index 2).",
          why: "Interactivity in Taichi apps is exactly this cheap — the sim never pauses; you're just changing the numbers the next frame reads.",
          see: "Press 2 — growth turns into dividing cell-like blobs (mitosis). 3 — fat worms. 4 — drifting waves. 5 — solitary spots. R — fresh islands any time. Tip: press R after switching, some presets prefer a fresh start.",
          checkpoint: "All five keys change the creature; R replants. Beat 5.",
          recovery: ["Keys dead — this block must be INSIDE the while loop, and e.key comparisons use quotes: \"r\", \"12345\".", "Crash on R — the reseed line ends with three closing parens: seed_pattern(...) inside apply_seed(...)."] }
      ]
    },
    {
      id: 6, title: "Paint with the mouse",
      build: "a brush that injects chemical V wherever you drag.",
      beat: "Wherever you drag, the pattern erupts.",
      steps: [
        { title: "The splat kernel", adding: "a brush size (with the constants), and a kernel that stamps V in a circle (add near splat's siblings, after step).",
          code: `BRUSH_RADIUS = 8.0
@ti.kernel
def splat(x: ti.f32, y: ti.f32, radius: ti.f32):
    for i, j in v:
        dx = i - x * N
        dy = j - y * N
        if dx * dx + dy * dy < radius * radius:
            v[i, j] = 1.0
            u[i, j] = 0.5`,
          does: "The mouse arrives as coordinates from 0 to 1, so x * N maps it onto the grid. Every cell checks 'am I inside the circle?' with the no-square-root trick (compare squared distances), and cells inside become fresh seed: V up, U down.",
          why: "Notice there's no loop over a circle — all cells check themselves in parallel. 'Everyone asks am I affected?' is how GPUs do brushes, explosions, and area effects; you'll reuse this shape constantly.",
          see: "Runs clean; not wired to the mouse yet.",
          checkpoint: "No red text.",
          recovery: ["All three arguments typed: x: ti.f32, y: ti.f32, radius: ti.f32.", "Squared compare: dx * dx + dy * dy < radius * radius — no sqrt anywhere."] },
        { title: "Wire the mouse", adding: "the drag check in main's loop (add right after the event block, before the preset lookup).",
          code: `        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            splat(mx, my, BRUSH_RADIUS)`,
          does: "is_pressed is true the whole time the left button is down (not just on the click), get_cursor_pos hands back that 0-to-1 position, and splat stamps the circle there — every frame while you drag.",
          why: "Press-and-hold plus a per-frame stamp is what makes it feel like a brush instead of a button.",
          see: "Drag across the window — a line of eruptions blooms behind your cursor and gets absorbed into the pattern. Try drawing through empty space with waves (4) selected.",
          checkpoint: "Drawing works while dragging. Beat 6.",
          recovery: ["Only stamps once per click — use gui.is_pressed(ti.GUI.LMB), not an event-loop branch.", "Eruptions mirror-flipped from your hand — the order is mx, my = gui.get_cursor_pos() and splat(mx, my, ...)."] }
      ]
    },
    {
      id: 7, title: "Color",
      build: "palette-mapped color, palette cycling, and a HUD that names what you're watching.",
      beat: "The pattern catches fire — palettes, and a HUD naming the creature.",
      steps: [
        { title: "Three palettes", adding: "palette data at top level (after PRESETS), and one more field placeholder (replace the placeholder block to add pal_stops).",
          code: `N_STOPS = 5
PALETTES = np.array(
    [
        [[0.00, 0.00, 0.05], [0.10, 0.00, 0.30], [0.80, 0.20, 0.10], [1.00, 0.70, 0.10], [1.00, 1.00, 0.90]],
        [[0.00, 0.02, 0.08], [0.00, 0.20, 0.45], [0.00, 0.60, 0.70], [0.40, 0.90, 0.85], [0.95, 1.00, 1.00]],
        [[0.02, 0.00, 0.05], [0.25, 0.00, 0.40], [0.10, 0.55, 0.20], [0.70, 0.95, 0.20], [1.00, 1.00, 0.75]],
    ],
    dtype=np.float32,
)
u = None
v = None
u_next = None
v_next = None
pixels = None
pal_stops = None`,
          does: "Each palette is 5 RGB stops — a color journey from 'no V' to 'lots of V': ember (black→red→white-hot), ocean (deep blue→foam), toxic (purple→acid green). pal_stops will be their home on the GPU.",
          why: "Grayscale showed data; palettes tell stories. Mapping a number through color stops is THE scientific-visualization move — every heat map you've ever seen is this.",
          see: "Runs clean, still grayscale.",
          checkpoint: "No red text.",
          recovery: ["Count the brackets: 3 palettes x 5 stops x 3 numbers, and dtype=np.float32 at the end.", "pal_stops = None joins the other placeholders at top level."] },
        { title: "Upload the palettes", adding: "pal_stops allocation and upload inside init_sim (replace the whole init_sim one last time).",
          code: `def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global u, v, u_next, v_next, pixels, pal_stops
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    u = ti.field(ti.f32, shape=(N, N))
    v = ti.field(ti.f32, shape=(N, N))
    u_next = ti.field(ti.f32, shape=(N, N))
    v_next = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))
    pal_stops = ti.Vector.field(3, ti.f32, shape=(len(PALETTES), N_STOPS))
    pal_stops.from_numpy(PALETTES)`,
          does: "A field of 3-vectors shaped (3 palettes, 5 stops), filled from the numpy table immediately — palettes never change, so one upload at startup is all they need.",
          why: "Same one-way bridge as the seeds: numpy authored it, from_numpy ships it, the GPU owns it from then on.",
          see: "Runs clean.",
          checkpoint: "Still grayscale, no red text.",
          recovery: ["The global line gains pal_stops at the end.", "Shape is (len(PALETTES), N_STOPS) — palettes first, stops second."] },
        { title: "Color mapping", adding: "the final render — V mapped through the palette (replace the render kernel one last time).",
          code: `@ti.kernel
def render(pal: ti.i32):
    for i, j in pixels:
        t = ti.math.clamp(v[i, j] / 0.4, 0.0, 1.0)
        x = t * (N_STOPS - 1)
        s = ti.min(int(x), N_STOPS - 2)
        f = x - s
        pixels[i, j] = pal_stops[pal, s] * (1.0 - f) + pal_stops[pal, s + 1] * f`,
          does: "Each pixel turns its V (usually 0 to ~0.4) into t from 0 to 1, finds where t lands between the 5 stops (s is the stop below, f how far toward the next), and blends the two stops. That blend — a*(1-f) + b*f — is linear interpolation, 'lerp'.",
          why: "Lerp between stops is the universal gradient trick; you'll lerp positions, colors, and forces all series long. The int(x) clamp to N_STOPS - 2 keeps s+1 legal at the very top of the range.",
          see: "Parses clean — but main still calls render() with no argument; two small steps to wire it.",
          checkpoint: "File parses; don't launch yet.",
          recovery: ["render now takes pal: ti.i32 — an integer, not f32.", "s uses ti.min(int(x), N_STOPS - 2) so the s + 1 lookup can't fall off the table."] },
        { title: "The P key", adding: "palette state and the cycle key (replace preset = 0 with the two state lines, and replace the event block to add the P branch).",
          code: `    preset = 0
    pal = 0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_pattern(N, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "p":
                pal = (pal + 1) % len(PALETTES)
            elif e.key in "12345":
                preset = int(e.key) - 1`,
          does: "pal remembers the current palette; the P branch bumps it and the % len(PALETTES) wraps 2 back around to 0 — an endless cycle through ember, ocean, toxic.",
          why: "The wrap-with-% pattern is the standard 'cycle through options' one-liner — you used % for grid edges in chapter 3; here it wraps a menu.",
          see: "Parses clean; last step wires render(pal) and the HUD.",
          checkpoint: "No syntax errors.",
          recovery: ["pal = 0 sits with preset = 0 in main's setup, before the while loop.", "The p branch goes before the \"12345\" branch, matching the order shown."] },
        { title: "Light it up", adding: "the final draw block — colored render plus a HUD (replace the render()/set_image/show lines at the bottom of the loop).",
          code: `        render(pal)
        gui.set_image(pixels)
        gui.text(f"{name}  F={feed:.4f} k={kill:.4f}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1-5] preset  [r] reseed  [p] palette  paint with mouse", (0.02, 0.94), color=0xAAAAAA)
        gui.show()`,
          does: "render(pal) paints through the chosen palette. gui.text overlays live text at a 0-to-1 position: the top line names the creature and its exact F/k (an f-string with :.4f formatting), the dim line below lists every control.",
          why: "A HUD is the difference between a demo and an instrument — you can now SEE which dials produced the pattern you're watching, and read F/k off the screen to hunt your own settings.",
          see: "Black window igniting into ember-red coral. P cycles ocean and toxic. The HUD names everything. Play: mitosis in ocean, worms in toxic, paint with the mouse. This is your reaction-diffusion instrument — project 01 complete.",
          checkpoint: "Color, palettes cycling, HUD reading out the preset. Final beat — the whole build.",
          recovery: ["TypeError about render arguments — main must call render(pal) now, not render().", "HUD missing — gui.text lines go between set_image and show, y positions 0.98 and 0.94.", "NameError: name — the preset lookup line (chapter 5) must still be in the loop above this block."] }
      ]
    }
  ]
};

// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["05-terrain-erosion"] = {
  project: "05-terrain-erosion",
  title: "Terrain Erosion",
  pitch: "Grow fractal mountains, then let rain, rivers, and deep time carve them into canyons.",
  tier: "medium",
  language: "Python",
  file: "terrain.py",
  chapters: [
    {
      id: 1, title: "Mountains from noise",
      build: "a fractal heightmap — octaves of noise, each finer and fainter — shown as raw grayscale.",
      beat: "A fractal mountain range, in raw grayscale height.",
      steps: [
        { title: "Load the GPU toolkit", adding: "the docstring and the Taichi import.",
          code: `"""Terrain erosion: fractal mountains weathered by rain, rivers, and time."""
import taichi as ti`,
          does: "Project 05 closes Arc 1 by combining both halves of the curriculum: procedural generation (like lightning) builds the mountains; grid simulation (like the fluids) weathers them.",
          why: "Generation gives you a world; simulation gives it a history. Terrain that's been rained on for ten thousand virtual years looks TRUE in a way pure noise never does.",
          see: "Runs clean.",
          checkpoint: "python3 terrain.py returns silently.",
          recovery: ["Venv first, as always."] },
        { title: "The heightmap", adding: "the grid size, placeholders, and init_sim with the height field.",
          code: `N = 512
h = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global h, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    h = ti.field(ti.f32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "h is elevation, 0 (sea level) to 1 (highest peak), one number per cell — a heightmap, the standard way games and GIS store terrain.",
          why: "A whole landscape in one scalar field. Every kernel in this project reads or rewrites h; everything you see is a portrait of it.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The standard opening — fifth time's a charm."] },
        { title: "Bilerp, the numpy edition", adding: "numpy (above import taichi) and a resize function that scales a small array up smoothly.",
          code: `import numpy as np
def resize_bilinear(a, n):
    """Pure numpy: smoothly resize a small square array up to n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1.0 - f)[:, None] + a[i1] * f[:, None]
    a = a[:, i0] * (1.0 - f)[None, :] + a[:, i1] * f[None, :]
    return a`,
          does: "For every output position, find the two source cells it lands between (i0 and i1), and blend by the fractional distance f. Done once along rows, once along columns — 2D bilerp as two 1D passes, all vectorized.",
          why: "This IS project 02's bilerp, wearing numpy clothes: same floor, same fraction, same blend. Seeing one idea in two dialects is how it becomes permanently yours. The separable-passes trick (rows then columns) is a classic you'll reuse for blurs.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The [:, None] and [None, :] reshape f so it broadcasts down rows first, then across columns.", "np.minimum(i0 + 1, m - 1) is the edge clamp — same job as % N, different boundary policy."] },
        { title: "Octaves", adding: "the terrain generator — layers of noise at doubling resolution and shrinking strength — plus the upload bridge.",
          code: `def fbm_terrain(n, rng_seed=0, octaves=7, roughness=0.55):
    """Pure numpy: fractal terrain — octaves of noise, each finer and fainter."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n, n), dtype=np.float32)
    amp = 1.0
    res = 4
    for _ in range(octaves):
        layer = rng.uniform(-1.0, 1.0, size=(res, res)).astype(np.float32)
        out += amp * resize_bilinear(layer, n)
        amp *= roughness
        res *= 2
    out -= out.min()
    out /= out.max()
    return out.astype(np.float32)
def apply_seed(h0):
    h.from_numpy(h0)`,
          does: "Seven passes: a 4x4 random grid stretched smooth (the continents), then 8x8 at 55% strength (the ranges), 16x16 fainter still (the ridges)… down to pixel-scale roughness. Sum, then normalize to [0, 1].",
          why: "This layering is fractal Brownian motion — fBm — and it's THE workhorse of procedural worlds: clouds, planets, marble, galaxies all start here. Same spirit as lightning's midpoint displacement (big features first, detail recursively finer), built with octave sums instead of recursion.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["amp *= roughness AND res *= 2 each pass — shrink the voice, sharpen the detail.", "Normalize at the end: subtract min THEN divide by max."] },
        { title: "See the land", adding: "a raw grayscale render and the standard main loop.",
          code: `@ti.kernel
def render():
    for i, j in pixels:
        hh = h[i, j]
        pixels[i, j] = ti.Vector([hh, hh, hh])
def main():
    init_sim()
    apply_seed(fbm_terrain(N))
    gui = ti.GUI("Terrain Erosion — taichi-academy", res=(N, N))
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "Height as brightness: white peaks, black basins.",
          why: "Raw data first, pretty later — you can already sanity-check the fractal (big blobs with fine grain on top) before any lighting flatters it.",
          see: "A cloudy grayscale relief map: bright ridges, dark valley systems, detail at every scale you zoom your eyes into.",
          checkpoint: "Fractal grayscale terrain. Beat 1.",
          recovery: ["Flat gray — check the normalize lines in fbm_terrain.", "Blocky squares — resize_bilinear's two blend passes aren't both running."] }
      ]
    },
    {
      id: 2, title: "Light on the land",
      build: "hillshading from the height gradient, and hand-tuned elevation colors.",
      beat: "Sunlit slopes, green valleys, snowy peaks.",
      steps: [
        { title: "Hillshade", adding: "a relief exaggeration dial and the classic cartographer's lighting (replace render).",
          code: `RELIEF = 300.0
@ti.kernel
def render():
    for i, j in pixels:
        ip = ti.min(i + 1, N - 1)
        jp = ti.min(j + 1, N - 1)
        dhdx = (h[ip, j] - h[i, j]) * RELIEF
        dhdy = (h[i, jp] - h[i, j]) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.35 + 0.65 * normal.dot(light), 0.0, 1.0)
        pixels[i, j] = ti.Vector([shade, shade, shade])`,
          does: "Each cell measures its slope (height difference to the next cell, exaggerated 300x because heights span 0-1 over 512 cells), builds a surface normal from it, and dots it against a fixed sun direction. Facing the sun → bright; facing away → shadow. The 0.35 floor is ambient light so shadows stay readable.",
          why: "normal-dot-light is THE lighting equation — the core of every shader you'll ever write, from this hillshade to the mandelbulb. Meet it here in 2D where you can see exactly what it does.",
          see: "The land leaps into 3D: sun from the lower-left, every ridge casting soft shade. Same data as last step — only the LIGHTING changed.",
          checkpoint: "Shaded relief. It looks touchable.",
          recovery: ["Normal is [-dhdx, -dhdy, 1.0] — minus signs on both slopes.", ".normalized() on BOTH vectors — a dot product only means cosine when lengths are 1.", "Inverted light (valleys glowing) — the light vector's z must be positive."] },
        { title: "Paint by elevation", adding: "a tiny blend helper and three color bands stacked by height (replace render).",
          code: `@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t
@ti.kernel
def render():
    for i, j in pixels:
        ip = ti.min(i + 1, N - 1)
        jp = ti.min(j + 1, N - 1)
        dhdx = (h[ip, j] - h[i, j]) * RELIEF
        dhdy = (h[i, jp] - h[i, j]) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.35 + 0.65 * normal.dot(light), 0.0, 1.0)
        hh = h[i, j]
        c = band(ti.Vector([0.76, 0.70, 0.50]), ti.Vector([0.30, 0.55, 0.25]), hh, 0.05, 0.35)
        c = band(c, ti.Vector([0.45, 0.42, 0.40]), hh, 0.45, 0.75)
        c = band(c, ti.Vector([0.95, 0.95, 0.98]), hh, 0.78, 0.92)
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)`,
          does: "band is a reusable lerp-with-window: below lo you're all c0, above hi all c1, smooth between. Chained three times: sand→grass, grass→rock, rock→snow. Color times shade gives lit, painted land.",
          why: "Chained bands are project 01's palette stops, written as a function — altitude zones are a palette indexed by height instead of chemistry. Multiplying color by shade (not adding) is how lighting works everywhere.",
          see: "A hand-drawn atlas: tan lowlands, green mid-slopes, gray crags, white summits, all sunlit.",
          checkpoint: "Colored, shaded mountains.",
          recovery: ["Each band call feeds the previous c in as its c0 — they chain.", "c * shade at the end — multiply, don't add."] },
        { title: "Reroll the world", adding: "the R key (replace the event block).",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))`,
          does: "A fresh seed, a fresh world — the whole generator reruns in a blink.",
          why: "Instant rerolls are the joy of procedural work: seven octaves of noise means every tap is a place nobody has ever seen.",
          see: "Tap R and watch continents you'll never visit twice.",
          checkpoint: "R makes new worlds. Beat 2.",
          recovery: ["The reseed line ends with three closing parens."] }
      ]
    },
    {
      id: 3, title: "Rain flows downhill",
      build: "a water layer, outflow flux to lower neighbors, and the gather trick that keeps it exact.",
      beat: "Rivers thread the valleys; lakes fill the bowls.",
      steps: [
        { title: "Water plumbing", adding: "water fields, a 4-slot flux field, and three little direction tables (placeholders, global line, field lines, and the tuples after apply_seed).",
          code: `h = None
w = None
w_next = None
flux = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global h, w, w_next, flux, pixels
    w = ti.field(ti.f32, shape=(N, N))
    w_next = ti.field(ti.f32, shape=(N, N))
    flux = ti.Vector.field(4, ti.f32, shape=(N, N))
DI = (1, -1, 0, 0)
DJ = (0, 0, 1, -1)
OPP = (1, 0, 3, 2)`,
          does: "w is standing water depth. flux is new: each cell stores FOUR numbers — how much water it's sending to each neighbor this tick (+x, -x, +y, -y). DI/DJ turn a direction index into an offset; OPP maps a direction to its reverse (0↔1, 2↔3).",
          why: "Publishing your outflows instead of directly pushing water is the key design move of this whole project — hold that thought for two steps and you'll see why.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["flux is ti.Vector.field(4, ...) — four channels, one per neighbor.", "The tuples live at top level, plain Python — kernels will read them via ti.static."] },
        { title: "Rain", adding: "two weather dials and the simplest kernel in the project (add after the tuples).",
          code: `RAIN = 0.0002
EVAP = 0.004
@ti.kernel
def rain():
    for i, j in w:
        w[i, j] += RAIN`,
          does: "Every cell gains a trace of water per tick. Evaporation (used shortly) will take a fraction back.",
          why: "Rain adds a constant; evaporation removes a proportion — so water level self-balances where those cancel. You built this same source/sink equilibrium in fire (burn vs cool); here it's the water table.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["RAIN is tiny (2e-4) — rivers come from ACCUMULATION, not downpours."] },
        { title: "Declare your outflows", adding: "the flux kernel — each cell decides where its water wants to go (add after rain).",
          code: `@ti.kernel
def compute_flux():
    for i, j in flux:
        total_h = h[i, j] + w[i, j]
        f = ti.Vector([0.0, 0.0, 0.0, 0.0])
        total = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                d = total_h - (h[ni, nj] + w[ni, nj])
                if d > 0.0:
                    f[k] = d
                    total += d
        scale = 0.0
        if total > 1e-9:
            scale = ti.min(w[i, j], 0.5 * total) / total
        flux[i, j] = f * scale`,
          does: "Water surface is ground PLUS water (h + w) — a lake's top is flat even when its floor isn't. Each cell compares its surface to all four neighbors, wants to send water down every drop, then scales all four sends so it never ships more than it has (min with w) and never overshoots past level (the 0.5). ti.static(range(4)) unrolls the little loop at compile time so DI/DJ stay plain tuples.",
          why: "Comparing h + w, not h, is what makes lakes fill and rivers route around obstacles. And the 0.5 is your old friend from Gray-Scott's stability lesson: move half the difference, never overshoot into oscillation.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Compare (h + w) on BOTH sides of the difference.", "Bounds check with if 0 <= ni < N — this world has edges, not wrap (mountains shouldn't drain into the opposite ocean).", "The limiter: ti.min(w[i, j], 0.5 * total) / total."] },
        { title: "The gather", adding: "the water mover and its copy-back — reading neighbors' fluxes instead of writing into them (add after compute_flux).",
          code: `@ti.kernel
def move_water():
    for i, j in w:
        inflow = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                inflow += flux[ni, nj][OPP[k]]
        w_next[i, j] = (w[i, j] - flux[i, j].sum() + inflow) * (1.0 - EVAP)
@ti.kernel
def copy_wet():
    for i, j in w:
        w[i, j] = w_next[i, j]`,
          does: "Each cell asks its neighbors 'how much did you send ME?' — my neighbor in direction k sent it through their OPP[k] slot. New water = old, minus everything I shipped, plus everything addressed to me, times evaporation's keep-rate.",
          why: "Here's the payoff of publishing fluxes. The obvious alternative — each cell ADDING water into its neighbors — has thousands of threads writing the same cells at once, and the additions happen in random order: answers change run to run. Reading is always safe; writing is the hazard. 'Publish, then gather' converts scatter into read-only — exact, deterministic, and it's why this project's tests can demand water be conserved to a part in a thousand.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Inflow reads flux[ni, nj][OPP[k]] — the NEIGHBOR'S field, THEIR slot pointing back at you.", "Evaporation multiplies the whole balance: * (1.0 - EVAP)."] },
        { title: "Weather, running", adding: "the tick conductor, a rain toggle flag (after the gui line), and the guarded call in the loop (after the event block).",
          code: `def step():
    rain()
    compute_flux()
    move_water()
    copy_wet()
    raining = True
        if raining:
            step()`,
          does: "Rain falls, cells declare flows, water moves, the future is adopted. The raining flag lets you freeze the weather to inspect the land.",
          why: "Four kernels, strict order — declare before you move, move before you adopt. Every multi-kernel sim you write from here has this same 'phases of a tick' shape.",
          see: "Nothing visible YET — water is flowing but the renderer is two steps behind. Patience: the reveal is worth it.",
          checkpoint: "Runs clean.",
          recovery: ["Three homes: step() at top level, raining = True in main's setup, the if-block inside the loop."] },
        { title: "Pause and reroll, properly", adding: "the space toggle, and R learning to dump the water with the old world (replace the event block).",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))
                w.fill(0.0)
            elif e.key == ti.GUI.SPACE:
                raining = not raining`,
          does: "w.fill(0.0) blanks the water field from Python in one call (no kernel needed for a plain fill), so new mountains don't inherit the old world's lakes.",
          why: "Reset hygiene again: R must reset ALL the state that belongs to the old world. Forgetting a field on reset is one of the great classic sim bugs — old water on new mountains would teleport lakes onto peaks.",
          see: "Space freezes the (still invisible) weather; R rolls dry new worlds.",
          checkpoint: "No red text.",
          recovery: ["w.fill(0.0) joins the r branch, after the reseed line."] },
        { title: "See the water", adding: "a visibility dial and the wet tint in render (replace render).",
          code: `WATER_VIS = 0.002
@ti.kernel
def render():
    for i, j in pixels:
        ip = ti.min(i + 1, N - 1)
        jp = ti.min(j + 1, N - 1)
        dhdx = (h[ip, j] - h[i, j]) * RELIEF
        dhdy = (h[i, jp] - h[i, j]) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.35 + 0.65 * normal.dot(light), 0.0, 1.0)
        hh = h[i, j]
        c = band(ti.Vector([0.76, 0.70, 0.50]), ti.Vector([0.30, 0.55, 0.25]), hh, 0.05, 0.35)
        c = band(c, ti.Vector([0.45, 0.42, 0.40]), hh, 0.45, 0.75)
        c = band(c, ti.Vector([0.95, 0.95, 0.98]), hh, 0.78, 0.92)
        wet = ti.math.clamp(w[i, j] / WATER_VIS * 0.2, 0.0, 0.85)
        c = c * (1.0 - wet) + ti.Vector([0.15, 0.35, 0.70]) * wet
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)`,
          does: "One new blend before the lighting: mix the land color toward lake-blue by water depth, capped at 85% so deep water still shows shading.",
          why: "Blend-toward (lerp) versus add-on-top (glow): water COVERS land, smoke ADDED light. Choosing the compositing op that matches the physics is a renderer's whole job.",
          see: "Give it a minute of rain: blue threads gather in every valley, join into rivers, and pool into lakes wherever the land makes a bowl. Tap space to freeze and admire the drainage network.",
          checkpoint: "Rivers and lakes in the valleys. Beat 3.",
          recovery: ["The wet blend goes BEFORE c * shade — water is a surface, light comes last.", "No visible water — it accumulates; wait 30+ seconds or check RAIN's value."] }
      ]
    },
    {
      id: 4, title: "The river carves",
      build: "sediment: flowing water picks up rock, carries it, and drops it where the flow slows.",
      beat: "Channels cut deeper; sediment fans spread below.",
      steps: [
        { title: "Three erosion dials and a cargo field", adding: "capacity/erode/deposit rates, and the suspended-sediment field with its twin.",
          code: `KC = 1.0
KE = 0.05
KD = 0.05
h = None
w = None
w_next = None
s = None
s_next = None
flux = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global h, w, w_next, s, s_next, flux, pixels
    s = ti.field(ti.f32, shape=(N, N))
    s_next = ti.field(ti.f32, shape=(N, N))`,
          does: "s is rock in transit — dissolved mountain riding the water. KC sets how much cargo flowing water can hold, KE how fast it erodes when under-loaded, KD how fast it drops cargo when over-loaded.",
          why: "These three dials ARE the geology. We tuned them the honest way — our first values (KC=30, KE=0.5) dissolved the entire mountain range into mud in three hundred ticks. Erosion operates on whispers: small rates, deep time.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The ritual once more: placeholders, global line, two field lines after w_next."] },
        { title: "Pick up, put down", adding: "the erosion kernel and its slot in step (add after rain's flux; replace step).",
          code: `@ti.kernel
def erode_deposit():
    for i, j in h:
        flow = flux[i, j].sum()
        cap = KC * flow
        if s[i, j] < cap:
            amount = ti.min(KE * (cap - s[i, j]), h[i, j])
            h[i, j] -= amount
            s[i, j] += amount
        else:
            amount = KD * (s[i, j] - cap)
            h[i, j] += amount
            s[i, j] -= amount
def step():
    rain()
    compute_flux()
    erode_deposit()
    move_water()
    copy_wet()`,
          does: "Capacity is proportional to flow through the cell. Under capacity: dissolve some ground into cargo (never more ground than exists — the ti.min). Over capacity: drop cargo back as new ground. Every unit that leaves h enters s and vice versa — rock is conserved, only relocated.",
          why: "This under/over-capacity seesaw is the entire secret of river shapes: fast water (steep, concentrated flow) digs; slow water (flat deltas, lake floors) builds. Canyons and floodplains are the same rule with opposite signs.",
          see: "Runs clean; carving has begun invisibly.",
          checkpoint: "No red text.",
          recovery: ["erode_deposit slots between compute_flux and move_water — it needs fresh fluxes and pre-move water.", "The ti.min(..., h[i, j]) guard: you cannot erode ground that isn't there."] },
        { title: "Cargo rides the current", adding: "sediment transport (gather again), its line in copy_wet, its slot in step, and s joining the R reset (replace all four).",
          code: `@ti.kernel
def move_sediment():
    for i, j in s:
        kept = s[i, j]
        if w[i, j] > 1e-9:
            kept = s[i, j] * (1.0 - flux[i, j].sum() / w[i, j])
        arriving = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                if w[ni, nj] > 1e-9:
                    arriving += s[ni, nj] * flux[ni, nj][OPP[k]] / w[ni, nj]
        s_next[i, j] = kept + arriving
@ti.kernel
def copy_wet():
    for i, j in w:
        w[i, j] = w_next[i, j]
        s[i, j] = s_next[i, j]
def step():
    rain()
    compute_flux()
    erode_deposit()
    move_water()
    move_sediment()
    copy_wet()
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(fbm_terrain(N, rng_seed=np.random.randint(1_000_000)))
                w.fill(0.0)
                s.fill(0.0)
            elif e.key == ti.GUI.SPACE:
                raining = not raining`,
          does: "Sediment moves as a passenger: if a third of my water left through some direction, a third of my cargo went with it. Same gather pattern as move_water — read the neighbors' fluxes, scale by THEIR water, sum what's addressed to me. And R now dumps the cargo too.",
          why: "The fraction flux/w converts water bookkeeping into cargo bookkeeping — one idea, reused. Guard every division: cells can be bone dry, and w > 1e-9 checks are the difference between a sim and a NaN generator.",
          see: "Let it run a few minutes: river channels visibly darken and deepen; where they empty into lakes, pale fans of dropped sediment build out. Freeze with space and compare slopes to a fresh R world.",
          checkpoint: "Carved channels and sediment fans. Beat 4.",
          recovery: ["TWO dryness guards — your own water for kept, each neighbor's for arriving.", "copy_wet adopts both w and s now; step gains move_sediment after move_water."] }
      ]
    },
    {
      id: 5, title: "Deep time",
      build: "thermal weathering — steep cliffs shed scree — plus the HUD that finishes Arc 1.",
      beat: "Cliffs shed scree — the world weathers itself.",
      steps: [
        { title: "The angle of repose", adding: "two dials, a height double-buffer, and the talus kernel with its copy-back.",
          code: `TALUS = 0.008
THERMAL_RATE = 0.25
h = None
h_next = None
w = None
w_next = None
s = None
s_next = None
flux = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global h, h_next, w, w_next, s, s_next, flux, pixels
    h_next = ti.field(ti.f32, shape=(N, N))
@ti.kernel
def thermal():
    for i, j in h:
        delta = 0.0
        for k in ti.static(range(4)):
            ni = i + DI[k]
            nj = j + DJ[k]
            if 0 <= ni < N and 0 <= nj < N:
                d = h[i, j] - h[ni, nj]
                if d > TALUS:
                    delta -= (d - TALUS) * 0.5 * THERMAL_RATE
                elif d < -TALUS:
                    delta += (-d - TALUS) * 0.5 * THERMAL_RATE
        h_next[i, j] = h[i, j] + delta
@ti.kernel
def copy_height():
    for i, j in h:
        h[i, j] = h_next[i, j]`,
          does: "Real slopes steeper than the 'angle of repose' shed rock until they aren't. Each cell checks all four neighbors: too far above one, give some height away; too far below, receive. Both cells of every pair compute the same exchange with opposite signs — so rock is conserved without any coordination.",
          why: "That symmetric-pair trick is the gather idea in its purest form: agreement through shared arithmetic instead of communication. Also note h finally needs a double-buffer — the moment a kernel writes what neighbors read, the twin appears. You could have predicted that.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Two mirrored branches: d > TALUS gives, d < -TALUS receives — same formula, opposite signs.", "Writes go to h_next; copy_height adopts."] },
        { title: "Geology joins the tick", adding: "thermal and its copy at the end of step (replace step).",
          code: `def step():
    rain()
    compute_flux()
    erode_deposit()
    move_water()
    move_sediment()
    copy_wet()
    thermal()
    copy_height()`,
          does: "The full weathering pipeline: rain, route, carve, transport, adopt, crumble, adopt.",
          why: "Two erosions with two personalities: hydraulic is CONCENTRATED (rivers dig lines), thermal is DIFFUSE (cliffs relax everywhere). Landscapes look real only when both act — sharp valleys, soft shoulders.",
          see: "Watch a steep young range age: knife ridges round off, scree aprons spread below crags, riverbanks slump. Let it run while you make coffee — deep time needs a little shallow time.",
          checkpoint: "Slopes soften over minutes.",
          recovery: ["thermal() then copy_height(), both at the end of step."] },
        { title: "The HUD", adding: "the weather readout (replace the draw block).",
          code: `        render()
        gui.set_image(pixels)
        sky = "raining" if raining else "paused"
        gui.text(f"weather: {sky}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[space] rain on/off  [r] new mountains", (0.02, 0.94), color=0xAAAAAA)
        gui.show()`,
          does: "The house HUD: state line on top, control legend dimmed below. Muscle memory by now.",
          why: "That's Arc 1 complete: kernels and fields (01), a real solver (02), forces and sources (03), procedural generation (04), and now generation + simulation fused into a world with a history. Arc 2 trades grids for PARTICLES — millions of them.",
          see: "Roll a world, rain on it for ten minutes, and look at what you built: rivers that found their own valleys, deltas that built themselves, cliffs that aged. Every line typed by hand.",
          checkpoint: "HUD reads out the weather. Final beat — project 05 and Arc 1 complete.",
          recovery: ["Text lines between set_image and show, as always."] }
      ]
    }
  ]
};

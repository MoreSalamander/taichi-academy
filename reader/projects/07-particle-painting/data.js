// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["07-particle-painting"] = {
  project: "07-particle-painting",
  title: "Particle Painting",
  pitch: "Paint fire, smoke, sparks, and water onto a live canvas — a particle pool, a ring buffer, and a splat instead of a grid.",
  tier: "easy-med",
  language: "Python",
  file: "particle_painting.py",
  chapters: [
    {
      id: 1, title: "A fountain of dots",
      build: "a pre-allocated particle pool, a ring-buffer emitter, gravity, and your first splat render.",
      beat: "A fountain of white dots — that never quite goes away.",
      steps: [
        { title: "No numpy this time", adding: "the docstring and one import.",
          code: `"""Particle Painting: paint fire, smoke, sparks, and water with your mouse."""
import taichi as ti`,
          does: "Just Taichi. Every project through 06 opened with import numpy as np too — this is the first one that doesn't need it.",
          why: "Every prior project generated its initial state ONCE, in numpy, then uploaded it. This project's whole point is a LIVE stream: particles are born continuously, under your mouse, for as long as you hold the button. There's no single 'seed' moment to hand off to numpy — so all the randomness this project needs will happen directly on the GPU instead. Watch for it a couple steps from now.",
          see: "Runs clean.",
          checkpoint: "python3 particle_painting.py returns silently.",
          recovery: ["Fresh project, fresh venv, same as always."] },
        { title: "A pool that's never freed", adding: "five dials, a particle pool sized once and for all, and init_sim.",
          code: `N = 512
MAX_PARTICLES = 20000
EMIT_RATE = 40
DT = 1.0
GRAVITY = 0.12
pos = None
vel = None
life = None
cursor = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, life, cursor, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=MAX_PARTICLES)
    vel = ti.Vector.field(2, ti.f32, shape=MAX_PARTICLES)
    life = ti.field(ti.f32, shape=MAX_PARTICLES)
    cursor = ti.field(ti.i32, shape=())
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "MAX_PARTICLES is a hard ceiling, allocated once — 20,000 slots that will be reused forever, never grown, never freed (the Metal rule from day one, still holding). life is new: a countdown per particle, 0 meaning 'dead, this slot is free to reuse.' cursor is a single scalar (shape=()) — a lone integer field, not an array, tracking where the NEXT particle will be written.",
          why: "This is a completely different memory strategy from every prior project's fixed arrays. There's no 'how many particles exist right now' counter, no allocation, no deallocation — just a fixed pool and a promise that dead slots get silently overwritten by new arrivals. You'll build the mechanism (a ring buffer) in the very next step.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["cursor = ti.field(ti.i32, shape=()) — empty tuple shape means a single value, accessed later as cursor[None]."] },
        { title: "The ring buffer", adding: "a full reset and the emitter — the ring-buffer trick in five lines.",
          code: `def clear():
    pixels.fill(0.0)
    life.fill(0.0)
    cursor[None] = 0
@ti.kernel
def emit(mx: ti.f32, my: ti.f32):
    for _ in range(1):
        for k in range(EMIT_RATE):
            slot = (cursor[None] + k) % MAX_PARTICLES
            pos[slot] = ti.Vector([mx * N, my * N])
            vel[slot] = ti.Vector([(ti.random() - 0.5) * 0.6, ti.random() * 0.5])
            life[slot] = 1.0
        cursor[None] = (cursor[None] + EMIT_RATE) % MAX_PARTICLES`,
          does: "emit writes EMIT_RATE new particles starting at cursor, wrapping around the pool with % MAX_PARTICLES — slot 19,999's next neighbor is slot 0. Every particle gets the SAME birthplace this step (mx, my) but a randomized velocity, drawn with ti.random() called right there inside the kernel. for _ in range(1): forces this loop to run serially (one thread, in order) — same trick project 06's prefix_sum used — because slot numbers must be assigned in a fixed, non-colliding sequence.",
          why: "This is the whole 'never freed' promise made concrete: once the cursor laps the pool, new particles silently overwrite the OLDEST ones, whether they've finished fading or not. No bookkeeping, no death list, no compaction — just a wraparound counter. And ti.random() inside a kernel is new: every call returns a fresh random float, seeded from Taichi's own GPU-side generator, no numpy round-trip required.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The % MAX_PARTICLES on BOTH the per-particle slot AND the cursor update — forget either and the pool overflows its bounds."] },
        { title: "Fall, and leave a mark", adding: "gravity, a floor, the plainest possible splat, and the main loop.",
          code: `@ti.kernel
def update():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            vel[p][1] -= GRAVITY
            life[p] -= 0.01
            newpos = pos[p] + vel[p] * DT
            if newpos[1] < 0.0:
                newpos[1] = 0.0
                vel[p] *= 0.0
            pos[p] = newpos
@ti.kernel
def splat():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            cx = ti.cast(pos[p][0], ti.i32)
            cy = ti.cast(pos[p][1], ti.i32)
            if 0 <= cx < N and 0 <= cy < N:
                pixels[cx, cy] = ti.Vector([1.0, 1.0, 1.0])
def main():
    init_sim()
    clear()
    gui = ti.GUI("Particle Painting — taichi-academy", res=(N, N))
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        emit(0.5, 0.9)
        update()
        splat()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "update loops over the WHOLE pool every frame (not just the living ones — checking life[p] > 0.0 is cheap, tracking a separate 'alive list' isn't worth it at this scale) applying gravity and stopping dead at the floor. splat does the opposite of every prior project's render(): instead of one kernel asking 'what color is THIS pixel', each particle SCATTERS itself onto its own pixel directly — a completely different rendering shape, because the data (a particle list) doesn't live on the same grid as the canvas.",
          why: "Watch what happens after a few seconds: the canvas fills up with permanent white flecks that never go away. That's not a bug you need to fix right now — splat only ever brightens pixels, nothing ever dims them, so every pixel a particle has ever touched stays lit forever. It's an honest first draft. Chapter 4 fixes it — and gets something much prettier in the same stroke.",
          see: "A little fountain of white dots falls from a fixed point near the top, bounces to a stop at the floor, and the screen slowly fills with white speckle.",
          checkpoint: "Falling dots, honestly a bit messy. Beat 1.",
          recovery: ["splat uses = (overwrite), not += — that comes later.", "Nothing visible — emit(0.5, 0.9) is hardcoded for now; mouse control arrives next chapter."] }
      ]
    },
    {
      id: 2, title: "Follow the mouse",
      build: "real mouse painting, particles that vanish off the sides, and a clear key.",
      beat: "Drag to paint a stream of dots that pile at the floor and clear on demand.",
      steps: [
        { title: "One conductor, driven by your hand", adding: "the step() conductor, and the mouse read replacing the fixed emit point.",
          code: `def step(mx, my, painting):
    if painting:
        emit(mx, my)
    update()
    splat()
        painting = gui.is_pressed(ti.GUI.LMB)
        mx, my = gui.get_cursor_pos()
        step(mx, my, painting)`,
          does: "step() bundles the tick into one call — emit only if the mouse is actually held, then always update and splat. gui.is_pressed(ti.GUI.LMB) is exactly the mouse-drag idiom project 02's fluid stirring and project 03's torch both used — a returning technique, not a new one.",
          why: "Gating emission behind painting (a bool) rather than always calling emit() is what turns a fountain into a BRUSH — the same mechanism, aimed by you instead of fixed in code.",
          see: "The fountain now follows your cursor instead of sitting at a fixed point — drag around and paint falling dots wherever you like.",
          checkpoint: "Mouse-driven fountain. No red text.",
          recovery: ["painting is read fresh every frame from is_pressed — no separate press/release event needed, just a live boolean."] },
        { title: "Off the sides, gone for good", adding: "one line in update: particles that drift past the left or right edge die.",
          code: `@ti.kernel
def update():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            vel[p][1] -= GRAVITY
            life[p] -= 0.01
            newpos = pos[p] + vel[p] * DT
            if newpos[1] < 0.0:
                newpos[1] = 0.0
                vel[p] *= 0.0
            if newpos[0] < 0.0 or newpos[0] > N:
                life[p] = 0.0
            pos[p] = newpos`,
          does: "life[p] = 0.0 is how a particle dies in this system — not removed from any list, just marked dead, invisible to splat, and eventually silently overwritten by the ring buffer whenever the cursor laps back around to its slot.",
          why: "Now that you're painting near the edges of the canvas (not just a fixed center point), particles WILL drift off the sides — and without this check they'd sit forever at a clamped x, piling up along the walls instead of vanishing cleanly.",
          see: "Dots that drift past either edge disappear instead of collecting along the walls.",
          checkpoint: "No red text.",
          recovery: ["Kill by SETTING life[p] = 0.0 — there's no 'remove from the array' operation here, the pool is fixed-size forever."] },
        { title: "A blank page on demand", adding: "the clear key.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "c":
                clear()`,
          does: "One key, one function call — clear() (written back in chapter 1, and unused until now) wipes the canvas AND kills every living particle in one shot.",
          why: "clear() resets pixels, life, and cursor together — miss any one of them and 'clearing' would leave ghosts: dead pixels with nothing painting them, or a cursor that's forgotten where it was.",
          see: "Tap C: the fleck-covered canvas from chapter 1 wipes instantly to black, ready for a clean stroke.",
          checkpoint: "C clears the canvas. Beat 2.",
          recovery: ["clear() was written in chapter 1 already — this step is purely wiring a key to it."] }
      ]
    },
    {
      id: 3, title: "Four elements",
      build: "a material per particle, and physics that genuinely differs by element.",
      beat: "Fire and smoke rise, sparks bounce, water falls and sticks — four distinct behaviors, still drawn in plain white.",
      steps: [
        { title: "Name the elements", adding: "four material IDs, the material field, and its line in init_sim, plus the emitter branching on which one you're painting.",
          code: `FIRE, SMOKE, SPARKS, WATER = 0, 1, 2, 3
material = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, life, material, cursor, pixels
    material = ti.field(ti.i32, shape=MAX_PARTICLES)
@ti.kernel
def emit(mx: ti.f32, my: ti.f32, mat: ti.i32):
    for _ in range(1):
        for k in range(EMIT_RATE):
            slot = (cursor[None] + k) % MAX_PARTICLES
            v = ti.Vector([0.0, 0.0])
            if mat == SPARKS:
                angle = ti.random() * 6.2831853
                speed = 2.0 + ti.random() * 3.0
                v = ti.Vector([ti.cos(angle), ti.sin(angle)]) * speed
            elif mat == FIRE:
                v = ti.Vector([(ti.random() - 0.5) * 0.6, 0.5 + ti.random() * 1.0])
            elif mat == SMOKE:
                v = ti.Vector([(ti.random() - 0.5) * 0.3, 0.2 + ti.random() * 0.3])
            else:
                v = ti.Vector([(ti.random() - 0.5) * 1.0, -0.2 - ti.random() * 0.5])
            pos[slot] = ti.Vector([mx * N, my * N])
            vel[slot] = v
            life[slot] = 1.0
            material[slot] = mat
        cursor[None] = (cursor[None] + EMIT_RATE) % MAX_PARTICLES`,
          does: "Four plain integers name the four brushes. material tags each particle the moment it's born. Sparks explode outward in a full circle (ti.random()*2π, any direction); fire jitters mostly upward; smoke drifts upward gently; water (the else branch) starts with a slight downward nudge.",
          why: "A particle's identity is decided ONCE, at birth, in emit — everything downstream (how it moves, how it looks) just reads material[p] and reacts. That's the same 'decide once, read everywhere' shape as project 06's species field.",
          see: "Runs clean (the physics still treats every material identically until the next step).",
          checkpoint: "No red text.",
          recovery: ["mat == SPARKS uses a full 2π angle range; the other three only jitter a SMALL velocity component — they're brushes, not explosions."] },
        { title: "Four kinds of falling (and rising)", adding: "three more dials and the material-aware update.",
          code: `FIRE_BUOYANCY = 0.10
SMOKE_BUOYANCY = 0.04
SPARK_BOUNCE = 0.5
@ti.kernel
def update():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            m = material[p]
            if m == FIRE:
                vel[p][1] += FIRE_BUOYANCY
                life[p] -= 0.02
            elif m == SMOKE:
                vel[p][1] += SMOKE_BUOYANCY
                vel[p][0] += (ti.random() - 0.5) * 0.05
                life[p] -= 0.008
            elif m == SPARKS:
                vel[p][1] -= GRAVITY
                life[p] -= 0.015
            else:
                vel[p][1] -= GRAVITY
                life[p] -= 0.004
            newpos = pos[p] + vel[p] * DT
            if newpos[1] < 0.0:
                newpos[1] = 0.0
                if m == SPARKS:
                    vel[p][1] *= -SPARK_BOUNCE
                else:
                    vel[p] *= 0.0
            elif newpos[1] > N:
                life[p] = 0.0
            if newpos[0] < 0.0 or newpos[0] > N:
                life[p] = 0.0
            pos[p] = newpos`,
          does: "Fire and smoke get buoyancy (positive y — this codebase's 'up', same convention project 03 established); smoke also gets a little per-frame horizontal jitter, a lazy drift instead of a straight column. Sparks and water both fall under GRAVITY, but hit the floor differently: sparks flip their vertical velocity and lose half its magnitude (* -SPARK_BOUNCE, a real bounce), everything else just stops dead (the old * 0.0). Rising materials can now die off the TOP of the screen too (elif newpos[1] > N).",
          why: "Four materials, four life spans (0.02 down to 0.004 per tick) — fire flickers out fast, water lingers. Notice the PHYSICS differs (branch by branch) while the overall SHAPE of update — check life, move by branch, handle the floor, handle the walls — hasn't changed since chapter 1. New behavior, same skeleton.",
          see: "Still all in plain white, but hold the mouse down: fire and smoke now billow upward, sparks and water fall — you just can't tell them apart by EYE yet.",
          checkpoint: "No red text — behavior is right, appearance isn't yet.",
          recovery: ["The bounce only applies to SPARKS (an elif inside the floor-contact block) — every other material just stops.", "life[p] -= differs per branch — sparks and fire should visibly outlive water by a wide margin once you can see life expire."] },
        { title: "Choose your element", adding: "the step() conductor gaining a material argument, a current-brush variable, the number-key selector, and the wired-up call.",
          code: `def step(mx, my, mat, painting):
    if painting:
        emit(mx, my, mat)
    update()
    splat()
    current = FIRE
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "1":
                current = FIRE
            elif e.key == "2":
                current = SMOKE
            elif e.key == "3":
                current = SPARKS
            elif e.key == "4":
                current = WATER
            elif e.key == "c":
                clear()
        painting = gui.is_pressed(ti.GUI.LMB)
        mx, my = gui.get_cursor_pos()
        step(mx, my, current, painting)`,
          does: "current just remembers which brush is selected; four number keys reassign it; step() now threads mat through to emit(). Four elif branches, one flat pattern — the same shape project 05's preset-cycling keys used, just fixed keys instead of a rotating index.",
          why: "This is the last piece behavior needed — everything downstream (emit's velocity branch, update's force branch) was ALREADY written to react to a material id back in the last two steps. Selecting a brush is just choosing which id to pass in.",
          see: "Press 1/2/3/4 and paint: fire and smoke billow up wherever you drag, sparks explode outward and bounce, water streams down and pools at the floor — four unmistakably different behaviors. Still all rendered in flat white, and the canvas is still filling up with permanent flecks.",
          checkpoint: "Four distinct behaviors, selectable. Beat 3.",
          recovery: ["Every branch is an elif off the SAME e.key check — a typo'd key string silently does nothing, which is easy to miss."] }
      ]
    },
    {
      id: 4, title: "Paint in color",
      build: "trails that actually fade, and a color ramp per material.",
      beat: "Glowing, fading, colored streaks — the real look, in three small steps.",
      steps: [
        { title: "Fix the flecks", adding: "a decay dial, the fade kernel, and its slot in step — the chapter-1 problem, solved.",
          code: `FADE = 0.90
@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE
def step(mx, my, mat, painting):
    if painting:
        emit(mx, my, mat)
    update()
    fade()
    splat()`,
          does: "One line, looped over every pixel every frame: multiply by 0.90. After ~65 frames any untouched pixel has decayed under 1/1000th of its brightness — invisible, without ever being explicitly cleared.",
          why: "This is the fix for the 'permanent white flecks' you were promised back in chapter 1 — and notice it ALSO buys you trails for free. fade() runs BEFORE splat() each tick, so a moving particle leaves a streak of slowly-dimming positions behind it, not a static dot.",
          see: "The permanent speckle is gone — dots now leave a soft fading tail behind them instead of accumulating forever.",
          checkpoint: "Trails, no more permanent buildup. No red text.",
          recovery: ["fade() must run BEFORE splat() in step() — fade the old picture, then paint the new positions on top."] },
        { title: "A color for every age", adding: "the color-ramp function — not called yet.",
          code: `@ti.func
def material_color(m, t) -> ti.math.vec3:
    c = ti.Vector([0.0, 0.0, 0.0])
    if m == FIRE:
        c = ti.Vector([1.0, 0.9, 0.3]) * (1.0 - t) + ti.Vector([0.6, 0.05, 0.0]) * t
    elif m == SMOKE:
        g = 0.5 * (1.0 - t)
        c = ti.Vector([g, g, g])
    elif m == SPARKS:
        c = ti.Vector([1.0, 1.0, 0.9]) * (1.0 - t) + ti.Vector([1.0, 0.4, 0.05]) * t
    else:
        c = ti.Vector([0.15, 0.35, 0.85])
    return c`,
          does: "t is age (0 = just born, 1 = about to die). Fire and sparks both lerp from a hot near-white through to a deep ember color as they age — fire toward dark red, sparks toward burnt orange. Smoke is flat gray, dimming toward black as t grows. Water doesn't age at all — a single steady blue.",
          why: "Two colors and a lerp is the entire trick behind every 'hot to cold' gradient you've ever seen in a game — you wrote a tinier version of this exact idea back in project 05's band() elevation colors. Here it's driven by TIME rather than height.",
          see: "Runs clean — nothing draws with it yet.",
          checkpoint: "No red text.",
          recovery: ["c = ... * (1.0 - t) + ... * t is a lerp: at t=0 you get 100% the first color, at t=1 you get 100% the second."] },
        { title: "See it", adding: "splat upgraded from a flat white overwrite to colored, additive, life-weighted paint.",
          code: `@ti.kernel
def splat():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            m = material[p]
            t = 1.0 - life[p]
            col = material_color(m, t)
            inten = life[p] * (0.6 if m == SMOKE else 1.0)
            cx = ti.cast(pos[p][0], ti.i32)
            cy = ti.cast(pos[p][1], ti.i32)
            if 0 <= cx < N and 0 <= cy < N:
                pixels[cx, cy] += col * inten`,
          does: "Two changes from chapter 1's splat: = became += (adding light instead of overwriting it — particles near each other now brighten the same pixel together), and the flat white became material_color(m, t) scaled by inten, which fades toward zero as life runs out (and is dimmed further for smoke, which should read as translucent, not solid).",
          why: "Additive blending (+=) is why overlapping fire particles glow brighter at their core instead of just looking like more white dots — it's the same idea as project 06's speed-glow blend, applied to overlapping splats instead of a single particle's own color.",
          see: "The real look arrives: fire licks upward from white-hot through orange to dark red as it cools and rises; smoke billows in soft fading gray; sparks burst in a shower of white-to-ember dots; water streams down in blue and pools at the floor.",
          checkpoint: "Full color, fading trails. Beat 4.",
          recovery: ["t = 1.0 - life[p] — life counts DOWN from 1, so age (t) counts UP from 0.", "Additive splat can wash out to solid white if too many particles overlap — that's what chapter 5's clamp is for."] }
      ]
    },
    {
      id: 5, title: "Glow and ship it",
      build: "a soft splat radius, a safety clamp, and the on-screen brush readout.",
      beat: "Soft glowing blobs instead of hard single pixels — the finished canvas.",
      steps: [
        { title: "Soften the dot into a glow", adding: "the splat kernel gaining a 3x3 falloff blob instead of one hard pixel.",
          code: `@ti.kernel
def splat():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            m = material[p]
            t = 1.0 - life[p]
            col = material_color(m, t)
            inten = life[p] * (0.6 if m == SMOKE else 1.0)
            cx = ti.cast(pos[p][0], ti.i32)
            cy = ti.cast(pos[p][1], ti.i32)
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                xi, yj = cx + di, cy + dj
                if 0 <= xi < N and 0 <= yj < N:
                    w = ti.max(0.0, 1.0 - (di * di + dj * dj) / 2.5)
                    pixels[xi, yj] += col * inten * w`,
          does: "ti.static(ti.ndrange((-1, 2), (-1, 2))) unrolls to the 9 offsets of a 3x3 block, centered on the particle. Each neighbor gets a weight that falls off with squared distance from center (w = 1 at the center, smaller at the corners, never negative) — a crude but effective soft circle instead of one hard-edged pixel.",
          why: "This is the exact same 'visit a small fixed neighborhood' shape as project 06's 27-cell search — here spent on RENDERING instead of physics, splatting each particle across a handful of pixels instead of one.",
          see: "Every dot is now a small soft glow instead of a single hard pixel — the whole canvas reads as smoke and fire instead of static.",
          checkpoint: "Soft glowing particles. No red text.",
          recovery: ["The weight formula uses di*di + dj*dj (squared distance) — at the corners (±1,±1) that's 2, giving w = 1 - 2/2.5 = 0.2, dim but nonzero."] },
        { title: "A safety net", adding: "a clamp kernel and its slot at the end of step.",
          code: `@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)
def step(mx, my, mat, painting):
    if painting:
        emit(mx, my, mat)
    update()
    fade()
    splat()
    clamp_pixels()`,
          does: "One line, run last every tick: no color channel is ever allowed above 1.0, no matter how many particles pile onto the same pixel.",
          why: "Hold the mouse still and paint one spot for a while — dozens of overlapping, additively-blended splats WILL blow a pixel past 1.0 without this. A GPU display can usually survive that (values just clip on screen), but an unbounded field is a landmine for anything that reads it later — a test, a screenshot, a future effect layered on top. Bound your state defensively, the same lesson project 06's determinism fix taught from a different angle.",
          see: "No visible change under normal painting — the safety net you (hopefully) never notice.",
          checkpoint: "No red text.",
          recovery: ["clamp_pixels() runs LAST in step(), after splat has added everything for this frame."] },
        { title: "The readout", adding: "a brush-name lookup and the HUD text.",
          code: `    names = {FIRE: "fire", SMOKE: "smoke", SPARKS: "sparks", WATER: "water"}
        gui.set_image(pixels)
        gui.text(f"brush: {names[current]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1-4] fire/smoke/sparks/water  drag to paint  [c] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()`,
          does: "A dict from material id to its display name, and two lines of HUD text — the current brush, and the control legend — the same two-line pattern (state on top, legend dimmed below) every prior project's HUD has used.",
          why: "That closes project 07: a completely different shape of sim from every project before it — a pool of independent, born-and-recycled particles splatted onto a canvas instead of a grid of cells reading their neighbors — built from ideas you already had (ti.static neighborhoods, additive blending, material branching, ring buffers) recombined for a new kind of state.",
          see: "Paint with all four elements at once: a bonfire of rising flame and smoke, a shower of bouncing sparks, and a stream of pooling water — the brush name always visible in the corner.",
          checkpoint: "HUD reads out the current brush. Final beat — project 07 complete.",
          recovery: ["names[current] will KeyError if current is ever anything but one of the four material ids — worth remembering if you extend this with a 5th brush."] }
      ]
    }
  ]
};

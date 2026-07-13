"""Code SOT for project 07 — particle painting.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 07-particle-painting`.

Evolutions: chapter 1 ships a single generic falling particle (no material,
no mouse) so the whole pipeline (pool, emit, update, splat, main loop) stands
up at its simplest — and is honest about the flaw that results (pixels never
fade, so the canvas fills with permanent white flecks). Chapter 2 wires up
mouse painting. Chapter 3 gives particles a material and distinct physics.
Chapter 4 fixes the fading problem AND adds color in the same breath. Chapter
5 softens the splat into a glow and ships the HUD.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="07-particle-painting",
    default_file="particle_painting.py",
    reference={"particle_painting.py": PROJECT_DIR / "reference" / "particle_painting.py"},
    chapter_steps={1: 4, 2: 3, 3: 3, 4: 3, 5: 3},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Particle Painting: paint fire, smoke, sparks, and water with your mouse."""'))
frag(((1, 1), "import taichi as ti"))

# --- constants -------------------------------------------------------------------------

frag(((1, 2), "N = 512"))
frag(((1, 2), "MAX_PARTICLES = 20000"))
frag(((1, 2), "EMIT_RATE = 40"))
frag(((1, 2), "DT = 1.0"))
frag(((1, 2), "GRAVITY = 0.12"))
frag(((3, 2), "FIRE_BUOYANCY = 0.10"))
frag(((3, 2), "SMOKE_BUOYANCY = 0.04"))
frag(((3, 2), "SPARK_BOUNCE = 0.5"))
frag(((4, 1), "FADE = 0.90"))
frag(((3, 1), "FIRE, SMOKE, SPARKS, WATER = 0, 1, 2, 3"))

# --- module-level fields ------------------------------------------------------------

frag(((1, 2), "pos = None"))
frag(((1, 2), "vel = None"))
frag(((1, 2), "life = None"))
frag(((3, 1), "material = None"))
frag(((1, 2), "cursor = None"))
frag(((1, 2), "pixels = None"))

DOC = '    """Start Taichi and allocate every field once (Metal can\'t free fields)."""'

frag(
    ((1, 2), f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, life, cursor, pixels"),
    ((3, 1), f"def init_sim(arch=None):\n{DOC}\n    global pos, vel, life, material, cursor, pixels"),
)
frag(
    (
        (1, 2),
        "    if arch is None:\n"
        "        try:\n"
        "            ti.init(arch=ti.gpu)\n"
        "        except Exception:\n"
        "            ti.init(arch=ti.cpu)\n"
        "    else:\n"
        "        ti.init(arch=arch)",
    )
)
frag(((1, 2), "    pos = ti.Vector.field(2, ti.f32, shape=MAX_PARTICLES)"))
frag(((1, 2), "    vel = ti.Vector.field(2, ti.f32, shape=MAX_PARTICLES)"))
frag(((1, 2), "    life = ti.field(ti.f32, shape=MAX_PARTICLES)"))
frag(((3, 1), "    material = ti.field(ti.i32, shape=MAX_PARTICLES)"))
frag(((1, 2), "    cursor = ti.field(ti.i32, shape=())"))
frag(((1, 2), "    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))"))

# --- reset -----------------------------------------------------------------------------

frag(((1, 3), "def clear():\n    pixels.fill(0.0)\n    life.fill(0.0)\n    cursor[None] = 0"))

# --- emission ----------------------------------------------------------------------

EMIT_V1 = """@ti.kernel
def emit(mx: ti.f32, my: ti.f32):
    for _ in range(1):
        for k in range(EMIT_RATE):
            slot = (cursor[None] + k) % MAX_PARTICLES
            pos[slot] = ti.Vector([mx * N, my * N])
            vel[slot] = ti.Vector([(ti.random() - 0.5) * 0.6, ti.random() * 0.5])
            life[slot] = 1.0
        cursor[None] = (cursor[None] + EMIT_RATE) % MAX_PARTICLES"""

EMIT_V2 = """@ti.kernel
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
        cursor[None] = (cursor[None] + EMIT_RATE) % MAX_PARTICLES"""

frag(((1, 3), EMIT_V1), ((3, 1), EMIT_V2))

# --- motion ------------------------------------------------------------------------

UPDATE_V1 = """@ti.kernel
def update():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            vel[p][1] -= GRAVITY
            life[p] -= 0.01
            newpos = pos[p] + vel[p] * DT
            if newpos[1] < 0.0:
                newpos[1] = 0.0
                vel[p] *= 0.0
            pos[p] = newpos"""

UPDATE_V2 = """@ti.kernel
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
            pos[p] = newpos"""

UPDATE_V3 = """@ti.kernel
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
            pos[p] = newpos"""

frag(((1, 4), UPDATE_V1), ((2, 2), UPDATE_V2), ((3, 2), UPDATE_V3))

# --- the fade (trails) ---------------------------------------------------------------

frag(((4, 1), "@ti.kernel\ndef fade():\n    for i, j in pixels:\n        pixels[i, j] *= FADE"))

# --- color ---------------------------------------------------------------------------

MATERIAL_COLOR = """@ti.func
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
    return c"""

frag(((4, 2), MATERIAL_COLOR))

# --- splat ---------------------------------------------------------------------------

SPLAT_V1 = """@ti.kernel
def splat():
    for p in range(MAX_PARTICLES):
        if life[p] > 0.0:
            cx = ti.cast(pos[p][0], ti.i32)
            cy = ti.cast(pos[p][1], ti.i32)
            if 0 <= cx < N and 0 <= cy < N:
                pixels[cx, cy] = ti.Vector([1.0, 1.0, 1.0])"""

SPLAT_V2 = """@ti.kernel
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
                pixels[cx, cy] += col * inten"""

SPLAT_V3 = """@ti.kernel
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
                    pixels[xi, yj] += col * inten * w"""

frag(((1, 4), SPLAT_V1), ((4, 3), SPLAT_V2), ((5, 1), SPLAT_V3))

# --- safety clamp ----------------------------------------------------------------------

frag(((5, 2), "@ti.kernel\ndef clamp_pixels():\n    for i, j in pixels:\n        pixels[i, j] = ti.min(pixels[i, j], 1.0)"))

# --- the tick ----------------------------------------------------------------------

frag(
    ((2, 1), "def step(mx, my, painting):\n    if painting:\n        emit(mx, my)\n    update()\n    splat()"),
    (
        (3, 3),
        "def step(mx, my, mat, painting):\n    if painting:\n        emit(mx, my, mat)\n    update()\n    splat()",
    ),
    (
        (4, 1),
        "def step(mx, my, mat, painting):\n    if painting:\n        emit(mx, my, mat)\n    update()\n    fade()\n    splat()",
    ),
    (
        (5, 2),
        "def step(mx, my, mat, painting):\n    if painting:\n        emit(mx, my, mat)\n    update()\n    fade()\n    splat()\n    clamp_pixels()",
    ),
)

# --- main (small sub-fragments so each step's diff stays readable) --------------------

frag(((1, 4), 'def main():\n    init_sim()\n    clear()\n    gui = ti.GUI("Particle Painting — taichi-academy", res=(N, N))'))
frag(((3, 3), "    current = FIRE"))
frag(((5, 3), '    names = {FIRE: "fire", SMOKE: "smoke", SPARKS: "sparks", WATER: "water"}'))
frag(((1, 4), "    while gui.running:"))

EVENTS_V1 = """        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False"""

EVENTS_V2 = '''        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "c":
                clear()'''

EVENTS_V3 = '''        for e in gui.get_events(ti.GUI.PRESS):
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
                clear()'''

frag(((1, 4), EVENTS_V1), ((2, 3), EVENTS_V2), ((3, 3), EVENTS_V3))

frag(
    ((1, 4), "        emit(0.5, 0.9)\n        update()\n        splat()"),
    ((2, 1), "        painting = gui.is_pressed(ti.GUI.LMB)\n        mx, my = gui.get_cursor_pos()\n        step(mx, my, painting)"),
    (
        (3, 3),
        "        painting = gui.is_pressed(ti.GUI.LMB)\n        mx, my = gui.get_cursor_pos()\n        step(mx, my, current, painting)",
    ),
)

RENDER_V1 = "        gui.set_image(pixels)\n        gui.show()"

RENDER_V2 = '''        gui.set_image(pixels)
        gui.text(f"brush: {names[current]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1-4] fire/smoke/sparks/water  drag to paint  [c] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()'''

frag(((1, 4), RENDER_V1), ((5, 3), RENDER_V2))
frag(((1, 4), 'if __name__ == "__main__":\n    main()'))

"""Particle Painting: paint fire, smoke, sparks, and water with your mouse."""

import taichi as ti

N = 512
MAX_PARTICLES = 20000
EMIT_RATE = 40
DT = 1.0
GRAVITY = 0.12
FIRE_BUOYANCY = 0.10
SMOKE_BUOYANCY = 0.04
SPARK_BOUNCE = 0.5
FADE = 0.90

FIRE, SMOKE, SPARKS, WATER = 0, 1, 2, 3

pos = None
vel = None
life = None
material = None
cursor = None
pixels = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, life, material, cursor, pixels
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
    material = ti.field(ti.i32, shape=MAX_PARTICLES)
    cursor = ti.field(ti.i32, shape=())
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))


def clear():
    pixels.fill(0.0)
    life.fill(0.0)
    cursor[None] = 0


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
        cursor[None] = (cursor[None] + EMIT_RATE) % MAX_PARTICLES


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
            pos[p] = newpos


@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE


@ti.func
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
    return c


@ti.kernel
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
                    pixels[xi, yj] += col * inten * w


@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)


def step(mx, my, mat, painting):
    if painting:
        emit(mx, my, mat)
    update()
    fade()
    splat()
    clamp_pixels()


def main():
    init_sim()
    clear()
    gui = ti.GUI("Particle Painting — taichi-academy", res=(N, N))
    current = FIRE
    names = {FIRE: "fire", SMOKE: "smoke", SPARKS: "sparks", WATER: "water"}
    while gui.running:
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
        step(mx, my, current, painting)
        gui.set_image(pixels)
        gui.text(f"brush: {names[current]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1-4] fire/smoke/sparks/water  drag to paint  [c] clear", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

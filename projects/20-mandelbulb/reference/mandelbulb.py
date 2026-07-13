"""Mandelbulb: a 3D fractal you can't mesh — ray-marched by asking 'how far is it, at least?'"""

import math

import numpy as np
import taichi as ti

RES = 400
POWER = 8.0
DE_ITERS = 12
BAILOUT = 2.0
MAX_STEPS = 128
EPS_BASE = 0.0004
MAX_DIST = 4.0
ZOOM = 1.6

CAM_HEIGHT_RATIO = 0.36
ORBIT_SPEED = 0.004
RADIUS_MIN = 1.35
RADIUS_MAX = 4.0

SUN = (0.5, -0.4, 0.75)

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
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))


@ti.func
def bulb_de(p):
    z = p
    dr = 1.0
    r = ti.max(z.norm(), 1e-9)
    for _ in range(DE_ITERS):
        if r < BAILOUT:
            theta = ti.acos(z[2] / r) * POWER
            phi = ti.atan2(z[1], z[0]) * POWER
            zr = ti.pow(r, POWER)
            dr = ti.pow(r, POWER - 1.0) * POWER * dr + 1.0
            z = zr * ti.Vector([ti.sin(theta) * ti.cos(phi), ti.sin(theta) * ti.sin(phi), ti.cos(theta)]) + p
            r = ti.max(z.norm(), 1e-9)
    return 0.5 * ti.log(r) * r / dr


@ti.func
def normal_at(p, e):
    dx = bulb_de(p + ti.Vector([e, 0.0, 0.0])) - bulb_de(p - ti.Vector([e, 0.0, 0.0]))
    dy = bulb_de(p + ti.Vector([0.0, e, 0.0])) - bulb_de(p - ti.Vector([0.0, e, 0.0]))
    dz = bulb_de(p + ti.Vector([0.0, 0.0, e])) - bulb_de(p - ti.Vector([0.0, 0.0, e]))
    return ti.Vector([dx, dy, dz]).normalized()


@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            p = cam + rd * t
            n = normal_at(p, eps)
            diffuse = ti.max(n.dot(sun), 0.0)
            ao = 1.0 - steps / float(MAX_STEPS)
            base = ti.Vector([0.85, 0.6, 0.95]) * 0.5 + 0.5 * ti.Vector([ti.abs(n[0]), ti.abs(n[1]), ti.abs(n[2])])
            col = base * (0.15 + 0.85 * diffuse) * ao
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)


def camera_position(theta, radius):
    return (
        radius * math.sin(theta),
        -radius * math.cos(theta),
        radius * CAM_HEIGHT_RATIO,
    )


def main():
    init_sim()
    gui = ti.GUI("Mandelbulb — taichi-academy", res=RES, background_color=0x000000)
    theta = 0.7
    radius = 2.2
    pmx = None
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in (ti.GUI.UP, "w"):
                radius = max(radius * 0.94, RADIUS_MIN)
            elif e.key in (ti.GUI.DOWN, "s"):
                radius = min(radius / 0.94, RADIUS_MAX)
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED
        cx, cy, cz = camera_position(theta, radius)
        eps = EPS_BASE * radius
        render(cx, cy, cz, *SUN, eps)
        gui.set_image(pixels)
        gui.text(f"radius {radius:.2f}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to orbit  [w/up] zoom in  [s/down] zoom out", (0.02, 0.94), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

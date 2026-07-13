"""Soft Body: spring rings plus internal pressure — jelly, rubber, and a balloon, one engine."""

import math

import numpy as np
import taichi as ti

N_RING = 28
N_BODIES = 3
N = N_RING * N_BODIES
DT = 5e-4
SUBSTEPS = 30
GRAVITY = 9.8
WORLD = 1.0
BODY_RADIUS = 0.08
CENTERS = [(0.25, 0.6), (0.5, 0.6), (0.75, 0.6)]

STIFFNESS_NP = np.array([800.0, 4000.0, 300.0], dtype=np.float32)
DAMPING_NP = np.array([6.0, 2.0, 3.0], dtype=np.float32)
GAS_NP = np.array([1.5, 1.0, 2.5], dtype=np.float32)
MASS_NP = np.array([0.02, 0.02, 0.02], dtype=np.float32)

GRAB_K = 400.0
GRAB_DAMP = 4.0
GRAB_RADIUS = 0.1

pos = None
vel = None
force = None
rest_len = None
body_id = None
stiffness = None
damping = None
gas = None
mass = None
grabbed = None
grab_target = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, force, rest_len, body_id, stiffness, damping, gas, mass
    global grabbed, grab_target
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    force = ti.Vector.field(2, ti.f32, shape=N)
    rest_len = ti.field(ti.f32, shape=N)
    body_id = ti.field(ti.i32, shape=N)
    stiffness = ti.field(ti.f32, shape=N_BODIES)
    damping = ti.field(ti.f32, shape=N_BODIES)
    gas = ti.field(ti.f32, shape=N_BODIES)
    mass = ti.field(ti.f32, shape=N_BODIES)
    grabbed = ti.field(ti.i32, shape=())
    grab_target = ti.Vector.field(2, ti.f32, shape=())


def seed_ring(cx, cy, radius, n):
    """Pure numpy: n points evenly spaced around a circle."""
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1).astype(np.float32)


def rest_lengths(ring):
    nxt = np.roll(ring, -1, axis=0)
    return np.linalg.norm(nxt - ring, axis=1).astype(np.float32)


def apply_seed():
    rings = [seed_ring(cx, cy, BODY_RADIUS, N_RING) for cx, cy in CENTERS]
    pos.from_numpy(np.concatenate(rings, axis=0))
    vel.fill(0.0)
    body_id.from_numpy(np.concatenate([np.full(N_RING, b, dtype=np.int32) for b in range(N_BODIES)]))
    rest_len.from_numpy(np.concatenate([rest_lengths(r) for r in rings]))
    stiffness.from_numpy(STIFFNESS_NP)
    damping.from_numpy(DAMPING_NP)
    gas.from_numpy(GAS_NP)
    mass.from_numpy(MASS_NP)
    grabbed[None] = -1


@ti.kernel
def compute_forces():
    for p in pos:
        b = body_id[p]
        force[p] = ti.Vector([0.0, -GRAVITY * mass[b]])

    for p in pos:
        b = body_id[p]
        q = (p // N_RING) * N_RING + (p + 1) % N_RING
        d = pos[q] - pos[p]
        dist = d.norm() + 1e-6
        dirn = d / dist
        stretch = dist - rest_len[p]
        rel_v = (vel[q] - vel[p]).dot(dirn)
        f = (stiffness[b] * stretch + damping[b] * rel_v) * dirn
        force[p] += f
        force[q] -= f

    for b in range(N_BODIES):
        area = 0.0
        base = b * N_RING
        for i in range(N_RING):
            p0, p1 = pos[base + i], pos[base + (i + 1) % N_RING]
            area += p0[0] * p1[1] - p1[0] * p0[1]
        area = ti.abs(area) * 0.5 + 1e-6
        pressure = gas[b] / area
        for i in range(N_RING):
            p0, p1 = pos[base + i], pos[base + (i + 1) % N_RING]
            edge = p1 - p0
            normal = ti.Vector([edge[1], -edge[0]])
            f = pressure * normal * 0.5
            force[base + i] += f
            force[base + (i + 1) % N_RING] += f

    if grabbed[None] >= 0:
        g = grabbed[None]
        pull = GRAB_K * (grab_target[None] - pos[g]) - GRAB_DAMP * vel[g]
        force[g] += pull


@ti.kernel
def integrate():
    for p in pos:
        b = body_id[p]
        vel[p] += DT * force[p] / mass[b]
        newp = pos[p] + DT * vel[p]
        if newp[1] < 0.0:
            newp[1] = 0.0
            vel[p][1] *= -0.3
            vel[p][0] *= 0.7
        if newp[0] < 0.0:
            newp[0] = 0.0
            vel[p][0] *= -0.3
        elif newp[0] > WORLD:
            newp[0] = WORLD
            vel[p][0] *= -0.3
        pos[p] = newp


def substep():
    compute_forces()
    integrate()


def grab_at(mx, my):
    p = pos.to_numpy()
    d2 = (p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2
    i = int(np.argmin(d2))
    if d2[i] < GRAB_RADIUS**2:
        grabbed[None] = i
    else:
        grabbed[None] = -1


def release():
    grabbed[None] = -1


def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Soft Body — taichi-academy", res=512, background_color=0x0A0A12)
    colors = np.zeros(N, dtype=np.uint32)
    colors[0:N_RING] = 0x8EC9FF
    colors[N_RING : 2 * N_RING] = 0xE8544A
    colors[2 * N_RING :] = 0xFFD35C
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed()
            elif e.key == ti.GUI.LMB:
                grab_at(*gui.get_cursor_pos())
        for e in gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                release()
        if gui.is_pressed(ti.GUI.LMB) and grabbed[None] >= 0:
            grab_target[None] = gui.get_cursor_pos()
        for _ in range(SUBSTEPS):
            substep()
        gui.circles(pos.to_numpy(), radius=2.5, color=colors)
        gui.text("drag a body to grab it  [r] reset", (0.02, 0.98), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

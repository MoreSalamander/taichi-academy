"""MPM Snow & Sand: one elastoplastic engine, particles AND a grid working together."""

import math

import numpy as np
import taichi as ti

N_GRID = 128
DX = 1.0 / N_GRID
INV_DX = float(N_GRID)
DT = 1e-4
SUBSTEPS = 25
P_VOL = (DX * 0.5) ** 2
P_MASS = P_VOL * 1.0
GRAVITY = 9.8

SNOW, SAND = 0, 1
N_PER_BLOCK = 6000
N_PARTICLES = N_PER_BLOCK * 2

E_SNOW, NU_SNOW = 1.4e4, 0.2
THETA_C_SNOW, THETA_S_SNOW = 2.5e-2, 4.5e-3
HARDEN_SNOW = 10.0

E_SAND, NU_SAND = 3.5e3, 0.3
FRICTION_DEG = 35.0
SAND_ALPHA = math.sqrt(2.0 / 3.0) * (2.0 * math.sin(math.radians(FRICTION_DEG))) / (
    3.0 - math.sin(math.radians(FRICTION_DEG))
)

STIR_RADIUS = 0.002

x = None
v = None
C = None
F = None
Jp = None
material = None
grid_v = None
grid_m = None


def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global x, v, C, F, Jp, material, grid_v, grid_m
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    x = ti.Vector.field(2, ti.f32, shape=N_PARTICLES)
    v = ti.Vector.field(2, ti.f32, shape=N_PARTICLES)
    C = ti.Matrix.field(2, 2, ti.f32, shape=N_PARTICLES)
    F = ti.Matrix.field(2, 2, ti.f32, shape=N_PARTICLES)
    Jp = ti.field(ti.f32, shape=N_PARTICLES)
    material = ti.field(ti.i32, shape=N_PARTICLES)
    grid_v = ti.Vector.field(2, ti.f32, shape=(N_GRID, N_GRID))
    grid_m = ti.field(ti.f32, shape=(N_GRID, N_GRID))


def seed_block(n, cx, cy, hx, hy, rng_seed):
    """Pure numpy: n random points inside a cx,cy-centered rectangle."""
    rng = np.random.default_rng(rng_seed)
    off = (rng.random((n, 2)).astype(np.float32) * 2 - 1) * np.array([hx, hy], dtype=np.float32)
    return off + np.array([cx, cy], dtype=np.float32)


@ti.kernel
def reset_particles():
    for p in x:
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        Jp[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, 2, 2)


def apply_seed(pos_snow, pos_sand):
    pos = np.concatenate([pos_snow, pos_sand], axis=0)
    mat = np.concatenate(
        [np.full(len(pos_snow), SNOW, dtype=np.int32), np.full(len(pos_sand), SAND, dtype=np.int32)]
    )
    x.from_numpy(pos)
    material.from_numpy(mat)
    reset_particles()


@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0


@ti.func
def snow_stress(p, U, sig, V):
    mu_0 = E_SNOW / (2 * (1 + NU_SNOW))
    lambda_0 = E_SNOW * NU_SNOW / ((1 + NU_SNOW) * (1 - 2 * NU_SNOW))
    h = ti.exp(HARDEN_SNOW * (1.0 - Jp[p]))
    mu, la = mu_0 * h, lambda_0 * h
    J = 1.0
    sig_c = sig
    for d in ti.static(range(2)):
        new_sig = ti.min(ti.max(sig_c[d, d], 1 - THETA_C_SNOW), 1 + THETA_S_SNOW)
        Jp[p] *= sig_c[d, d] / new_sig
        sig_c[d, d] = new_sig
        J *= new_sig
    F[p] = U @ sig_c @ V.transpose()
    return mu, la, J


@ti.func
def sand_stress(p, U, sig, V):
    mu_0 = E_SAND / (2 * (1 + NU_SAND))
    lambda_0 = E_SAND * NU_SAND / ((1 + NU_SAND) * (1 - 2 * NU_SAND))
    e0 = ti.log(ti.max(sig[0, 0], 1e-4))
    e1 = ti.log(ti.max(sig[1, 1], 1e-4))
    eps = ti.Vector([e0, e1])
    eps_trace = eps[0] + eps[1]
    eps_hat = eps - (eps_trace / 2.0) * ti.Vector([1.0, 1.0])
    eps_hat_norm = eps_hat.norm() + 1e-20
    new_eps = eps
    if eps_trace > 0.0:
        new_eps = ti.Vector([0.0, 0.0])
        Jp[p] *= ti.exp(eps_trace)
    else:
        delta_gamma = eps_hat_norm + (2.0 * lambda_0 + 2.0 * mu_0) / (2.0 * mu_0) * eps_trace * SAND_ALPHA
        if delta_gamma > 0.0:
            new_eps = eps - (delta_gamma / eps_hat_norm) * eps_hat
    new_sig0, new_sig1 = ti.exp(new_eps[0]), ti.exp(new_eps[1])
    F[p] = U @ ti.Matrix([[new_sig0, 0.0], [0.0, new_sig1]]) @ V.transpose()
    return mu_0, lambda_0, new_sig0 * new_sig1


@ti.kernel
def p2g():
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        F[p] = (ti.Matrix.identity(ti.f32, 2) + DT * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        mu, la, J = 0.0, 0.0, 1.0
        if material[p] == SNOW:
            mu, la, J = snow_stress(p, U, sig, V)
        else:
            mu, la, J = sand_stress(p, U, sig, V)

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
        stress += ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
        stress = (-DT * P_VOL * 4 * INV_DX * INV_DX) * stress
        affine = stress + P_MASS * C[p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * DX
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (P_MASS * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * P_MASS


@ti.kernel
def grid_forces(mx: ti.f32, my: ti.f32, fx_: ti.f32, fy_: ti.f32, stirring: ti.i32):
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1.0 / grid_m[i, j]) * grid_v[i, j]
        grid_v[i, j][1] -= DT * GRAVITY
        if stirring == 1:
            gx, gy = i * DX, j * DX
            d2 = (gx - mx) ** 2 + (gy - my) ** 2
            grid_v[i, j] += ti.exp(-d2 / STIR_RADIUS) * ti.Vector([fx_, fy_])
        if i < 3 and grid_v[i, j][0] < 0:
            grid_v[i, j][0] = 0
        if i > N_GRID - 3 and grid_v[i, j][0] > 0:
            grid_v[i, j][0] = 0
        if j < 3 and grid_v[i, j][1] < 0:
            grid_v[i, j][1] = 0
        if j > N_GRID - 3 and grid_v[i, j][1] > 0:
            grid_v[i, j][1] = 0


@ti.kernel
def g2p():
    for p in x:
        base = (x[p] * INV_DX - 0.5).cast(int)
        fx = x[p] * INV_DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * INV_DX * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += DT * v[p]


def substep(mx=0.0, my=0.0, fx=0.0, fy=0.0, stirring=False):
    clear_grid()
    p2g()
    grid_forces(mx, my, fx, fy, 1 if stirring else 0)
    g2p()


def default_seed(rng_seed=0):
    pos_snow = seed_block(N_PER_BLOCK, 0.28, 0.12, 0.12, 0.05, rng_seed)
    pos_sand = seed_block(N_PER_BLOCK, 0.72, 0.12, 0.12, 0.05, rng_seed + 1)
    return pos_snow, pos_sand


def main():
    init_sim()
    apply_seed(*default_seed())
    gui = ti.GUI("MPM Snow & Sand — taichi-academy", res=512, background_color=0x0A0A12)
    colors = np.zeros(N_PARTICLES, dtype=np.uint32)
    colors[:N_PER_BLOCK] = 0xE8F3FF
    colors[N_PER_BLOCK:] = 0xD8A650
    pmx, pmy = 0.0, 0.0
    dragging = False
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(*default_seed(rng_seed=np.random.randint(1_000_000)))
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            fx, fy = 0.0, 0.0
            if dragging:
                fx, fy = (mx - pmx) * 40.0 / SUBSTEPS, (my - pmy) * 40.0 / SUBSTEPS
            for _ in range(SUBSTEPS):
                substep(mx, my, fx, fy, stirring=True)
            pmx, pmy = mx, my
            dragging = True
        else:
            dragging = False
            for _ in range(SUBSTEPS):
                substep()
        gui.circles(x.to_numpy(), radius=1.5, color=colors)
        gui.text("drag to stir  [r] reset", (0.02, 0.98), color=0xAAAAAA)
        gui.show()


if __name__ == "__main__":
    main()

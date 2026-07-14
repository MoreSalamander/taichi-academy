"""Digital Brain: a sheet of spiking neurons — local excitation, surround inhibition, a
refractory pause — self-organizes into travelling waves, and Hebbian plasticity carves pathways."""

import numpy as np
import taichi as ti

N = 256
R = 5                    # coupling radius
TAU = 12.0               # membrane leak time constant
THRESH = 1.0             # firing threshold
REF = 6                  # refractory steps after a spike
SYN_DECAY = 0.8          # synaptic conductance decay per step
BIAS = 0.02              # tonic background drive
NOISE = 0.06             # random jitter on the input
W_EXC = 1.8              # excitatory centre strength
W_INH = 0.28             # inhibitory surround strength
SIG_E = 1.8              # excitatory gaussian width
SIG_I = 4.0              # inhibitory gaussian width
LR = 0.006               # Hebbian learning rate
HOMEO = 0.02             # homeostatic relaxation of weights toward 1
W_MIN, W_MAX = 0.5, 2.0

p = None
ref = None
fired = None
g = None
w = None
kernel = None
pixels = None


def init_sim(arch=None):
    """Start Taichi, allocate every field once (Metal can't free fields), build the coupling kernel."""
    global p, ref, fired, g, w, kernel, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    p = ti.field(ti.f32, shape=(N, N))
    ref = ti.field(ti.i32, shape=(N, N))
    fired = ti.field(ti.i32, shape=(N, N))
    g = ti.field(ti.f32, shape=(N, N))
    w = ti.field(ti.f32, shape=(N, N))
    kernel = ti.field(ti.f32, shape=(2 * R + 1, 2 * R + 1))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))
    build_kernel()


def mexican_hat():
    """Pure numpy: a difference-of-gaussians — excite close neighbours, inhibit distant ones."""
    k = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            r2 = di * di + dj * dj
            k[di + R, dj + R] = (W_EXC * np.exp(-r2 / (2 * SIG_E ** 2))
                                 - W_INH * np.exp(-r2 / (2 * SIG_I ** 2)))
    k[R, R] = 0.0  # a neuron does not synapse onto itself
    return k


def build_kernel():
    kernel.from_numpy(mexican_hat())


@ti.kernel
def apply_seed(rng_seed: ti.i32):
    for i, j in p:
        p[i, j] = 0.2 * ti.random()
        ref[i, j] = 0
        g[i, j] = 0.0
        w[i, j] = 1.0
        fired[i, j] = 0


@ti.kernel
def synapse_step(sx: ti.i32, sy: ti.i32, srad: ti.f32, si: ti.f32):
    """Every neuron gathers its neighbours' recent spikes through the kernel, integrates, and
    fires if it crosses threshold — unless it is still in its refractory pause."""
    for i, j in p:
        syn = 0.0
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    syn += kernel[di + R, dj + R] * g[ni, nj]
        ext = 0.0
        if (i - sx) ** 2 + (j - sy) ** 2 < srad * srad:
            ext = si
        f = 0
        if ref[i, j] > 0:
            ref[i, j] -= 1
            p[i, j] = 0.0
        else:
            drive = syn + ext + BIAS + NOISE * (ti.random() - 0.5)
            pp = p[i, j] + (drive - p[i, j] / TAU)
            if pp >= THRESH:
                p[i, j] = 0.0
                ref[i, j] = REF
                f = 1
            else:
                p[i, j] = ti.max(pp, 0.0)
        fired[i, j] = f


@ti.kernel
def conductance_step():
    """A spike leaves a decaying trace, scaled by the neuron's plastic weight — its 'loudness'."""
    for i, j in g:
        g[i, j] = g[i, j] * SYN_DECAY + fired[i, j] * w[i, j]


@ti.kernel
def plasticity_step():
    """Hebbian: a neuron firing amid active neighbours strengthens its output; all weights
    relax toward 1 (homeostasis) so nothing runs away. Fire together, wire together."""
    for i, j in w:
        local = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    local += g[ni, nj]
        dw = LR * fired[i, j] * local - HOMEO * (w[i, j] - 1.0)
        w[i, j] = ti.min(ti.max(w[i, j] + dw, W_MIN), W_MAX)


def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)
    conductance_step()
    plasticity_step()


def firing_rate():
    """Pure numpy: fraction of neurons that spiked this step."""
    return float(fired.to_numpy().mean())


def mean_weight(i0, i1, j0, j1):
    """Pure numpy: average plastic weight over a rectangular patch."""
    return float(w.to_numpy()[i0:i1, j0:j1].mean())


@ti.kernel
def render():
    for i, j in pixels:
        a = ti.min(g[i, j], 1.0)
        wn = (w[i, j] - W_MIN) / (W_MAX - W_MIN)
        cool = ti.Vector([0.15, 0.45, 0.95])
        warm = ti.Vector([1.0, 0.35, 0.55])
        col = (cool * (1.0 - wn) + warm * wn) * a + ti.Vector([0.05, 0.05, 0.08])
        pixels[i, j] = ti.min(col, 1.0)


def main():
    init_sim()
    apply_seed(0)
    gui = ti.GUI("Digital Brain — taichi-academy", res=(N, N), background_color=0x08080F)
    frame = 0
    while gui.running:
        frame += 1
        sx, sy, srad, si = -1000, -1000, 0.0, 0.0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(frame)
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            sx, sy, srad, si = int(mx * N), int(my * N), 6.0, 1.2  # a pacemaker under the cursor
        step(sx, sy, srad, si)
        render()
        gui.set_image(pixels)
        gui.text(f"firing {firing_rate() * 100:.1f}%   drag: stimulate   [r] reset", (0.02, 0.98), color=0xFFFFFF)
        gui.show()


if __name__ == "__main__":
    main()

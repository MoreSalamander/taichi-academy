"""Code SOT for project 28 — digital brain.

Fragments in FINAL document order; versions keyed by (chapter, step).
Verify with `python tools/build_fulls.py --project 28-digital-brain`.

Arc: chapter 1 is a single spiking neuron model on a grid — leaky integrate-and-fire with a
refractory pause, driven only by a stimulus, so a poke makes a local patch flicker and die.
Chapter 2 wires the neurons together through the Mexican-hat kernel and a decaying spike trace,
and the sheet erupts into self-sustaining travelling waves. Chapter 3 adds Hebbian plasticity:
weights that strengthen with use and relax by homeostasis — the finished reference.

The coupling kernel is built in chapter 1 (init_sim fills it) but not READ until chapter 2's
coupled synapse_step, so a few pieces are keyed a chapter before they matter.
"""

from pathlib import Path

from fragment_lib import FragmentSet

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPEC = FragmentSet(
    project_id="28-digital-brain",
    default_file="digital_brain.py",
    reference={"digital_brain.py": PROJECT_DIR / "reference" / "digital_brain.py"},
    chapter_steps={1: 6, 2: 2, 3: 2},
)
frag = SPEC.frag

# --- module head ---------------------------------------------------------------------

frag(((1, 1), '"""Digital Brain: a sheet of spiking neurons — local excitation, surround inhibition, a\nrefractory pause — self-organizes into travelling waves, and Hebbian plasticity carves pathways."""'))
frag(((1, 1), "import numpy as np"))
frag(((1, 1), "import taichi as ti"))

# --- constants + fields --------------------------------------------------------------

frag((
    (1, 2),
    "N = 256\n"
    "R = 5                    # coupling radius\n"
    "TAU = 12.0               # membrane leak time constant\n"
    "THRESH = 1.0             # firing threshold\n"
    "REF = 6                  # refractory steps after a spike\n"
    "SYN_DECAY = 0.8          # synaptic conductance decay per step\n"
    "BIAS = 0.02              # tonic background drive\n"
    "NOISE = 0.06             # random jitter on the input\n"
    "W_EXC = 1.8              # excitatory centre strength\n"
    "W_INH = 0.28             # inhibitory surround strength\n"
    "SIG_E = 1.8              # excitatory gaussian width\n"
    "SIG_I = 4.0              # inhibitory gaussian width\n"
    "LR = 0.006               # Hebbian learning rate\n"
    "HOMEO = 0.02             # homeostatic relaxation of weights toward 1\n"
    "W_MIN, W_MAX = 0.5, 2.0",
))

for _name in ("p", "ref", "fired", "g", "w", "kernel", "pixels"):
    frag(((1, 2), f"{_name} = None"))

# --- init ----------------------------------------------------------------------------

INIT_SIM = '''def init_sim(arch=None):
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
    build_kernel()'''

frag(((1, 3), INIT_SIM))

# --- the coupling kernel -------------------------------------------------------------

MEXICAN_HAT = '''def mexican_hat():
    """Pure numpy: a difference-of-gaussians — excite close neighbours, inhibit distant ones."""
    k = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            r2 = di * di + dj * dj
            k[di + R, dj + R] = (W_EXC * np.exp(-r2 / (2 * SIG_E ** 2))
                                 - W_INH * np.exp(-r2 / (2 * SIG_I ** 2)))
    k[R, R] = 0.0  # a neuron does not synapse onto itself
    return k'''

frag(((1, 4), MEXICAN_HAT))
frag(((1, 4), "def build_kernel():\n    kernel.from_numpy(mexican_hat())"))

# --- seed ----------------------------------------------------------------------------

APPLY_SEED = '''@ti.kernel
def apply_seed(rng_seed: ti.i32):
    for i, j in p:
        p[i, j] = 0.2 * ti.random()
        ref[i, j] = 0
        g[i, j] = 0.0
        w[i, j] = 1.0
        fired[i, j] = 0'''

frag(((1, 5), APPLY_SEED))

# --- the neuron: leaky integrate-and-fire (+ coupling in chapter 2) -------------------

SYNAPSE_V1 = '''@ti.kernel
def synapse_step(sx: ti.i32, sy: ti.i32, srad: ti.f32, si: ti.f32):
    """A neuron integrates its input and fires past threshold — unless it is mid-refractory pause."""
    for i, j in p:
        ext = 0.0
        if (i - sx) ** 2 + (j - sy) ** 2 < srad * srad:
            ext = si
        f = 0
        if ref[i, j] > 0:
            ref[i, j] -= 1
            p[i, j] = 0.0
        else:
            drive = ext + BIAS + NOISE * (ti.random() - 0.5)
            pp = p[i, j] + (drive - p[i, j] / TAU)
            if pp >= THRESH:
                p[i, j] = 0.0
                ref[i, j] = REF
                f = 1
            else:
                p[i, j] = ti.max(pp, 0.0)
        fired[i, j] = f'''

SYNAPSE_V2 = '''@ti.kernel
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
        fired[i, j] = f'''

frag(((1, 5), SYNAPSE_V1), ((2, 1), SYNAPSE_V2))

# --- the spike trace (chapter 2), scaled by plastic weight (chapter 3) ----------------

COND_V1 = '''@ti.kernel
def conductance_step():
    for i, j in g:
        g[i, j] = g[i, j] * SYN_DECAY + fired[i, j]'''

COND_V2 = '''@ti.kernel
def conductance_step():
    """A spike leaves a decaying trace, scaled by the neuron's plastic weight — its 'loudness'."""
    for i, j in g:
        g[i, j] = g[i, j] * SYN_DECAY + fired[i, j] * w[i, j]'''

frag(((2, 1), COND_V1), ((3, 1), COND_V2))

# --- plasticity (chapter 3) ----------------------------------------------------------

PLASTICITY = '''@ti.kernel
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
        w[i, j] = ti.min(ti.max(w[i, j] + dw, W_MIN), W_MAX)'''

frag(((3, 1), PLASTICITY))

# --- the tick: three versions --------------------------------------------------------

STEP_V1 = """def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)"""

STEP_V2 = """def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)
    conductance_step()"""

STEP_V3 = """def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)
    conductance_step()
    plasticity_step()"""

frag(((1, 6), STEP_V1), ((2, 2), STEP_V2), ((3, 2), STEP_V3))

FIRING_RATE = '''def firing_rate():
    """Pure numpy: fraction of neurons that spiked this step."""
    return float(fired.to_numpy().mean())'''

frag(((1, 6), FIRING_RATE))

MEAN_WEIGHT = '''def mean_weight(i0, i1, j0, j1):
    """Pure numpy: average plastic weight over a rectangular patch."""
    return float(w.to_numpy()[i0:i1, j0:j1].mean())'''

frag(((3, 2), MEAN_WEIGHT))

# --- render: two versions ------------------------------------------------------------

RENDER_V1 = """@ti.kernel
def render():
    for i, j in pixels:
        m = ti.min(ti.max(p[i, j], 0.0), 1.0)
        s = 1.0 * fired[i, j]
        pixels[i, j] = ti.Vector([0.1 + 0.9 * s, 0.15 + 0.5 * m, 0.2 + 0.3 * s])"""

RENDER_V2 = """@ti.kernel
def render():
    for i, j in pixels:
        a = ti.min(g[i, j], 1.0)
        wn = (w[i, j] - W_MIN) / (W_MAX - W_MIN)
        cool = ti.Vector([0.15, 0.45, 0.95])
        warm = ti.Vector([1.0, 0.35, 0.55])
        col = (cool * (1.0 - wn) + warm * wn) * a + ti.Vector([0.05, 0.05, 0.08])
        pixels[i, j] = ti.min(col, 1.0)"""

frag(((1, 6), RENDER_V1), ((2, 2), RENDER_V2))

# --- main ----------------------------------------------------------------------------

MAIN = '''def main():
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
        gui.show()'''

frag(((1, 6), MAIN))

frag(((1, 6), 'if __name__ == "__main__":\n    main()'))

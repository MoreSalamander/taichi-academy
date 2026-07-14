// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["28-digital-brain"] = {
  project: "28-digital-brain",
  title: "Digital Brain",
  pitch: "A quarter-million spiking neurons on a sheet. Wire each to its neighbours with a pinch of excitation and a ring of inhibition, add a refractory pause, and the tissue erupts into travelling waves — then learns, strengthening the paths it uses most.",
  tier: "hard",
  language: "Python",
  file: "digital_brain.py",
  chapters: [
    {
      id: 1, title: "A neuron that spikes",
      build: "the leaky integrate-and-fire model on a grid — membrane, threshold, and the refractory pause — driven by a stimulus you paint.",
      beat: "Poke the sheet and a patch of neurons flickers with spikes, then falls silent — no spread yet.",
      steps: [
        { title: "The floor beneath everything", adding: "the docstring and imports.",
          code: `"""Digital Brain: a sheet of spiking neurons — local excitation, surround inhibition, a
refractory pause — self-organizes into travelling waves, and Hebbian plasticity carves pathways."""
import numpy as np
import taichi as ti`,
          does: "The first Arc 7 capstone models the organ that models everything else. A real cortex is billions of neurons, each a leaky bag of charge that fires when it fills up; the magic is entirely in how they're wired to each other. We build a 256x256 sheet — 65,536 neurons — where every neuron runs the same tiny rule, and watch brain-like behaviour fall out of the connections.",
          why: "This is the emergence thesis of the whole curriculum at its most literal: no neuron 'knows' about waves or memory. Those are what a wired-up crowd of dumb integrate-and-fire cells DOES. numpy builds the wiring diagram once; Taichi runs 65k neurons in lockstep every frame.",
          see: "Runs clean.",
          checkpoint: "python3 digital_brain.py returns silently.",
          recovery: ["Usual venv setup: source .venv/bin/activate, then run from the project folder."] },
        { title: "The dials and the fields", adding: "every neuron parameter and field.",
          code: `N = 256
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
pixels = None`,
          does: "Each neuron carries a membrane potential p (charge, leaking away with time constant TAU), a refractory countdown ref, a fired flag, a synaptic trace g (its recent spiking, for chapter 2's coupling), and a plastic weight w (chapter 3). kernel is the shared wiring stamp — how strongly a neuron at each offset influences its neighbours. Most of these dials won't be touched until later chapters; they're all here so the physics never has to be re-plumbed.",
          why: "The whole model is a handful of scalars per cell. That frugality is the point: a neuron is genuinely almost this simple, and the richness is in R (how far each reaches), the excitation/inhibition balance (W_EXC vs W_INH), and REF (the enforced silence after a spike). Those three numbers decide whether the sheet does nothing, seizes, or ripples with waves.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["THRESH = 1 sets the scale; every other magnitude (BIAS, NOISE, the weights) is relative to 'how much charge is one spike away.'", "REF is measured in steps, not milliseconds — this is a discrete-time model, one tick per frame."] },
        { title: "Allocate once", adding: "init_sim.",
          code: `def init_sim(arch=None):
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
    build_kernel()`,
          does: "The familiar allocate-once pattern. Every per-neuron field is N x N; the kernel is a tiny (2R+1) x (2R+1) stamp shared by all neurons. init_sim finishes by building that kernel — the wiring diagram — even though nothing reads it until chapter 2.",
          why: "One coupling kernel for the entire sheet is the key economy: instead of storing a synapse list per neuron (billions of numbers), every neuron uses the SAME little stamp of offset-weights, applied around its own location. That's a convolution — and it's why a quarter-million coupled neurons cost almost nothing.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["If build_kernel raises a NameError, it just isn't defined yet — it arrives in the next step.", "kernel is shape (11, 11) for R=5 — small enough to reuse everywhere, which is the whole idea."] },
        { title: "The wiring stamp", adding: "the Mexican-hat kernel.",
          code: `def mexican_hat():
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
    kernel.from_numpy(mexican_hat())`,
          does: "The wiring: a difference of two gaussians. A narrow, strong positive bump (excite your close neighbours) sits inside a wide, weaker negative dish (inhibit the ones farther out). The centre is zeroed — no neuron drives itself. This 'Mexican hat' is stamped identically around every neuron.",
          why: "Local excitation plus broader inhibition is one of the most important motifs in all of neuroscience — it's how the retina sharpens edges, how the cortex forms orientation columns, and, here, how a smear of activity organizes into a crisp travelling front instead of a formless blob. The excited centre pushes activity outward; the inhibitory surround stops it from filling everything at once, carving the leading edge of a wave. We build the wiring now; chapter 2 connects the neurons to it.",
          see: "Still assembling — the kernel exists but no neuron reads it yet.",
          checkpoint: "No red text.",
          recovery: ["SIG_E < SIG_I (1.8 vs 4.0) is what makes it a hat and not a bowl: excitation must be tighter than inhibition.", "If you later get a seizure (everything firing), the surround inhibition is too weak — raise W_INH or widen SIG_I."] },
        { title: "Integrate and fire", adding: "the seed and the neuron update.",
          code: `@ti.kernel
def apply_seed(rng_seed: ti.i32):
    for i, j in p:
        p[i, j] = 0.2 * ti.random()
        ref[i, j] = 0
        g[i, j] = 0.0
        w[i, j] = 1.0
        fired[i, j] = 0
@ti.kernel
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
        fired[i, j] = f`,
          does: "The leaky integrate-and-fire neuron, the workhorse of computational neuroscience. Each step a neuron adds its input drive and leaks a fraction (p/TAU) of what it holds. If that pushes it over THRESH, it SPIKES: emits a 1, resets to zero, and enters a REF-step refractory pause during which it's pinned at zero and deaf to input. For now the only input is ext — a stimulus disc under the cursor — plus a whisper of bias and noise.",
          why: "Two details do all the work. The LEAK makes the neuron a forgetful integrator — steady input builds toward a ceiling, and a spike needs input arriving faster than it drains, so timing matters. The REFRACTORY pause is what will make waves possible in chapter 2: a neuron that just fired can't immediately re-fire, so a wave can't bounce backward into the tissue it came from — it's forced to keep moving forward. That single rule is the difference between a rippling brain and a flickering mess.",
          see: "Assembling — the loop that runs this is next.",
          checkpoint: "No red text. apply_seed and synapse_step compile.",
          recovery: ["The refractory branch pins p to 0 and decrements ref — no integration happens while a neuron is 'resting.'", "ext only lands inside the stimulus disc; everywhere else the neuron drifts on bias and noise alone, staying quiet."] },
        { title: "Poke it and watch", adding: "the tick, a spike render, and the main loop.",
          code: `def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)
def firing_rate():
    """Pure numpy: fraction of neurons that spiked this step."""
    return float(fired.to_numpy().mean())
@ti.kernel
def render():
    for i, j in pixels:
        m = ti.min(ti.max(p[i, j], 0.0), 1.0)
        s = 1.0 * fired[i, j]
        pixels[i, j] = ti.Vector([0.1 + 0.9 * s, 0.15 + 0.5 * m, 0.2 + 0.3 * s])
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
    main()`,
          does: "step is just the neuron update for now. render draws each neuron: a dim glow for its sub-threshold membrane, a bright flash the moment it spikes. main lets you drag the mouse to inject a stimulus disc (a pacemaker) and press [r] to reset.",
          why: "This is the control experiment before the interesting part: neurons that integrate and fire but AREN'T yet wired to each other. Poking them proves the single-neuron dynamics work — and makes vivid what's missing. Without coupling, a spike is a private event; it can't recruit its neighbours, so activity can't travel. That absence is exactly what chapter 2 fills in.",
          see: "Drag the mouse across the sheet: under the cursor, a disc of neurons flickers with spikes as fast as their refractory pause allows — a shimmering patch that vanishes the instant you lift the mouse. Beautiful, but inert: the spikes stay trapped where you put them, never spreading. It's a field of disconnected cells.",
          checkpoint: "An interactive field of independent spiking neurons. Chapter 1 complete.",
          recovery: ["If the whole screen twinkles without any stimulus, NOISE is too high — it's randomly tipping neurons over threshold.", "Nothing spreading is CORRECT here: there's no coupling yet, so each neuron is an island."] }
      ]
    },
    {
      id: 2, title: "Waves across the sheet",
      build: "the lateral coupling — each neuron feels its neighbours' recent spikes through the Mexican-hat kernel — and the sheet bursts into self-sustaining travelling waves.",
      beat: "One poke and rings of activity ripple outward across the whole sheet, on and on, long after you let go.",
      steps: [
        { title: "Wire them together", adding: "the coupled neuron update and the spike trace.",
          code: `@ti.kernel
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
    for i, j in g:
        g[i, j] = g[i, j] * SYN_DECAY + fired[i, j]`,
          does: "The one addition that changes everything: syn. Before integrating, each neuron sweeps the (2R+1)x(2R+1) block around it, multiplying each neighbour's recent spiking g by the kernel weight for that offset, and sums it into its drive. conductance_step maintains that g — a per-neuron spike trace that jumps up when a neuron fires and decays a bit each step, so a spike keeps 'shouting' to its neighbours for a few frames after it happens.",
          why: "This is a convolution of the spike-trace field with the Mexican-hat kernel, done the direct way — every neuron reading its neighbourhood. The decaying trace g matters as much as the kernel: an instantaneous spike would be gone before its neighbours could integrate it, but a trace that lingers for a few frames gives the excitation time to build up next door and tip those neurons over. Coupling strength times persistence is what lets a spike RECRUIT its neighbours instead of just flashing alone.",
          see: "Assembling — the tick still needs to run the trace; next step.",
          checkpoint: "No red text. The coupled synapse_step and conductance_step compile.",
          recovery: ["The neighbour loops are plain range loops (not ti.static), so Taichi keeps them as a real loop instead of unrolling 121 copies — much faster to compile.", "g reads from the PREVIOUS step's spikes (it's updated after the neurons fire), so every neuron sees a consistent snapshot — no neuron reacts to a spike that happens in the same instant."] },
        { title: "Run the trace — and it lives", adding: "the fuller tick and the wave render.",
          code: `def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)
    conductance_step()
@ti.kernel
def render():
    for i, j in pixels:
        a = ti.min(g[i, j], 1.0)
        wn = (w[i, j] - W_MIN) / (W_MAX - W_MIN)
        cool = ti.Vector([0.15, 0.45, 0.95])
        warm = ti.Vector([1.0, 0.35, 0.55])
        col = (cool * (1.0 - wn) + warm * wn) * a + ti.Vector([0.05, 0.05, 0.08])
        pixels[i, j] = ti.min(col, 1.0)`,
          does: "step now updates the spike trace after the neurons fire, closing the loop: spikes feed g, g feeds next step's drive, drive makes new spikes. render switches to showing g — the wave of recent activity — tinted by weight (uniform for now, since every weight is still 1). The dim base keeps quiet tissue visible.",
          why: "With the loop closed, the sheet becomes an excitable medium — the same class of system as a heart muscle or a forest fire. A single stimulus lights a patch; the patch excites its ring of neighbours, which fire and excite THEIR ring, while the refractory pause behind the front stops the wave from flowing backward. The result is a self-propagating wave that outlives its trigger. Tune the excitation up and it never stops; this is, in miniature, the mechanism behind the travelling waves real cortex shows in sleep and under anaesthesia.",
          see: "Give the sheet one poke and let go: concentric rings of activity bloom outward from the spot and roll across the entire sheet, wave after wave, sustaining themselves indefinitely from that one pacemaker. Poke a second spot and its rings collide with the first's, breaking into spirals. The tissue is alive with motion you no longer have to feed.",
          checkpoint: "Self-sustaining travelling waves from a single stimulus. Chapter 2 complete.",
          recovery: ["If the waves die out, coupling is too weak or the trace decays too fast: nudge W_EXC up or SYN_DECAY toward 1.", "If the whole sheet strobes on and off together, it's synchronizing instead of forming waves — stronger surround inhibition (W_INH) breaks the symmetry into travelling fronts."] }
      ]
    },
    {
      id: 3, title: "A brain that learns",
      build: "Hebbian plasticity — weights that grow where neurons fire together and relax by homeostasis — so the paths the brain uses most grow strongest.",
      beat: "Stimulate one spot again and again, and its pathways brighten and strengthen: the sheet remembers where it's been driven.",
      steps: [
        { title: "Fire together, wire together", adding: "the weighted trace and the plasticity rule.",
          code: `@ti.kernel
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
        w[i, j] = ti.min(ti.max(w[i, j] + dw, W_MIN), W_MAX)`,
          does: "Two changes give the sheet a memory. conductance_step now scales each neuron's spike trace by its weight w — a strengthened neuron 'shouts louder' to its neighbours. plasticity_step is the learning: a neuron's weight grows (LR term) when it fires WHILE its neighbourhood is active — the Hebbian condition, fire-together-wire-together — and every weight is gently pulled back toward 1 (the HOMEO term) so strengths can't run away. Weights are clamped to a sane band.",
          why: "This is Hebb's 1949 rule, the oldest idea in learning theory and still the beating heart of it: synapses that participate in successful firing get stronger. The homeostatic decay is the essential counterweight — without it every active synapse would climb to the ceiling and the network would seize; with it, weights settle at a level that reflects how OFTEN a neuron is usefully active. A spot driven over and over wins the tug-of-war between potentiation and decay and stays strong; quiet tissue relaxes back to baseline. That balance — plasticity pulling up, homeostasis pulling down — is how a real brain learns without frying itself.",
          see: "Assembling — the tick needs to run the plasticity, next step.",
          checkpoint: "No red text. The weighted trace and plasticity_step compile.",
          recovery: ["The Hebbian term needs BOTH fired[i,j] and active neighbours (local) to be nonzero — a lone spike in silence barely strengthens.", "If weights all pin to W_MAX, HOMEO is too weak relative to LR — the decay must be able to hold back a busy neuron."] },
        { title: "The learning brain", adding: "the full tick and a weight probe.",
          code: `def step(sx=-1000, sy=-1000, srad=0.0, si=0.0):
    synapse_step(sx, sy, srad, si)
    conductance_step()
    plasticity_step()
def mean_weight(i0, i1, j0, j1):
    """Pure numpy: average plastic weight over a rectangular patch."""
    return float(w.to_numpy()[i0:i1, j0:j1].mean())`,
          does: "The finished tick runs all three stages every frame: the neurons fire (synapse_step), their traces update (conductance_step), and their weights learn (plasticity_step). mean_weight is a probe for reading how much a patch has strengthened — the handle the tests and the headless check use to confirm learning happened.",
          why: "Order matters: neurons fire based on the CURRENT weights, then those weights adjust based on who just fired — perception then learning, every tick. Run it with a fixed pacemaker and you can watch the driven region's weights climb well above a distant corner the waves reach only later: the brain is recording, in its synapses, where it has been stimulated. That's the capstone idea — structure (waves) and memory (weights) emerging together from the same local rule, no supervisor anywhere. This is the deepest 'more is different' in the curriculum, and it closes the first of the three dream projects.",
          see: "Hold the stimulus on one spot for a while: that region's waves grow visibly brighter and warmer than the rest as its weights strengthen, and even after you move away, activity keeps preferring the path you carved. Drag a slow loop and you can almost draw a strengthened circuit into the tissue. The sheet doesn't just react — it remembers.",
          checkpoint: "A spiking sheet that forms travelling waves AND learns the pathways it uses. Project 28 complete — the first Arc 7 capstone.",
          recovery: ["If learning seems to do nothing, check that conductance_step multiplies by w (so strengthened neurons actually project harder) and that step calls plasticity_step.", "mean_weight over the driven patch should exceed a far corner after ~100 steps — if not, LR is too small or the pacemaker too weak to fire that patch more than the background waves."] }
      ]
    }
  ]
};

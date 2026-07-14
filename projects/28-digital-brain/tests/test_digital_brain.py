import numpy as np

import digital_brain as db


def pacemaker(n, cx=None, cy=None, si=0.9):
    cx = db.N // 2 if cx is None else cx
    cy = db.N // 2 if cy is None else cy
    for _ in range(n):
        db.step(cx, cy, 5.0, si)


# --- the coupling kernel -------------------------------------------------------------


def test_mexican_hat_excites_near_inhibits_far():
    k = db.mexican_hat()
    assert k[db.R, db.R] == 0.0, "a neuron has no self-synapse"
    near = k[db.R + 1, db.R]          # one cell away
    far = k[db.R, 0]                  # R cells away (the rim)
    assert near > 0.0, "close neighbours excite"
    assert far < near, "distant neighbours are relatively inhibitory"
    assert k.shape == (2 * db.R + 1, 2 * db.R + 1)


# --- the seed ------------------------------------------------------------------------


def test_seed_is_a_blank_resting_sheet():
    assert np.allclose(db.w.to_numpy(), 1.0), "every weight starts at baseline 1"
    assert (db.ref.to_numpy() == 0).all(), "no neuron starts refractory"
    assert (db.fired.to_numpy() == 0).all(), "no spikes yet"
    p = db.p.to_numpy()
    assert p.min() >= 0.0 and p.max() < db.THRESH, "membranes start below threshold"


# --- spiking + refractory ------------------------------------------------------------


def test_a_supra_threshold_neuron_fires_then_rests():
    """Push one neuron above threshold: it spikes, then sits out its refractory pause."""
    pp = db.p.to_numpy()
    pp[100, 100] = 2.0
    db.p.from_numpy(pp)
    db.synapse_step(-1000, -1000, 0.0, 0.0)
    assert db.fired.to_numpy()[100, 100] == 1, "it crossed threshold and spiked"
    assert db.ref.to_numpy()[100, 100] == db.REF, "it entered the refractory period"
    db.synapse_step(-1000, -1000, 0.0, 0.0)
    assert db.fired.to_numpy()[100, 100] == 0, "it cannot spike again while refractory"
    assert db.p.to_numpy()[100, 100] == 0.0, "its membrane is pinned during the pause"


def test_stimulus_makes_neurons_fire():
    db.step(db.N // 2, db.N // 2, 6.0, 1.5)
    assert db.fired.to_numpy().sum() > 0, "a strong stimulus ignites the sheet"


# --- the headline: self-sustaining waves ---------------------------------------------


def test_activity_self_sustains_into_waves():
    """Drive a pacemaker and the sheet lights up with ongoing activity — not a single blip
    that dies, but sustained travelling waves (a healthy fraction firing every step)."""
    pacemaker(150)
    rate = db.firing_rate()
    assert rate > 0.02, f"the sheet should keep firing, got {rate:.3f}"
    assert rate < 0.6, f"but not seize into everything-at-once, got {rate:.3f}"
    assert np.all(np.isfinite(db.p.to_numpy()))


# --- plasticity ----------------------------------------------------------------------


def test_hebbian_learning_strengthens_the_driven_site():
    """The repeatedly-stimulated pacemaker learns: its weights climb above a distant, quiet
    corner the waves reach only later."""
    pacemaker(120)
    center = db.mean_weight(db.N // 2 - 12, db.N // 2 + 12, db.N // 2 - 12, db.N // 2 + 12)
    corner = db.mean_weight(0, 20, 0, 20)
    assert center > corner + 0.1, f"the driven site should outlearn the corner: {center:.3f} vs {corner:.3f}"
    assert center <= db.W_MAX and corner >= db.W_MIN, "weights stay clamped in range"


def test_weights_relax_without_activity():
    """With no spikes at all, homeostasis holds every weight at its baseline of 1."""
    wv = db.w.to_numpy()
    wv[:] = 1.5
    db.w.from_numpy(wv)
    # freeze spiking by making the tissue silent: no fire, so dw = -HOMEO*(w-1) each step
    for _ in range(250):
        db.plasticity_step()
    assert abs(db.mean_weight(0, db.N, 0, db.N) - 1.0) < 0.05, "quiet weights decay back to baseline"


# --- render --------------------------------------------------------------------------


def test_render_is_finite_and_bounded():
    pacemaker(40)
    db.render()
    px = db.pixels.to_numpy()
    assert np.all(np.isfinite(px)) and px.min() >= 0.0 and px.max() <= 1.0

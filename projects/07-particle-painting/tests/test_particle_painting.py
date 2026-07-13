import numpy as np

import particle_painting as pp


def one(material, x=250.0, y=250.0, vx=0.0, vy=0.0, life=1.0, slot=0):
    pp.pos[slot] = [x, y]
    pp.vel[slot] = [vx, vy]
    pp.life[slot] = life
    pp.material[slot] = material


# --- emission / ring buffer -----------------------------------------------------------


def test_emit_fills_slots_and_advances_cursor():
    pp.emit(0.5, 0.5, pp.FIRE)
    life = pp.life.to_numpy()
    mat = pp.material.to_numpy()
    assert np.all(life[: pp.EMIT_RATE] == 1.0)
    assert np.all(mat[: pp.EMIT_RATE] == pp.FIRE)
    assert np.all(life[pp.EMIT_RATE :] == 0.0)
    assert pp.cursor[None] == pp.EMIT_RATE


def test_emit_wraps_around_the_buffer():
    pp.cursor[None] = pp.MAX_PARTICLES - 5
    pp.emit(0.5, 0.5, pp.SPARKS)
    assert pp.cursor[None] == pp.EMIT_RATE - 5
    life = pp.life.to_numpy()
    assert life[-5:].sum() == 5.0, "the last 5 slots wrap to the front of the buffer"
    assert life[: pp.EMIT_RATE - 5].sum() == pp.EMIT_RATE - 5


def test_emitted_positions_land_at_the_cursor():
    pp.emit(0.25, 0.75, pp.WATER)
    positions = pp.pos.to_numpy()[: pp.EMIT_RATE]
    assert np.allclose(positions[:, 0], 0.25 * pp.N)
    assert np.allclose(positions[:, 1], 0.75 * pp.N)


# --- per-material motion -----------------------------------------------------------


def test_fire_rises():
    one(pp.FIRE, y=100.0, vy=0.1)
    for _ in range(20):
        pp.update()
    assert pp.pos[0][1] > 100.0
    assert pp.vel[0][1] > 0.1, "buoyancy keeps adding upward velocity"


def test_sparks_bounce_off_the_floor():
    one(pp.SPARKS, y=1.0, vy=-5.0)
    pp.update()
    assert pp.pos[0][1] == 0.0
    assert pp.vel[0][1] > 0.0, "downward velocity flips sign on floor contact"


def test_water_sticks_to_the_floor():
    one(pp.WATER, y=1.0, vy=-5.0)
    pp.update()
    assert pp.pos[0][1] == 0.0
    assert pp.vel[0][0] == 0.0 and pp.vel[0][1] == 0.0, "water stops dead, no bounce"


def test_particles_die_when_they_leave_the_sides():
    one(pp.FIRE, x=1.0, y=250.0, vx=-50.0, vy=0.0)
    pp.update()
    assert pp.life[0] == 0.0


def test_life_decays_and_frozen_once_dead():
    one(pp.SPARKS, y=250.0)
    pp.update()
    first = pp.life[0]
    assert 0.0 < first < 1.0
    pp.life[0] = 0.0
    before = pp.pos[0][1]
    pp.update()
    assert pp.pos[0][1] == before, "a dead particle (life <= 0) is skipped entirely"


# --- rendering -----------------------------------------------------------------------


def test_fade_multiplies_every_pixel():
    pp.pixels.fill(0.4)
    pp.fade()
    px = pp.pixels.to_numpy()
    assert np.allclose(px, 0.4 * pp.FADE, atol=1e-5)


def test_splat_stays_finite_and_bounded():
    pp.emit(0.5, 0.5, pp.FIRE)
    for _ in range(30):
        pp.update()
        pp.fade()
        pp.splat()
        pp.clamp_pixels()
    px = pp.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0


def test_material_colors_are_visually_distinct():
    """Fire skews red, water skews blue, smoke stays gray — paint one of each and check."""
    results = {}
    for material in (pp.FIRE, pp.WATER, pp.SMOKE):
        pp.clear()
        cx, cy = pp.N // 2, pp.N // 2
        one(material, x=float(cx), y=float(cy), vx=0.0, vy=0.0)
        pp.splat()
        pp.clamp_pixels()
        results[material] = tuple(pp.pixels[cx, cy])
    assert results[pp.FIRE][0] > results[pp.FIRE][2], "fire: red channel beats blue"
    assert results[pp.WATER][2] > results[pp.WATER][0], "water: blue channel beats red"
    g = results[pp.SMOKE]
    assert abs(g[0] - g[1]) < 1e-5 and abs(g[1] - g[2]) < 1e-5, "smoke is neutral gray"


# --- reset -----------------------------------------------------------------------------


def test_clear_resets_particles_pixels_and_cursor():
    pp.emit(0.5, 0.5, pp.FIRE)
    pp.splat()
    pp.clear()
    assert pp.life.to_numpy().sum() == 0.0
    assert pp.pixels.to_numpy().sum() == 0.0
    assert pp.cursor[None] == 0

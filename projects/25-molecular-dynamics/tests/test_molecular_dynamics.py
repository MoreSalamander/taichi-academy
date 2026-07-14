import numpy as np

import molecular_dynamics as md


def run(n, thermo=True):
    for _ in range(n):
        md.step(thermo=thermo)


def isolate_pair(r):
    """Place atom 0 and atom 1 at separation r; pile the rest at one far point so the
    r2 < 1e-4 guard drops them (co-located atoms exert no force) and none reach atom 0."""
    p = np.full((md.N, 2), 30.0, dtype=np.float32)
    p[0] = [5.0, 5.0]
    p[1] = [5.0 + r, 5.0]
    md.pos.from_numpy(p)
    md.build_grid()
    md.compute_forces()


# --- pure numpy generation ---------------------------------------------------------


def test_lattice_positions_fill_the_box():
    xy = md.lattice_positions(md.N, md.L)
    assert xy.shape == (md.N, 2)
    assert xy.min() >= 0.0 and xy.max() <= md.L
    # distinct cells
    assert len({tuple(np.round(p, 3)) for p in xy}) == md.N


def test_maxwell_velocities_temperature_and_zero_momentum():
    rng = np.random.default_rng(1)
    v = md.maxwell_velocities(20000, 2.0, rng)
    ke_per_atom = 0.5 * (v**2).sum() / 20000
    assert abs(ke_per_atom - 2.0) < 0.05, f"KE/atom should equal the temperature, got {ke_per_atom:.3f}"
    assert np.abs(v.mean(axis=0)).max() < 1e-4, "net momentum removed"


# --- the Lennard-Jones force ---------------------------------------------------------


def test_lj_repels_when_too_close():
    isolate_pair(1.0)  # inside the potential minimum (~1.122)
    fx = md.acc.to_numpy()[0, 0]
    assert fx < -1.0, "an atom crowded from its right (+x) is shoved left (repulsion)"


def test_lj_attracts_at_medium_range():
    isolate_pair(1.5)  # past the minimum, inside the cutoff
    fx = md.acc.to_numpy()[0, 0]
    assert fx > 0.05, "an atom pulls toward a neighbor at medium range (attraction)"


def test_lj_vanishes_beyond_the_cutoff():
    isolate_pair(3.0)  # > RCUT = 2.5
    assert abs(md.acc.to_numpy()[0, 0]) < 1e-5, "no force beyond the cutoff"


def test_force_minimum_is_near_2_to_the_1_6():
    """The LJ force crosses zero at r = 2^(1/6) sigma — repulsive inside, attractive out."""
    isolate_pair(2 ** (1 / 6) * md.SIGMA)
    assert abs(md.acc.to_numpy()[0, 0]) < 0.2, "force is ~zero at the potential minimum"


# --- integrator + spatial hash --------------------------------------------------------


def test_grid_buckets_every_atom_once():
    md.build_grid()
    assert md.cell_count.to_numpy().sum() == md.N
    assert sorted(md.sorted_idx.to_numpy().tolist()) == list(range(md.N))


def test_verlet_conserves_energy_at_equilibrium():
    """Velocity-Verlet (project 16's family) conserves total energy: after equilibrating,
    running with the thermostat OFF keeps temperature stable — no systematic drift."""
    md.apply_seed(rng_seed=1, temperature=0.6)
    md.temp_target[None] = 0.6
    run(400)  # equilibrate with thermostat
    temps = []
    for _ in range(300):
        md.step(thermo=False)  # pure NVE
        temps.append(md.measure_temp())
    temps = np.array(temps)
    assert abs(temps.mean() - 0.6) < 0.15, f"NVE temperature should hold near 0.6, got {temps.mean():.3f}"
    assert temps.std() < 0.1, "no runaway drift"


def test_atoms_stay_in_the_periodic_box():
    run(200)
    p = md.pos.to_numpy()
    assert np.all(np.isfinite(p))
    assert p.min() >= 0.0 and p.max() < md.L


# --- temperature & thermostat ---------------------------------------------------------


def test_measure_temp_is_mean_kinetic_energy():
    v = np.zeros((md.N, 2), dtype=np.float32)
    v[:, 0] = 2.0  # every atom speed 2 -> KE/atom = 0.5*4 = 2
    md.vel.from_numpy(v)
    assert abs(md.measure_temp() - 2.0) < 1e-4


def test_thermostat_heats_and_cools():
    md.apply_seed(rng_seed=1, temperature=1.0)
    md.temp_target[None] = 2.5
    run(300)
    hot = md.measure_temp()
    md.temp_target[None] = 0.3
    run(300)
    cold = md.measure_temp()
    assert hot > 1.8, f"thermostat should drive up to hot target, got {hot:.2f}"
    assert cold < 0.8, f"thermostat should drive down to cold target, got {cold:.2f}"


# --- the headline: crystallization ----------------------------------------------------


def test_cooling_crystallizes_the_liquid():
    """The payoff: a hot disordered liquid has few fully-coordinated atoms; quench it and
    a hexagonal lattice forms — the fraction of 6-neighbor atoms jumps."""
    md.apply_seed(rng_seed=2, temperature=1.0)
    md.temp_target[None] = 3.0
    run(200)  # melt to a hot liquid
    hot = md.crystalline_fraction()
    for target in np.linspace(2.5, 0.1, 12):
        md.temp_target[None] = target
        run(50)
    cold = md.crystalline_fraction()
    assert hot < 0.3, f"a hot liquid should be mostly disordered, got {hot:.2f}"
    assert cold > hot * 1.8, f"cooling should crystallize: {hot:.2f} -> {cold:.2f}"


# --- interaction / render ------------------------------------------------------------


def test_heat_gun_speeds_up_nearby_atoms():
    md.apply_seed(rng_seed=1, temperature=0.5)
    p = md.pos.to_numpy()
    dx = np.abs(p[:, 0] - 0.5 * md.L)
    dy = np.abs(p[:, 1] - 0.5 * md.L)
    dx = np.minimum(dx, md.L - dx)
    dy = np.minimum(dy, md.L - dy)
    near = np.hypot(dx, dy) < md.HEAT_RADIUS
    speed_before = np.linalg.norm(md.vel.to_numpy()[near], axis=1).mean()
    md.heat(0.5, 0.5)
    speed_after = np.linalg.norm(md.vel.to_numpy()[near], axis=1).mean()
    assert abs(speed_after - speed_before * md.HEAT_BOOST) < 1e-4, "the heat gun scales local speeds by HEAT_BOOST"


def test_render_is_finite_and_bounded():
    run(100)
    md.render()
    px = md.pixels.to_numpy()
    assert np.all(np.isfinite(px))
    assert px.min() >= 0.0 and px.max() <= 1.0

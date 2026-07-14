// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["25-molecular-dynamics"] = {
  project: "25-molecular-dynamics",
  title: "Molecular Dynamics",
  pitch: "One pair force between atoms and a symplectic integrator — and matter melts, boils, freezes, and grows crystals, all from the bottom up.",
  tier: "medium",
  language: "Python",
  file: "molecular_dynamics.py",
  chapters: [
    {
      id: 1, title: "Atoms and a force law",
      build: "the Lennard-Jones pair force and velocity-Verlet on the spatial hash — a jiggling liquid.",
      beat: "A box of atoms jostles and jiggles — a liquid drop held together by nothing but one force curve.",
      steps: [
        { title: "The floor beneath everything", adding: "the docstring and imports.",
          code: `"""Molecular Dynamics: one pair force + Verlet, and matter melts, freezes, and crystallizes."""
import math
import numpy as np
import taichi as ti`,
          does: "This is the physics floor of the whole curriculum: real atoms, interacting through the Lennard-Jones potential — the standard model of a simple substance. Give atoms that one force law and a good integrator, and everything above emerges bottom-up: pressure, temperature, the three states of matter, melting, boiling, and crystals growing atom by atom. No rules for 'liquid' or 'solid' are written anywhere; they are what a box of atoms DOES.",
          why: "Arc 5 has climbed from ants to traffic to evolving brains to self-assembling cells — emergent order at ever-lower levels. This project reaches bedrock: the atoms those cells and creatures were metaphors for. It closes the arc by showing that even the phases of matter are emergent, not fundamental — a fitting last word on 'more is different.'",
          see: "Runs clean.",
          checkpoint: "python3 molecular_dynamics.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "Reduced units", adding: "the physics dials and every field.",
          code: `RES = 600
L = 40.0
N = 1400
SIGMA = 1.0
EPS = 1.0
RCUT = 2.5
COORD_SHELL = 1.3
DT = 0.004
GRID = 16
CELL = L / GRID
NCELLS = GRID * GRID
pos = None
vel = None
acc = None
coord = None
cell_count = None
cell_start = None
cell_cursor = None
sorted_idx = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, acc, coord, cell_count, cell_start, cell_cursor, sorted_idx, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pos = ti.Vector.field(2, ti.f32, shape=N)
    vel = ti.Vector.field(2, ti.f32, shape=N)
    acc = ti.Vector.field(2, ti.f32, shape=N)
    coord = ti.field(ti.i32, shape=N)
    cell_count = ti.field(ti.i32, shape=NCELLS)
    cell_start = ti.field(ti.i32, shape=NCELLS)
    cell_cursor = ti.field(ti.i32, shape=NCELLS)
    sorted_idx = ti.field(ti.i32, shape=N)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "SIGMA = EPS = 1: the atom's size and the depth of its attraction are both set to ONE, so everything is measured in atom-widths and bond-energies — 'reduced units,' the standard MD convention. Each atom carries position, velocity, and acceleration (velocity-Verlet needs all three), plus coord for its neighbor count. The cell_* / sorted_idx fields are project 06's spatial hash again — force computation needs each atom's nearby neighbors, and GRID = 16 makes each cell 2.5 wide, exactly the force cutoff RCUT.",
          why: "Reduced units aren't a shortcut, they're the professional standard: with SIGMA and EPS as your rulers, one simulation stands in for argon, krypton, or any simple fluid — you just multiply by that substance's real sigma and epsilon at the end. The physics is universal; the units are a costume you put on last.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["CELL = L/GRID = 2.5 = RCUT exactly — an atom's force partners always live in its own cell plus the 8 neighbors, so the 3x3 search is complete.", "acc is stored, not recomputed on the fly — velocity-Verlet reuses last step's acceleration, so it must persist between ticks."] },
        { title: "Lennard-Jones and Verlet", adding: "the seeders, the hash, the pair force, velocity-Verlet, a plain render, and the tick.",
          code: `def lattice_positions(n, box):
    """Pure numpy: n atoms on a square grid filling the box, jittered off perfection."""
    side = int(math.ceil(math.sqrt(n)))
    spacing = box / side
    xy = np.array([[(i + 0.5) * spacing, (j + 0.5) * spacing] for i in range(side) for j in range(side)])
    return xy[:n].astype(np.float32)
def maxwell_velocities(n, temperature, rng):
    """Pure numpy: gaussian velocities at a temperature, with net momentum removed."""
    v = rng.normal(0.0, math.sqrt(temperature), (n, 2)).astype(np.float32)
    v -= v.mean(axis=0)
    return v
def apply_seed(rng_seed=0, temperature=1.0):
    rng = np.random.default_rng(rng_seed)
    xy = lattice_positions(N, L) + rng.normal(0, 0.05, (N, 2)).astype(np.float32)
    pos.from_numpy(xy % L)
    vel.from_numpy(maxwell_velocities(N, temperature, rng))
    acc.fill(0.0)
    build_grid()
    compute_forces()
@ti.func
def flat_cell(p):
    ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
    cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
    return ci * GRID + cj
@ti.kernel
def count_cells():
    for c in cell_count:
        cell_count[c] = 0
    for p in pos:
        cell_count[flat_cell(p)] += 1
@ti.kernel
def prefix_sum():
    for _ in range(1):
        a = 0
        for c in range(NCELLS):
            cell_start[c] = a
            a += cell_count[c]
@ti.kernel
def scatter():
    for c in cell_cursor:
        cell_cursor[c] = cell_start[c]
    for p in pos:
        idx = flat_cell(p)
        slot = ti.atomic_add(cell_cursor[idx], 1)
        sorted_idx[slot] = p
def build_grid():
    count_cells()
    prefix_sum()
    scatter()
@ti.func
def wrapd(a, b):
    d = a - b
    if d > 0.5 * L:
        d -= L
    if d < -0.5 * L:
        d += L
    return d
@ti.kernel
def compute_forces():
    for p in pos:
        f = ti.Vector([0.0, 0.0])
        nc = 0
        ci = ti.min(ti.max(ti.cast(pos[p][0] / CELL, ti.i32), 0), GRID - 1)
        cj = ti.min(ti.max(ti.cast(pos[p][1] / CELL, ti.i32), 0), GRID - 1)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ni = (ci + di) % GRID
            nj = (cj + dj) % GRID
            nidx = ni * GRID + nj
            for s in range(cell_start[nidx], cell_start[nidx] + cell_count[nidx]):
                q = sorted_idx[s]
                if q != p:
                    dx = wrapd(pos[p][0], pos[q][0])
                    dy = wrapd(pos[p][1], pos[q][1])
                    r2 = dx * dx + dy * dy
                    if 1e-4 < r2 < RCUT * RCUT:
                        inv2 = SIGMA * SIGMA / r2
                        inv6 = inv2 * inv2 * inv2
                        fmag = 24.0 * EPS * (2.0 * inv6 * inv6 - inv6) / r2
                        f += fmag * ti.Vector([dx, dy])
                        if r2 < COORD_SHELL * COORD_SHELL:
                            nc += 1
        acc[p] = f
        coord[p] = nc
@ti.kernel
def half_kick():
    for p in vel:
        vel[p] += 0.5 * DT * acc[p]
@ti.kernel
def drift():
    for p in pos:
        newp = pos[p] + DT * vel[p]
        pos[p] = ti.Vector([newp[0] % L, newp[1] % L])
def step():
    half_kick()
    drift()
    build_grid()
    compute_forces()
    half_kick()
@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.03, 0.03, 0.05])
    for p in pos:
        xi = ti.cast(pos[p][0] / L * RES, ti.i32)
        yi = ti.cast(pos[p][1] / L * RES, ti.i32)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = ti.Vector([0.7, 0.8, 1.0])
def main():
    init_sim()
    apply_seed(temperature=1.0)
    gui = ti.GUI("Molecular Dynamics — taichi-academy", res=RES, background_color=0x08080F)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step()
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "The heart is compute_forces: for every close pair, the Lennard-Jones force fmag = 24*EPS*(2*(sigma/r)^12 - (sigma/r)^6)/r^2. That one curve does everything — it's HUGELY repulsive when atoms overlap (the r^12 term: atoms can't interpenetrate) and gently attractive at medium range (the r^6 term: atoms stick), crossing zero at the bond length r = 2^(1/6). step() is velocity-Verlet, the same symplectic-integrator family as project 16's leapfrog: half-kick the velocity, drift the position, recompute forces, half-kick again. maxwell_velocities seeds a temperature as random gaussian speeds — the microscopic meaning of heat.",
          why: "Velocity-Verlet, like leapfrog, is TIME-SYMMETRIC, and that's why an MD simulation can run for millions of steps without the energy drifting — the same lesson project 16 taught with orbits, now keeping a thousand atoms honest. Plain Euler here would heat the system up from nothing and 'boil' it artificially; the tests confirm this integrator holds energy steady. The whole reason molecular dynamics is trustworthy science rests on that one property.",
          see: "1,400 pale-blue atoms fill the box, jostling and jiggling and flowing around each other — a liquid, held together purely by the attractive tail of the force curve, kept apart purely by its repulsive core. Watch two atoms approach: they slow, hesitate at the bond length, and rebound. That hesitation is the potential well, visible.",
          checkpoint: "A jiggling liquid. Beat 1.",
          recovery: ["fmag's 2*(sigma/r)^12 term is repulsion, the -(sigma/r)^6 is attraction — drop the 12-power term and atoms collapse into a singularity.", "compute_forces uses pos[p] - pos[q] (force ON p points away from q when repulsive) — the sign convention is what makes overlap push apart, not together."] }
      ]
    },
    {
      id: 2, title: "Temperature is motion",
      build: "temperature as kinetic energy, a thermostat, and heat/cool controls — melting and freezing on command.",
      beat: "Turn up the heat and the crystal-liquid melts to a churning gas; turn it down and it stills.",
      steps: [
        { title: "Heat, defined", adding: "the kinetic-energy thermometer and the coordination-colored render (replace render).",
          code: `@ti.kernel
def measure_temp() -> ti.f32:
    ke = 0.0
    for p in vel:
        ke += 0.5 * vel[p].dot(vel[p])
    return ke / N
@ti.kernel
def render():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.03, 0.03, 0.05])
    for p in pos:
        xi = ti.cast(pos[p][0] / L * RES, ti.i32)
        yi = ti.cast(pos[p][1] / L * RES, ti.i32)
        c = coord[p]
        col = ti.Vector([0.3, 0.5, 1.0])
        if c >= 6:
            col = ti.Vector([1.0, 0.85, 0.3])
        elif c >= 4:
            col = ti.Vector([0.5, 0.9, 0.5])
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x, y = xi + di, yi + dj
            if 0 <= x < RES and 0 <= y < RES:
                pixels[x, y] = col`,
          does: "measure_temp is the single most important idea in thermodynamics, in three lines: temperature IS the average kinetic energy of the atoms. There is no separate 'heat' quantity — heat is just how fast the atoms happen to be moving, averaged. The render now colors each atom by its coordination number (computed back in compute_forces): blue when it has few close neighbors (fluid), gold when it has a full hexagonal shell of six (crystal).",
          why: "That temperature = mean kinetic energy is worth pausing on — it's the bridge from the microscopic (individual atom velocities) to the macroscopic (the number on a thermometer), and it's LITERAL, not a metaphor. When you feel heat, you are feeling atoms hit your skin faster. The coordination coloring is your microscope for the phase transitions coming next: watch the gold appear and you're watching a solid form.",
          see: "The liquid, now colored by local order — mostly greens and blues with flickers of transient gold where atoms momentarily pack tight. No control over it yet; that's the next step.",
          checkpoint: "A thermometer and a phase microscope. No red text.",
          recovery: ["measure_temp divides total KE by N — it's the per-atom average, the quantity that equals temperature in reduced units.", "coord was already computed in chapter 1's compute_forces; this render just finally uses it."] },
        { title: "A thermostat", adding: "the temperature target, the velocity-rescaling thermostat, and the thermostat-wired tick.",
          code: `THERMO_RATE = 0.05
TEMP_STEP = 0.1
TEMP_MIN = 0.02
TEMP_MAX = 4.0
temp_target = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pos, vel, acc, coord, cell_count, cell_start, cell_cursor, sorted_idx, temp_target, pixels
    temp_target = ti.field(ti.f32, shape=())
def apply_seed(rng_seed=0, temperature=1.0):
    rng = np.random.default_rng(rng_seed)
    xy = lattice_positions(N, L) + rng.normal(0, 0.05, (N, 2)).astype(np.float32)
    pos.from_numpy(xy % L)
    vel.from_numpy(maxwell_velocities(N, temperature, rng))
    acc.fill(0.0)
    temp_target[None] = temperature
    build_grid()
    compute_forces()
@ti.kernel
def thermostat(cur_temp: ti.f32):
    scale = 1.0
    if cur_temp > 1e-6:
        scale = ti.sqrt(temp_target[None] / cur_temp)
    s = 1.0 + THERMO_RATE * (scale - 1.0)
    for p in vel:
        vel[p] *= s
def step(thermo=True):
    half_kick()
    drift()
    build_grid()
    compute_forces()
    half_kick()
    if thermo:
        thermostat(measure_temp())`,
          does: "A thermostat controls temperature by controlling SPEED — since temperature is kinetic energy, to heat the system you scale every velocity UP, to cool it you scale DOWN. The rescale factor sqrt(target/current) would snap to the target instantly; THERMO_RATE = 0.05 eases only 5% of the way each tick, so heating and cooling look gradual and physical. step() gains a thermo flag (the tests flip it off to prove the raw dynamics conserve energy).",
          why: "Velocity rescaling is the simplest real thermostat (the Berendsen family), and it makes the temperature=motion idea tangible: there is no other knob to turn. You cannot add 'heat' to these atoms except by making them move faster, because that is all heat has ever been. Cooling toward zero literally means freezing the motion.",
          see: "Runs clean; the thermostat is holding the seed temperature steady, but nothing lets you CHANGE the target yet — one keypress away.",
          checkpoint: "A working thermostat. No red text.",
          recovery: ["The eased rate (1 + 0.05*(scale-1)) is what makes phase transitions watchable — snap straight to target and the melt happens in one invisible frame.", "step's thermo=True default keeps main working; tests pass thermo=False for pure energy-conserving dynamics."] },
        { title: "Melt it, freeze it", adding: "the up/down temperature keys and the temperature/crystalline HUD.",
          code: `def crystalline_fraction():
    """Pure numpy: fraction of atoms with a full hexagonal shell of 6 close neighbors."""
    return float((coord.to_numpy() == 6).mean())
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                temp_target[None] = min(temp_target[None] + TEMP_STEP, TEMP_MAX)
            elif e.key == ti.GUI.DOWN:
                temp_target[None] = max(temp_target[None] - TEMP_STEP, TEMP_MIN)
        gui.text(
            f"target T {temp_target[None]:.2f}   crystalline {crystalline_fraction() * 100:.0f}%",
            (0.02, 0.98), color=0xFFFFFF,
        )`,
          does: "Up and down arrows raise and lower the target temperature; the thermostat chases it. The HUD reports the target and the crystalline fraction — the share of atoms with a full six-neighbor shell, the numeric signature of a solid.",
          why: "Now you can drive matter through its phases by hand and watch the coloring respond: crank the heat and the gold vanishes as the ordered solid melts to churning blue liquid, then to a sparse hot gas; drop it and gold spreads as atoms lock into place. The crystalline-fraction readout quantifies what your eyes see — a phase transition, measured. Melting and freezing were never programmed; they are what this atom count at this density DOES at each temperature.",
          see: "Hold the up-arrow: the tissue of atoms loosens, gold fades to green fades to blue, and the whole box starts to churn and expand like a boiling pot. Hold down-arrow and it settles, stills, and gold order creeps back in. You are running a substance up and down its phase diagram with two keys.",
          checkpoint: "Melting and freezing on command. Beat 2.",
          recovery: ["crystalline_fraction counts coord == 6 exactly — a perfect hexagonal shell; 5 or 7 are defects and grain boundaries, deliberately excluded."] }
      ]
    },
    {
      id: 3, title: "Grow a crystal",
      build: "a mouse heat gun and a remelt key — carve into the lattice and watch it heal.",
      beat: "Cool slowly into a hexagonal crystal, then blast a hole in it with the mouse and watch it refreeze.",
      steps: [
        { title: "A laser to the lattice", adding: "the heat-gun dials, the local-heating kernel, and its mouse wiring.",
          code: `HEAT_RADIUS = 4.0
HEAT_BOOST = 1.4
@ti.kernel
def heat(mx: ti.f32, my: ti.f32):
    for p in pos:
        dx = wrapd(pos[p][0], mx * L)
        dy = wrapd(pos[p][1], my * L)
        if dx * dx + dy * dy < HEAT_RADIUS * HEAT_RADIUS:
            vel[p] *= HEAT_BOOST
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            heat(mx, my)`,
          does: "Every atom within HEAT_RADIUS of the cursor has its velocity scaled up by 40% — a localized injection of kinetic energy, exactly like focusing a laser on one spot of a material. Because temperature IS velocity (chapter 2), speeding these atoms up IS heating them, and locally melts whatever they were part of.",
          why: "The heat gun makes the phase transition INTERACTIVE and local, which is where the intuition really lands: you can melt a hole in a solid crystal and, because the surrounding cold lattice acts as its own thermostat (the global thermostat pulls the average back down), watch the puddle re-freeze and the lattice heal across the scar. That's real physics — it's how zone refining purifies silicon and how a welded joint recrystallizes as it cools.",
          see: "Cool the box to a gold crystal, then drag the mouse through it: a molten blue trail of fast atoms follows your cursor, tearing the lattice open — and behind you, as the global thermostat drains the heat away, the blue re-orders into gold and the crystal knits back together.",
          checkpoint: "An interactive heat gun. No red text.",
          recovery: ["mx, my from get_cursor_pos are in 0..1, so mx*L converts to box coordinates before the distance check.", "HEAT_BOOST multiplies velocity (not adds) — it scales up whatever motion an atom already had, heating hot spots more, which feels natural."] },
        { title: "Grow it from the melt", adding: "the remelt key and the control legend.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.UP:
                temp_target[None] = min(temp_target[None] + TEMP_STEP, TEMP_MAX)
            elif e.key == ti.GUI.DOWN:
                temp_target[None] = max(temp_target[None] - TEMP_STEP, TEMP_MIN)
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000), temperature=1.0)
        gui.text("[up/down] heat/cool  drag: heat gun  [r] remelt   gold: crystal, blue: fluid", (0.02, 0.94), color=0xAAAAAA)`,
          does: "R reseeds a fresh warm liquid to grow a new crystal from, and the HUD spells out the controls.",
          why: "The definitive experiment: melt fully (up-arrow to T ~ 2), then cool slowly (tap down-arrow) and WATCH a crystal nucleate and grow — gold seeds appearing, spreading, meeting at mismatched angles to form the grain boundaries and defects real polycrystals have. Cool too fast and you freeze in a disordered glass instead; cool slowly and you grow big clean grains. That control — cooling RATE decides crystal quality — is the single most important knob in real metallurgy, and here it's your down-arrow. That closes Arc 5, and with it every arc but the capstones: from ant to atom, order that no one designed.",
          see: "Reroll to a fresh melt and cool it patiently: the first gold specks appear, fan out into hexagonal patches, and collide into a mosaic of tilted crystal grains seamed with defects — a whole polycrystalline microstructure, grown atom by atom in front of you, from one force curve and a symmetric integrator.",
          checkpoint: "A grown crystal, with grains and defects. Final beat — project 25 and Arc 5 complete.",
          recovery: ["Same reseed idiom as every project since 01 — apply_seed lays a fresh warm liquid to crystallize from.", "If cooling never crystallizes, cool SLOWER — quenching too fast traps the atoms in a glassy disorder, which is itself a correct and interesting outcome to observe."] }
      ]
    }
  ]
};

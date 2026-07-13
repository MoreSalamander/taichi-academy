// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["18-ocean-currents"] = {
  project: "18-ocean-currents",
  title: "Ocean Currents",
  pitch: "Wind bands push, Coriolis bends, continents block — and your fluid solver becomes a climate machine that carries heat to the poles.",
  tier: "hard",
  language: "Python",
  file: "ocean_currents.py",
  chapters: [
    {
      id: 1, title: "A world to stir",
      build: "fbm continents as a land mask, a sun-driven temperature map, and the climate-map renderer.",
      beat: "A still world: brown continents, hot red equator seas fading to icy blue poles.",
      steps: [
        { title: "The solver becomes a planet", adding: "the docstring and imports.",
          code: `"""Ocean Currents: wind bands + Coriolis + continents turn a fluid box into a climate map."""
import numpy as np
import taichi as ti`,
          does: "This is the stable-fluids solver's third appearance (02: stirred by hand, 11: driven by a vortex function) — this time driven by planetary-scale forces: latitude-banded winds, the Coriolis effect, and continents that simply refuse to flow. The payload it carries isn't ink or dust but TEMPERATURE, and the emergent story is heat transport: the reason Northern Europe isn't Siberia.",
          why: "Arc 3 closes the way it opened — by recombining tools you already own into something none of them could do alone: project 05's fbm draws the continents, projects 02/11's solver moves the water, and three small new kernels (wind, Coriolis, land) turn the whole thing into a climate model you can poke.",
          see: "Runs clean.",
          checkpoint: "python3 ocean_currents.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "The map's state", adding: "grid dials and the core fields.",
          code: `N = 256
vel = None
temp = None
land = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, temp, land, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    vel = ti.Vector.field(2, ti.f32, shape=(N, N))
    temp = ti.field(ti.f32, shape=(N, N))
    land = ti.field(ti.i32, shape=(N, N))
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "The familiar fluid pair (vel for currents, temp as the advected payload — where project 02 had dye) plus one genuinely new kind of field: land, an integer mask that marks cells as immovable continent. The x axis will mean longitude, the y axis latitude.",
          why: "A boolean-ish mask field is the cheapest possible representation of geography, and it's all a fluid solver needs: every kernel that touches water will simply skip (or zero) cells where land == 1. No mesh coastlines, no special boundary objects — just a per-cell yes/no.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["temp is this project's 'dye' — same advection machinery, but the value MEANS something physical now, and a later kernel will push it toward what the sun dictates."] },
        { title: "Continents and a sun", adding: "fbm land, latitude temperature, and the climate renderer.",
          code: `OCEAN_FRACTION = 0.65
def resize_bilinear(a, n):
    """Pure numpy: smoothly resize a small square array up to n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1 - f)[:, None] + a[i1] * f[:, None]
    a = a[:, i0] * (1 - f)[None, :] + a[:, i1] * f[None, :]
    return a
def fbm2d(n, rng_seed=0, octaves=5, roughness=0.55):
    """Pure numpy: fractal 2D noise — octaves of noise, each finer and fainter."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n, n), dtype=np.float32)
    amp, res = 1.0, 4
    for _ in range(octaves):
        layer = rng.uniform(0, 1, size=(res, res)).astype(np.float32)
        out += amp * resize_bilinear(layer, n)
        amp *= roughness
        res *= 2
    out -= out.min()
    out /= out.max()
    return out
def seed_continents(n, rng_seed=0):
    """Pure numpy: fbm noise thresholded at a fixed ocean fraction — the land mask."""
    noise = fbm2d(n, rng_seed)
    sea = np.quantile(noise, OCEAN_FRACTION)
    return (noise > sea).astype(np.int32)
def seed_temperature(n):
    """Pure numpy: warm equator, cold poles — the sun's job, one line of latitude math."""
    jj = np.arange(n)
    lat = np.abs(jj - n / 2) / (n / 2)
    return ((1.0 - lat)[None, :] * np.ones((n, n))).astype(np.float32)
def apply_seed(rng_seed=0):
    land.from_numpy(seed_continents(N, rng_seed))
    temp.from_numpy(seed_temperature(N))
    vel.fill(0.0)
@ti.func
def latitude(j):
    return (j - N / 2.0) / (N / 2.0)
@ti.kernel
def render():
    for i, j in pixels:
        if land[i, j] == 1:
            pixels[i, j] = ti.Vector([0.25, 0.22, 0.18])
        else:
            t = ti.math.clamp(temp[i, j], 0.0, 1.0)
            cold = ti.Vector([0.05, 0.15, 0.45])
            warm = ti.Vector([0.9, 0.35, 0.15])
            c = cold * (1 - t) + warm * t
            spd = vel[i, j].norm()
            c += ti.min(spd * 0.5, 0.35) * ti.Vector([1.0, 1.0, 1.0])
            pixels[i, j] = ti.math.clamp(c, 0.0, 1.0)
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Ocean Currents — taichi-academy", res=N, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "seed_continents is project 13's quantile-anchor trick in 2D: fbm noise thresholded so exactly 65% of the world is ocean, every seed. seed_temperature paints the sun's influence — 1.0 at the equator row, 0.0 at both pole rows, no noise needed. The renderer maps sea temperature onto a cold-blue-to-warm-red ramp, adds a white sheen where currents run fast (nothing yet), and paints land a flat dark earth-brown. latitude(j) — a one-line helper mapping row to [-1, +1] — will quietly become the most-used function in the project.",
          why: "Three physical ideas got their own tiny functions on purpose: geography (seed_continents), insolation (seed_temperature), and the coordinate frame (latitude). Every force this project adds — wind, Coriolis, relaxation — will be expressed in terms of latitude(j), so pinning that abstraction down before any physics keeps the physics readable.",
          see: "A striking still map: dark continents over a smooth thermal gradient — deep red equatorial seas banding out through violet to inky polar blue.",
          checkpoint: "A static climate map. Beat 1.",
          recovery: ["Both seeders are pure numpy, testable without Taichi — the discipline every project has kept since 01.", "The white speed-sheen line is already in render — dormant until water moves."] }
      ]
    },
    {
      id: 2, title: "Wind sets the water moving",
      build: "latitude-banded wind forcing, the transport core with wrap/clamp geometry, land enforcement, and the sun's slow pull.",
      beat: "Currents flow east and west in great bands, splitting and deflecting around every continent.",
      steps: [
        { title: "Trades, westerlies, and a new edge rule", adding: "wind dials, the wrap/clamp sampler, bilerp, advection, and the wind kernel.",
          code: `DT = 1.0
WIND = 0.06
VEL_DECAY = 0.995
PI = 3.14159265
vel_next = None
temp_next = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, land, pixels
    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))
    temp_next = ti.field(ti.f32, shape=(N, N))
@ti.func
def sample(f: ti.template(), i, j):
    ci = ((i % N) + N) % N
    cj = min(max(j, 0), N - 1)
    return f[ci, cj]
@ti.func
def bilerp(f: ti.template(), x, y):
    x0 = int(ti.floor(x))
    y0 = int(ti.floor(y))
    fx = x - x0
    fy = y - y0
    a = sample(f, x0, y0)
    b = sample(f, x0 + 1, y0)
    c = sample(f, x0, y0 + 1)
    d = sample(f, x0 + 1, y0 + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
@ti.kernel
def wind_forcing():
    for i, j in vel:
        if land[i, j] == 0:
            zonal = -ti.cos(latitude(j) * 3.0 * PI)
            vel[i, j][0] += DT * WIND * zonal
@ti.kernel
def advect_all():
    for i, j in vel:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        vel_next[i, j] = bilerp(vel, x, y)
        temp_next[i, j] = bilerp(temp, x, y)
@ti.kernel
def copy_back():
    for i, j in vel:
        vel[i, j] = vel_next[i, j] * VEL_DECAY
        temp[i, j] = temp_next[i, j]
@ti.kernel
def enforce_land():
    for i, j in vel:
        if land[i, j] == 1:
            vel[i, j] = ti.Vector([0.0, 0.0])
def step():
    wind_forcing()
    advect_all()
    copy_back()
    enforce_land()`,
          does: "sample() gets this project's signature geometry: WRAP in x (sail east far enough and you come home — longitude), CLAMP in y (the poles are ends, not portals — latitude). Third boundary policy in three fluid projects: 02 wrapped both, 11 clamped both, this one mixes. wind_forcing pushes only x-velocity, only on water, with -cos(3·pi·lat) — a profile that flips sign three times per hemisphere: easterly trades at the equator, westerlies at mid-latitudes, polar easterlies above. That's Earth's actual wind-belt structure, one cosine.",
          why: "A subtle test-writing lesson surfaced here during this project's development: an early test sampled the wind at 75% latitude to check the westerlies — and failed, because that latitude is exactly a band's ZERO CROSSING, where the profile passes through nothing on its way between belts. When a function oscillates, test its peaks, not wherever's convenient.",
          see: "Runs clean; step() isn't wired into main yet — one more piece (the pressure solve) keeps it honest first.",
          checkpoint: "No red text.",
          recovery: ["enforce_land comes AFTER copy_back — advection can smear velocity onto coastal land cells; zeroing them every tick is what makes coastlines truly solid."] },
        { title: "Keep it incompressible", adding: "the pressure projection from projects 02/11, plus a second land pass.",
          code: `JACOBI = 30
pressure = None
pressure_next = None
divergence = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, temp, temp_next, pressure, pressure_next, divergence, land, pixels
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))
def apply_seed(rng_seed=0):
    land.from_numpy(seed_continents(N, rng_seed))
    temp.from_numpy(seed_temperature(N))
    vel.fill(0.0)
    pressure.fill(0.0)
@ti.kernel
def compute_divergence():
    for i, j in vel:
        divergence[i, j] = (
            sample(vel, i + 1, j)[0] - vel[i, j][0] + sample(vel, i, j + 1)[1] - vel[i, j][1]
        )
@ti.kernel
def pressure_jacobi():
    for i, j in pressure:
        pressure_next[i, j] = (
            sample(pressure, i + 1, j)
            + sample(pressure, i - 1, j)
            + sample(pressure, i, j + 1)
            + sample(pressure, i, j - 1)
            - divergence[i, j]
        ) * 0.25
@ti.kernel
def copy_pressure():
    for i, j in pressure:
        pressure[i, j] = pressure_next[i, j]
@ti.kernel
def subtract_gradient():
    for i, j in vel:
        grad = ti.Vector(
            [pressure[i, j] - sample(pressure, i - 1, j), pressure[i, j] - sample(pressure, i, j - 1)]
        )
        vel[i, j] -= grad
def project():
    compute_divergence()
    for _ in range(JACOBI):
        pressure_jacobi()
        copy_pressure()
    subtract_gradient()
def step():
    wind_forcing()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()`,
          does: "The exact projection stack from projects 02 and 11 (30 Jacobi passes here — a world map tolerates a little more divergence than an ink box). Note step() now enforces land TWICE: once after advection, once after projection, because the pressure solve — blind to geography — happily pushes velocity back onto continent cells while canceling divergence.",
          why: "That second enforce_land is the step's real lesson: every operation that WRITES velocity needs the constraint re-applied after it. Constraints aren't set-and-forget; they're re-asserted after every pass that could violate them — the same reason project 10's solver looped its constraints instead of solving them once.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["If you skipped the second enforce_land, nothing would crash — currents would just slowly leak across coastlines, wrong in a way that takes minutes to notice. The quiet bugs are the ones worth ritual defenses."] },
        { title: "The sun keeps score", adding: "temperature relaxation and the full wiring.",
          code: `TEMP_RELAX = 0.005
@ti.kernel
def relax_temp():
    for i, j in temp:
        target = 1.0 - ti.abs(latitude(j))
        temp[i, j] += TEMP_RELAX * (target - temp[i, j])
def step():
    wind_forcing()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()
    relax_temp()
        step()`,
          does: "Every cell's temperature drifts 0.5% per tick back toward what its latitude 'should' be — the sun endlessly reheating the equator and the poles endlessly radiating away. Advection fights this pull by carrying water to latitudes it doesn't belong to; the balance between the two IS the climate.",
          why: "Without relaxation, advection would eventually stir the ocean to one uniform lukewarm gray — no gradient left to transport. Without advection, temperature would be a boring latitude ramp. Source-versus-transport equilibrium, the same tension as project 03's burn-vs-cool and project 05's rain-vs-evaporation, at planetary scale.",
          see: "The map comes alive: currents shear east and west in the wind bands, pile against continents and deflect along their coasts, and warm equatorial water begins visibly streaming into colder latitudes — the first hints of gyres curling in the basins.",
          checkpoint: "Flowing, heat-carrying currents. Beat 2.",
          recovery: ["relax_temp runs on land cells too — harmless (land temperature is never rendered) and one less branch."] }
      ]
    },
    {
      id: 3, title: "The Coriolis twist",
      build: "the rotation-of-the-Earth force — flows curl into basin-wide gyres.",
      beat: "Straight wind-driven currents bend into closed circulating loops, mirrored across the equator.",
      steps: [
        { title: "The deflecting force", adding: "the three-line Coriolis kernel.",
          code: `CORIOLIS = 0.05
@ti.kernel
def coriolis():
    for i, j in vel:
        if land[i, j] == 0:
            f = CORIOLIS * latitude(j)
            v = vel[i, j]
            vel[i, j] += DT * f * ti.Vector([v[1], -v[0]])`,
          does: "[v.y, -v.x] is the perpendicular of v — the rotate-90-degrees trick yet again — scaled by CORIOLIS times latitude. The latitude factor does all the geophysics: positive in the north (deflect right), negative in the south (deflect left), zero at the equator (no deflection at all), stronger toward the poles. Exactly how the real Coriolis parameter behaves.",
          why: "Coriolis is the most famously misunderstood force in physics (no, it doesn't drain your bathtub), and here it's three lines: a perpendicular push proportional to your speed and your latitude. It never speeds anything up or slows it down — perpendicular forces only BEND paths — which is why it can't create currents, only shape what the wind already made.",
          see: "Runs clean; not in the tick yet.",
          checkpoint: "No red text.",
          recovery: ["The deflection must be computed from a SAVED copy of v — reading vel[i,j] twice mid-update would deflect the already-deflected value."] },
        { title: "Gyres", adding: "coriolis in the tick, right after the wind.",
          code: `def step():
    wind_forcing()
    coriolis()
    advect_all()
    copy_back()
    enforce_land()
    project()
    enforce_land()
    relax_temp()`,
          does: "One line in the tick, placed directly after wind_forcing so the deflection acts on freshly-forced flow before transport and projection tidy up.",
          why: "Now watch the basins: eastward and westward wind bands, bent continuously toward the right (north) or left (south), curl into closed basin-scale loops — GYRES, the defining structures of real ocean circulation, spinning clockwise in the north and counterclockwise in the south exactly like the Atlantic's and Pacific's. Nobody drew a circle; a cosine, a perpendicular push, and some continents in the way did.",
          see: "Basin-wide rotating loops of current, mirrored across the equator, each hugging its continent's coast on one side — and along those coasts, tongues of warm red water streaking poleward: your world's Gulf Streams.",
          checkpoint: "Gyres and boundary currents. Beat 3.",
          recovery: ["If your gyres spin the same way in both hemispheres, the latitude factor is missing or unsigned — it's the sign flip at the equator that mirrors them."] }
      ]
    },
    {
      id: 4, title: "Storms on demand",
      build: "click-to-spawn cyclones whose spin follows the hemisphere, plus reseed and HUD.",
      beat: "Poke the ocean and watch a correctly-handed cyclone swirl heat into a spiral.",
      steps: [
        { title: "A cyclone under the cursor", adding: "storm dials, the spinning splat, and its mouse wiring.",
          code: `STORM_STRENGTH = 1.2
STORM_RADIUS = 14.0
@ti.kernel
def storm(mx: ti.f32, my: ti.f32):
    ci = mx * N
    cj = my * N
    spin = 1.0
    if latitude(cj) < 0:
        spin = -1.0
    for i, j in vel:
        if land[i, j] == 0:
            dx = i - ci
            dy = j - cj
            r2 = dx * dx + dy * dy
            w = ti.exp(-r2 / (STORM_RADIUS * STORM_RADIUS))
            vel[i, j] += spin * STORM_STRENGTH * w * ti.Vector([-dy, dx]) / STORM_RADIUS
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            storm(mx, my)`,
          does: "A gaussian-weighted TANGENTIAL push around the cursor — project 11's vortex forcing, made transient and portable — with one delicious detail: the spin sign flips with the hemisphere. Click north of the equator and the storm turns counterclockwise; click south and it turns clockwise, exactly as real cyclones do and for the same underlying reason your Coriolis kernel encodes.",
          why: "The hemisphere check costs two lines because latitude(j) already existed and the rotate-90 idiom is muscle memory by now — a feature that sounds like a bullet point ('hemisphere-correct cyclone rotation!') falling out of infrastructure built for other reasons. That's what accumulated vocabulary buys.",
          see: "Hold the mouse over open ocean: a white-cored swirl spins up under the cursor and winds the local temperature field into a spiral — release, and the gyres slowly reabsorb it.",
          checkpoint: "Hemisphere-correct storms. No red text.",
          recovery: ["The storm force is tangential ([-dy, dx]) — a pure spin, no inflow. Add a small radial term yourself if you want your cyclones to gather water toward the eye."] },
        { title: "New geography", adding: "the reseed key and the HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("click the sea to spawn a storm", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new continents", (0.02, 0.94), color=0xAAAAAA)`,
          does: "R deals new continents onto the same physics — new basins, new gyres, new boundary currents, all discovered by the same wind and the same spin of the same planet.",
          why: "That's Arc 3 complete: clouds, a planet, a galaxy, a stellar nursery, a solar system, drifting continents, and now a working climate. Every one of them the same recipe — a procedural seed, a handful of forces, and a renderer — recombined. Arc 4 trades physics for pure mathematics: strange attractors and the mandelbulb.",
          see: "Reroll geographies and watch the climate adapt: a world with one huge ocean grows one huge gyre per hemisphere; an archipelago world shatters the circulation into dozens of small eddies threading the islands.",
          checkpoint: "A pokeable climate machine. Final beat — project 18 and Arc 3 complete.",
          recovery: ["Same reseed idiom as every project since 01 — twelfth and final time this arc."] }
      ]
    }
  ]
};

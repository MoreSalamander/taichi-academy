// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["29-earth-simulator"] = {
  project: "29-earth-simulator",
  title: "Earth Simulator",
  pitch: "A whole planet's climate from an energy budget: sunlight in, heat radiated out, winds and oceans moving warmth around, ice that begets ice. Out come climate bands, migrating seasons, ice caps, and green continents.",
  tier: "epic",
  language: "Python",
  file: "earth_simulator.py",
  chapters: [
    {
      id: 1, title: "A world in the sun",
      build: "the energy balance — sunlight absorbed by latitude and albedo, heat radiated back to space — on a lat-lon grid of land and ocean.",
      beat: "A planet map glows hot at the equator and freezes at the poles — the raw climate before anything moves it around.",
      steps: [
        { title: "The floor beneath everything", adding: "the docstring and imports.",
          code: `"""Earth Simulator: an energy-balance climate on a lat-lon grid — sunlight, radiation, winds,
ice-albedo feedback, and a water cycle conspire into climate bands, ice caps, and seasons."""
import numpy as np
import taichi as ti`,
          does: "An 'epic' capstone: a planet's climate, built from the single most important equation in climate science — energy in equals energy out. Sunlight warms the ground; the ground radiates heat back to space; where they balance sets the temperature. Layer on that a few moving parts — winds, oceans, ice, water — and the whole familiar pattern of Earth's climate falls out. We work on a flat longitude-by-latitude grid, the same projection as a world map.",
          why: "This is the ultimate emergence demonstration: nobody programs 'a tropical rainforest belt' or 'polar ice caps' or 'summer.' Those are what an energy budget DOES once you let heat flow and water evaporate. It's the same thesis as every project before it — local rules, global pattern — applied to the most complex system humans try to model.",
          see: "Runs clean.",
          checkpoint: "python3 earth_simulator.py returns silently.",
          recovery: ["Usual venv setup: source .venv/bin/activate, then run from the project folder."] },
        { title: "The planet's dials", adding: "every climate constant and field.",
          code: `W, H = 256, 128          # longitude x latitude
YEAR = 240.0             # steps per orbit
TILT = 23.5 * np.pi / 180
SOLAR = 350.0            # peak insolation scale
A_OLR = 193.0            # outgoing-radiation offset (W/m2); the greenhouse knob (lower = warmer)
B_OLR = 2.2              # outgoing radiation per degC
C_OCEAN = 80.0           # heat capacity — ocean is a slow flywheel
C_LAND = 20.0            # land heats and cools fast
DIFF = 0.22              # heat diffusion per pass (2D explicit limit is 0.25)
DIFF_ITERS = 6           # diffusion passes per step -> strong poleward transport
WIND_AMP = 0.5           # zonal wind speed (cells/step); CFL keeps it < 1
FREEZE = -2.0            # below this, a cell counts as ice
ALB_OCEAN, ALB_LAND, ALB_ICE = 0.08, 0.25, 0.42  # weak ice contrast avoids a snowball runaway
T = None
T2 = None
land = None
moist = None
moist2 = None
veg = None
pixels = None
clock = None`,
          does: "The entire model is a Budyko energy-balance climate, one of the oldest and most elegant in the field. T is temperature (in Celsius); land marks continents; moist and veg are the water cycle and its greenery. The key physics lives in a handful of numbers: A_OLR and B_OLR set how a warm surface radiates heat away (this is the GREENHOUSE knob), the albedos set how much sunlight each surface reflects, and the heat capacities set thermal inertia — why oceans lag the seasons and deserts swing wildly.",
          why: "A real global climate model has millions of lines; this captures the qualitative behaviour in fifteen constants because it keeps only what matters at the largest scale — the flow of energy. The comments flag two subtle, hard-won choices: DIFF stays under 0.25 (an explicit-diffusion stability limit we'll hit again), and the ice albedo is deliberately WEAK, because a strong one tips the whole planet into a frozen 'snowball' — a real instability of these models we have to design around.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["A_OLR is the CO2 dial in disguise: lower it (more greenhouse gas trapping heat) and the whole planet warms; the interactive keys will drive it.", "C_OCEAN >> C_LAND is why coasts have mild climates and continental interiors have brutal winters — water is a thermal flywheel."] },
        { title: "Allocate once", adding: "init_sim.",
          code: `def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global T, T2, land, moist, moist2, veg, pixels, clock
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    T = ti.field(ti.f32, shape=(W, H))
    T2 = ti.field(ti.f32, shape=(W, H))
    land = ti.field(ti.i32, shape=(W, H))
    moist = ti.field(ti.f32, shape=(W, H))
    moist2 = ti.field(ti.f32, shape=(W, H))
    veg = ti.field(ti.f32, shape=(W, H))
    pixels = ti.Vector.field(3, ti.f32, shape=(W, H))
    clock = ti.field(ti.f32, shape=())`,
          does: "The allocate-once pattern again. T2 and moist2 are double buffers for the transport steps; clock is a single scalar counting steps, which drives the seasons in chapter 2.",
          why: "Everything is a W x H grid — one number per surface cell — so a whole planet fits in a few megabytes and updates in parallel. The moisture and vegetation fields are allocated now but stay zero until chapter 3; reserving them here keeps init_sim final.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["clock is a scalar field (shape=()) so kernels and the seasonal calculation can read the same calendar.", "The paired T/T2 buffers are the tell that transport is coming — you diffuse from one into the other, then swap."] },
        { title: "Continents and a starting climate", adding: "the land map and the seed.",
          code: `def make_land(seed=3):
    """Pure numpy: low-frequency noise thresholded into continents, with open polar oceans."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((W // 16, H // 16)).astype(np.float32)
    big = np.kron(base, np.ones((16, 16), np.float32))[:W, :H]
    for _ in range(4):
        big = 0.25 * (np.roll(big, 1, 0) + np.roll(big, -1, 0) + np.roll(big, 1, 1) + np.roll(big, -1, 1))
    m = (big > 0.15).astype(np.int32)
    m[:, :6] = 0
    m[:, -6:] = 0
    return m
@ti.kernel
def _seed_fields():
    for i, j in T:
        lat = lat_of(j)
        T[i, j] = 30.0 - 60.0 * (lat / (0.5 * np.pi)) ** 2
        moist[i, j] = 0.0
        veg[i, j] = 0.0
def apply_seed(seed=3):
    """Lay down continents and a warm-equator/cold-pole starting climate; reset the calendar."""
    land.from_numpy(make_land(seed))
    _seed_fields()
    clock[None] = 0.0`,
          does: "make_land builds continents by thresholding smoothed random noise — coarse blobs blurred into landmasses, with the top and bottom rows forced to ocean (open polar seas). _seed_fields lays down a plausible starting climate: a warm equator falling off toward cold poles, which the physics will then take over and refine.",
          why: "We seed a REASONABLE climate rather than starting from a cold, dead planet, for the same reason you preheat an oven: an energy-balance model can be bistable (it has both a temperate state and a frozen 'snowball' state), and starting warm lands us in the climate we actually want to explore. The land map gives the planet longitudinal structure — without continents, every point at a given latitude would be identical and the map would be dull horizontal stripes.",
          see: "Still assembling — no window yet, but the fields now hold continents and a first guess at their temperatures.",
          checkpoint: "No red text.",
          recovery: ["_seed_fields calls lat_of, defined in the next step — that's fine, Taichi resolves it when the kernel first runs, by which point the whole file has loaded.", "Change the seed and you get an entirely different arrangement of continents to run the climate on."] },
        { title: "Sunlight in, heat out", adding: "latitude, albedo, and the radiation balance.",
          code: `@ti.func
def lat_of(j):
    return (ti.cast(j, ti.f32) / (H - 1) - 0.5) * np.pi
@ti.func
def albedo(i, j):
    base = ALB_LAND if land[i, j] == 1 else ALB_OCEAN
    # ice albedo ramps in smoothly from +2C down to -10C — a soft feedback, not a hard tip
    icefrac = ti.max(0.0, ti.min((2.0 - T[i, j]) / 12.0, 1.0))
    return base + icefrac * (ALB_ICE - base)
@ti.kernel
def radiate_step(decl: ti.f32, a_olr: ti.f32):
    """Absorb sunlight (by latitude, season, and albedo), radiate heat to space, update temperature."""
    for i, j in T:
        s = ti.max(0.0, ti.cos(lat_of(j) - decl))     # noon sun height at this latitude
        absorbed = SOLAR * s * (1.0 - albedo(i, j))
        olr = a_olr + B_OLR * T[i, j]
        cap = C_LAND if land[i, j] == 1 else C_OCEAN
        T[i, j] = T[i, j] + (absorbed - olr) / cap`,
          does: "The beating heart of the model, and it's just book-keeping of energy. Each cell ABSORBS sunlight — the sun's height s falls off with latitude (cos of the angle from the sub-solar point), and albedo reflects some fraction away. It RADIATES heat to space as a simple linear function of its temperature (A_OLR + B_OLR*T — warmer things glow more). The difference, divided by heat capacity, nudges the temperature. albedo also folds in the ice feedback: as a cell cools past freezing, it whitens and reflects more, cooling further.",
          why: "Two profound ideas hide in these few lines. First, the linear radiation law A + B*T is a famous simplification of the true fourth-power blackbody curve — accurate enough near Earth's temperatures, and A is literally the greenhouse: raise the trapping and the planet must warm to shed the same heat. Second, the ice-albedo feedback is a genuine POSITIVE feedback — cold makes ice makes more cold — and it's why we made the contrast weak and the onset gradual. Push it too hard and the feedback runs away, freezing the equator; that snowball state really happened on Earth, 700 million years ago.",
          see: "Assembling — the tick that runs this is next.",
          checkpoint: "No red text. lat_of, albedo, and radiate_step compile.",
          recovery: ["s = max(0, cos(lat - decl)) is zero on the night side of the sub-solar latitude — the polar night, when a pole gets no sun for months.", "decl (the sub-solar latitude) is passed in: it's 0 for now, and chapter 2 will swing it with the seasons."] },
        { title: "The bare climate", adding: "a static tick, the probes, the map render, and the main loop.",
          code: `def step(a_olr=A_OLR):
    radiate_step(0.0, a_olr)
def ice_fraction():
    """Pure numpy: fraction of the surface frozen (below FREEZE)."""
    return float((T.to_numpy() < FREEZE).mean())
def band_temp(j0, j1):
    """Pure numpy: mean temperature over a band of latitudes."""
    return float(T.to_numpy()[:, j0:j1].mean())
@ti.kernel
def render():
    for i, j in pixels:
        t = T[i, j]
        col = ti.Vector([0.9, 0.93, 0.97])                # ice
        if t >= FREEZE:
            if land[i, j] == 1:
                green = ti.Vector([0.25, 0.55, 0.2])
                desert = ti.Vector([0.7, 0.62, 0.4])
                v = veg[i, j]
                col = desert * (1.0 - v) + green * v
            else:
                warm = ti.Vector([0.1, 0.5, 0.75])
                cold = ti.Vector([0.03, 0.1, 0.4])
                f = ti.max(ti.min((t + 5.0) / 35.0, 1.0), 0.0)
                col = cold * (1.0 - f) + warm * f
        pixels[i, j] = col
def main():
    init_sim()
    apply_seed()
    a_olr = A_OLR
    gui = ti.GUI("Earth Simulator — taichi-academy", res=(W, H), background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "=":
                a_olr = max(a_olr - 3.0, 170.0)   # more CO2 -> warmer
            elif e.key == "-":
                a_olr = min(a_olr + 3.0, 215.0)   # less CO2 -> colder
            elif e.key == "r":
                apply_seed(np.random.randint(1_000_000))
                a_olr = A_OLR
        step(a_olr)
        render()
        gui.set_image(pixels)
        day = clock[None] % YEAR
        eq = band_temp(H // 2 - 4, H // 2 + 4)
        gui.text(f"day {day:.0f}/{YEAR:.0f}   equator {eq:.0f}C   ice {ice_fraction() * 100:.0f}%   "
                 f"greenhouse [=/-]  [r] new world", (0.02, 0.98), color=0xFFFFFF)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "step for now is just the radiation balance with a fixed overhead sun (decl=0). render paints the world map: ice white, ocean blue (deep to warm), land from desert-tan to green by vegetation. main runs it, with keys to turn the greenhouse up and down ([=] and [-]) and reroll the continents ([r]).",
          why: "This is the climate with the machinery switched OFF — pure local radiation, nothing sharing heat between places. It's instructive precisely because it's WRONG in a specific way: the poles, which barely see the sun, plummet to impossibly cold temperatures, and the equator bakes, because no winds or oceans carry warmth from where there's a surplus to where there's a deficit. That missing transport is the entire subject of chapter 2 — and you can already feel its absence in the too-steep gradient.",
          see: "A world map: a blazing equatorial band of warm blue ocean and tan land, fading through the mid-latitudes into vast white ice sheets swallowing both poles. Tap [=] to crank up the greenhouse and watch the ice retreat; tap [-] and the caps march toward the equator. It's a planet — just one where the poles are unrealistically brutal, because nothing yet moves heat toward them.",
          checkpoint: "An interactive energy-balance planet with a greenhouse dial. Chapter 1 complete.",
          recovery: ["If the whole map is white, the greenhouse is too weak (A_OLR too high) or the seed didn't take — press [=] a few times.", "The poles being absurdly cold is EXPECTED here: there's no heat transport yet, so they sit at pure radiative equilibrium in the dark."] }
      ]
    },
    {
      id: 2, title: "Winds and seasons",
      build: "poleward heat transport (diffusion + prevailing winds) that flattens the gradient into real climate bands, and the axial-tilt cycle that brings seasons.",
      beat: "Heat floods toward the poles, softening the ice caps into realistic bands — and the warm zone marches north and south with the seasons.",
      steps: [
        { title: "Moving the heat around", adding: "prevailing winds, diffusion, and advection.",
          code: `@ti.func
def zonal_wind(lat):
    # easterly trade winds in the tropics, westerlies in the mid-latitudes
    return -WIND_AMP * ti.cos(3.0 * lat)
@ti.kernel
def diffuse_step():
    """One pass of heat diffusion — the atmosphere and oceans smearing warmth toward the poles."""
    for i, j in T:
        jm = ti.max(j - 1, 0)
        jp = ti.min(j + 1, H - 1)
        im = (i - 1 + W) % W
        ip = (i + 1) % W
        lap = T[im, j] + T[ip, j] + T[i, jm] + T[i, jp] - 4.0 * T[i, j]
        T2[i, j] = T[i, j] + DIFF * lap
    for i, j in T:
        T[i, j] = T2[i, j]
@ti.kernel
def advect_step():
    """Prevailing winds carry heat around each latitude circle (upwind, longitude wraps)."""
    for i, j in T:
        im = (i - 1 + W) % W
        ip = (i + 1) % W
        u = zonal_wind(lat_of(j))
        adv = 0.0
        if u > 0:
            adv = -u * (T[i, j] - T[im, j])
        else:
            adv = -u * (T[ip, j] - T[i, j])
        T2[i, j] = T[i, j] + adv
    for i, j in T:
        T[i, j] = T2[i, j]`,
          does: "Two ways heat travels. diffuse_step smears temperature toward its neighbours — the net effect of countless turbulent eddies in atmosphere and ocean carrying warmth from the hot equator toward the cold poles. advect_step blows it around each latitude circle with prevailing winds: easterly trades in the tropics, westerlies in the mid-latitudes (the zonal_wind pattern). Longitude wraps around, and both use double buffers so every cell reads a consistent snapshot.",
          why: "Diffusion is where a stability limit bites: an explicit Laplacian update is only stable if the coefficient stays under 0.25 in 2D — push past it and the temperature field oscillates and explodes. But one gentle pass barely transports heat, and real oceans move a LOT of it, so we run several passes per step (next step). The advection uses an 'upwind' scheme — always sampling the cell the wind blows FROM — which is the standard trick for stability: sample the downwind side instead and the simulation blows up. These two numerical rules, the diffusion limit and the upwind rule, are the same ones that govern every fluid simulation in the curriculum.",
          see: "Assembling — the tick needs to call these; next step.",
          checkpoint: "No red text. zonal_wind, diffuse_step, and advect_step compile.",
          recovery: ["Diffusion's DIFF=0.22 is just under the 0.25 explicit limit — nudge it higher and you'll watch the map dissolve into noise.", "Upwind advection (sampling the cell the wind comes from) is not a stylistic choice — the downwind version is numerically unstable."] },
        { title: "Transport and the turning year", adding: "the seasonal tick.",
          code: `def step(a_olr=A_OLR):
    decl = TILT * float(np.sin(2 * np.pi * clock[None] / YEAR))
    radiate_step(decl, a_olr)
    for _ in range(DIFF_ITERS):
        diffuse_step()
    advect_step()
    clock[None] += 1.0`,
          does: "The tick grows up. decl — the sub-solar latitude — now swings between the tropics as the planet orbits, driven by the axial TILT and the clock: that IS the seasons. After the radiation balance, we run several diffusion passes (DIFF_ITERS) to move a realistic amount of heat poleward, one advection pass for the winds, and advance the calendar.",
          why: "Two payoffs land at once. Running diffusion six times per step is the cheap, stable way to get strong transport — six small stable passes instead of one big unstable one — and it's what finally warms the poles to sane temperatures and flattens the ice caps into believable bands. And swinging decl with the axial tilt gives seasons for free: the sun's overhead point migrates north in the northern summer and south half a year later, exactly as Earth's does because it's tilted 23.5 degrees. Tilt is the entire reason seasons exist, and here it's one sine wave.",
          see: "The climate transforms. Heat pours poleward: the brutal ice sheets of chapter 1 pull back into tidy caps, and a broad temperate band opens up in the mid-latitudes. Then watch over a simulated year — the warm equatorial bulge slides north, the northern ice shrinks while the southern grows, and half a year later it all reverses. The planet breathes with the seasons.",
          checkpoint: "Realistic climate bands and a working seasonal cycle. Chapter 2 complete.",
          recovery: ["If the poles are still frozen solid, raise DIFF_ITERS — you need enough transport passes to overpower the polar radiative cooling.", "No seasonal motion? Check that decl uses clock[None] and that the clock actually increments at the end of the tick."] }
      ]
    },
    {
      id: 3, title: "The living planet",
      build: "the water cycle — evaporation, wind-borne moisture, rain, and the vegetation it grows — turning a climate into a biosphere.",
      beat: "Oceans breathe moisture into the winds, rain falls downwind, and green spreads across the temperate continents.",
      steps: [
        { title: "Evaporation, rain, and green", adding: "the water cycle.",
          code: `@ti.kernel
def moisture_step():
    """Oceans evaporate, winds carry vapor, it rains where the air is over-saturated, and rain
    plus warmth grows vegetation on land."""
    for i, j in moist:
        u = zonal_wind(lat_of(j))
        im = (i - 1 + W) % W
        ip = (i + 1) % W
        evap = 0.0
        if land[i, j] == 0 and T[i, j] > 0.0:
            evap = 0.04 * T[i, j]
        cap = ti.max(0.0, 2.0 + 0.2 * T[i, j])         # warm air holds more vapor
        adv = 0.0
        if u > 0:
            adv = -u * (moist[i, j] - moist[im, j])
        else:
            adv = -u * (moist[ip, j] - moist[i, j])
        m = moist[i, j] + evap + adv
        rain = 0.0
        if m > cap:
            rain = 0.4 * (m - cap)
            m -= rain
        moist2[i, j] = ti.max(m, 0.0)
        if land[i, j] == 1:
            suit = 0.0
            if 2.0 < T[i, j] < 42.0:
                suit = ti.min(rain * 6.0, 1.0)
            veg[i, j] += 0.03 * (suit - veg[i, j])
    for i, j in moist:
        moist[i, j] = moist2[i, j]`,
          does: "The hydrological cycle in one kernel. Warm oceans EVAPORATE moisture into the air. The same prevailing winds that carry heat now carry that vapor around (upwind advection again). Air can only hold so much — a capacity that grows with temperature — and any excess RAINS out. On land, rain plus comfortable warmth grows VEGETATION, which relaxes toward how suitable each spot is: wet temperate ground turns green, cold or bone-dry ground stays barren.",
          why: "This closes the loop from physics to biology. Notice the chain of dependencies the model reproduces without being told to: oceans supply water, winds set WHERE it goes, temperature decides where it falls (warm air carries vapor poleward before it cools and dumps it), and only the temperate, rained-on land greens. That's why deserts sit where dry air descends and rainforests sit under the wet tropics — geography of life, emerging from the geography of heat and water. The 'warm air holds more vapor' rule (capacity rising with T) is the real Clausius-Clapeyron relation, the same physics behind why a warming climate means heavier downpours.",
          see: "Assembling — the tick needs to run it; last step.",
          checkpoint: "No red text. moisture_step compiles.",
          recovery: ["Evaporation only happens over warm ocean, so an all-land or frozen world grows nothing — vegetation needs a wet, warm supply.", "The temperature window (2 to 42 C) for vegetation is why both the ice and the hottest deserts stay bare."] },
        { title: "The whole world turning", adding: "the complete tick.",
          code: `def step(a_olr=A_OLR):
    decl = TILT * float(np.sin(2 * np.pi * clock[None] / YEAR))
    radiate_step(decl, a_olr)
    for _ in range(DIFF_ITERS):
        diffuse_step()
    advect_step()
    moisture_step()
    clock[None] += 1.0`,
          does: "The finished tick, every subsystem in order: sunlight and radiation set the temperatures, diffusion and winds move that heat around, and the water cycle turns the resulting climate into rain and greenery — then the calendar advances and it all runs again.",
          why: "This ordering IS the planet's causal chain: energy first (it drives everything), then transport (it reshapes the energy), then water (it rides on the temperatures and winds the first two produced). Run it and you get a world that no single line describes: tropical green belts, mid-latitude temperate zones, subtropical deserts where the air is dry, polar ice, and all of it sliding with the seasons and shifting when you turn the greenhouse dial. From an energy budget and a water budget, a living planet — and, tuning that one A_OLR knob, a hands-on model of climate change. This is the largest system in the curriculum, and it closes the second dream project.",
          see: "The finished Earth: blue oceans, white polar caps, and continents mottled green and tan — rainforest where the tropics are wet, desert where the descending air is dry, tundra fading to ice toward the poles. Watch a year turn and the whole biosphere pulses with the seasons; press [=] to load the atmosphere with greenhouse gas and watch the ice caps melt back and the green creep poleward. A planet you can hold in a window.",
          checkpoint: "A complete climate-and-biosphere simulation: bands, seasons, ice caps, deserts, forests, and a greenhouse dial. Project 29 complete — the second Arc 7 capstone.",
          recovery: ["If nothing greens, confirm step calls moisture_step and that some ocean is warm enough to evaporate — a too-cold planet is a dead one.", "Order matters: moisture_step reads the temperatures radiate_step and transport just set, so it must come after them in the tick."] }
      ]
    }
  ]
};

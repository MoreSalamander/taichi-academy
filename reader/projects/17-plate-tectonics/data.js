// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["17-plate-tectonics"] = {
  project: "17-plate-tectonics",
  title: "Plate Tectonics",
  pitch: "Voronoi plates with drift vectors: collisions raise mountains, partings tear rift oceans, and the higher plate wins.",
  tier: "hard",
  language: "Python",
  file: "plate_tectonics.py",
  chapters: [
    {
      id: 1, title: "A cracked world",
      build: "a voronoi plate map, continents and oceans assigned per plate, and project 05's renderer to see it.",
      beat: "A world map already divided into plates — some land, some sea, all still.",
      steps: [
        { title: "Geology as a rule set", adding: "the docstring and imports.",
          code: `"""Plate Tectonics: voronoi plates drift, collide into mountains, tear open rifts."""
import numpy as np
import taichi as ti`,
          does: "Plate tectonics reduced to a cellular rule set: the world is a grid of cells, each OWNED by one of a handful of rigid plates; each plate has one drift direction it pushes forever; and everything interesting — mountains, rifts, quakes — happens only at the boundaries where two owners meet.",
          why: "This is the same shape of idea as project 06's species rules (identity + an interaction law = emergent structure), applied to geology. The deep lesson of the arc returns: a tiny table of WHO and a tiny rule of HOW is enough to grow something that looks unmistakably like an atlas.",
          see: "Runs clean.",
          checkpoint: "python3 plate_tectonics.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "Ownership and velocity", adding: "grid dials and the core fields.",
          code: `N = 256
N_PLATES = 7
plate_id = None
height = None
plate_vel = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global plate_id, height, plate_vel, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    plate_id = ti.field(ti.i32, shape=(N, N))
    height = ti.field(ti.f32, shape=(N, N))
    plate_vel = ti.Vector.field(2, ti.f32, shape=N_PLATES)
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))`,
          does: "Three kinds of state: plate_id (which of the 7 plates owns each cell — an INTEGER field, like project 06's species), height (the terrain, exactly project 05's h), and plate_vel — just SEVEN vectors, one drift direction per plate, because plates are rigid: every cell of a plate moves identically.",
          why: "Seven numbers driving a 65,536-cell world is the leverage to notice here. Real plate motion is measured in centimeters per year, uniform across continent-sized slabs — the physics genuinely is 'a handful of shared velocity vectors', so the data structure mirrors reality unusually directly.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["plate_vel is shape=N_PLATES — per PLATE, not per cell. Cells look their motion up through their owner's id."] },
        { title: "Crack the crust", adding: "toroidal voronoi, the world seeder, and the hillshade renderer from project 05.",
          code: `SEA = 0.48
RELIEF = 40.0
def voronoi_plates(n, seeds):
    """Pure numpy: nearest-seed labeling with toroidal (wraparound) distance."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    best = np.full((n, n), 1e18)
    pid = np.zeros((n, n), dtype=np.int32)
    for k in range(len(seeds)):
        dx = np.abs(ii - seeds[k, 0])
        dy = np.abs(jj - seeds[k, 1])
        dx = np.minimum(dx, n - dx)
        dy = np.minimum(dy, n - dy)
        d = dx * dx + dy * dy
        m = d < best
        best[m] = d[m]
        pid[m] = k
    return pid
def seed_world(rng_seed=0):
    """Pure numpy: plates from voronoi, continents vs oceans, one drift vector per plate."""
    rng = np.random.default_rng(rng_seed)
    seeds = rng.uniform(0, N, size=(N_PLATES, 2)).astype(np.float32)
    pid = voronoi_plates(N, seeds)
    is_continent = rng.random(N_PLATES) < 0.4
    h = np.where(is_continent[pid], 0.58, 0.30).astype(np.float32)
    h += rng.normal(0, 0.02, (N, N)).astype(np.float32)
    v = rng.uniform(-1, 1, size=(N_PLATES, 2)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return pid, h, v
def apply_seed(rng_seed=0):
    pid, h, v = seed_world(rng_seed)
    plate_id.from_numpy(pid)
    height.from_numpy(h)
    plate_vel.from_numpy(v)
@ti.func
def wrap(i):
    return ((i % N) + N) % N
@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t
@ti.kernel
def render():
    for i, j in pixels:
        hh = height[i, j]
        c = ti.Vector([0.05, 0.15, 0.4])
        if hh > SEA:
            land = (hh - SEA) / (1.0 - SEA)
            c = band(ti.Vector([0.55, 0.6, 0.3]), ti.Vector([0.45, 0.4, 0.35]), land, 0.1, 0.5)
            c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.55, 0.8)
        else:
            c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), hh / SEA, 0.3, 1.0)
        dhdx = (height[wrap(i + 1), j] - hh) * RELIEF
        dhdy = (height[i, wrap(j + 1)] - hh) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.4 + 0.6 * normal.dot(light), 0.0, 1.0)
        pixels[i, j] = ti.math.clamp(c * shade, 0.0, 1.0)
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Plate Tectonics — taichi-academy", res=N, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "voronoi_plates assigns each cell to its NEAREST seed point — the classic space-partitioning trick — with one twist: distance wraps around the map's edges (this world is a torus, so plates crossing the right edge continue on the left, like project 01's chemicals did). Each plate is dealt a fate (40% chance continent, else ocean floor) and a random unit drift vector. The renderer is project 05's — bands by elevation, normal-dot-light hillshade — with wrap() replacing the edge-clamp.",
          why: "wrap() is defined HERE, up front, because the render's hillshade gradient already needs it — a lesson learned the honest way: an earlier draft of this project introduced wrap alongside the physics in chapter 2, and the chapter-1 build crashed at runtime because render referenced a function that didn't exist yet. Compile-checks catch syntax; only RUNNING each chapter catches missing names.",
          see: "A political map made physical: seven irregular plates, some sandy-green continents, some deep blue ocean floor, faint noise texture, sharp height cliffs at the plate borders.",
          checkpoint: "A static cracked world. Beat 1.",
          recovery: ["Toroidal distance: dx = min(dx, N - dx) per axis BEFORE squaring — that's the wraparound.", "is_continent indexes per-plate fates through the pid map: is_continent[pid] broadcasts 7 booleans onto 65,536 cells in one stroke."] }
      ]
    },
    {
      id: 2, title: "Boundaries do the work",
      build: "convergence detection at plate boundaries — uplift where plates collide, rifting where they part — plus erosion.",
      beat: "Mountain ranges grow along colliding borders; trenches deepen where plates part.",
      steps: [
        { title: "Collision or divorce?", adding: "physics dials, the height double-buffer, and the boundary-classification kernel.",
          code: `DT = 0.02
UPLIFT = 0.22
RIFT = 0.15
height_next = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global plate_id, height, height_next, plate_vel, pixels
    height_next = ti.field(ti.f32, shape=(N, N))
@ti.kernel
def boundary_forces():
    for i, j in height:
        me = plate_id[i, j]
        delta = 0.0
        for k in ti.static(range(4)):
            di = (1, -1, 0, 0)[k]
            dj = (0, 0, 1, -1)[k]
            ni, nj = wrap(i + di), wrap(j + dj)
            other = plate_id[ni, nj]
            if other != me:
                rel = plate_vel[me] - plate_vel[other]
                conv = rel[0] * di + rel[1] * dj
                if conv > 0:
                    delta += UPLIFT * conv * DT * (1.0 - height[i, j])
                else:
                    delta += RIFT * conv * DT * height[i, j]
        height_next[i, j] = ti.math.clamp(height[i, j] + delta, 0.0, 1.0)`,
          does: "Each cell checks its four neighbors; where a neighbor belongs to a DIFFERENT plate, this cell sits on a boundary. conv projects the two plates' relative velocity onto the direction toward that neighbor: positive means the plates are closing (convergent — uplift), negative means opening (divergent — rift). The (1 - height) and (height) factors make both effects SATURATE — mountains asymptote toward a ceiling instead of growing forever.",
          why: "Those saturation factors fix a bug this project's first draft actually had: without them, boundary uplift compounded unbounded and heights blew past 10 (on a 0-to-1 scale) within a thousand frames — Himalayas to the moon. 'Growth proportional to remaining headroom' is the standard fix, the same logistic-shaped damping that shows up everywhere from population models to neural activations.",
          see: "Runs clean; not in a tick yet.",
          checkpoint: "No red text.",
          recovery: ["conv > 0 means closing: rel dotted with the OUTWARD neighbor direction. Sign errors here swap mountains and trenches — worth double-checking."] },
        { title: "Weather, cheaply", adding: "one-knob erosion.",
          code: `EROSION = 0.10
@ti.kernel
def erode():
    for i, j in height:
        avg = 0.25 * (
            height_next[wrap(i + 1), j]
            + height_next[wrap(i - 1), j]
            + height_next[i, wrap(j + 1)]
            + height_next[i, wrap(j - 1)]
        )
        height[i, j] = height_next[i, j] * (1 - EROSION) + avg * EROSION`,
          does: "Blend every cell 10% toward its neighborhood average, while copying the double-buffer back — diffusion doing double duty as erosion AND the buffer swap.",
          why: "Project 05 spent three chapters on hydraulic erosion because erosion WAS that project's subject. Here it's one supporting line, because this project's subject is the plates. Knowing how much fidelity a supporting actor needs is a design skill — the fancy version would add nothing to this lesson but noise.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["erode reads height_next (boundary_forces' output) and writes height — it IS the copy-back, with smoothing folded in."] },
        { title: "Watch the ranges rise", adding: "the tick and its wiring.",
          code: `def step(frame):
    boundary_forces()
    erode()
    frame = 0
        step(frame)
        frame += 1`,
          does: "Classify boundaries, apply their verdicts, smooth — every frame. (step already takes a frame argument that nothing uses yet; chapter 3's drift schedule will want it.)",
          why: "The map's border cliffs — which chapter 1 left as raw height STEPS between continent and ocean plates — now evolve by the boundary's actual politics: closing borders sprout brightening mountain ridges, opening ones darken into deepening trenches, and borders sliding past each other (transform boundaries) stay quiet. One dot product sorted all three fates.",
          see: "White-capped ranges grow along some borders while others sink into deep-blue trench lines — the map is developing a geologic history.",
          checkpoint: "Growing mountains and trenches. Beat 2.",
          recovery: ["frame = 0 goes in main() before the loop; the two call lines replace the loop's plain render-only body."] }
      ]
    },
    {
      id: 3, title: "The continents move",
      build: "actual drift — each cell asks which plate's material ARRIVES, with subduction on collision and fresh crust in the gaps.",
      beat: "Continents visibly migrate across the map, plowing up ranges and leaving young oceans behind.",
      steps: [
        { title: "Who arrives here?", adding: "drift dials and the arrival-based motion kernel.",
          code: `DRIFT_STEP = 1.0
DRIFT_EVERY = 6
NEW_CRUST = 0.25
plate_id_next = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global plate_id, plate_id_next, height, height_next, plate_vel, pixels
    plate_id_next = ti.field(ti.i32, shape=(N, N))
@ti.kernel
def drift():
    for i, j in plate_id:
        best_h = -1.0
        best_id = -1
        for k in range(N_PLATES):
            si = wrap(i - ti.cast(ti.round(plate_vel[k][0] * DRIFT_STEP), ti.i32))
            sj = wrap(j - ti.cast(ti.round(plate_vel[k][1] * DRIFT_STEP), ti.i32))
            if plate_id[si, sj] == k:
                if height[si, sj] > best_h:
                    best_h = height[si, sj]
                    best_id = k
        if best_id >= 0:
            plate_id_next[i, j] = best_id
            height_next[i, j] = best_h
        else:
            plate_id_next[i, j] = plate_id[i, j]
            height_next[i, j] = NEW_CRUST
@ti.kernel
def copy_drift():
    for i, j in plate_id:
        plate_id[i, j] = plate_id_next[i, j]
        height[i, j] = height_next[i, j]`,
          does: "For each cell, try all seven plates: 'if plate k moved one step, would ITS material land on me?' (check the cell one plate-k-step upstream and confirm plate k actually owns it). ZERO arrivals means the plates parted here — fresh, low ocean crust wells up (real seafloor spreading). MULTIPLE arrivals means a collision — and the HIGHER plate wins, which is real subduction: dense low ocean floor dives beneath buoyant continental crust.",
          why: "This kernel replaced a simpler first attempt that failed in two instructive ways. Naive per-cell gather ('sample upstream along MY plate's velocity') produced garbage stripes at every trailing edge — the upstream cell belongs to a DIFFERENT plate moving differently, so boundary cells flickered between owners. And sub-cell drift speeds rounded to zero, so nothing moved at all. Enumerate-the-arrivals fixes both, and the collision rule falls out as physics instead of a hack.",
          see: "Runs clean; not scheduled yet.",
          checkpoint: "No red text.",
          recovery: ["The upstream check is plate_id[si, sj] == k — plate k can only deliver material it actually owns.", "best_h starts at -1 so even a height-0 arrival beats 'nobody came'."] },
        { title: "Set the clock", adding: "a post-drift smoothing pass and the drift schedule.",
          code: `@ti.kernel
def smooth_after_drift():
    for i, j in height:
        height_next[i, j] = height[i, j]
    for i, j in height:
        avg = 0.25 * (
            height_next[wrap(i + 1), j]
            + height_next[wrap(i - 1), j]
            + height_next[i, wrap(j + 1)]
            + height_next[i, wrap(j - 1)]
        )
        height[i, j] = height_next[i, j] * 0.5 + avg * 0.5
def step(frame):
    boundary_forces()
    erode()
    if frame % DRIFT_EVERY == 0:
        drift()
        copy_drift()
        smooth_after_drift()`,
          does: "Drift fires every sixth frame — continents creep, they don't race — and each drift is chased by one strong smoothing pass (a 50% blend toward the neighborhood, much stronger than the per-frame erosion) to soften the single-cell staircase edges that quantized plate motion leaves behind.",
          why: "The one-cell-at-a-time quantized motion plus per-boundary uplift leaves periodic ridge artifacts — 'growth rings' — every drift period. The post-drift smoothing pass knocks those down while leaving the big structures alone. Cleaning up after a discrete operation with one targeted smoothing step is a workhorse trick well beyond geology.",
          see: "The continents MOVE. Watch one plow into a neighbor and pile a bright mountain range at the collision front while a widening strip of young dark-blue ocean opens behind it — Atlantic-style seafloor spreading, live.",
          checkpoint: "Drifting continents. Beat 3.",
          recovery: ["Two separate loops inside smooth_after_drift — the first snapshots height into height_next so the second reads stable values (the double-buffer discipline, in miniature)."] }
      ]
    },
    {
      id: 4, title: "Quakes and eruptions",
      build: "an activity layer that flashes at violent boundaries and decays, plus reseed and HUD.",
      beat: "Orange flashes crackle along the most violent boundaries as the world grinds on.",
      steps: [
        { title: "Stress that snaps", adding: "quake dials, the activity field, and flash-triggering in boundary_forces.",
          code: `QUAKE_CONV = 0.8
QUAKE_PROB = 0.002
ACTIVITY_DECAY = 0.92
activity = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global plate_id, plate_id_next, height, height_next, plate_vel, activity, pixels
    activity = ti.field(ti.f32, shape=(N, N))
def apply_seed(rng_seed=0):
    pid, h, v = seed_world(rng_seed)
    plate_id.from_numpy(pid)
    height.from_numpy(h)
    plate_vel.from_numpy(v)
    activity.fill(0.0)
@ti.kernel
def boundary_forces():
    for i, j in height:
        me = plate_id[i, j]
        delta = 0.0
        for k in ti.static(range(4)):
            di = (1, -1, 0, 0)[k]
            dj = (0, 0, 1, -1)[k]
            ni, nj = wrap(i + di), wrap(j + dj)
            other = plate_id[ni, nj]
            if other != me:
                rel = plate_vel[me] - plate_vel[other]
                conv = rel[0] * di + rel[1] * dj
                if conv > 0:
                    delta += UPLIFT * conv * DT * (1.0 - height[i, j])
                else:
                    delta += RIFT * conv * DT * height[i, j]
                if ti.abs(conv) > QUAKE_CONV and ti.random() < QUAKE_PROB:
                    activity[i, j] = 1.0
        height_next[i, j] = ti.math.clamp(height[i, j] + delta, 0.0, 1.0)`,
          does: "activity is a per-cell 'something violent just happened here' scalar. Boundary cells where the plates' relative speed is high (|conv| past a threshold — convergent OR divergent, both are violent) roll a small die each tick; winners flash to 1.0. It's the same threshold-plus-probability pattern as project 15's star ignition.",
          why: "Real earthquakes are exactly this shape of process: stress accumulates continuously at locked boundaries and releases in sudden discrete snaps at unpredictable moments. A deterministic 'flash every N frames' would look mechanical; threshold-gated randomness reads as geology.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["ti.abs(conv) — divergent boundaries quake too; only near-zero (transform-ish, slow) boundaries stay quiet."] },
        { title: "Flash and fade", adding: "activity decay, the lava tint in render, and the full tick.",
          code: `@ti.kernel
def decay_activity():
    for i, j in activity:
        activity[i, j] *= ACTIVITY_DECAY
@ti.kernel
def render():
    for i, j in pixels:
        hh = height[i, j]
        c = ti.Vector([0.05, 0.15, 0.4])
        if hh > SEA:
            land = (hh - SEA) / (1.0 - SEA)
            c = band(ti.Vector([0.55, 0.6, 0.3]), ti.Vector([0.45, 0.4, 0.35]), land, 0.1, 0.5)
            c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.55, 0.8)
        else:
            c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), hh / SEA, 0.3, 1.0)
        dhdx = (height[wrap(i + 1), j] - hh) * RELIEF
        dhdy = (height[i, wrap(j + 1)] - hh) * RELIEF
        normal = ti.Vector([-dhdx, -dhdy, 1.0]).normalized()
        light = ti.Vector([-0.5, -0.5, 0.8]).normalized()
        shade = ti.math.clamp(0.4 + 0.6 * normal.dot(light), 0.0, 1.0)
        c = c * shade
        c += activity[i, j] * ti.Vector([1.0, 0.35, 0.05])
        pixels[i, j] = ti.math.clamp(c, 0.0, 1.0)
def step(frame):
    boundary_forces()
    erode()
    if frame % DRIFT_EVERY == 0:
        drift()
        copy_drift()
        smooth_after_drift()
    decay_activity()`,
          does: "Each flash decays by 8% a frame — bright for an instant, gone in a second — and renders as an ADDITIVE lava-orange glow on top of the lit terrain, the glow-vs-cover compositing choice from projects 03 and 05 made once more.",
          why: "The activity layer is pure presentation state — it never feeds back into height or drift. Keeping 'what happened' (physics) separate from 'what flashes' (drama) means you can retune the spectacle freely without touching, or risking, the geology.",
          see: "Orange sparks crackle along the fastest-grinding boundaries — clustered, irregular, dying out and re-igniting, like a seismic activity map played at a million years a minute.",
          checkpoint: "Quakes flash along violent boundaries. No red text.",
          recovery: ["The activity tint adds AFTER shade multiplies — eruptions glow with their own light; they're not lit by the sun."] },
        { title: "A new geology", adding: "the reseed key and the HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("mountains rise where plates meet", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[r] new world", (0.02, 0.94), color=0xAAAAAA)`,
          does: "R deals a new voronoi crack pattern, new plate fates, new drift vectors — an entirely different tectonic future from the same rules.",
          why: "Leave one world running for ten minutes and read its history off the map like a geologist: mountain belts mark ancient collisions, deep trenches mark subduction fronts, and stripes of young flat seafloor trail behind every migrating continent. Nothing wrote that history — seven drift vectors and two boundary rules did.",
          see: "Every reroll is a different tectonic saga — sometimes one supercontinent grinding against a world-ocean, sometimes an archipelago of small plates flashing with quakes at every seam.",
          checkpoint: "A living tectonic world. Final beat — project 17 complete.",
          recovery: ["Same reseed idiom as every project since 01 — and apply_seed clears the activity layer too, so no ghost quakes survive a reroll."] }
      ]
    }
  ]
};

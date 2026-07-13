// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["11-tornado"] = {
  project: "11-tornado",
  title: "Tornado",
  pitch: "Your project 02 fluid solver, given a forcing function instead of a mouse — plus debris that rides the wind it makes.",
  tier: "medium",
  language: "Python",
  file: "tornado.py",
  chapters: [
    {
      id: 1, title: "State first",
      build: "the fluid grid, a debris ring, and a first static render.",
      beat: "A ring of motionless white dots, waiting for a wind that doesn't exist yet.",
      steps: [
        { title: "An old solver, a new job", adding: "the docstring and imports.",
          code: `"""Tornado: a self-sustaining vortex in a stable-fluids grid, with debris riding the wind."""
import numpy as np
import taichi as ti`,
          does: "This project doesn't invent a new fluid solver — it reuses project 02's almost line for line (advection, pressure projection, vorticity confinement, all coming back unchanged). What's new is what DRIVES it: instead of a mouse dragging force into the grid, a permanent forcing function keeps a vortex spinning on its own, forever, with no hand required.",
          why: "This is the payoff of building the fluid solver as a general tool back in Arc 1: 'stir a box of ink' and 'sustain a tornado' turn out to be nearly the same simulation, differing only in WHERE the force comes from. Recognizing that reuse is worth more than the code itself.",
          see: "Runs clean.",
          checkpoint: "python3 tornado.py returns silently.",
          recovery: ["Standard venv setup, same as every project."] },
        { title: "A grid, and something to throw around", adding: "sizing dials and every field this project needs.",
          code: `N = 512
CX, CY = N * 0.5, N * 0.5
CORE_R = 60.0
N_DEBRIS = 3000
vel = None
vel_next = None
dye = None
dye_next = None
pressure = None
pressure_next = None
divergence = None
dpos = None
dvel = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence
    global dpos, dvel
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    vel = ti.Vector.field(2, ti.f32, shape=(N, N))
    vel_next = ti.Vector.field(2, ti.f32, shape=(N, N))
    dye = ti.Vector.field(3, ti.f32, shape=(N, N))
    dye_next = ti.Vector.field(3, ti.f32, shape=(N, N))
    pressure = ti.field(ti.f32, shape=(N, N))
    pressure_next = ti.field(ti.f32, shape=(N, N))
    divergence = ti.field(ti.f32, shape=(N, N))
    dpos = ti.Vector.field(2, ti.f32, shape=N_DEBRIS)
    dvel = ti.Vector.field(2, ti.f32, shape=N_DEBRIS)`,
          does: "The grid fields (vel/dye/pressure/divergence, plus their _next twins) are exactly project 02's set. dpos/dvel are new: NOT grid cells — a separate pool of N_DEBRIS free-floating particles, positioned in continuous (not per-cell) coordinates, that will ride the fluid rather than compose it.",
          why: "CX, CY, and CORE_R describe the vortex's home position and size before a single kernel uses them for physics — needed now because the debris ring (next step) is seeded relative to them. Two totally different kinds of 'things' (a grid of cells, a pool of particles) coexisting and interacting in one simulation is this project's whole shape.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["dpos/dvel are Vector.field(2, ...) — CONTINUOUS 2D positions/velocities, not grid indices; nothing here is per-cell."] },
        { title: "See the ring", adding: "the debris seeder, seed application, and a first static render.",
          code: `def seed_debris(rng_seed=0):
    """Pure numpy: N_DEBRIS points scattered in a ring around the vortex core."""
    rng = np.random.default_rng(rng_seed)
    ang = rng.uniform(0.0, 2 * np.pi, N_DEBRIS)
    rad = rng.uniform(CORE_R * 1.2, N * 0.45, N_DEBRIS)
    x = CX + rad * np.cos(ang)
    y = CY + rad * np.sin(ang)
    return np.stack([x, y], axis=1).astype(np.float32)
def apply_seed(rng_seed=0):
    dye.fill(0.0)
    vel.fill(0.0)
    pressure.fill(0.0)
    dpos.from_numpy(seed_debris(rng_seed))
    dvel.fill(0.0)
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Tornado — taichi-academy", res=N, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        gui.circles(dpos.to_numpy() / N, radius=1.5, color=0xFFFFFF)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "Random angle, random radius between 1.2x and 0.45xN out from the core — a donut of debris, close enough to feel the vortex, far enough not to start inside it. gui.circles wants normalized [0,1] coordinates, so dpos.to_numpy() / N does that conversion at render time.",
          why: "Notice there's no fluid visualization at all yet — no pixels field, no gui.set_image. That's deliberate: this beat is purely about the SECOND kind of state (debris) existing and being visible, before the FIRST kind (the fluid grid) gets a renderer of its own next chapter.",
          see: "A ring of small white dots, floating in silence over a black canvas.",
          checkpoint: "A static debris ring. Beat 1.",
          recovery: ["gui.circles expects coordinates in [0, 1] — forgetting the / N divide would try to plot points at pixel coordinates 60-230 on a canvas that only understands 0-1, and nothing would appear on screen."] }
      ]
    },
    {
      id: 2, title: "A vortex that sustains itself",
      build: "the reused fluid-transport core, a permanent forcing function, and the pressure solve — a self-sustaining vortex.",
      beat: "A glowing, spiraling column of dust — no mouse required.",
      steps: [
        { title: "The transport core, unchanged", adding: "a timestep dial and the sampling/advection machinery, straight from project 02.",
          code: `DT = 1.0
@ti.func
def sample(f: ti.template(), i, j):
    ci = min(max(i, 0), N - 1)
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
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy
@ti.kernel
def advect(f: ti.template(), f_next: ti.template()):
    for i, j in f:
        x = i - DT * vel[i, j][0]
        y = j - DT * vel[i, j][1]
        f_next[i, j] = bilerp(f, x, y)
@ti.kernel
def copy_back():
    for i, j in dye:
        dye[i, j] = dye_next[i, j]
        vel[i, j] = vel_next[i, j]`,
          does: "sample/bilerp/advect/copy_back are semi-Lagrangian advection, word for word from project 02: trace each cell's content backward along the velocity field, sample where it came from, adopt the result. One change worth noticing: sample here CLAMPS out-of-range indices to the grid's edge (min/max) instead of project 02's WRAPAROUND (modulo). A tornado in an open field has walls, not an infinite looping plane.",
          why: "That clamp-vs-wrap difference is small in code but changes the whole feel: wraparound (project 02's ink box) makes a borderless, infinite canvas; clamping makes a bounded room with edges the fluid can pile up against — the right choice when 'debris flying off one side and reappearing on the other' would look absurd.",
          see: "Runs clean; nothing calls these yet.",
          checkpoint: "No red text.",
          recovery: ["This step is nearly a transcription exercise if you did project 02 — the only genuinely new line is sample's min/max clamp replacing the old modulo wrap."] },
        { title: "A wind that never stops", adding: "vortex dials and the forcing kernel that replaces the mouse.",
          code: `TANGENT_STRENGTH = 1.0
INFLOW_STRENGTH = 0.2
@ti.kernel
def vortex_forcing():
    for i, j in vel:
        rx, ry = float(i) - CX, float(j) - CY
        r = ti.sqrt(rx * rx + ry * ry) + 1e-3
        falloff = r / (r * r + CORE_R * CORE_R) * CORE_R
        tangent = ti.Vector([-ry, rx]) / r
        radial_in = ti.Vector([-rx, -ry]) / r
        vel[i, j] += DT * falloff * (TANGENT_STRENGTH * tangent + INFLOW_STRENGTH * radial_in)
@ti.kernel
def seed_dye():
    for i, j in dye:
        rx, ry = float(i) - CX, float(j) - CY
        r2 = rx * rx + ry * ry
        if r2 < (CORE_R * 1.5) ** 2:
            w = ti.exp(-r2 / (CORE_R * CORE_R))
            dye[i, j] = ti.min(dye[i, j] + 0.02 * w * ti.Vector([0.8, 0.75, 0.6]), 1.0)`,
          does: "tangent ([-ry, rx], a 90-degree rotation of the radius vector — the exact same rotation trick project 09's pressure force used) points every cell to swirl AROUND the core rather than toward or away from it. radial_in pulls straight at the center. falloff peaks at r = CORE_R and fades on both sides, so the push is strongest at the vortex's characteristic radius, not at its exact center (which would be a singularity) or far away (where a tornado has no business reaching). seed_dye replenishes a capped (ti.min(..., 1.0)) trickle of dust near the core every tick — a permanent SOURCE, not a one-time splat.",
          why: "vortex_forcing runs EVERY tick, forever, unlike a mouse-driven splat that only fires while you're clicking — that persistence is the entire difference between 'a box you stir' and 'a tornado that sustains itself.' And that ti.min cap on seed_dye matters more than it looks: without it, a source added every tick with only gentle decay to remove it would accumulate past 1.0 and wash the whole render out to white — a real bug the reference implementation hit during tuning.",
          see: "Runs clean; not wired into a tick yet.",
          checkpoint: "No red text.",
          recovery: ["Both tangent and radial_in divide by r, guarded by the +1e-3 in its definition — undefined direction at the exact center is a divide-by-zero waiting to happen without that guard."] },
        { title: "Make it real fluid: the pressure solve", adding: "the incompressibility solver, straight from project 02.",
          code: `JACOBI_ITERS = 40
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
    for _ in range(JACOBI_ITERS):
        pressure_jacobi()
        copy_pressure()
    subtract_gradient()`,
          does: "Unmodified from project 02: measure how much each cell's velocity would create or destroy fluid (divergence), solve for a pressure field that cancels it (40 Jacobi relaxation passes), then subtract that pressure's gradient back out of velocity. Real incompressible flow, not just 'stuff that swirls.'",
          why: "vortex_forcing alone would happily create fluid out of nowhere at the core and destroy it at the edges — a physically nonsensical vacuum-then-explosion. project() is what keeps the vortex an actual FLOW: air pushed toward the center has to go somewhere, and the pressure solve is what works that out, every single tick.",
          see: "Runs clean; still not wired into a tick.",
          checkpoint: "No red text.",
          recovery: ["If any of this looks unfamiliar, it's worth a quick trip back to project 02's own version — this IS that code, unchanged."] },
        { title: "Let it spin", adding: "decay dials, the tick conductor, a renderer, and the wiring that makes it all visible.",
          code: `DYE_DECAY = 0.985
VEL_DECAY = 0.97
@ti.kernel
def decay():
    for i, j in dye:
        dye[i, j] *= DYE_DECAY
        vel[i, j] *= VEL_DECAY
def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    project()
    decay()
@ti.kernel
def render(pixels: ti.template()):
    for i, j in pixels:
        pixels[i, j] = ti.math.clamp(dye[i, j], 0.0, 1.0)
    pixels = ti.Vector.field(3, ti.f32, shape=(N, N))
        step()
        render(pixels)
        gui.set_image(pixels)`,
          does: "step() is the whole recipe in order: force the wind, replenish dust, carry both along the flow, make it incompressible, let everything fade a little. render just clamps dye into a displayable image — pixels finally exists, and gui.set_image finally has something to show.",
          why: "This is the moment the abstract math becomes a picture. Everything from chapter 1 (the debris ring) and this chapter (the vortex forcing, the pressure solve) has been building toward this single beat: a real, continuously-running fluid simulation, sustaining its own rotation with no outside input.",
          see: "A warm, glowing spiral blooms at the center of the screen and keeps spinning, unprompted — dust drawn in, swirled, and gently fading at the edges, forever. The debris ring from chapter 1 sits motionless on top of it, for now.",
          checkpoint: "A self-sustaining spinning vortex. Beat 2.",
          recovery: ["pixels is declared in main(), between apply_seed() and the gui = ti.GUI(...) line — it needs to exist before the render loop can write into it.", "Debris still isn't moving — advect_debris doesn't exist yet. That's next chapter, on purpose."] }
      ]
    },
    {
      id: 3, title: "Debris rides the wind",
      build: "a fluid-to-particle coupling — debris that samples the vortex's velocity and gets thrown around by it.",
      beat: "The ring of dots stops sitting still and starts orbiting the storm.",
      steps: [
        { title: "Sample the wind where you stand", adding: "coupling dials and the debris-advection kernel.",
          code: `DRAG = 0.12
DEBRIS_PULL = 0.015
DEBRIS_HOME_R = N * 0.4
@ti.kernel
def advect_debris():
    for p in dpos:
        fluid_v = bilerp(vel, dpos[p][0], dpos[p][1])
        dvel[p] += (fluid_v - dvel[p]) * DRAG
        offset = dpos[p] - ti.Vector([CX, CY])
        r = offset.norm() + 1e-3
        if r > DEBRIS_HOME_R:
            dvel[p] -= DEBRIS_PULL * (r - DEBRIS_HOME_R) * (offset / r)
        dpos[p] += DT * dvel[p]
        for a in ti.static(range(2)):
            if dpos[p][a] < 0:
                dpos[p][a] = 0
                dvel[p][a] *= -0.5
            if dpos[p][a] >= N:
                dpos[p][a] = N - 1
                dvel[p][a] *= -0.5`,
          does: "bilerp(vel, ...) is the exact same interpolation function the fluid grid uses on ITSELF — reused here to ask 'what's the wind doing at this exact (non-grid-aligned) point?' DRAG blends debris velocity toward that sampled wind gradually, not instantly — real debris has its own momentum, it doesn't teleport onto the flow. The DEBRIS_PULL term is a second, independent force: a soft leash keeping debris from drifting past DEBRIS_HOME_R and out to the walls, where the fluid's large-scale circulation would otherwise strand it for good.",
          why: "This is ONE-WAY coupling — debris reads the fluid's velocity, but never writes anything back into it — and it's a deliberate simplification, the same 'water COVERS land, doesn't move it' spirit as project 07's blend choices. Real two-way coupling (debris disturbing the air around it) is a much harder, genuinely different technique, saved for a future project. The leash force exists because this project's reference implementation tried WITHOUT one first: debris drifted steadily outward over hundreds of frames and piled up uselessly against the walls, never orbiting anything — a small explicit 'stay near home' force fixed it completely.",
          see: "Runs clean; not called from step() yet.",
          checkpoint: "No red text.",
          recovery: ["DRAG is a BLEND rate (0 to 1), not a target — dvel[p] += (fluid_v - dvel[p]) * DRAG nudges debris velocity a FRACTION of the way toward the fluid's each tick, never fully matching it in one step."] },
        { title: "Set the ring free", adding: "advect_debris's slot in step.",
          code: `def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    project()
    decay()
    advect_debris()`,
          does: "One line, at the very end of the tick — debris responds to whatever the fluid did THIS tick, always one step behind, exactly the way real inertia works.",
          why: "The order matters: advect_debris runs AFTER project() and decay(), so debris always samples the FINAL, already-incompressible, already-decayed velocity field for this tick — not some intermediate, not-yet-physical state.",
          see: "The debris ring stops being a static decoration and starts genuinely living inside the storm: dots drift, spiral, and orbit the glowing core, some catching more wind than others, no two paths quite alike.",
          checkpoint: "Debris orbits the vortex. Beat 3.",
          recovery: ["If debris looks glued to the wall instead of orbiting, double check DEBRIS_HOME_R and DEBRIS_PULL made it into advect_debris — that's the leash keeping them in the storm's neighborhood."] }
      ]
    },
    {
      id: 4, title: "Sharpen the spin, stir it up",
      build: "vorticity confinement for a tighter core, mouse-driven gusts, and a HUD.",
      beat: "A crisper, more violent funnel you can personally disturb.",
      steps: [
        { title: "Fight numerical smoothing", adding: "a curl field and confinement dial, plus the curl/vorticity kernels — verbatim from project 02.",
          code: `CURL_STRENGTH = 0.3
curl = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global vel, vel_next, dye, dye_next, pressure, pressure_next, divergence, curl
    global dpos, dvel
    curl = ti.field(ti.f32, shape=(N, N))
@ti.kernel
def compute_curl():
    for i, j in vel:
        curl[i, j] = (
            sample(vel, i + 1, j)[1] - sample(vel, i - 1, j)[1] - sample(vel, i, j + 1)[0] + sample(vel, i, j - 1)[0]
        ) * 0.5
@ti.kernel
def apply_vorticity(strength: ti.f32):
    for i, j in vel:
        grad = (
            ti.Vector(
                [
                    ti.abs(sample(curl, i + 1, j)) - ti.abs(sample(curl, i - 1, j)),
                    ti.abs(sample(curl, i, j + 1)) - ti.abs(sample(curl, i, j - 1)),
                ]
            )
            * 0.5
        )
        n = grad / (grad.norm() + 1e-5)
        vel[i, j] += DT * strength * curl[i, j] * ti.Vector([n[1], -n[0]])`,
          does: "curl measures local spin at every cell; apply_vorticity finds where spin is INCREASING outward and gives it a small extra push in that direction — actively fighting the way numerical advection quietly smooths sharp rotation away over time.",
          why: "You built vortex_forcing yourself this project, but compute_curl/apply_vorticity are unchanged from project 02 — proof that 'sharpen the swirl' is a completely general fluid-solver feature, independent of WHERE the swirl came from in the first place (a mouse drag there, a permanent forcing function here).",
          see: "Runs clean; CURL_STRENGTH isn't in the tick yet.",
          checkpoint: "No red text.",
          recovery: ["curl joins init_sim's global statement AND gets its own allocation line — same two-part ritual every field has followed since project 01."] },
        { title: "Wire it in", adding: "the curl/vorticity calls, inserted mid-tick.",
          code: `def step():
    vortex_forcing()
    seed_dye()
    advect(dye, dye_next)
    advect(vel, vel_next)
    copy_back()
    compute_curl()
    apply_vorticity(CURL_STRENGTH)
    project()
    decay()
    advect_debris()`,
          does: "Unlike advect_debris (appended at the end last chapter), this pair goes IN THE MIDDLE — right after copy_back adopts this tick's velocity, right before project() re-enforces incompressibility.",
          why: "Order is physics here, not style: curl needs to read the CURRENT tick's velocity (available right after copy_back), and apply_vorticity's extra push needs to happen BEFORE the pressure solve, so project() gets the final say on removing any divergence that confinement might have introduced.",
          see: "The funnel visibly tightens — a narrower, more defined core with crisper spiral arms, closer to a real tornado's silhouette than the softer swirl from chapter 2.",
          checkpoint: "A sharper vortex. No red text.",
          recovery: ["Insert compute_curl/apply_vorticity between copy_back() and project() — not at the end, where advect_debris lives."] },
        { title: "A hand in the storm", adding: "stir dials and a gust kernel.",
          code: `STIR_RADIUS = 20.0
STIR_FORCE = 200.0
@ti.kernel
def stir(mx: ti.f32, my: ti.f32, fx: ti.f32, fy: ti.f32):
    for i, j in vel:
        dx, dy = float(i) - mx * N, float(j) - my * N
        w = ti.exp(-(dx * dx + dy * dy) / (STIR_RADIUS * STIR_RADIUS))
        vel[i, j] += w * ti.Vector([fx, fy])
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            fx, fy = np.random.uniform(-1, 1, 2) * STIR_FORCE
            stir(mx, my, float(fx), float(fy))`,
          does: "A small Gaussian bump of RANDOM directional force wherever you hold the mouse — same 'torch' shape as projects 02/03/08's stirring, but with the push direction re-rolled every frame instead of following the drag delta, so it reads as gusty turbulence, not a deliberate shove.",
          why: "Random-direction stirring, rather than drag-direction stirring, is a small but deliberate choice: this project's vortex already has a strong, dominant rotation of its own — a directional user push would just get absorbed into that spin unnoticed, while chaotic gusts visibly perturb the funnel's shape in a way you can actually see.",
          see: "Runs clean; not wired to a mouse event yet.",
          checkpoint: "No red text.",
          recovery: ["stir adds force with += — it's one more push added on top of everything vortex_forcing already applied, not a replacement."] },
        { title: "Reroll and read out", adding: "the reset key and the HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("drag to stir  [r] new debris", (0.02, 0.98), color=0xAAAAAA)`,
          does: "R calls apply_seed again — clearing the fluid grid back to stillness AND scattering a fresh random debris ring, in one call, since apply_seed already does both.",
          why: "That's the project: your Arc-1 fluid solver, unmodified in its two hardest parts (advection, pressure projection), carrying a permanently-forced vortex AND a second, entirely different kind of physics (drag-coupled debris particles) riding on top of it — two techniques from opposite ends of this curriculum, in one file.",
          see: "Hold the mouse down anywhere and watch gusts visibly distort the funnel's shape; tap R for an entirely fresh storm. The vortex itself never stops, no matter how hard you disturb it.",
          checkpoint: "A fully interactive, self-sustaining tornado. Final beat — project 11 complete.",
          recovery: ["apply_seed(rng_seed=np.random.randint(1_000_000)) — the reseed pattern every project since 01 has used for its 'r' key."] }
      ]
    }
  ]
};

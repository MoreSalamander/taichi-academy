// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["14-galaxy-creator"] = {
  project: "14-galaxy-creator",
  title: "Galaxy Creator",
  pitch: "A logarithmic spiral, differential rotation, and 60,000 additive stars — the recipe behind every galaxy photo you've ever loved.",
  tier: "medium",
  language: "Python",
  file: "galaxy.py",
  chapters: [
    {
      id: 1, title: "Sixty thousand stars",
      build: "star state in polar coordinates, a log-spiral seeder, and an additive splat — a frozen spiral.",
      beat: "A motionless two-armed spiral of blue-white stars around a golden core.",
      steps: [
        { title: "A breather that glitters", adding: "the docstring and imports.",
          code: `"""Galaxy Creator: star particles on spiral arms, differential rotation, additive light."""
import numpy as np
import taichi as ti`,
          does: "After two hard ray-marching projects, this one is deliberately light on new machinery: particles (Arc 2's bread and butter), an additive splat (project 07's), and a fade (project 07's again). The new content is ASTRONOMY — why galaxies look the way they do — expressed in about a dozen lines of the right math.",
          why: "Two ideas carry the whole project: a LOGARITHMIC SPIRAL (arm angle proportional to log of radius — the shape spiral arms actually trace) and DIFFERENTIAL ROTATION (inner stars orbit faster than outer ones — the reason those arms wind and smear the way photographs show). Everything else is presentation.",
          see: "Runs clean.",
          checkpoint: "python3 galaxy.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "Polar state", adding: "star-count dials and fields — radius and angle per star, not x and y.",
          code: `RES = 512
N_STARS = 60000
SPIRAL = 0
radius_f = None
angle_f = None
color_f = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global radius_f, angle_f, color_f, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    radius_f = ti.field(ti.f32, shape=N_STARS)
    angle_f = ti.field(ti.f32, shape=N_STARS)
    color_f = ti.Vector.field(3, ti.f32, shape=N_STARS)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "Every particle project so far stored positions as (x, y). This one stores (radius, angle) — POLAR coordinates — because every question this project asks is polar: how far from the core? how fast should this orbit? Rotation becomes a single addition to angle instead of a 2x2 matrix multiply.",
          why: "Choosing coordinates that match the problem's symmetry is a quiet superpower. A galaxy is round; its physics is 'stuff orbiting a center'; storing state as radius-and-angle makes the upcoming rotation kernel THREE lines. The same physics in x/y would need sin/cos gymnastics every tick.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["SPIRAL = 0 looks lonely — it grows into a trio of galaxy types in chapter 3."] },
        { title: "The spiral recipe", adding: "star colors, disk radii, the log-spiral seeder, and the upload bridge.",
          code: `def star_colors(r, rng, core_scale=0.12, core_col=(0.7, 0.6, 0.4), arm_col=(0.5, 0.6, 1.0)):
    """Pure numpy: blend core color to arm color by radius, dimmed per-star at random."""
    n = len(r)
    core = np.exp(-r / core_scale)
    col = np.zeros((n, 3), dtype=np.float32)
    for ch in range(3):
        col[:, ch] = core_col[ch] * core + arm_col[ch] * (1 - core)
    brightness = rng.uniform(0.3, 1.0, n) ** 2
    return (col * brightness[:, None]).astype(np.float32)
def disk_radii(n, rng, scale=0.18, r_min=0.01, r_max=0.85):
    """Pure numpy: exponential-falloff radii, re-rolling any that land outside the disk."""
    r = rng.exponential(scale, n)
    bad = (r < r_min) | (r > r_max)
    r[bad] = rng.uniform(r_min, r_max, bad.sum())
    return r.astype(np.float32)
def seed_spiral(n, rng_seed=0, arms=2, twist=3.5):
    """Pure numpy: stars scattered along logarithmic spiral arms."""
    rng = np.random.default_rng(rng_seed)
    r = disk_radii(n, rng)
    arm = rng.integers(0, arms, n)
    theta = arm * (2 * np.pi / arms) + twist * np.log(r / 0.01)
    theta = theta + rng.normal(0, 0.25, n) * (0.3 + r)
    return r, theta.astype(np.float32), star_colors(r, rng)
def seed_galaxy(kind, rng_seed=0):
    return seed_spiral(N_STARS, rng_seed)
def apply_seed(seed):
    r, theta, col = seed
    radius_f.from_numpy(r)
    angle_f.from_numpy(theta)
    color_f.from_numpy(col)
    pixels.fill(0.0)`,
          does: "disk_radii draws radii from an exponential distribution (dense core, sparse rim — how real disk galaxies distribute their stars), then RE-ROLLS any that fall outside the disk instead of clipping them. seed_spiral is the log spiral itself: theta = arm offset + twist * log(r), plus gaussian scatter so the arms are fuzzy bands, not pen strokes. star_colors blends a golden core (old stars) into blue-white arms (young stars) — real stellar-population colors.",
          why: "disk_radii's re-roll fixes a bug found while building this project: np.clip on an exponential PILES every out-of-range star at exactly r_max, which rendered as a crisp, bright, obviously-artificial circle around the whole galaxy. Re-rolling scatters them instead. Distribution edge behavior is a classic silent procgen bug — the histogram looks fine, the PICTURE gives it away.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["The scatter grows with radius — (0.3 + r) — arms are tight near the core, loose at the rim, as in photographs.", "twist * np.log(r / 0.01): the log is the entire difference between a lazy Archimedean spiral and the real thing."] },
        { title: "Additive starlight", adding: "the splat kernel and a static render loop.",
          code: `SPLAT_GAIN = 0.35
DISK_SCALE = 0.55
@ti.kernel
def splat():
    for s in radius_f:
        r = radius_f[s]
        a = angle_f[s]
        x = 0.5 + r * ti.cos(a) * DISK_SCALE
        y = 0.5 + r * ti.sin(a) * DISK_SCALE
        xi = ti.cast(x * RES, ti.i32)
        yi = ti.cast(y * RES, ti.i32)
        if 0 <= xi < RES and 0 <= yi < RES:
            pixels[xi, yi] += color_f[s] * SPLAT_GAIN
def step(dt=0.016):
    splat()
def main():
    init_sim()
    kind = SPIRAL
    apply_seed(seed_galaxy(kind))
    gui = ti.GUI("Galaxy Creator — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        step()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "Each star converts its polar state to screen x/y (cos/sin — the ONLY place the conversion happens) and adds its color into one pixel. ADDS: where thousands of dim stars overlap, light accumulates — the core blooms bright not because any star is bright but because so many share those pixels.",
          why: "Additive blending is physically honest for stars: light from separate sources really does sum. Project 07 chose additive for fire and glow; here it does the heavy lifting of making 60,000 identical dots read as a luminous object with structure — density becomes brightness for free.",
          see: "A frozen spiral galaxy: two grand arms of blue-white stars winding out of a blazing golden core. No motion yet.",
          checkpoint: "A static spiral. Beat 1.",
          recovery: ["Nothing fades yet, and splat runs every frame — brightness slowly saturates toward white. Chapter 2 fixes exactly this.", "DISK_SCALE shrinks the galaxy to fit the frame with margin."] }
      ]
    },
    {
      id: 2, title: "Differential rotation",
      build: "the three-line rotation kernel, then the fade that keeps the canvas from saturating.",
      beat: "The galaxy turns — inner stars lapping outer ones — with comet-tail motion trails.",
      steps: [
        { title: "Inner stars lap outer stars", adding: "rotation dials and the kernel that spins the disk.",
          code: `ROT_SPEED = 0.35
ROT_SOFTEN = 0.05
DT = 0.016
@ti.kernel
def rotate(dt: ti.f32):
    for s in radius_f:
        omega = ROT_SPEED / (radius_f[s] + ROT_SOFTEN)
        angle_f[s] += omega * dt
def step(dt=DT):
    rotate(dt)
    splat()`,
          does: "omega ~ 1/r encodes a FLAT ROTATION CURVE: stars at different radii move at roughly the same linear speed, so inner stars sweep far more ANGLE per second than outer ones. One addition per star per tick — the payoff of chapter 1's polar coordinates.",
          why: "Flat rotation curves are one of astronomy's most famous facts (they're the original evidence for dark matter — the outer stars orbit 'too fast' for the visible mass). Here that physics is one division. And differential rotation is what WINDS spiral structure: watch the arms stretch and shear as inner stars pull ahead.",
          see: "The galaxy rotates — but every star still deposits light every frame with nothing erasing it, so the disk rapidly smears into a solid, saturated blur. Real motion, unreadable picture.",
          checkpoint: "It spins, and it smears to a blur. That's the expected failure. No red text.",
          recovery: ["ROT_SOFTEN in the denominator keeps the innermost stars from spinning at near-infinite speed — 1/r explodes at r=0 without it."] },
        { title: "Fade, so motion can exist", adding: "the fade and clamp kernels, completing the tick.",
          code: `FADE = 0.88
@ti.kernel
def fade():
    for i, j in pixels:
        pixels[i, j] *= FADE
@ti.kernel
def clamp_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.min(pixels[i, j], 1.0)
def step(dt=DT):
    rotate(dt)
    fade()
    splat()
    clamp_pixels()`,
          does: "fade multiplies every pixel by 0.88 each frame — old light decays exponentially, so a star's past positions linger as a dimming trail while its current position stays bright. clamp keeps the additive pile-up in the core from exceeding displayable range. Project 07's exact canvas discipline (fade, splat, clamp), reused.",
          why: "The fade constant is an artistic dial disguised as bookkeeping: at 0.88 each star wears a short comet tail that makes the differential rotation VISIBLE as curved streaks — the inner disk's tight circles versus the rim's lazy arcs. Try 0.6 (crisp dots, no trails) and 0.98 (long ghostly smears) to feel it.",
          see: "The blur snaps into focus: a rotating galaxy with motion trails, the core a bright whirl of tight fast circles, the arms sweeping grandly. This is the money shot.",
          checkpoint: "A spinning, trailing spiral. Beat 2.",
          recovery: ["Order matters in step: fade BEFORE splat, so this frame's star positions land at full brightness on an already-dimmed canvas."] },
        { title: "Reroll", adding: "the reseed key.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))`,
          does: "A fresh seed, a fresh galaxy — apply_seed already clears the canvas (pixels.fill(0.0)), so no ghost of the old galaxy survives.",
          why: "Reset hygiene, the recurring lesson: apply_seed owning the canvas clear means every code path that reseeds — this key now, the type-switcher next chapter — gets a clean slate without remembering to ask.",
          see: "Tap R for new arm patterns — same physics, different roll.",
          checkpoint: "R rerolls. No red text.",
          recovery: ["Same reseed idiom as every project since 01."] }
      ]
    },
    {
      id: 3, title: "A zoo of galaxies",
      build: "elliptical and ring types sharing the same star engine, plus the type-switcher keys.",
      beat: "Three galaxy species — spiral, elliptical, ring — one keystroke apart.",
      steps: [
        { title: "The elliptical", adding: "the armless golden blob and a two-way dispatch.",
          code: `SPIRAL, ELLIPTICAL, RING = 0, 1, 2
def seed_elliptical(n, rng_seed=0):
    """Pure numpy: a smooth, armless, golden-old-star blob."""
    rng = np.random.default_rng(rng_seed)
    r = disk_radii(n, rng, scale=0.22)
    theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
    col = star_colors(r, rng, core_scale=0.3, core_col=(0.9, 0.75, 0.5), arm_col=(0.8, 0.6, 0.4))
    return r, theta, col
def seed_galaxy(kind, rng_seed=0):
    if kind == SPIRAL:
        return seed_spiral(N_STARS, rng_seed)
    return seed_elliptical(N_STARS, rng_seed)`,
          does: "An elliptical galaxy is what you get by REMOVING structure: theta is uniform random (no arms, no correlation with radius), and the palette is golds and ambers throughout — real ellipticals are old-star populations with no gas left to form hot young blue ones.",
          why: "Compare the two seeders line by line: the entire difference between a spiral and an elliptical is whether theta depends on log(r). One term. The rotation, the splat, the fade — every kernel downstream is untouched, which is exactly what a clean data/physics split buys.",
          see: "Runs clean; no key reaches it yet.",
          checkpoint: "No red text.",
          recovery: ["star_colors' defaults get overridden per type — the function was built parameterized in chapter 1 for exactly this moment."] },
        { title: "The ring", adding: "the thin blue ring with a sparse core, completing the dispatch.",
          code: `def seed_ring(n, rng_seed=0):
    """Pure numpy: a thin ring of hot blue stars with a sparse old core."""
    rng = np.random.default_rng(rng_seed)
    n_core = n // 5
    n_ring = n - n_core
    r_ring = rng.normal(0.55, 0.045, n_ring)
    r_core = rng.exponential(0.06, n_core)
    r = np.clip(np.concatenate([r_ring, r_core]), 0.01, 0.85).astype(np.float32)
    theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
    col = star_colors(r, rng, core_scale=0.1, core_col=(0.9, 0.8, 0.6), arm_col=(0.4, 0.65, 1.0))
    return r, theta, col
def seed_galaxy(kind, rng_seed=0):
    if kind == SPIRAL:
        return seed_spiral(N_STARS, rng_seed)
    if kind == ELLIPTICAL:
        return seed_elliptical(N_STARS, rng_seed)
    return seed_ring(N_STARS, rng_seed)`,
          does: "Ring galaxies (think Hoag's Object) are two populations in one: a thin gaussian band of radii around 0.55 holding hot blue young stars, and a small exponential core of old ones — built here by literally concatenating two numpy draws before upload.",
          why: "Three galaxy types, three DISTRIBUTIONS, zero new kernels. This is the procgen thesis of the whole arc in miniature: the renderer and physics are a fixed instrument; what you feed it — a distribution over radius and angle — is the entire identity of the thing on screen.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["np.concatenate joins ring stars first, core stars second — order doesn't matter downstream since every star is independent."] },
        { title: "The switchboard", adding: "the name table, the 1/2/3 keys, and the HUD.",
          code: `NAMES = {SPIRAL: "spiral", ELLIPTICAL: "elliptical", RING: "ring"}
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in "123":
                kind = int(e.key) - 1
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))
            elif e.key == "r":
                apply_seed(seed_galaxy(kind, rng_seed=np.random.randint(1_000_000)))
        gui.text(f"galaxy: {NAMES[kind]}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("[1] spiral  [2] elliptical  [3] ring  [r] reroll", (0.02, 0.94), color=0xAAAAAA)`,
          does: "1/2/3 pick a species and reroll it fresh; R rerolls the current one; the HUD names what you're looking at — the same preset-switch pattern as project 01's Gray-Scott presets, five arcs later.",
          why: "Watch what differential rotation does to each type: the spiral's arms wind tighter and tighter; the elliptical, having no structure to shear, just shimmers; the ring stays a ring because all its stars share a radius and therefore a speed. One physics rule, three different fates — decided entirely by the initial distribution.",
          see: "Flip between species and watch each respond to the same spin in its own way. Park on the spiral and watch the arms slowly wind up over a minute — that's the real reason old galaxies' arms are so tightly wrapped.",
          checkpoint: "Three switchable galaxy types. Final beat — project 14 complete.",
          recovery: ["e.key in '123' then int(e.key) - 1 — the same key-to-index trick as project 07's material picker."] }
      ]
    }
  ]
};

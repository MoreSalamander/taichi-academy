// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["13-planet-generator"] = {
  project: "13-planet-generator",
  title: "Planet Generator",
  pitch: "Sample 3D noise ON the sphere's surface and map seams simply cannot exist — a whole world from one seed.",
  tier: "hard",
  language: "Python",
  file: "planet.py",
  chapters: [
    {
      id: 1, title: "Noise for a world",
      build: "the noise volume, its generator, and an empty window waiting for a sphere.",
      beat: "Black space, a full 3D terrain volume already loaded behind it.",
      steps: [
        { title: "The map-seam problem", adding: "the docstring and imports.",
          code: `"""Planet Generator: a whole world from one seed — 3D noise on a sphere, no map seams."""
import math
import numpy as np
import taichi as ti`,
          does: "The obvious way to texture a planet — generate a 2D map, wrap it around a sphere — has two famous defects: a visible seam where the map's left and right edges meet, and ugly pinching at the poles where the whole top row of the map squeezes into a single point. This project sidesteps BOTH with one idea: generate noise in 3D, and ask for its value AT each point on the sphere's surface. A sphere sitting inside a continuous 3D field has no seams and no poles — every surface point is just a coordinate like any other.",
          why: "This is the standard professional solution (solid texturing), and you already own every ingredient: project 12 built 3D fbm noise and trilinear sampling; project 05 built elevation coloring. The new pieces are a ray-SPHERE intersection (project 12 did ray-box) and real surface lighting.",
          see: "Runs clean.",
          checkpoint: "python3 planet.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "A volume and a canvas", adding: "resolution dials and both fields.",
          code: `RES = 400
VOL_N = 64
elevation = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global elevation, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    elevation = ti.field(ti.f32, shape=(VOL_N, VOL_N, VOL_N))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "elevation is a 3D scalar field, exactly like project 12's density — except the planet only ever READS the thin spherical shell of it that its surface passes through. The interior and corners of the cube are generated and never looked at.",
          why: "That 'waste' is the whole trick: generating the full volume is cheap (it's just numpy noise), and in exchange, every point ON the sphere gets a well-defined, smoothly-varying value with no seams possible by construction — the sphere never knows a 2D map ever existed.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["VOL_N (64) is slightly finer than project 12's 48 — surface terrain rewards detail more than fog does."] },
        { title: "Fill the volume", adding: "the 3D fbm stack from project 12 and an empty render loop.",
          code: `def resize_trilinear(a, n):
    """Pure numpy: trilinear resize of a small 3D array up to n x n x n."""
    m = a.shape[0]
    x = np.linspace(0.0, m - 1.0, n)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = (x - i0).astype(np.float32)
    a = a[i0] * (1 - f)[:, None, None] + a[i1] * f[:, None, None]
    a = a[:, i0] * (1 - f)[None, :, None] + a[:, i1] * f[None, :, None]
    a = a[:, :, i0] * (1 - f)[None, None, :] + a[:, :, i1] * f[None, None, :]
    return a
def fbm3d(n, rng_seed=0, octaves=5, roughness=0.55):
    """Pure numpy: fractal 3D noise — octaves of noise, each finer and fainter."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros((n, n, n), dtype=np.float32)
    amp = 1.0
    res = 4
    for _ in range(octaves):
        layer = rng.uniform(0.0, 1.0, size=(res, res, res)).astype(np.float32)
        out += amp * resize_trilinear(layer, n)
        amp *= roughness
        res *= 2
    out -= out.min()
    out /= out.max()
    return out.astype(np.float32)
def apply_seed(rng_seed=0):
    elevation.from_numpy(fbm3d(VOL_N, rng_seed))
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Planet Generator — taichi-academy", res=RES, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "resize_trilinear and fbm3d are project 12's, verbatim (with one more octave, 5, for extra terrain detail). apply_seed fills the volume. The render loop shows... nothing: pixels was allocated to zero and nothing writes it yet, so the window is pure black space.",
          why: "Deliberately anticlimactic — the DATA for a whole world now exists in memory, and there is no way to see it. Chapter 2's entire job is building the eye: a camera, a ray per pixel, and the intersection test that finds the sphere. Starting from black space makes the moment the planet appears land harder.",
          see: "A black window. Space, waiting.",
          checkpoint: "An empty black canvas — that's correct. Beat 1.",
          recovery: ["If typing fbm3d felt like deja vu, it should — it's project 12's, one octave deeper. Fifth project to reuse the fbm idea (05 in 2D, 12 in 3D, now here)."] }
      ]
    },
    {
      id: 2, title: "A sphere in the void",
      build: "ray-sphere intersection and a diffuse-lit bare rock, orbited by the project 12 camera.",
      beat: "A gray moon hangs in space, lit from one side, slowly orbited.",
      steps: [
        { title: "Where does a ray meet a sphere?", adding: "the planet radius and the quadratic intersection test.",
          code: `PLANET_R = 1.0
@ti.func
def ray_sphere(origin, rd):
    b = origin.dot(rd)
    c = origin.dot(origin) - PLANET_R * PLANET_R
    disc = b * b - c
    t = -1.0
    if disc > 0:
        t = -b - ti.sqrt(disc)
    return t`,
          does: "A point along the ray is origin + rd*t; asking 'when is that point at distance PLANET_R from the center?' expands into a quadratic in t. b and c are that quadratic's coefficients (simplified because rd is unit-length and the sphere sits at the origin); disc is its discriminant — negative means the ray misses entirely. The smaller root (-b - sqrt(disc)) is the NEAR intersection, the side facing the camera.",
          why: "Project 12's ray_box was solved with interval logic (slabs); a sphere is solved with algebra (a quadratic). These two intersection tests — box and sphere — are the two atoms nearly every ray-traced scene is built from, and now you've written both from scratch.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Taking -b - sqrt(disc), not -b + sqrt(disc), matters: the + root is the FAR side of the sphere, where the ray exits — drawing that would show you the planet inside-out."] },
        { title: "First light", adding: "camera dials, a diffuse-lit gray sphere, and the orbit loop from project 12.",
          code: `CAM_RADIUS = 2.6
CAM_HEIGHT = 0.5
ORBIT_SPEED = 0.004
ZOOM = 1.8
SUN = (0.6, 0.3, -0.6)
@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32, sunx: ti.f32, suny: ti.f32, sunz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        col = ti.Vector([0.02, 0.02, 0.05])
        t = ray_sphere(cam, rd)
        if t > 0:
            p = cam + rd * t
            n = p.normalized()
            diffuse = ti.max(n.dot(sun), 0.0)
            shade = 0.06 + 0.94 * diffuse
            col = ti.Vector([0.6, 0.6, 0.62]) * shade

        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)
def camera_position(theta):
    return (
        CAM_RADIUS * math.sin(theta),
        CAM_HEIGHT,
        -CAM_RADIUS * math.cos(theta),
    )
    theta = 0.0
        theta += ORBIT_SPEED
        cx, cy, cz = camera_position(theta)
        render(cx, cy, cz, *SUN)`,
          does: "The camera-basis construction (forward/right/up, a ray per pixel) is project 12's, looking at the origin. The one profound simplification a sphere buys: its surface normal is FREE — n = p.normalized(), the hit point's own direction from the center. diffuse = n dot sun is project 05's hillshade equation, on a sphere. ZOOM (multiplying forward) narrows the field of view so the planet fills the frame.",
          why: "n dot sun is now the THIRD appearance of the normal-dot-light idea (05's terrain, 12's implicit sun march, now here) — by the mandelbulb it should feel like breathing. Notice how much of this 40-line kernel you've written before; the genuinely new content this chapter is ray_sphere's six lines of algebra.",
          see: "A bare gray moon hangs in the black, one side sunlit, the terminator (day/night line) curving across it, orbiting slowly. No terrain, no color — a stage waiting for a world.",
          checkpoint: "A lit gray sphere in space. Beat 2.",
          recovery: ["theta = 0.0 goes in main() before the loop; the theta update and the render call replace the loop body's dead air.", "Planet looks tiny — check ZOOM multiplies forward INSIDE the rd sum, not the whole rd."] }
      ]
    },
    {
      id: 3, title: "A world on the surface",
      build: "surface sampling, a quantile-anchored sea level, and elevation-banded terrain with polar ice.",
      beat: "The gray moon becomes a living planet — oceans, continents, ice caps.",
      steps: [
        { title: "Read the noise where the ray landed", adding: "trilinear sampling on the sphere's surface.",
          code: `@ti.func
def sample_elevation(p):
    q = ti.math.clamp(p * 0.5 + 0.5, 0.0, 1.0)
    x = q * (VOL_N - 1)
    x0 = ti.cast(ti.floor(x), ti.i32)
    x1 = ti.min(x0 + 1, VOL_N - 1)
    f = x - x0
    c00 = elevation[x0[0], x0[1], x0[2]] * (1 - f[0]) + elevation[x1[0], x0[1], x0[2]] * f[0]
    c10 = elevation[x0[0], x1[1], x0[2]] * (1 - f[0]) + elevation[x1[0], x1[1], x0[2]] * f[0]
    c01 = elevation[x0[0], x0[1], x1[2]] * (1 - f[0]) + elevation[x1[0], x0[1], x1[2]] * f[0]
    c11 = elevation[x0[0], x1[1], x1[2]] * (1 - f[0]) + elevation[x1[0], x1[1], x1[2]] * f[0]
    c0 = c00 * (1 - f[1]) + c10 * f[1]
    c1 = c01 * (1 - f[1]) + c11 * f[1]
    return c0 * (1 - f[2]) + c1 * f[2]`,
          does: "Project 12's trilinear sampler with one new line up front: q = p * 0.5 + 0.5 maps the sphere's surface coordinates (each component in [-1, 1]) into the volume's [0, 1] cube. The unit sphere sits perfectly inscribed in the noise volume; every surface point lands somewhere inside it.",
          why: "This single function IS the no-seams guarantee: two surface points that are neighbors in 3D space get near-identical elevation values, no matter which 'side' of any would-be map they'd have fallen on. There is no wraparound edge because there is no map — just one continuous field.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Compare against project 12's sample_density — the ONLY differences are the field name and the [-1,1] to [0,1] remap. Worth diffing by eye to convince yourself."] },
        { title: "Every planet gets the same ocean", adding: "the quantile-anchored terrain seeder.",
          code: `OCEAN_FRACTION = 0.62
SEA_LEVEL = 0.5
ELEV_GAIN = 1.5
def seed_terrain(n, rng_seed=0):
    """Pure numpy: fbm noise, re-anchored so a fixed fraction of the world is ocean."""
    noise = fbm3d(n, rng_seed)
    sea = np.quantile(noise, OCEAN_FRACTION)
    return np.clip(SEA_LEVEL + (noise - sea) * ELEV_GAIN, 0.0, 1.0).astype(np.float32)
def apply_seed(rng_seed=0):
    elevation.from_numpy(seed_terrain(VOL_N, rng_seed))`,
          does: "np.quantile(noise, 0.62) finds the exact value below which 62% of the noise lies — THIS seed's own natural sea level. Shifting the noise so that value lands exactly at SEA_LEVEL (0.5) means every planet, regardless of seed, is 62% ocean by construction. ELEV_GAIN stretches the remaining variation so mountains still reach the upper color bands.",
          why: "This fixes a real bug found while building this project: raw fbm's value distribution DRIFTS between seeds (one seed's median was 0.56, another's 0.48), so any fixed sea-level constant gave some planets sprawling supercontinents and others near-total ocean. Anchoring to a percentile instead of a value makes the LOOK of the world a controlled design decision and leaves only the SHAPE to the seed — exactly the kind of determinism this curriculum keeps choosing.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["seed_terrain REPLACES the direct fbm3d call inside apply_seed — chapter 1's version of apply_seed gets rewritten, not appended to."] },
        { title: "Paint the world", adding: "the band helper, the full surface-color function, and terrain in the renderer.",
          code: `ICE_LAT = 0.72
@ti.func
def band(c0, c1, hh, lo, hi):
    t = ti.math.clamp((hh - lo) / (hi - lo), 0.0, 1.0)
    return c0 * (1.0 - t) + c1 * t
@ti.func
def surface_color(h, lat):
    c = ti.Vector([0.05, 0.15, 0.45])
    if h > SEA_LEVEL:
        land = (h - SEA_LEVEL) / (1.0 - SEA_LEVEL)
        c = band(ti.Vector([0.75, 0.7, 0.45]), ti.Vector([0.2, 0.5, 0.2]), land, 0.0, 0.35)
        c = band(c, ti.Vector([0.45, 0.4, 0.35]), land, 0.45, 0.75)
        c = band(c, ti.Vector([0.95, 0.95, 0.98]), land, 0.8, 0.95)
    else:
        shallow = h / SEA_LEVEL
        c = band(ti.Vector([0.02, 0.08, 0.3]), ti.Vector([0.1, 0.4, 0.6]), shallow, 0.4, 1.0)
    if lat > ICE_LAT:
        ice = (lat - ICE_LAT) / (1.0 - ICE_LAT)
        c = c * (1.0 - ice) + ti.Vector([0.95, 0.97, 1.0]) * ice
    return c
@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32, sunx: ti.f32, suny: ti.f32, sunz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        col = ti.Vector([0.02, 0.02, 0.05])
        t = ray_sphere(cam, rd)
        if t > 0:
            p = cam + rd * t
            n = p.normalized()
            h = sample_elevation(n)
            c = surface_color(h, ti.abs(n[1]))

            diffuse = ti.max(n.dot(sun), 0.0)
            shade = 0.06 + 0.94 * diffuse
            col = c * shade

        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)`,
          does: "band is project 05's elevation-palette helper, character for character. surface_color chains it: beach sand to grass to rock to snow above sea level; deep-to-shallow blues below. And one thing 2D terrain never had: LATITUDE. On a sphere, n[1] (the surface point's height along the planet's axis) IS latitude — |n[1]| > ICE_LAT means polar, and the color blends toward ice regardless of what's underneath.",
          why: "Ice caps cost three lines because the sphere's geometry hands you latitude for free — the same n vector drives lighting, elevation sampling, AND climate. That's the density of information a good representation buys: one unit vector, three different physical meanings.",
          see: "The gray moon turns into a WORLD: blue oceans, sandy coastlines, green interiors, gray-white mountain ridges, white caps at both poles — day side lit, night side dark, orbiting slowly.",
          checkpoint: "A full terrain planet. Beat 3.",
          recovery: ["surface_color takes ti.abs(n[1]) — both poles, north and south, freeze symmetrically.", "render is a REPLACE of chapter 2's — the terrain lines slot in where the flat gray color was."] }
      ]
    },
    {
      id: 4, title: "Atmosphere and hands",
      build: "ocean specular, a blue atmosphere rim, drag-to-orbit, and a reseed key.",
      beat: "A planet with glinting seas and a glowing atmosphere, spun by hand.",
      steps: [
        { title: "Sun-glint and a blue halo", adding: "specular reflection on water and a fresnel-style rim (replace render).",
          code: `@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32, sunx: ti.f32, suny: ti.f32, sunz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        col = ti.Vector([0.02, 0.02, 0.05])
        t = ray_sphere(cam, rd)
        if t > 0:
            p = cam + rd * t
            n = p.normalized()
            h = sample_elevation(n)
            c = surface_color(h, ti.abs(n[1]))

            diffuse = ti.max(n.dot(sun), 0.0)
            spec = 0.0
            if h <= SEA_LEVEL:
                refl = 2.0 * n.dot(sun) * n - sun
                spec = ti.pow(ti.max(refl.dot(-rd), 0.0), 32) * 0.6
            shade = 0.06 + 0.94 * diffuse
            col = c * shade + ti.Vector([1.0, 0.95, 0.85]) * spec

            rim = ti.pow(1.0 - ti.max(n.dot(-rd), 0.0), 3)
            col += ti.Vector([0.3, 0.5, 1.0]) * rim * 0.8 * (0.3 + 0.7 * diffuse)

        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)`,
          does: "Two new light behaviors. SPECULAR (water only): refl mirrors the sun direction across the surface normal — the textbook reflection formula — and refl dot -rd asks 'does that mirror bounce aim at MY eye?', sharpened by pow(.., 32) into a tight glint. RIM: n dot -rd measures how face-on the surface is to the camera; 1 minus that, cubed, is large only at the silhouette edge — where, on a real planet, you'd be looking through the most atmosphere.",
          why: "Diffuse, specular, and fresnel-rim are the three workhorse terms of practically every lighting model ever shipped — and here each one is doing a legible, physical job: diffuse says 'day side', specular says 'this is liquid', rim says 'this has air'. Land doesn't glint (skipped by the h <= SEA_LEVEL gate) because rock is matte; that one conditional is what makes the oceans read as WATER.",
          see: "A bright sun-glint slides across the oceans as the planet turns, and a thin blue halo hugs the sunlit edge of the disc — unmistakably a planet with an atmosphere now, not a painted ball.",
          checkpoint: "Glinting seas, glowing rim. No red text.",
          recovery: ["The specular gate is h <= SEA_LEVEL — put it on land too and the whole planet looks like polished plastic.", "rim uses n dot MINUS rd — the ray points AT the surface; the comparison needs the direction back toward the camera."] },
        { title: "Take the wheel", adding: "drag-to-orbit, replacing the automatic-only camera.",
          code: `    pmx = None
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED`,
          does: "The identical drag-to-orbit block from project 12 — hold to steer, release to resume the slow automatic spin.",
          why: "Fourth appearance of the pmx idiom (08, 09/10's grab, 12, now here). This is what 'reusing muscles' means in practice: entire interaction patterns, not just functions, become vocabulary.",
          see: "Drag to spin the planet under your hand; the sun stays fixed, so you can chase the terminator or park over the glint.",
          checkpoint: "Mouse-driven orbit. No red text.",
          recovery: ["Both pmx lines — the declaration in main() AND the reset in the else branch."] },
        { title: "Infinite worlds", adding: "the reseed key and the HUD.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))
        gui.text("drag to orbit  [r] new planet", (0.02, 0.98), color=0xFFFFFF)`,
          does: "R rebuilds the terrain volume from a fresh seed — new continents, same 62% ocean, same physics of light.",
          why: "Because of chapter 3's quantile anchor, every reroll is a PLAUSIBLE world — never all-land, never all-sea — so the differences you notice between planets are the interesting ones: where the continents sit, how the archipelagos chain, which pole got the bigger cap.",
          see: "Tap R and tour worlds nobody will ever see again: island chains, one-continent worlds, twin landmasses split by a strait — every one of them seamless, poleless, and yours.",
          checkpoint: "A planet-a-keystroke generator. Final beat — project 13 complete.",
          recovery: ["Same reseed idiom as every project since 01."] }
      ]
    }
  ]
};

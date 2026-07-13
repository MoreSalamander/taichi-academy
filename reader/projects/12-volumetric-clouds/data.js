// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["12-volumetric-clouds"] = {
  project: "12-volumetric-clouds",
  title: "Volumetric Clouds",
  pitch: "No mesh, no surface — a cloud is a NUMBER at every point in space, and a camera ray reads through it.",
  tier: "hard",
  language: "Python",
  file: "volumetric_clouds.py",
  chapters: [
    {
      id: 1, title: "A cloud made of noise",
      build: "a 3D density field — fractal noise shaped into a cloud layer — and a cheap top-down proof it looks right.",
      beat: "A fluffy, cloud-shaped blob, seen from directly above, no camera needed yet.",
      steps: [
        { title: "A different kind of 'solid'", adding: "the docstring and imports.",
          code: `"""Volumetric Clouds: 3D fractal noise, ray marching, and a light march for real shading."""
import math
import numpy as np
import taichi as ti`,
          does: "Every project so far that drew a shape — MPM's snow, soft body's rings, cloth's grid — represented it with PARTICLES or a MESH: discrete points with an agreed-upon connection between them. A cloud has no such boundary. It's represented as a single NUMBER (density: how much cloud-stuff is here) defined at every point in a 3D volume — mostly zero, thicker in some regions, with no edges to speak of.",
          why: "This is a genuinely different representation from everything in this curriculum so far, and it demands a genuinely different renderer: you can't hand a density FIELD to a triangle rasterizer. You have to sample it directly, along a ray, which is exactly what this whole project builds toward.",
          see: "Runs clean.",
          checkpoint: "python3 volumetric_clouds.py returns silently.",
          recovery: ["Usual venv setup."] },
        { title: "A box of numbers", adding: "resolution dials, the density field, and the render canvas.",
          code: `RES = 400
VOL_N = 48
density = None
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global density, pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    density = ti.field(ti.f32, shape=(VOL_N, VOL_N, VOL_N))
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "density is a genuine 3D field — VOL_N cubed scalars, one 'how much cloud is here' value per lattice point in a unit cube. pixels is the ordinary 2D screen canvas every project since 01 has used.",
          why: "VOL_N (48) is deliberately much smaller than RES (400) — the density field is a coarse LATTICE that gets smoothly interpolated when sampled (next chapter), the same 'few control points, smooth interpolation' idea as every bilerp/trilerp this curriculum has used, just one dimension higher.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["density is a plain scalar field (ti.field), not a Vector.field — one number per lattice point, not a color or position."] },
        { title: "Prove the noise looks like a cloud", adding: "3D fractal noise, vertical shaping, seeding, and a cheap top-down debug render.",
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
def fbm3d(n, rng_seed=0, octaves=4, roughness=0.55):
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
def seed_density(n, rng_seed=0):
    """Pure numpy: fbm noise shaped into a cloud layer by a vertical profile."""
    noise = fbm3d(n, rng_seed)
    yv = np.linspace(0.0, 1.0, n)
    profile = np.clip(1.0 - np.abs(yv - 0.45) / 0.35, 0.0, 1.0) ** 1.2
    shaped = noise * profile[None, :, None]
    return np.clip((shaped - 0.22) / 0.78, 0.0, 1.0).astype(np.float32)
def apply_seed(rng_seed=0):
    density.from_numpy(seed_density(VOL_N, rng_seed))
@ti.kernel
def render():
    for i, j in pixels:
        di = ti.min(ti.cast(i / RES * VOL_N, ti.i32), VOL_N - 1)
        dj = ti.min(ti.cast(j / RES * VOL_N, ti.i32), VOL_N - 1)
        m = 0.0
        for k in range(VOL_N):
            m = ti.max(m, density[di, k, dj])
        pixels[i, j] = ti.Vector([m, m, m])
def main():
    init_sim()
    apply_seed()
    gui = ti.GUI("Volumetric Clouds — taichi-academy", res=RES, background_color=0x0A0A12)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "resize_trilinear/fbm3d are project 05's resize_bilinear/fbm_terrain, promoted from 2D to 3D — three resize passes (one per axis) instead of two, otherwise identical in spirit. seed_density then SHAPES that raw noise: a vertical profile (peaked around y=0.45, tapering over a 0.35 band) multiplies it into a distinct LAYER rather than filling the whole cube, and a threshold (subtract 0.22, rescale) carves 'coverage' — turning smooth noise into distinct puffy clumps with real empty space between them. render, for now, is a cheap trick: for each screen pixel, take the BRIGHTEST density value anywhere along that vertical column — a top-down max-intensity projection, not a real camera at all, just enough to sanity-check the shape.",
          why: "This mirrors project 05's own opening move: 'raw data first, pretty later — you can already sanity-check the fractal before any lighting flatters it.' A real ray-marched camera is chapter 2's whole subject; committing to it before confirming the underlying noise even LOOKS like a cloud would make debugging chapter 2 much harder to reason about.",
          see: "A soft, irregular white blob with fluffy, cauliflower-like edges — recognizably cloud-shaped, floating on black, viewed from directly above.",
          checkpoint: "A cloud-shaped noise blob, top-down. Beat 1.",
          recovery: ["seed_density's threshold, (shaped - 0.22) / 0.78, is doing double duty: 0.22 sets HOW MUCH of the noise counts as 'cloud' (raise it for wispier, patchier cover), and dividing by 0.78 rescales what's left back to fill [0, 1].", "resize_trilinear needs THREE blend passes (x, then y, then z) where resize_bilinear needed two — one more dimension, one more pass, same blend-by-fraction idea each time."] }
      ]
    },
    {
      id: 2, title: "A camera enters the volume",
      build: "trilinear sampling, a ray/box intersection, and a perspective ray marcher — real 3D, flat lighting.",
      beat: "The cloud becomes a real, orbitable 3D object — flat-lit, but genuinely volumetric.",
      steps: [
        { title: "Read the field at any point, and where a ray meets the box", adding: "march dials, trilinear sampling, and the box-intersection test.",
          code: `STEPS = 96
ABSORPTION = 18.0
AMBIENT = 0.25
@ti.func
def sample_density(p):
    v = 0.0
    if 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 and 0.0 <= p[2] <= 1.0:
        x = p * (VOL_N - 1)
        x0 = ti.cast(ti.floor(x), ti.i32)
        x1 = ti.min(x0 + 1, VOL_N - 1)
        f = x - x0
        c00 = density[x0[0], x0[1], x0[2]] * (1 - f[0]) + density[x1[0], x0[1], x0[2]] * f[0]
        c10 = density[x0[0], x1[1], x0[2]] * (1 - f[0]) + density[x1[0], x1[1], x0[2]] * f[0]
        c01 = density[x0[0], x0[1], x1[2]] * (1 - f[0]) + density[x1[0], x0[1], x1[2]] * f[0]
        c11 = density[x0[0], x1[1], x1[2]] * (1 - f[0]) + density[x1[0], x1[1], x1[2]] * f[0]
        c0 = c00 * (1 - f[1]) + c10 * f[1]
        c1 = c01 * (1 - f[1]) + c11 * f[1]
        v = c0 * (1 - f[2]) + c1 * f[2]
    return v
@ti.func
def ray_box(origin, rd):
    inv = 1.0 / rd
    t0 = (ti.Vector([0.0, 0.0, 0.0]) - origin) * inv
    t1 = (ti.Vector([1.0, 1.0, 1.0]) - origin) * inv
    tmin = ti.min(t0, t1)
    tmax = ti.max(t0, t1)
    t_enter = ti.max(ti.max(tmin[0], tmin[1]), tmin[2])
    t_exit = ti.min(ti.min(tmax[0], tmax[1]), tmax[2])
    return t_enter, t_exit`,
          does: "sample_density is trilinear interpolation — bilerp's 4-corner blend, extended to 8 corners of a cube, read at any CONTINUOUS point (not just lattice indices) — and it returns exactly 0 outside the unit cube, so a ray never needs a separate bounds check once it's inside. ray_box is the classic 'slab method': for each of the 3 axis-aligned pairs of box faces, find where the ray crosses them (t0, t1), and the ray is INSIDE the box exactly where all three per-axis intervals overlap — t_enter is the latest entry, t_exit the earliest exit.",
          why: "Every camera ray needs to know WHERE to start and stop marching — without ray_box, you'd either march from negative infinity (wasteful) or hardcode a range (wrong for every camera angle). This is the standard technique behind every ray-traced/ray-marched box or bounding volume you've ever seen rendered.",
          see: "Runs clean; nothing calls these yet.",
          checkpoint: "No red text.",
          recovery: ["1.0 / rd can produce +-infinity when a ray is exactly axis-aligned (rd[axis] == 0) — that's fine and intentional; IEEE float infinity arithmetic makes the min/max comparisons still work out correctly, no special-casing needed."] },
        { title: "March through the fog", adding: "the real ray marcher — replacing the top-down debug view.",
          code: `@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    center = ti.Vector([CENTER[0], CENTER[1], CENTER[2]])
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (center - cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward + u * right + v * up).normalized()

        t_enter, t_exit = ray_box(cam, rd)
        t_enter = ti.max(t_enter, 0.0)

        transmittance = 1.0
        color = ti.Vector([0.0, 0.0, 0.0])
        if t_exit > t_enter:
            step_size = (t_exit - t_enter) / STEPS
            t = t_enter
            for _s in range(STEPS):
                p = cam + rd * t
                d = sample_density(p)
                if d > 0.001:
                    alpha = 1.0 - ti.exp(-d * step_size * ABSORPTION)
                    color += transmittance * alpha * AMBIENT * ti.Vector([1.0, 0.97, 0.92])
                    transmittance *= 1.0 - alpha
                    if transmittance < 0.01:
                        break
                t += step_size

        sky_t = ti.max(rd[1], 0.0)
        sky = ti.Vector([0.5, 0.65, 0.9]) * (1.0 - sky_t) + ti.Vector([0.2, 0.35, 0.7]) * sky_t
        pixels[i, j] = ti.math.clamp(color + transmittance * sky, 0.0, 1.0)`,
          does: "For every pixel: build a camera ray (forward/right/up from camera to a fixed look-at CENTER — the same basis-vector construction GGUI's camera used, hand-built here since this is a flat 2D canvas project, not 3D GGUI), find where it crosses the cloud's bounding box, then step along it STEPS times. Each step where density is nonzero absorbs some light (alpha, via Beer's law — ti.exp(-density * distance)) and adds its own glow (AMBIENT), while transmittance — how much background light still gets through — shrinks multiplicatively. transmittance < 0.01 triggers an early exit: fully opaque fog has nothing left to reveal behind it.",
          why: "This loop — accumulate color, shrink transmittance, multiply by Beer's law each step — is THE standard volumetric rendering integral, approximated by marching. It's the same idea behind every volumetric fog, smoke, or cloud effect in modern games and film, and it works on ANY density field, not just this fbm cloud.",
          see: "The flat top-down blob becomes a real 3D shape you're looking at from an angle — puffy, dimensional, with a visible silhouette against the sky gradient below. But it's flat-shaded: no shadow detail, no bright and dark sides, just a uniformly lit cloud.",
          checkpoint: "A real 3D cloud, flatly lit. No red text yet — nothing calls render with a moving camera.",
          recovery: ["t_enter = ti.max(t_enter, 0.0) matters when the camera is INSIDE the box (t_enter negative) — marching should start at the camera, not behind it.", "The early-exit break only fires INSIDE the if d > 0.001 block — empty space between cloud puffs still costs a loop iteration, just a cheap one."] },
        { title: "Give it an orbit", adding: "camera dials and the position formula, wired into a moving main loop.",
          code: `CENTER = (0.5, 0.35, 0.5)
CAM_RADIUS = 1.1
CAM_HEIGHT = 0.35
ORBIT_SPEED = 0.01
def camera_position(theta):
    return (
        CENTER[0] + CAM_RADIUS * math.sin(theta),
        CENTER[1] + CAM_HEIGHT,
        CENTER[2] - CAM_RADIUS * math.cos(theta),
    )
    theta = 0.0
        theta += ORBIT_SPEED
        cx, cy, cz = camera_position(theta)
        render(cx, cy, cz)`,
          does: "camera_position walks a circle of CAM_RADIUS around CENTER at a fixed CAM_HEIGHT — sin/cos of a slowly-increasing angle, the oldest trick for 'orbit around a point' there is. theta creeps forward by ORBIT_SPEED every single frame, with no user input at all yet.",
          why: "An automatic, ever-so-slow orbit turns a single static render into something that reads immediately as 'a real 3D object you can walk around' — motion sells volume the way a single still frame never quite can, especially for something as boundary-less as a cloud.",
          see: "The camera slowly circles the cloud on its own, revealing its lumpy, irregular 3D silhouette from every angle — a real object floating in the sky, orbitable, still flatly lit.",
          checkpoint: "An auto-orbiting 3D cloud. Beat 2.",
          recovery: ["camera_position returns a plain 3-tuple, unpacked at the call site as cx, cy, cz = camera_position(theta) — render still wants three separate float arguments, not a vector."] }
      ]
    },
    {
      id: 3, title: "Light finds its way through",
      build: "a light march — a second, shorter ray toward the sun — for real volumetric shading.",
      beat: "Bright, sunlit tops; soft, shadowed undersides. A real cloud.",
      steps: [
        { title: "Ask the sun's side of the story", adding: "light-march dials and the self-shadowing function.",
          code: `LIGHT_STEPS = 6
LIGHT_STEP_SIZE = 0.06
SUN_INTENSITY = 1.3
@ti.func
def march_light(p, sun):
    lt = 1.0
    for k in range(LIGHT_STEPS):
        lp = p + sun * LIGHT_STEP_SIZE * (k + 1)
        ld = sample_density(lp)
        lt *= ti.exp(-ld * LIGHT_STEP_SIZE * ABSORPTION)
    return lt`,
          does: "From any sample point INSIDE the cloud, march a SECOND, much shorter ray — this time toward the sun, not the camera — asking 'how much cloud is between here and the light source?' Each step along that light ray absorbs light the same Beer's-law way the main ray does, so lt ends up near 1.0 (nothing blocking the sun) or near 0.0 (buried deep in cloud, in its own shadow).",
          why: "This is the entire secret behind volumetric clouds looking three-dimensional instead of flat: a point near the SUNWARD surface has almost nothing blocking it (bright), while a point deep inside or on the far side has a long way to march through dense cloud first (dark) — self-shadowing, computed from first principles, no baked lightmaps or hacks.",
          see: "Runs clean; render doesn't call this yet.",
          checkpoint: "No red text.",
          recovery: ["march_light reuses sample_density — the SAME field, the SAME interpolation — just walked in a different direction. One density field, two different rays reading it for two different purposes."] },
        { title: "Let the sun in", adding: "sun intensity, the final render, and its wiring.",
          code: `@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32, sunx: ti.f32, suny: ti.f32, sunz: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz])
    center = ti.Vector([CENTER[0], CENTER[1], CENTER[2]])
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (center - cam).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward)
        rd = (forward + u * right + v * up).normalized()

        t_enter, t_exit = ray_box(cam, rd)
        t_enter = ti.max(t_enter, 0.0)

        transmittance = 1.0
        color = ti.Vector([0.0, 0.0, 0.0])
        if t_exit > t_enter:
            step_size = (t_exit - t_enter) / STEPS
            t = t_enter
            for _s in range(STEPS):
                p = cam + rd * t
                d = sample_density(p)
                if d > 0.001:
                    lt = march_light(p, sun)
                    luminance = AMBIENT + lt * SUN_INTENSITY
                    alpha = 1.0 - ti.exp(-d * step_size * ABSORPTION)
                    color += transmittance * alpha * luminance * ti.Vector([1.0, 0.97, 0.92])
                    transmittance *= 1.0 - alpha
                    if transmittance < 0.01:
                        break
                t += step_size

        sky_t = ti.max(rd[1], 0.0)
        sky = ti.Vector([0.5, 0.65, 0.9]) * (1.0 - sky_t) + ti.Vector([0.2, 0.35, 0.7]) * sky_t
        pixels[i, j] = ti.math.clamp(color + transmittance * sky, 0.0, 1.0)
    sun = (0.5, 0.5, 0.2)
        render(cx, cy, cz, *sun)`,
          does: "One change to the per-step lighting: luminance is no longer just AMBIENT, it's AMBIENT + lt * SUN_INTENSITY — every sample point now asks march_light how exposed IT specifically is, and brightens accordingly. Everything else — the primary march, Beer's law, the early exit — is untouched.",
          why: "This single luminance formula is doing real physical work for very little code: a fully-shadowed point (lt near 0) still gets its ambient floor, so shadows read as dim, not pitch black — and a fully-exposed point (lt near 1) gets the full sun on top of that. That gradient, computed per-sample, is what turns a flat white blob into something with volume, form, and a visible light direction.",
          see: "The cloud transforms: sunlit tops and edges glow warm and bright, undersides and interior pockets fall into soft blue-gray shadow — a real, dimensional, believably lit cloud, orbiting slowly against a gradient sky.",
          checkpoint: "Fully volumetric-shaded clouds. Beat 3.",
          recovery: ["luminance = AMBIENT + lt * SUN_INTENSITY, not lt * (AMBIENT + SUN_INTENSITY) — ambient light reaches every point regardless of sun visibility; only the SUN's contribution is gated by lt."] }
      ]
    },
    {
      id: 4, title: "Make it yours",
      build: "mouse-drag orbit control, a reset key, and a HUD.",
      beat: "A cloud you can personally spin around and regenerate on demand.",
      steps: [
        { title: "Take the wheel", adding: "drag state and mouse-controlled orbit, replacing the automatic-only version.",
          code: `    pmx = None
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED`,
          does: "Holding the mouse down steers theta directly from horizontal drag distance (scaled 4x for a responsive feel); letting go falls back to the same automatic creep from chapter 2. pmx (previous mouse x) tracks frame-to-frame delta, reset to None the instant the button releases so the very next drag doesn't inherit a stale delta.",
          why: "This exact pmx-tracks-last-frame pattern is the same one MPM's stir(), soft body's grab, and cloth's drag interaction all used — a small, recurring idiom for 'turn a continuous mouse gesture into a per-frame delta' that shows up any time a project wants hands-on control instead of (or in addition to) automatic motion.",
          see: "Drag left and right across the canvas — the cloud swings around to follow, staying obedient to your hand; let go, and it drifts back into its own slow automatic spin.",
          checkpoint: "Mouse-driven orbit. No red text.",
          recovery: ["pmx = None (both the initial declaration AND the reset in the else branch) is what prevents a first-frame-of-a-new-drag jump — without it, the FIRST drag frame would compute a delta against last time's stale, possibly very different, mouse position."] },
        { title: "A fresh sky anytime", adding: "the reset key.",
          code: `        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == "r":
                apply_seed(rng_seed=np.random.randint(1_000_000))`,
          does: "R regenerates the entire density field from a new random seed — a completely different cloud, same physics.",
          why: "Because fbm3d and the vertical/coverage shaping are fully general (they don't hardcode any particular cloud's shape), every reseed produces a genuinely different-looking formation — sparse wisps some rolls, a dense unbroken deck others — for free, from the same few lines.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["Same reseed idiom as every project since 01: rng_seed=np.random.randint(1_000_000)."] },
        { title: "Read the controls", adding: "the HUD text.",
          code: `        gui.text("drag to orbit  [r] new clouds", (0.02, 0.98), color=0xFFFFFF)`,
          does: "One line, the last piece of polish every project in this curriculum ends on.",
          why: "That's Arc 3 opened and closed in one project: procedural generation (fbm, promoted to 3D), a genuinely new representation (a density FIELD instead of particles or a mesh), and a genuinely new rendering technique (ray marching with a light march) — built from pieces (bilerp's logic, project 05's fbm, the mouse-drag idiom) this curriculum had already taught, recombined into something none of those projects could do alone.",
          see: "Drag to spin a living, breathing cloud formation from any angle; tap R for an entirely new sky. No two clouds the same, and none of them a single triangle.",
          checkpoint: "A fully interactive volumetric cloud renderer. Final beat — project 12 complete.",
          recovery: ["gui.text goes between gui.set_image(pixels) and gui.show() — HUD drawn after the render, on top of it, same order as every project before this one."] }
      ]
    }
  ]
};

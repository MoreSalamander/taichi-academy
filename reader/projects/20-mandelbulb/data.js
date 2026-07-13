// Project SOT (prose). Code fields must match lessons/fragments.py verbatim —
// verified by tools/check_lessons.py. Cumulative code lives in generated fulls.js.
window.ACADEMY_SOT = window.ACADEMY_SOT || {};
window.ACADEMY_SOT["20-mandelbulb"] = {
  project: "20-mandelbulb",
  title: "Mandelbulb Explorer",
  pitch: "A shape with infinite detail and no mesh — rendered by an oracle that only ever answers 'you're at least THIS far away.'",
  tier: "hard",
  language: "Python",
  file: "mandelbulb.py",
  chapters: [
    {
      id: 1, title: "The distance oracle",
      build: "the power-8 distance estimator, and a picture of the distance field itself — before any camera exists.",
      beat: "A glowing 2D slice through the fractal's distance field: the oracle, visualized.",
      steps: [
        { title: "A shape you can't mesh", adding: "the docstring and imports.",
          code: `"""Mandelbulb: a 3D fractal you can't mesh — ray-marched by asking 'how far is it, at least?'"""
import math
import numpy as np
import taichi as ti`,
          does: "The mandelbulb is the 3D cousin of the Mandelbrot set: iterate z -> z^8 + c in three dimensions and keep the points that never escape. Its surface has structure at EVERY magnification — no triangle mesh can hold it, no scanline renderer can draw it. The only practical way in is a DISTANCE ESTIMATOR: a function that, for any point in space, returns a guaranteed lower bound on the distance to the fractal. Never the exact distance — just 'at least this far.'",
          why: "That one guarantee is enough to render anything: if the set is at least d away, a ray can safely leap d forward without passing through it. Repeat until the bound gets tiny (you've arrived) or you've flown past everything. This is SPHERE TRACING, the technique behind the entire demoscene/shadertoy school of impossible geometry — and this project builds it from bare math.",
          see: "Runs clean.",
          checkpoint: "python3 mandelbulb.py returns silently.",
          recovery: ["Usual venv setup. Note there is NO numpy seeding in this project — the mandelbulb is one fixed, eternal object; the only thing that ever changes is where you stand."] },
        { title: "The iteration dials", adding: "fractal constants and the lone pixel field.",
          code: `RES = 400
POWER = 8.0
DE_ITERS = 12
BAILOUT = 2.0
pixels = None
def init_sim(arch=None):
    """Start Taichi and allocate every field once (Metal can't free fields)."""
    global pixels
    if arch is None:
        try:
            ti.init(arch=ti.gpu)
        except Exception:
            ti.init(arch=ti.cpu)
    else:
        ti.init(arch=arch)
    pixels = ti.Vector.field(3, ti.f32, shape=(RES, RES))`,
          does: "POWER=8 is the classic mandelbulb exponent (the value that made the 2009 renders famous). DE_ITERS caps how many times the iteration runs per distance query; BAILOUT is the escape radius — once the iterated point flies past it, it's never coming back. And the entire simulation state is... one pixel field. No particles, no grids, no volumes.",
          why: "This is the leanest state of any project in the curriculum, and that's the lesson: the shape isn't STORED anywhere. It exists only as the behavior of a function, evaluated fresh wherever a ray asks. Implicit geometry — the same idea as project 12's density field, taken to its logical extreme where even the field is replaced by pure computation.",
          see: "Runs clean.",
          checkpoint: "No red text.",
          recovery: ["12 iterations is plenty: each iteration raises the radius to the 8th power, so escape happens FAST — the bound converges in single digits for most points."] },
        { title: "See the oracle itself", adding: "the distance estimator and a flat slice through its field.",
          code: `@ti.func
def bulb_de(p):
    z = p
    dr = 1.0
    r = ti.max(z.norm(), 1e-9)
    for _ in range(DE_ITERS):
        if r < BAILOUT:
            theta = ti.acos(z[2] / r) * POWER
            phi = ti.atan2(z[1], z[0]) * POWER
            zr = ti.pow(r, POWER)
            dr = ti.pow(r, POWER - 1.0) * POWER * dr + 1.0
            z = zr * ti.Vector([ti.sin(theta) * ti.cos(phi), ti.sin(theta) * ti.sin(phi), ti.cos(theta)]) + p
            r = ti.max(z.norm(), 1e-9)
    return 0.5 * ti.log(r) * r / dr
@ti.kernel
def render_slice():
    for i, j in pixels:
        x = (i / RES - 0.5) * 3.0
        y = (j / RES - 0.5) * 3.0
        d = bulb_de(ti.Vector([x, y, 0.0]))
        v = ti.math.clamp(d * 0.8, 0.0, 1.0)
        c = ti.Vector([v, v, v])
        if d < 0.01:
            c = ti.Vector([0.9, 0.5, 1.0])
        pixels[i, j] = c
def main():
    init_sim()
    gui = ti.GUI("Mandelbulb — taichi-academy", res=RES, background_color=0x000000)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        render_slice()
        gui.set_image(pixels)
        gui.show()
if __name__ == "__main__":
    main()`,
          does: "bulb_de is the mandelbulb's whole recipe: convert the running point z to spherical coordinates (r, theta, phi), raise r to the 8th power and multiply both angles by 8 — that IS 'z to the 8th' in 3D — convert back, add the original point p, repeat. dr tracks the running derivative alongside, and the final formula 0.5·log(r)·r/dr converts 'how fast did it escape' into 'how far away is the set, at least.' render_slice ignores 3D entirely: it paints the z=0 plane's distance values as brightness, with points effectively ON the set (d < 0.01) in violet.",
          why: "Two things worth staring at. One: the loop uses a GUARD (if r < BAILOUT) instead of a break — a real fix from this project's development, because Taichi forbids break in a kernel's outermost parallel loop, and a bare scalar test-probe kernel makes the DE's loop exactly that. Guards are break-proof by construction. Two: the slice image is the honest introduction to the whole technique — you're looking at the oracle's answers directly: dark near the set, bright far from it, with the fractal's silhouette emerging as the black valley floor.",
          see: "A luminous gray field with a dark eight-lobed flower at its center — the fractal's z=0 cross-section, drawn purely as distance, its edge glowing violet where the oracle answers 'you have arrived.'",
          checkpoint: "The distance field, visualized. Beat 1.",
          recovery: ["Both r = ti.max(z.norm(), 1e-9) guards matter — acos(z/r) and log(r) both detonate at r = 0, and the origin is a point someone will eventually query.", "The 0.5·log(r)·r/dr formula is the standard escape-time distance bound — worth accepting on faith today and deriving the day you fall down the fractal rabbit hole."] }
      ]
    },
    {
      id: 2, title: "March until you arrive",
      build: "sphere tracing, normals from the gradient, then free ambient occlusion — the full 3D render, staged.",
      beat: "A silhouette, then a moon, then the full crawling-detail mandelbulb.",
      steps: [
        { title: "Leap by what the oracle allows", adding: "march dials, the orbit camera, and the sphere tracer (flat white hits, replacing the slice).",
          code: `MAX_STEPS = 128
EPS_BASE = 0.0004
MAX_DIST = 4.0
ZOOM = 1.6
CAM_HEIGHT_RATIO = 0.36
ORBIT_SPEED = 0.004
SUN = (0.5, -0.4, 0.75)
@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            col = ti.Vector([0.9, 0.9, 0.9])
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)
def camera_position(theta, radius):
    return (
        radius * math.sin(theta),
        -radius * math.cos(theta),
        radius * CAM_HEIGHT_RATIO,
    )
    theta = 0.7
    radius = 2.2
        theta += ORBIT_SPEED
        cx, cy, cz = camera_position(theta, radius)
        eps = EPS_BASE * radius
        render(cx, cy, cz, *SUN, eps)`,
          does: "The camera scaffold is projects 12/13's (a basis, a ray per pixel — note z is 'up' this time, matching the bulb's axis). The march itself is three lines of consequence: ask the oracle (d = bulb_de(p)), arrive if the answer is tiny (d < eps), otherwise LEAP EXACTLY d forward — the maximum step the guarantee permits. Compare project 12's marcher, which plodded through the volume in fixed steps; this one takes hundred-length strides through empty space and tiptoes automatically near the surface.",
          why: "That adaptive stride is why sphere tracing is fast enough for infinite detail: step size IS the distance bound, so effort concentrates precisely where the geometry is. The break inside this loop is legal (the pixel struct-for is the outermost loop here) — the same statement chapter 1's bulb_de couldn't use, a distinction worth having felt once.",
          see: "A stark white silhouette of the mandelbulb against near-black — unmistakably 3D in outline, utterly flat in surface, slowly orbiting.",
          checkpoint: "A white paper cutout of the bulb, orbiting. No red text.",
          recovery: ["eps arrives as an argument (EPS_BASE * radius) — adaptive precision that chapter 3's zoom will cash in.", "Flat white is intentional this step — resist fixing it; the next two steps ARE the fix, staged."] },
        { title: "Normals from nothing", adding: "the gradient normal and diffuse light (replace render's hit branch).",
          code: `@ti.func
def normal_at(p, e):
    dx = bulb_de(p + ti.Vector([e, 0.0, 0.0])) - bulb_de(p - ti.Vector([e, 0.0, 0.0]))
    dy = bulb_de(p + ti.Vector([0.0, e, 0.0])) - bulb_de(p - ti.Vector([0.0, e, 0.0]))
    dz = bulb_de(p + ti.Vector([0.0, 0.0, e])) - bulb_de(p - ti.Vector([0.0, 0.0, e]))
    return ti.Vector([dx, dy, dz]).normalized()
@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            p = cam + rd * t
            n = normal_at(p, eps)
            diffuse = ti.max(n.dot(sun), 0.0)
            col = ti.Vector([0.85, 0.8, 0.9]) * (0.15 + 0.85 * diffuse)
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)`,
          does: "A surface normal is the gradient of the distance field — 'which way does distance increase fastest' points straight off the surface — and central differences (six extra oracle calls) estimate it numerically. Then the oldest equation in the curriculum: normal dot light, the same line as project 05's hillshade, project 13's planet, project 12's sun.",
          why: "No mesh means no stored normals — yet the DE contains them implicitly, recoverable at any point by differentiation. Everything a renderer usually precomputes, an implicit surface can derive on demand. That trade — storage for computation — is the entire philosophy of this school of rendering.",
          see: "The cutout becomes a MOON: lit from the upper side, every great lobe and equatorial ridge shaded into three-dimensionality. Still smooth-looking — the fine detail is there, just not yet legible.",
          checkpoint: "A diffusely lit bulb. No red text.",
          recovery: ["normal_at's probe distance reuses eps — probing much finer than the hit tolerance just measures noise."] },
        { title: "Free ambient occlusion", adding: "step-count AO and the normal-tinted palette (replace render's hit branch).",
          code: `@ti.kernel
def render(camx: ti.f32, camy: ti.f32, camz: ti.f32,
           sunx: ti.f32, suny: ti.f32, sunz: ti.f32, eps: ti.f32):
    cam = ti.Vector([camx, camy, camz])
    sun = ti.Vector([sunx, suny, sunz]).normalized()
    for i, j in pixels:
        u = (i / RES - 0.5) * 2.0
        v = (j / RES - 0.5) * 2.0
        forward = (-cam).normalized()
        right = forward.cross(ti.Vector([0.0, 0.0, 1.0])).normalized()
        up = right.cross(forward)
        rd = (forward * ZOOM + u * right + v * up).normalized()

        t = 0.0
        steps = 0
        hit = 0
        for s in range(MAX_STEPS):
            p = cam + rd * t
            d = bulb_de(p)
            if d < eps:
                hit = 1
                steps = s
                break
            t += d
            steps = s
            if t > MAX_DIST:
                break

        col = ti.Vector([0.01, 0.01, 0.03])
        if hit == 1:
            p = cam + rd * t
            n = normal_at(p, eps)
            diffuse = ti.max(n.dot(sun), 0.0)
            ao = 1.0 - steps / float(MAX_STEPS)
            base = ti.Vector([0.85, 0.6, 0.95]) * 0.5 + 0.5 * ti.Vector([ti.abs(n[0]), ti.abs(n[1]), ti.abs(n[2])])
            col = base * (0.15 + 0.85 * diffuse) * ao
        pixels[i, j] = ti.math.clamp(col, 0.0, 1.0)`,
          does: "Two finishing moves, both nearly free. AO: a ray that reached the surface in FEW steps flew through open space; one that needed MANY steps was creeping through a crevice, the distance bound shrinking around it — so 1 - steps/MAX_STEPS darkens exactly the nooks real ambient light can't reach, a shadow term the march computed as a side effect. The palette tints by |normal|, so differently-oriented surfaces take different hues.",
          why: "Step-count AO is the most famous free lunch in ray marching: genuinely useful global-illumination-flavored shading from a counter you were already incrementing. It's also what makes fractal detail READ — without the crevice darkening, the fine structure blends into mush; with it, every fold and floret pops.",
          see: "The moon becomes the mandelbulb of the posters: violet-green-rose surface crawling with self-similar florets, every crevice deepening into shadow, slowly rotating.",
          checkpoint: "The full fractal render. Beat 2.",
          recovery: ["steps must be captured INSIDE the march loop — it's the AO signal, not just debris.", "If the whole image is dim, check ao multiplies the LIT color, not the light direction."] }
      ]
    },
    {
      id: 3, title: "Lean closer",
      build: "zoom with adaptive precision, drag-to-orbit, and the HUD — an explorer, not a screensaver.",
      beat: "Zoom toward the surface and watch new florets resolve out of what looked like texture.",
      steps: [
        { title: "Zoom that sharpens as it closes", adding: "radius limits and the zoom keys.",
          code: `RADIUS_MIN = 1.35
RADIUS_MAX = 4.0
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key in (ti.GUI.UP, "w"):
                radius = max(radius * 0.94, RADIUS_MIN)
            elif e.key in (ti.GUI.DOWN, "s"):
                radius = min(radius / 0.94, RADIUS_MAX)`,
          does: "W or up-arrow multiplies the orbit radius by 0.94 (zooming is multiplicative — each press covers 6% of the REMAINING distance, so it never slams into the surface); S backs out. And the eps = EPS_BASE * radius line from chapter 2 quietly becomes the star: closer camera, finer hit tolerance, MORE resolved detail. The fractal has structure at every scale; adaptive precision is what lets you actually see it.",
          why: "RADIUS_MIN = 1.35 records this project's funniest bug: the first draft allowed zooming to radius 0.15 — INSIDE the bulb (its surface reaches to about 1.2) — where every ray 'hit' at step zero and the screen rendered a perfectly flat, featureless gray wall. An infinite-detail object, displaying nothing. The floor keeps the camera just outside the surface, where the zoom actually means something. Know where your object IS before you fly at it.",
          see: "Hold W: the surface swells toward you, and what read as noise resolves into ranks of florets — each wearing smaller florets, which wear smaller ones still. Back out with S and the whole bulb reassembles.",
          checkpoint: "Working zoom with adaptive detail. No red text.",
          recovery: ["Both key spellings (arrow and letter) are checked with `in` against a tuple — ti.GUI key constants and character keys mix freely."] },
        { title: "Take the wheel", adding: "drag-to-orbit and the HUD.",
          code: `    pmx = None
        if gui.is_pressed(ti.GUI.LMB):
            mx, _my = gui.get_cursor_pos()
            if pmx is not None:
                theta -= (mx - pmx) * 4.0
            pmx = mx
        else:
            pmx = None
            theta += ORBIT_SPEED
        gui.text(f"radius {radius:.2f}", (0.02, 0.98), color=0xFFFFFF)
        gui.text("drag to orbit  [w/up] zoom in  [s/down] zoom out", (0.02, 0.94), color=0xAAAAAA)`,
          does: "The sixth and final appearance of the pmx drag idiom, plus a radius readout so you know how deep you've flown.",
          why: "That's Arc 4, complete — and with it, twenty projects. Chaos rendered as light (19) and infinity rendered as a surface (20), both built from nothing but iterated arithmetic and the rendering vocabulary the first eighteen projects assembled. Arc 5 changes species entirely: ants, traffic, evolution — systems that LEARN.",
          see: "Drag to any face of the bulb, lean in with W until florets fill the screen, and consider: every pixel you're looking at was computed fresh this frame, from a twelve-iteration loop, and there is no bottom — only the float32 floor.",
          checkpoint: "A full fractal explorer. Final beat — project 20 and Arc 4 complete.",
          recovery: ["The HUD's radius readout is your depth gauge — at 1.35 you're skimming the surface; at 4.0 the whole object floats in view."] }
      ]
    }
  ]
};

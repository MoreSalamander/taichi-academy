# taichi-academy — series roadmap

One repo, ~30 projects, each taught as hand-typed lessons. Order is a skills ladder:
every project reuses muscles built by the ones before it. Status: ✅ available · 🔨 in
progress · ⬜ planned.

## Arc 1 — GPU kernels & grids

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 01 | `01-reaction-diffusion` | Gray-Scott chemicals paint coral, mitosis, worms | easy | ✅ |
| 02 | `02-fluid` | Real-time smoke/dye fluid — advection + pressure solve | medium | ✅ |
| 03 | `03-fire` | Volumetric fire & smoke — buoyancy, temperature color | medium | ⬜ |
| 04 | `04-lightning` | Branching bolts — charge propagation + glow | easy-med | ⬜ |
| 05 | `05-terrain-erosion` | Rainfall carves rivers and canyons into noise terrain | medium | ⬜ |

## Arc 2 — particles at scale

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 06 | `06-particle-life-3d` | Millions of particles, species rules, emergent ecology in 3D | medium | ⬜ |
| 07 | `07-particle-painting` | Paint with fire, water, smoke, sparks | easy-med | ⬜ |
| 08 | `08-mpm-snow-sand` | Material Point Method — avalanches, dunes, crumbling | hard | ⬜ |
| 09 | `09-soft-body` | Jelly, rubber, balloons — deformable bodies | medium | ⬜ |
| 10 | `10-cloth-rope` | Flags, capes, nets — constraint physics | medium | ⬜ |
| 11 | `11-tornado` | Rotating vortex, flying debris, pressure gradients | medium | ⬜ |

## Arc 3 — procedural worlds

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 12 | `12-volumetric-clouds` | 3D noise + ray marching + dynamic light | hard | ⬜ |
| 13 | `13-planet-generator` | Continents, oceans, atmosphere, ice caps | hard | ⬜ |
| 14 | `14-galaxy-creator` | Spirals, ellipticals, nebulae, star clusters | medium | ⬜ |
| 15 | `15-star-nursery` | Stars ignite inside collapsing molecular clouds | hard | ⬜ |
| 16 | `16-solar-system` | Accurate orbits, planet formation, comets | medium | ⬜ |
| 17 | `17-plate-tectonics` | Continental drift, mountains, quakes, volcanoes | hard | ⬜ |
| 18 | `18-ocean-currents` | Global currents, temperature, salinity, storms | hard | ⬜ |

## Arc 4 — mathematical art

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 19 | `19-strange-attractors` | Lorenz, Clifford, Thomas, Aizawa in motion | easy | ⬜ |
| 20 | `20-mandelbulb` | Ray-marched 3D fractal with infinite zoom | hard | ⬜ |

## Arc 5 — AI & emergence

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 21 | `21-ant-colony` | Pheromone trails, foraging, adaptation | medium | ⬜ |
| 22 | `22-traffic` | AI drivers, lights, congestion, routing | medium | ⬜ |
| 23 | `23-evolution` | Neural creatures, mutation, predators vs prey | hard | ⬜ |
| 24 | `24-artificial-life` | Millions of organisms eat, reproduce, mutate | hard | ⬜ |
| 25 | `25-molecular-dynamics` | Atoms, bonds, heat, crystal formation | medium | ⬜ |

## Arc 6 — game tech

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 26 | `26-destruction` | Buildings fracture under explosions and quakes | hard | ⬜ |
| 27 | `27-voxel-sandbox` | Destructible voxel world — water, lava, fire, sand | hard | ⬜ |

## Arc 7 — dream projects (capstones)

| # | Project | Pitch | Tier | Status |
|---|---------|-------|------|--------|
| 28 | `28-digital-brain` | Millions of neurons firing with plasticity | hard | ⬜ |
| 29 | `29-earth-simulator` | Atmosphere + oceans + rivers + life + cities | epic | ⬜ |
| 30 | `30-universe-sandbox` | Galaxies, black holes, stellar evolution, civilizations | epic | ⬜ |

## The pipeline (every project repeats it)

1. **Reference impl** in `projects/<id>/reference/` — verified working + tests green.
2. **Fragments** — decompose into `(chapter, step)` pieces in `lessons/fragments.py`;
   `tools/build_fulls.py --project <id>` must pass.
3. **Prose** — author `reader/projects/<id>/data.js`; `tools/check_lessons.py` must pass.
4. **Ship** — flip the card in `reader/manifest.js` to `available`.

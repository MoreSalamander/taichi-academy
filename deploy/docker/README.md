# taichi-academy, in a container

Run project 01 (Gray-Scott reaction-diffusion) yourself, no local Taichi/Python setup needed.

## Run it

```bash
docker build -t taichi-academy:local -f deploy/docker/Dockerfile .
docker run --rm taichi-academy:local
```

Expect something like:
```
OK — 200 steps of 'coral', v in [0.000, 0.941], pattern alive
```

## A note on Apple Silicon

Taichi ships no Linux arm64 wheel (only manylinux x86_64) as of 1.7.4, so this image is pinned to
`linux/amd64` in the Dockerfile itself. On an Apple Silicon Mac this runs under Docker Desktop's
x86_64 emulation — slower than native, but confirmed correct.

## What this ships, and what it doesn't

This is the **headless/CPU path** only — `scripts/verify_headless.py`, which forces
`ti.init(arch=ti.cpu)` and runs 200 real simulation steps, asserting the field stays finite and the
pattern is actually alive. It's a genuine, correct run of the algorithm, just not the live rendered
window.

The **live GUI window** (`python projects/01-reaction-diffusion/reference/gray_scott.py`, no args)
needs a real display — Taichi's GGUI wants Vulkan or Metal, which a plain container doesn't have.
`init_sim()` already falls back from `ti.gpu` to `ti.cpu` on any exception, but that only changes
the compute backend — it doesn't give the container a window to draw into. Getting the live window
running in Docker would need Xvfb plus a software Vulkan/OpenGL stack, a much heavier image than
"run this to see it work." Run the reference script directly on your own machine (with a display)
for the live version.

## Run the full test suite instead

```bash
docker run --rm taichi-academy:local pytest projects/01-reaction-diffusion/tests -v
```

## The guided-lesson experience isn't containerized

`SESSION_PROTOCOL.md`'s hand-typing-with-Claude-Code workflow (checkpointed via
`tools/check_step.py`) is a human+Claude session over the checked-out repo, not a service Docker
can package — this image only ships the verified reference simulation and its tests.

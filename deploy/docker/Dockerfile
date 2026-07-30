# taichi-academy, packaged so a visitor with Docker can run project 01
# (Gray-Scott reaction-diffusion) themselves — `docker build && docker run`.
#
# Delivers the HEADLESS/CPU path only, not the live GUI window. gray_scott.py's
# own init_sim() tries `ti.gpu` and falls back to `ti.cpu` on any exception,
# but Taichi's GGUI window still needs a real display (Vulkan/Metal) that a
# plain container doesn't have — opening it here would either fail or need
# Xvfb + a software Vulkan stack, a materially heavier ask than "run this."
# scripts/verify_headless.py and the project's pytest suite already force
# arch=ti.cpu deliberately for exactly this CI/headless case — that's the
# real, honest experience this image ships: 200 steps of a real simulation,
# proven correct, no window required. See deploy/docker/README.md.
#
# Pinned to linux/amd64: Taichi ships no Linux arm64 wheel (only manylinux
# x86_64) as of 1.7.4, so this runs under emulation on Apple Silicon hosts —
# slower, but correct (confirmed: a full build + run works via Docker
# Desktop's Rosetta emulation). Without this pin, `pip install` fails outright
# on an arm64 host with "no matching distribution found for taichi".
FROM --platform=linux/amd64 python:3.11-slim

# Taichi's compiled core links against X11/GL even for the pure-CPU backend
# used here — it fails to IMPORT at all without these, not just fails to open
# a window. python:3.11-slim has none of them by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libx11-6 libxrandr2 libxi6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[dev]"

CMD ["python", "projects/01-reaction-diffusion/scripts/verify_headless.py"]

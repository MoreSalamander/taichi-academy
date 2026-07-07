# CLAUDE.md — taichi-academy

A lesson-driven series of ~30 Taichi GPU-simulation projects, each taught as chapters of
hand-typed steps. Two delivery modes share one source of truth per project:
a static build-it-style reader (`reader/`) and guided Claude sessions
(`curriculum/SESSION_PROTOCOL.md` — read it before teaching).

## Non-negotiable environment facts

- **Use `python3.11`**, not the default `python3`. `.python-version`=3.11. One shared venv at
  `.venv/`; `pip install -e ".[dev]"`. All projects share it; heavy per-project deps are extras.
- **Taichi GPU = Metal** on this M4 Pro: `ti.init(arch=ti.gpu)` → `Arch.metal`. Reference apps
  fall back to `ti.cpu` if GPU init fails. **All tests/scripts use `ti.cpu`** (headless, CI-safe),
  via an autouse pytest fixture that re-`ti.init()`s per test.
- **Metal can't `destroy_snode_tree`.** Allocate fields **once at capacity**; reseed by
  re-uploading into the same fields. To genuinely resize: `ti.init()` again. Never use
  `FieldsBuilder.destroy()` — it throws on Metal.
- If a project mixes Taichi with another GLFW-bundling lib (e.g. Open3D), **import the other
  lib first** — the ObjC runtime binds GLFW classes to whichever loads first (see universe-forge).

## Layout

- `projects/<id>/reference/` — the verified teaching target (write this FIRST, before lessons).
- `projects/<id>/lessons/fragments.py` — code SOT: versioned fragments keyed by `(chapter, step)`.
- `projects/<id>/tests/` + `scripts/verify_headless.py` — ti.cpu, no window.
- `projects/<id>/my_build/` — the learner's hand-typed workspace. **Gitignored. Never write
  learner files for them; never paste whole solutions during sessions.**
- `reader/projects/<id>/data.js` — prose SOT (chapters/steps); `fulls.js` is GENERATED.
- `tools/` — `build_fulls.py` (fragments → fulls.js, py_compile per step, final must equal the
  normalized reference), `render_step.py`, `check_step.py` (deterministic checkpoint verdict),
  `check_lessons.py` (data.js ↔ fragments anti-drift).
- `curriculum/ROADMAP.md` — the full ordered series.

## Hard rules

- **Reference first.** A project's reference impl must run and its tests pass before any
  fragments or prose are authored.
- **Never edit `fulls.js` by hand** — regenerate via `python tools/build_fulls.py --project <id>`.
- **`data.js` carries no `full` field** — the reader reads generated fulls exclusively.
- **Checkpoints are deterministic.** `check_step.py` decides pass/fail; Claude (or the reader's
  LLM helper) only voices the verdict. Same philosophy as build-it.
- Every chapter must end runnable-and-visible: chapter-end renders are smoke-run on Metal.

## Verification

- `pytest projects/<id>/tests` — invariants on ti.cpu.
- `python projects/<id>/scripts/verify_headless.py` — no window sanity.
- `python tools/build_fulls.py --project <id>` — every step compiles; final == reference.
- `python tools/check_lessons.py --project <id>` — prose/code seam intact.
- `cd reader && python3 -m http.server` — click through the reader.

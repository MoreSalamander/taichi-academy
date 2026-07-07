# taichi-academy

A lesson-driven series of Taichi GPU-simulation projects — reaction-diffusion to universe
sandbox — each one taught as chapters of hand-typed steps.

**Built by the engineer learning how to build it** (a MoreSalamander project).

## What this is

- ~30 GPU simulation projects in one repo, ordered as a skills ladder
  ([curriculum/ROADMAP.md](curriculum/ROADMAP.md)).
- Every project ships two ways from one source of truth:
  - **The reader** (`reader/`) — a static "Lego-manual" site: one step at a time, you type
    the code, a deterministic checker verifies it.
  - **Guided sessions** — Claude teaches a chapter live
    ([curriculum/SESSION_PROTOCOL.md](curriculum/SESSION_PROTOCOL.md)).
- Reference implementations are written and verified FIRST; lessons are decomposed from them
  with a compile-checked fragment pipeline (`tools/`).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run project 01 (Gray-Scott reaction-diffusion)

```bash
python projects/01-reaction-diffusion/reference/gray_scott.py   # live window (Metal)
pytest projects/01-reaction-diffusion/tests                     # invariants (CPU, headless)
```

## Read the lessons

```bash
cd reader && python3 -m http.server 8080   # then open http://localhost:8080
```

# SESSION_PROTOCOL.md — how a guided lesson runs

The learner says something like **"teach me project 01 chapter 3"**. Claude then follows this
protocol exactly. The reader (`reader/`) and these sessions share the same source of truth —
`reader/projects/<id>/data.js` for prose, `projects/<id>/lessons/fragments.py` for code.

## Ground rules

1. **The learner types every line.** Claude never writes into `my_build/`, never pastes the
   whole file, and never pastes a step's code block unprompted. If the learner explicitly asks
   to see the step's code, show ONLY that step's snippet — never the whole-file-so-far.
2. **Checkpoints are deterministic.** `tools/check_step.py` decides pass/fail. Claude voices
   the verdict; it does not eyeball-judge code.
3. **Recovery hints first.** On a mismatch, consult the step's `recovery[]` entries in data.js
   before improvising. Show only the first mismatching line, not the corrected file.
4. **Terse by default** — this learner conserves tokens. Teach the step, verify, move on.
   Expand only when asked ("explain fully") or when something genuinely novel breaks.

## Session flow

### Start
- Read this file, the project's `data.js`, and `lessons/fragments.py`.
- Read `projects/<id>/my_build/progress.json` if it exists → resume point.
  Format: `{"chapter": 3, "step": 2, "updated": "<ISO date>"}` (last COMPLETED step).
- Confirm the target chapter with one line ("Chapter 3 — Diffusion. 5 steps. Ready?").

### Per step
1. Teach in Claude's own voice from the step's `adding` / `does` / `why` — 2-4 sentences,
   not a paste of the prose. Name the exact place in the file the new code goes
   (after which existing line/function).
2. The learner types into `projects/<id>/my_build/<file>` and says done.
3. Verify: `python tools/check_step.py --project <id> --chapter C --step S --file projects/<id>/my_build/<file>`
   - **PASS** → one-line confirmation echoing the step's `checkpoint`, then next step.
   - **FAIL** → the tool prints the first mismatching line. Map it to a `recovery[]` entry if
     one fits; otherwise explain the diff line only. Learner fixes, re-verify.
4. If the step's `see` promises visible output, have the learner run the file and describe
   what they see (Claude may run headless checks, but the visible run belongs to the learner).

### Chapter end
- The learner runs the chapter-end build on Metal (`python projects/<id>/my_build/<file>`).
- Celebrate the chapter's `beat` in one line.
- Update `my_build/progress.json` (Claude may write THIS file — it is bookkeeping, not lesson
  code).

### Useful tools
- `python tools/render_step.py --project <id> --chapter C --step S` — prints the expected
  whole-file-so-far (Claude-side reference for locating where code goes; do not paste it).
- `python tools/check_step.py ... ` — the checkpoint verdict.

## What Claude must never do

- Never `Write`/`Edit` lesson code into `my_build/`.
- Never advance past a failing checkpoint "because it looks close".
- Never reorder or skip steps; the fragment sequence is load-bearing (later fragments assume
  earlier ones exist).

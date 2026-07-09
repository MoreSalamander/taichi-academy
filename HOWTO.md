# How to use taichi-academy

Two ways to work through a project. Pick either, or mix — they share the same source of
truth, so switching mid-project is safe.

- **The reader** — a browser page you click through solo, one step at a time.
- **Guided sessions** — you ask Claude to teach a chapter, live, in this kind of conversation.

Both put you in the same place: hand-typing real code into a file that becomes a working
simulation.

## 0. One-time setup

Four commands, once, in a terminal:

```bash
git clone https://github.com/MoreSalamander/taichi-academy.git
cd taichi-academy
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Needs Python 3.11 specifically (not 3.12, not 3.9) — grab it from
[python.org](https://www.python.org/downloads/release/python-3119/) if you don't have it, then
run the five lines above. That's it — no further checks needed; if `pip install` finishes
without red text, you're set up.

The reader (`cd reader && python3 -m http.server 8080`, then `localhost:8080`) has the same
four commands on its "Set up the repo first →" page, with a button that drops you straight
into project 01, chapter 1, step 1.

## 1. Learning via the reader

Open `localhost:8080`, click a project card (only ones marked available are clickable — the
rest are the roadmap, dimmed).

Each step on the page has the same shape:
- **What you're adding** / **the code** — the new lines for this step, shown standalone
- **Your whole file so far** (collapsed by default) — the complete file up to this point, in
  case you lose your place
- **What it does** / **why it matters** — the explanation
- **Run it — what you'll see** — what should happen when you run the file
- **Checkpoint** — the one-sentence "did it work" test
- **It's not right?** (collapsed) — the specific fixes for mistakes THIS step tends to get

**Where you actually type:** create the file the reader names (top of each code block, e.g.
`gray_scott.py`) inside that project's `my_build/` folder —
`projects/01-reaction-diffusion/my_build/gray_scott.py` — and keep adding to it as you go.
`reference/` next door is the finished, correct version; don't open it unless you're well and
truly stuck, since it spoils the next several steps.

Run what you've typed from inside `my_build/`:
```bash
cd projects/01-reaction-diffusion/my_build
python gray_scott.py
```

**The Helper** (bottom of each step, "🛟 Helper — want a check, or stuck?") — paste your code
or an error message. "Check my code" does an instant, deterministic check in your browser
(the "something's not right" LLM version only works if this reader is deployed with an
Anthropic API key behind it — locally it'll just tell you it's offline and fall back to the
step's recovery list, which is usually enough).

**Progress** is saved per-project in your browser automatically. Chapters unlock in order as
you finish the one before; closing the tab and coming back picks up where you left off.

## 2. Learning via guided sessions (with me)

Just say something like:

> teach me project 01 chapter 1

I'll follow [curriculum/SESSION_PROTOCOL.md](curriculum/SESSION_PROTOCOL.md), which boils
down to a few promises:

- **I never type your code for you.** I won't paste a step's code block unless you explicitly
  ask to see it — and even then, only that one step's snippet, never the whole file.
- **I tell you what's changing and why**, in a couple of sentences, and exactly where in the
  file it goes (after which existing line).
- **You type it, then tell me you're done.** I verify with the same deterministic tool the
  reader uses:
  ```bash
  python tools/check_step.py --project 01-reaction-diffusion --chapter 3 --step 2 \
      --file projects/01-reaction-diffusion/my_build/gray_scott.py
  ```
  A pass moves us to the next step; a mismatch prints the exact line that's off, and I'll map
  it to a known fix rather than guess.
- **At the end of a chapter**, you run the file yourself so you see the payoff live, and I
  jot down where we stopped in `my_build/progress.json` so a later session can resume cold.

You can run the check yourself too, any time — you don't need me in the loop for it. It's the
same command above; a `PASS ✅` or a `MISMATCH at line N` with expected-vs-yours is all it
prints.

## 3. A typical session, start to finish

```bash
source .venv/bin/activate                     # once per terminal
mkdir -p projects/01-reaction-diffusion/my_build
```
1. Open the reader to project 01, chapter 1, step 1 (or ask me to teach it).
2. Type the step's code into `my_build/gray_scott.py`.
3. Check it — via the reader's Helper, my `check_step.py` call, or running the command
   yourself.
4. Fix if needed, using the step's recovery list.
5. Run the file when a step promises visible output — that's the fun part.
6. Repeat for every step; at a chapter's last step you get the chapter's "beat" — the payoff
   moment (a window opening, coral growing, a bolt striking).
7. Move to the next chapter, or stop and resume later — nothing is lost.

## Reference

| Command | What it does |
|---|---|
| `cd reader && python3 -m http.server 8080` | Serve the reader locally |
| `python tools/check_step.py --project <id> --chapter C --step S --file <path>` | Deterministic checkpoint |
| `python tools/render_step.py --project <id> --chapter C --step S` | Print the expected whole-file-so-far (for me, not you — no spoilers) |
| `pytest projects/<id>/tests` | Run that project's invariant tests |
| `python projects/<id>/reference/<file>` | Run the finished answer key, if you want to see the destination |

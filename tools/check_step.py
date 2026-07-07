"""Deterministic checkpoint: does the learner's file match the expected state
after a given step? Comparison ignores blank lines and #-comments.

Usage: python tools/check_step.py --project 01-reaction-diffusion --chapter 3 --step 2 \
           --file projects/01-reaction-diffusion/my_build/gray_scott.py

Exit 0 = PASS, exit 1 = mismatch (prints the first differing line).
"""

import argparse
from pathlib import Path

from fragment_lib import load_spec, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--file", required=True, help="the learner's hand-typed file")
    ap.add_argument("--as-file", default=None, help="which project file this is (default: the project's main file)")
    args = ap.parse_args()

    fs = load_spec(args.project)
    fname = args.as_file or fs.default_file
    expected = fs.render((args.chapter, args.step)).get(fname)
    if expected is None:
        raise SystemExit(f"{fname} does not exist yet at ch{args.chapter} step{args.step}")

    learner_path = Path(args.file)
    if not learner_path.exists():
        raise SystemExit(f"FAIL — no file at {learner_path}")

    exp_lines = norm(expected).split("\n")
    got_lines = norm(learner_path.read_text()).split("\n")
    if exp_lines == got_lines:
        print(f"PASS ✅ — {fname} matches ch{args.chapter} step{args.step}")
        return

    for idx in range(max(len(exp_lines), len(got_lines))):
        exp = exp_lines[idx] if idx < len(exp_lines) else "<nothing — file should end here>"
        got = got_lines[idx] if idx < len(got_lines) else "<missing — file ends too early>"
        if exp != got:
            print(f"MISMATCH at (non-blank) line {idx + 1}:")
            print(f"  expected: {exp}")
            print(f"  yours:    {got}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()

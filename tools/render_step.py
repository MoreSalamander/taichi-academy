"""Print the expected whole-file-so-far after a given step (Claude-side session tool).

Usage: python tools/render_step.py --project 01-reaction-diffusion --chapter 3 --step 2 [--file gray_scott.py]
"""

import argparse

from fragment_lib import load_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    fs = load_spec(args.project)
    files = fs.render((args.chapter, args.step))
    fname = args.file or fs.default_file
    if fname not in files:
        raise SystemExit(f"{fname} does not exist yet at ch{args.chapter} step{args.step}")
    print(files[fname], end="")


if __name__ == "__main__":
    main()

"""Compile a project's fragments into reader fulls.

Usage: python tools/build_fulls.py --project 01-reaction-diffusion
"""

import argparse

from fragment_lib import REPO_ROOT, emit_fulls, load_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    args = ap.parse_args()
    fs = load_spec(args.project)
    emit_fulls(fs, REPO_ROOT / "reader" / "projects" / args.project / "fulls.js")


if __name__ == "__main__":
    main()

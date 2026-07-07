"""Anti-drift check: the prose SOT (reader data.js) must match the code SOT (fragments.py).

Asserts, per project:
- chapter count and per-chapter step counts match fragments' chapter_steps
- every step's `code` equals the fragment text(s) introduced at that (chapter, step),
  joined in document order (compared normalized: blank/comment lines ignored)
- no step carries a `full` field (fulls are generated, never authored)

Usage: python tools/check_lessons.py --project 01-reaction-diffusion
"""

import argparse
import json
import subprocess
import sys

from fragment_lib import REPO_ROOT, load_spec, norm


def load_sot(project_id):
    data_js = REPO_ROOT / "reader" / "projects" / project_id / "data.js"
    if not data_js.exists():
        sys.exit(f"no data.js at {data_js}")
    out = subprocess.run(
        ["node", "-e", f"global.window = {{}}; require({json.dumps(str(data_js))}); "
                       f"console.log(JSON.stringify(window.ACADEMY_SOT[{json.dumps(project_id)}]))"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"data.js failed to evaluate:\n{out.stderr}")
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    args = ap.parse_args()

    fs = load_spec(args.project)
    sot = load_sot(args.project)
    problems = []

    chapters = sot.get("chapters", [])
    if len(chapters) != len(fs.chapter_steps):
        problems.append(f"chapter count: data.js has {len(chapters)}, fragments expect {len(fs.chapter_steps)}")

    for ch in chapters:
        c = ch["id"]
        want = fs.chapter_steps.get(c)
        got = len(ch.get("steps", []))
        if want != got:
            problems.append(f"ch{c}: data.js has {got} steps, fragments expect {want}")
            continue
        for s, step in enumerate(ch["steps"], start=1):
            if "full" in step:
                problems.append(f"ch{c} step{s}: carries a 'full' field — fulls are generated only")
            expected = norm("\n".join(fs.new_at((c, s))))
            got_code = norm(step.get("code", ""))
            if expected != got_code:
                problems.append(
                    f"ch{c} step{s} ('{step.get('title', '?')}'): code differs from fragments"
                )

    if problems:
        print("check_lessons: FAIL")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    n = sum(fs.chapter_steps.values())
    print(f"check_lessons: OK — {len(chapters)} chapters / {n} steps, prose matches fragments ✅")


if __name__ == "__main__":
    main()

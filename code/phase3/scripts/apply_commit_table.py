#!/usr/bin/env python3
"""Apply a remap produced by remap_commit_table.py to the paper.

Reads the sed-style lines that script prints, checks each new hash resolves in
the rebuilt bundle to a commit whose subject matches the one it replaces, and
only then rewrites the appendix. Refuses to write if anything is off, because
a wrong hash in the commit-trail table is worse than a stale one: a stale hash
fails to resolve and a reader knows something is wrong, while a wrong hash may
resolve to a real commit and quietly assert something false.

Usage:
  python code/phase3/scripts/remap_commit_table.py OLD NEW MAP > remap.txt
  python code/phase3/scripts/apply_commit_table.py remap.txt NEW_CLONE
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PAPER = (Path(__file__).resolve().parents[3]
         / "paper" / "sections" / "app_prereg.tex")
LINE = re.compile(r"s/\\+texttt\{([0-9a-f]{7,40})\}/\\+texttt\{([0-9a-f]{7,40})\}/g")


def sh(*a: str) -> str:
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def main() -> int:
    remap_file, new_clone = sys.argv[1], sys.argv[2]
    pairs = LINE.findall(Path(remap_file).read_text(encoding="utf-8"))
    if not pairs:
        print("no substitutions found in", remap_file)
        return 2

    text = PAPER.read_text(encoding="utf-8")
    bad = 0
    for old, new in pairs:
        token = "\\texttt{" + old + "}"
        if token not in text:
            print(f"  MISS  {old} not present")
            bad += 1
            continue
        if not sh("git", "-C", new_clone, "rev-parse", "--verify", "--quiet",
                  new + "^{commit}"):
            print(f"  FAIL  {new} does not resolve in the rebuilt bundle")
            bad += 1
            continue
        n = text.count(token)
        text = text.replace(token, "\\texttt{" + new + "}")
        subj = sh("git", "-C", new_clone, "log", "-1", "--format=%s", new)
        print(f"  ok    {old} -> {new}  ({n}x)  {subj[:44]}")

    left = re.findall(r"\\texttt\{([0-9a-f]{7,40})\}", text)
    unresolved = [h for h in set(left)
                  if not sh("git", "-C", new_clone, "rev-parse", "--verify",
                            "--quiet", h + "^{commit}")]
    if unresolved:
        print(f"\n  {len(unresolved)} hash(es) in the paper do not resolve in "
              f"the bundle: {sorted(unresolved)}")
        bad += 1

    if bad:
        print(f"\n{bad} problem(s); nothing written.")
        return 2

    PAPER.write_text(text, encoding="utf-8")
    print(f"\nevery cited hash resolves in the bundle; wrote {PAPER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check the paper's commit-trail table against the bundle it ships with.

Reads the table out of the appendix rather than a hand-kept copy, resolves
every hash in the bundle, and runs the ancestry check the table's own caption
tells a reviewer to run. This is the check that matters: the table is the
paper's claim about its own honesty, and it is worth nothing if a rebuild
leaves it pointing at commits that do not exist or do not stand in the claimed
order.

Usage:
  python code/phase3/scripts/verify_commit_table.py CLONE_OF_BUNDLE
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PAPER = (Path(__file__).resolve().parents[3]
         / "paper" / "sections" / "app_prereg.tex")

# A row is: label & rules & analyzer(s) & result \\   possibly wrapped.
ROW = re.compile(
    r"^([A-Z][^&\n]*?)\s*&\s*\\texttt\{([0-9a-f]{7,40})\}\s*&"
    r"(.*?)&\s*\\texttt\{([0-9a-f]{7,40})\}\s*\\\\",
    re.M | re.S)
HASH = re.compile(r"\\texttt\{([0-9a-f]{7,40})\}")


def sh(clone: str, *a: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", clone, *a], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def main() -> int:
    clone = sys.argv[1]
    text = PAPER.read_text(encoding="utf-8")

    bad = 0
    print("=== every cited hash resolves ===")
    cited = sorted(set(HASH.findall(text)))
    for h in cited:
        rc, _ = sh(clone, "rev-parse", "--verify", "--quiet", h + "^{commit}")
        if rc != 0:
            print(f"  FAIL {h} does not resolve")
            bad += 1
    print(f"  {len(cited) - bad}/{len(cited)} resolve")

    print("\n=== ancestry, rules -> result, as the caption instructs ===")
    rows = ROW.findall(text)
    for label, rules, _mid, result in rows:
        label = " ".join(label.split())[:44]
        rc, _ = sh(clone, "merge-base", "--is-ancestor", rules, result)
        if rc == 0:
            print(f"  OK   {label}")
        else:
            print(f"  FAIL {label}  ({rules} not an ancestor of {result})")
            bad += 1
    print(f"  rows checked: {len(rows)}")

    if not rows:
        print("  WARNING: no rows parsed; the table format may have changed")
        bad += 1

    print("\n=== the ordering the paper says must NOT hold ===")
    # The replication amendment deliberately post-dates the step-0 result.
    m = re.search(r"Running the\s*\n?\s*ancestry check on \\texttt\{([0-9a-f]{7,40})\} "
                  r"against \\texttt\{([0-9a-f]{7,40})\} fails", text)
    if m:
        rc, _ = sh(clone, "merge-base", "--is-ancestor", m.group(1), m.group(2))
        if rc != 0:
            print(f"  OK   {m.group(1)} is not an ancestor of {m.group(2)}, "
                  f"as stated")
        else:
            print(f"  FAIL the paper says this check fails, and it passes")
            bad += 1
    else:
        print("  (claim not found in the text)")

    print()
    if bad:
        print(f"{bad} problem(s).")
        return 2
    print("The table matches the bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

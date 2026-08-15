#!/usr/bin/env python3
"""Bridge the paper's commit-trail hashes across a bundle rebuild.

The table cites the ANONYMISED bundle's hashes, not the working repository's,
because anonymisation rewrites every commit. So after a rebuild the old hashes
belong to a hash space that no longer exists anywhere, and they cannot simply
be looked up in the new bundle or in the working repo: they resolve in neither.

What survives the rewrite is the commit MESSAGE. Anonymisation rewrites
identifying strings inside messages but leaves the rest, and in this project
every commit subject is distinct. So the bridge is:

    old bundle hash --(subject)--> working repo commit --(commit-map)--> new hash

This script does that, and refuses to emit a row it cannot resolve uniquely.
A silent mismatch here would put a wrong hash in the one table whose whole
purpose is to be checkable.

Usage:
  python code/phase3/scripts/remap_commit_table.py OLD_CLONE NEW_CLONE COMMIT_MAP
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# filter-repo rewrites commit hashes MENTIONED IN commit messages to their new
# values, so a subject like "generator + PBS (prereg f9d230e precedes)" reads
# differently in each hash space and cannot be matched literally. Mask any
# hex-looking token before comparing.
_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


def norm(subject: str) -> str:
    return _HEX.sub("<hash>", subject.strip())

# The table's rows, as (label, old-bundle hashes in column order).
ROWS = [
    ("Replication, step 0",        ["0d9924b", "175453f"]),
    ("Replication, n=3",           ["d503347", "7edf68a"]),
    ("DARE with TIES",             ["100cd43", "b94331c", "672a72c", "3a0822e"]),
    ("Conditioning and the ridge", ["fa5f3c3", "6e51df3", "de2aa10", "25d7de0"]),
    ("Rate exponent",              ["fa5f3c3", "40eae48", "dc2991a"]),
    ("Solver replication",         ["0c8b9e8", "c597e31", "2a49b3e", "a5e1ea1"]),
    ("Untruncated and gate",       ["0c8b9e8", "fb21c0f", "23c251b"]),
    ("Repaired KnOTS",             ["0c8b9e8", "e8c968a", "30d2c04"]),
    ("Merge matrix at T=3",        ["0c8b9e8", "7a12e2c", "0d35789"]),
]


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def main() -> int:
    old_clone, new_clone, cmap_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # working-repo full hash -> new bundle hash
    cmap: dict[str, str] = {}
    for line in Path(cmap_path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            cmap[parts[0]] = parts[1]

    # subject -> working-repo full hash, from the repository this ran in
    repo = Path(__file__).resolve().parents[3]
    subjects: dict[str, list[str]] = {}
    out = sh("git", "-C", str(repo), "log", "--all", "--format=%H%x00%s")
    for line in out.splitlines():
        if "\x00" not in line:
            continue
        h, s = line.split("\x00", 1)
        subjects.setdefault(norm(s), []).append(h)

    print(f"{'old':<9} {'new':<9}  status  subject")
    print("-" * 78)
    resolved: dict[str, str] = {}
    problems = 0
    seen: set[str] = set()

    for label, hashes in ROWS:
        for old in hashes:
            if old in seen:
                continue
            seen.add(old)
            subj = sh("git", "-C", old_clone, "log", "-1", "--format=%s", old)
            if not subj:
                print(f"{old:<9} {'-':<9}  NO SUBJECT in the old bundle")
                problems += 1
                continue
            cands = subjects.get(norm(subj), [])
            if len(cands) != 1:
                print(f"{old:<9} {'-':<9}  {len(cands)} working-repo matches"
                      f"  {subj[:44]}")
                problems += 1
                continue
            new = cmap.get(cands[0])
            if not new or set(new) == {"0"}:
                print(f"{old:<9} {'-':<9}  not in commit-map  {subj[:44]}")
                problems += 1
                continue
            check = sh("git", "-C", new_clone, "log", "-1", "--format=%s", new)
            if norm(check) != norm(subj):
                print(f"{old:<9} {new[:7]:<9}  SUBJECT MISMATCH after remap")
                problems += 1
                continue
            resolved[old] = new[:7]
            print(f"{old:<9} {new[:7]:<9}  ok      {subj[:44]}")

    print()
    if problems:
        print(f"{problems} hash(es) unresolved. Do NOT regenerate the table.")
        return 2

    print("sed script to rewrite the table in place:")
    for old, new in resolved.items():
        print(f"  s/\\\\texttt{{{old}}}/\\\\texttt{{{new}}}/g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

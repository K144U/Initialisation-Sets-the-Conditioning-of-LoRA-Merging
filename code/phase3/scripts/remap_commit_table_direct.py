#!/usr/bin/env python3
"""Bridge the commit-trail hashes between two anonymised bundles directly.

remap_commit_table.py goes old bundle -> subject -> working repo -> commit-map
-> new bundle, which needs filter-repo's commit-map. That map lives in the
build script's temporary directory and is gone once the build finishes, so
after the fact the three-hop route is unavailable.

Both bundles are anonymised from the same repository by the same rules, so the
commit messages agree and the middle hop is unnecessary:

    old bundle hash --(subject)--> new bundle hash

Same guarantees as the three-hop version. Hex-looking tokens are masked before
comparison, because filter-repo rewrites hashes mentioned inside commit
messages and a subject like "(prereg f9d230e precedes)" reads differently in
each hash space. Any subject that is not unique on both sides is refused
rather than guessed: a wrong hash in this table is worse than a stale one,
since a stale hash fails to resolve and tells the reader something is wrong,
while a wrong one may resolve and quietly assert something false.

Usage:
  python code/phase3/scripts/remap_commit_table_direct.py OLD_CLONE NEW_CLONE > remap.txt
  python code/phase3/scripts/apply_commit_table.py remap.txt NEW_CLONE
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PAPER = (Path(__file__).resolve().parents[3]
         / "paper" / "sections" / "app_prereg.tex")

_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")
_CITED = re.compile(r"\\texttt\{([0-9a-f]{7,40})\}")


def norm(subject: str) -> str:
    return _HEX.sub("<hash>", subject.strip())


def cited_hashes() -> list[str]:
    """Read the hashes out of the paper rather than keeping a second copy."""
    text = PAPER.read_text(encoding="utf-8")
    seen, out = set(), []
    for h in _CITED.findall(text):
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def subjects_of(clone: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    log = sh("git", "-C", clone, "log", "--all", "--format=%H%x1f%s")
    for line in log.splitlines():
        if "\x1f" not in line:
            continue
        h, s = line.split("\x1f", 1)
        out.setdefault(norm(s), []).append(h)
    return out


def main() -> int:
    old_clone, new_clone = sys.argv[1], sys.argv[2]
    new_by_subject = subjects_of(new_clone)

    cited = cited_hashes()
    ok = ambiguous = unresolved = 0
    notes: list[str] = []

    for h in cited:
        subj = sh("git", "-C", old_clone, "log", "-1", "--format=%s", h)
        if not subj:
            notes.append(f"# {h}: not in the old bundle either")
            unresolved += 1
            continue
        cands = new_by_subject.get(norm(subj), [])
        if len(cands) == 1:
            new = cands[0][: len(h)]
            if new == h:
                notes.append(f"# {h}: unchanged")
            else:
                print(f"s/\\\\texttt{{{h}}}/\\\\texttt{{{new}}}/g")
            ok += 1
        elif len(cands) > 1:
            notes.append(f"# {h}: {len(cands)} commits share the subject "
                         f"{subj[:60]!r}, refusing to guess")
            ambiguous += 1
        else:
            notes.append(f"# {h}: no commit in the new bundle has subject "
                         f"{subj[:60]!r}")
            unresolved += 1

    for n in notes:
        print(n, file=sys.stderr)
    print(f"\n# resolved {ok}/{len(cited)}; ambiguous {ambiguous}; "
          f"unresolved {unresolved}", file=sys.stderr)
    return 0 if (ambiguous == 0 and unresolved == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())

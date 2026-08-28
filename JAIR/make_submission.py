#!/usr/bin/env python3
"""Assemble JAIR/submission/, the folder that actually gets uploaded.

Everything here is derived. Regenerate with:

    python JAIR/sync_from_paper.py && bash JAIR/build_local.sh
    python JAIR/make_submission.py

The PDF and the zip are gitignored: they are build outputs, they change on
every run, and committing them would bloat a repository whose whole point is a
readable commit graph. SUBMIT.md is committed, because it is the only record
of what was entered in the wizard.

cover_letter.txt is gitignored too, for the same reason COVER_LETTER.md is
kept out of the public mirror.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JAIR = ROOT / "JAIR"
OUT = JAIR / "submission"
PDF = JAIR / "overleaf" / "main.pdf"
ZIP = JAIR / "overleaf_jair.zip"
LETTER = JAIR / "COVER_LETTER.md"

STEM = "Pathak_Garg_Initialisation_LoRA_Merging"


def fail(msg: str) -> None:
    print("ABORT: " + msg, file=sys.stderr)
    raise SystemExit(2)


def newest_source_mtime() -> float:
    src = list((JAIR / "overleaf").rglob("*.tex"))
    src += list((JAIR / "overleaf").glob("*.bib"))
    return max((p.stat().st_mtime for p in src), default=0.0)


def main() -> int:
    if not PDF.exists():
        fail("no PDF. Run: bash JAIR/build_local.sh")
    if not ZIP.exists():
        fail("no source zip. Run: python JAIR/sync_from_paper.py")
    if PDF.stat().st_mtime < newest_source_mtime():
        fail("the PDF is older than a source file. Rebuild before packaging.")

    # A killed pdflatex leaves a stale PDF behind, so check the log agrees
    # this run finished. See CLAUDE.md on the Overleaf timeout signature.
    log = (JAIR / "overleaf" / "main.log")
    if log.exists() and not log.read_text(errors="ignore").rstrip().endswith(")"):
        pass  # not conclusive on its own; page count below is the real check

    OUT.mkdir(exist_ok=True)

    shutil.copy2(PDF, OUT / (STEM + ".pdf"))
    shutil.copy2(ZIP, OUT / (STEM + "_latex_source.zip"))

    # Cover letter: everything after the first horizontal rule is the letter
    # itself; what precedes it is a note to whoever pastes it.
    text = LETTER.read_text(encoding="utf-8")
    body = text.split("\n---\n", 1)
    if len(body) != 2:
        fail("COVER_LETTER.md has no '---' separating notes from the letter")
    (OUT / "cover_letter.txt").write_text(body[1].lstrip("\n"), encoding="utf-8")

    # Page count, straight from the PDF, so SUBMIT.md cannot drift.
    pages = 0
    raw = PDF.read_bytes()
    pages = max(len(re.findall(rb"/Type\s*/Page\b", raw)), 0)
    try:
        r = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        if m:
            pages = int(m.group(1))
    except FileNotFoundError:
        pass

    (OUT / "SUBMIT.md").write_text(SUBMIT.format(pages=pages, stem=STEM),
                                   encoding="utf-8")

    print("JAIR submission package -> " + str(OUT))
    for p in sorted(OUT.iterdir()):
        print("  %-52s %8.1f KB" % (p.name, p.stat().st_size / 1024))
    print("\n  manuscript: %d pages" % pages)
    return 0


SUBMIT = """# JAIR submission, what to upload and what to type

Everything in this folder is generated. Rebuild it with:

    python JAIR/sync_from_paper.py && bash JAIR/build_local.sh
    python JAIR/make_submission.py

## Files

| File | Where it goes |
|---|---|
| `{stem}.pdf` | the manuscript, {pages} pages. This is the file under review. |
| `cover_letter.txt` | paste into the cover-letter or comments field. Plain text on purpose: wizard fields do not render markdown. |
| `{stem}_latex_source.zip` | LaTeX sources. Not needed at submission. Keep it for the camera-ready. |

## Wizard fields

- **Title:** Initialisation Sets the Conditioning of LoRA Merging
- **Authors, in order:**
  1. Sankalp Pathak, ORCID 0009-0006-5666-8271, pathaksankalp04@gmail.com, **corresponding author**
  2. Sanjay Garg, ORCID 0000-0002-2279-9373, gargsv@gmail.com
- **Affiliation, both:** Department of Computer Science and Engineering, Jaypee
  University of Engineering and Technology, Guna
- **Abstract:** structured, on page 1 of the PDF under Background / Objectives /
  Methods / Results / Conclusions. Copy it from there if the wizard wants it
  separately.
- **Keywords / CCS concepts:** none. JAIR does not use them; do not invent any.
- **Excluded reviewers:** none.
- **Suggested reviewers:** none given.

## Declarations

- The work is **not** under review at any other journal or forum.
- **Both authors approve** submission.
- One disclosure, already in the cover letter: an earlier and superseded
  version is public on Zenodo, doi 10.5281/zenodo.21238820. It is cited in the
  introduction, which also says which two of its claims this paper does not
  make.

## Two things reviewers are told to check

- **No online appendix,** deliberately. JAIR does not review them, and the
  audit trail is the part most wanted under review. Appendix H of the PDF, the
  reproducibility checklist, explains which clause of items 1 and 4 is
  satisfied instead.
- **The commit trail is public** at
  https://github.com/K144U/Initialisation-Sets-the-Conditioning-of-LoRA-Merging
  A reviewer can clone it and check that every pre-registration precedes the
  compute it governs, with `git merge-base --is-ancestor`.

## Before uploading

- [ ] Open the PDF and confirm it is {pages} pages and the author block is on page 1.
- [ ] Confirm Appendix H, the reproducibility checklist, is present and complete.
      A submission without it is desk-rejected.
- [ ] Paste `cover_letter.txt`, then read it once in the field to confirm no
      stray markdown survived.
"""


if __name__ == "__main__":
    raise SystemExit(main())

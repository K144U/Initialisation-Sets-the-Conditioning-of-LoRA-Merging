#!/usr/bin/env python3
"""Assemble the JAIR Overleaf project from paper/ and JAIR/format/.

Same contract as tmlr_submission/sync_from_paper.py: paper/ is the source of
truth and this folder is a derived copy, regenerated whenever paper/ changes.
The TMLR copy drifted from paper/ once already, which is why neither folder is
edited by hand.

Two files are renamed on the way across, because Overleaf picks a project root
by looking for a \\documentclass and prefers the name main.tex:

    main.tex  <-  paper/jair.tex
    body.tex  <-  paper/main.tex

and \\input{main} in the root is rewritten to \\input{body} to match. Without
the rewrite the root would input itself.

Unlike the TMLR sync, nothing here is anonymised. JAIR hides reviewers from
authors and not the other way round, so the real author block and the real
repository URL are what should ship. That does mean the repository the paper
points a reader at has to be public before this is uploaded.

Usage:
  python JAIR/sync_from_paper.py
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
FORMAT = REPO / "JAIR" / "format"
DEST = REPO / "JAIR" / "overleaf"
ZIP = REPO / "JAIR" / "overleaf_jair.zip"

# Shipped verbatim from JAIR's own template download. acmart.cls is carried
# rather than relied on from Overleaf's TeX Live: their tree may hold a
# different acmart revision than the one jair.cls was written against, and a
# class mismatch here is the kind of thing that surfaces as an inscrutable
# error hours before a deadline.
CLASS_FILES = [
    "jair.cls",
    "acmart.cls",
    "acmauthoryear.bbx",
    "acmauthoryear.cbx",
    "acmdatamodel.dbx",
]

# Inputs the JAIR root pulls in directly, outside anything body.tex names.
EXTRA_SECTIONS = ["abstract_jair.tex", "app_checklist.tex"]

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
LSTINPUT_RE = re.compile(r"\\lstinputlisting(?:\[[^]]*\])?\{([^}]+)\}")


def body_sections() -> list[str]:
    """Section files paper/main.tex actually inputs, in order."""
    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        for m in INPUT_RE.finditer(line):
            target = m.group(1)
            if target.startswith("sections/"):
                out.append(target.split("/", 1)[1] + ".tex"
                           if not target.endswith(".tex")
                           else target.split("/", 1)[1])
    return out


def copy_tree(sections: list[str]) -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "sections").mkdir(parents=True)

    (DEST / "main.tex").write_text(
        (PAPER / "jair.tex").read_text(encoding="utf-8")
        .replace("\\input{main}", "\\input{body}"),
        encoding="utf-8",
    )
    shutil.copy2(PAPER / "main.tex", DEST / "body.tex")
    shutil.copy2(PAPER / "preamble.tex", DEST / "preamble.tex")
    shutil.copy2(PAPER / "references.bib", DEST / "references.bib")

    for name in sections + EXTRA_SECTIONS:
        shutil.copy2(PAPER / "sections" / name, DEST / "sections" / name)

    for name in CLASS_FILES:
        src = FORMAT / name
        if not src.exists():
            sys.exit("missing class file: %s" % src)
        shutil.copy2(src, DEST / name)


def copy_referenced_assets() -> list[str]:
    """Copy every figure and listing the copied sources actually reference.

    Copying figures/ wholesale would ship unused ones; naming them by hand
    means a new figure is silently missing. So the sources are scanned.
    """
    copied, missing = [], []
    for tex in sorted(DEST.rglob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        for rx in (GRAPHIC_RE, LSTINPUT_RE):
            for m in rx.finditer(text):
                ref = m.group(1)
                for cand in (ref, ref + ".pdf", ref + ".png", ref + ".jpg"):
                    src = PAPER / cand
                    if src.exists():
                        dst = DEST / cand
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            shutil.copy2(src, dst)
                            copied.append(cand)
                        break
                else:
                    missing.append("%s -> %s" % (tex.name, ref))
    if missing:
        print("  WARNING: referenced but not found in paper/:")
        for m in missing:
            print("    " + m)
    return copied


def check_inputs_resolve() -> list[str]:
    """Every \\input in the copied tree must resolve inside the copied tree."""
    broken = []
    for tex in sorted(DEST.rglob("*.tex")):
        for m in INPUT_RE.finditer(tex.read_text(encoding="utf-8")):
            ref = m.group(1)
            if not ((DEST / ref).exists() or (DEST / (ref + ".tex")).exists()):
                broken.append("%s -> %s" % (tex.name, ref))
    return broken


def make_zip() -> None:
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DEST).as_posix())


def main() -> None:
    sections = body_sections()
    copy_tree(sections)
    figures = copy_referenced_assets()
    broken = check_inputs_resolve()

    print("JAIR Overleaf project -> %s" % DEST)
    print("  %d section files from body.tex" % len(sections))
    print("  %d extra sections (%s)" % (len(EXTRA_SECTIONS),
                                        ", ".join(EXTRA_SECTIONS)))
    print("  %d class/style files" % len(CLASS_FILES))
    print("  %d assets: %s" % (len(figures), ", ".join(figures) or "none"))
    if broken:
        print("  BROKEN \\input references:")
        for b in broken:
            print("    " + b)
        sys.exit(1)
    print("  all \\input references resolve")

    make_zip()
    n = len(zipfile.ZipFile(ZIP).namelist())
    print("  zip: %s (%d files, %.0f KB)"
          % (ZIP.name, n, ZIP.stat().st_size / 1024))


if __name__ == "__main__":
    main()

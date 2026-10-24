#!/usr/bin/env python3
"""Pre-submission checks for the Neurocomputing build.

    python paper/nc/check.py

Checks what the journal's guide asks for and what this project's rules ask
for, from the built PDF and the sources:
  - abstract at most 250 words; 3-5 highlights of at most 85 characters
  - main-text word count (Introduction to Conclusion), for the length target
  - no em or en dash used as sentence punctuation in the prose (a project
    rule; numeric ranges written with -- are fine, and none are used here)
  - every [TO CONFIRM] marker still in the PDF, listed so none ships by accident
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz

NC = Path(__file__).resolve().parent
PAPER = NC.parent
PDF = PAPER / "neurocomputing.pdf"


def strip_tex(s: str) -> str:
    s = re.sub(r"(?m)%.*$", "", s)
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    return s


def main() -> int:
    ok = True
    abstract = strip_tex((NC / "abstract.tex").read_text(encoding="utf-8"))
    n_abs = len(abstract.split())
    print(f"abstract: {n_abs} words (limit 250)")
    ok &= n_abs <= 250

    hl = [l for l in (NC / "highlights.tex").read_text(encoding="utf-8").splitlines()
          if l.strip().startswith("\\item")]
    print(f"highlights: {len(hl)} (3-5)")
    ok &= 3 <= len(hl) <= 5
    for l in hl:
        text = l.strip()[len("\\item"):].strip().replace("\\%", "%")
        flag = "" if len(text) <= 85 else "  <-- over 85"
        ok &= len(text) <= 85
        print(f"  {len(text):3d}  {text}{flag}")

    doc = fitz.open(PDF)
    full = "\n".join(p.get_text() for p in doc)
    a = full.find("1. Introduction")
    b = full.find("CRediT authorship")
    body = full[a:b] if a >= 0 and b > a else ""
    print(f"main text (Introduction..Conclusion, incl. captions and tables): "
          f"{len(body.split())} words")
    print(f"pages: {len(doc)}")

    dash_hits = []
    for f in sorted(NC.glob("*.tex")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("%", 1)[0]
            if "\u2014" in code or "\u2013" in code or re.search(r"[A-Za-z] ?--- ?[A-Za-z]", code):
                dash_hits.append(f"{f.name}:{i}: {line.strip()[:70]}")
    print(f"dash punctuation in prose: {len(dash_hits)}")
    for h in dash_hits:
        print("  " + h)
    ok &= not dash_hits

    todo = " ".join(full.split()).count("[TO CONFIRM]")
    print(f"[TO CONFIRM] markers in the PDF: {todo} (must be 0 before submission)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

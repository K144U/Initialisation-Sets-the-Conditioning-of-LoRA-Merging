#!/usr/bin/env python3
"""Re-sync the standalone submission folder from paper/.

paper/ is the source of truth. This folder is a derived copy that exists so
the user can upload one zip to Overleaf with no configuration, so it has to be
regenerated whenever paper/ changes, and it drifted once already.

Two files are renamed on the way across, because Overleaf picks a project's
root by looking for a class declaration and prefers the name main.tex:

    main.tex  <-  paper/tmlr.tex
    body.tex  <-  paper/main.tex

Everything else keeps its name. Two edits are applied to the copy:

  * \\input{main} becomes \\input{body} in the root, to match the rename;
  * the real repository URL in preamble.tex is replaced with an anonymous
    placeholder, since the uploaded source must leak no identity even though
    no section in the build renders either macro.

Only what the tmlr build actually consumes is copied. The arXiv and legacy
ICLR roots, the appendix sections the current body does not input, and the
unused figures stay in paper/.

Usage:
  python tmlr_submission/sync_from_paper.py
  ./tmlr_submission/make_zips.sh
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
DEST = REPO / "tmlr_submission" / "overleaf"

SECTIONS = [
    "abstract.tex", "intro.tex", "related_work.tex", "setup.tex",
    "theory_brief.tex", "regimes.tex", "method_tests.tex",
    "discussion.tex", "reproducibility.tex", "app_rd.tex", "app_proofs.tex",
    "app_synthetic.tex", "app_geometry.tex", "app_rate_exponent.tex",
    "app_prereg.tex", "app_manifest.tex",
]
FIGURES = ["figure_two_regimes.pdf"]
SUPPORT = ["tmlr.sty", "tmlr.bst", "fancyhdr.sty", "references.bib"]

# Files this script writes itself and must not clobber from paper/.
KEEP = {"README.md", "build.sh", "main.tex", "body.tex", "preamble.tex"}


def title_of(root: Path) -> str:
    m = re.search(r"\\title\{(.*?)\}", root.read_text(encoding="utf-8"), re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def sections_in_build() -> list[str]:
    """What paper/main.tex actually inputs, in order."""
    body = (PAPER / "main.tex").read_text(encoding="utf-8")
    return [f"{m}.tex" for m in
            re.findall(r"\\input\{sections/([A-Za-z0-9_]+)\}", body)]


def main() -> int:
    changed = []

    # SECTIONS has gone stale before. Check it against the root rather than
    # trusting it: a missing entry ships a build with dangling references, and
    # a stale entry crashes this script on a file that no longer exists.
    actual = sections_in_build()
    missing = [s for s in actual if s not in SECTIONS]
    extra = [s for s in SECTIONS if s not in actual]
    if missing or extra:
        print("SECTIONS does not match paper/main.tex:")
        for s in missing:
            print(f"  main.tex inputs {s}, SECTIONS does not list it")
        for s in extra:
            print(f"  SECTIONS lists {s}, main.tex does not input it")
        print("\nFix SECTIONS and re-run. Nothing copied.")
        return 2


    for name in SUPPORT:
        dst = DEST / name
        src = PAPER / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)
            changed.append(name)

    (DEST / "sections").mkdir(parents=True, exist_ok=True)
    for name in SECTIONS:
        src, dst = PAPER / "sections" / name, DEST / "sections" / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)
            changed.append(f"sections/{name}")

    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        src, dst = PAPER / "figures" / name, DEST / "figures" / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)
            changed.append(f"figures/{name}")

    # Appendix D used to \lstinputlisting all ten pre-registrations. It now
    # summarises them and points at the supplementary material, so the build no
    # longer reads paper/prereg/ and shipping it here would put a second,
    # non-authoritative copy in the Overleaf project. The documents travel in
    # the supplementary zip instead, alongside the bundle that dates them.
    stale_pre = DEST / "prereg"
    if stale_pre.exists():
        shutil.rmtree(stale_pre)
        changed.append("prereg/ (removed: no longer inputted)")

    # The registrations live in notes/ and are mirrored into paper/prereg/.
    # That mirror went stale once: a registration was committed to notes/ and
    # never copied across, so the supplementary shipped eleven of twelve while
    # the paper claimed twelve. The bundle had it, which is what matters for
    # the audit trail, but a reader counting files in the supplementary would
    # have found one missing and no way to know it was an oversight rather
    # than a document we chose not to show. Check rather than trust.
    notes_pre = sorted(p.name for p in (REPO / "notes").glob("prereg_*.md"))
    mirror_pre = sorted(p.name for p in (PAPER / "prereg").glob("*.md"))
    if notes_pre != mirror_pre:
        print("paper/prereg/ does not mirror notes/:")
        for n in sorted(set(notes_pre) - set(mirror_pre)):
            print(f"  in notes/ only: {n}")
        for n in sorted(set(mirror_pre) - set(notes_pre)):
            print(f"  in paper/prereg/ only: {n}")
        print("
Fix the mirror and re-run. Nothing copied.")
        return 2

    # They do ship in the supplementary, and that copy is kept in sync here so
    # the two cannot drift.
    sup_pre = REPO / "tmlr_submission" / "supplementary" / "prereg"
    sup_pre.mkdir(parents=True, exist_ok=True)
    for src in sorted((PAPER / "prereg").glob("*.md")):
        dst = sup_pre / src.name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)
            changed.append(f"supplementary/prereg/{src.name}")

    # Root: paper/tmlr.tex -> main.tex, with the body input renamed.
    root = (PAPER / "tmlr.tex").read_text(encoding="utf-8")
    assert "\\input{main}" in root, "tmlr.tex no longer inputs main"
    root = root.replace("\\input{main}", "\\input{body}")
    header = (
        "% ===========================================================\n"
        "% ROOT FILE. Compile this one.\n"
        "%\n"
        "% Generated by tmlr_submission/sync_from_paper.py from paper/.\n"
        "% Do not edit here and expect it to survive: edit paper/ and re-sync.\n"
        "%\n"
        "%     this file  main.tex   <-  paper/tmlr.tex\n"
        "%     body.tex              <-  paper/main.tex\n"
        "% ===========================================================\n"
    )
    new_root = header + root
    if not (DEST / "main.tex").exists() or \
            (DEST / "main.tex").read_text(encoding="utf-8") != new_root:
        (DEST / "main.tex").write_text(new_root, encoding="utf-8")
        changed.append("main.tex")

    # Body: paper/main.tex -> body.tex.
    body = (PAPER / "main.tex").read_text(encoding="utf-8")
    # paper/main.tex explains what it must not contain by naming the control
    # sequences literally. Harmless there; not here. Overleaf picks a
    # project's root by looking for a class declaration, and a literal one in
    # a comment makes this file a candidate root alongside main.tex. Spell it
    # out instead. Without this the rename buys nothing.
    literal = ("% This file is \\input from inside \\begin{document}. It must "
               "not contain\n% \\documentclass, preamble packages, or "
               "\\begin{document}/\\end{document}.")
    spelled = ("% This file is input from inside the document environment. It "
               "must not\n% contain a class declaration, preamble packages, or "
               "the document\n% environment itself. Those are spelled out rather "
               "than written as\n% control sequences because Overleaf picks a "
               "root file by looking for\n% a class declaration, and a literal "
               "one here would make this file a\n% candidate.")
    if literal in body:
        body = body.replace(literal, spelled)
    assert "\\documentclass" not in body, \
        "body.tex still contains a literal class declaration"
    body_header = (
        "% Document body, \\input from main.tex (the root). Called main.tex in\n"
        "% paper/; renamed here so the root can take the name Overleaf looks\n"
        "% for. Generated by tmlr_submission/sync_from_paper.py.\n"
        "%\n"
    )
    new_body = body_header + body
    if not (DEST / "body.tex").exists() or \
            (DEST / "body.tex").read_text(encoding="utf-8") != new_body:
        (DEST / "body.tex").write_text(new_body, encoding="utf-8")
        changed.append("body.tex")

    # Preamble, with the repository URL anonymised.
    pre = (PAPER / "preamble.tex").read_text(encoding="utf-8")
    blk = re.search(
        r"% The one place the release URL is written\..*?"
        r"\\newcommand\{\\repourl\}\{[^\n]*\}\n", pre, re.S)
    assert blk, "repo URL block not found in paper/preamble.tex"
    anon = (
        "% Anonymised in this standalone copy. No section in the TMLR build\n"
        "% uses either macro, so nothing renders from them; they stay defined\n"
        "% so reinstating a section cannot break the build, and stay anonymous\n"
        "% so the uploaded source leaks no identity. Restore the real values\n"
        "% from paper/preamble.tex at camera-ready.\n"
        "\\newcommand{\\repohost}{anonymous.repository.invalid}\n"
        "\\newcommand{\\repourl}{https://\\repohost}\n"
    )
    pre = pre[: blk.start()] + anon + pre[blk.end():]
    if not (DEST / "preamble.tex").exists() or \
            (DEST / "preamble.tex").read_text(encoding="utf-8") != pre:
        (DEST / "preamble.tex").write_text(pre, encoding="utf-8")
        changed.append("preamble.tex")

    title = title_of(PAPER / "tmlr.tex")
    print(f"current title: {title}")
    if changed:
        print(f"\nre-synced {len(changed)} file(s):")
        for c in changed:
            print(f"  {c}")
        print("\nNow run ./tmlr_submission/make_zips.sh and rebuild paper.pdf.")
    else:
        print("\nalready in sync, nothing copied")

    # The title appears in prose in three READMEs; flag rather than rewrite,
    # since their wording is not mechanical.
    # Compare with whitespace collapsed: a README may wrap the title across
    # two lines, which a substring match reads as a stale title forever.
    flat_title = " ".join(title.split())
    stale = []
    for p in (REPO / "tmlr_submission" / "README.md",
              REPO / "tmlr_submission" / "supplementary" / "README.md",
              DEST / "README.md"):
        if not p.exists():
            continue
        if flat_title not in " ".join(p.read_text(encoding="utf-8").split()):
            stale.append(str(p.relative_to(REPO)))
    if stale:
        print("\nREADMEs still carrying an older title:")
        for s in stale:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

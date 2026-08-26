#!/usr/bin/env bash
# Build the JAIR PDF locally, without Overleaf.
#
# Overleaf's free tier gives the whole latexmk run about a minute, and this
# paper needs more than that: the symptom is no PDF, no error, and output.log
# stopping mid-word because the process was killed rather than failing. Nothing
# is wrong with the document, so the fix is simply to compile somewhere with no
# stopwatch.
#
# Compiles JAIR/overleaf/, which is what sync_from_paper.py produces and what
# gets uploaded, rather than paper/. Building the thing you ship is the point:
# if the bundle is missing a file, this fails the same way Overleaf would.
#
# pdfLaTeX, never XeLaTeX: under XeTeX acmart wants libertinusmath-regular.otf
# and dies before the bibliography. biblatex+biber, not BibTeX, because
# jair.cls loads acmart with natbib=false.
#
# Usage:  bash JAIR/build_local.sh [--clean]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/overleaf"
JOB=main

# TeX Live lives on D: because C: had under 7 GB free at install time. Respect
# an existing installation on PATH first, so this keeps working if TeX Live is
# ever moved or installed properly.
if ! command -v pdflatex >/dev/null 2>&1; then
  for d in "/d/texlive/2026/bin/windows" "/d/texlive/2026/bin/win32"; do
    [ -d "$d" ] && PATH="$d:$PATH" && break
  done
fi
export PATH

for c in pdflatex biber; do
  command -v "$c" >/dev/null 2>&1 || {
    echo "missing $c. Is TeX Live installed and on PATH?" >&2; exit 2; }
done

cd "$SRC"

if [ "${1:-}" = "--clean" ]; then
  rm -f "$JOB".{aux,bbl,bcf,blg,log,out,run.xml,toc,fls,fdb_latexmk,synctex.gz}
  echo "cleaned"
fi

# Manual passes rather than latexmk. latexmk decides how many runs to do by
# watching the .aux settle, which is the right default, but it also hides which
# pass a failure came from. These four are what this document actually needs:
# one to lay out and emit citations, biber, then two so that hyperref anchors
# and the \ref-heavy appendix tables both stabilise.
run() {
  echo "--- $1"
  # nonstopmode so a recoverable error does not hang on a prompt. It also means
  # a FAILED build still exits 0, so the exit status is checked below on the
  # PDF and the log, never on pdflatex's own return code.
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$JOB" \
    > /dev/null 2>&1 || true
}

run "pass 1 of 3"
echo "--- biber"
biber --quiet "$JOB" 2>&1 | grep -vE "^$" | head -5 || true
run "pass 2 of 3"
run "pass 3 of 3"

echo
if [ ! -f "$JOB.pdf" ]; then
  echo "NO PDF. Errors:" >&2
  grep -E "^(\./)?[^ ]+:[0-9]+:|^! " "$JOB.log" | head -20 >&2
  exit 1
fi

# A LaTeX error does not stop nonstopmode, and a PDF can be produced from a
# broken run, so report the errors even on success rather than trusting the
# file's existence.
errs="$(grep -cE "^(\./)?[^ ]+:[0-9]+:|^! " "$JOB.log" || true)"
pages="$(grep -oE "Output written on $JOB\.pdf \([0-9]+ page" "$JOB.log" \
         | grep -oE "[0-9]+" | tail -1 || true)"

echo "PDF:    $SRC/$JOB.pdf"
echo "pages:  ${pages:-unknown}"
echo "errors: $errs"
if [ "${errs:-0}" -gt 0 ]; then
  echo
  echo "first errors:"
  grep -E "^(\./)?[^ ]+:[0-9]+:|^! " "$JOB.log" | head -10
fi

echo
echo "undefined references and citations:"
grep -cE "Warning: (Reference|Citation) .* undefined" "$JOB.log" || echo "  0"

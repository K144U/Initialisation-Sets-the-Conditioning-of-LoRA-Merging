#!/usr/bin/env bash
# Build the Neurocomputing short paper: paper/neurocomputing.pdf.
#
#   bash paper/nc/build.sh            # figures + PDF
#   bash paper/nc/build.sh --no-figs  # PDF only
#
# pdfLaTeX + BibTeX (elsarticle-harv, natbib). Runs from paper/, because the
# roots resolve \input and \includegraphics against the working directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="$(dirname "$HERE")"

if ! command -v pdflatex >/dev/null 2>&1; then
  for d in "/d/texlive/2026/bin/windows" "/d/texlive/2026/bin/win32"; do
    [ -d "$d" ] && PATH="$d:$PATH" && break
  done
fi
export PATH

if [ "${1:-}" != "--no-figs" ]; then
  python "$HERE/make_figures.py"
fi

cd "$PAPER"
# The Elsevier CAS class (cas-sc.cls, cas-common.sty) lives in the downloaded
# bundle; put it on the search path. kpathsea on Windows separates with ';',
# and the trailing separator keeps TeX Live's default paths.
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) SEP=";" ;; *) SEP=":" ;; esac
export TEXINPUTS=".${SEP}./nc/NCTemplate/els-cas-templates//${SEP}"
pdflatex -interaction=nonstopmode -halt-on-error neurocomputing >/dev/null || {
  echo "pdflatex failed (pass 1); see paper/neurocomputing.log" >&2
  grep -n "^!" neurocomputing.log | head -20 >&2; exit 1; }
bibtex neurocomputing >/dev/null || { echo "bibtex failed; see neurocomputing.blg" >&2; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error neurocomputing >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error neurocomputing >/dev/null

# Report, from the log rather than the return code.
echo "PDF: $PAPER/neurocomputing.pdf"
printf '  undefined refs/cites: %s\n' "$(grep -cE 'undefined|Citation .* undefined' neurocomputing.log || true)"
printf '  overfull boxes:       %s\n' "$(grep -c 'Overfull' neurocomputing.log || true)"
grep -E "Warning: (Reference|Citation)" neurocomputing.log | sort -u | head -10 || true

#!/usr/bin/env bash
# Build the TMLR submission from this directory.
#
#   ./build.sh          -> out/main.pdf
#
# Uses tectonic if available (self-contained, no TeX install needed),
# otherwise falls back to pdflatex + bibtex.
#
# Everything sits flat here on purpose. tectonic resolves \input against the
# source file's directory while pdflatex resolves against the CWD, so a build
# subdirectory works under one and breaks under the other. Flat means tectonic,
# pdflatex and Overleaf all resolve identically, with no TEXINPUTS juggling.

set -euo pipefail
cd "$(dirname "$0")"

OUT=out
mkdir -p "$OUT"

if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --outdir "$OUT" main.tex
else
  pdflatex -interaction=nonstopmode -output-directory "$OUT" main.tex
  ( cd "$OUT" && BIBINPUTS="..:" BSTINPUTS="..:" bibtex main )
  pdflatex -interaction=nonstopmode -output-directory "$OUT" main.tex
  pdflatex -interaction=nonstopmode -output-directory "$OUT" main.tex
fi

echo "--- $OUT/main.pdf"
# Surface the things that actually bite. Match LaTeX's own warning wording,
# not a bare "undefined": the log always carries a cosmetic "Font shape
# `T1/ptm/m/scit' undefined" line (Times has no small-caps italic), which is
# expected and must not be counted as a broken reference.
if [ -f "$OUT/main.log" ]; then
  echo "    overfull hboxes  : $(grep -c 'Overfull \\hbox' "$OUT/main.log" || true)"
  echo "    undefined refs   : $(grep -c 'Reference.*undefined' "$OUT/main.log" || true)"
  echo "    undefined cites  : $(grep -c 'Citation.*undefined' "$OUT/main.log" || true)"
  echo "    rerun needed     : $(grep -c 'Rerun to get' "$OUT/main.log" || true)"
fi

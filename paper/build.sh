#!/usr/bin/env bash
# Build a paper variant from the single shared source.
#
#   ./build.sh iclr     -> out/iclr2027.pdf   (double-blind submission)
#   ./build.sh arxiv    -> out/arxiv.pdf      (non-anonymous preprint)
#   ./build.sh both
#
# Uses tectonic if available (self-contained, no TeX install needed),
# otherwise falls back to pdflatex + bibtex.
#
# Run from this directory; every path is relative to paper/.
#
# All sources sit flat in paper/ on purpose. tectonic resolves \input
# against the source file's directory while pdflatex resolves against the
# CWD, so a build/ subdirectory works under one and breaks under the other.
# Flat means tectonic, pdflatex and Overleaf all resolve identically, with
# no TEXINPUTS juggling.

set -euo pipefail
cd "$(dirname "$0")"

OUT=out
mkdir -p "$OUT"

build_one () {
  local target="$1"
  echo "=== building $target ==="
  if command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs --outdir "$OUT" "${target}.tex"
  else
    pdflatex -interaction=nonstopmode -output-directory "$OUT" "${target}.tex"
    ( cd "$OUT" && BIBINPUTS="..:" BSTINPUTS="..:" bibtex "${target}" )
    pdflatex -interaction=nonstopmode -output-directory "$OUT" "${target}.tex"
    pdflatex -interaction=nonstopmode -output-directory "$OUT" "${target}.tex"
  fi
  echo "--- $OUT/${target}.pdf"
  # Surface the things that actually bite. Match LaTeX's own warning
  # wording, not a bare "undefined": the log always carries a cosmetic
  # "Font shape `T1/ptm/m/scit' undefined" line (Times has no small-caps
  # italic), which is expected and must not be counted as a broken ref.
  if [ -f "$OUT/${target}.log" ]; then
    echo "    overfull hboxes  : $(grep -c 'Overfull \\hbox' "$OUT/${target}.log" || true)"
    echo "    undefined refs   : $(grep -c 'Reference.*undefined' "$OUT/${target}.log" || true)"
    echo "    undefined cites  : $(grep -c 'Citation.*undefined' "$OUT/${target}.log" || true)"
    echo "    rerun needed     : $(grep -c 'Rerun to get' "$OUT/${target}.log" || true)"
  fi
}

case "${1:-tmlr}" in
  tmlr)  build_one tmlr ;;
  iclr)  build_one iclr2027 ;;
  arxiv) build_one arxiv ;;
  both)  build_one iclr2027; build_one arxiv ;;
  all)   build_one tmlr; build_one arxiv; build_one iclr2027 ;;
  *) echo "usage: $0 {tmlr|arxiv|iclr|both|all}" >&2; exit 2 ;;
esac

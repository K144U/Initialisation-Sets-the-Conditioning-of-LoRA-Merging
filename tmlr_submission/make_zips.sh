#!/usr/bin/env bash
# Regenerate both upload zips from the folders next to this script.
#
#   overleaf/      -> overleaf_tmlr.zip   (upload to Overleaf)
#   supplementary/ -> supplementary.zip   (upload to OpenReview)
#
# The overleaf zip is written FLAT, with no wrapping top-level directory:
# Overleaf creates the project from whatever is at the root of the zip, and a
# wrapper would bury main.tex one level down where its main-file detection does
# not look. Build outputs are excluded so a stale out/main.pdf never ships.

set -euo pipefail
cd "$(dirname "$0")"

rm -f overleaf_tmlr.zip supplementary.zip

python - <<'PY'
import zipfile
from pathlib import Path

SKIP_DIRS = {"out", "_build", "__pycache__", ".git"}
SKIP_SUFFIX = {".aux", ".log", ".blg", ".bbl", ".out", ".synctex.gz", ".fls",
               ".fdb_latexmk", ".toc"}


def pack(src: str, zip_name: str) -> None:
    root = Path(src)
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if SKIP_DIRS & set(rel.parts):
            continue
        if p.suffix in SKIP_SUFFIX:
            continue
        files.append((p, rel))
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in files:
            z.write(p, rel.as_posix())
    size = Path(zip_name).stat().st_size
    print(f"{zip_name}: {len(files)} files, {size/1e6:.2f} MB")
    for _, rel in files:
        print(f"    {rel.as_posix()}")


pack("overleaf", "overleaf_tmlr.zip")
pack("supplementary", "supplementary.zip")
PY

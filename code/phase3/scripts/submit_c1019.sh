#!/bin/bash
# Submit a 2026-10-19 manifest, optionally restricted to some bases, as N jobs.
#
#   bash code/phase3/scripts/submit_c1019.sh <manifest.json> <tag> <n_jobs> [base ...]
#
# With bases given, a filtered copy <manifest>__<tag>.json keeps only cells
# whose name starts with one of them (cells are named <base>__...). Each job is
# one ORCH_SHARD of that manifest, so no cell runs twice; the orchestrator
# skips cells whose done-file exists, so resubmitting is safe.
set -euo pipefail
cd "$HOME/projects/rdmerge_v2"
M="$1"; TAG="$2"; N="$3"; shift 3
SRC="code/phase3/configs/$M"
[ -f "$SRC" ] || { echo "no manifest $SRC"; exit 2; }
if [ "$#" -gt 0 ]; then
  OUT="code/phase3/configs/${M%.json}__${TAG}.json"
  python3 - "$SRC" "$OUT" "$@" <<'EOF'
import json, sys
src, out, bases = sys.argv[1], sys.argv[2], sys.argv[3:]
cells = [c for c in json.load(open(src)) if any(c["name"].startswith(b + "__") for b in bases)]
json.dump(cells, open(out, "w"), indent=1)
print(f"{out}: {len(cells)} cells for {bases}")
EOF
else
  OUT="$SRC"
fi
for ((s=0; s<N; s++)); do
  qsub -N "r_${TAG}${s}" \
    -v MANIFEST="$OUT",ORCH_STATE="state_${TAG}_${s}.json",ORCH_SHARD="$s/$N",ORCH_SENTINEL="_DONE_${TAG}_${s}" \
    -o "$HOME/projects/rdmerge_v2/logs/pbs/" code/phase3/scripts/pbs_v2.sh
done

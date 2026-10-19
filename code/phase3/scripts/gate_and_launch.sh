#!/bin/bash
# Restage gate for one base, then queue its stage-1 cells.
#
#   bash code/phase3/scripts/gate_and_launch.sh <base>
#
# Passes only if the restage cell's worst-task excess and every nll_tau entry
# are within 0.002 nats of the published cell (notes/campaign_2026-10-19_redteam.md).
# On a pass: re-key that base's nll_tau cache, then submit one two-worker job
# with its heterogeneity (Part M), lambda-selection and WUDI cells. On a fail:
# print why and launch nothing.
set -euo pipefail
cd "$HOME/projects/rdmerge_v2"
B="$1"
export PYTHONNOUSERSITE=1 RDMERGE_ROOT=$HOME/projects/rdmerge_v2
P=.conda/envs/rdmerge/bin/python
$P - "$B" <<'EOF'
import json, sys
b = sys.argv[1]
a = json.load(open(f"results/phase3/restage_check/{b}__rmdf_l0p01__seed1.json"))
p = json.load(open(f"results/phase3/eval_regmean_real/{b}__rmdf_l0p01__seed1.json"))
dw = abs(a["worst_task_excess"] - p["worst_task_excess"])
dt = max(abs(a["nll_tau"][s][t] - p["nll_tau"][s][t]) for s in p["nll_tau"] for t in p["nll_tau"][s])
print(f"{b}: worst {a['worst_task_excess']:.5f} vs {p['worst_task_excess']:.5f}, max d nll_tau {dt:.5f}")
sys.exit(0 if dw <= 0.002 and dt <= 0.002 else 1)
EOF
$P code/phase3/scripts/rekey_tau_cache.py "$B" | head -1
python3 - "$B" <<'EOF'
import json, sys
b = sys.argv[1]
d = "code/phase3/configs/"
pick = lambda m: [c for c in json.load(open(d + m)) if c["name"].startswith(b + "__")]
cells = pick("heterogeneity_manifest.json") + pick("lambda_selection_manifest.json") + pick("wudi_manifest.json")
json.dump(cells, open(d + f"stage1_{b}_manifest.json", "w"), indent=1)
print(len(cells), "cells queued for", b)
EOF
WORKERS=2 bash code/phase3/scripts/submit_c1019.sh "stage1_${B}_manifest.json" "s1${B:0:3}" 1

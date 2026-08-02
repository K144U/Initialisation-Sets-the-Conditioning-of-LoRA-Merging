#!/usr/bin/env python3
"""Smoke gate for the review-response campaign.

Hard-won rule #2 (context.md): smoke-first is MANDATORY for any new merge or
eval path, one cheap GPU cell with an explicit PASS/FAIL bar before any batch.
Four paths here are new, so this runs one Llama cell of each and one control.

  ta_alpha0p25   CONTROL. TA at alpha = 0.25 is exactly the published default,
                 so this must reproduce Table 1's TA row (seed1 = 0.2132). If
                 it does not, the harness has drifted and nothing downstream
                 can be trusted. Bar: |excess - 0.2132| < 0.005.
  rd_renorm      NEW kwarg renorm="ta". Bar: runs, and the logged
                 w_over_ta_norm is ~2.4 (the offline reconstruction says
                 ||W*||/||TA|| = 2.44 on Llama at lambda* = 0.05).
  knots_ties     NEW path. Bar: differs from TA by > 0.005 nats. The published
                 knots-linear cell sits 0.00014 from TA.
  rd_ridge_b4    NEW path, finite b with the ridge on and realize=rank_deff.
                 Bar: runs to completion and lands between the b=inf cell
                 (0.094) and the lambda=0 b=4 cell (0.405).
  ta_gsm8k_em    NEW scorer. Bar: extraction-failure rate falls well below the
                 published 0.611 on this cell.

Cells reuse the already-generated configs, so a PASS here is a PASS for the
identical cell in the full manifest; the orchestrator's done-file check then
skips it on the real run rather than repeating it.

Usage:
  python code/phase3/scripts/gen_smoke_review.py
  qsub code/phase3/scripts/pbs_smoke_review.sh
  python code/phase3/scripts/check_smoke_review.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"

# (cell name, config path relative to CFG, runner)
SMOKE = [
    ("llama31_8b__ta_alpha0p25__seed1",
     "eval_w1_alpha/llama31_8b__ta_alpha0p25__seed1.yaml", "run_eval_cell"),
    ("llama31_8b__rd_renorm__seed1",
     "eval_w1_alpha/llama31_8b__rd_renorm__seed1.yaml", "run_eval_cell"),
    ("llama31_8b__knots_ties__seed1",
     "eval_a2_knots_ties/llama31_8b__knots_ties__seed1.yaml", "run_eval_cell"),
    ("llama31_8b__rd_ridge_b4__seed1",
     "eval_w3_rate/llama31_8b__rd_ridge_b4__seed1.yaml", "run_eval_cell"),
    ("llama31_8b__ta__gsm8k_em__seed1",
     "eval_downstream_v2/llama31_8b__ta__gsm8k_em__seed1.yaml",
     "run_downstream_cell"),
]

RUNNER = {"run_eval_cell": "code/phase3/eval/run_eval_cell.py",
          "run_downstream_cell": "code/phase3/eval/run_downstream_cell.py"}


def main() -> int:
    manifest, problems = [], []
    for name, rel, runner in SMOKE:
        p = CFG / rel
        if not p.exists():
            problems.append(f"missing config {p} (run the gen_* scripts first)")
            continue
        cfg = json.loads(json.dumps(__import__("yaml").safe_load(p.read_text())))
        out = cfg.get("output_path")
        if not out:
            problems.append(f"no output_path in {p}")
            continue
        done = Path(out)
        try:
            done_rel = done.relative_to(ROOT).as_posix()
        except ValueError:
            done_rel = done.as_posix()
        manifest.append({
            "name": f"smoke_{name}",
            "cmd": f"python {RUNNER[runner]} --config {p.relative_to(ROOT).as_posix()}",
            "done": done_rel,
            "min_free_gb": 25.0,
        })

    man_p = CFG / "smoke_review_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2))
    print(f"[smoke] {len(manifest)} cells -> {man_p}")
    for m in manifest:
        print(f"[smoke]   {m['name']}")
    if problems:
        print("[smoke] PROBLEMS:")
        for x in problems:
            print("  - " + x)
        return 1
    print("[smoke] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

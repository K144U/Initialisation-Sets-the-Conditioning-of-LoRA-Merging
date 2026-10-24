#!/usr/bin/env python3
"""Cells of notes/check_solve_precision_result_2026-10-24.md: the float64
rebuild of the one merge the B1 rule flagged (Yi seed1, encoder at lambda 0),
plus two unregistered controls.

  rd_l0_f64       registered: float64 solve, exact truncation
  rd_l0_rerun     control: the production configuration unchanged
  rd_l0_f32exact  control: float32 solve, exact truncation

Template: the published cell eval_ridge_cond/yi15_9b__rd_l0__seed1.json.
"""
from __future__ import annotations

import sys

from _c1019 import CACHE, INVARIANT, RES, Manifest, template

OUT_DIR = "eval_precision_f64"
BASE, COHORT = "yi15_9b", "seed1"
ARMS = {
    "rd_l0_f64": {"solve_dtype": "float64", "truncation": "exact"},
    "rd_l0_rerun": {},
    "rd_l0_f32exact": {"truncation": "exact"},
}


def main() -> int:
    m = Manifest()
    t = template(RES / "eval_ridge_cond" / f"{BASE}__rd_l0__{COHORT}.json")
    assert t["method"] == "rd_encoder" and float(t["method_kwargs"]["ridge_lambda"]) == 0.0
    for tag, extra in ARMS.items():
        name = f"{BASE}__{tag}__{COHORT}"
        cfg = {k: v for k, v in t.items() if k != "output_path"}
        cfg.update(method_kwargs={**t["method_kwargs"], **extra},
                   nll_tau_cache=str(CACHE / f"{BASE}__{COHORT}.json"),
                   output_path=str(RES / OUT_DIR / f"{name}.json"))
        for k in INVARIANT:
            assert cfg.get(k) == t.get(k), (name, k)
        m.add(name, cfg, OUT_DIR, "run_eval_cell.py", min_free=40.0)
    m.write("precision_f64_manifest.json", expect=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cells of notes/prereg_lambda_selection_2026-10-19.md: 48 NLL cells.
Stage 2 (accuracy) is generated separately, and only on branch 1."""
from __future__ import annotations

import sys

from _c1019 import (BASES, CACHE, COHORTS, INVARIANT, RES, Manifest, retarget,
                    template)

NLL_DIR = "eval_lambda_sel"
RULES = ["gcv", "lcurve"]


def main() -> int:
    m = Manifest()
    for base in BASES:
        t = template(RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json")
        for cohort in COHORTS:
            specs = retarget(t["adapter_specs"], "seed1", cohort)
            for rule in RULES:
                name = f"{base}__rm_{rule}__{cohort}"
                cfg = {k: v for k, v in t.items() if k != "output_path"}
                cfg.update(method="regmean", method_kwargs={"ridge_mode": rule},
                           adapter_specs=specs,
                           nll_tau_cache=str(CACHE / f"{base}__{cohort}.json"),
                           output_path=str(RES / NLL_DIR / f"{name}.json"))
                for k in INVARIANT:
                    assert cfg.get(k) == t.get(k), (name, k)
                m.add(name, cfg, NLL_DIR, "run_eval_cell.py")
    m.write("lambda_selection_manifest.json", expect=48)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cells of notes/prereg_wudi_2026-10-19.md: 72 NLL + 96 accuracy.

Templates are the published adapter-only RegMean cells (eval_regmean_real and
eval_regmean_real_ds), so loader, eval sets, prompts and seeds are inherited;
only the method, its kwargs, the cohort and the output path change.
"""
from __future__ import annotations

import sys

from _c1019 import (BASES, CACHE, COHORTS, INVARIANT, RES, Manifest, retarget,
                    template)

NLL_DIR, DS_DIR = "eval_wudi", "eval_wudi_ds"
METHODS = {
    "wudi_pub": {"mode": "adam", "iters": 300, "lr": 1e-5},
    "wudi_cf": {"mode": "closed", "omega": 1e-6},
    "wudi_3k": {"mode": "adam", "iters": 3000, "lr": 1e-5},
}
DS_METHODS = ["wudi_pub", "wudi_cf"]
BENCH = ["gsm8k", "humaneval"]


def main() -> int:
    m = Manifest()
    for base in BASES:
        t = template(RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json")
        for cohort in COHORTS:
            specs = retarget(t["adapter_specs"], "seed1", cohort)
            for tag, kw in METHODS.items():
                name = f"{base}__{tag}__{cohort}"
                cfg = {k: v for k, v in t.items() if k != "output_path"}
                cfg.update(method="wudi", method_kwargs=dict(kw), adapter_specs=specs,
                           nll_tau_cache=str(CACHE / f"{base}__{cohort}.json"),
                           output_path=str(RES / NLL_DIR / f"{name}.json"))
                for k in INVARIANT:
                    assert cfg.get(k) == t.get(k), (name, k)
                m.add(name, cfg, NLL_DIR, "run_eval_cell.py")
        for bench in BENCH:
            td = template(RES / "eval_regmean_real_ds" / f"{base}__rmdf_l0p01_{bench}__seed1.json")
            for cohort in COHORTS:
                specs = retarget(td["adapter_specs"], "seed1", cohort)
                for tag in DS_METHODS:
                    name = f"{base}__{tag}_{bench}__{cohort}"
                    cfg = {k: v for k, v in td.items() if k != "output_path"}
                    cfg.update(method="wudi", method_kwargs=dict(METHODS[tag]),
                               adapter_specs=specs,
                               output_path=str(RES / DS_DIR / f"{name}.json"))
                    for k in INVARIANT + ["metric", "n_eval_metric", "metric_kwargs"]:
                        assert cfg.get(k) == td.get(k), (name, k)
                    m.add(name, cfg, DS_DIR, "run_downstream_cell.py")
    # NLL first: cheap, and it decides W1 to W4.
    m.cells.sort(key=lambda c: (DS_DIR + "/" in c["done"], c["name"]))
    m.write("wudi_manifest.json", expect=72 + 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())

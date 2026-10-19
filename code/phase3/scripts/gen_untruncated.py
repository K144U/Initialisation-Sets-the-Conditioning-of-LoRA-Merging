#!/usr/bin/env python3
"""Cells of notes/prereg_untruncated_2026-10-19.md: H2 (72 accuracy) + H3 (40 NLL).

H2 templates: the TOST accuracy cells (cohorts 2-3) and the heur_downstream
cells (cohort 1), with "untruncated": True added; nothing else changes.
H3 templates: the published NLL template, with each heuristic's settings read
from its heur_downstream config.
"""
from __future__ import annotations

import sys

from _c1019 import (BASES, CACHE, INVARIANT, RES, Manifest, load_cfg, restage,
                    retarget, template)

ACC_DIR, NLL_DIR = "eval_untrunc_ds", "eval_untrunc"
PAIRS = [("task_arithmetic", "humaneval"), ("dare", "humaneval"), ("ties", "gsm8k")]
TAGS = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots_ties"]
COH = ["seed1", "seed2", "seed3", "indep1", "indep2", "indep3"]


def acc_template(base, tag, bench, cohort):
    d = "eval_heur_downstream" if cohort in ("seed1", "indep1") else "eval_heur_tost"
    p = RES / d / f"{base}__{tag}_{bench}__{cohort}.json"
    return template(p)


def main() -> int:
    m = Manifest()
    for base in BASES:
        for tag, bench in PAIRS:
            for cohort in COH:
                t = acc_template(base, tag, bench, cohort)
                name = f"{base}__{tag}_{bench}__{cohort}"
                cfg = {k: v for k, v in t.items() if k != "output_path"}
                kw = dict(cfg.get("method_kwargs", {}))
                kw["untruncated"] = True
                cfg.update(method_kwargs=kw, output_path=str(RES / ACC_DIR / f"{name}.json"))
                m.add(name, cfg, ACC_DIR, "run_downstream_cell.py", min_free=40.0)
        tn = template(RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json")
        for tag in TAGS:
            src = load_cfg(RES / "eval_heur_downstream" / f"{base}__{tag}_gsm8k__seed1.json")
            for cohort in ("seed1", "indep1"):
                specs = retarget(tn["adapter_specs"], "seed1", cohort)
                name = f"{base}__{tag}__{cohort}"
                cfg = {k: v for k, v in tn.items() if k != "output_path"}
                kw = dict(src.get("method_kwargs", {}))
                kw["untruncated"] = True
                cfg.update(method=src["method"], method_kwargs=kw, adapter_specs=specs,
                           nll_tau_cache=str(CACHE / f"{base}__{cohort}.json"),
                           output_path=str(RES / NLL_DIR / f"{name}.json"))
                if "weights" in src:
                    cfg["weights"] = src["weights"]
                for k in INVARIANT:
                    assert cfg.get(k) == tn.get(k), (name, k)
                m.add(name, cfg, NLL_DIR, "run_eval_cell.py", min_free=40.0)
    m.cells.sort(key=lambda c: (ACC_DIR + "/" in c["done"], c["name"]))
    m.write("untruncated_manifest.json", expect=72 + 40)
    return 0


if __name__ == "__main__":
    sys.exit(main())

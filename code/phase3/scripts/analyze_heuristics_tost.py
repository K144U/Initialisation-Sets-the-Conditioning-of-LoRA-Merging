#!/usr/bin/env python3
"""Apply notes/prereg_heuristics_tost_2026-09-23.md.

Refuses to report until all 160 new cells exist. Margins, gates and the
sign test are hard-coded from the registration.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from math import comb
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
NEW, OLD = RES / "eval_heur_tost", RES / "eval_heur_downstream"
OUT = RES / "heur_tost_summary.json"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
METHODS = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots_ties"]
BENCH = {"gsm8k": 0.03, "humaneval": 0.05}          # equivalence margins
SHARED, INDEP = ["seed1", "seed2", "seed3"], ["indep1", "indep2", "indep3"]
Z = 1.645


def acc(stem, cohort):
    p = (OLD if cohort in ("seed1", "indep1") else NEW) / f"{stem}__{cohort}.json"
    return json.loads(p.read_text())["metric_score"]


def sign_test(pos, neg):
    n = pos + neg
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    need = [NEW / f"{b}__{m}_{k}__{c}.json" for b in BASES for m in METHODS
            for k in BENCH for c in ("seed2", "seed3", "indep2", "indep3")]
    have = sum(p.exists() for p in need)
    if have < len(need):
        print(f"{have}/{len(need)} cells; refusing to report")
        return 1
    rep, dvals = {"rules": "notes/prereg_heuristics_tost_2026-09-23.md"}, []
    equiv, effects = {}, 0
    for m in METHODS:
        for k, M in BENCH.items():
            ds, ses, per = [], [], {}
            for b in BASES:
                s = [acc(f"{b}__{m}_{k}", c) for c in SHARED]
                i = [acc(f"{b}__{m}_{k}", c) for c in INDEP]
                d = st.mean(i) - st.mean(s)
                se = math.sqrt(st.variance(s) / 3 + st.variance(i) / 3)
                eff = abs(d) > max(0.05, 2 * se)
                effects += eff
                per[b] = {"shared": s, "indep": i, "d": d, "se": se, "effect": eff}
                ds.append(d); ses.append(se); dvals.append(d)
            dbar = st.mean(ds)
            se = math.sqrt(sum(x * x for x in ses)) / 4
            ok = dbar - Z * se > -M and dbar + Z * se < M
            equiv[f"{m}/{k}"] = {"per_base": per, "dbar": dbar, "se": se, "margin": M,
                                 "equivalent": ok,
                                 "smallest_margin": abs(dbar) + Z * se}
    pos = sum(d > 0 for d in dvals)
    neg = sum(d < 0 for d in dvals)
    p = sign_test(pos, neg)
    E1 = sum(equiv[f"{m}/gsm8k"]["equivalent"] for m in METHODS) >= 4
    E2 = sum(equiv[f"{m}/humaneval"]["equivalent"] for m in METHODS) >= 4
    E3 = (40 - effects) >= 38
    E4 = p >= 0.05
    rep.update(equivalence=equiv, n_effect_cells=effects,
               sign={"indep_better": pos, "shared_better": neg, "p": p},
               verdict={"E1": E1, "E2": E2, "E3": E3, "E4": E4})
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["verdict"], indent=1))
    for key, v in equiv.items():
        print(f"{key:<26} dbar {v['dbar']:+.4f}  se {v['se']:.4f}  "
              f"{'EQUIV' if v['equivalent'] else 'not'} at {v['margin']}  "
              f"(smallest margin {v['smallest_margin']:.3f})")
    print(f"sign: indep better {pos}, shared better {neg}, p = {p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

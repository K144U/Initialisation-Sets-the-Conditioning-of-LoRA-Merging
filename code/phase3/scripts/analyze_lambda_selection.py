#!/usr/bin/env python3
"""Verdicts L1 to L4 (and the branch) of notes/prereg_lambda_selection_2026-10-19.md.

Comparators: eval_regmean_real (lambda 0.01 and alpha 0.9, three cohorts per
arm) and eval_regmean_agram (lambda 1e-6 and the 7-point sweep, seed1/indep1).
Refuses partial reads.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys

from _c1019 import BASES, INDEP, RES, SHARED

SEL, REAL, AGRAM = RES / "eval_lambda_sel", RES / "eval_regmean_real", RES / "eval_regmean_agram"
GRID = ["l1em6", "l0p01", "l0p03", "l0p05", "l0p13", "l0p30", "l1p00"]


def w(p):
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["worst_task_excess"]


def lambdas(p):
    d = json.loads(p.read_text()).get("merge_diagnostics") or {}
    return [v["lambda"] for v in d.values() if isinstance(v, dict) and "lambda" in v]


def paired_gate(a, b):
    dc = [x - y for x, y in zip(a, b)]
    return st.mean(dc), max(0.005, 2 * st.stdev(dc) / math.sqrt(len(dc)))


def agram(base, cohort, tag):
    return w(AGRAM / f"{base}__rm_{tag}__{cohort}.json")


def main() -> int:
    out = {"rules": "notes/prereg_lambda_selection_2026-10-19.md", "per_rule": {}}
    try:
        for rule in ["gcv", "lcurve"]:
            per = {}
            for b in BASES:
                rs = {c: w(SEL / f"{b}__rm_{rule}__{c}.json") for c in SHARED + INDEP}
                l01 = {c: w(REAL / f"{b}__rmdf_l0p01__{c}.json") for c in SHARED + INDEP}
                a09 = {c: w(REAL / f"{b}__rmdf_a09__{c}.json") for c in SHARED + INDEP}
                sweep_s = {g: agram(b, "seed1", g) for g in GRID}
                sweep_i = {g: agram(b, "indep1", g) for g in GRID}
                orc_s = min(sweep_s.values())
                d2s, g2s = paired_gate([rs[c] for c in SHARED], [l01[c] for c in SHARED])
                d2i, g2i = paired_gate([rs[c] for c in INDEP], [l01[c] for c in INDEP])
                per[b] = {
                    "rule_shared": [rs[c] for c in SHARED], "rule_indep": [rs[c] for c in INDEP],
                    "L1": rs["seed1"] < sweep_s["l1em6"] - 0.005,
                    "L2": d2s <= g2s,
                    "L3": rs["seed1"] <= orc_s + max(0.005, 0.1 * (sweep_s["l1em6"] - orc_s)),
                    "L4": (rs["indep1"] <= sweep_i["l1em6"] + 0.005) and d2i <= g2i,
                    "vs_alpha09_shared": st.mean(rs[c] - a09[c] for c in SHARED),
                    "oracle_shared": orc_s, "min_ridge_shared": sweep_s["l1em6"],
                    "lambda_median_seed1": st.median(lambdas(SEL / f"{b}__rm_{rule}__seed1.json")),
                    "lambda_median_indep1": st.median(lambdas(SEL / f"{b}__rm_{rule}__indep1.json")),
                }
            v = {k: sum(per[b][k] for b in BASES) >= 3 for k in ["L1", "L2", "L3", "L4"]}
            out["per_rule"][rule] = {"per_base": per, "verdict": v}
    except FileNotFoundError as e:
        print(f"incomplete ({e.args[0]}); refusing to report")
        return 2
    branches = {}
    for rule, r in out["per_rule"].items():
        v = r["verdict"]
        branches[rule] = 1 if all(v.values()) else (2 if v["L1"] and v["L4"] else 3)
    out["branch"] = branches
    (RES / "lambda_selection_summary.json").write_text(json.dumps(out, indent=1))
    for rule, r in out["per_rule"].items():
        print(rule, r["verdict"], "branch", branches[rule])
        for b, x in r["per_base"].items():
            print(f"  {b:11s} seed1 {x['rule_shared'][0]:.4f} (1e-6 {x['min_ridge_shared']:.4f},"
                  f" oracle {x['oracle_shared']:.4f})  lambda med {x['lambda_median_seed1']:.2e}"
                  f" / {x['lambda_median_indep1']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

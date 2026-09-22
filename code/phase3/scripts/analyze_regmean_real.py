#!/usr/bin/env python3
"""Apply notes/prereg_regmean_real_2026-09-23.md to its cells.

Refuses to report a part until every cell that part needs exists: Q1-Q3 and
Q1m need all 72 NLL cells, Q4-Q5 all 144 downstream cells. Thresholds are
hard-coded from the registration.

Usage:
  RDMERGE_ROOT=~/projects/rdmerge_v2 python code/phase3/scripts/analyze_regmean_real.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
NLL, DS = RES / "eval_regmean_real", RES / "eval_regmean_real_ds"
OUT = RES / "regmean_real_summary.json"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SHARED = ["seed1", "seed2", "seed3"]
INDEP = ["indep1", "indep2", "indep3"]
TAGS = ["rma", "rmdf_a09", "rmdf_l0p01"]
BENCH = ["gsm8k", "humaneval"]
NLL_TIE, ACC_TIE = 0.005, 0.05


def se(xs):
    return st.stdev(xs) / math.sqrt(len(xs))


def arm_contrast(s, i, tie):
    d = st.mean(s) - st.mean(i)
    gate = max(tie, 2 * math.sqrt(se(s) ** 2 + se(i) ** 2))
    return {"shared": s, "indep": i, "d": d, "gate": gate, "effect": abs(d) > gate,
            "ratio": st.mean(s) / st.mean(i) if st.mean(i) else float("inf")}


def paired(a, b, tie):
    """a - b paired by cohort."""
    dc = [x - y for x, y in zip(a, b)]
    d = st.mean(dc)
    gate = max(tie, 2 * se(dc))
    return {"a": a, "b": b, "d": d, "gate": gate, "clears": d > gate}


def load(path, key):
    return json.loads(path.read_text())[key]


def main() -> int:
    report = {"rules": "notes/prereg_regmean_real_2026-09-23.md"}
    need_nll = [NLL / f"{b}__{t}__{c}.json" for b in BASES for t in TAGS
                for c in SHARED + INDEP]
    missing = [p.name for p in need_nll if not p.exists()]
    if missing:
        print(f"NLL part: {len(need_nll) - len(missing)}/72 cells; refusing to report.")
        report["nll"] = {"complete": False, "have": len(need_nll) - len(missing)}
    else:
        v = {(b, t, c): load(NLL / f"{b}__{t}__{c}.json", "worst_task_excess")
             for b in BASES for t in TAGS for c in SHARED + INDEP}
        amp = {}
        for b in BASES:
            meds = []
            for c in SHARED:
                diag = json.loads((NLL / f"{b}__rma__{c}.json").read_text()).get(
                    "merge_diagnostics") or {}
                meds.append(st.median(x["amp_pre_trunc"] for x in diag.values()))
            amp[b] = st.median(meds)
        q1 = {b: arm_contrast([v[b, "rma", c] for c in SHARED],
                              [v[b, "rma", c] for c in INDEP], NLL_TIE) for b in BASES}
        q2 = {b: arm_contrast([v[b, "rmdf_a09", c] for c in SHARED],
                              [v[b, "rmdf_a09", c] for c in INDEP], NLL_TIE) for b in BASES}
        q3 = {b: paired([v[b, "rmdf_a09", c] for c in SHARED],
                        [v[b, "rmdf_l0p01", c] for c in SHARED], NLL_TIE) for b in BASES}
        Q1 = (sum(not x["effect"] for x in q1.values()) >= 3
              and all(x["ratio"] < 2 for x in q1.values()))
        Q1m = all(a <= 3 for a in amp.values())
        Q2 = sum(x["effect"] and x["d"] > 0 and x["ratio"] >= 2 for x in q2.values()) >= 3
        Q3 = sum(x["clears"] for x in q3.values()) >= 3
        Q1_fail_hard = sum(x["effect"] and x["d"] > 0 for x in q1.values()) >= 2
        report["nll"] = {"complete": True, "Q1": q1, "Q1m_amp_shared": amp, "Q2": q2,
                         "Q3": q3, "verdict": {"Q1": Q1, "Q1m": Q1m, "Q2": Q2, "Q3": Q3,
                                               "Q1_hurt_on_2plus": Q1_fail_hard}}
        # reproducibility check against the published Table 4 cells (reported only)
        repro = {}
        for b in BASES:
            for c in ("seed1", "indep1"):
                pub = RES / "eval_regmean_cond" / f"{b}__rm_l0p01__{c}.json"
                if pub.exists():
                    repro[f"{b}/{c}"] = {"published": load(pub, "worst_task_excess"),
                                         "rerun": v[b, "rmdf_l0p01", c]}
        report["nll"]["repro_vs_table4"] = repro
        print(json.dumps(report["nll"]["verdict"], indent=1))

    need_ds = [DS / f"{b}__{t}_{k}__{c}.json" for b in BASES for t in TAGS
               for k in BENCH for c in SHARED + INDEP]
    missing = [p.name for p in need_ds if not p.exists()]
    if missing:
        print(f"downstream part: {len(need_ds) - len(missing)}/144 cells; refusing to report.")
        report["downstream"] = {"complete": False, "have": len(need_ds) - len(missing)}
    else:
        a = {(b, t, k, c): load(DS / f"{b}__{t}_{k}__{c}.json", "metric_score")
             for b in BASES for t in TAGS for k in BENCH for c in SHARED + INDEP}
        q4 = {k: {b: paired([a[b, "rmdf_l0p01", k, c] for c in SHARED],
                            [a[b, "rmdf_a09", k, c] for c in SHARED], ACC_TIE)
                  for b in BASES} for k in BENCH}
        q5 = {k: {b: arm_contrast([a[b, "rma", k, c] for c in SHARED],
                                  [a[b, "rma", k, c] for c in INDEP], ACC_TIE)
                  for b in BASES} for k in BENCH}
        Q4 = any(sum(x["clears"] for x in q4[k].values()) >= 3 for k in BENCH)
        Q5 = all(sum(not x["effect"] for x in q5[k].values()) >= 3 for k in BENCH)
        table = {f"{b}/{t}/{k}": {"shared": [a[b, t, k, c] for c in SHARED],
                                  "indep": [a[b, t, k, c] for c in INDEP]}
                 for b in BASES for t in TAGS for k in BENCH}
        report["downstream"] = {"complete": True, "Q4": q4, "Q5": q5, "table": table,
                                "verdict": {"Q4": Q4, "Q5": Q5}}
        print(json.dumps(report["downstream"]["verdict"], indent=1))

    if report["nll"].get("complete") and report["downstream"].get("complete"):
        vn, vd = report["nll"]["verdict"], report["downstream"]["verdict"]
        if vn["Q1_hurt_on_2plus"]:
            branch = 2
        elif not vn["Q2"]:
            branch = 3
        elif not vd["Q4"]:
            branch = 4
        elif vn["Q1"] and vd["Q5"] and vn["Q3"]:
            branch = 1
        else:
            branch = "no single registered branch; report each prediction as it landed"
        report["branch"] = branch
        print("branch:", branch)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

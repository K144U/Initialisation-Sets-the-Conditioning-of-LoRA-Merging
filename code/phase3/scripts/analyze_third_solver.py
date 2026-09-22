#!/usr/bin/env python3
"""Apply notes/prereg_third_solver_2026-09-23.md. No partial reads."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_regmean_real import (ACC_TIE, BASES, INDEP, NLL_TIE, RES, SHARED,  # noqa: E402
                                  arm_contrast, load)

NLL, DS = RES / "eval_third_solver", RES / "eval_third_solver_ds"
OUT = RES / "third_solver_summary.json"
TAGS = ["tsv", "mats", "fisherdiag"]


def binom_gate(p1, p2, n):
    return max(0.05, 2 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n))


def main() -> int:
    rep = {"rules": "notes/prereg_third_solver_2026-09-23.md"}
    need = [NLL / f"{b}__{t}__{c}.json" for b in BASES for t in TAGS for c in SHARED + INDEP]
    have = sum(p.exists() for p in need)
    if have < len(need):
        print(f"NLL part {have}/72; refusing to report")
        rep["nll"] = {"complete": False, "have": have}
    else:
        v = {(b, t, c): load(NLL / f"{b}__{t}__{c}.json", "worst_task_excess")
             for b in BASES for t in TAGS for c in SHARED + INDEP}
        con = {t: {b: arm_contrast([v[b, t, c] for c in SHARED],
                                   [v[b, t, c] for c in INDEP], NLL_TIE) for b in BASES}
               for t in TAGS}
        S1 = sum(x["effect"] and x["d"] > 0 for x in con["tsv"].values()) >= 3

        def immune(t):
            return (sum(not x["effect"] for x in con[t].values()) >= 3
                    and all(x["ratio"] < 2 for x in con[t].values()))
        rep["nll"] = {"complete": True, "contrasts": con,
                      "verdict": {"S1": S1, "S2": immune("mats"), "S3": immune("fisherdiag")}}
        print(json.dumps(rep["nll"]["verdict"]))

    need = ([DS / f"{b}__tsv_{k}__{c}.json" for b in BASES for k in ("gsm8k", "humaneval")
             for c in SHARED + INDEP]
            + [DS / f"{b}__mats_{k}__{c}.json" for b in BASES for k in ("gsm8k", "humaneval")
               for c in ("seed1", "indep1")])
    have = sum(p.exists() for p in need)
    if have < len(need):
        print(f"downstream part {have}/64; refusing to report")
        rep["downstream"] = {"complete": False, "have": have}
    else:
        tsv = {k: {b: arm_contrast([load(DS / f"{b}__tsv_{k}__{c}.json", "metric_score") for c in SHARED],
                                   [load(DS / f"{b}__tsv_{k}__{c}.json", "metric_score") for c in INDEP],
                                   ACC_TIE) for b in BASES} for k in ("gsm8k", "humaneval")}
        S4 = any(sum(x["effect"] and x["d"] < 0 for x in tsv[k].values()) >= 2 for k in tsv)
        mats = {}
        for k, n in (("gsm8k", 500), ("humaneval", 164)):
            for b in BASES:
                s = load(DS / f"{b}__mats_{k}__seed1.json", "metric_score")
                i = load(DS / f"{b}__mats_{k}__indep1.json", "metric_score")
                g = binom_gate(s, i, n)
                mats[f"{b}/{k}"] = {"shared": s, "indep": i, "d": s - i, "gate": g,
                                    "effect": abs(s - i) > g}
        rep["downstream"] = {"complete": True, "tsv": tsv, "mats": mats, "verdict": {"S4": S4}}
        print(json.dumps(rep["downstream"]["verdict"]))
    OUT.write_text(json.dumps(rep, indent=1, default=str))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verdicts of notes/prereg_training_regimes_2026-10-19.md (+ amendment).

Inputs: regimes_geometry.json (measure_regimes.py --cohorts),
regimes_calibration.json, eval_regimes cells, eval_hetero_m indep1 references,
library_defaults.json. Each part is reported only when all its inputs exist.
"""
from __future__ import annotations

import json
import statistics as st
import sys

from _c1019 import RES

SHARED_K = ["s_sast", "k_lr033", "k_lr3", "k_ep1", "k_ep4", "k_amult4", "k_loraplus16", "k_scale025"]
INDEP_INIT = ["s_iast", "ki_lr3", "ki_amult4", "indep1"]


def w(name):
    p = (RES / "eval_regimes" / f"llama31_8b__{name}.json")
    if name.endswith("__indep1"):
        p = RES / "eval_hetero_m" / f"llama31_8b__{name}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["worst_task_excess"]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def main() -> int:
    geo = json.loads((RES / "regimes_geometry.json").read_text())
    cal = json.loads((RES / "regimes_calibration.json").read_text())
    v, out = {}, {"rules": ["notes/prereg_training_regimes_2026-10-19.md",
                            "notes/prereg_training_regimes_amendment_2026-10-19.md"]}
    try:
        s = {c: geo[c] for c in ["s_sast", "s_sait", "s_iast", "indep1"]}
        rm = {c: w(f"rm1em6__{c}") for c in s}
        ta = {c: w(f"ta__{c}") for c in s}
        v["S1"] = s["s_sast"]["median_cos"] >= 0.9 and s["s_sait"]["median_cos"] >= 0.9 \
            and s["s_iast"]["median_cos"] <= 0.5
        v["S2"] = all(s[a]["kappa"] >= 10 * s[b]["kappa"] for a in ("s_sast", "s_sait") for b in ("s_iast", "indep1"))
        v["S3"] = all(rm[a] >= 2 * rm[b] and rm[a] - rm[b] >= 0.01
                      for a in ("s_sast", "s_sait") for b in ("s_iast", "indep1"))
        v["S4"] = max(ta.values()) - min(ta.values()) < 0.01
        out["S"] = {"geometry": s, "rm1em6": rm, "ta": ta}
    except (KeyError, FileNotFoundError) as e:
        print(f"Part S incomplete ({e}); not reported")
    try:
        k = {c: geo[c] for c in SHARED_K}
        ki = {c: geo[c] for c in ("ki_lr3", "ki_amult4")}
        v["K1"] = all(x["median_cos"] >= 0.9 for x in k.values()) and all(x["median_cos"] <= 0.5 for x in ki.values())
        pred = {c: {"kappa_pred": cal["C_k"] / x["t"] ** 2, "amp_pred": cal["C_a"] / x["t"]} for c, x in k.items()}
        kok = sum(0.5 <= k[c]["kappa"] / pred[c]["kappa_pred"] <= 2 for c in k)
        aok = sum(1 / 1.25 <= k[c]["amp"] / pred[c]["amp_pred"] <= 1.25 for c in k)
        v["K2"] = kok >= 6 and aok >= 7
        rho = spearman([k[c]["t"] for c in SHARED_K], [k[c]["kappa"] for c in SHARED_K])
        v["K3"] = rho <= -0.8
        kap_ind = st.median(geo[c]["kappa"] for c in INDEP_INIT)
        t_star = (cal["C_k"] / kap_ind) ** 0.5
        v["K4"] = t_star >= 3 * max(x["t"] for x in k.values())
        v["K5"] = all(w(f"rm1em6__{a}") >= 2 * w(f"rm1em6__{b}")
                      for a, b in (("k_lr3", "ki_lr3"), ("k_amult4", "ki_amult4")))
        out["K"] = {"geometry": k, "indep": ki, "pred": pred, "kappa_within_2x": kok,
                    "amp_within_1.25x": aok, "spearman_t_kappa": rho,
                    "kappa_indep_median": kap_ind, "t_star": t_star}
    except (KeyError, FileNotFoundError) as e:
        print(f"Part K incomplete ({e}); not reported")
    try:
        v["C1"] = geo["c_cont"]["median_cos"] >= 0.9
        v["C2"] = w("rm1em6__c_cont") >= 2 * w("rm1em6__indep1")
        out["C"] = {"geometry": geo["c_cont"], "rm1em6": w("rm1em6__c_cont")}
    except (KeyError, FileNotFoundError) as e:
        print(f"Part C incomplete ({e}); not reported")
    try:
        v["D1"] = geo["d_shared"]["median_cos"] >= 0.9 and geo["d_indep"]["median_cos"] <= 0.5
        out["D"] = {"shared": geo["d_shared"], "indep": geo["d_indep"]}
    except KeyError as e:
        print(f"Part D incomplete ({e}); not reported")
    p = RES / "library_defaults.json"
    if p.exists():
        lib = json.loads(p.read_text())
        u = lib.get("unsloth", {})
        if u:
            v["L1_unsloth"] = u["A_creation"]["gsm8k_vs_alpaca"] >= 0.99
            v["L2_unsloth"] = u["A_step50"]["gsm8k_vs_alpaca"] >= 0.9
        lf = lib.get("llamafactory", {})
        if lf:
            v["L1_llamafactory"] = lf["A_creation"]["gsm8k_vs_alpaca"] >= 0.99
            v["L2_llamafactory"] = lf["A_step50"]["gsm8k_vs_alpaca"] >= 0.9
        out["L"] = lib
    out["verdict"] = v
    (RES / "training_regimes_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

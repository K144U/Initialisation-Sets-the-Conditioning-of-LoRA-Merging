#!/usr/bin/env python3
"""Verdicts of notes/prereg_heterogeneity_2026-10-19.md (G1-G4, M-a to M-c).

Cells: eval_hetero_g, eval_hetero_m. Geometry (median and maximum pairwise
principal cosine, kappa(Hbar)) and the public A-norm spread (G4) are computed
here on CPU. Refuses partial reads.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

import torch
from safetensors.torch import load_file

from _c1019 import BASES, RES, ROOT

G, M = RES / "eval_hetero_g", RES / "eval_hetero_m"
G_BASES = ["llama31_8b", "qwen25_7b"]
G_COH = ["indep1", "indep2", "indep3"]
TASK_DIRS = {"gsm8k": "gsm8k", "alpaca": "alpaca", "magicoder": "magicoder", "translation": "flores"}
MIXES = {"M1": {"gsm8k": "seed1", "magicoder": "seed1", "alpaca": "indep1", "translation": "indep1"},
         "M2": {"gsm8k": "seed1", "magicoder": "seed1", "alpaca": "seed1", "translation": "indep1"},
         "indep1": {k: "indep1" for k in TASK_DIRS}}


def w(p):
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["worst_task_excess"]


def se(xs):
    return st.stdev(xs) / math.sqrt(len(xs))


def a_mats(path):
    sd = load_file(str(path))
    return {k: v.double() for k, v in sd.items() if ".lora_A." in k}


def geometry(base, plan):
    mats = [a_mats(ROOT / "artifacts" / "lora" / base / TASK_DIRS[t] / c / "adapter_model.safetensors")
            for t, c in plan.items()]
    keys = sorted(set.intersection(*[set(m) for m in mats]))
    cos, kap = [], []
    for k in keys:
        Q = [torch.linalg.qr(m[k].T)[0] for m in mats]
        for i in range(len(Q)):
            for j in range(i + 1, len(Q)):
                cos.append((i, j, float(torch.linalg.svdvals(Q[i].T @ Q[j]).clamp(max=1).median())))
        if ".q_proj." in k:
            basis = torch.linalg.qr(torch.cat(Q, 1))[0]
            hb = sum(basis.T @ q @ q.T @ basis for q in Q) / len(Q)
            ev = torch.linalg.eigvalsh(hb)
            kap.append(float(ev[-1] / ev[0]))
    vals = sorted(c for _, _, c in cos)
    pair_med = {}
    for i, j, c in cos:
        pair_med.setdefault((i, j), []).append(c)
    return {"median_cos": vals[len(vals) // 2],
            "max_pair_cos": max(st.median(v) for v in pair_med.values()),
            "kappa_hbar_median": st.median(kap)}


def public_spread():
    os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf_home"))
    from huggingface_hub import hf_hub_download
    rows = json.loads((RES / "public_merge" / "public_merge_summary.json").read_text())["rows"]
    out = []
    for r in rows:
        F = (r.get("F") or {}).get("rm_1em6")
        if F is None:
            continue
        blob = json.loads((RES / "public_merge" / f"{r['cohort']}.json").read_text())
        mats = [a_mats(hf_hub_download(rp, "adapter_model.safetensors")) for rp in blob["adapters"]]
        keys = sorted(set.intersection(*[set(m) for m in mats]))
        spreads = []
        for k in keys:
            n = [float(m[k].norm()) for m in mats]
            if min(n) > 0:
                spreads.append(max(n) / min(n))
        out.append({"cohort": r["cohort"], "F": F, "spread": st.median(spreads) if spreads else None})
    return out


def main() -> int:
    v, out = {}, {"rules": "notes/prereg_heterogeneity_2026-10-19.md"}
    try:
        g = {}
        for b in G_BASES:
            ref_rm = [w(G / f"{b}__rm1em6_ref__{c}.json") for c in G_COH]
            ref_ta = [w(G / f"{b}__ta_ref__{c}.json") for c in G_COH]
            for pert in ["gauge", "scale", "canon", "rank", "module"]:
                rm = [w(G / f"{b}__rm1em6_{pert}__{c}.json") for c in G_COH]
                dc = [x - y for x, y in zip(rm, ref_rm)]
                e = {"rm": rm, "ref": ref_rm, "ratio": st.mean(rm) / st.mean(ref_rm),
                     "d": st.mean(dc), "gate": max(0.005, 2 * se(dc))}
                if pert in ("scale", "rank", "module"):
                    ta = [w(G / f"{b}__ta_{pert}__{c}.json") for c in G_COH]
                    e["ta_ratio"] = st.mean(ta) / st.mean(ref_ta)
                g[(b, pert)] = e
        out["G"] = {f"{b}/{p}": x for (b, p), x in g.items()}
        v["G1"] = all(g[(b, "gauge")]["ratio"] >= 2 and g[(b, "gauge")]["d"] > g[(b, "gauge")]["gate"]
                      for b in G_BASES)
        v["G2"] = all(1 / 1.5 <= g[(b, "canon")]["ratio"] <= 1.5 for b in G_BASES)
        v["G3"] = all(g[(b, p)]["ratio"] <= 2 * g[(b, p)]["ta_ratio"]
                      for b in G_BASES for p in ("scale", "rank", "module"))
        mres = {}
        for b in BASES:
            ind = w(M / f"{b}__rm1em6__indep1.json")
            for mix in ("M1", "M2"):
                geo = geometry(b, MIXES[mix])
                mres[(b, mix)] = {"rm": w(M / f"{b}__rm1em6__{mix}.json"),
                                  "ta": w(M / f"{b}__ta__{mix}.json"), "rm_indep1": ind, **geo}
            mres[(b, "indep1")] = geometry(b, MIXES["indep1"])
        out["M"] = {f"{b}/{k}": x for (b, k), x in mres.items()}
        v["M-a"] = sum(mres[(b, "M1")]["median_cos"] < 0.5
                       and mres[(b, "M1")]["rm"] >= 2 * mres[(b, "M1")]["rm_indep1"]
                       for b in BASES) >= 3
        v["M-b"] = all(mres[(b, k)]["max_pair_cos"] >= 0.9 for b in BASES for k in ("M1", "M2"))
        v["M-c"] = sum(mres[(b, "M1")]["kappa_hbar_median"]
                       >= 10 * mres[(b, "indep1")]["kappa_hbar_median"] for b in BASES) >= 3
    except FileNotFoundError as e:
        print(f"incomplete ({e.args[0]}); refusing to report")
        return 2
    pub = public_spread()
    fail = [x["spread"] for x in pub if x["F"] >= 2 and x["spread"]]
    ok = [x["spread"] for x in pub if x["F"] < 2 and x["spread"]]
    out["G4_rows"] = pub
    v["G4"] = bool(fail and ok and st.median(fail) >= 2 * st.median(ok))
    out["G4_medians"] = {"failing": st.median(fail) if fail else None,
                         "not_failing": st.median(ok) if ok else None,
                         "n": [len(fail), len(ok)]}
    out["verdict"] = v
    (RES / "heterogeneity_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(v, indent=1))
    for k, x in out["G"].items():
        print(f"G {k:22s} ratio {x['ratio']:6.2f} d {x['d']:+.4f} gate {x['gate']:.4f}"
              + (f" ta_ratio {x['ta_ratio']:.2f}" if "ta_ratio" in x else ""))
    for k, x in out["M"].items():
        print(f"M {k:18s} " + " ".join(f"{kk} {vv:.4g}" for kk, vv in x.items()))
    print("G4", out["G4_medians"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

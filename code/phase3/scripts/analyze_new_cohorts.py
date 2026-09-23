#!/usr/bin/env python3
"""Apply notes/prereg_new_cohorts_2026-09-23.md (T1-T5, R1-R5, P1-P2) and
notes/prereg_lambda_rule_2026-09-23.md (V1-V3).

Each part reports only when every file it needs exists. Geometry uses the
functions of drift_theory.py (same layer sample, same definitions).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dt", HERE / "drift_theory.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)

ROOT = Path(os.environ.get("RDMERGE_ROOT", HERE.parents[2]))
RES = ROOT / "results" / "phase3"
NLL, DS = RES / "eval_new_cohorts", RES / "eval_new_cohorts_ds"
NEW = ROOT / "artifacts_new" / "lora"
OLD = Path("/home/sanjay.g/projects/rdmerge/artifacts/lora")
OUT = RES / "new_cohorts_summary.json"
TASK_DIR = {"gsm8k": "gsm8k", "alpaca": "alpaca", "magicoder": "magicoder", "translation": "flores"}
HEUR = ["ta", "ties", "dare", "tvq_b2", "knots_ties"]
NEW_TASKS = ["dolly", "xsum", "codealpaca", "translation_fr"]


def worst(name):
    p = NLL / f"{name}.json"
    return json.loads(p.read_text())["worst_task_excess"] if p.exists() else None


def acc(name):
    p = DS / f"{name}.json"
    return json.loads(p.read_text())["metric_score"] if p.exists() else None


def geometry(dirs: list[Path]) -> dict:
    loaded = [dt.load_ab(d / "adapter_model.safetensors") for d in dirs]
    cos, kr, ar, kap, amp = [], [], [], [], []
    for key in dt.layer_keys(loaded[0][0]):
        As = [A[key] for A, _ in loaded]
        Bs = [B[key] for _, B in loaded]
        V = [dt.orth_rows(A) for A in As]
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                cos.append(float(torch.linalg.svdvals(V[i].T @ V[j]).median()))
        Ds = [B @ A for A, B in zip(As, Bs)]
        ex = dt.exact(As, Ds)
        kap.append(ex["kappa"]); amp.append(ex["amp"])
        pr = dt.predicted(As, Ds, dt.consensus_W0(As, As[0].shape[0]))
        kr.append(pr["kappa_hat"] / ex["kappa"]); ar.append(pr["amp_hat"] / ex["amp"])
    return {"cos": st.median(cos), "kappa": st.median(kap), "amp": st.median(amp),
            "kappa_ratio": st.median(kr), "amp_ratio": st.median(ar), "T": len(dirs)}


def binom_gate(p1, p2, n):
    return max(0.05, 2 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n))


def part_T(rep):
    need = ([f"{b}__{m}__{c}" for b in ("llama31_8b", "mistral_7b") for c in ("seed1fx", "indep1fx")
             for m in ["rm_1em6", "rm_0p01", "rd_l0", "rd_lstar"] + HEUR])
    ds_need = [f"{b}__rd_{w}_{k}__seed1fx" for b in ("llama31_8b", "mistral_7b")
               for w in ("l0", "lstar") for k in ("gsm8k", "humaneval")]
    if any(worst(n) is None for n in need) or any(acc(n) is None for n in ds_need):
        rep["T"] = "incomplete"
        return
    out = {}
    t1 = []
    for b in ("llama31_8b", "mistral_7b"):
        for c in ("seed1fx", "indep1fx"):
            d = json.loads((NLL / f"{b}__ta__{c}.json").read_text())["nll_tau"]
            base_nll = d["base"]["translation"]
            own = d["translation"]["translation"]
            best = min((s for s in d if s != "base"), key=lambda s: d[s]["translation"])
            t1.append({"cohort": f"{b}/{c}", "gain": 1 - own / base_nll, "best": best,
                       "ok": (1 - own / base_nll) >= 0.05 and best == "translation"})
    out["T1"] = {"rows": t1, "holds": all(x["ok"] for x in t1)}
    geo = {f"{b}/{c}": geometry([NEW / b / "flores" / c if t == "translation" else
                                 OLD / b / TASK_DIR[t] / c.replace("fx", "")
                                 for t in TASK_DIR])
           for b in ("llama31_8b", "mistral_7b") for c in ("seed1fx", "indep1fx")}
    out["geometry"] = geo
    out["T2"] = all(geo[f"{b}/seed1fx"]["cos"] > 0.9 and geo[f"{b}/indep1fx"]["cos"] < 0.1
                    for b in ("llama31_8b", "mistral_7b"))
    t3 = {}
    for b in ("llama31_8b", "mistral_7b"):
        s1, i1 = worst(f"{b}__rm_1em6__seed1fx"), worst(f"{b}__rm_1em6__indep1fx")
        gs = s1 - worst(f"{b}__rm_0p01__seed1fx")
        gi = i1 - worst(f"{b}__rm_0p01__indep1fx")
        t3[b] = {"ratio": s1 / i1, "gain_s": gs, "gain_i": gi, "ok": s1 / i1 >= 2 and gs > gi}
    out["T3"] = {"rows": t3, "holds": all(x["ok"] for x in t3.values())}
    t4 = {f"{b}/{m}": abs(worst(f"{b}__{m}__seed1fx") - worst(f"{b}__{m}__indep1fx"))
          for b in ("llama31_8b", "mistral_7b") for m in HEUR}
    out["T4"] = {"rows": t4, "holds": sum(v < 0.02 for v in t4.values()) >= 8}
    t5 = {}
    for b in ("llama31_8b", "mistral_7b"):
        for k, n in (("gsm8k", 500), ("humaneval", 164)):
            a0, a1 = acc(f"{b}__rd_l0_{k}__seed1fx"), acc(f"{b}__rd_lstar_{k}__seed1fx")
            t5[f"{b}/{k}"] = {"l0": a0, "lstar": a1, "gate": binom_gate(a0, a1, n),
                              "clears": a1 - a0 > binom_gate(a0, a1, n)}
    out["T5"] = {"rows": t5, "holds": all(any(t5[f"{b}/{k}"]["clears"] for k in ("gsm8k", "humaneval"))
                                          for b in ("llama31_8b", "mistral_7b"))}
    rep["T"] = out


def part_R(rep):
    b = "llama31_8b"
    cohorts = {"T2": ("_T2", ["gsm8k", "alpaca"]), "T4": ("", list(TASK_DIR)),
               "T8": ("_T8", list(TASK_DIR) + NEW_TASKS), "r8": ("_r8", list(TASK_DIR)),
               "r64": ("_r64", list(TASK_DIR))}

    def dirs(arm, key):
        suffix, tasks = cohorts[key]
        out = []
        for t in tasks:
            if key.startswith("r"):
                out.append(NEW / b / TASK_DIR[t] / f"{arm}{suffix}")
            elif t in NEW_TASKS:
                out.append(NEW / b / t / arm)
            elif t == "translation":
                out.append(NEW / b / "flores" / arm)
            else:
                out.append(OLD / b / TASK_DIR[t] / arm.replace("fx", ""))
        return out

    names = {k: {arm: (f"{arm}{cohorts[k][0]}" if k != "T4" else arm) for arm in ("seed1fx", "indep1fx")}
             for k in cohorts}
    for k in cohorts:
        for arm in ("seed1fx", "indep1fx"):
            if any(not (d / "adapter_model.safetensors").exists() for d in dirs(arm, k)):
                rep["R"] = "incomplete (adapters)"
                return
            for m in ("rm_1em6", "ta"):
                if worst(f"{b}__{m}__{names[k][arm]}") is None:
                    rep["R"] = "incomplete (cells)"
                    return
    geo = {f"{k}/{arm}": geometry(dirs(arm, k)) for k in cohorts for arm in ("seed1fx", "indep1fx")}
    out = {"geometry": geo}
    out["R1"] = all(geo[f"{k}/seed1fx"]["cos"] > 0.9 and geo[f"{k}/indep1fx"]["cos"] < 0.3 for k in cohorts)
    out["R2"] = geo["T8/seed1fx"]["kappa"] > geo["T4/seed1fx"]["kappa"] > geo["T2/seed1fx"]["kappa"]
    out["R3"] = all(0.75 <= geo[f"{k}/seed1fx"][q] <= 1.33 for k in cohorts
                    for q in ("kappa_ratio", "amp_ratio"))
    out["R4"] = all(abs(geo[f"{k}/indep1fx"]["amp"] - geo[f"{k}/indep1fx"]["T"])
                    <= 0.05 * geo[f"{k}/indep1fx"]["T"] for k in ("T2", "T4", "T8"))
    r5 = {k: {"rm_ratio": worst(f"{b}__rm_1em6__{names[k]['seed1fx']}") / worst(f"{b}__rm_1em6__{names[k]['indep1fx']}"),
              "ta_diff": abs(worst(f"{b}__ta__{names[k]['seed1fx']}") - worst(f"{b}__ta__{names[k]['indep1fx']}"))}
          for k in cohorts}
    out["R5"] = {"rows": r5, "holds": sum(v["rm_ratio"] >= 2 for v in r5.values()) >= 4
                 and sum(v["ta_diff"] < 0.02 for v in r5.values()) >= 4}
    rep["R"] = out


def part_P(rep):
    b = "llama31_8b"
    dirs = [NEW / b / TASK_DIR[t] / "pissa_indep" for t in TASK_DIR]
    if any(not (d / "adapter_model.safetensors").exists() for d in dirs):
        rep["P"] = "incomplete"
        return
    geo = geometry(dirs)
    # P2: trained row spaces against the base weight's top-r right singular vectors
    from safetensors import safe_open
    base_dir = Path("/home/sanjay.g/projects/rdmerge/models/Llama-3.1-8B-Instruct")
    idx = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    A0 = dt.load_ab(dirs[0] / "adapter_model.safetensors")[0]
    cos = []
    for key in dt.layer_keys(A0):
        wname = key.split("base_model.model.")[-1].replace(".lora_A.weight", ".weight")
        with safe_open(str(base_dir / idx[wname]), framework="pt") as f:
            W = f.get_tensor(wname).double()
        r = A0[key].shape[0]
        Vw = torch.linalg.svd(W, full_matrices=False)[2][:r].T
        for d in dirs:
            A = dt.load_ab(d / "adapter_model.safetensors")[0][key]
            cos.append(float(torch.linalg.svdvals(dt.orth_rows(A).T @ Vw).median()))
    rep["P"] = {"geometry": geo, "P1": geo["cos"] > 0.9,
                "cos_to_base_top_r": st.median(cos), "P2": st.median(cos) >= 0.9}


def rule(rep):
    shared = [("llama31_8b", "seed1fx"), ("mistral_7b", "seed1fx")] + \
             [("llama31_8b", f"seed1fx_{k}") for k in ("T2", "T8", "r8", "r64")]
    indep = [(b, c.replace("seed1fx", "indep1fx")) for b, c in shared]
    need = [f"{b}__{m}__{c}" for b, c in shared + indep for m in ("rm_rule", "rm_a09", "rm_0p01", "rm_1em6")]
    if any(worst(n) is None for n in need):
        rep["rule"] = "incomplete"
        return
    v1 = [worst(f"{b}__rm_rule__{c}") <= worst(f"{b}__rm_0p01__{c}") + 0.005 for b, c in shared]
    v2 = [worst(f"{b}__rm_rule__{c}") < worst(f"{b}__rm_a09__{c}") - 0.005 for b, c in shared]
    v3 = [abs(worst(f"{b}__rm_rule__{c}") - worst(f"{b}__rm_1em6__{c}")) <= 0.005 for b, c in indep]
    rep["rule"] = {"V1": sum(v1) >= 0.8 * len(v1), "V2": sum(v2) >= 0.6 * len(v2),
                   "V3": sum(v3) >= 0.8 * len(v3),
                   "counts": {"V1": [sum(v1), len(v1)], "V2": [sum(v2), len(v2)], "V3": [sum(v3), len(v3)]}}


def main() -> int:
    torch.set_default_dtype(torch.float64)
    rep = {"rules": ["notes/prereg_new_cohorts_2026-09-23.md", "notes/prereg_lambda_rule_2026-09-23.md"]}
    for f in (part_T, part_R, part_P, rule):
        f(rep)
    OUT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items()
                                                         if isinstance(vv, (bool, str))})
                      for k, v in rep.items() if k != "rules"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Geometry for prereg_training_regimes_2026-10-19 (and its amendment), CPU.

For a cohort (one adapter dir per task): Algorithm 1's median principal cosine
over all modules; kappa(Hbar), the exact amplification and the tilt norm t
(consensus W0, the amendment's definition) on the 8 sampled layers of the drift
analysis, as medians over layers. Functions are drift_theory.py's.

  --calibrate : C_k = median(kappa t^2), C_a = median(amp t) over the four
                calibration cohorts; writes results/phase3/regimes_calibration.json
  --cohorts   : measure every Part S/K/C/D cohort and the references;
                writes results/phase3/regimes_geometry.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_theory as dt  # noqa: E402  (sets float64 default)
from _c1019 import RES, ROOT  # noqa: E402

TASK_DIRS = ["gsm8k", "alpaca", "magicoder", "flores"]
LORA = ROOT / "artifacts" / "lora" / "llama31_8b"
REG = ROOT / "artifacts" / "lora_regimes" / "llama31_8b"


def calib_cohorts():
    fixed = Path(__import__("os").environ.get(
        "FLORES_FIX", ROOT / "artifacts_new" / "lora" / "llama31_8b" / "flores" / "seed1fx"))
    out = {f"seed{k}": [LORA / t / f"seed{k}" for t in TASK_DIRS] for k in (1, 2, 3)}
    out["seed1_flores_fix"] = [LORA / t / "seed1" for t in TASK_DIRS[:3]] + [fixed]
    return out


def median_cosine(dirs):
    mats = [dt.load_ab(d / "adapter_model.safetensors")[0] for d in dirs]
    keys = sorted(set.intersection(*[set(m) for m in mats]))
    cos = []
    for k in keys:
        Q = [torch.linalg.qr(m[k].T)[0] for m in mats]
        for i in range(len(Q)):
            for j in range(i + 1, len(Q)):
                cos.append(float(torch.linalg.svdvals(Q[i].T @ Q[j]).clamp(max=1).median()))
    cos.sort()
    return cos[len(cos) // 2]


def theory(dirs):
    ks, amps, ts = [], [], []
    for key, As, Bs, _ in dt.cohort_layers([d / "adapter_model.safetensors" for d in dirs]):
        Ds = [B @ A for A, B in zip(As, Bs)]
        ex = dt.exact(As, Ds)
        W0 = dt.consensus_W0(As, As[0].shape[0])
        pr = dt.predicted(As, Ds, W0)
        ks.append(ex["kappa"]); amps.append(ex["amp"]); ts.append(pr["eps_tilt_median"])
    return {"kappa": st.median(ks), "amp": st.median(amps), "t": st.median(ts)}


def measure(dirs):
    return {"median_cos": median_cosine(dirs), **theory(dirs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--cohorts", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        pts = {}
        for name, dirs in calib_cohorts().items():
            pts[name] = measure(dirs)
            p = pts[name]
            print(f"{name:18s} cos {p['median_cos']:.3f} kappa {p['kappa']:.4g} "
                  f"amp {p['amp']:.3f} t {p['t']:.4f}", flush=True)
        C_k = st.median(p["kappa"] * p["t"] ** 2 for p in pts.values())
        C_a = st.median(p["amp"] * p["t"] for p in pts.values())
        out = {"rules": ["notes/prereg_training_regimes_2026-10-19.md",
                         "notes/prereg_training_regimes_amendment_2026-10-19.md"],
               "points": pts, "C_k": C_k, "C_a": C_a}
        (RES / "regimes_calibration.json").write_text(json.dumps(out, indent=1))
        print(f"C_k = {C_k:.6g}   C_a = {C_a:.6g}")
    if a.cohorts:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gen_training_regimes import COHORTS as REGIMES  # noqa: E402
        res = {}
        for name in REGIMES:
            dirs = [REG / t / name for t in TASK_DIRS]
            if not all((d / "adapter_model.safetensors").exists() for d in dirs):
                print(f"{name}: incomplete, skipped", flush=True)
                continue
            res[name] = measure(dirs)
            print(name, res[name], flush=True)
        res["indep1"] = measure([LORA / t / "indep1" for t in TASK_DIRS])
        (RES / "regimes_geometry.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

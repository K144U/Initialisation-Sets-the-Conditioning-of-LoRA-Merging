#!/usr/bin/env python3
"""A1 / W2 readout: how much do a cohort's task subspaces actually overlap?

Reports, per base and per layer, from adapter artifacts only (CPU, no base
model, no GPU):

  * pairwise principal cosines between the tasks' top-r right-singular
    subspaces (1.0 = identical subspace, 0 = orthogonal)
  * relative distance between the raw LoRA A factors, which is what exposes a
    shared initialisation (~0.18 when shared, ~1.41 = sqrt(2) when independent)
  * hard d_eff as a function of the rank threshold epsilon, over six decades,
    which is exactly what the review's W2 asks for
  * soft d_eff, the threshold-free participation ratio
  * the Lemma 2 floor B^2 (1 - d_eff/(Tr)) under both statistics

Motivation: the published cohorts report hard d_eff = Tr = 64 in every layer,
so floor = 0, and the paper reads that as "the task subspaces are linearly
independent". Measured principal cosines are median 0.996 and the participation
ratio is 16.3 of 64, i.e. the subspaces are near-coincident and the rank test
only says otherwise because its tolerance is ~1e-3 sigma_1. Run this on the
independent-init cohort to see whether the geometry was ever a property of
fine-tuning or purely of the shared LoRA A init.

Usage:
  python code/phase3/scripts/measure_subspace_geometry.py --cohort seed1
  python code/phase3/scripts/measure_subspace_geometry.py --cohort indep1
  python code/phase3/scripts/measure_subspace_geometry.py --cohort seed1 --layers 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
ART = ROOT / "artifacts/lora"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASK_DIRS = ["gsm8k", "alpaca", "magicoder", "flores"]
EPS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]


def principal_cosines(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """X, Y: (d, r) with orthonormal columns. Returns r cosines, descending."""
    return torch.linalg.svdvals(X.T @ Y).clamp(max=1.0)


def analyse_layer(A: dict[str, torch.Tensor], D: dict[str, torch.Tensor],
                  r: int) -> dict:
    tasks = list(A)
    T = len(tasks)
    Tr = T * r
    # Delta_t = B_t A_t is exactly rank r, so its right-singular span is
    # rowspace(A_t); a QR is exact here and far cheaper than a full SVD.
    V = {t: torch.linalg.qr(A[t].T)[0][:, :r] for t in tasks}

    cos_pairs = []
    for i in range(T):
        for j in range(i + 1, T):
            cos_pairs.append(float(principal_cosines(V[tasks[i]], V[tasks[j]]).median()))
    a_rel = []
    for i in range(1, T):
        a_rel.append(float((A[tasks[i]] - A[tasks[0]]).norm() / A[tasks[0]].norm()))

    M = torch.cat([V[t] for t in tasks], dim=1)
    S = torch.linalg.svdvals(M)
    S2 = S * S
    soft = float((S2.sum() ** 2) / (S2 * S2).sum())
    B2 = max(float(D[t].norm()) ** 2 for t in tasks)

    hard = {}
    floor = {}
    for e in EPS:
        de = int((S > e * S[0]).sum())
        hard[e] = de
        floor[e] = B2 * max(0.0, 1.0 - de / Tr)

    return {
        "median_principal_cosine": sum(cos_pairs) / len(cos_pairs),
        "A_rel_distance": sum(a_rel) / len(a_rel),
        "sigma_max": float(S[0]),
        "sqrt_T": T ** 0.5,
        "soft_d_eff": soft,
        "soft_floor": B2 * max(0.0, 1.0 - soft / Tr),
        "B2": B2,
        "Tr": Tr,
        "hard_d_eff": hard,
        "hard_floor": floor,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="seed1",
                    help="adapter subdirectory, e.g. seed1, seed2, indep1, v1")
    ap.add_argument("--layers", type=int, default=8,
                    help="how many layers to sample per base (0 = all)")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    out: dict[str, dict] = {}
    print(f"cohort = {args.cohort}\n")
    print(f"{'base':<13}{'cos':>7}{'|dA|':>7}{'smax':>7}{'soft':>8}"
          f"{'softFloor':>11}   hard d_eff by eps "
          + " ".join(f"{e:>7.0e}" for e in EPS))
    for base in BASES:
        paths = {t: ART / base / t / args.cohort / "adapter_model.safetensors"
                 for t in TASK_DIRS}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            print(f"{base:<13} SKIP, missing: {missing[0]}")
            continue
        mats = {t: load_file(str(p)) for t, p in paths.items()}
        keys = sorted(k for k in mats[TASK_DIRS[0]] if k.endswith("lora_A.weight"))
        if args.layers and args.layers < len(keys):
            step = max(1, len(keys) // args.layers)
            keys = keys[::step][:args.layers]

        per_layer = []
        for ka in keys:
            kb = ka.replace("lora_A", "lora_B")
            A = {t: mats[t][ka].float() for t in TASK_DIRS}
            D = {t: mats[t][kb].float() @ mats[t][ka].float() for t in TASK_DIRS}
            per_layer.append(analyse_layer(A, D, args.rank))

        n = len(per_layer)
        agg = {
            "n_layers": n,
            "median_principal_cosine": sum(l["median_principal_cosine"] for l in per_layer) / n,
            "A_rel_distance": sum(l["A_rel_distance"] for l in per_layer) / n,
            "sigma_max": sum(l["sigma_max"] for l in per_layer) / n,
            "sqrt_T": per_layer[0]["sqrt_T"],
            "soft_d_eff": sum(l["soft_d_eff"] for l in per_layer) / n,
            "soft_floor": sum(l["soft_floor"] for l in per_layer) / n,
            "Tr": per_layer[0]["Tr"],
            "hard_d_eff": {str(e): sum(l["hard_d_eff"][e] for l in per_layer) / n
                           for e in EPS},
            "hard_floor": {str(e): sum(l["hard_floor"][e] for l in per_layer) / n
                           for e in EPS},
        }
        out[base] = agg
        print(f"{base:<13}{agg['median_principal_cosine']:>7.3f}"
              f"{agg['A_rel_distance']:>7.2f}{agg['sigma_max']:>7.3f}"
              f"{agg['soft_d_eff']:>8.1f}{agg['soft_floor']:>11.4f}   "
              + " " * 17
              + " ".join(f"{agg['hard_d_eff'][str(e)]:>7.1f}" for e in EPS))

    if out:
        print(f"\nreading: cos ~ 1.0 and soft ~ r means the task subspaces are")
        print(f"near-coincident; |dA| ~ 0.2 means a shared LoRA A init, ~1.41 independent.")
        print(f"sigma_max ~ sqrt(T) = {out[list(out)[0]]['sqrt_T']:.2f} is the "
              f"signature of identical subspaces.")
        p = RES / f"subspace_geometry_{args.cohort}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\n[geom] wrote {p}")
    else:
        print("\n[geom] no cohorts found; nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

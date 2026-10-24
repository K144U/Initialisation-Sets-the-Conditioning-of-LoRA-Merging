#!/usr/bin/env python3
"""Per-layer subspace statistics for the Neurocomputing figures (1b, 1c, 2a).

Reads only the LoRA factors of the trained adapters. For every base model and
cohort it records, over the same eight sampled layers as
code/phase3/scripts/measure_subspace_geometry.py:

  cos      all principal cosines between every pair of task row spaces
  svals    singular values of the stacked orthonormal bases M = [V_1 .. V_T]
  eig      exact nonzero eigenvalues of Hbar = (1/T) sum_t P_t, = svals^2 / T
  pred     (shared cohorts only) the first-order prediction of the bottom
           (T-1)r eigenvalues: those of Sigma_X = Y Y^T / T, with the
           reference-free consensus W0 of code/phase3/scripts/drift_theory.py

Nothing here trains or merges; it is linear algebra on the A factors, a few
minutes of CPU. Runs locally (FIG_THREADS caps the threads, default 4):

  python paper/nc/geometry_stats.py --art artifacts/lora \
      --out paper/nc/data/geometry_stats.json

The independent cohorts (indep1-3) were copied read-only from the cluster on
2026-10-06; no computation for this script runs there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

import torch
from safetensors.torch import load_file

# Shares a machine with other experiments: keep to a few threads.
torch.set_num_threads(int(os.environ.get("FIG_THREADS", "4")))

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
COHORTS = ["seed1", "seed2", "seed3", "indep1", "indep2", "indep3"]
N_LAYERS = 8
R = 16


def orth_rows(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(A.T)[0][:, :A.shape[0]]


def consensus_W0(As, r):
    V = torch.cat([orth_rows(A) for A in As], dim=1)
    U, _, _ = torch.linalg.svd(V, full_matrices=False)
    return U[:, :r]


def predicted_bottom(As, W0):
    T, r = len(As), As[0].shape[0]
    Pperp = torch.eye(W0.shape[0], dtype=W0.dtype) - W0 @ W0.T
    X = [Pperp @ A.T @ torch.linalg.inv(W0.T @ A.T) for A in As]
    Xbar = sum(X) / T
    Y = torch.cat([x - Xbar for x in X], dim=1)
    sv = torch.linalg.svdvals(Y)
    k = (T - 1) * r
    return (sv[:k] ** 2 / T).tolist()


def rnd(xs, n=6):
    return [float(f"{x:.{n}g}") for x in xs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    art = Path(args.art)
    out: dict = {"layers_sampled": N_LAYERS, "rank": R, "tasks": TASKS, "data": {}}
    for base in BASES:
        for coh in COHORTS:
            paths = [art / base / t / coh / "adapter_model.safetensors" for t in TASKS]
            if not all(p.exists() for p in paths):
                print("skip", base, coh, flush=True)
                continue
            sds = [load_file(str(p)) for p in paths]
            keys = sorted(k for k in sds[0] if k.endswith("lora_A.weight"))
            step = max(1, len(keys) // N_LAYERS)
            keys = keys[::step][:N_LAYERS]
            rec = {"layers": keys, "cos": [], "svals": [], "eig": [], "pred": []}
            for k in keys:
                As = [sd[k].double() for sd in sds]
                V = [orth_rows(A) for A in As]
                for i in range(len(V)):
                    for j in range(i + 1, len(V)):
                        rec["cos"].extend(rnd(torch.linalg.svdvals(V[i].T @ V[j]).clamp(max=1).tolist()))
                S = torch.linalg.svdvals(torch.cat(V, dim=1))
                rec["svals"].append(rnd(S.tolist()))
                rec["eig"].append(rnd((S ** 2 / len(V)).tolist()))
                if coh.startswith("seed"):
                    rec["pred"].append(rnd(predicted_bottom(As, consensus_W0(As, R))))
            out["data"][f"{base}/{coh}"] = rec
            cos = torch.tensor(rec["cos"])
            print(f"{base:12s} {coh:7s} median cos {float(cos.median()):.4f}", flush=True)
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

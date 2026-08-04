#!/usr/bin/env python3
"""Why is DARE numerically indistinguishable from Task Arithmetic in our matrix?

Context. On three independently initialised cohorts, task_arithmetic, dare and
knots agree to within 0.0025 nats on all four bases. The KnOTS no-op is
established (A2: V V^T = I cancels under inner_combination="linear"). This
script tests the competing explanation for DARE.

Our drop-and-rescale operator was checked against the official implementation
(yule-BUAA/MergeLM, model_merging_methods/mask_weights_utils.py) and is
algebraically identical: keep each element with probability `density`, divide
survivors by `density`. Official uses mask_rate p and divides by (1 - p); our
density is (1 - p). Our merge is mask + task_arithmetic, which is the official
mask_merging wrapper with mask_apply_method="task_arithmetic". So the operator
is faithful.

The one structural difference is that the official pipeline merges full weight
matrices and applies NO low-rank truncation anywhere, whereas every method in
our pipeline is SVD-truncated back to rank r after the merge, because the
output has to be a LoRA adapter again.

That difference should matter specifically and only for DARE. Rescaling makes
DARE an unbiased estimator of the task vector:

    E[delta * mask / density] = delta

so DARE cannot change the merged delta in expectation. Its entire effect is
variance, injected as a dense, near-isotropic perturbation spread over all
min(d_out, d_in) singular directions. Projecting onto the top-r subspace keeps
roughly r / min(d_out, d_in) of that energy. With r = 16 and d = 4096 that is
0.4 percent. TIES survives the same truncation because trimming and sign
election are BIASED operations that move the dominant singular directions;
DARE's is not.

Measured here, per base and layer, on real adapters, CPU only:

  rel_pre    ||D_dare - D_ta||_F / ||D_ta||_F        before truncation
  rel_post   ||T_r(D_dare) - T_r(D_ta)||_F / ||T_r(D_ta)||_F   after
  survive    ||T_r(D_dare) - T_r(D_ta)||^2 / ||D_dare - D_ta||^2

`survive` is the fraction of everything DARE injects that survives the rank-r
projection. Truncation uses the production routine svd_truncate_to_rank, so
this measures what the pipeline actually did, not an idealisation.

Swept over density, because our matrix used 0.2 (drop 80 percent) while the
DARE paper's headline results drop 90 to 99 percent. The analytic prediction is
rel_pre = sqrt((1 - density) / density), which the run should reproduce and
which is a check that the operator does what it claims.

Usage:
  python code/phase3/scripts/diagnose_dare_truncation.py
  python code/phase3/scripts/diagnose_dare_truncation.py --cohort indep2 --layers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
sys.path.insert(0, str(ROOT / "code/phase3"))
from merging._adapter_utils import svd_truncate_to_rank  # noqa: E402

ART = ROOT / "artifacts/lora"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASK_DIRS = ["gsm8k", "alpaca", "magicoder", "flores"]
DENSITIES = [0.2, 0.1, 0.05, 0.01]     # 0.2 is ours; 0.1-0.01 is the paper's range
SEED = 20260518                        # the seed the matrix cells used


def dare_op(delta: torch.Tensor, density: float, g: torch.Generator) -> torch.Tensor:
    """Byte-for-byte the operator in merging/dare.py."""
    mask = (torch.rand(delta.shape, generator=g, dtype=torch.float32) < density)
    return (delta * mask.to(delta.dtype)) / density


def trunc(delta: torch.Tensor, r: int) -> torch.Tensor:
    A, B = svd_truncate_to_rank(delta, r)
    return B @ A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="indep1")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    w = 1.0 / len(TASK_DIRS)
    out: dict = {"cohort": args.cohort, "rank": args.rank, "bases": {}}

    print(f"cohort = {args.cohort}   rank = {args.rank}   weights = uniform {w}")
    print("DARE vs Task Arithmetic, before and after the rank-r truncation")
    print("survive = fraction of DARE's injected perturbation energy that the")
    print("          rank-r projection lets through\n")
    print(f"{'base':<13}{'density':>8}{'rel_pre':>10}{'predicted':>11}"
          f"{'rel_post':>10}{'survive':>10}")

    for base in BASES:
        paths = {t: ART / base / t / args.cohort / "adapter_model.safetensors"
                 for t in TASK_DIRS}
        if any(not p.exists() for p in paths.values()):
            print(f"{base:<13} SKIP, adapters missing")
            continue
        mats = {t: load_file(str(p)) for t, p in paths.items()}
        keys = sorted(k for k in mats[TASK_DIRS[0]] if k.endswith("lora_A.weight"))
        if args.layers and args.layers < len(keys):
            step = max(1, len(keys) // args.layers)
            keys = keys[::step][:args.layers]

        per_density: dict[str, list[dict]] = {f"{d}": [] for d in DENSITIES}
        for ka in keys:
            kb = ka.replace("lora_A", "lora_B")
            deltas = [mats[t][kb].float() @ mats[t][ka].float() for t in TASK_DIRS]
            D_ta = sum(w * d for d in deltas)
            T_ta = trunc(D_ta, args.rank)
            n_ta = float(D_ta.norm())
            n_T_ta = float(T_ta.norm())

            for dens in DENSITIES:
                g = torch.Generator().manual_seed(SEED)
                D_dare = sum(w * dare_op(d, dens, g) for d in deltas)
                T_dare = trunc(D_dare, args.rank)
                pre = float((D_dare - D_ta).norm())
                post = float((T_dare - T_ta).norm())
                per_density[f"{dens}"].append({
                    "rel_pre": pre / n_ta,
                    "rel_post": post / n_T_ta,
                    "survive": (post ** 2) / (pre ** 2) if pre > 0 else 0.0,
                    "ta_energy_kept": n_T_ta / n_ta,
                })

        out["bases"][base] = {}
        for dens in DENSITIES:
            rows = per_density[f"{dens}"]
            n = len(rows)
            agg = {k: sum(r[k] for r in rows) / n for k in rows[0]}
            agg["n_layers"] = n
            out["bases"][base][f"{dens}"] = agg
            pred = ((1.0 - dens) / dens) ** 0.5
            print(f"{base:<13}{dens:>8.2f}{agg['rel_pre']:>10.3f}{pred:>11.3f}"
                  f"{agg['rel_post']:>10.3f}{agg['survive']:>10.4f}")
        print()

    r, dmin = args.rank, 4096
    print(f"isotropic reference: a rank-{r} projection of a dense perturbation in")
    print(f"{dmin} dimensions passes about {r / dmin:.4f} of its energy.")
    print("TA itself loses energy to the same truncation (it is a sum of four")
    print("rank-16 deltas, so rank <= 64 going in):")
    for base in out["bases"]:
        k = out["bases"][base][f"{DENSITIES[0]}"]["ta_energy_kept"]
        print(f"  {base:<13} ||T_r(D_ta)|| / ||D_ta|| = {k:.4f}")

    p = RES / f"dare_truncation_diagnosis_{args.cohort}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[dare] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does RegMean's own published default already regularise the collapse?

A referee is entitled to ask this and we did not answer it. Our practical
recommendation reduces to "regularise a solve-based merge", and RegMean ships
a regulariser of its own. If its default were already strong enough, the
recommendation would collapse to the much smaller claim that the audit tells
you when the default matters.

The published regulariser is *not* a Tikhonov ridge. RegMean (Jin et al.,
2023, Sec. 3.3) scales the **non-diagonal** entries of each per-task Gram by a
scalar alpha, default 0.9:

    G_t  ->  alpha * G_t + (1 - alpha) * diag(G_t)

which is a diagonal shrinkage whose size is set by the data rather than by a
free parameter. Our implementation instead adds lambda * I, which we chose;
the paper called it "part of RegMean's own published formulation", and that
was wrong.

This script compares the two on the same Grams, at the same layers, for both
cohorts. It is CPU-only and reads only the LoRA A factors. It reports
conditioning and solution norm, not NLL: no merge is evaluated here, and no
claim about task loss is made from it.

Usage:
  python code/phase3/scripts/regmean_published_default.py [--bases ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
LORA = REPO / "artifacts" / "lora"
OUT = REPO / "results" / "phase3" / "regmean_published_default.json"

TASKS = ["alpaca", "gsm8k", "magicoder", "flores"]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
ALPHA = 0.9          # RegMean's published default
LAMBDAS = [1e-6, 1e-3, 1e-2]


def a_factors(base: str, cohort: str) -> dict[str, list[torch.Tensor]]:
    """lora_A per layer, one entry per task, for one cohort."""
    per_layer: dict[str, list[torch.Tensor]] = {}
    for task in TASKS:
        d = LORA / base / task / cohort / "adapter_model.safetensors"
        if not d.exists():
            return {}
        sd = load_file(str(d))
        for k, v in sd.items():
            if "lora_A" not in k:
                continue
            layer = k.split(".lora_A")[0]
            per_layer.setdefault(layer, []).append(v.float())
    # keep only layers present for every task
    return {k: v for k, v in per_layer.items() if len(v) == len(TASKS)}


def stats(As: list[torch.Tensor]) -> dict:
    """Conditioning of the RegMean system under each regulariser.

    The Gram is summed over tasks exactly as the merge builds it. Its rank is
    at most T*r, far below the input dimension, so the unregularised system is
    singular rather than merely ill-conditioned; that is why the comparison is
    between regularisers and never against lambda = 0.
    """
    G = sum(A.T @ A for A in As)
    d = G.shape[0]
    eig = torch.linalg.eigvalsh(G)
    scale = float(eig.max())

    out = {"dim": d, "rank_numeric": int((eig > 1e-8 * scale).sum()),
           "gram_sigma_max": scale}

    # RegMean's published alpha-shrinkage.
    G_alpha = ALPHA * G + (1.0 - ALPHA) * torch.diag(torch.diag(G))
    e = torch.linalg.eigvalsh(G_alpha)
    out["alpha0.9"] = {
        "lambda_min": float(e.min()), "lambda_max": float(e.max()),
        "cond": float(e.max() / e.min()) if e.min() > 0 else float("inf"),
        # the equivalent isotropic ridge, for comparability with our sweep
        "equiv_ridge_mean": float((1.0 - ALPHA) * torch.diag(G).mean()),
    }

    for lam in LAMBDAS:
        e = torch.linalg.eigvalsh(G + lam * torch.eye(d))
        out[f"ridge{lam:g}"] = {
            "lambda_min": float(e.min()), "lambda_max": float(e.max()),
            "cond": float(e.max() / e.min()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="*", default=BASES)
    ap.add_argument("--layers", type=int, default=8,
                    help="layers sampled per cohort, as the geometry script does")
    args = ap.parse_args()

    report: dict = {"alpha": ALPHA, "lambdas": LAMBDAS, "bases": {}}
    for base in args.bases:
        entry = {}
        for cohort, label in (("seed1", "shared"), ("indep1", "independent")):
            per_layer = a_factors(base, cohort)
            if not per_layer:
                print(f"  {base}/{cohort}: adapters not present, skipped")
                continue
            names = sorted(per_layer)[: args.layers]
            rows = [stats(per_layer[n]) for n in names]
            agg = {
                "n_layers": len(rows),
                "rank_numeric": rows[0]["rank_numeric"],
                "dim": rows[0]["dim"],
                "alpha_cond_median": float(
                    torch.tensor([r["alpha0.9"]["cond"] for r in rows]).median()),
                "alpha_equiv_ridge_median": float(
                    torch.tensor([r["alpha0.9"]["equiv_ridge_mean"]
                                  for r in rows]).median()),
                "gram_sigma_max_median": float(
                    torch.tensor([r["gram_sigma_max"] for r in rows]).median()),
            }
            for lam in LAMBDAS:
                agg[f"cond_ridge{lam:g}_median"] = float(
                    torch.tensor([r[f"ridge{lam:g}"]["cond"]
                                  for r in rows]).median())
            entry[label] = agg
            print(f"  {base}/{label}: alpha cond {agg['alpha_cond_median']:.3e}, "
                  f"equiv ridge {agg['alpha_equiv_ridge_median']:.4g}, "
                  f"ridge1e-2 cond {agg['cond_ridge0.01_median']:.3e}")
        if entry:
            report["bases"][base] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

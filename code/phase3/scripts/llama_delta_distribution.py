"""Llama-3.1 anomaly hypothesis 2 test — Δ magnitude distribution flatness.

For each (model, task) v1 adapter, materialize per-layer
Δ = scaling · B @ A, and compute distribution statistics of |Δ|:

  mean, median, p95, p99            — basic location
  std, coefficient of variation      — spread
  kurtosis                           — peakedness (Pearson, excess > 0 = peaked)
  p99 / median                       — tail heaviness (higher = more peaked)
  top20_threshold / median           — where TIES's cut lands vs mass center
  fraction near top20 threshold      — sensitivity-to-noise proxy

Aggregate per model across the 4 task adapters and all attention
projection layers. If Llama-3.1's distribution is measurably FLATTER
(lower kurtosis, lower p99/median, lower top20/median, more mass
near the top-20% threshold), hypothesis 2 is supported: TIES's
magnitude threshold lands in a noisy regime, making the selection
stochastic and reasoning-bit-discarding under merging.

CPU only. Reads from artifacts/lora/<model>/<task>/v1/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/sanjay.g/projects/rdmerge/code/phase3")
from eval.deff_analysis import load_adapter_factors


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
OUT_JSON = PROJECT_ROOT / "results/phase3/llama_delta_distribution.json"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]   # flores = translation v1
BASE_MODELS = {
    "llama31_8b": "Llama-3.1-8B-Instruct",
    "mistral_7b": "Mistral-7B-Instruct-v0.3",
    "qwen25_7b":  "Qwen2.5-7B-Instruct",
    "yi15_9b":    "Yi-1.5-9B-Chat",
}


def materialize_delta(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Returns {layer -> scaling * B @ A} in fp32 on CPU."""
    factors, scaling_map = load_adapter_factors(str(adapter_dir))
    out: dict[str, torch.Tensor] = {}
    for layer, f in factors.items():
        A = f["A"].to(torch.float32)
        B = f["B"].to(torch.float32)
        scale = scaling_map[layer]
        out[layer] = scale * (B @ A)
    return out


def layer_stats(delta: torch.Tensor) -> dict:
    """Per-tensor distribution stats on |Δ|. Uses quantile (O(n)) for
    speed; subsamples to 1M coords for kurtosis to bound memory."""
    a = delta.abs().flatten()
    n = a.numel()
    qs = torch.quantile(a, torch.tensor([0.50, 0.80, 0.95, 0.99],
                                         dtype=a.dtype))
    median, top20_thresh, p95, p99 = (float(q) for q in qs)
    mean = float(a.mean())
    std = float(a.std())
    cv = std / mean if mean > 0 else float("inf")
    # Excess kurtosis on subsample for speed (kurtosis converges
    # quickly; we only need 3-decimal precision).
    if n > 1_000_000:
        idx = torch.randperm(n)[:1_000_000]
        sub = a[idx]
    else:
        sub = a
    if std > 0:
        z = (sub - mean) / std
        kurt = float((z ** 4).mean() - 3)
    else:
        kurt = 0.0
    top20_over_median = top20_thresh / max(median, 1e-12)
    # Sensitivity-to-noise: fraction within [0.9, 1.1] of threshold.
    lo = 0.9 * top20_thresh
    hi = 1.1 * top20_thresh
    near_fraction = float(((a >= lo) & (a <= hi)).float().mean())
    return {
        "mean": mean, "median": median, "p95": p95, "p99": p99,
        "std": std, "cv": cv, "kurtosis": kurt,
        "top20_threshold": top20_thresh,
        "top20_over_median": top20_over_median,
        "p99_over_median": p99 / max(median, 1e-12),
        "near_threshold_fraction": near_fraction,
        "n_coords": n,
    }


def aggregate(stats_list: list[dict]) -> dict:
    """Mean over a list of per-layer stats dicts."""
    keys = [k for k in stats_list[0].keys() if k != "n_coords"]
    out = {k: sum(s[k] for s in stats_list) / len(stats_list) for k in keys}
    out["n_layers"] = len(stats_list)
    return out


def main() -> int:
    summary: dict[str, dict] = {}
    print(f"{'model':<14}{'task':<12}{'mean(|Δ|)':<12}{'p99/med':<10}"
          f"{'kurt':<10}{'top20/med':<12}{'near%':<8}")
    print("-" * 80)
    for model in MODELS:
        per_task_aggregates = []
        for task in TASKS:
            adir = PROJECT_ROOT / "artifacts/lora" / model / task / "v1"
            if not adir.exists():
                print(f"  [warn] missing {adir}", file=sys.stderr)
                continue
            deltas = materialize_delta(adir)
            layer_stats_list = [layer_stats(d) for d in deltas.values()]
            agg = aggregate(layer_stats_list)
            per_task_aggregates.append((task, agg))
            print(f"{model:<14}{task:<12}{agg['mean']:<12.5f}"
                  f"{agg['p99_over_median']:<10.2f}{agg['kurtosis']:<10.2f}"
                  f"{agg['top20_over_median']:<12.2f}"
                  f"{agg['near_threshold_fraction']*100:<8.2f}")
        # Aggregate across the 4 tasks
        keys = [k for k in per_task_aggregates[0][1].keys()
                if k != "n_layers"]
        model_agg = {k: sum(a[1][k] for a in per_task_aggregates) / len(per_task_aggregates)
                     for k in keys}
        summary[model] = {
            "per_task": {t: a for t, a in per_task_aggregates},
            "across_tasks": model_agg,
        }

    print()
    print("=" * 80)
    print("Per-model averages (across 4 tasks × all attention projections)")
    print("=" * 80)
    print(f"{'model':<14}{'mean(|Δ|)':<12}{'p99/med':<10}{'kurt':<10}"
          f"{'top20/med':<12}{'near%':<8}{'cv':<8}")
    for model in MODELS:
        a = summary[model]["across_tasks"]
        print(f"{model:<14}{a['mean']:<12.5f}{a['p99_over_median']:<10.2f}"
              f"{a['kurtosis']:<10.2f}{a['top20_over_median']:<12.2f}"
              f"{a['near_threshold_fraction']*100:<8.2f}{a['cv']:<8.2f}")
    print()
    print("Interpretation cheatsheet:")
    print("  LOW p99/med + LOW kurtosis + LOW top20/med + HIGH near% = FLATTER")
    print("  Flatter Δ distribution ⇒ TIES's top-20% threshold lands closer")
    print("  to median, more coordinates near the threshold, selection is")
    print("  noisier ⇒ supports hypothesis 2.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

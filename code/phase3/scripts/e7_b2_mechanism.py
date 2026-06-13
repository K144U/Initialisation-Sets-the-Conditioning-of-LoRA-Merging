"""E7 — b=2 TVQ mechanism test (SKELETON).

Per master_plan §E7: the b=2 'less is more' dip survives E2 multi-seed
(confirmed 2026-06-13: 4/4 models show b2 < b4 worst-excess). Test two
falsifiable predictions of the "implicit TIES" hypothesis.

Prediction 1 (sparsity):
  - Measure fraction of coordinates zeroed by 2-bit TVQ per (model, layer).
  - Correlate dip depth (excess_b4 / excess_b2) with sparsity across the
    4 models, then across the 16 model-seed cells once E2 multi-seed
    completes.
  - Direct manipulation: magnitude pruning at the *same* per-layer
    sparsity as b=2 TVQ, no quantization. If pruning reproduces the dip,
    the mechanism is confirmed.

Prediction 2 (win-pattern):
  - Per-task excess of b=2 TVQ correlates with per-task excess of TIES.
  - Method: paired per-task differences across methods (one row per
    (model, task)), Pearson rank correlation between
    excess[b2] - excess[b4] and excess[TIES] - excess[TA].

Status: SKELETON. Implementation deferred. Total cost per master plan:
30-50 GPU-h. Slot when E5 main run breathes (likely week 6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --- Sparsity probe ------------------------------------------------------


def measure_tvq_b2_sparsity(adapter_dir: Path, b: int = 2) -> dict[str, float]:
    """For each LoRA layer, compute the fraction of merged delta
    coefficients that quantize to zero at b bits. Returns dict[layer -> frac]."""
    # TODO: load adapter, apply TVQ-b at every layer's delta, count zero coords.
    raise NotImplementedError


def magnitude_prune_at_sparsity(adapter_dir: Path,
                                 target_sparsity: dict[str, float]) -> dict:
    """Per-layer magnitude pruning that hits target_sparsity[layer] within
    1% tolerance, no quantization on the surviving weights. Returns
    in-memory merged factors ready for eval."""
    # TODO: per-layer threshold search; produce LoRA factors.
    raise NotImplementedError


# --- Correlation harness -------------------------------------------------


def dip_depth(excess_b2: float, excess_b4: float) -> float:
    """Dip ratio: how much better b=2 is than b=4 (higher = deeper dip)."""
    return excess_b4 / max(excess_b2, 1e-12)


def per_task_excess_correlation(matrix_results_dir: Path) -> dict:
    """Spearman correlation between:
        per-task excess[b2] - excess[b4]
    and
        per-task excess[TIES] - excess[TA]
    across (model, task) cells. r_s > 0.7 confirms Prediction 2."""
    # TODO: load all 80 matrix result JSONs (or 140 with cross-seed),
    # build paired arrays, return scipy.stats.spearmanr result.
    raise NotImplementedError


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sparsity", "prune", "correlation"],
                   required=True)
    p.add_argument("--matrix-dir", type=Path,
                   default=Path("/home/sanjay.g/projects/rdmerge/results/"
                                "phase3/eval_matrix_seeds"))
    args = p.parse_args()
    if args.mode == "sparsity":
        # TODO: iterate adapters, write sparsity table.
        raise NotImplementedError
    elif args.mode == "prune":
        # TODO: build pruned merges, eval them, write JSONs into
        # results/phase3/eval_e7_prune/<model>.json.
        raise NotImplementedError
    else:  # correlation
        out = per_task_excess_correlation(args.matrix_dir)
        json.dump(out, sys.stdout, indent=2)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

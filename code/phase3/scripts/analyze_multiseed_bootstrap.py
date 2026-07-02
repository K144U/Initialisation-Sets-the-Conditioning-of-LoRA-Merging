"""Multi-seed bootstrap analysis on the §6.1 T=4 matrix.

Reads seed1, seed2, and (when available) seed3 cells of the
4-base × 5-method matrix from results/phase3/eval_matrix_seeds/,
computes the per-cell mean across seeds and a non-parametric bootstrap
95% confidence interval for each method-method gap on each base.

Decision rules:
  - "Confident gap": |delta| > CI_half_width AND CI doesn't cross 0
  - "Borderline":    CI crosses 0 but |delta| > 0.5 * CI_half_width
  - "Inconclusive":  CI crosses 0 AND |delta| <= 0.5 * CI_half_width

Writes results/phase3/multiseed_bootstrap_summary.json with per-base
per-method-pair gap + CI.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Portable root: .../code/phase3/scripts/analyze_multiseed_bootstrap.py -> repo root.
ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "results/phase3/eval_matrix_seeds"
OUT = ROOT / "results/phase3/multiseed_bootstrap_summary.json"
N_BOOT = 10000
RNG_SEED = 20260625

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
METHOD_MAP = [
    ("ta",     "task_arithmetic"),
    ("ties",   "ties"),
    ("dare",   "dare"),
    ("knots",  "knots"),
    ("tvq_b2", "tvq_b2"),
]


def load_seed(base: str, mat_name: str, seed: int) -> float | None:
    p = MATRIX / f"{base}__{mat_name}__seed{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("worst_task_excess")


def bootstrap_diff_ci(a: list[float], b: list[float], n_boot: int = N_BOOT
                       ) -> tuple[float, float, float]:
    """Returns (point_diff, ci_lo, ci_hi) for a - b across paired seeds."""
    if not a or not b or len(a) != len(b):
        return float("nan"), float("nan"), float("nan")
    n = len(a)
    point = (sum(a) - sum(b)) / n
    rng = random.Random(RNG_SEED)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ai = [a[i] for i in idx]
        bi = [b[i] for i in idx]
        diffs.append((sum(ai) - sum(bi)) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return point, lo, hi


def main() -> int:
    # Find which seeds are available per (base, method)
    seeds = [1, 2, 3]
    table: dict[str, dict[str, list[float]]] = {}
    for base in BASES:
        table[base] = {}
        for tag, mat_name in METHOD_MAP:
            cells = []
            for s in seeds:
                v = load_seed(base, mat_name, s)
                if v is not None:
                    cells.append(v)
            table[base][tag] = cells

    print("=" * 88)
    print(f"Multi-seed §6.1 matrix — worst-task NLL excess per cell")
    print("=" * 88)
    tags = [t for t, _ in METHOD_MAP]
    print(f"{'base':<12}{'method':<12}" +
          " ".join(f"seed{s:<6}" for s in seeds))
    for base in BASES:
        for tag in tags:
            row = [f"{v:.4f}" if v is not None else "--"
                   for v in table[base][tag] + [None] * (3 - len(table[base][tag]))]
            print(f"{base:<12}{tag:<12}" + " ".join(f"{r:<10}" for r in row))
    print()

    # Per-base, per-method-pair bootstrap CI
    print("=" * 88)
    print("Per-base method-pair bootstrap 95% CI (delta = first - second)")
    print("=" * 88)
    summary = {}
    for base in BASES:
        summary[base] = {}
        # Each base: pairs of (TA, TIES), (TIES, TVQ_b2), (TA, TVQ_b2),
        # (TIES, DARE), (TIES, KnOTS)
        pairs = [
            ("ta",     "ties"),
            ("ta",     "tvq_b2"),
            ("ties",   "tvq_b2"),
            ("ties",   "dare"),
            ("ties",   "knots"),
        ]
        for a_tag, b_tag in pairs:
            a = table[base][a_tag]
            b = table[base][b_tag]
            if len(a) < 2 or len(b) < 2:
                continue
            # Pair by index — match seed-i to seed-i
            n_pair = min(len(a), len(b))
            point, lo, hi = bootstrap_diff_ci(a[:n_pair], b[:n_pair])
            ci_half = (hi - lo) / 2
            verdict = (
                "CONFIDENT" if (point > 0 and lo > 0) or (point < 0 and hi < 0)
                else "BORDERLINE" if abs(point) > 0.5 * ci_half
                else "INCONCLUSIVE"
            )
            summary[base][f"{a_tag}_minus_{b_tag}"] = {
                "delta": point, "lo": lo, "hi": hi,
                "ci_half_width": ci_half, "verdict": verdict,
                "n_paired": n_pair,
            }
            print(f"  {base:<12} {a_tag:<8} - {b_tag:<8}  "
                  f"delta = {point:+.4f}  [95%: {lo:+.4f}, {hi:+.4f}]  "
                  f"({verdict}, n={n_pair})")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_cell": table, "method_pair_cis": summary},
              open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

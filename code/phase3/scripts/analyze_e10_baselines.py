"""E10 added baselines analysis.

Compares Fisher-magnitude and DELLA against the published §6.1 matrix
(TA, TIES, DARE, KnOTS, TVQ-b=2) on worst-task NLL excess across 4
bases (Llama-3.1-8B-Instruct, Mistral-7B, Qwen-2.5-7B, Yi-1.5-9B-Chat).

Decision rules:
  Fisher-avg / DELLA in TOP-2 on >=3 bases -> add to §6.8 as winners.
  Fisher-avg / DELLA neither best nor worst -> add to §6.8 as "competitive
    but not headline" — closes the reviewer comment without disrupting
    R1/R3 ordering.
  Either method WORST on >=3 bases -> highlight as "baseline that hurts"
    (useful contra-example).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
E10_DIR = PROJECT_ROOT / "results/phase3/eval_e10_baselines"
MATRIX_DIR = PROJECT_ROOT / "results/phase3/eval_matrix_seeds"
OUT_JSON = PROJECT_ROOT / "results/phase3/e10_baselines_summary.json"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
PUBLISHED_METHODS = [
    ("ta",     "task_arithmetic"),
    ("ties",   "ties"),
    ("dare",   "dare"),
    ("knots",  "knots"),
    ("tvq_b2", "tvq_b2"),
]
NEW_METHODS = ["fisher_avg", "della"]


def load_published(model: str, mat_name: str, seed: int = 1) -> float | None:
    p = MATRIX_DIR / f"{model}__{mat_name}__seed{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("worst_task_excess")


def load_new(model: str, method: str) -> float | None:
    p = E10_DIR / f"{model}__{method}__seed1.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("worst_task_excess")


def main() -> int:
    table: dict[str, dict[str, float | None]] = {}
    for model in MODELS:
        row = {}
        for tag, mat_name in PUBLISHED_METHODS:
            row[tag] = load_published(model, mat_name)
        for m in NEW_METHODS:
            row[m] = load_new(model, m)
        table[model] = row

    all_methods = [t for t, _ in PUBLISHED_METHODS] + NEW_METHODS
    print("=" * 96)
    print("E10 baselines vs published — worst-task NLL excess (seed-1, lower is better)")
    print("=" * 96)
    print(f"{'model':<12}" + "".join(f"{m:<11}" for m in all_methods))
    for model in MODELS:
        row = table[model]
        cells = []
        for m in all_methods:
            v = row[m]
            if v is None:
                cells.append(f"{'--':<11}")
            else:
                cells.append(f"{v:<11.4f}")
        print(f"{model:<12}" + "".join(cells))
    print()

    # Rank each method per base
    print("=" * 96)
    print("Per-base rank (1 = best, 7 = worst)")
    print("=" * 96)
    print(f"{'model':<12}" + "".join(f"{m:<11}" for m in all_methods))
    rank_table = {}
    for model in MODELS:
        valid_pairs = [(m, table[model][m]) for m in all_methods
                       if table[model][m] is not None]
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
        ranks = {m: i + 1 for i, (m, _) in enumerate(sorted_pairs)}
        rank_table[model] = ranks
        cells = [f"{ranks.get(m, '--'):<11}" for m in all_methods]
        print(f"{model:<12}" + "".join(str(c) for c in cells))
    print()

    # New methods verdict
    print("=" * 96)
    print("New methods verdict")
    print("=" * 96)
    for m in NEW_METHODS:
        wins = sum(1 for model in MODELS if rank_table[model].get(m) == 1)
        top2 = sum(1 for model in MODELS if (rank_table[model].get(m) or 99) <= 2)
        bottom = sum(1 for model in MODELS
                     if (rank_table[model].get(m) or 0) == len(all_methods))
        print(f"  {m:<12} best-of-7 on {wins}/4 bases, top-2 on {top2}/4, "
              f"worst on {bottom}/4")

    # Best method per base (incl. new methods)
    print()
    print("=" * 96)
    print("Best method per base (incl. new methods)")
    print("=" * 96)
    bests = {}
    for model in MODELS:
        best_m, best_v = min(
            ((m, table[model][m]) for m in all_methods if table[model][m] is not None),
            key=lambda x: x[1])
        bests[model] = (best_m, best_v)
        print(f"  {model:<12} best={best_m:<12} excess={best_v:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "excess_table": table,
        "rank_table": rank_table,
        "best_per_base": {k: {"method": v[0], "excess": v[1]} for k, v in bests.items()},
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

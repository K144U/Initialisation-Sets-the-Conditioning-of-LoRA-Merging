"""E3 GSM8K em sweep analysis.

Reads the 20 cells (4 models x 5 methods) from results/phase3/eval_e3_gsm8k/,
builds the 4x5 accuracy table, computes per-model Spearman rank
correlation between accuracy and NLL-excess (the matrix's seed-1
worst-task excess), and writes a JSON summary.

Decision rule (master plan §E3):
  per-model rho:
    >= 0.7  -> NLL conclusions hold on accuracy
    0.4-0.7 -> report alongside NLL
    < 0.4   -> NLL not predictive
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
E3_DIR = PROJECT_ROOT / "results/phase3/eval_e3_gsm8k"
MATRIX_DIR = PROJECT_ROOT / "results/phase3/eval_matrix_seeds"
OUT_JSON = PROJECT_ROOT / "results/phase3/e3_gsm8k_summary.json"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
# (tag in E3 filenames, matrix method name)
METHOD_MAP = [
    ("ta",     "task_arithmetic"),
    ("ties",   "ties"),
    ("dare",   "dare"),
    ("knots",  "knots"),
    ("tvq_b2", "tvq_b2"),
]


def spearmanr(a: list[float], b: list[float]) -> float:
    def ranks(xs):
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    return pearsonr(ranks(a), ranks(b))


def pearsonr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def load_accuracy(model: str, tag: str) -> float | None:
    p = E3_DIR / f"{model}__{tag}__gsm8k_em.json"
    if not p.exists():
        return None
    return json.load(open(p))["metric_score"]


def load_nll_excess(model: str, matrix_method: str, seed: int = 1) -> float | None:
    p = MATRIX_DIR / f"{model}__{matrix_method}__seed{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("worst_task_excess")


def main() -> int:
    acc_table: dict[str, dict[str, float | None]] = {}
    nll_table: dict[str, dict[str, float | None]] = {}
    for model in MODELS:
        acc_table[model] = {}
        nll_table[model] = {}
        for tag, mat_name in METHOD_MAP:
            acc_table[model][tag] = load_accuracy(model, tag)
            nll_table[model][tag] = load_nll_excess(model, mat_name)

    # Print the 4x5 accuracy table
    print("=" * 72)
    print(f"GSM8K em accuracy (n=500) — 4 models × 5 methods")
    print("=" * 72)
    tags = [t for t, _ in METHOD_MAP]
    print(f"{'model':<14}" + " ".join(f"{t:<8}" for t in tags))
    for model in MODELS:
        row = []
        for tag in tags:
            v = acc_table[model][tag]
            row.append(f"{v:<8.3f}" if v is not None else f"{'--':<8}")
        print(f"{model:<14}" + " ".join(row))
    print()

    # NLL excess reference (from matrix)
    print("=" * 72)
    print("NLL worst-task excess (seed-1, from §6.1 matrix)")
    print("=" * 72)
    print(f"{'model':<14}" + " ".join(f"{t:<8}" for t in tags))
    for model in MODELS:
        row = []
        for tag in tags:
            v = nll_table[model][tag]
            row.append(f"{v:<8.3f}" if v is not None else f"{'--':<8}")
        print(f"{model:<14}" + " ".join(row))
    print()

    # Per-model Spearman: accuracy ↔ -NLL excess (higher acc ↔ lower excess
    # → positive correlation expected)
    print("=" * 72)
    print("Per-model Spearman: accuracy ↔ -NLL excess (across 5 methods)")
    print("=" * 72)
    spearman_per_model = {}
    for model in MODELS:
        accs = [acc_table[model][t] for t, _ in METHOD_MAP]
        nlls = [nll_table[model][t] for t, _ in METHOD_MAP]
        if any(a is None or n is None for a, n in zip(accs, nlls)):
            print(f"  {model}: incomplete data")
            spearman_per_model[model] = None
            continue
        neg_nlls = [-n for n in nlls]
        rho = spearmanr(accs, neg_nlls)
        verdict = ("STRONG ≥0.7" if rho >= 0.7
                   else "MODERATE 0.4-0.7" if rho >= 0.4
                   else "WEAK <0.4")
        print(f"  {model:<14} rho = {rho:+.3f}  ({verdict})")
        spearman_per_model[model] = rho

    # Best method per model
    print()
    print("=" * 72)
    print("Best method per model (by accuracy)")
    print("=" * 72)
    bests = {}
    for model in MODELS:
        valid = [(t, acc_table[model][t]) for t, _ in METHOD_MAP
                 if acc_table[model][t] is not None]
        if not valid:
            print(f"  {model}: no data")
            continue
        best_tag, best_acc = max(valid, key=lambda x: x[1])
        bests[model] = {"tag": best_tag, "accuracy": best_acc}
        print(f"  {model:<14} best={best_tag:<8} acc={best_acc:.3f}")

    # Verdict
    print()
    print("=" * 72)
    valid_rhos = [r for r in spearman_per_model.values() if r is not None]
    if valid_rhos:
        n_strong = sum(1 for r in valid_rhos if r >= 0.7)
        n_mod = sum(1 for r in valid_rhos if 0.4 <= r < 0.7)
        n_weak = sum(1 for r in valid_rhos if r < 0.4)
        print(f"VERDICT: {n_strong}/{len(valid_rhos)} models show STRONG agreement"
              f" (NLL → accuracy holds)")
        print(f"         {n_mod}/{len(valid_rhos)} MODERATE, {n_weak}/{len(valid_rhos)} WEAK")
        if n_strong == len(valid_rhos):
            print("  -> Paper §6.5 lands as confirmation: NLL conclusions survive on accuracy.")
        elif n_strong + n_mod == len(valid_rhos):
            print("  -> §6.5 reports accuracy alongside NLL, noting moderate agreement.")
        else:
            print("  -> §6.5 reports accuracy as a separate story; NLL not fully predictive.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "accuracy": acc_table,
        "nll_excess": nll_table,
        "spearman_per_model": spearman_per_model,
        "best_per_model": bests,
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

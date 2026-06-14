"""E7 Phase 1 — TVQ b=2 dip ↔ TIES win-pattern correlation.

Master plan Prediction 2 (implicit-TIES hypothesis):
  per-task excess pattern of b=2 TVQ is correlated with per-task
  excess pattern of TIES.

Test: for each (model, task, seed) cell, compute:
  dip_depth = excess[b=4] - excess[b=2]      (positive when b=2 helps)
  ties_win  = excess[TA]  - excess[TIES]      (positive when TIES helps)
Then Spearman + Pearson correlation across the 32 cells
(4 models x 4 tasks x 2 seeds).

CPU only. Reads from results/phase3/eval_matrix_seeds/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MATRIX_DIR = Path("/home/sanjay.g/projects/rdmerge/results/phase3/eval_matrix_seeds")
OUT_JSON = Path("/home/sanjay.g/projects/rdmerge/results/phase3/e7_phase1_correlation.json")

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASKS = ["gsm8k", "alpaca", "magicoder", "translation"]


def load_excess(model: str, method: str, seed: int) -> dict[str, float] | None:
    p = MATRIX_DIR / f"{model}__{method}__seed{seed}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d.get("excess_per_task")


def spearmanr(a: list[float], b: list[float]) -> float:
    """Spearman = Pearson of ranks. Hand-rolled to avoid scipy dep."""
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
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def main() -> int:
    cells = []
    for model in MODELS:
        for seed in (1, 2):
            ex_b2 = load_excess(model, "tvq_b2", seed)
            ex_b4 = load_excess(model, "tvq_b4", seed)
            ex_ties = load_excess(model, "ties", seed)
            ex_ta = load_excess(model, "task_arithmetic", seed)
            if not all([ex_b2, ex_b4, ex_ties, ex_ta]):
                print(f"[warn] missing cell at {model} seed{seed}",
                      file=sys.stderr)
                continue
            for task in TASKS:
                dip = ex_b4[task] - ex_b2[task]
                ties_win = ex_ta[task] - ex_ties[task]
                cells.append({
                    "model": model, "task": task, "seed": seed,
                    "excess_b2": ex_b2[task],
                    "excess_b4": ex_b4[task],
                    "excess_ties": ex_ties[task],
                    "excess_ta": ex_ta[task],
                    "dip_depth": dip,
                    "ties_win": ties_win,
                })

    dips = [c["dip_depth"] for c in cells]
    wins = [c["ties_win"] for c in cells]
    rho = spearmanr(dips, wins)
    r = pearsonr(dips, wins)

    # Per-task break-out
    per_task: dict[str, dict] = {}
    for task in TASKS:
        sub = [c for c in cells if c["task"] == task]
        d_t = [c["dip_depth"] for c in sub]
        w_t = [c["ties_win"] for c in sub]
        per_task[task] = {
            "n": len(sub),
            "spearman": spearmanr(d_t, w_t) if len(sub) >= 4 else None,
            "pearson": pearsonr(d_t, w_t) if len(sub) >= 4 else None,
            "mean_dip": sum(d_t) / len(d_t),
            "mean_ties_win": sum(w_t) / len(w_t),
        }

    # Per-model break-out
    per_model: dict[str, dict] = {}
    for model in MODELS:
        sub = [c for c in cells if c["model"] == model]
        d_m = [c["dip_depth"] for c in sub]
        w_m = [c["ties_win"] for c in sub]
        per_model[model] = {
            "n": len(sub),
            "spearman": spearmanr(d_m, w_m) if len(sub) >= 4 else None,
            "pearson": pearsonr(d_m, w_m) if len(sub) >= 4 else None,
            "mean_dip": sum(d_m) / len(d_m),
            "mean_ties_win": sum(w_m) / len(w_m),
        }

    out = {
        "n_cells": len(cells),
        "overall_spearman": rho,
        "overall_pearson": r,
        "per_task": per_task,
        "per_model": per_model,
        "decision_rule": {
            "threshold": 0.7,
            "result": "CONFIRMED" if rho >= 0.7 else
                      "PARTIAL" if rho >= 0.4 else "REJECTED",
        },
        "cells": cells,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    print(f"\nE7 Phase 1 — TVQ b=2 dip vs TIES win across "
          f"{len(cells)} (model, task, seed) cells")
    print(f"  overall Spearman rho = {rho:+.3f}")
    print(f"  overall Pearson r    = {r:+.3f}")
    print(f"  decision (threshold 0.7): {out['decision_rule']['result']}")
    print()
    print(f"{'task':<14}{'n':<4}{'rho':<8}{'r':<8}{'mean_dip':<12}{'mean_ties_win':<12}")
    for task in TASKS:
        s = per_task[task]
        rho_t = s["spearman"] if s["spearman"] is not None else float("nan")
        r_t = s["pearson"] if s["pearson"] is not None else float("nan")
        print(f"{task:<14}{s['n']:<4}{rho_t:<+8.3f}{r_t:<+8.3f}"
              f"{s['mean_dip']:<+12.4f}{s['mean_ties_win']:<+12.4f}")
    print()
    print(f"{'model':<14}{'n':<4}{'rho':<8}{'r':<8}{'mean_dip':<12}{'mean_ties_win':<12}")
    for model in MODELS:
        s = per_model[model]
        rho_m = s["spearman"] if s["spearman"] is not None else float("nan")
        r_m = s["pearson"] if s["pearson"] is not None else float("nan")
        print(f"{model:<14}{s['n']:<4}{rho_m:<+8.3f}{r_m:<+8.3f}"
              f"{s['mean_dip']:<+12.4f}{s['mean_ties_win']:<+12.4f}")
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

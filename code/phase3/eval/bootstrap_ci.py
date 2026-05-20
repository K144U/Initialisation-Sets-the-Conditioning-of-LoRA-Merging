"""Bootstrap confidence intervals for the v3 eval matrix.

Each v3 cell JSON stores per-example NLL arrays:
  per_example_nll_merged[task]            = [{"nll", "n_answer", "skipped"}, ...]
  per_example_nll_tau[state][task]        = same, for each tau-adapter state

The cell's reported metric is token-weighted excess per task:
  excess_task = wmean(merged on task) - wmean(tau_task's-own-adapter on task)
where wmean(x) = Σ_i nll_i * n_answer_i / Σ_i n_answer_i.

worst_task_excess = max_task excess_task.

We bootstrap by resampling EXAMPLE INDICES with replacement, independently per
task (the 4 tasks are disjoint eval sets), recomputing token-weighted excess
per task, and taking the max for worst_task_excess. B resamples → percentile CIs.

merged and tau are evaluated on the SAME examples per task (same tokenization),
so per-example excess is paired: we resample a shared index set per task and
apply it to both the merged and tau arrays.

Output: JSON with point estimate + [2.5, 97.5] CI for every per-task excess and
the worst_task_excess of every cell. CPU-only, ~couple minutes for 40 cells.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
TASKS = ["gsm8k", "alpaca", "magicoder", "translation"]


def _arrays(per_example_list: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (nll, n_answer) float arrays over NON-skipped examples, in order."""
    nll, na = [], []
    for e in per_example_list:
        if e.get("skipped"):
            continue
        nll.append(e["nll"])
        na.append(e["n_answer"])
    return np.asarray(nll, dtype=np.float64), np.asarray(na, dtype=np.float64)


def _wmean(nll: np.ndarray, na: np.ndarray) -> float:
    s = na.sum()
    return float((nll * na).sum() / s) if s > 0 else float("nan")


def bootstrap_cell(cell: dict, n_boot: int, rng: np.random.Generator) -> dict:
    pe_merged = cell.get("per_example_nll_merged")
    pe_tau = cell.get("per_example_nll_tau")
    if pe_merged is None or pe_tau is None:
        return {"error": "no per-example arrays in this cell"}

    # Point estimates + bootstrap distributions per task
    boot_excess = {t: None for t in TASKS}
    point_excess = {}
    for t in TASKS:
        if t not in pe_merged or t not in pe_tau.get(t, {}):
            continue
        m_nll, m_na = _arrays(pe_merged[t])
        tau_nll, tau_na = _arrays(pe_tau[t][t])
        # merged and tau are on the same examples; align to the shorter length
        # (skipped-example filtering should be identical, but guard anyway)
        n = min(len(m_nll), len(tau_nll))
        m_nll, m_na = m_nll[:n], m_na[:n]
        tau_nll, tau_na = tau_nll[:n], tau_na[:n]

        point_excess[t] = _wmean(m_nll, m_na) - _wmean(tau_nll, tau_na)

        # Bootstrap: resample shared indices, recompute paired excess
        idx = rng.integers(0, n, size=(n_boot, n))
        # vectorized token-weighted means over resampled sets
        m_num = (m_nll[idx] * m_na[idx]).sum(axis=1)
        m_den = m_na[idx].sum(axis=1)
        tau_num = (tau_nll[idx] * tau_na[idx]).sum(axis=1)
        tau_den = tau_na[idx].sum(axis=1)
        boot_excess[t] = (m_num / m_den) - (tau_num / tau_den)

    # worst_task_excess bootstrap = elementwise max across tasks' bootstrap draws
    valid_tasks = [t for t in TASKS if boot_excess[t] is not None]
    boot_matrix = np.stack([boot_excess[t] for t in valid_tasks], axis=1)  # (n_boot, n_tasks)
    boot_worst = boot_matrix.max(axis=1)

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    out = {
        "point_worst_task_excess": float(max(point_excess.values())),
        "boot_worst_task_excess_mean": float(boot_worst.mean()),
        "boot_worst_task_excess_ci95": ci(boot_worst),
        "per_task": {
            t: {
                "point_excess": float(point_excess[t]),
                "boot_mean": float(boot_excess[t].mean()),
                "ci95": ci(boot_excess[t]),
                "n_examples": int(len(_arrays(pe_merged[t])[0])),
            }
            for t in valid_tasks
        },
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path,
                   default=PROJECT_ROOT / "results/phase3/eval_matrix_n1k_v3_perexample")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "results/phase3/bootstrap_ci_v3.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260520)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cells = sorted(args.eval_dir.glob("*.json"))
    print(f"[bootstrap] {len(cells)} cells in {args.eval_dir}", flush=True)

    results = {}
    for cell_path in cells:
        cell = json.loads(cell_path.read_text())
        name = cell_path.stem
        res = bootstrap_cell(cell, args.n_boot, rng)
        results[name] = res
        if "error" in res:
            print(f"  {name:<32} ERROR: {res['error']}", flush=True)
        else:
            lo, hi = res["boot_worst_task_excess_ci95"]
            print(f"  {name:<32} worst={res['point_worst_task_excess']:.4f}  "
                  f"95%CI=[{lo:.4f}, {hi:.4f}]", flush=True)

    args.out.write_text(json.dumps({"params": vars(args), "cells": results},
                                   indent=2, default=str))
    print(f"\n[bootstrap] wrote {args.out}", flush=True)
    print("BOOTSTRAP_CI_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

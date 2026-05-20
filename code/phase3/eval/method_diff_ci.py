"""Paired bootstrap CI for method-vs-method per-task differences.

For two merging methods A, B evaluated on the SAME task eval set, the
difference in excess simplifies because the shared tau baseline cancels:

    excess_A(task) - excess_B(task)
      = (merged_A_nll - tau_nll) - (merged_B_nll - tau_nll)
      = merged_A_nll - merged_B_nll   (token-weighted, paired per example)

So we load per_example_nll_merged[task] from cell A and cell B, pair by
example index, and bootstrap the token-weighted-mean difference. A 95% CI
that excludes 0 means A and B differ significantly on that (model, task).

Default comparison: knots vs task_arithmetic (tests F4 / criterion C3).
Negative diff = KnOTS has lower excess = KnOTS better.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
MODELS = ["llama31_8b", "qwen25_7b", "mistral_7b", "yi15_9b"]
TASKS = ["gsm8k", "alpaca", "magicoder", "translation"]


def arrays(pe_list):
    nll, na = [], []
    for e in pe_list:
        if e.get("skipped"):
            continue
        nll.append(e["nll"]); na.append(e["n_answer"])
    return np.asarray(nll, np.float64), np.asarray(na, np.float64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path,
                   default=PROJECT_ROOT / "results/phase3/eval_matrix_n1k_v3_perexample")
    p.add_argument("--method-a", default="knots")
    p.add_argument("--method-b", default="task_arithmetic")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "results/phase3/method_diff_ci_v3.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260520)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    results = {}
    n_sig = 0
    n_total = 0
    print(f"[method_diff] {args.method_a} vs {args.method_b}  (negative = {args.method_a} better)\n", flush=True)
    for model in MODELS:
        ca = json.loads((args.eval_dir / f"{model}__{args.method_a}.json").read_text())
        cb = json.loads((args.eval_dir / f"{model}__{args.method_b}.json").read_text())
        results[model] = {}
        for task in TASKS:
            a_nll, a_na = arrays(ca["per_example_nll_merged"][task])
            b_nll, b_na = arrays(cb["per_example_nll_merged"][task])
            n = min(len(a_nll), len(b_nll))
            a_nll, a_na, b_nll, b_na = a_nll[:n], a_na[:n], b_nll[:n], b_na[:n]
            point = (a_nll * a_na).sum() / a_na.sum() - (b_nll * b_na).sum() / b_na.sum()
            idx = rng.integers(0, n, size=(args.n_boot, n))
            a_wm = (a_nll[idx] * a_na[idx]).sum(1) / a_na[idx].sum(1)
            b_wm = (b_nll[idx] * b_na[idx]).sum(1) / b_na[idx].sum(1)
            diff = a_wm - b_wm
            lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            sig = (lo > 0) or (hi < 0)
            better = args.method_a if point < 0 else args.method_b
            n_total += 1
            if sig:
                n_sig += 1
            results[model][task] = {
                "point_diff": float(point), "ci95": [lo, hi],
                "significant": bool(sig), "better": better if sig else "tie",
            }
            tag = "SIG" if sig else "ns "
            print(f"  {model:<12} {task:<12} diff={point:+.4f} CI=[{lo:+.4f},{hi:+.4f}] {tag} "
                  f"{'('+better+')' if sig else '(within noise)'}", flush=True)
        print()

    args.out.write_text(json.dumps({"params": {k: str(v) for k, v in vars(args).items()},
                                    "results": results,
                                    "n_significant": n_sig, "n_total": n_total}, indent=2))
    print(f"[method_diff] {n_sig}/{n_total} (model,task) cells show a SIGNIFICANT {args.method_a} vs {args.method_b} difference", flush=True)
    print(f"[method_diff] wrote {args.out}", flush=True)
    print("METHOD_DIFF_CI_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

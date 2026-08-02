#!/usr/bin/env python3
"""W5 verdict: what the scorer fixes change in the downstream table.

Prints the published table, the re-scored table, and the delta, plus the two
diagnostics that motivated the re-run (extraction-failure rate, empty-completion
rate). Also recomputes the Spearman correlations behind the paper's "7 of 8
cells positive" claim on both scorings.

Decision rules, fixed before the cells run:

  D1  If the (Llama-3.1, GSM8K EM) cell's Spearman rho turns positive under the
      fixed scorer, the paper's one "unexplained outlier" was an artifact.
      Limitation (5) and the H1/H2/H3 falsification appendix come out, and W5
      is answered rather than conceded.

  D2  If rd-ridge's across-seed SD on its two unstable cells drops below 0.03
      (published: 0.074 Llama GSM8K, 0.084 Mistral HumanEval), the
      "norm-amplifying construction yields degenerate greedy generations"
      explanation was wrong and should be retracted with the rest.

  D3  If the HumanEval spread between {TA,DARE,KnOTS} and {TIES,TVQ2,rd-ridge}
      narrows from the published 2-40x to under 10x, the "far more deployable
      code merges" sentence must be rewritten.

Usage:  python code/phase3/scripts/analyze_w5_rescore.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SEEDS = ["seed1", "seed2", "seed3"]
METHODS = ["ta", "ties", "dare", "knots", "tvq_b2", "rd_ridge"]
# The paper computes Spearman across the FIVE matrix baselines only
# (Table 2 caption), excluding rd-ridge, which is selected on NLL. Match that
# definition exactly or the before/after comparison is not apples-to-apples.
RHO_METHODS = ["ta", "ties", "dare", "knots", "tvq_b2"]
NEW = "eval_downstream_v2"

# (metric label, published dir for the 5 matrix methods, published dir for
#  rd-ridge, filename infix)
METRICS = [
    ("GSM8K em", "eval_e3_gsm8k_seed", "eval_e3b_gsm8k_rdridge_seed", "gsm8k_em"),
    ("HumanEval pass@1", "eval_b4_humaneval_seed",
     "eval_b4b_humaneval_rdridge_seed", "humaneval"),
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def cell_files(base: str, method: str, infix: str, pub5: str, pubrd: str,
               new: bool) -> list[Path]:
    d = RES / (NEW if new else (pubrd if method == "rd_ridge" else pub5))
    return [d / f"{base}__{method}__{infix}__{s}.json" for s in SEEDS]


def score_and_diag(paths: list[Path]) -> tuple[float | None, float | None, int]:
    """Returns (mean score, mean failure rate, n seeds found).

    Failure rate is extraction failure for GSM8K (pred is None) and empty
    completion for HumanEval, i.e. the fraction of generations the harness
    threw away.
    """
    scores, fails = [], []
    for p in paths:
        d = load(p)
        if d is None:
            continue
        scores.append(float(d["metric_score"]))
        pe = d.get("per_example", [])
        if not pe:
            continue
        if "pred" in pe[0]:
            fails.append(sum(1 for e in pe if e.get("pred") is None) / len(pe))
        elif "completion_preview" in pe[0]:
            fails.append(sum(1 for e in pe
                             if not (e.get("completion_preview") or "").strip())
                         / len(pe))
    return (statistics.mean(scores) if scores else None,
            statistics.mean(fails) if fails else None,
            len(scores))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def nll_excess(base: str, method: str) -> float | None:
    """3-seed worst-task NLL excess for the Spearman correlation."""
    if method == "rd_ridge":
        for d, pat in [("eval_seed_rdridge_regmean", "{b}__rd_ridge__{s}.json"),
                       ("eval_ridge_seed", "{b}__ridge_l0p05__{s}.json")]:
            v = [load(RES / d / pat.format(b=base, s=s)) for s in SEEDS]
            v = [float(x["worst_task_excess"]) for x in v if x]
            if v:
                return statistics.mean(v)
        return None
    name = "task_arithmetic" if method == "ta" else method
    v = [load(RES / "eval_matrix_seeds" / f"{base}__{name}__{s}.json") for s in SEEDS]
    v = [float(x["worst_task_excess"]) for x in v if x]
    return statistics.mean(v) if v else None


def main() -> int:
    any_new = False
    for label, pub5, pubrd, infix in METRICS:
        print("=" * 92)
        print(f"{label}: published -> re-scored (3-seed means)")
        print("=" * 92)
        print(f"{'base':<13}" + "".join(f"{m:>13}" for m in METHODS)
              + f"{'rho pub':>9}{'rho new':>9}")
        for base in BASES:
            row, pub_s, new_s, nlls = f"{base:<13}", [], [], []
            for m in METHODS:
                p, _, _ = score_and_diag(cell_files(base, m, infix, pub5, pubrd, False))
                q, _, nq = score_and_diag(cell_files(base, m, infix, pub5, pubrd, True))
                any_new = any_new or nq > 0
                if p is None:
                    row += f"{'--':>13}"
                elif q is None:
                    row += f"{p:>8.3f}{'  --':>5}"
                else:
                    row += f"{p:>6.3f}->{q:<6.3f}"
                x = nll_excess(base, m)
                if m in RHO_METHODS and p is not None and x is not None:
                    pub_s.append(p); nlls.append(-x)
                    new_s.append(q if q is not None else p)
            rp = spearman(nlls, pub_s) if len(pub_s) >= 3 else None
            rn = spearman(nlls, new_s) if len(new_s) >= 3 else None
            row += (f"{rp:>+9.2f}" if rp is not None else f"{'--':>9}")
            row += (f"{rn:>+9.2f}" if rn is not None else f"{'--':>9}")
            print(row)

        print(f"\n  harness discard rate (extraction failure / empty completion)")
        print(f"  {'base':<13}" + "".join(f"{m:>13}" for m in METHODS))
        for base in BASES:
            row = f"  {base:<13}"
            for m in METHODS:
                _, fp, _ = score_and_diag(cell_files(base, m, infix, pub5, pubrd, False))
                _, fq, _ = score_and_diag(cell_files(base, m, infix, pub5, pubrd, True))
                if fp is None:
                    row += f"{'--':>13}"
                elif fq is None:
                    row += f"{fp:>8.2f}{'  --':>5}"
                else:
                    row += f"{fp:>6.2f}->{fq:<6.2f}"
            print(row)
        print()

    # D2: seed stability on rd-ridge's two published outlier cells.
    print("=" * 92)
    print("D2  rd-ridge across-seed SD on its two published unstable cells")
    print("=" * 92)
    for base, label, pub5, pubrd, infix, pubsd in [
        ("llama31_8b", "GSM8K em", "eval_e3_gsm8k_seed",
         "eval_e3b_gsm8k_rdridge_seed", "gsm8k_em", 0.074),
        ("mistral_7b", "HumanEval", "eval_b4_humaneval_seed",
         "eval_b4b_humaneval_rdridge_seed", "humaneval", 0.084),
    ]:
        vals = []
        for p in cell_files(base, "rd_ridge", infix, pub5, pubrd, True):
            d = load(p)
            if d:
                vals.append(float(d["metric_score"]))
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        verdict = ("--" if sd is None
                   else "artifact" if sd < 0.03 else "real instability")
        print(f"  {base:<13}{label:<12} published SD {pubsd:.3f}  "
              f"re-scored SD {sd if sd is not None else float('nan'):.3f}  {verdict}")

    if not any_new:
        print("\n[analyze] no eval_downstream_v2 cells found yet; "
              "showing published values only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

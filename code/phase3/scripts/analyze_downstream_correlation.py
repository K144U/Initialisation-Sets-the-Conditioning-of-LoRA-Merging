"""R10: does worst-task NLL excess predict downstream accuracy?

Limitation 2 of the paper says the available Spearman correlations are computed
over the five or six methods in a row, that SD(rho) = 1/sqrt(n-1) = 0.45 at
n = 6, and that every value is within about 1.5 SD of zero. It then draws no
conclusion, which is correct at n = 6.

The referee's point is that the correlation does not have to be computed within
a row. There are 4 bases x 3 cohorts = 12 blocks of 6 methods each in
eval_downstream_v2, so the block structure raises the effective sample size at
zero compute cost.

Two estimators, both reported:

  1. BLOCK-WISE. Spearman within each (base, cohort) block over its 6 methods,
     then the mean over 12 blocks with SE = sd/sqrt(12). This is the honest
     one: blocks are independent draws, and the within-block correlation is
     what "does the metric rank methods correctly" actually means.
  2. POOLED. Rank within block, pool all 72 pairs, correlate. Reported with a
     block bootstrap CI, because the 72 pairs are not independent.

Sign convention: lower NLL excess should mean higher accuracy, so the predicted
correlation is NEGATIVE. A positive correlation would mean the metric ranks
methods backwards.

One confound is stated rather than hidden: the rd_ridge NLL cells are
realize=rank_deff (rank 64) while every other method is rank 16, which is audit
finding A3. The correlation is therefore also reported with rd_ridge dropped.

Usage:  python code/phase3/scripts/analyze_downstream_correlation.py [results/phase3]
"""
import json
import random
import statistics
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / "results" / "phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
COHORTS = ["seed1", "seed2", "seed3"]
METRICS = ["gsm8k_em", "humaneval"]
# downstream name -> (nll directory, nll method name)
METHODS = {
    "ta": ("eval_matrix_seeds", "task_arithmetic"),
    "ties": ("eval_matrix_seeds", "ties"),
    "dare": ("eval_matrix_seeds", "dare"),
    "tvq_b2": ("eval_matrix_seeds", "tvq_b2"),
    "knots": ("eval_matrix_seeds", "knots"),
    "rd_ridge": ("eval_seed_rdridge_regmean", "rd_ridge"),
}
RANK_CONFOUNDED = "rd_ridge"   # audit A3: rank 64 against everyone else's 16


def load(path, key):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(key)


def ranks(xs):
    """Average ranks, so ties do not shift the correlation."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def collect(root, metric, drop=()):
    """(base, cohort) -> [(method, nll_excess, score)]"""
    blocks, missing = {}, []
    for base in BASES:
        for cohort in COHORTS:
            rows = []
            for dname, (nll_dir, nll_method) in METHODS.items():
                if dname in drop:
                    continue
                nll = load(root / nll_dir / f"{base}__{nll_method}__{cohort}.json",
                           "worst_task_excess")
                score = load(
                    root / "eval_downstream_v2" /
                    f"{base}__{dname}__{metric}__{cohort}.json", "metric_score")
                if nll is None or score is None:
                    missing.append(f"{base}/{cohort}/{dname}/{metric}")
                    continue
                rows.append((dname, nll, score))
            if len(rows) >= 3:
                blocks[(base, cohort)] = rows
    return blocks, missing


def report(root, metric, drop=()):
    blocks, missing = collect(root, metric, drop)
    label = metric + ("" if not drop else f"  (without {', '.join(drop)})")
    print("=" * 100)
    print(f"{label}: {len(blocks)} blocks, "
          f"{sum(len(v) for v in blocks.values())} pairs")
    if missing:
        print(f"  {len(missing)} cells missing, e.g. {missing[:3]}")
    if not blocks:
        return
    print("=" * 100)

    rhos = []
    for (base, cohort), rows in sorted(blocks.items()):
        r = spearman([x[1] for x in rows], [x[2] for x in rows])
        rhos.append(r)
        print(f"  {base:<13}{cohort:<9} n={len(rows)}  rho = {r:+.3f}")

    mean = statistics.fmean(rhos)
    sd = statistics.stdev(rhos) if len(rhos) > 1 else float("nan")
    se = sd / len(rhos) ** 0.5
    print(f"\n  BLOCK-WISE   mean rho {mean:+.3f}   sd {sd:.3f}   "
          f"SE {se:.3f}   mean/SE = {mean / se:+.2f}")
    print(f"               95% interval {mean - 1.96 * se:+.3f} to "
          f"{mean + 1.96 * se:+.3f}")
    print(f"               at n=6 in a single row the SE would be "
          f"{1 / (6 - 1) ** 0.5:.3f}, so this is "
          f"{(1 / (6 - 1) ** 0.5) / se:.1f}x tighter")

    # pooled, ranked within block
    px, py = [], []
    for rows in blocks.values():
        px += ranks([x[1] for x in rows])
        py += ranks([x[2] for x in rows])
    pooled = pearson(px, py)
    keys = sorted(blocks)
    rng = random.Random(20260814)
    boot = []
    for _ in range(2000):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        bx, by = [], []
        for k in pick:
            rows = blocks[k]
            bx += ranks([x[1] for x in rows])
            by += ranks([x[2] for x in rows])
        boot.append(pearson(bx, by))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    print(f"\n  POOLED       rho {pooled:+.3f}   block-bootstrap 95% "
          f"{lo:+.3f} to {hi:+.3f}")

    crosses = lo <= 0 <= hi
    print(f"\n  VERDICT: the interval {'includes' if crosses else 'excludes'} "
          f"zero.")
    if not crosses and pooled < 0:
        print("  The metric predicts downstream accuracy in the expected "
              "direction.")
    elif not crosses:
        print("  The metric predicts downstream accuracy BACKWARDS, which is "
              "worse than not predicting it.")
    else:
        print("  Still no detectable relationship, now at 12 blocks rather "
              "than one row.")
    print()


def main(root):
    print("R10: NLL excess against downstream accuracy, pooled across blocks")
    print("Predicted sign is NEGATIVE: lower excess should mean higher score.\n")
    for metric in METRICS:
        report(root, metric)
        report(root, metric, drop=(RANK_CONFOUNDED,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

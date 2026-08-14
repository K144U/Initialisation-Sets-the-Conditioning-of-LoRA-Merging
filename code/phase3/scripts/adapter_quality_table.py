"""Table 9 of the paper: per-adapter task quality, from existing eval cells.

Companion to `adapter_quality.py`, which reads `eval_a1_indep` on the cluster.
This one reconstructs the same table from whichever canonical cohort
directories are present, so it also runs on a partial mirror of `results/`.
It prints the appendix table (App. C.3) and the two summary statements made
around it.

`nll_tau` is a property of the adapters, not of the merge, so any cell of a
given (base, cohort) carries the same block. Two directories are deliberately
excluded:

  eval_e6_pilot    a different task set (dolly, not alpaca)
  eval_e11*        indexed by merge alpha, not by cohort
  eval_ridge_cond  a different evaluation draw: its nll_tau disagrees with the
                   canonical cells by up to 0.02 nats on the same nominal
                   cohort, which on Yi is enough to change which adapter is
                   best on the translation task. Do not average the two.

The same caveat applies inside the canonical set, at smaller scale: the
llama31_8b seed3 cells in `eval_e1_seed` sit on a different evaluation draw
from the ones in `eval_matrix_seeds` (0.9912 against 0.9732 nats for the
alpaca specialist). DIRS order is therefore load-bearing, not cosmetic: the
first directory listed that holds a given (base, cohort) is the one used, so
one draw is picked and stuck to. The consistency check below prints any such
disagreement rather than hiding it in a mean.

Usage:  python adapter_quality_table.py [results/phase3]
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / "results" / "phase3"
DIRS = ["eval_matrix_seeds", "eval_density_sweep", "eval_dare_ties",
        "eval_e1_seed", "eval_e1"]
ARMS = [("shared", ["seed1", "seed2", "seed3"]),
        ("independent", ["indep1", "indep2", "indep3"])]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
CELL = re.compile(r"([a-z0-9_]+?)__(.+)__([a-z0-9]+)\.json$")


def collect(root):
    """(base, cohort) -> one nll_tau block, plus every source seen."""
    found = defaultdict(list)
    for name in DIRS:
        for path in (root / name).glob("*.json"):
            if path.name.endswith("_summary.json"):
                continue
            m = CELL.match(path.name)
            if not m:
                continue
            try:
                cell = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if isinstance(cell, dict) and "nll_tau" in cell:
                base, _, cohort = m.groups()
                found[(base, cohort)].append(cell["nll_tau"])
    return found


def check_consistency(found, tol=0.002):
    """Every cell of one (base, cohort) must agree, or the mean is meaningless."""
    worst, offenders = 0.0, []
    for key, blocks in sorted(found.items()):
        for task in blocks[0]["base"]:
            vals = [b[task][task] for b in blocks]
            spread = max(vals) - min(vals)
            worst = max(worst, spread)
            if spread > tol:
                offenders.append((key, task, spread, len(blocks)))
    return worst, offenders


def main(root):
    found = collect(root)
    if not found:
        sys.exit(f"no usable cells under {root}")

    worst, offenders = check_consistency(found)
    print(f"nll_tau agreement within each (base, cohort): max spread "
          f"{worst:.5f} nats")
    for key, task, spread, n in offenders:
        print(f"  DISAGREE {key} {task}: {spread:.4f} over {n} cells")

    gains, usurped = [], []
    for arm, cohorts in ARMS:
        print(f"\n### {arm}  ({', '.join(cohorts)})")
        print(f"{'base':<12}{'task':<12}{'base':>8}{'spec':>8}{'gain%':>7}"
              f"   best adapter on this task")
        for base in BASES:
            blocks = [found[(base, c)][0] for c in cohorts if found.get((base, c))]
            if not blocks:
                print(f"{base:<12} no cells")
                continue
            if len(blocks) != len(cohorts):
                print(f"{base:<12} only {len(blocks)} of {len(cohorts)} cohorts")
            for task in blocks[0]["base"]:
                bn = statistics.fmean(b["base"][task] for b in blocks)
                sn = statistics.fmean(b[task][task] for b in blocks)
                per_ad = {a: statistics.fmean(b[a][task] for b in blocks)
                          for a in blocks[0]["base"]}
                best = min(per_ad, key=per_ad.get)
                pct = 100 * (bn - sn) / bn
                if best == task:
                    gains.append(pct)
                    note = ""
                else:
                    usurped.append((arm, base, task, best, sn - per_ad[best]))
                    note = f", by {sn - per_ad[best]:.3f}"
                print(f"{base:<12}{task:<12}{bn:>8.4f}{sn:>8.4f}{pct:>6.1f}%"
                      f"   {best}{note}")

    print(f"\nspecialist best on its own task in {len(gains)} cells, "
          f"gaining {min(gains):.1f}% to {max(gains):.1f}% over base")
    print(f"specialist beaten on its own task in {len(usurped)} cells:")
    for arm, base, task, best, gap in usurped:
        print(f"  {arm:<12}{base:<12}{task:<12}beaten by {best} by {gap:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)

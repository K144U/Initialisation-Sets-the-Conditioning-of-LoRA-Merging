"""R4: is the repaired KnOTS a real arm, or task arithmetic under another name?

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), section R4.

  The repaired KnOTS enters the null of R3 as an independent arm if its
  worst-task excess differs from task arithmetic by more than the tie
  threshold on at least 3 of 4 bases, on the shared arm. Otherwise it is
  removed and the null is restated over the 16 cells that remain.

  Either way the DEFAULT-configuration KnOTS comes out of the null: it is
  algebraically task arithmetic and matches it to four decimals, so its four
  cells were the same data point counted twice.

The verdict is read on the shared arm, as registered. The independent arm is
reported alongside because it exists now and because a method that separates
from task arithmetic on one arm and not the other would itself be a finding.

Usage:  python code/phase3/scripts/analyze_r4_knots.py [results/phase3]
"""
import json
import statistics
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / "results" / "phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TIE = 0.005
ARMS = [
    ("shared", ["seed1", "seed2", "seed3"],
     "eval_a2_knots_ties", "eval_matrix_seeds"),
    ("independent", ["indep1", "indep2", "indep3"],
     "eval_a2_knots_ties_indep", "eval_a1_indep"),
]


def vals(root, subdir, base, method, cohorts):
    out = []
    for c in cohorts:
        p = root / subdir / f"{base}__{method}__{c}.json"
        if not p.exists():
            return None
        out.append(json.loads(p.read_text())["worst_task_excess"])
    return out


def main(root):
    verdicts = {}
    for arm, cohorts, kdir, tadir in ARMS:
        print("=" * 96)
        print(f"{arm} arm ({', '.join(cohorts)})")
        print("=" * 96)
        print(f"{'base':<13}{'knots_ties':>12}{'task arith':>12}{'diff':>10}"
              f"{'2xSE':>9}   separates?")
        n_sep = 0
        complete = True
        for base in BASES:
            k = vals(root, kdir, base, "knots_ties", cohorts)
            t = vals(root, tadir, base, "task_arithmetic", cohorts)
            if k is None or t is None:
                print(f"{base:<13} INCOMPLETE")
                complete = False
                continue
            d = [a - b for a, b in zip(k, t)]
            md = statistics.fmean(d)
            se2 = 2 * statistics.stdev(d) / len(d) ** 0.5
            sep = abs(md) > TIE
            n_sep += sep
            print(f"{base:<13}{statistics.fmean(k):>12.4f}"
                  f"{statistics.fmean(t):>12.4f}{md:>+10.4f}{se2:>9.4f}"
                  f"   {'YES' if sep else 'no'}")
        verdicts[arm] = (n_sep, complete)
        print(f"\n  separates from task arithmetic on {n_sep}/4 bases\n")

    n_shared, complete = verdicts.get("shared", (0, False))
    if not complete:
        print("Shared arm incomplete; no verdict.")
        return 2
    print("=" * 96)
    if n_shared >= 3:
        print("R4 VERDICT: the repaired KnOTS IS a real arm")
        print("  It enters the regime null as an independent method, and the")
        print("  subspace-alignment family is represented in the comparison")
        print("  after all. Discussion limitation 4 is rewritten accordingly.")
    else:
        print("R4 VERDICT: the repaired KnOTS is NOT a separate arm either")
        print("  Even with inner_combination=ties it does not separate from")
        print("  task arithmetic by more than the tie threshold on 3 of 4")
        print("  bases. It stays out of the null, which is restated over 16")
        print("  cells, and the paper says no subspace-alignment method is")
        print("  represented in the comparison. That is a stronger version of")
        print("  the limitation the paper already carries, not a weaker one.")
    print("\n  Default-configuration KnOTS comes out of the null regardless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

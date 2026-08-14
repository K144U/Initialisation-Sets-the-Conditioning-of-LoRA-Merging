"""R3: the regime null, with the noise gate the paper could not previously apply.

Section 7.2 reports twenty base-by-method cells with ONE cohort per arm and
concludes indifference. Three referee objections follow from that: no gate could
be computed, the reported mean difference of 0.0014 nats sits above the paper's
own stated metric resolution of about 0.001 nats, and a null without a minimum
detectable effect is not evidence of absence.

No new cells are needed. eval_matrix_seeds holds the shared arm at seed1/2/3 and
eval_a1_indep holds the independent arm at indep1/2/3, so n = 3 per arm already
exists.

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), section R3.
  effect if |mean difference| > max(0.005, 2 x SE), else no detectable effect
  the minimum detectable effect is reported for every cell whatever the outcome

On the SE. The registration says "SE = sd/sqrt(3)", which is the paired form,
but the cohorts are not paired: seed1 and indep1 are independent draws that
share nothing but an index. The two-sample form sqrt(sd_s^2/3 + sd_i^2/3) is
the correct one and is used as PRIMARY here; the paired form is printed beside
it so a reader can apply the registration literally. Where the two disagree on a
cell's verdict, the cell is reported as disagreeing rather than resolved
silently. The two-sample form is the more conservative of the two, so this
choice cannot manufacture an effect.

R4 also bears on this table: KnOTS in its default configuration is task
arithmetic to four decimals, so its four cells are duplicates and the null is
restated without them.

Usage:  python code/phase3/scripts/analyze_regime_null.py [results/phase3]
"""
import json
import statistics
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / "results" / "phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
METHODS = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots"]
SHARED = ("eval_matrix_seeds", ["seed1", "seed2", "seed3"])
INDEP = ("eval_a1_indep", ["indep1", "indep2", "indep3"])
TIE = 0.005
DUPLICATE = "knots"   # R4: algebraically task arithmetic under the default


def excess(root, subdir, base, method, cohort):
    p = root / subdir / f"{base}__{method}__{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("worst_task_excess")


def main(root):
    rows, missing = [], []
    for base in BASES:
        for method in METHODS:
            s = [excess(root, SHARED[0], base, method, c) for c in SHARED[1]]
            i = [excess(root, INDEP[0], base, method, c) for c in INDEP[1]]
            if any(v is None for v in s + i):
                missing.append(f"{base}__{method}")
                continue
            ms, mi = statistics.fmean(s), statistics.fmean(i)
            ss, si = statistics.stdev(s), statistics.stdev(i)
            diff = mi - ms                      # positive: independent worse
            se2 = (ss ** 2 / len(s) + si ** 2 / len(i)) ** 0.5
            se1 = statistics.stdev([a - b for a, b in zip(i, s)]) / len(s) ** 0.5
            rows.append(dict(base=base, method=method, ms=ms, mi=mi, ss=ss,
                             si=si, diff=diff, se2=se2, se1=se1))

    if missing:
        print(f"INCOMPLETE: {len(missing)} cells missing, no verdict computed")
        for m in missing[:10]:
            print("   ", m)
        return 2

    print("=" * 108)
    print("R3: independent minus shared, worst-task NLL excess, n = 3 per arm")
    print("    positive difference = the independent cohort is WORSE")
    print("=" * 108)
    print(f"{'base':<13}{'method':<17}{'shared':>9}{'indep':>9}{'diff':>9}"
          f"{'2xSE':>9}{'gate':>9}   verdict")

    counts = {"independent better": 0, "independent worse": 0, "no effect": 0}
    counts_nodup = dict(counts)
    disagree = []
    gates, diffs, diffs_nodup = [], [], []
    for r in rows:
        gate = max(TIE, 2 * r["se2"])
        gate1 = max(TIE, 2 * r["se1"])
        if abs(r["diff"]) > gate:
            verdict = "independent worse" if r["diff"] > 0 else "independent better"
        else:
            verdict = "no effect"
        if (abs(r["diff"]) > gate) != (abs(r["diff"]) > gate1):
            disagree.append(f"{r['base']}/{r['method']}")
        counts[verdict] += 1
        gates.append(gate)
        diffs.append(r["diff"])
        if r["method"] != DUPLICATE:
            counts_nodup[verdict] += 1
            diffs_nodup.append(r["diff"])
        mark = "" if (abs(r["diff"]) > gate1) == (abs(r["diff"]) > gate) else "  (*)"
        print(f"{r['base']:<13}{r['method']:<17}{r['ms']:>9.4f}{r['mi']:>9.4f}"
              f"{r['diff']:>+9.4f}{2 * r['se2']:>9.4f}{gate:>9.4f}   "
              f"{verdict}{mark}")

    n = len(rows)
    print()
    print("=" * 108)
    print(f"VERDICT over all {n} cells")
    print("=" * 108)
    for k, v in counts.items():
        print(f"  {k:<20}{v:>3} of {n}")
    print(f"  mean difference     {statistics.fmean(diffs):>+8.4f} nats")
    print(f"  minimum detectable effect: median {statistics.median(gates):.4f}, "
          f"range {min(gates):.4f} to {max(gates):.4f} nats")

    m = len(diffs_nodup)
    print()
    print(f"R4: the same table with {DUPLICATE} removed as a duplicate of task")
    print(f"    arithmetic, leaving {m} independent cells")
    for k, v in counts_nodup.items():
        print(f"  {k:<20}{v:>3} of {m}")
    print(f"  mean difference     {statistics.fmean(diffs_nodup):>+8.4f} nats")

    if disagree:
        print()
        print("Cells where the paired and two-sample gates disagree, reported "
              "rather than resolved:")
        for d in disagree:
            print("   ", d)

    print()
    print("HOW TO STATE THIS IN THE PAPER:")
    worst_gate = max(gates)
    print(f"  Not \"no effect\". The design can detect a difference of "
          f"{statistics.median(gates):.4f} nats in the median cell and "
          f"{worst_gate:.4f} in the worst.")
    print(f"  Anything smaller than that is invisible to it, and the correct")
    print(f"  sentence is \"no effect detectable at {worst_gate:.3f} nats\".")
    print(f"  The 0.0014 nat mean difference the paper currently quotes is "
          f"{'below' if abs(statistics.fmean(diffs)) < statistics.median(gates) else 'above'}")
    print("  the median gate, so it is a number the design cannot resolve, "
          "which is")
    print("  what resolves its tension with the 0.001 nat metric resolution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

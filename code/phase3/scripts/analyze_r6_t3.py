"""R6 analyzer. Committed before any T = 3 merge cell has landed.

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), section R6.

  The primary result remains the T = 4 matrix, as run and as registered on
  2026-08-03. T = 3 is a robustness arm. If any per-base verdict (win, tie,
  loss) differs between T = 4 and T = 3, BOTH are reported in Table 3 and the
  difference is stated in the text; the T = 4 verdict is not replaced.

Two things this analyzer will not do, because they would be wrong.

It does not compare the LEVEL of worst-task excess between T = 4 and T = 3.
The T = 3 maximum is taken over three tasks rather than four, so it is a
different quantity, and a lower number would mean nothing on its own. What
transfers is the VERDICT: which method beats which, by more than the gate.

It does not re-select the champion at T = 3 in a way that hides a flip. The
champion is chosen within each T separately, and if the identity of the
champion changes that is reported as its own finding rather than folded into
the win/tie/loss counts.

Usage:  python code/phase3/scripts/analyze_r6_t3.py
"""
import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path.home() / "projects" / "rdmerge"))
RES = ROOT / "results" / "phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
COHORTS = ["indep1", "indep2", "indep3"]
BASELINES = ["task_arithmetic", "ties", "dare", "tvq_b2"]
ENCODERS = ["rd_ridge", "rd_rank16"]
TIE = 0.005


def vals(subdir, base, method):
    out = []
    for c in COHORTS:
        p = RES / subdir / f"{base}__{method}__{c}.json"
        if not p.exists():
            return None
        out.append(json.loads(p.read_text())["worst_task_excess"])
    return out


def verdict(diffs):
    d = statistics.fmean(diffs)
    gate = max(TIE, 2 * statistics.stdev(diffs) / len(diffs) ** 0.5)
    if abs(d) <= gate:
        return "ties", d, gate
    return ("wins" if d > 0 else "loses"), d, gate


def table(subdir, label, encoder):
    print("=" * 104)
    print(f"{label}   encoder = {encoder}")
    print("=" * 104)
    print(f"{'base':<13}{'encoder':>9}{'champion':>10}  {'name':<18}"
          f"{'d':>9}{'gate':>9}   verdict")
    out = {}
    for base in BASES:
        e = vals(subdir, base, encoder)
        if e is None:
            print(f"{base:<13} INCOMPLETE")
            out[base] = None
            continue
        champ, cv = None, None
        for m in BASELINES:
            v = vals(subdir, base, m)
            if v is None:
                continue
            if cv is None or statistics.fmean(v) < statistics.fmean(cv):
                champ, cv = m, v
        if cv is None:
            print(f"{base:<13} INCOMPLETE (no baseline)")
            out[base] = None
            continue
        diffs = [c - x for c, x in zip(cv, e)]
        v, d, gate = verdict(diffs)
        out[base] = (v, champ, d)
        print(f"{base:<13}{statistics.fmean(e):>9.4f}{statistics.fmean(cv):>10.4f}"
              f"  {champ:<18}{d:>+9.4f}{gate:>9.4f}   {v}")
    print()
    return out


def main():
    for encoder in ENCODERS:
        t4 = table("eval_a1_indep", "T = 4, as published", encoder)
        t3 = table("eval_r6_t3", "T = 3, translation dropped", encoder)
        if any(v is None for v in t3.values()):
            print("T = 3 incomplete; no comparison drawn.\n")
            continue

        print("-" * 104)
        print(f"VERDICT COMPARISON for {encoder}")
        flips, champ_changes = [], []
        for base in BASES:
            if t4.get(base) is None:
                continue
            v4, c4, d4 = t4[base]
            v3, c3, d3 = t3[base]
            if v4 != v3:
                flips.append(f"{base}: {v4} at T=4 -> {v3} at T=3 "
                             f"(d {d4:+.4f} -> {d3:+.4f})")
            if c4 != c3:
                champ_changes.append(f"{base}: champion {c4} -> {c3}")
        if flips:
            print("  VERDICTS DIFFER. Both go in Table 3 and the text says so;")
            print("  the T = 4 verdict is NOT replaced (registered rule).")
            for f in flips:
                print("   ", f)
        else:
            print("  No per-base verdict changes. The performance conclusions")
            print("  are robust to dropping the adapter that never learned,")
            print("  which is what 6.6 currently shows only for the geometry.")
        if champ_changes:
            print("  Champion identity changes, reported separately:")
            for c in champ_changes:
                print("   ", c)
        print()

    print("NOTE: levels are not compared across T. The T = 3 worst-task excess")
    print("is a maximum over three tasks, not four, so it is a different")
    print("quantity. Only verdicts transfer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

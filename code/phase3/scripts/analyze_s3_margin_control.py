#!/usr/bin/env python3
"""S3: the margin-aware stability control.

Rules: notes/prereg_tmlr_2026-08-14.md (0c8b9e8), section S3. Registered with
the indep1/2/3 matrix already visible, which the registration states about
itself and which this script repeats in its output. It is a secondary control
and it cannot delete or replace the primary Q4 verdict.

S3 as registered, verbatim in the rule it fixes:

    A base counts as unstable only if its top-1 method changes across cohorts
    AND the margin between the top two methods exceeds the tie threshold of
    0.005 nats in at least one cohort where the change occurs.

    If at most 1 of 4 bases is unstable under this rule, the
    benchmark-confounding claim is reinstated as PROVISIONAL, with both the
    original margin-blind verdict and this one reported side by side, and with
    the original identified as the one registered first.

    Under no outcome is the original verdict deleted or replaced.

Same three cohorts, same seven methods, same top-1 comparison and same source
directories as the Q4 control in analyze_a1_matrix_3cohort.py. No new cells.

Usage:
  python code/phase3/scripts/analyze_s3_margin_control.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BASELINES = ["task_arithmetic", "ties", "dare", "knots", "tvq_b2"]
ALL7 = BASELINES + ["rd_ridge", "rd_rank16"]
COHORTS = ["indep1", "indep2", "indep3"]
TIE = 0.005


def read_excess(p: Path) -> float | None:
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["worst_task_excess"])
    except Exception:
        return None


def main() -> int:
    print("=" * 92)
    print("S3  margin-aware stability control     rules: prereg_tmlr_2026-08-14 (0c8b9e8)")
    print("=" * 92)
    print("Registered AFTER the indep1/2/3 matrix had been read. It is a secondary")
    print("control and cannot replace the Q4 verdict, which was registered first.")
    print()

    missing = []
    V: dict = {}
    for b in BASES:
        V[b] = {}
        for m in ALL7:
            V[b][m] = {}
            for c in COHORTS:
                v = read_excess(RES / "eval_a1_indep" / f"{b}__{m}__{c}.json")
                V[b][m][c] = v
                if v is None:
                    missing.append(f"{b}__{m}__{c}")
    if missing:
        print(f"MISSING {len(missing)} cells, first five: {missing[:5]}")
        print("NO VERDICT: the control needs the full matrix.")
        return 1

    k_blind = 0     # the original Q4 count, recomputed here for the side-by-side
    k_margin = 0    # the S3 count
    rows = []

    for b in BASES:
        tops, margins = [], {}
        for c in COHORTS:
            vals = {m: V[b][m][c] for m in ALL7}
            order = sorted(vals, key=lambda m: vals[m])
            tops.append(order[0])
            margins[c] = vals[order[1]] - vals[order[0]]

        changed = len(set(tops)) > 1
        k_blind += changed

        # "the margin between the top two exceeds the tie threshold in at
        # least one cohort WHERE THE CHANGE OCCURS". The cohorts where the
        # change occurs are those whose top-1 is not the modal top-1.
        modal = max(set(tops), key=tops.count)
        change_cohorts = [c for c, t in zip(COHORTS, tops) if t != modal]
        decisive = [c for c in change_cohorts if margins[c] > TIE]
        unstable = changed and bool(decisive)
        k_margin += unstable

        rows.append({"base": b, "top1_by_cohort": tops, "changed": changed,
                     "margins": margins, "change_cohorts": change_cohorts,
                     "decisive_cohorts": decisive, "unstable_s3": unstable})

        print(f"{b}")
        print(f"  top-1 by cohort   {', '.join(tops)}")
        print("  top-two margin    " + "  ".join(
            f"{c} {margins[c]:.4f}{'*' if margins[c] > TIE else ''}"
            for c in COHORTS))
        print(f"  Q4 (margin-blind) {'UNSTABLE' if changed else 'stable'}")
        print(f"  S3 (margin-aware) {'UNSTABLE' if unstable else 'stable'}"
              + (f"   decisive in {decisive}" if decisive else
                 ("   every flip is inside the tie threshold" if changed else "")))
        print()

    print("=" * 92)
    print(f"Q4, margin-blind, registered first : {k_blind}/4 bases unstable")
    print(f"S3, margin-aware, registered second: {k_margin}/4 bases unstable")
    print()

    if k_margin <= 1:
        verdict = "REINSTATED AS PROVISIONAL"
        print(f"S3 VERDICT: {verdict}")
        print("  At most 1 of 4 bases is unstable under the margin-aware rule, so by")
        print("  the registered rule the benchmark-confounding claim is reinstated as")
        print("  PROVISIONAL. Both verdicts are reported side by side and the")
        print("  margin-blind one is identified as the one registered first. It is")
        print("  not deleted and not replaced.")
    else:
        verdict = "NOT REINSTATED"
        print(f"S3 VERDICT: {verdict}")
        print("  More than 1 of 4 bases is unstable even after requiring the flip to")
        print("  exceed the tie threshold, so the margin-aware control agrees with the")
        print("  margin-blind one. The claim stays withdrawn.")

    out = RES / "s3_margin_control_summary.json"
    out.write_text(json.dumps(
        {"verdict": verdict, "k_margin_blind_q4": k_blind,
         "k_margin_aware_s3": k_margin, "tie_threshold": TIE,
         "per_base": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the registered rule to the heuristics-downstream cells.

Rules: notes/prereg_heuristics_downstream_2026-08-16.md (8ff7fb4). This file
is committed before any cell of the run lands, so the thresholds below cannot
have been chosen with the data in view.

Registered, and hard-coded here rather than passed in, so that a command-line
flag cannot move a threshold:

  * effect threshold      |d| > 0.05          (5 accuracy points)
  * noise gate            |d| > 2 * SE_binom  (one-directional: may only
                                               downgrade an effect to none)
  * d                     acc_indep - acc_shared
  * P1  at most 4 of 40 cells show an effect
  * P2  no method shows an effect on 3 or more of its 8 cells
  * P3  cells that clear favour the independent arm (secondary, directional)

The four branches of the decision rule are printed with the verdict, so the
consequence is read off the registration and not decided afterwards.

Usage:
  python code/phase3/scripts/analyze_heuristics_downstream.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3" / "eval_heur_downstream"
OUT = ROOT / "results/phase3" / "heur_downstream_summary.json"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
METHODS = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots_ties"]
BENCH = {"gsm8k": 500, "humaneval": 164}

THRESHOLD = 0.05
P1_MAX_EFFECTS = 4
P2_MAX_PER_METHOD = 3


def read_cell(base: str, method: str, bench: str, cohort: str) -> dict | None:
    p = RES / f"{base}__{method}_{bench}__{cohort}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())

    # The score key differs by harness generation; take the registered metric
    # and fail loudly rather than silently reading a missing key as zero.
    score = d.get("metric_score", d.get("metric_value"))
    if score is None:
        raise SystemExit(f"{p.name}: no metric_score in cell")

    # Discard accounting, per binding constraint 4. The two harnesses store
    # per-example records under different schemas; see
    # analyze_downstream_conditioning.py for the same split.
    ex = d.get("per_example") or []
    n_empty = n_fail = 0
    for e in ex:
        if bench == "gsm8k":
            if not str(e.get("gen_text", "")).strip():
                n_empty += 1
            if e.get("pred") is None or not str(e.get("pred", "")).strip():
                n_fail += 1
        else:
            if not str(e.get("gen_raw", e.get("completion_preview", ""))).strip():
                n_empty += 1
    return {"score": float(score), "n": len(ex) or BENCH[bench],
            "n_empty": n_empty, "n_unparsed": n_fail}


def se_binom(p1: float, n1: int, p2: float, n2: int) -> float:
    return math.sqrt(max(p1 * (1 - p1) / max(n1, 1), 0.0)
                     + max(p2 * (1 - p2) / max(n2, 1), 0.0))


def main() -> int:
    rows, missing = [], []
    for base in BASES:
        for method in METHODS:
            for bench in BENCH:
                sh = read_cell(base, method, bench, "seed1")
                ind = read_cell(base, method, bench, "indep1")
                if sh is None or ind is None:
                    missing.append(f"{base}/{method}/{bench}")
                    continue
                d = ind["score"] - sh["score"]
                se = se_binom(ind["score"], ind["n"], sh["score"], sh["n"])
                gate = 2 * se
                effect = abs(d) > THRESHOLD and abs(d) > gate
                rows.append({
                    "base": base, "method": method, "bench": bench,
                    "acc_shared": sh["score"], "acc_indep": ind["score"],
                    "d": d, "gate": gate, "effect": effect,
                    "favours": ("independent" if d > 0 else "shared")
                               if effect else "",
                    "empty_shared": sh["n_empty"], "empty_indep": ind["n_empty"],
                    "unparsed_shared": sh["n_unparsed"],
                    "unparsed_indep": ind["n_unparsed"],
                })

    if missing:
        print(f"INCOMPLETE: {len(missing)} of 40 comparisons missing")
        for m in missing[:12]:
            print("   ", m)
        print("\nBinding constraint 1: no cell is read until all have landed.")
        print("Reporting nothing further.")
        return 1

    n_eff = sum(r["effect"] for r in rows)
    per_method = {m: sum(r["effect"] for r in rows if r["method"] == m)
                  for m in METHODS}
    worst = max(per_method.values())
    p1 = n_eff <= P1_MAX_EFFECTS
    p2 = worst < P2_MAX_PER_METHOD
    shared_favouring = [r for r in rows if r["effect"] and r["d"] < 0]

    print(f"{'base':<13}{'method':<17}{'bench':<11}"
          f"{'shared':>8}{'indep':>8}{'d':>9}{'gate':>8}  verdict")
    for r in sorted(rows, key=lambda r: (not r["effect"], r["base"])):
        print(f"{r['base']:<13}{r['method']:<17}{r['bench']:<11}"
              f"{r['acc_shared']:>8.3f}{r['acc_indep']:>8.3f}"
              f"{r['d']:>+9.3f}{r['gate']:>8.3f}  "
              f"{'EFFECT (' + r['favours'] + ')' if r['effect'] else '-'}")

    print(f"\ncells with a detectable effect: {n_eff}/40")
    for m in METHODS:
        print(f"   {m:<17} {per_method[m]}/8")
    print(f"\nP1 (<= {P1_MAX_EFFECTS} of 40):            "
          f"{'HOLDS' if p1 else 'FAILS'}")
    print(f"P2 (< {P2_MAX_PER_METHOD} per method):        "
          f"{'HOLDS' if p2 else 'FAILS'}  (worst {worst}/8)")
    print(f"P3 (clearing cells favour indep): "
          f"{'HOLDS' if not shared_favouring else 'FAILS'}")
    if shared_favouring:
        print("   reported first, per the registration:")
        for r in shared_favouring:
            print(f"     {r['base']} {r['method']} {r['bench']} d={r['d']:+.3f}")

    if p1 and p2:
        branch, action = 1, ("The null holds in a second unit. Section 7.2 "
                             "gains the accuracy table, with the MDE sentence.")
    elif p1 and not p2:
        offenders = [m for m, c in per_method.items()
                     if c >= P2_MAX_PER_METHOD]
        branch, action = 2, (f"Remove {', '.join(offenders)} from the null and "
                             f"restate contribution 3 over the rest.")
    elif n_eff < 10:
        branch, action = 3, ("The null does not carry over. Section 1 and the "
                             "abstract must say the invisibility claim holds "
                             "in NLL and is not confirmed in accuracy.")
    else:
        branch, action = 4, ("Withdraw the invisibility claim, including from "
                             "the abstract. The practical message reduces to "
                             "the solver half.")
    print(f"\nregistered branch {branch}: {action}")

    # The sentence a null must be written in, fixed in the registration.
    g = {b: 2 * se_binom(0.5, n, 0.5, n) for b, n in BENCH.items()}
    print(f"\nMDE, for the write-up: no effect detectable at about "
          f"{g['gsm8k']:.3f} on GSM8K and {g['humaneval']:.3f} on HumanEval "
          f"(worst case, p = 0.5).")

    tot_empty = sum(r["empty_shared"] + r["empty_indep"] for r in rows)
    print(f"empty generations across all 80 cells: {tot_empty}")

    OUT.write_text(json.dumps({
        "threshold": THRESHOLD, "rows": rows, "n_effects": n_eff,
        "per_method": per_method, "P1": p1, "P2": p2,
        "P3": not shared_favouring, "branch": branch, "action": action,
        "mde": g, "total_empty": tot_empty,
    }, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

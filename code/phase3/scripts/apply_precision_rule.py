#!/usr/bin/env python3
"""Apply the decision rule of notes/check_solve_precision_2026-10-24.md to the
per-base outputs of check_solve_precision.py and write one summary.

  python apply_precision_rule.py <out.json> <solve_precision_<base>.json> ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REL_MAX = 1e-2          # median per-module rel_trunc must be below this
RATIO_BAND = (0.5, 2.0)  # (shared/indep amp, float64) / (shared/indep amp, float32)


def main() -> int:
    out, *ins = map(Path, sys.argv[1:])
    cases = {}
    for p in ins:
        cases.update(json.loads(p.read_text()))
    bases = sorted({k.split("/")[0] for k in cases})
    methods = sorted({m for v in cases.values() for m in v if isinstance(v[m], dict)})
    assert len(cases) == 2 * len(bases) == 8, sorted(cases)
    rows, failures = [], []
    for b in bases:
        for m in methods:
            sh, ind = cases[f"{b}/seed1"][m], cases[f"{b}/indep1"][m]
            r32 = sh["amp32"]["median"] / ind["amp32"]["median"]
            r64 = sh["amp64"]["median"] / ind["amp64"]["median"]
            ratio_ok = RATIO_BAND[0] <= r64 / r32 <= RATIO_BAND[1]
            for arm, c in (("seed1", sh), ("indep1", ind)):
                rel_ok = c["rel_trunc"]["median"] < REL_MAX
                row = {"base": b, "arm": arm, "method": m,
                       "rel_trunc_median": c["rel_trunc"]["median"],
                       "rel_trunc_max": c["rel_trunc"]["max"],
                       "amp_ratio_32": r32, "amp_ratio_64": r64,
                       "prod_gap_median": c["prod_gap"]["median"],
                       "amp_prod_over_amp32": c["amp_prod"]["median"] / c["amp32"]["median"],
                       "trunc_check_max": c["trunc_check"]["max"],
                       "pass": rel_ok and ratio_ok}
                rows.append(row)
                if not row["pass"]:
                    failures.append(f"{b}/{arm}/{m}")
    verdict = ("PRECISION_NOT_THE_CAUSE" if not failures else "REBUILD_AFFECTED_IN_FLOAT64")
    summary = {"rule": "notes/check_solve_precision_2026-10-24.md",
               "thresholds": {"rel_trunc_median_below": REL_MAX, "ratio_band": RATIO_BAND},
               "verdict": verdict, "affected": failures, "rows": rows, "cases": cases}
    out.write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{r['base']:<11} {r['arm']:<7} {r['method']:<13} rel_trunc {r['rel_trunc_median']:.2e}"
              f"  ratio64/32 {r['amp_ratio_64'] / r['amp_ratio_32']:.4f}"
              f"  prod_gap {r['prod_gap_median']:.3f}  {'pass' if r['pass'] else 'FAIL'}")
    print(f"verdict: {verdict}  affected: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare the four restage cells with their published counterparts."""
from __future__ import annotations

import json
import sys

from _c1019 import BASES, RES

TOL = 0.002


def main() -> int:
    ok = True
    report = {}
    for base in BASES:
        new = RES / "restage_check" / f"{base}__rmdf_l0p01__seed1.json"
        old = RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json"
        if not new.exists():
            print(f"missing {new}")
            return 2
        a, b = json.loads(new.read_text()), json.loads(old.read_text())
        dw = abs(a["worst_task_excess"] - b["worst_task_excess"])
        dt = max(abs(a["nll_tau"][s][t] - b["nll_tau"][s][t])
                 for s in b["nll_tau"] for t in b["nll_tau"][s])
        report[base] = {"d_worst": dw, "max_d_nll_tau": dt,
                        "new": a["worst_task_excess"], "published": b["worst_task_excess"]}
        ok &= dw <= TOL and dt <= TOL
        print(f"{base:12s} worst {a['worst_task_excess']:.5f} vs {b['worst_task_excess']:.5f}"
              f"  |d|={dw:.5f}  max|d nll_tau|={dt:.5f}")
    report["pass"] = bool(ok)
    (RES / "restage_check_summary.json").write_text(json.dumps(report, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

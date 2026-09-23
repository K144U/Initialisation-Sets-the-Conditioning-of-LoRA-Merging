#!/usr/bin/env python3
"""Apply notes/prereg_public_merge_2026-09-23.md (U1-U4) and V4 of
notes/prereg_lambda_rule_2026-09-23.md to the public-cohort merges.

Reports only when every selected cohort has a result file. A method that
errored on a cohort is counted as an error, never as a pass.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
OUT = ROOT / "results" / "phase3" / "public_merge"


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for the top-left cell of [[a, b], [c, d]] under the null."""
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def main() -> int:
    sel = [d for d in json.loads((OUT / "selection.json").read_text()) if d["selected"]]
    rows, missing = [], []
    for d in sel:
        f = OUT / f"{d['cohort']}.json"
        if not f.exists():
            missing.append(d["cohort"])
            continue
        r = json.loads(f.read_text())["results"]
        ta = r.get("ta", {}).get("worst")
        ratio = {k: (v["worst"] / ta if "worst" in v and ta and ta > 0 else None)
                 for k, v in r.items() if k != "ta"}
        rows.append({"cohort": d["cohort"], "regime": d["regime"], "ta_worst": ta,
                     "F": ratio, "errors": {k: v["error"] for k, v in r.items() if "error" in v}})
    if missing:
        print(f"{len(missing)} of {len(sel)} cohorts missing; refusing to report")
        return 1
    col = [x for x in rows if x["regime"] == "collapsed"]
    sep = [x for x in rows if x["regime"] == "separated"]

    def frac(group, key, pred):
        ok = [x for x in group if x["F"].get(key) is not None]
        return sum(pred(x["F"][key]) for x in ok), len(group)

    u1c = frac(col, "rm_1em6", lambda f: f >= 2)
    u1s = frac(sep, "rm_1em6", lambda f: f >= 2)
    u2 = frac(col, "rm_0p01", lambda f: f <= 1.5)
    u3 = frac(col, "rm_a09", lambda f: f >= 2)
    v4 = frac(col, "rm_rule", lambda f: f <= 1.5)
    verdict = {"U1": u1c[0] >= 0.6 * u1c[1] and u1s[0] <= 0.2 * u1s[1],
               "U2": u2[0] >= 0.8 * u2[1], "U3": u3[0] >= 0.4 * u3[1],
               "V4": v4[0] >= 0.8 * v4[1]}
    if len(col) >= 5 and len(sep) >= 5:
        p = fisher_one_sided(u1c[0], u1c[1] - u1c[0], u1s[0], u1s[1] - u1s[0])
        verdict["U4"], verdict["U4_p"] = p < 0.05, p
    else:
        verdict["U4"] = "not applicable (fewer than 5 cohorts in a group)"
    rep = {"rules": ["notes/prereg_public_merge_2026-09-23.md",
                     "notes/prereg_lambda_rule_2026-09-23.md (V4)"],
           "counts": {"U1_collapsed": u1c, "U1_separated": u1s, "U2": u2, "U3": u3, "V4": v4},
           "verdict": verdict, "rows": rows}
    (OUT / "public_merge_summary.json").write_text(json.dumps(rep, indent=1, default=str))
    for x in rows:
        f = {k: (round(v, 2) if v is not None else None) for k, v in x["F"].items()}
        print(f"{x['regime']:<10} {x['cohort'][:55]:<55} TA {x['ta_worst']:.4f}  F {f}  {x['errors'] or ''}")
    print(json.dumps(rep["counts"]), json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

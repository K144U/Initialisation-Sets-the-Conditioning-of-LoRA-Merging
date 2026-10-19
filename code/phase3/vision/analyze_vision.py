#!/usr/bin/env python3
"""Verdicts V1 to V6 of notes/prereg_vision_2026-10-19.md. Refuses partial reads."""
from __future__ import annotations

import json
import math
import statistics as st
import sys

from vision_lora import METHODS, RES

S, I = ["shared1", "shared2", "shared3"], ["indep1", "indep2", "indep3"]
GRID = ["rm_1em6", "rm_1em3", "rm_1em2", "rm_3em2", "rm_1em1", "rm_3em1", "rm_1"]


def se(x):
    return st.stdev(x) / math.sqrt(len(x))


def acc(c, tag):
    p = RES / f"eval__{c}__{tag}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["mean"]


def contrast(tag):
    s, i = [acc(c, tag) for c in S], [acc(c, tag) for c in I]
    sd = math.sqrt(se(s) ** 2 + se(i) ** 2)
    d = st.mean(i) - st.mean(s)
    return {"shared": s, "indep": i, "d": d, "gate": max(0.01, 2 * sd),
            "tost90": [d - 1.645 * sd, d + 1.645 * sd]}


def main() -> int:
    try:
        geo = json.loads((RES / "geometry.json").read_text())
        c = {tag: contrast(tag) for tag in METHODS}
    except FileNotFoundError as e:
        print(f"incomplete ({e.args[0]}); refusing to report")
        return 2
    v = {}
    v["V1"] = all(geo[x] >= 0.9 for x in S) and all(geo[x] <= 0.5 for x in I)
    v["V2"] = c["rm_1em6"]["d"] > c["rm_1em6"]["gate"]
    best_s = max(GRID, key=lambda g: st.mean(c[g]["shared"]))
    best_i = max(GRID, key=lambda g: st.mean(c[g]["indep"]))
    gs = [acc(x, best_s) - acc(x, "rm_1em6") for x in S]
    gi = [acc(x, best_i) - acc(x, "rm_1em6") for x in I]
    did = st.mean(gs) - st.mean(gi)
    v["V3"] = did > max(0.01, 2 * math.sqrt(se(gs) ** 2 + se(gi) ** 2))
    v["V4"] = all(-0.02 < c[t]["tost90"][0] and c[t]["tost90"][1] < 0.02 for t in ("ta", "ties"))
    v["V5"] = all(c[t]["d"] <= c[t]["gate"] for t in ("rma", "tsv"))
    v["V6"] = c["wudi_cf"]["d"] > c["wudi_cf"]["gate"] and c["wudi_pub"]["d"] <= c["wudi_pub"]["gate"]
    branch = 1 if v["V1"] and v["V2"] and v["V3"] else (2 if v["V1"] and not v["V2"] else (3 if not v["V1"] else 0))
    out = {"rules": "notes/prereg_vision_2026-10-19.md", "geometry": geo, "contrasts": c,
           "best_lambda": {"shared": best_s, "indep": best_i}, "ridge_did": did,
           "verdict": v, "branch": branch}
    (RES / "vision_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(v, indent=1), "branch", branch)
    for t, x in c.items():
        print(f"{t:9s} shared {st.mean(x['shared']):.4f} indep {st.mean(x['indep']):.4f} "
              f"d {x['d']:+.4f} gate {x['gate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verdicts H1, H2 (and H3 table) of notes/prereg_untruncated_2026-10-19.md."""
from __future__ import annotations

import json
import statistics as st
import sys

from _c1019 import BASES, INDEP, RES, SHARED

PAIRS = [("task_arithmetic", "humaneval"), ("dare", "humaneval"), ("ties", "gsm8k")]
TAGS = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots_ties"]


def acc(d, base, tag, bench, cohort):
    p = RES / d / f"{base}__{tag}_{bench}__{cohort}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["metric_score"]


def truncated_dir(cohort):
    return "eval_heur_downstream" if cohort in ("seed1", "indep1") else "eval_heur_tost"


def main() -> int:
    out, v = {"rules": "notes/prereg_untruncated_2026-10-19.md"}, {}
    te = json.loads((RES / "truncation_energy.json").read_text())
    h1 = {}
    for tag in TAGS:
        wins = 0
        for b in BASES:
            s = st.mean(te[f"{b}/{c}/{tag}"] for c in SHARED)
            i = st.mean(te[f"{b}/{c}/{tag}"] for c in INDEP)
            h1[f"{b}/{tag}"] = {"shared": s, "indep": i}
            wins += i > s
        h1[tag] = wins
    v["H1"] = all(h1[t] >= 3 for t in TAGS)
    out["H1"] = h1
    try:
        h2 = {}
        for tag, bench in PAIRS:
            def dbar(dirf):
                return st.mean(st.mean(acc(dirf(c), b, tag, bench, c) for c in INDEP)
                               - st.mean(acc(dirf(c), b, tag, bench, c) for c in SHARED)
                               for b in BASES)
            dt = dbar(truncated_dir)
            du = dbar(lambda c: "eval_untrunc_ds")
            h2[f"{tag}/{bench}"] = {"d_trunc": dt, "d_untrunc": du, "halved": du >= dt / 2}
        v["H2"] = sum(x["halved"] for x in h2.values()) >= 2
        out["H2"] = h2
    except FileNotFoundError as e:
        print(f"H2 incomplete ({e.args[0]}); refusing to report H2")
        v["H2"] = None
    h3 = {}
    for b in BASES:
        for tag in TAGS:
            for c in ("seed1", "indep1"):
                p = RES / "eval_untrunc" / f"{b}__{tag}__{c}.json"
                if p.exists():
                    h3[f"{b}/{tag}/{c}"] = json.loads(p.read_text())["worst_task_excess"]
    out["H3_untruncated_nll"] = h3
    branch = 1 if v["H1"] and v["H2"] else (2 if v["H1"] else 3)
    out["verdict"], out["branch"] = v, branch
    (RES / "untruncated_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(v), "branch", branch)
    for k, x in out.get("H2", {}).items():
        print(f"{k:28s} d_trunc {x['d_trunc']:+.4f}  d_untrunc {x['d_untrunc']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

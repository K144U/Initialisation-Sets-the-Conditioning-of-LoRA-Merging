#!/usr/bin/env python3
"""Verdicts W1 to W6 of notes/prereg_wudi_2026-10-19.md. Refuses partial reads."""
from __future__ import annotations

import json
import math
import statistics as st
import sys

from _c1019 import BASES, INDEP, RES, SHARED

NLL, DS = RES / "eval_wudi", RES / "eval_wudi_ds"


def se(xs):
    return st.stdev(xs) / math.sqrt(len(xs))


def load(p, key):
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())[key]


def diag_mean(p, field):
    d = json.loads(p.read_text()).get("merge_diagnostics") or {}
    vals = [v[field] for v in d.values() if isinstance(v, dict) and field in v]
    return sum(vals) / len(vals) if vals else None


def nll_arm(base, tag):
    s = [load(NLL / f"{base}__{tag}__{c}.json", "worst_task_excess") for c in SHARED]
    i = [load(NLL / f"{base}__{tag}__{c}.json", "worst_task_excess") for c in INDEP]
    d = st.mean(s) - st.mean(i)
    gate = max(0.005, 2 * math.sqrt(se(s) ** 2 + se(i) ** 2))
    tail = [diag_mean(NLL / f"{base}__{tag}__{c}.json", "tail_frac") for c in SHARED]
    off = [diag_mean(NLL / f"{base}__{tag}__{c}.json", "off_range_frac") for c in SHARED]
    amp = [diag_mean(NLL / f"{base}__{tag}__{c}.json", "amp") for c in SHARED + INDEP]
    return {"shared": s, "indep": i, "d": d, "gate": gate,
            "ratio": st.mean(s) / st.mean(i), "effect": d > gate,
            "tail_shared": st.mean(tail), "off_range_shared": st.mean(off),
            "amp_shared": st.mean(amp[:3]), "amp_indep": st.mean(amp[3:])}


def acc_arm(base, tag, bench):
    s = [load(DS / f"{base}__{tag}_{bench}__{c}.json", "metric_score") for c in SHARED]
    i = [load(DS / f"{base}__{tag}_{bench}__{c}.json", "metric_score") for c in INDEP]
    d = st.mean(i) - st.mean(s)
    gate = max(0.05, 2 * math.sqrt(se(s) ** 2 + se(i) ** 2))
    return {"shared": s, "indep": i, "d_acc": d, "gate": gate, "harm": d > gate}


def main() -> int:
    out = {"rules": "notes/prereg_wudi_2026-10-19.md"}
    try:
        nll = {(b, t): nll_arm(b, t) for b in BASES for t in ["wudi_pub", "wudi_cf", "wudi_3k"]}
    except FileNotFoundError as e:
        print(f"NLL incomplete ({e.args[0]}); refusing to report W1-W4")
        return 2
    out["nll"] = {f"{b}/{t}": v for (b, t), v in nll.items()}
    cf = [nll[(b, "wudi_cf")] for b in BASES]
    pub = [nll[(b, "wudi_pub")] for b in BASES]
    k3 = [nll[(b, "wudi_3k")] for b in BASES]
    v = {}
    v["W1"] = sum(x["ratio"] >= 2 and x["effect"] for x in cf) >= 3
    v["W2"] = sum(not x["effect"] for x in pub) >= 3
    v["W3"] = (sum(x["tail_shared"] >= 0.5 for x in cf) >= 3
               and sum(x["tail_shared"] <= 0.1 for x in pub) >= 3)
    v["W4"] = sum(min(p["ratio"], c["ratio"]) < k["ratio"] < max(p["ratio"], c["ratio"])
                  for p, c, k in zip(pub, cf, k3)) >= 3
    try:
        acc = {(b, t, k): acc_arm(b, t, k) for b in BASES
               for t in ["wudi_pub", "wudi_cf"] for k in ["gsm8k", "humaneval"]}
        out["acc"] = {f"{b}/{t}/{k}": x for (b, t, k), x in acc.items()}
        v["W5"] = any(sum(acc[(b, "wudi_cf", k)]["harm"] for b in BASES) >= 3
                      for k in ["gsm8k", "humaneval"])
        v["W6"] = all(sum(not acc[(b, "wudi_pub", k)]["harm"] for b in BASES) >= 3
                      for k in ["gsm8k", "humaneval"])
    except FileNotFoundError as e:
        print(f"accuracy incomplete ({e.args[0]}); W5, W6 not reported")
    if v["W1"] and v["W2"]:
        branch = 1
    elif not v["W2"]:
        branch = 2
    elif not v["W1"]:
        branch = 3
    else:
        branch = 4
    if v["W1"] and v["W2"] and not v["W3"]:
        branch = 4
    out["verdict"], out["branch"] = v, branch
    (RES / "wudi_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(v, indent=1))
    for b in BASES:
        for t in ["wudi_pub", "wudi_3k", "wudi_cf"]:
            x = nll[(b, t)]
            print(f"{b:11s} {t:8s} shared {st.mean(x['shared']):8.4f} indep {st.mean(x['indep']):7.4f}"
                  f" ratio {x['ratio']:6.2f} d {x['d']:+.4f} gate {x['gate']:.4f}"
                  f" tail {x['tail_shared']:.3f} off {x['off_range_shared']:.3f}"
                  f" amp {x['amp_shared']:.2f}/{x['amp_indep']:.2f}")
    print(f"branch: {branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

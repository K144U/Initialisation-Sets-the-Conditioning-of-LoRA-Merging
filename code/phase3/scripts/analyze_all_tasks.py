#!/usr/bin/env python3
"""Verdicts T1, T2 of notes/prereg_all_tasks_2026-10-19.md, plus reference rows.

Gates: chrF max(2 points, 2 bootstrap SE of the difference, 1000 resamples of
sentences, seed 0); IFEval max(0.05, 2 binomial SE of the difference).
"""
from __future__ import annotations

import json
import math
import random
import sys

from _c1019 import BASES, RES

OUT = RES / "eval_all_tasks"


def load(name):
    p = OUT / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def chrf_gap(a, b):
    import sacrebleu
    ra, rb = a["per_example"], b["per_example"]
    assert [x["index"] for x in ra] == [x["index"] for x in rb]
    d0 = a["metric_score"] - b["metric_score"]
    rng, n, ds = random.Random(0), len(ra), []
    for _ in range(1000):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sacrebleu.corpus_chrf([ra[i]["hyp"] for i in idx], [[ra[i]["ref"] for i in idx]]).score
        sb = sacrebleu.corpus_chrf([rb[i]["hyp"] for i in idx], [[rb[i]["ref"] for i in idx]]).score
        ds.append(sa - sb)
    m = sum(ds) / len(ds)
    se = math.sqrt(sum((x - m) ** 2 for x in ds) / (len(ds) - 1))
    return d0, max(2.0, 2 * se)


def ifeval_gap(a, b):
    pa, pb, n = a["metric_score"], b["metric_score"], a["n_eval_metric"]
    se = math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
    return pa - pb, max(0.05, 2 * se)


GAP = {"wmt_chrf": chrf_gap, "ifeval_strict": ifeval_gap}


def main() -> int:
    v, out = {}, {"rules": "notes/prereg_all_tasks_2026-10-19.md", "cells": {}}
    try:
        for metric, gap in GAP.items():
            t1 = 0
            t2 = {"task_arithmetic": 0, "ties": 0}
            for b in BASES:
                d, g = gap(load(f"{b}__enc_lstar_{metric}__seed1"), load(f"{b}__enc_l0_{metric}__seed1"))
                out["cells"][f"{b}/{metric}/enc_gain_seed1"] = {"d": d, "gate": g}
                t1 += d > g
                for h in t2:
                    d, g = gap(load(f"{b}__{h}_{metric}__seed1"), load(f"{b}__{h}_{metric}__indep1"))
                    out["cells"][f"{b}/{metric}/{h}_shared_minus_indep"] = {"d": d, "gate": g}
                    t2[h] += abs(d) <= g
            v[f"T1_{metric}"] = t1 >= 3
            v[f"T2_{metric}"] = all(c >= 3 for c in t2.values())
    except FileNotFoundError as e:
        print(f"incomplete ({e.args[0]}); refusing to report")
        return 2
    refs = {}
    for b in BASES:
        for metric in list(GAP) + ["gsm8k", "humaneval"]:
            for tag in ["base", "ind_gsm8k", "ind_alpaca", "ind_magicoder", "ind_translation"]:
                p = OUT / f"{b}__{tag}_{metric}.json"
                if p.exists():
                    refs[f"{b}/{metric}/{tag}"] = json.loads(p.read_text())["metric_score"]
    out["reference_rows"] = refs
    out["verdict"] = v
    (RES / "all_tasks_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

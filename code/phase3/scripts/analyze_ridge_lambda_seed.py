#!/usr/bin/env python3
"""Aggregate the matched-seed Llama-3.1 rd-encoder ridge lambda-sweep
(results/phase3/eval_ridge_seed/, 30 cells = 10 lambdas x seed1/2/3) into the
3-seed-mean numbers for paper Table tab:rd-ridge-sweep, the Llama row of
tab:rd-ridge-heldout, and the salvage figure. Replaces the legacy v1 numbers.

For each lambda we report the 3-seed mean of excess_per_task[t] for each of the
four tasks and of worst_task_excess (mean of each seed's per-seed worst)."""
from __future__ import annotations
import json
from pathlib import Path

RES = Path("/home/sanjay.g/projects/rdmerge/results/phase3/eval_ridge_seed")
BASE = "llama31_8b"
LAMBDAS = [0.001, 0.01, 0.05, 0.07, 0.1, 0.13, 0.17, 0.2, 0.3, 1.0]
SEEDS = ["seed1", "seed2", "seed3"]
TASKS = ["gsm8k", "alpaca", "magicoder", "translation"]


def ltag(l):
    return "l1" if l == 1.0 else "l" + ("%g" % l).replace("0.", "0p").replace(".", "p")


def mean(xs):
    return sum(xs) / len(xs)


hdr = "{:>7} | {:>7} {:>7} {:>7} {:>7} | {:>11}  per-seed-worst".format(
    "lambda", "GSM8K", "Alpaca", "Magic", "Trans", "worst(mean)")
print(hdr)
print("-" * 92)
rows = {}
for l in LAMBDAS:
    per_task = {t: [] for t in TASKS}
    worst = []
    miss = []
    for s in SEEDS:
        f = RES / "{}__ridge_{}__{}.json".format(BASE, ltag(l), s)
        if not f.exists():
            miss.append(s)
            continue
        d = json.loads(f.read_text())
        for t in TASKS:
            per_task[t].append(d["excess_per_task"][t])
        worst.append(d["worst_task_excess"])
    if miss:
        print("{:>7} | MISSING seeds: {}".format(l, miss))
        continue
    mt = {t: mean(per_task[t]) for t in TASKS}
    wm = mean(worst)
    rows[l] = (mt, wm, worst)
    ps = " ".join("{:.4f}".format(w) for w in worst)
    print("{:>7g} | {:+7.3f} {:+7.3f} {:+7.3f} {:+7.3f} | {:>11.4f}  [{}]".format(
        l, mt["gsm8k"], mt["alpaca"], mt["magicoder"], mt["translation"], wm, ps))

print()
print("=== LaTeX rows for tab:rd-ridge-sweep (3-seed means) ===")
for l in LAMBDAS:
    if l not in rows:
        continue
    mt, wm, _ = rows[l]
    wcell = "$\\mathbf{{{:.3f}}}$".format(wm) if l == min(rows, key=lambda x: rows[x][1]) else "${:.3f}$".format(wm)
    print("${:g}$ & ${:+.2f}$ & ${:+.2f}$ & ${:+.2f}$ & ${:+.2f}$ & {} \\\\".format(
        l, mt["gsm8k"], mt["alpaca"], mt["magicoder"], mt["translation"], wcell))

print()
best_l = min(rows, key=lambda l: rows[l][1])
print("=== KEY NUMBERS (3-seed means) ===")
print("lambda* (min worst-task) = {:g}  -> worst-task {:.4f}".format(best_l, rows[best_l][1]))
for probe in (0.05, 0.13):
    if probe in rows:
        print("lambda={:g}              -> worst-task {:.4f}".format(probe, rows[probe][1]))

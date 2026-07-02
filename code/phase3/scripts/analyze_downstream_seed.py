#!/usr/bin/env python3
"""Regenerate the §6.5 downstream accuracy tables (tab:e3-gsm8k GSM8K em +
tab:b4-humaneval HumanEval pass@1) on the MATCHED seed1/2/3 adapters, replacing
the legacy all-v1 numbers. 3-seed means per (base, method); rd-ridge column from
the *_rdridge_seed dirs. Spearman rho per base across the 5 matrix baselines,
accuracy vs -NLL-excess (3-seed mean worst-task excess from eval_matrix_seeds)."""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

ROOT = Path("/home/sanjay.g/projects/rdmerge")
RES = ROOT / "results/phase3"
MATRIX = RES / "eval_matrix_seeds"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
MODEL_LABEL = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B-v0.3",
               "qwen25_7b": "Qwen-2.5-7B", "yi15_9b": "Yi-1.5-9B"}
# display order; (file-method-tag, matrix-method-name for NLL)
METHODS = [("ta", "task_arithmetic"), ("ties", "ties"), ("dare", "dare"),
           ("knots", "knots"), ("tvq_b2", "tvq_b2"), ("rd_ridge", None)]
BASELINES = ["ta", "ties", "dare", "knots", "tvq_b2"]
MLABEL = {"ta": "TA", "ties": "TIES", "dare": "DARE", "knots": "KnOTS",
          "tvq_b2": "TVQ $b=2$", "rd_ridge": "rd-ridge"}
SEEDS = ["seed1", "seed2", "seed3"]


def spearmanr(a, b):
    def ranks(xs):
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    return pearsonr(ranks(a), ranks(b))


def pearsonr(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_acc(dirs):
    """dirs: list of result dirs. Returns {base: {method: 3-seed-mean score}}."""
    acc = {m: {} for m in MODELS}
    raw = {m: {} for m in MODELS}
    for d in dirs:
        for f in glob.glob(str(RES / d / "*.json")):
            stem = os.path.basename(f)[:-5]
            parts = stem.split("__")  # base, method, metric, seed
            if len(parts) != 4:
                continue
            base, method, _metric, seed = parts
            if base not in acc:
                continue
            raw[base].setdefault(method, {})[seed] = json.load(open(f))["metric_score"]
    for base in MODELS:
        for method, seedmap in raw[base].items():
            vals = [seedmap[s] for s in SEEDS if s in seedmap]
            acc[base][method] = (mean(vals), len(vals))
    return acc


def load_nll_excess(base, matrix_method):
    vals = []
    for s in SEEDS:
        p = MATRIX / f"{base}__{matrix_method}__{s}.json"
        if p.exists():
            v = json.load(open(p)).get("worst_task_excess")
            if v is not None:
                vals.append(v)
    return mean(vals) if vals else None


def report(title, dirs):
    print("=" * 78)
    print(title)
    print("=" * 78)
    acc = load_acc(dirs)
    # table
    hdr = f"{'base':<16}" + "".join(f"{MLABEL[t]:>12}" for t, _ in METHODS) + f"{'rho':>8}"
    print(hdr)
    latex = []
    for base in MODELS:
        cells = {}
        for tag, _ in METHODS:
            entry = acc[base].get(tag)
            cells[tag] = entry[0] if entry else None
        # spearman over baselines vs -NLL excess (3-seed)
        accs = [cells[t] for t in BASELINES]
        nlls = [load_nll_excess(base, dict(METHODS)[t]) for t in BASELINES]
        if all(a is not None for a in accs) and all(n is not None for n in nlls):
            rho = spearmanr(accs, [-n for n in nlls])
        else:
            rho = None
        rowstr = f"{base:<16}"
        for tag, _ in METHODS:
            v = cells[tag]
            rowstr += (f"{v:>12.3f}" if v is not None else f"{'--':>12}")
        rowstr += (f"{rho:>+8.2f}" if rho is not None else f"{'--':>8}")
        print(rowstr)
        # best (max) and worst (min) among all 6 for highlighting
        present = {t: cells[t] for t, _ in METHODS if cells[t] is not None}
        best_tag = max(present, key=present.get)
        worst_val = min(present.values())
        latexcells = []
        for tag, _ in METHODS:
            v = cells[tag]
            if v is None:
                latexcells.append("--"); continue
            s = f"{v:.3f}"
            if tag == best_tag:
                s = "\\best{$%s$}" % s
            elif v == worst_val:
                s = "\\worst{$%s$}" % s
            else:
                s = "$%s$" % s
            latexcells.append(s)
        rho_s = f"${rho:+.2f}$" if rho is not None else "--"
        latex.append(f"{MODEL_LABEL[base]:<15} & " + " & ".join(latexcells) + f" & {rho_s} \\\\")
    # seed-count sanity
    print("\nseed counts (should be 3 each):")
    for base in MODELS:
        cnts = {t: (acc[base].get(t)[1] if acc[base].get(t) else 0) for t, _ in METHODS}
        print(f"  {base:<14} " + " ".join(f"{t}:{cnts[t]}" for t, _ in METHODS))
    print("\nLaTeX rows:")
    for r in latex:
        print(r)
    print()


report("GSM8K em accuracy (3-seed mean, n=500) — tab:e3-gsm8k",
       ["eval_e3_gsm8k_seed", "eval_e3b_gsm8k_rdridge_seed"])
report("HumanEval pass@1 (3-seed mean, n=164) — tab:b4-humaneval",
       ["eval_b4_humaneval_seed", "eval_b4b_humaneval_rdridge_seed"])

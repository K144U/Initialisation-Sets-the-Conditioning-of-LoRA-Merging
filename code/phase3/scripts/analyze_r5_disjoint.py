"""R5: re-score on the evaluation examples that were NOT in training.

The Reproducibility Statement discloses a 10 to 14.5 percent train/eval overlap
and then asserts "we do not believe the overlap affects any comparison between
methods, which is what all of our claims rest on". Nothing supports that
sentence. Identical exposure across methods establishes that the exposure is
equal; it does not establish that the benefit is, because methods differ in how
much of each task vector they retain and therefore in how much memorised
training content survives the merge.

No retraining and no GPU. Every cell already stores per-example NLL in
evaluation order, so the overlapping positions can be dropped and the whole
matrix re-aggregated.

THE MECHANISM, from data_loaders.py:

    shuffled  = ds[train_split].shuffle(seed=seed)
    train_raw = shuffled.select(range(n_train))
    eval_raw  = shuffled.select(range(n_train, n_train + n_eval))

Disjointness holds only when both processes shuffle with the same seed. They do
not: training passes seeds.global, evaluation passes its own seed. alpaca and
magicoder set split_eval == split_train, so their "held-out" 1000 examples are a
different permutation's slice of the same pool. gsm8k (test) and translation
(validation) draw from genuinely different splits and are unaffected.

Training seeds are per cohort and, for the independent cohorts, per TASK:

    seed1/2/3        every task shares 1 / 2 / 3
    indep1           gsm8k 101, alpaca 102, magicoder 103, flores 104
    indep2 / indep3  201-204 / 301-304

SELF-CHECK FIRST. Before any filtered number is reported, the script rebuilds
each cell's unfiltered mean from its own per-example records and compares it to
the stored scalar. If the aggregation does not reproduce, the filtered numbers
are meaningless and the script stops.

Usage:
  python code/phase3/scripts/analyze_r5_disjoint.py
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG, RES = ROOT / "code/phase3/configs", ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
AFFECTED = ["alpaca", "magicoder"]        # split_eval == split_train
N_TRAIN, N_EVAL = 7500, 1000

# (results dir, config dir, cohorts, methods)
MATRICES = [
    ("eval_a1_indep", "eval_a1_indep", ["indep1", "indep2", "indep3"],
     ["task_arithmetic", "ties", "dare", "tvq_b2", "knots", "rd_ridge",
      "rd_rank16"]),
    ("eval_matrix_seeds", "eval_matrix_seeds", ["seed1", "seed2", "seed3"],
     ["task_arithmetic", "ties", "dare", "tvq_b2", "knots"]),
]

TRAIN_SEED = {
    **{c: {t: s for t in ["gsm8k", "alpaca", "magicoder", "translation"]}
       for c, s in [("seed1", 1), ("seed2", 2), ("seed3", 3)]},
    **{f"indep{i}": {"gsm8k": 100 * i + 1, "alpaca": 100 * i + 2,
                     "magicoder": 100 * i + 3, "translation": 100 * i + 4}
       for i in (1, 2, 3)},
}


def key(row, task_cfg):
    """Stable identity for an example, independent of column order."""
    p = str(row.get(task_cfg["prompt_field"], ""))
    a = str(row.get(task_cfg["answer_field"], ""))
    return hashlib.sha1((p + "\x00" + a).encode("utf-8")).hexdigest()


_ds_cache: dict = {}


def overlap_mask(task_cfg, train_seed, eval_seed):
    """True at eval positions whose example also appears in the train draw."""
    ck = (task_cfg["dataset"], task_cfg.get("config"))
    if ck not in _ds_cache:
        ds = (load_dataset(task_cfg["dataset"], task_cfg["config"])
              if task_cfg.get("config") else load_dataset(task_cfg["dataset"]))
        _ds_cache[ck] = ds[task_cfg["split_train"]]
    pool = _ds_cache[ck]

    train = pool.shuffle(seed=train_seed).select(range(min(N_TRAIN, len(pool))))
    train_keys = {key(r, task_cfg) for r in train}
    ev = pool.shuffle(seed=eval_seed).select(
        range(N_TRAIN, min(N_TRAIN + N_EVAL, len(pool))))
    return [key(r, task_cfg) in train_keys for r in ev]


def weighted(per_example, mask=None):
    """Token-weighted mean NLL, the aggregation compute_nll uses."""
    num = den = 0.0
    for i, e in enumerate(per_example):
        if e.get("skipped") or e.get("nll") is None:
            continue
        if mask is not None and i < len(mask) and mask[i]:
            continue
        w = e.get("n_answer") or 1
        num += e["nll"] * w
        den += w
    return num / den if den else float("nan")


def main() -> int:
    masks: dict = {}
    rows: dict = {}
    checked = failed = 0

    for resdir, cfgdir, cohorts, methods in MATRICES:
        for base in BASES:
            for cohort in cohorts:
                for method in methods:
                    rp = RES / resdir / f"{base}__{method}__{cohort}.json"
                    cp = CFG / cfgdir / f"{base}__{method}__{cohort}.yaml"
                    if not rp.exists() or not cp.exists():
                        continue
                    cell = json.loads(rp.read_text())
                    cfg = yaml.safe_load(cp.read_text())
                    eval_seed = cfg.get("seed", 20260518)
                    specs = {s["name"]: s["task_cfg"]
                             for s in cfg["adapter_specs"]}

                    excess_full, excess_clean = {}, {}
                    for task, tcfg in specs.items():
                        pm = cell["per_example_nll_merged"][task]
                        pt = cell["per_example_nll_tau"][task][task]

                        # self-check against the stored scalars
                        for got, want in ((weighted(pm),
                                           cell["nll_merged"][task]),
                                          (weighted(pt),
                                           cell["nll_tau"][task][task])):
                            checked += 1
                            if abs(got - want) > 1e-6:
                                failed += 1
                                print(f"SELF-CHECK FAILED {rp.name} {task}: "
                                      f"rebuilt {got:.6f} vs stored {want:.6f}")

                        excess_full[task] = weighted(pm) - weighted(pt)
                        if task in AFFECTED:
                            mk = (task, TRAIN_SEED[cohort][task], eval_seed)
                            if mk not in masks:
                                masks[mk] = overlap_mask(tcfg, mk[1], mk[2])
                                print(f"  mask {mk}: "
                                      f"{sum(masks[mk])}/{len(masks[mk])} "
                                      f"overlapping")
                            m = masks[mk]
                        else:
                            m = None
                        excess_clean[task] = weighted(pm, m) - weighted(pt, m)

                    rows[(resdir, base, cohort, method)] = (
                        max(excess_full.values()), max(excess_clean.values()),
                        max(excess_full, key=excess_full.get),
                        max(excess_clean, key=excess_clean.get))

    print(f"\nself-check: {checked - failed}/{checked} aggregations reproduced")
    if failed:
        print("ABORT: the aggregation does not reproduce the stored numbers, "
              "so no filtered number here can be trusted.")
        return 2

    print("\noverlap measured, per (task, train seed, eval seed):")
    for mk, m in sorted(masks.items()):
        print(f"  {mk[0]:<10} train {mk[1]:<6} eval {mk[2]}: "
              f"{100 * sum(m) / len(m):.1f}%")

    print("\n" + "=" * 100)
    print("METHOD ORDERING, all cells vs overlap-free cells")
    print("=" * 100)
    flips = 0
    for resdir, _cfgdir, cohorts, methods in MATRICES:
        for base in BASES:
            for cohort in cohorts:
                have = [(m, rows[(resdir, base, cohort, m)])
                        for m in methods if (resdir, base, cohort, m) in rows]
                if len(have) < 2:
                    continue
                o_full = [m for m, _ in sorted(have, key=lambda x: x[1][0])]
                o_clean = [m for m, _ in sorted(have, key=lambda x: x[1][1])]
                same = o_full == o_clean
                flips += not same
                if not same:
                    print(f"{base} {cohort}: ORDER CHANGES")
                    print(f"   all cells: {' < '.join(o_full)}")
                    print(f"   disjoint : {' < '.join(o_clean)}")
    print(f"\nordering changes in {flips} of the "
          f"{sum(1 for _ in rows) // max(1, len(MATRICES[0][3]))} rows examined")

    print("\nworst-task argmax changes:")
    n_arg = sum(1 for v in rows.values() if v[2] != v[3])
    print(f"  {n_arg} of {len(rows)} cells change which task is worst")

    print("\nper-cell excess shift (clean minus full), nats:")
    d = [v[1] - v[0] for v in rows.values()]
    print(f"  mean {statistics.fmean(d):+.4f}, "
          f"range {min(d):+.4f} to {max(d):+.4f}")

    out = RES / "r5_disjoint_summary.json"
    out.write_text(json.dumps(
        {"masks": {f"{k[0]}|{k[1]}|{k[2]}": sum(v) for k, v in masks.items()},
         "cells": {"|".join(k): v for k, v in rows.items()}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

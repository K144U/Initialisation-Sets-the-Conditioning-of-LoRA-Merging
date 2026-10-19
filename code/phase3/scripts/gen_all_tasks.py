#!/usr/bin/env python3
"""Cells of notes/prereg_all_tasks_2026-10-19.md: 144 accuracy cells.

Per base: for wmt_chrf and ifeval_strict, the base model (task arithmetic with
zero weights), the four seed1 adapters alone (one-hot weights), task
arithmetic and TIES on seed1/indep1 (heur_downstream settings), and the encoder
at lambda = 0 and lambda* on seed1/indep1 (lambda* read from eval_ridge_cond
exactly as gen_downstream_conditioning did; template eval_a1_indep rd_ridge).
Plus base and single-adapter reference rows on GSM8K and HumanEval.
"""
from __future__ import annotations

import json
import sys

from _c1019 import BASES, RES, Manifest, retarget, template

OUT = "eval_all_tasks"
NEW = {"wmt_chrf": {"metric_task_spec": "translation",
                    "metric_kwargs": {"n": 500, "seed": 0, "max_new_tokens": 256}},
       "ifeval_strict": {"metric_task_spec": "alpaca",
                         "metric_kwargs": {"n": 300, "seed": 0, "max_new_tokens": 512}}}
TASKS = ["gsm8k", "alpaca", "magicoder", "translation"]
TAGS = ["0", "0p01", "0p03", "0p05", "0p13", "0p30", "1p00"]
LAM = {"0": 0.0, "0p01": 0.01, "0p03": 0.03, "0p05": 0.05, "0p13": 0.13, "0p30": 0.30, "1p00": 1.00}


def lstar(base, cohort):
    v = {t: json.loads((RES / "eval_ridge_cond" / f"{base}__rd_l{t}__{cohort}.json").read_text())
         ["worst_task_excess"] for t in TAGS}
    return min(v, key=v.get)


def with_metric(cfg, metric):
    cfg = dict(cfg)
    for k in ("metric_task_cfg",):
        cfg.pop(k, None)
    cfg.update(metric=metric, n_eval_metric=NEW[metric]["metric_kwargs"]["n"], **NEW[metric])
    return cfg


def main() -> int:
    m = Manifest()
    for base in BASES:
        ta = template(RES / "eval_heur_downstream" / f"{base}__task_arithmetic_gsm8k__seed1.json")
        refs = {"base": [0.0] * 4, **{f"ind_{t}": [1.0 if i == k else 0.0 for i in range(4)]
                                     for k, t in enumerate(TASKS)}}
        for metric in NEW:
            for tag, w in refs.items():
                cfg = with_metric({k: v for k, v in ta.items() if k != "output_path"}, metric)
                cfg.update(method="task_arithmetic", method_kwargs={}, weights=w,
                           output_path=str(RES / OUT / f"{base}__{tag}_{metric}.json"))
                m.add(f"{base}__{tag}_{metric}", cfg, OUT, "run_downstream_cell.py")
            for htag in ("task_arithmetic", "ties"):
                for cohort in ("seed1", "indep1"):
                    t = template(RES / "eval_heur_downstream" / f"{base}__{htag}_gsm8k__{cohort}.json")
                    cfg = with_metric({k: v for k, v in t.items() if k != "output_path"}, metric)
                    cfg["output_path"] = str(RES / OUT / f"{base}__{htag}_{metric}__{cohort}.json")
                    m.add(f"{base}__{htag}_{metric}__{cohort}", cfg, OUT, "run_downstream_cell.py")
            enc = template(RES / "eval_a1_indep" / f"{base}__rd_ridge__indep1.json")
            for cohort in ("seed1", "indep1"):
                specs = retarget(enc["adapter_specs"], "indep1", cohort)
                for which, lt in (("l0", "0"), ("lstar", lstar(base, cohort))):
                    cfg = {k: v for k, v in enc.items() if k != "output_path"}
                    kw = dict(cfg.get("method_kwargs", {}))
                    kw.update(ridge_lambda=LAM[lt], realize="rank_r")
                    cfg.update(method="rd_encoder", method_kwargs=kw, adapter_specs=specs)
                    cfg = with_metric(cfg, metric)
                    cfg["output_path"] = str(RES / OUT / f"{base}__enc_{which}_{metric}__{cohort}.json")
                    m.add(f"{base}__enc_{which}_{metric}__{cohort}", cfg, OUT, "run_downstream_cell.py")
        for bench in ("gsm8k", "humaneval"):
            tb = template(RES / "eval_heur_downstream" / f"{base}__task_arithmetic_{bench}__seed1.json")
            for tag, w in refs.items():
                cfg = {k: v for k, v in tb.items() if k != "output_path"}
                cfg.update(method="task_arithmetic", method_kwargs={}, weights=w,
                           output_path=str(RES / OUT / f"{base}__{tag}_{bench}.json"))
                m.add(f"{base}__{tag}_{bench}", cfg, OUT, "run_downstream_cell.py")
    m.write("all_tasks_manifest.json", expect=144)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Training configs (stage train) for notes/prereg_new_cohorts_2026-09-23.md.

32 adapters, each cloned from the committed config of the adapter it
replaces or mirrors, with only the registered changes applied:

  Part T  translation, stream_shuffle_buffer 50000, llama + mistral,
          seed1 (seeds 1/1) and indep1 (104, data 20260518)            4
  Part R  llama: 4 new tasks for T = 8, shared (1/1) and indep (105-108) 8
          llama: r in {8, 64} x 4 tasks x shared/indep (101-104)       16
  Part P  llama: PiSSA init, 4 tasks, independent seeds 101-104         4

Templates are read from the original tree (read-only); outputs go under the
work tree: artifacts_new/lora/<base>/<task>/<cohort>.

Usage:
  RDMERGE_ROOT=~/projects/rdmerge_v2 python code/phase3/scripts/gen_new_cohorts.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
SRC = Path("/home/sanjay.g/projects/rdmerge/code/phase3/configs")
CFG = ROOT / "code" / "phase3" / "configs" / "lora_new"
BUF = 50000
IND_SEED = {"gsm8k": 101, "alpaca": 102, "magicoder": 103, "flores": 104}
NEW_TASKS = {
    "dolly": ({"name": "dolly", "format": "dolly",
               "dataset": "databricks/databricks-dolly-15k", "config": None,
               "split_train": "train", "split_eval": "train",
               "n_train": 7500, "n_eval": 200}, 105),
    "xsum": ({"name": "xsum", "dataset": "EdinburghNLP/xsum", "config": None,
              "split_train": "train", "split_eval": "validation",
              "n_train": 7500, "n_eval": 200,
              "prompt_field": "document", "answer_field": "summary"}, 106),
    "codealpaca": ({"name": "codealpaca", "format": "alpaca",
                    "dataset": "sahil2801/CodeAlpaca-20k", "config": None,
                    "split_train": "train", "split_eval": "train",
                    "n_train": 7500, "n_eval": 200}, 107),
    "translation_fr": ({"name": "translation_fr", "format": "translation",
                        "dataset": "wmt/wmt14", "config": "fr-en",
                        "split_train": "train", "split_eval": "validation",
                        "n_train": 7500, "n_eval": 200, "src_lang": "en",
                        "tgt_lang": "fr", "streaming": True,
                        "stream_shuffle_buffer": BUF}, 108),
}


def template(base: str, task: str, arm: str) -> dict:
    p = (SRC / "lora_seeds" / f"{base}_{task}_seed1.yaml" if arm == "shared"
         else SRC / "lora_indep" / f"{base}_{task}_indep1.yaml")
    return yaml.safe_load(p.read_text())


def finish(cfg: dict, base: str, task: str, cohort: str) -> dict:
    cfg["output"] = {
        "adapter_dir": f"artifacts_new/lora/{base}/{task}/{cohort}",
        "results_path": f"results/phase3/lora_train_v2/{base}_{task}_{cohort}.json"}
    cfg["train"]["min_free_gb"] = 40
    return cfg


def main() -> int:
    CFG.mkdir(parents=True, exist_ok=True)
    cells = []

    def add(cfg, base, task, cohort, part):
        cfg = finish(cfg, base, task, cohort)
        name = f"{base}__{task}__{cohort}"
        p = CFG / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cells.append({"name": f"train_{name}", "part": part,
                      "cmd": f"python code/phase3/training/train_lora.py --config {p.relative_to(ROOT)}",
                      "done": cfg["output"]["results_path"], "min_free_gb": 45.0})

    # Part T
    for base in ("llama31_8b", "mistral_7b"):
        for arm, cohort in (("shared", "seed1fx"), ("indep", "indep1fx")):
            cfg = copy.deepcopy(template(base, "flores", arm))
            cfg["task"]["stream_shuffle_buffer"] = BUF
            add(cfg, base, "flores", cohort, "T")

    base = "llama31_8b"
    # Part R, T = 8: four new tasks, hyperparameters from the gsm8k template
    for task, (tcfg, iseed) in NEW_TASKS.items():
        for arm, cohort in (("shared", "seed1fx"), ("indep", "indep1fx")):
            cfg = copy.deepcopy(template(base, "gsm8k", arm))
            cfg["task"] = dict(tcfg)
            # Data seed 20260518 in both arms: dolly and codealpaca carve their
            # eval split from the train split, and only a shared data seed with
            # the eval cells keeps the two disjoint (see train_lora.py).
            if arm == "indep":
                cfg["seeds"] = {"global": iseed, "train": iseed, "data": 20260518}
            else:
                cfg["seeds"] = {"global": 1, "train": 1, "data": 20260518}
            add(cfg, base, task, cohort, "R")
    # Part R, rank sweep at T = 4 (translation fixed)
    for r in (8, 64):
        for task in ("gsm8k", "alpaca", "magicoder", "flores"):
            for arm, cohort in (("shared", f"seed1fx_r{r}"), ("indep", f"indep1fx_r{r}")):
                cfg = copy.deepcopy(template(base, task, arm))
                cfg["lora"]["r"], cfg["lora"]["alpha"] = r, 2 * r
                if task == "flores":
                    cfg["task"]["stream_shuffle_buffer"] = BUF
                add(cfg, base, task, cohort, "R")
    # Part P
    for task in ("gsm8k", "alpaca", "magicoder", "flores"):
        cfg = copy.deepcopy(template(base, task, "indep"))
        cfg["lora"]["init"] = "pissa"
        if task == "flores":
            cfg["task"]["stream_shuffle_buffer"] = BUF
        add(cfg, base, task, "pissa_indep", "P")

    counts = {k: sum(c["part"] == k for c in cells) for k in "TRP"}
    if counts != {"T": 4, "R": 24, "P": 4}:
        raise SystemExit(f"unexpected counts {counts}")
    # Part T first: Part R's T = 4 and r = 16 cells depend on the fixed translation.
    cells.sort(key=lambda c: "TRP".index(c["part"]))
    out = ROOT / "code" / "phase3" / "configs" / "new_cohorts_train_manifest.json"
    out.write_text(json.dumps(cells, indent=1))
    print(f"wrote {out}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

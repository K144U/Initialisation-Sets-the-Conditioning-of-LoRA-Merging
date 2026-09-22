#!/usr/bin/env python3
"""Smoke cells for regmean_act, on a cohort OUTSIDE the registered set.

notes/prereg_regmean_real_2026-09-23.md binds the smoke to a cohort that is
not registered and to checking only that the code runs and the merged model
generates. The cohort is the registration's named one, llama32_1b_smoke
(Llama-3.2-1B, T = 2). Sample sizes are tiny on purpose: these numbers answer
"does it run", nothing else.

Four cells: NLL with regmean_act (plain loader), NLL with adapter-only
alpha 0.9, and GSM8K with regmean_act under each loader, because Unsloth's
fused kernels may bypass the forward hooks the Grams need.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
CFG = ROOT / "code" / "phase3" / "configs" / "smoke_regmean_real"
OUT = RES / "smoke_regmean_real"
S = "/home/sanjay.g/projects/rdmerge"
BASE = (f"{S}/cache/hf_home/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/"
        "5a8abab4a5d6f164389b1079fb721cfab8d7126c")
SMOKE_TASKS = ("gsm8k", "alpaca")          # the only two adapters that cohort has


def main():
    CFG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t = json.loads((RES / "eval_regmean_cond" / "llama31_8b__rm_l0p01__seed1.json")
                   .read_text())["meta"]["config"]
    specs = []
    for s in t["adapter_specs"]:
        if s["name"] not in SMOKE_TASKS:
            continue
        s = dict(s)
        s["dir"] = f"{S}/artifacts/lora/llama32_1b_smoke/{s['name']}"
        s["task_cfg"] = dict(s["task_cfg"], n_eval=20)
        assert Path(s["dir"], "adapter_model.safetensors").exists(), s["dir"]
        specs.append(s)
    common = {"base_model": BASE, "max_seq_length": 1024, "seed": t["seed"],
              "min_free_gb": 30.0, "adapter_specs": specs}
    cells = {
        "smoke_rma_nll": dict(common, loader="plain", method="regmean_act",
                              method_kwargs={"alpha": 0.9, "n_gram_samples": 16}),
        "smoke_rmdf_a09_nll": dict(common, loader="plain", method="regmean",
                                   method_kwargs={"alpha": 0.9, "ridge_lambda": 0.0}),
        "smoke_rma_gsm8k_plain": dict(common, loader="plain", method="regmean_act",
                                      method_kwargs={"alpha": 0.9, "n_gram_samples": 16},
                                      metric="gsm8k_em", metric_task_spec="gsm8k",
                                      metric_kwargs={"max_new_tokens": 128},
                                      n_eval_metric=8),
        # notes/prereg_third_solver_2026-09-23.md binds its smoke to this cohort too
        "smoke_tsv_nll": dict(common, loader="plain", method="tsv",
                              method_kwargs={"alpha": 1.0}),
        "smoke_mats_nll": dict(common, loader="plain", method="mats",
                               method_kwargs={"kfac": True, "num_iterations": 100,
                                              "n_gram_samples": 16, "gram_max_len": 512}),
        "smoke_rma_gsm8k_unsloth": dict(common, method="regmean_act",
                                        method_kwargs={"alpha": 0.9, "n_gram_samples": 16},
                                        metric="gsm8k_em", metric_task_spec="gsm8k",
                                        metric_kwargs={"max_new_tokens": 128},
                                        n_eval_metric=8),
    }
    manifest = []
    for name, cfg in cells.items():
        cfg = dict(cfg, output_path=str(OUT / f"{name}.json"))
        p = CFG / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        runner = "run_downstream_cell.py" if "metric" in cfg else "run_eval_cell.py"
        manifest.append({"name": name,
                         "cmd": f"python code/phase3/eval/{runner} --config {p.relative_to(ROOT)}",
                         "done": str((OUT / f"{name}.json").relative_to(ROOT)),
                         "min_free_gb": 30.0})
    m = CFG.parent / "smoke_regmean_real_manifest.json"
    m.write_text(json.dumps(manifest, indent=1))
    print("wrote", m, len(manifest), "cells")


if __name__ == "__main__":
    main()

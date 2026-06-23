"""E3 — One downstream-metric eval cell.

Like run_eval_cell.py but evaluates a downstream metric (accuracy /
pass@1 / COMET / IFEval) instead of NLL. Re-applies the merge for each
cell so the pipeline matches the matrix cells exactly.

Config schema (same as eval_matrix_seeds with three new fields):
  metric:               one of {gsm8k_em, humaneval_pass1, comet22, ifeval_strict}
  metric_kwargs:        per-metric extras (e.g. max_new_tokens)
  metric_task_spec:     adapter_spec entry whose task_cfg defines the eval split
                        (typically gsm8k for gsm8k_em, magicoder for humaneval_pass1)
  n_eval_metric:        sample count (default: 1000 for production, 200 for smoke)
"""

# Unsloth must import before transformers.
import unsloth  # noqa: F401

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel

from merging.peft_model_view import PeftModelView
from merging.registry import REGISTRY, DEFAULT_KWARGS
from training.data_loaders import load_task_split
from utils.io_helpers import write_result
from utils.pbs_env import assert_env

from eval.downstream_metrics import METRIC_FNS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    env_info = assert_env(min_free_gb=cfg.get("min_free_gb", 25.0))
    print(f"[downstream_cell] env OK: {env_info}", flush=True)

    project_root = Path(os.environ["PROJECT_ROOT"])
    base_path = cfg["base_model"]
    method = cfg["method"]
    method_kwargs = dict(DEFAULT_KWARGS.get(method, {}))
    method_kwargs.update(cfg.get("method_kwargs", {}))

    metric = cfg["metric"]
    metric_kwargs = cfg.get("metric_kwargs", {})
    n_eval_metric = int(cfg.get("n_eval_metric", 1000))
    metric_task_name = cfg["metric_task_spec"]

    # Load base + first task adapter (plain loader required for mixed-rank
    # methods like rd_encoder rank_deff).
    t_load = time.time()
    if cfg.get("loader") == "plain":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[downstream_cell] loading base+{cfg['adapter_specs'][0]['name']} (plain)",
              flush=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"], torch_dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa",
            low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(
            model, cfg["adapter_specs"][0]["dir"], adapter_name="default")
        model.eval()
    else:
        print(f"[downstream_cell] loading base+{cfg['adapter_specs'][0]['name']} (unsloth)",
              flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["adapter_specs"][0]["dir"],
            max_seq_length=cfg.get("max_seq_length", 2048),
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
    print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)

    for spec in cfg["adapter_specs"][1:]:
        print(f"[downstream_cell] loading adapter {spec['name']}", flush=True)
        model.load_adapter(spec["dir"], adapter_name=spec["name"])
    print(f"  peft_config keys: {list(model.peft_config)}", flush=True)

    # Apply merge
    view = PeftModelView(model)
    adapter_names = ["default"] + [s["name"] for s in cfg["adapter_specs"][1:]]
    n_tasks = len(adapter_names)
    weights = cfg.get("weights", [1.0 / n_tasks] * n_tasks)
    merged_name = "__merged__"
    print(f"[downstream_cell] merging via {method} kwargs={method_kwargs}",
          flush=True)
    t_merge = time.time()
    REGISTRY[method](view, adapter_names, weights, merged_name, **method_kwargs)
    print(f"  merge done in {time.time()-t_merge:.1f}s", flush=True)
    model.set_adapter(merged_name)

    # Locate the eval split for the metric's task. Two schemas supported:
    #   (a) metric_task_spec names an adapter in adapter_specs (existing E3
    #       behavior — eval comes from one of the merged tasks)
    #   (b) metric_task_cfg is a top-level override (B4 HumanEval, where the
    #       eval data is from a different dataset than any merged adapter)
    if "metric_task_cfg" in cfg:
        target_task_cfg = cfg["metric_task_cfg"]
    else:
        target_spec = next(s for s in cfg["adapter_specs"]
                           if s["name"] == metric_task_name)
        target_task_cfg = target_spec["task_cfg"]
    _train, ev_full = load_task_split(target_task_cfg,
                                       seed=cfg.get("seed", 20260518))
    ev = ev_full[:n_eval_metric]
    print(f"[downstream_cell] {metric} on {metric_task_name} "
          f"n={len(ev)} (of {len(ev_full)} available)", flush=True)

    if metric not in METRIC_FNS:
        raise ValueError(f"unknown metric {metric!r}")
    metric_fn = METRIC_FNS[metric]

    t_eval = time.time()
    score, per_example = metric_fn(model, tokenizer, ev, **metric_kwargs)
    print(f"[downstream_cell] {metric}={score:.4f} in {time.time()-t_eval:.0f}s",
          flush=True)

    # Write JSON
    results_path = project_root / cfg["output_path"]
    payload = {
        "method": method,
        "method_kwargs": method_kwargs,
        "weights": weights,
        "base_model": base_path,
        "adapter_dirs": [s["dir"] for s in cfg["adapter_specs"]],
        "metric": metric,
        "metric_kwargs": metric_kwargs,
        "metric_task": metric_task_name,
        "metric_score": score,
        "n_eval_metric": len(ev),
        "per_example": per_example,
        "env": env_info,
    }
    write_result(results_path, payload, config=cfg)
    print(f"\n[downstream_cell] wrote {results_path}", flush=True)
    print(f"[downstream_cell] {metric}={score:.4f}", flush=True)
    print(f"DOWNSTREAM_CELL_DONE method={method} metric={metric}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

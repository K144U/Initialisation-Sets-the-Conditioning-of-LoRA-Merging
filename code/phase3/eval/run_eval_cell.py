"""One cell of the Phase 3 eval matrix.

Input: a YAML config naming (base_model, 4 task adapter dirs, 1 merging method,
optional rate_bits). The script loads everything, applies the merge, evaluates
NLL on all 4 task held-outs, and writes one JSON.

Each cell is independent — Day 5's submit_eval_matrix.sh fans these out.
"""

# Unsloth must import before transformers.
import unsloth  # noqa: F401

import argparse
import gc
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


def compute_nll(model, tokenizer, eval_data: list[dict], max_len: int) -> tuple[float, list[dict]]:
    """Mean NLL nats per assistant-response token. Template-agnostic.

    Returns (mean_nll, per_example) where per_example is a list of
    {"nll": float|None, "n_answer": int, "skipped": bool} dicts, one per
    eval_data row in order. This per-example info enables bootstrap CIs
    downstream (resample examples, recompute weighted mean of nll with
    weights = n_answer). Added 2026-05-19 per user direction (no
    compute shyness; per-example arrays are basically free).

    Same logic as training/train_lora.py:_compute_nll — duplicated here to keep
    eval/ standalone and avoid cross-package imports of Unsloth-dependent modules.
    """
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    per_example: list[dict] = []
    model.eval()
    with torch.no_grad():
        for ex in eval_data:
            user_msgs = [{"role": "user", "content": ex["prompt"]}]
            full_msgs = user_msgs + [{"role": "assistant", "content": ex["answer"]}]
            prompt_text = tokenizer.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=max_len, add_special_tokens=False,
            ).input_ids.to(device)
            prompt_len = prompt_ids.shape[1]
            if prompt_len >= ids.shape[1]:
                per_example.append({"nll": None, "n_answer": 0, "skipped": True, "reason": "no_answer_tokens_after_truncation"})
                continue
            labels = ids.clone()
            labels[:, :prompt_len] = -100
            n_answer = int((labels != -100).sum().item())
            if n_answer == 0:
                per_example.append({"nll": None, "n_answer": 0, "skipped": True, "reason": "n_answer_zero"})
                continue
            out = model(input_ids=ids, labels=labels)
            ex_nll = float(out.loss.item())
            total_nll += ex_nll * n_answer
            total_tokens += n_answer
            per_example.append({"nll": ex_nll, "n_answer": n_answer, "skipped": False})
    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    return mean_nll, per_example


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    env_info = assert_env(min_free_gb=cfg.get("min_free_gb", 25.0))
    print(f"[eval_cell] env OK: {env_info}", flush=True)

    project_root = Path(os.environ["PROJECT_ROOT"])
    base_path = cfg["base_model"]
    method = cfg["method"]
    method_kwargs = dict(DEFAULT_KWARGS.get(method, {}))
    method_kwargs.update(cfg.get("method_kwargs", {}))

    # 1. Load base + the first task adapter via Unsloth (sets up apply_qkv etc.)
    print(f"[eval_cell] loading base+{cfg['adapter_specs'][0]['name']} via Unsloth from "
          f"{cfg['adapter_specs'][0]['dir']}", flush=True)
    t_load = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["adapter_specs"][0]["dir"],
        max_seq_length=cfg.get("max_seq_length", 1024),
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)

    # First adapter is "default" inside PEFT; rename via load_adapter for clarity
    # actually keep "default" as the first task name to avoid a rename dance
    first_task = cfg["adapter_specs"][0]["name"]
    print(f"  peft_config keys initially: {list(model.peft_config)}", flush=True)

    # 2. Load remaining task adapters
    for spec in cfg["adapter_specs"][1:]:
        print(f"[eval_cell] loading adapter {spec['name']} from {spec['dir']}", flush=True)
        model.load_adapter(spec["dir"], adapter_name=spec["name"])
    print(f"  peft_config keys now: {list(model.peft_config)}", flush=True)

    # 3. Pre-eval each individual tau (so we can compute excess later)
    eval_data: dict[str, list[dict]] = {}
    for spec in cfg["adapter_specs"]:
        # Use cfg task hooks to load eval data
        task_cfg = spec["task_cfg"]
        _train, ev = load_task_split(task_cfg, seed=cfg.get("seed", 20260518))
        eval_data[spec["name"]] = ev
        print(f"  task {spec['name']}: n_eval={len(ev)}", flush=True)

    max_len = cfg.get("max_seq_length", 1024)

    nll_tau: dict[str, dict[str, float]] = {}   # nll_tau[state][task]
    per_example_tau: dict[str, dict[str, list[dict]]] = {}  # per_example[state][task] = [...]
    # Evaluate each task-specific adapter on each task
    print(f"[eval_cell] evaluating tau adapters", flush=True)
    state_names_for_nll_tau = ["base"] + [s["name"] for s in cfg["adapter_specs"]]
    for state in state_names_for_nll_tau:
        if state == "base":
            ctx = model.disable_adapter()
        else:
            model.set_adapter(state if state != cfg["adapter_specs"][0]["name"] else "default")
            ctx = _NullCtx()
        with ctx:
            nll_tau[state] = {}
            per_example_tau[state] = {}
            for task_name, ev in eval_data.items():
                v, per_ex = compute_nll(model, tokenizer, ev, max_len)
                nll_tau[state][task_name] = v
                per_example_tau[state][task_name] = per_ex
                print(f"  {state:<12} on {task_name:<14} nll={v:.4f}", flush=True)

    # 4. Apply the merging method
    print(f"[eval_cell] merging via {method} kwargs={method_kwargs}", flush=True)
    view = PeftModelView(model)
    # Use PEFT's internal adapter names: first task is "default", rest are by spec name
    adapter_names = ["default"] + [s["name"] for s in cfg["adapter_specs"][1:]]
    n_tasks = len(adapter_names)
    weights = cfg.get("weights", [1.0 / n_tasks] * n_tasks)
    merged_name = "__merged__"
    t_merge = time.time()
    REGISTRY[method](view, adapter_names, weights, merged_name, **method_kwargs)
    print(f"  merge done in {time.time()-t_merge:.1f}s", flush=True)

    # 5. Eval merged model on each task
    print(f"[eval_cell] evaluating merged adapter", flush=True)
    model.set_adapter(merged_name)
    nll_merged: dict[str, float] = {}
    per_example_merged: dict[str, list[dict]] = {}
    for task_name, ev in eval_data.items():
        v, per_ex = compute_nll(model, tokenizer, ev, max_len)
        nll_merged[task_name] = v
        per_example_merged[task_name] = per_ex
        print(f"  merged on {task_name:<14} nll={v:.4f}", flush=True)

    # 6. Compute excess = NLL_merged - NLL_tau_t
    excess: dict[str, float] = {}
    for spec in cfg["adapter_specs"]:
        task_name = spec["name"]
        tau_state = task_name if task_name != cfg["adapter_specs"][0]["name"] else cfg["adapter_specs"][0]["name"]
        # nll_tau is keyed by state (which == task name); excess against tau's own task
        excess[task_name] = nll_merged[task_name] - nll_tau[tau_state][task_name]

    worst_excess = max(excess.values())
    avg_excess = sum(excess.values()) / len(excess)

    # 7. Write JSON
    results_path = project_root / cfg["output_path"]
    payload = {
        "method": method,
        "method_kwargs": method_kwargs,
        "weights": weights,
        "base_model": base_path,
        "adapter_dirs": [s["dir"] for s in cfg["adapter_specs"]],
        "task_names": [s["name"] for s in cfg["adapter_specs"]],
        "nll_tau": nll_tau,           # nll_tau[state][task] (mean over n_eval)
        "nll_merged": nll_merged,      # nll_merged[task] (mean over n_eval)
        "per_example_nll_tau": per_example_tau,
        "per_example_nll_merged": per_example_merged,
        "excess_per_task": excess,
        "worst_task_excess": worst_excess,
        "avg_task_excess": avg_excess,
        "n_eval_per_task": {n: len(e) for n, e in eval_data.items()},
        "env": env_info,
    }
    write_result(results_path, payload, config=cfg)

    print(f"\n[eval_cell] wrote {results_path}", flush=True)
    print(f"[eval_cell] worst_task_excess={worst_excess:.4f}  avg={avg_excess:.4f}", flush=True)
    print(f"EVAL_CELL_DONE method={method}", flush=True)
    return 0


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    sys.exit(main())

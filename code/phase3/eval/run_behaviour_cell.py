"""One public cohort: task-free behaviour-preservation excess of several merges.

Rules: notes/prereg_public_merge_2026-09-23.md. For each member adapter t,
greedy completions y_t (<= max_new_tokens) on a fixed prompt set are generated
from base + adapter t. For a merge m,

    excess_t(m) = NLL_m(y_t | prompt) - NLL_t(y_t | prompt)

in nats per completion token, and worst = max_t excess_t(m). Public adapters
rarely ship their training data, so this asks only how well a merge preserves
each adapter's own behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from merging.peft_model_view import PeftModelView
from merging.registry import REGISTRY, DEFAULT_KWARGS
from training.data_loaders import load_task_split
from utils.io_helpers import write_result
from utils.pbs_env import assert_env

PROMPT_TASK = {"name": "alpaca", "dataset": "yahma/alpaca-cleaned", "config": None,
               "split_train": "train", "split_eval": "train", "n_train": 7500,
               "n_eval": 1000, "prompt_field": "instruction", "answer_field": "output"}


def fmt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
    return prompt + "\n"


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new: int, bs: int = 16):
    tokenizer.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tokenizer(prompts[i:i + bs], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for row in gen[:, enc.input_ids.shape[1]:]:
            ids = row.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
            ids = [t for t in ids if t != tokenizer.pad_token_id or t == tokenizer.eos_token_id]
            outs.append(ids)
    return outs


@torch.no_grad()
def nll(model, tokenizer, prompts: list[str], completions: list[list[int]]) -> float:
    tot, n = 0.0, 0
    for p, c in zip(prompts, completions):
        if not c:
            continue
        pid = tokenizer(p, add_special_tokens=False).input_ids
        ids = torch.tensor([pid + c], device=model.device)
        labels = ids.clone()
        labels[:, :len(pid)] = -100
        out = model(input_ids=ids, labels=labels)
        tot += float(out.loss) * len(c)
        n += len(c)
    return tot / n if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    cfg = yaml.safe_load(ap.parse_args().config.read_text())
    env = assert_env(min_free_gb=cfg.get("min_free_gb", 30.0))
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", attn_implementation="sdpa")
    names = ["default"] + [f"a{i}" for i in range(1, len(cfg["adapters"]))]
    model = PeftModel.from_pretrained(model, cfg["adapters"][0], adapter_name="default")
    for nm, repo in zip(names[1:], cfg["adapters"][1:]):
        model.load_adapter(repo, adapter_name=nm)
    model.eval()
    print(f"[behaviour] loaded base + {len(names)} adapters in {time.time() - t0:.0f}s", flush=True)

    _, ev = load_task_split(PROMPT_TASK, seed=cfg["seed"])
    prompts = [fmt(tok, ex["prompt"]) for ex in ev[: cfg["n_prompts"]]]

    refs, self_nll = {}, {}
    for nm in names:
        model.set_adapter(nm)
        refs[nm] = generate(model, tok, prompts, cfg["max_new_tokens"])
        self_nll[nm] = nll(model, tok, prompts, refs[nm])
        print(f"  {nm}: {sum(map(len, refs[nm]))} ref tokens, self NLL {self_nll[nm]:.4f}", flush=True)

    view = PeftModelView(model)
    weights = [1.0 / len(names)] * len(names)
    results = {}
    for tag, (method, kw) in cfg["methods"].items():
        kwargs = dict(DEFAULT_KWARGS.get(method, {}), **kw)
        t = time.time()
        try:
            REGISTRY[method](view, names, weights, "__merged__", **kwargs)
            model.set_adapter("__merged__")
            exc = {nm: nll(model, tok, prompts, refs[nm]) - self_nll[nm] for nm in names}
            results[tag] = {"excess": exc, "worst": max(exc.values()),
                            "mean": sum(exc.values()) / len(exc)}
            print(f"  {tag:<9} worst {results[tag]['worst']:.4f} ({time.time() - t:.0f}s)", flush=True)
        except Exception as e:  # recorded, not hidden: the analyzer counts it
            results[tag] = {"error": f"{type(e).__name__}: {e}"[:500]}
            print(f"  {tag:<9} ERROR {results[tag]['error']}", flush=True)
        if "__merged__" in model.peft_config:
            model.delete_adapter("__merged__")

    payload = {"base_model": cfg["base_model"], "adapters": cfg["adapters"],
               "regime": cfg["regime"], "audit_cos": cfg["audit_cos"],
               "self_nll": self_nll, "n_ref_tokens": {k: sum(map(len, v)) for k, v in refs.items()},
               "results": results, "env": env}
    write_result(Path(cfg["output_path"]), payload, config=cfg)
    print("BEHAVIOUR_CELL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

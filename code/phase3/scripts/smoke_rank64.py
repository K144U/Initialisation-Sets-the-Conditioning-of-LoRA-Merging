"""GPU smoke test for the rd_encoder v3 rank-d_eff adapter realization.

Validates the NEW plumbing (add_adapter_rank + exact W* factors) on the
real unsloth/PEFT stack before any batch cells are queued -- the lesson of
the v2 base-patch failure, whose CPU test passed while the GPU path was
broken.

Loads llama + 4 adapters exactly like run_eval_cell, merges with
realize="rank_deff" bits=32, evaluates gsm8k on n=50, and compares to the
v1 (rank-16) cell's merged NLL on the same task. PASS if the absolute NLL
is in a sane LM range and not worse than v1 by more than 0.3 nats (v2's
breakage was +8 nats, so the discrimination margin is enormous).

Run inside a PBS job with CUDA_VISIBLE_DEVICES set.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import unsloth  # noqa: F401
import torch
import yaml
from unsloth import FastLanguageModel

from eval.run_eval_cell import compute_nll
from merging.peft_model_view import PeftModelView
from merging.registry import REGISTRY
from training.data_loaders import load_task_split

ROOT = "/home/sanjay.g/projects/rdmerge"
CFG = f"{ROOT}/code/phase3/configs/eval_e1/llama31_8b__rd_b32.yaml"
N_SMOKE = 50


def main():
    cfg = yaml.safe_load(open(CFG))
    print("[smoke] loading base + adapters", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["adapter_specs"][0]["dir"],
        max_seq_length=cfg.get("max_seq_length", 1024),
        dtype=torch.bfloat16, load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    for spec in cfg["adapter_specs"][1:]:
        model.load_adapter(spec["dir"], adapter_name=spec["name"])

    view = PeftModelView(model)
    names = ["default"] + [s["name"] for s in cfg["adapter_specs"][1:]]
    REGISTRY["rd_encoder"](view, names, [0.25] * 4, "rd64",
                           bits=32, realize="rank_deff", seed=20260611)
    model.set_adapter("rd64")

    task_cfg = cfg["adapter_specs"][0]["task_cfg"]   # gsm8k
    _train, ev = load_task_split(task_cfg, seed=cfg.get("seed", 20260518))
    ev = ev[:N_SMOKE]
    nll, _per = compute_nll(model, tokenizer, ev,
                            cfg.get("max_seq_length", 1024))
    print(f"[smoke] rank-deff merged NLL on gsm8k[:{N_SMOKE}] = {nll:.4f}",
          flush=True)

    v1 = json.load(open(
        f"{ROOT}/results/phase3/eval_e1/llama31_8b__rd_b32.json"))
    v1_nll = v1["nll_merged"]["gsm8k"]
    print(f"[smoke] v1 (rank-16) merged NLL on gsm8k (n=1000) = {v1_nll:.4f}",
          flush=True)
    if nll < v1_nll + 0.3 and nll < 2.5:
        print("SMOKE PASS: rank-deff path is sane "
              f"(delta vs v1 = {nll - v1_nll:+.3f})", flush=True)
        return 0
    print(f"SMOKE FAIL: nll={nll:.4f} vs v1={v1_nll:.4f}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

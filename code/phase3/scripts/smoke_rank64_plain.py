"""Smoke v3b: rank-d_eff realization on PLAIN transformers+PEFT (no unsloth).

Two checks:
  1. Stored-weights check: after add_adapter_rank, the materialized delta
     of the merged adapter must equal the offline W* (pins the v3a failure
     on unsloth's forward, not on the stored factors).
  2. NLL check: merged NLL on gsm8k[:50] within sane range of the v1 cell.

PASS on both -> the 12 full-rank cells run with a plain-PEFT loader.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.run_eval_cell import compute_nll
from merging.peft_model_view import PeftModelView
from merging.registry import REGISTRY
from training.data_loaders import load_task_split

ROOT = "/home/sanjay.g/projects/rdmerge"
CFG = f"{ROOT}/code/phase3/configs/eval_e1/llama31_8b__rd_b32.yaml"
BASE = f"{ROOT}/models/Llama-3.1-8B-Instruct"
N_SMOKE = 50


def main():
    cfg = yaml.safe_load(open(CFG))
    print("[smoke3b] loading PLAIN base + adapters", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(
        model, cfg["adapter_specs"][0]["dir"], adapter_name="default")
    for spec in cfg["adapter_specs"][1:]:
        model.load_adapter(spec["dir"], adapter_name=spec["name"])
    model.eval()

    view = PeftModelView(model)
    names = ["default"] + [s["name"] for s in cfg["adapter_specs"][1:]]
    REGISTRY["rd_encoder"](view, names, [0.25] * 4, "rd64",
                           bits=32, realize="rank_deff", seed=20260611)

    # Check 1: stored weights realize W* (spot-check 3 layers)
    import math
    layers = view.layer_names()
    spot = [layers[0], layers[len(layers) // 2], layers[-1]]
    for ly in spot:
        deltas = [view.get_delta(n, ly).to(torch.float32) for n in names]
        Vs = []
        for d in deltas:
            _, _, vh = torch.linalg.svd(d.float(), full_matrices=False)
            Vs.append(vh[:16, :].T)
        Mw = torch.cat([V / math.sqrt(4) for V in Vs], dim=1)
        G = Mw.T @ Mw
        S, U = torch.linalg.eigh(G)
        keep = S > 1e-6 * S.max()
        S, U = S[keep], U[:, keep]
        Q = Mw @ U @ torch.diag(S.rsqrt())
        tau = (sum(deltas) / 4) @ (Q @ torch.diag(1.0 / S) @ Q.T)
        got = view.get_delta("rd64", ly).to(torch.float32)
        rel = float((got - tau).norm() / tau.norm())
        print(f"[smoke3b] {ly}: stored-vs-W* rel={rel:.2e}", flush=True)
        # bf16 storage of factors costs ~1e-2 relative; the v3a/v2 failures
        # were O(1) wrong, so 5e-2 separates cleanly.
        assert rel < 5e-2, f"stored weights wrong at {ly}: rel={rel}"
    print("[smoke3b] stored-weights check PASS", flush=True)

    model.set_adapter("rd64")
    task_cfg = cfg["adapter_specs"][0]["task_cfg"]
    _train, ev = load_task_split(task_cfg, seed=cfg.get("seed", 20260518))
    ev = ev[:N_SMOKE]
    nll, _ = compute_nll(model, tokenizer, ev, cfg.get("max_seq_length", 1024))
    v1_nll = json.load(open(
        f"{ROOT}/results/phase3/eval_e1/llama31_8b__rd_b32.json"))[
        "nll_merged"]["gsm8k"]
    print(f"[smoke3b] plain-PEFT rank-deff NLL={nll:.4f} "
          f"(v1 unsloth rank-16: {v1_nll:.4f})", flush=True)
    if nll < v1_nll + 0.3 and nll < 2.5:
        print(f"SMOKE PASS (delta vs v1 = {nll - v1_nll:+.3f})", flush=True)
        return 0
    print("SMOKE FAIL", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

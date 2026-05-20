"""Day-3.7 GPU smoke: load two real Llama-3.1-8B LoRA adapters, run each of
the 5 merging methods, verify (a) no exception, (b) merged adapter writes
back into PeftModel with the right rank.

Runs as a 1-minute PBS job. No NLL eval here (that's Day 4).
"""

# Unsloth must import before transformers.
import unsloth  # noqa: F401

import json
import os
import sys
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from merging.peft_model_view import PeftModelView
from merging.registry import REGISTRY, DEFAULT_KWARGS
from utils.pbs_env import assert_env


# Use the 1B smoke adapters: they're the cheapest real PEFT-compatible
# adapters we have, and the smoke only validates the code path (not 8B-
# specific behavior). 8B + 4096-dim fp32 SVD per layer pushes past
# walltime; 1B + 2048-dim runs in ~2 min for all 5 methods.
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
GSM_DIR = PROJECT_ROOT / "artifacts/lora/llama32_1b_smoke/gsm8k"
ALP_DIR = PROJECT_ROOT / "artifacts/lora/llama32_1b_smoke/alpaca"
RESULTS = PROJECT_ROOT / "results/phase3/smoke/merging_gpu_smoke.json"


def main() -> int:
    env_info = assert_env(min_free_gb=20.0)
    print(f"[merge_smoke] env OK: {env_info}", flush=True)

    print(f"[merge_smoke] loading base + gsm8k via Unsloth", flush=True)
    t0 = time.time()
    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(GSM_DIR),
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    print(f"  loaded in {time.time()-t0:.1f}s; adapters={list(model.peft_config)}", flush=True)

    print(f"[merge_smoke] attaching alpaca adapter", flush=True)
    model.load_adapter(str(ALP_DIR), adapter_name="alpaca")
    print(f"  adapters now = {list(model.peft_config)}", flush=True)

    # PEFT names the first-loaded adapter "default"; rename mentally to "gsm8k"
    adapter_names = ["default", "alpaca"]
    weights = [0.5, 0.5]

    view = PeftModelView(model)
    n_layers = len(view.layer_names())
    print(f"[merge_smoke] view sees {n_layers} LoRA layers", flush=True)
    if n_layers == 0:
        print("ERROR: no LoRA layers found via PeftModelView", file=sys.stderr)
        return 2

    # Pre-flight: confirm get_delta works on a single layer
    sample_layer = view.layer_names()[0]
    d = view.get_delta("default", sample_layer)
    print(f"  sample delta {sample_layer}: shape={tuple(d.shape)} dtype={d.dtype} norm={d.norm():.4e}", flush=True)

    results: list[dict] = []
    for method_name, fn in REGISTRY.items():
        merged_name = f"merged_{method_name}"
        # add the rate axis for TVQ
        kwargs = dict(DEFAULT_KWARGS[method_name])
        if method_name == "tvq":
            kwargs["rate_bits"] = 4
        t_method = time.time()
        try:
            fn(view, adapter_names, weights, merged_name, **kwargs)
            elapsed = time.time() - t_method
            # validate: merged adapter exists, sample layer has finite values and right rank
            assert merged_name in model.peft_config, f"merged adapter {merged_name} not registered in peft_config"
            f = view._layers[sample_layer]
            A = f.lora_A[merged_name].weight
            B = f.lora_B[merged_name].weight
            assert torch.isfinite(A).all().item(), f"NaN/Inf in merged A"
            assert torch.isfinite(B).all().item(), f"NaN/Inf in merged B"
            eff_rank = torch.linalg.matrix_rank((B.float() @ A.float()).cpu()).item()
            out_dim, in_dim, r = view.layer_topology[sample_layer]
            assert eff_rank <= r, f"rank {eff_rank} > r={r}"
            results.append({
                "method": method_name,
                "ok": True,
                "elapsed_s": round(elapsed, 2),
                "sample_layer": sample_layer,
                "sample_rank": int(eff_rank),
                "sample_max_abs_A": float(A.abs().max()),
                "sample_max_abs_B": float(B.abs().max()),
            })
            print(f"PASS  {method_name:<18} elapsed={elapsed:.1f}s  rank={eff_rank}/{r}", flush=True)
        except Exception as e:
            elapsed = time.time() - t_method
            import traceback as tb
            results.append({
                "method": method_name,
                "ok": False,
                "elapsed_s": round(elapsed, 2),
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb.format_exc(),
            })
            print(f"FAIL  {method_name:<18} elapsed={elapsed:.1f}s  err={type(e).__name__}: {e}", flush=True)

    n_pass = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_pass

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "env": env_info,
        "n_layers": n_layers,
        "adapter_names": adapter_names,
        "results": results,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "wallclock_s": round(time.time() - t0, 1),
    }, indent=2))

    print(f"\n[merge_smoke] wrote {RESULTS}", flush=True)
    print(f"[merge_smoke] {n_pass}/{len(results)} methods passed", flush=True)
    if n_fail == 0:
        print("MERGING_GPU_SMOKE_GREEN", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

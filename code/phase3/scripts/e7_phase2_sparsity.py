"""E7 Phase 2a — TVQ b=2 implicit sparsity per model + correlation with dip depth.

Master plan Prediction 1 (implicit-TIES hypothesis, sparsity branch):
  At b=2 quantization, a non-trivial fraction of LoRA-delta coordinates
  are "implicitly zeroed" (their quantized value lands in the bucket
  containing 0). The dip depth (b=4 worst-excess - b=2 worst-excess)
  should correlate with this sparsity fraction across the 4 models.

We report three sparsity metrics per (model, task) because the master
plan doesn't pin a single definition:
  metric A (bucket): frac whose b=2 dequantized value falls in the
                     same quantization bucket that contains 0
  metric B (small):  frac whose |post-quant| < 0.5 * |pre-quant|
  metric C (lost):   frac whose |post-quant - pre-quant| > |pre-quant|
                     (quantization moved them past zero / sign-flip)

Aggregates per model (mean over the 4 tasks), then correlate with each
model's mean dip depth. CPU only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
ADAPTER_ROOT = PROJECT_ROOT / "artifacts/lora"
PHASE1_JSON = PROJECT_ROOT / "results/phase3/e7_phase1_correlation.json"
OUT_JSON = PROJECT_ROOT / "results/phase3/e7_phase2_sparsity.json"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASKS = ["gsm8k", "alpaca", "magicoder", "translation", "flores"]

BITS = 2


def materialize_delta(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Load A, B for each lora layer; return {module -> scale * B @ A} (fp32)."""
    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        # older format may use adapter_model.bin
        bin_path = adapter_dir / "adapter_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"no adapter file in {adapter_dir}")
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        state = load_file(str(st_path), device="cpu")

    # Read scaling from adapter_config.json (lora_alpha / r)
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    scaling = cfg["lora_alpha"] / cfg["r"]

    # Find A/B pairs by stripping the lora-specific suffix
    A_by_mod: dict[str, torch.Tensor] = {}
    B_by_mod: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if "lora_A" in k:
            mod = k.replace(".lora_A.weight", "").replace(".weight", "")
            A_by_mod[mod] = v.to(torch.float32)
        elif "lora_B" in k:
            mod = k.replace(".lora_B.weight", "").replace(".weight", "")
            B_by_mod[mod] = v.to(torch.float32)

    deltas: dict[str, torch.Tensor] = {}
    for mod in A_by_mod:
        if mod in B_by_mod:
            deltas[mod] = scaling * (B_by_mod[mod] @ A_by_mod[mod])
    return deltas


def tvq_quantize(delta: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor min-max uniform scalar quantization (matches merging/tvq.py)."""
    if bits >= 32:
        return delta
    levels = (1 << bits) - 1
    min_val = delta.min()
    max_val = delta.max()
    if (max_val - min_val).abs() < 1e-12:
        return delta
    step = (max_val - min_val) / levels
    q = torch.round((delta - min_val) / step)
    return q * step + min_val


def sparsity_metrics(delta: torch.Tensor, bits: int) -> dict[str, float]:
    """Compute metrics A/B/C for one tensor."""
    q = tvq_quantize(delta, bits)
    # bucket containing 0:
    min_val = delta.min()
    max_val = delta.max()
    rng = (max_val - min_val).item()
    if rng < 1e-12:
        return {"A": 1.0, "B": 1.0, "C": 0.0}
    levels = (1 << bits) - 1
    step = rng / levels
    # bucket index of 0: floor((0 - min) / step)
    k0 = int(math.floor((0.0 - min_val.item()) / step))
    k0 = max(0, min(levels - 1, k0))
    # Post-quant value for bucket k: k * step + min_val
    zero_bucket_val = k0 * step + min_val.item()
    # Metric A: frac with post-quant value == zero_bucket_val (i.e., in bucket k0)
    tol = step * 0.5
    in_zero_bucket = (q - zero_bucket_val).abs() < tol
    frac_A = float(in_zero_bucket.float().mean())
    # Metric B: |post-quant| < 0.5 * |pre-quant|
    eps = 1e-12
    frac_B = float(((q.abs() + eps) < 0.5 * (delta.abs() + eps)).float().mean())
    # Metric C: |post - pre| > |pre|  (quantization moved past zero or large)
    frac_C = float(((q - delta).abs() > delta.abs()).float().mean())
    return {"A": frac_A, "B": frac_B, "C": frac_C}


def find_task_v1_dir(model: str, task: str) -> Path | None:
    """Returns artifacts/lora/<model>/<task>/v1 if it exists, else flores fallback."""
    p = ADAPTER_ROOT / model / task / "v1"
    if p.exists():
        return p
    # translation may be stored under "flores"
    if task == "translation":
        p = ADAPTER_ROOT / model / "flores" / "v1"
        if p.exists():
            return p
    return None


def main() -> int:
    phase1 = json.load(open(PHASE1_JSON))
    per_model_dip = {m: phase1["per_model"][m]["mean_dip"] for m in MODELS}

    per_model_metrics: dict[str, dict] = {}
    for model in MODELS:
        per_task_A: list[float] = []
        per_task_B: list[float] = []
        per_task_C: list[float] = []
        for task in ("gsm8k", "alpaca", "magicoder", "translation"):
            adir = find_task_v1_dir(model, task)
            if adir is None:
                print(f"[warn] missing v1 adapter for {model}/{task}",
                      file=sys.stderr)
                continue
            print(f"  loading {model}/{task}", flush=True)
            deltas = materialize_delta(adir)
            layer_metrics = [sparsity_metrics(d, BITS) for d in deltas.values()]
            mean_A = sum(m["A"] for m in layer_metrics) / len(layer_metrics)
            mean_B = sum(m["B"] for m in layer_metrics) / len(layer_metrics)
            mean_C = sum(m["C"] for m in layer_metrics) / len(layer_metrics)
            per_task_A.append(mean_A)
            per_task_B.append(mean_B)
            per_task_C.append(mean_C)
        per_model_metrics[model] = {
            "A_zero_bucket": sum(per_task_A) / len(per_task_A),
            "B_small_post_quant": sum(per_task_B) / len(per_task_B),
            "C_quant_lost_signal": sum(per_task_C) / len(per_task_C),
            "mean_dip": per_model_dip[model],
        }
        print(f"  -> {model}: A={per_model_metrics[model]['A_zero_bucket']:.3f} "
              f"B={per_model_metrics[model]['B_small_post_quant']:.3f} "
              f"C={per_model_metrics[model]['C_quant_lost_signal']:.3f}",
              flush=True)

    # Pearson correlation across 4 models
    def pearsonr(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
        dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
        return num / (dx * dy) if dx > 0 and dy > 0 else 0.0

    dips = [per_model_metrics[m]["mean_dip"] for m in MODELS]
    rA = pearsonr([per_model_metrics[m]["A_zero_bucket"] for m in MODELS], dips)
    rB = pearsonr([per_model_metrics[m]["B_small_post_quant"] for m in MODELS], dips)
    rC = pearsonr([per_model_metrics[m]["C_quant_lost_signal"] for m in MODELS], dips)

    rs = {"A": rA, "B": rB, "C": rC}
    out = {
        "bits": BITS,
        "per_model": per_model_metrics,
        "pearson_A_vs_dip": rA,
        "pearson_B_vs_dip": rB,
        "pearson_C_vs_dip": rC,
        "decision_rule": {
            "threshold": 0.7,
            "best_metric": max(rs, key=lambda k: abs(rs[k])),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    print(f"\nE7 Phase 2a — TVQ b={BITS} sparsity ↔ dip depth across 4 models")
    print(f"{'model':<14}{'A':<8}{'B':<8}{'C':<8}{'mean_dip':<12}")
    for m in MODELS:
        s = per_model_metrics[m]
        print(f"{m:<14}{s['A_zero_bucket']:<8.3f}"
              f"{s['B_small_post_quant']:<8.3f}{s['C_quant_lost_signal']:<8.3f}"
              f"{s['mean_dip']:<+12.4f}")
    print()
    print(f"  Pearson r(A vs dip) = {rA:+.3f}")
    print(f"  Pearson r(B vs dip) = {rB:+.3f}")
    print(f"  Pearson r(C vs dip) = {rC:+.3f}")
    print(f"  best |r|: {out['decision_rule']['best_metric']}")
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

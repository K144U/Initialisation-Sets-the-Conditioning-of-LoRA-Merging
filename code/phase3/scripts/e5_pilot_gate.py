"""E5 Arm 2 pilot gate analysis.

After the 12 pilot trainings complete (4 tasks x 3 alpha = 12 adapters),
load the 4 alpha=0.9 adapters and the 4 alpha=0.0 (baseline) adapters,
compute d_eff/(Tr) per layer for each alpha, and fire the gate decision:

  GO if   alpha=0.9 -> d_eff/(Tr) < 0.8 in MAJORITY of layers (>64/128)
  NO-GO otherwise (Arm 3 geometric forcing becomes the primary arm)

Also reports alpha=0.5 as a midpoint for the monotonicity check.

Outputs:
  - results/phase3/e5_pilot_gate.json   — machine-readable verdict
  - prints decision to stdout
  - intended to be appended to decisions.md by the operator
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/sanjay.g/projects/rdmerge/code/phase3")

from eval.deff_analysis import (
    load_adapter_factors,
    compute_layer_metrics,
)

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
QWEN_BASE = PROJECT_ROOT / "models/Qwen2.5-7B-Instruct"
TASKS = ("gsm8k", "alpaca", "magicoder", "translation")
ALPHAS = (0, 50, 90)
LORA_RANK = 16


def adapter_dir(task: str, alpha_pct: int) -> Path:
    return (PROJECT_ROOT / "artifacts/lora/qwen25_7b/e5_pilot"
            / f"{task}__alpha{alpha_pct:02d}/v1")


def analyze_alpha(alpha_pct: int) -> dict:
    """Compute per-layer d_eff/(Tr) for the 4 adapters at this alpha."""
    print(f"\n=== alpha={alpha_pct/100:.2f} ===", flush=True)
    # Load all 4 task adapters at this alpha
    per_task_factors: list[dict[str, dict[str, "torch.Tensor"]]] = []
    scales: list[dict[str, float]] = []
    for task in TASKS:
        d = adapter_dir(task, alpha_pct)
        if not d.exists():
            raise FileNotFoundError(f"missing adapter: {d}")
        f, s = load_adapter_factors(str(d))
        per_task_factors.append(f)
        scales.append(s)

    # Layers in common (should be identical across tasks since same base)
    layers = sorted(per_task_factors[0].keys())
    Tr = len(TASKS) * LORA_RANK   # = 64

    per_layer = []
    for layer in layers:
        metrics = compute_layer_metrics(
            layer, per_task_factors, scales, list(TASKS), rank=LORA_RANK,
        )
        if "error" in metrics:
            print(f"  [warn] {layer}: {metrics['error']}", flush=True)
            continue
        per_layer.append(metrics)

    deff_frac = [m["d_eff_over_Tr"] for m in per_layer]
    soft_deff_frac = [m["soft_d_eff_over_Tr"] for m in per_layer]
    n_layers = len(deff_frac)
    n_below_0p8 = sum(1 for v in deff_frac if v < 0.8)
    n_below_0p9 = sum(1 for v in deff_frac if v < 0.9)
    return {
        "alpha": alpha_pct / 100.0,
        "n_layers": n_layers,
        "d_eff_over_Tr_mean": sum(deff_frac) / n_layers,
        "d_eff_over_Tr_min": min(deff_frac),
        "d_eff_over_Tr_max": max(deff_frac),
        "soft_d_eff_over_Tr_mean": sum(soft_deff_frac) / n_layers,
        "soft_d_eff_over_Tr_min": min(soft_deff_frac),
        "n_layers_below_0p8": n_below_0p8,
        "n_layers_below_0p9": n_below_0p9,
        "per_layer": per_layer,
    }


def gate_decision(by_alpha: dict[int, dict]) -> str:
    """Pre-registered: GO if alpha=0.9 has d_eff/(Tr) < 0.8 in majority of layers."""
    a90 = by_alpha[90]
    if a90["n_layers_below_0p8"] > a90["n_layers"] / 2:
        return "GO"
    return "NO-GO"


def main() -> int:
    out_path = PROJECT_ROOT / "results/phase3/e5_pilot_gate.json"
    by_alpha = {}
    for a in ALPHAS:
        by_alpha[a] = analyze_alpha(a)
        s = by_alpha[a]
        print(f"  d_eff/(Tr) mean={s['d_eff_over_Tr_mean']:.3f} "
              f"min={s['d_eff_over_Tr_min']:.3f} "
              f"max={s['d_eff_over_Tr_max']:.3f}", flush=True)
        print(f"  layers < 0.8: {s['n_layers_below_0p8']}/{s['n_layers']}",
              flush=True)
        print(f"  soft d_eff/(Tr) mean={s['soft_d_eff_over_Tr_mean']:.3f}",
              flush=True)

    decision = gate_decision(by_alpha)
    print(f"\n*** GATE DECISION: {decision} ***", flush=True)
    print(f"  alpha=0.9: {by_alpha[90]['n_layers_below_0p8']}/"
          f"{by_alpha[90]['n_layers']} layers below 0.8 threshold")
    if decision == "GO":
        print("  -> proceed to Arm 2 main run (5-alpha sweep, 3 seeds)")
    else:
        print("  -> fall back to Arm 3 geometric forcing; "
              "report Arm 2 null as a finding")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": decision,
        "by_alpha": {str(k): {k2: v2 for k2, v2 in v.items()
                              if k2 != "per_layer"}
                     for k, v in by_alpha.items()},
        "thresholds": {"d_eff_over_Tr": 0.8, "majority_fraction": 0.5},
        "per_layer_by_alpha": {str(k): v["per_layer"]
                                for k, v in by_alpha.items()},
    }
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

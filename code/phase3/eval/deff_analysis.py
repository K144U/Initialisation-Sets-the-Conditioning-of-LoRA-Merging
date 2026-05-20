"""Tier-1 item A + A2 — empirical d_eff (hard) and soft d_eff (participation ratio).

For each (model, task) LoRA adapter, materialize the per-layer delta
Δ_t^ℓ = scaling · B_t^ℓ @ A_t^ℓ, extract V_t^ℓ as the top-r right-singular
subspace of Δ_t^ℓ. Then compute two metrics per layer:

  HARD: d_eff^ℓ = rank(Σ_t P_{V_t^ℓ})  =  rank([V_1 | ... | V_T])

  SOFT (T1.A2): soft_d_eff^ℓ = (Σ σ_i²)² / Σ σ_i⁴   (participation ratio of
        singular values σ_i of M = [V_1 | ... | V_T] ∈ R^{in_dim × T·r}).

The bound's floor prediction is

    floor_predicted^ℓ = B^2 · (1 - d_eff^ℓ / (T·r))

where B is the LoRA-delta norm bound (we take per-layer B = max_t ||Δ_t^ℓ||_F,
the empirical worst-case norm). Hard d_eff saturates at Tr generically; the
soft variant ranges continuously in [1, Tr] and is sensitive to subspace
overlap angle even when the V_t's remain linearly independent.

Output: a JSON summary + figures (per-layer hard/soft d_eff hists +
soft-d_eff-vs-observed-excess scatter). Runs CPU-only.
"""

# Unsloth import not needed — pure PEFT loading, no training
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")

# (model_key, base_path, list of (task_name, adapter_dir))
ADAPTER_SETS = [
    (
        "llama31_8b",
        "/home/sanjay.g/projects/rdmerge/models/Llama-3.1-8B-Instruct",
        [
            ("gsm8k",       "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/gsm8k/v1"),
            ("alpaca",      "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/alpaca/v1"),
            ("magicoder",   "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/magicoder/v1"),
            ("translation", "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/flores/v1"),
        ],
    ),
    (
        "qwen25_7b",
        "/home/sanjay.g/projects/rdmerge/models/Qwen2.5-7B-Instruct",
        [
            ("gsm8k",       "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/gsm8k/v1"),
            ("alpaca",      "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/alpaca/v1"),
            ("magicoder",   "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/magicoder/v1"),
            ("translation", "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/flores/v1"),
        ],
    ),
    # T1.D / T1.C addition 2026-05-19: 2 more architecture families
    (
        "mistral_7b",
        "/home/sanjay.g/projects/rdmerge/models/Mistral-7B-Instruct-v0.3",
        [
            ("gsm8k",       "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/gsm8k/v1"),
            ("alpaca",      "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/alpaca/v1"),
            ("magicoder",   "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/magicoder/v1"),
            ("translation", "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/flores/v1"),
        ],
    ),
    (
        "yi15_9b",
        "/home/sanjay.g/projects/rdmerge/models/Yi-1.5-9B-Chat",
        [
            ("gsm8k",       "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/gsm8k/v1"),
            ("alpaca",      "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/alpaca/v1"),
            ("magicoder",   "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/magicoder/v1"),
            ("translation", "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/flores/v1"),
        ],
    ),
]
TASK_NAMES_ORDER = ["gsm8k", "alpaca", "magicoder", "translation"]
RANK = 16   # LoRA rank used in training
T_TASKS = 4


def load_adapter_factors(adapter_dir: str) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    """Read adapter_model.safetensors + adapter_config.json.

    Returns:
      factors: {module_qualified_name: {'A': Tensor, 'B': Tensor}}
      scaling: {module_qualified_name: float}  # alpha/r per the config
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    weights_path = Path(adapter_dir) / "adapter_model.safetensors"
    cfg = json.loads(cfg_path.read_text())
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = alpha / r

    weights = load_file(str(weights_path))
    # Keys look like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    factors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    scaling: dict[str, float] = {}
    for k, v in weights.items():
        if k.endswith(".lora_A.weight"):
            module = k[: -len(".lora_A.weight")]
            factors[module]["A"] = v.to(torch.float32)
            scaling[module] = scale
        elif k.endswith(".lora_B.weight"):
            module = k[: -len(".lora_B.weight")]
            factors[module]["B"] = v.to(torch.float32)
            scaling[module] = scale
    # Sanity: every module has both A and B
    for module, f in factors.items():
        assert "A" in f and "B" in f, f"missing A or B for {module}"
    return dict(factors), scaling


def compute_layer_metrics(
    deltas_by_task: dict[str, torch.Tensor],   # task -> (out_dim, in_dim) on device
    rank: int,
) -> dict:
    """For one layer, compute V_t per task, d_eff, predicted floor, etc.

    Uses top-r truncated SVD (svd_lowrank, ~50x faster than full SVD on
    4096-dim matrices). Tensors stay on whatever device they're given.
    """
    tasks = sorted(deltas_by_task.keys())
    T = len(tasks)
    V_list = []
    norms = []
    for t in tasks:
        delta = deltas_by_task[t]
        norms.append(float(delta.norm()))
        try:
            # top-r truncated SVD: returns U (out,r), S (r,), V (in,r)
            # — V is what we want (right-singular subspace).
            _U, _S, V_t = torch.svd_lowrank(delta, q=rank, niter=4)
        except Exception as e:
            return {"error": f"svd_failed: {type(e).__name__}", "tasks": tasks}
        # V_t shape: (in_dim, rank) — already orthonormal columns
        V_list.append(V_t)

    # rank(P_total) = rank(M) where M = [V_1 | V_2 | ... | V_T] (shape: in_dim x T·r).
    # Computing SVD of M is O(in_dim · (T·r)²) — much cheaper than the in_dim x in_dim
    # eigvalsh on P_total. (Sum of T rank-r projectors has rank at most T·r.)
    in_dim = V_list[0].shape[0]
    M = torch.cat(V_list, dim=1)          # (in_dim, T·r)
    S = torch.linalg.svdvals(M)            # (T·r,) singular values
    tol = float(S[0]) * S.numel() * torch.finfo(S.dtype).eps * 100
    d_eff = int((S > tol).sum().item())

    # T1.A2 — SOFT d_eff: participation ratio of squared singular values
    # PR = (Σ σ_i²)² / Σ σ_i⁴
    # = Tr when σ_i are all equal (mutually orthogonal V_t's, the "most independent"
    #   case; e.g. random-Stiefel with T·r ≤ in_dim).
    # < Tr when overlap concentrates energy in fewer singular directions.
    # Bounds: 1 ≤ soft_d_eff ≤ Tr.
    S2 = S * S
    pr_num = float(S2.sum()) ** 2
    pr_den = float((S2 * S2).sum())
    soft_d_eff = pr_num / pr_den if pr_den > 0 else 0.0

    # B^2 — per-layer worst-case norm-squared across tasks
    B2 = max(n ** 2 for n in norms)

    # Predicted floor for this layer using HARD d_eff (theorem's literal formula).
    Tr = T * rank
    if d_eff >= Tr:
        floor_predicted = 0.0   # all subspaces span the full Tr-space, no floor
    else:
        floor_predicted = B2 * (1.0 - d_eff / Tr)

    # SOFT floor: same formula with soft_d_eff in place of hard d_eff.
    # Empirically motivated; the theorem proves the hard version.
    floor_predicted_soft = B2 * max(0.0, 1.0 - soft_d_eff / Tr)

    return {
        "tasks": tasks,
        "T": T,
        "rank": rank,
        "d_eff": d_eff,
        "soft_d_eff": soft_d_eff,
        "Tr": Tr,
        "d_eff_over_Tr": d_eff / Tr,
        "soft_d_eff_over_Tr": soft_d_eff / Tr,
        "norms_per_task": norms,
        "B2_layer": B2,
        "floor_predicted_layer": floor_predicted,
        "floor_predicted_soft_layer": floor_predicted_soft,
        "in_dim": int(in_dim),
        "out_dim": int(deltas_by_task[tasks[0]].shape[0]),
    }


def analyze_one_model(model_key: str, base_path: str, adapters: list) -> dict:
    print(f"[deff] analyzing {model_key}", flush=True)
    # Load each task's per-module factors
    per_task_factors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    scaling_per_task: dict[str, dict[str, float]] = {}
    for task_name, adapter_dir in adapters:
        print(f"  loading {task_name} from {adapter_dir}", flush=True)
        f, s = load_adapter_factors(adapter_dir)
        per_task_factors[task_name] = f
        scaling_per_task[task_name] = s

    # Intersect module names across tasks (should be identical for same model)
    common_modules = set.intersection(*(set(f.keys()) for f in per_task_factors.values()))
    common_modules = sorted(common_modules)
    print(f"  {len(common_modules)} common LoRA modules across tasks", flush=True)

    per_layer = []
    for module in common_modules:
        deltas = {}
        for task_name in TASK_NAMES_ORDER:
            f = per_task_factors[task_name][module]
            A, B = f["A"], f["B"]
            scale = scaling_per_task[task_name][module]
            deltas[task_name] = scale * (B @ A)
        m = compute_layer_metrics(deltas, RANK)
        m["module"] = module
        per_layer.append(m)

    # Aggregate
    Tr = T_TASKS * RANK
    d_eff_vals = [m["d_eff"] for m in per_layer]
    soft_d_eff_vals = [m["soft_d_eff"] for m in per_layer]
    floor_predicted_per_layer = [m["floor_predicted_layer"] for m in per_layer]
    floor_predicted_soft_per_layer = [m["floor_predicted_soft_layer"] for m in per_layer]
    B2_per_layer = [m["B2_layer"] for m in per_layer]

    agg = {
        "n_layers": len(per_layer),
        "T": T_TASKS,
        "rank": RANK,
        "Tr": Tr,
        "d_eff_mean": float(np.mean(d_eff_vals)),
        "d_eff_median": float(np.median(d_eff_vals)),
        "d_eff_min": int(min(d_eff_vals)),
        "d_eff_max": int(max(d_eff_vals)),
        "d_eff_over_Tr_mean": float(np.mean(d_eff_vals)) / Tr,
        "soft_d_eff_mean": float(np.mean(soft_d_eff_vals)),
        "soft_d_eff_median": float(np.median(soft_d_eff_vals)),
        "soft_d_eff_min": float(min(soft_d_eff_vals)),
        "soft_d_eff_max": float(max(soft_d_eff_vals)),
        "soft_d_eff_over_Tr_mean": float(np.mean(soft_d_eff_vals)) / Tr,
        "B2_mean": float(np.mean(B2_per_layer)),
        "floor_predicted_per_layer_mean": float(np.mean(floor_predicted_per_layer)),
        "floor_predicted_per_layer_sum": float(np.sum(floor_predicted_per_layer)),
        "floor_predicted_soft_per_layer_mean": float(np.mean(floor_predicted_soft_per_layer)),
        "floor_predicted_soft_per_layer_sum": float(np.sum(floor_predicted_soft_per_layer)),
    }
    return {"model": model_key, "per_layer": per_layer, "aggregate": agg}


def compare_to_observed(model_key: str, predicted: dict, eval_dir: Path) -> dict:
    """Compare predicted floor to observed worst_excess for non-TVQ methods."""
    methods = ["task_arithmetic", "ties", "dare", "knots"]
    observed = {}
    for meth in methods:
        f = eval_dir / f"{model_key}__{meth}.json"
        if f.exists():
            d = json.loads(f.read_text())
            observed[meth] = {
                "worst_excess": d["worst_task_excess"],
                "avg_excess": d["avg_task_excess"],
            }
    # TVQ at b=32 = no-quantization baseline
    f32 = eval_dir / f"{model_key}__tvq_b32.json"
    if f32.exists():
        observed["tvq_b32"] = {
            "worst_excess": json.loads(f32.read_text())["worst_task_excess"],
        }
    return {"observed": observed, "predicted_aggregate": predicted["aggregate"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        default=str(PROJECT_ROOT / "results/phase3/eval_matrix_n1k_v2"),
        help="Directory of eval_matrix JSONs to compare against",
    )
    parser.add_argument(
        "--out-json",
        default=str(PROJECT_ROOT / "results/phase3/deff_analysis.json"),
    )
    parser.add_argument(
        "--out-fig",
        default=str(PROJECT_ROOT / "code/phase3/figures/deff_vs_floor.png"),
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    # Fall back to n=200 if n=1k_v2 not ready
    if not (eval_dir / "llama31_8b__task_arithmetic.json").exists():
        eval_dir = PROJECT_ROOT / "results/phase3/eval_matrix"
        print(f"[deff] n=1k_v2 eval not found, using {eval_dir}", flush=True)

    all_models: dict = {}
    for model_key, base_path, adapters in ADAPTER_SETS:
        t0 = time.time()
        analysis = analyze_one_model(model_key, base_path, adapters)
        comparison = compare_to_observed(model_key, analysis, eval_dir)
        all_models[model_key] = {
            "analysis": analysis,
            "comparison": comparison,
            "elapsed_s": time.time() - t0,
        }
        agg = analysis["aggregate"]
        obs = comparison["observed"]
        print(f"\n[deff] {model_key}:")
        print(f"  hard d_eff per layer: mean={agg['d_eff_mean']:.1f} median={agg['d_eff_median']:.1f} range=[{agg['d_eff_min']}, {agg['d_eff_max']}]")
        print(f"  hard d_eff/(Tr): {agg['d_eff_over_Tr_mean']:.3f} (Tr = {agg['Tr']})")
        print(f"  SOFT d_eff per layer: mean={agg['soft_d_eff_mean']:.2f} median={agg['soft_d_eff_median']:.2f} range=[{agg['soft_d_eff_min']:.2f}, {agg['soft_d_eff_max']:.2f}]")
        print(f"  soft d_eff/(Tr): {agg['soft_d_eff_over_Tr_mean']:.3f}")
        print(f"  hard predicted floor per layer (mean): {agg['floor_predicted_per_layer_mean']:.4f}")
        print(f"  soft predicted floor per layer (mean): {agg['floor_predicted_soft_per_layer_mean']:.4f}")
        if obs:
            we = {k: v.get('worst_excess') for k, v in obs.items()}
            print(f"  observed worst_excess: {we}")
        print(f"  elapsed: {time.time() - t0:.1f}s", flush=True)

    # Save
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(all_models, indent=2, default=str))
    print(f"\n[deff] wrote {args.out_json}", flush=True)

    # Plot: 2x2 grid
    # (top-left) hard d_eff per-layer hist across all models
    # (top-right) SOFT d_eff per-layer hist across all models
    # (bottom-left) soft d_eff/(Tr) mean vs observed worst_excess (TA) — should correlate
    # (bottom-right) per-model mean soft d_eff bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "llama31_8b":  "#1f77b4",
        "qwen25_7b":   "#ff7f0e",
        "mistral_7b":  "#2ca02c",
        "yi15_9b":     "#d62728",
    }
    model_order = list(all_models.keys())

    ax = axes[0, 0]
    for model_key in model_order:
        d_effs = [m["d_eff"] for m in all_models[model_key]["analysis"]["per_layer"]]
        ax.hist(d_effs, bins=20, alpha=0.5, label=model_key,
                color=colors.get(model_key, "gray"))
    ax.axvline(T_TASKS * RANK, color="black", linestyle="--", alpha=0.5, label=f"Tr={T_TASKS*RANK}")
    ax.set_xlabel("hard d_eff per layer (= rank of stacked V)")
    ax.set_ylabel("count of layers")
    ax.set_title("Hard d_eff per-layer distribution (saturates at Tr generically)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for model_key in model_order:
        sd = [m["soft_d_eff"] for m in all_models[model_key]["analysis"]["per_layer"]]
        ax.hist(sd, bins=30, alpha=0.5, label=model_key,
                color=colors.get(model_key, "gray"))
    ax.axvline(T_TASKS * RANK, color="black", linestyle="--", alpha=0.5, label=f"Tr={T_TASKS*RANK}")
    ax.set_xlabel("SOFT d_eff per layer (participation ratio of stacked-V σ's)")
    ax.set_ylabel("count of layers")
    ax.set_title("Soft d_eff per-layer distribution (continuous in [1, Tr])")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Soft d_eff/Tr vs observed worst_excess for TA — does the continuous metric
    # correlate with the actual merging difficulty?
    ax = axes[1, 0]
    for model_key in model_order:
        agg = all_models[model_key]["analysis"]["aggregate"]
        comp = all_models[model_key]["comparison"]["observed"]
        x = agg["soft_d_eff_over_Tr_mean"]
        if "task_arithmetic" in comp:
            y = comp["task_arithmetic"]["worst_excess"]
            ax.scatter(x, y, color=colors.get(model_key, "gray"), s=120,
                       label=model_key, edgecolor="black")
            ax.annotate(model_key, (x, y), fontsize=8, xytext=(5, 5),
                        textcoords="offset points")
    ax.set_xlabel("mean soft d_eff / (T·r)  ←  more overlap   |   more independence  →")
    ax.set_ylabel("observed worst_excess (TA, at b=32)")
    ax.set_title("Soft d_eff vs observed TA worst_excess (F2 mechanism check)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Per-model soft d_eff bar chart for orientation
    ax = axes[1, 1]
    means = [all_models[m]["analysis"]["aggregate"]["soft_d_eff_mean"] for m in model_order]
    ax.bar(model_order, means, color=[colors.get(m, "gray") for m in model_order],
           edgecolor="black")
    ax.axhline(T_TASKS * RANK, color="black", linestyle="--", alpha=0.5, label=f"Tr={T_TASKS*RANK}")
    ax.set_ylabel("mean soft d_eff per layer")
    ax.set_title("Mean soft d_eff per architecture")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("d_eff analysis (4 models × 4 tasks): hard saturates at Tr, soft varies continuously",
                 fontsize=12)
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=130, bbox_inches="tight")
    print(f"[deff] wrote {args.out_fig}", flush=True)
    print("DEFF_ANALYSIS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

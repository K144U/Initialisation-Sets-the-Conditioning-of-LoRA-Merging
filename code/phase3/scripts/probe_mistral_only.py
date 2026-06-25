"""Mistral-7B-only TIES sign-election probe at T=4.

Verifies the R ≈ 0.041 value used in the Mistral T=7 pre-registration
(decisions.md 2026-06-25 evening). Loads the 4 v1 Mistral adapters
(alpaca, gsm8k, magicoder, flores), computes TIES trim at density=0.2,
and reports per-task win-share + win-share range.

CPU-only, faster than the full 3-base probe (no Yi/Llama crunching).
"""

import json
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT = Path("/home/sanjay.g/projects/rdmerge")
ADAPTERS = {
    "alpaca": "artifacts/lora/mistral_7b/alpaca/v1",
    "gsm8k": "artifacts/lora/mistral_7b/gsm8k/v1",
    "magicoder": "artifacts/lora/mistral_7b/magicoder/v1",
    "translation": "artifacts/lora/mistral_7b/flores/v1",
}
DENSITY = 0.2


def adapter_scaling(adir: Path) -> float:
    cfg = json.load(open(adir / "adapter_config.json"))
    return cfg["lora_alpha"] / cfg["r"]


def load_deltas(adir: Path) -> dict[str, torch.Tensor]:
    sd = load_file(str(adir / "adapter_model.safetensors"))
    s = adapter_scaling(adir)
    out = {}
    for k in sd:
        if "lora_A.weight" in k:
            base = k.replace(".lora_A.weight", "")
            A = sd[base + ".lora_A.weight"].float()
            B = sd[base + ".lora_B.weight"].float()
            out[base] = s * (B @ A)
    return out


def trim_topk(delta: torch.Tensor, density: float) -> torch.Tensor:
    flat = delta.abs().flatten()
    k = max(1, int(round(density * flat.numel())))
    thr = torch.topk(flat, k, largest=True).values[-1]
    return delta * (delta.abs() >= thr).to(delta.dtype)


def main() -> None:
    tasks = sorted(ADAPTERS.keys())
    T = len(tasks)
    print(f"Loading {T} Mistral adapters...", flush=True)
    per_task = {t: load_deltas(ROOT / ADAPTERS[t]) for t in tasks}
    layers = sorted(set.intersection(*(set(per_task[t].keys()) for t in tasks)))
    print(f"  {T} adapters, {len(layers)} shared layers", flush=True)
    print(f"  applying TIES trim density={DENSITY}, accumulating sign votes...",
          flush=True)

    win_count = {t: 0 for t in tasks}
    active_count = {t: 0 for t in tasks}
    n_active_total = 0

    for li, layer in enumerate(layers):
        deltas = torch.stack([per_task[t][layer] for t in tasks], dim=0)
        # TIES trim per task
        trimmed = torch.stack(
            [trim_topk(per_task[t][layer], DENSITY) for t in tasks], dim=0)
        # Elected sign per coordinate (magnitude-weighted = sign of sum)
        elected = torch.sign(trimmed.sum(dim=0))
        # Active coordinates: where elected sign is non-zero
        active_mask = (elected != 0)
        n_active_total += int(active_mask.sum().item())
        # Per-task: count active coords where this task's sign matches elected
        for ti, t in enumerate(tasks):
            task_sign = torch.sign(trimmed[ti])
            task_active = (task_sign != 0) & active_mask
            match = task_active & (task_sign == elected)
            win_count[t] += int(match.sum().item())
            active_count[t] += int(task_active.sum().item())
        if (li + 1) % 16 == 0:
            print(f"  layer {li+1}/{len(layers)}", flush=True)

    win_share = {t: win_count[t] / max(active_count[t], 1) for t in tasks}
    ws = list(win_share.values())
    R = max(ws) - min(ws)

    print(f"\n=== Mistral-7B-Instruct-v0.3 (T={T}, density={DENSITY}) ===")
    print(f"n_active_coords (elected non-zero, summed across layers): {n_active_total:,}")
    print(f"\nPer-task win share (fraction of task's active coords matching elected sign):")
    for t in sorted(tasks, key=lambda x: -win_share[x]):
        print(f"  {t:<12} {win_share[t]:.4f}")
    print(f"\nWin-share range R = {R:.4f}")
    print(f"  max task: {max(win_share, key=win_share.get)} (R_max = {max(ws):.4f})")
    print(f"  min task: {min(win_share, key=win_share.get)} (R_min = {min(ws):.4f})")
    print(f"\nTd2 prediction window R* in [0.025, 0.075] for T=4 at density 0.2.")
    if 0.025 <= R <= 0.075:
        print(f"  -> R = {R:.4f} INSIDE the predicted ambiguity window.")
        print(f"     Pre-registered prediction: Mistral T=7 should show TIES neither")
        print(f"     clearly inverting nor clearly winning. (Test pending T=7 sweep.)")
    elif R < 0.025:
        print(f"  -> R = {R:.4f} BELOW the predicted window.")
        print(f"     Pre-registered model predicts Mistral T=7: TIES should remain best.")
    else:
        print(f"  -> R = {R:.4f} ABOVE the predicted window.")
        print(f"     Pre-registered model predicts Mistral T=7: TIES should invert to worst.")

    out_path = ROOT / "results/phase3/mistral_t4_ties_probe.json"
    json.dump({
        "base": "mistral_7b",
        "T": T,
        "density": DENSITY,
        "tasks": tasks,
        "win_share": win_share,
        "win_share_range_R": R,
        "n_active_coords_total": n_active_total,
    }, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

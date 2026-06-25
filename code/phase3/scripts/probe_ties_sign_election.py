"""B1 + B3 — TIES sign-election mechanism probe (Yi + Llama, T=7).

Tests the §6.6 v2 conjecture that TIES sign-election consensus degrades
with T more on Yi than on Llama because of Yi-Chat's instruction-tuning
saturation (compressed per-task headroom amplifies sign-election noise).

For each (base, T=7) configuration:
  - Load 7 trained adapters
  - Per layer, compute Δ_t = scaling * B @ A
  - Apply TIES trim at density=0.2 (the default in code/phase3/merging/ties.py)
  - Tally per-coordinate signs across T tasks
  - Compute elected sign via 'total' (sign of magnitude-weighted sum)
  - Per task: fraction of (active) coords where sign matches the elected sign
  - Vote-margin distribution: |pos - neg| / (pos + neg) per non-zero coord
  - Mass-share distribution: |Δ_t| / Σ |Δ_t| per coord per task

Reports compact summary tables to stdout + writes
results/phase3/ties_sign_election_probe.json for any downstream analysis.

CPU-only. Expected runtime: a few minutes (loads 14 adapter files, no
NLL eval).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path("/home/sanjay.g/projects/rdmerge")
OUT = ROOT / "results/phase3/ties_sign_election_probe.json"

# Adapter pools — same task pool as E6 T=7_all cells
BASES = {
    "yi15_9b": {
        "alpaca": "artifacts/lora/yi15_9b/alpaca/v1",
        "codealpaca": "artifacts/lora/yi15_9b/e6_pilot/codealpaca/v1",
        "dolly": "artifacts/lora/yi15_9b/e6_pilot/dolly/v1",
        "gsm8k": "artifacts/lora/yi15_9b/gsm8k/v1",
        "magicoder": "artifacts/lora/yi15_9b/magicoder/v1",
        "translation": "artifacts/lora/yi15_9b/flores/v1",
        "xsum": "artifacts/lora/yi15_9b/e6_pilot/xsum/v1",
    },
    "llama31_8b": {
        "alpaca": "artifacts/lora/llama31_8b/alpaca/v1",
        "codealpaca": "artifacts/lora/llama31_8b/e6_pilot/codealpaca/v1",
        "dolly": "artifacts/lora/llama31_8b/e6_pilot/dolly/v1",
        "gsm8k": "artifacts/lora/llama31_8b/gsm8k/v1",
        "magicoder": "artifacts/lora/llama31_8b/magicoder/v1",
        "translation": "artifacts/lora/llama31_8b/flores/v1",
        "xsum": "artifacts/lora/llama31_8b/e6_pilot/xsum/v1",
    },
    # Mistral at T=4: only the 4 v1 adapters exist; pre-registered
    # Mistral T=7 prediction will use a separate probe after the 3
    # pilot adapters (codealpaca, dolly, xsum) are trained on Mistral.
    "mistral_7b_T4": {
        "alpaca": "artifacts/lora/mistral_7b/alpaca/v1",
        "gsm8k": "artifacts/lora/mistral_7b/gsm8k/v1",
        "magicoder": "artifacts/lora/mistral_7b/magicoder/v1",
        "translation": "artifacts/lora/mistral_7b/flores/v1",
    },
}

DENSITY = 0.2  # TIES default in code/phase3/merging/ties.py


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


def probe_one_base(base: str, paths: dict[str, str]) -> dict:
    print(f"\n=== {base} ===", flush=True)
    tasks = sorted(paths.keys())
    T = len(tasks)
    per_task = {t: load_deltas(ROOT / paths[t]) for t in tasks}
    layers = sorted(set.intersection(*(set(d.keys()) for d in per_task.values())))
    print(f"  T={T} tasks, {len(layers)} shared layers", flush=True)

    # Vote-split bucket: (T+1, T+1) torch tensor, indexed by (n_pos, n_neg).
    # Vectorized accumulation via bincount on flattened index = n_pos*(T+1)+n_neg.
    split_hist = torch.zeros((T + 1) * (T + 1), dtype=torch.int64)
    n_coords_active = 0
    n_coords_total = 0
    margin_sum = 0.0
    margin_count = 0
    per_task_win = torch.zeros(T, dtype=torch.int64)
    per_task_active = torch.zeros(T, dtype=torch.int64)
    mass_share_sum = torch.zeros(T, dtype=torch.float64)
    mass_share_count = 0

    for li, k in enumerate(layers):
        trimmed = torch.stack(
            [trim_topk(per_task[t][k], DENSITY) for t in tasks], dim=0
        )  # (T, out, in)
        sign_t = torch.sign(trimmed)
        nonzero_t = (sign_t != 0)

        n_active = nonzero_t.sum(dim=0)  # (out, in)
        active_mask = (n_active > 0)
        n_coords_total += int(n_active.numel())
        n_coords_active += int(active_mask.sum().item())

        n_pos = (sign_t > 0).sum(dim=0).to(torch.int64)
        n_neg = (sign_t < 0).sum(dim=0).to(torch.int64)

        # Vectorized vote-split histogram via bincount
        flat_idx = (n_pos * (T + 1) + n_neg).flatten()
        # Restrict to active coords only
        flat_idx = flat_idx[active_mask.flatten()]
        split_hist += torch.bincount(flat_idx, minlength=(T + 1) * (T + 1))

        # Margin
        pn_sum = (n_pos + n_neg).clamp_min(1)
        margin = (n_pos - n_neg).abs().float() / pn_sum.float()
        margin_sum += float(margin[active_mask].sum().item())
        margin_count += int(active_mask.sum().item())

        # Elected sign + per-task win/active counts
        elected = torch.sign(trimmed.sum(dim=0))
        for i in range(T):
            m = (sign_t[i] == elected) & (sign_t[i] != 0) & active_mask
            per_task_win[i] += int(m.sum().item())
            per_task_active[i] += int((nonzero_t[i] & active_mask).sum().item())

        # Mass share per task — vectorized
        abs_stack = trimmed.abs()
        denom = abs_stack.sum(dim=0)
        nz = (denom > 0)
        if nz.any():
            # share per task: |Δ_t[nz]| / denom[nz], mean per task accumulated
            share_per_task = (abs_stack[:, nz] / denom[nz].unsqueeze(0)).sum(dim=1)
            mass_share_sum += share_per_task.to(torch.float64)
            mass_share_count += int(nz.sum().item())

        if (li + 1) % 48 == 0:
            print(f"  layer {li+1}/{len(layers)} done", flush=True)

    # Summaries
    mean_margin = margin_sum / max(1, margin_count)
    win_share = {t: int(per_task_win[i]) / max(1, int(per_task_active[i]))
                 for i, t in enumerate(tasks)}
    mass_share = {t: float(mass_share_sum[i]) / max(1, mass_share_count)
                  for i, t in enumerate(tasks)}

    # Vote-split: collapse split_hist (length (T+1)^2, indexed by p*(T+1)+n)
    # into unanimous / thin-split / wide-split using vectorized ops
    H = split_hist.reshape(T + 1, T + 1)
    ps = torch.arange(T + 1).view(-1, 1)
    ns = torch.arange(T + 1).view(1, -1)
    active = (ps + ns) > 0
    unanimous_mask = active & ((ps == 0) | (ns == 0))
    diff = (ps - ns).abs()
    thin_mask = active & ~unanimous_mask & (diff <= 1)
    wide_mask = active & ~unanimous_mask & (diff > 1)
    unanimous = int(H[unanimous_mask].sum().item())
    split_thin = int(H[thin_mask].sum().item())
    split_wide = int(H[wide_mask].sum().item())
    total_active = unanimous + split_thin + split_wide
    pct = lambda x: 100.0 * x / max(1, total_active)

    print(f"  n_active_coords:    {n_coords_active:>10,}")
    print(f"  n_total_coords:     {n_coords_total:>10,}")
    print(f"  mean vote margin:   {mean_margin:.4f}")
    print(f"  unanimous:          {pct(unanimous):.2f}%  ({unanimous:,})")
    print(f"  split thin (±1):    {pct(split_thin):.2f}%  ({split_thin:,})")
    print(f"  split wide (>1):    {pct(split_wide):.2f}%  ({split_wide:,})")
    print(f"  per-task win share (elected sign / active coords for that task):")
    for t in tasks:
        print(f"    {t:<12} {win_share[t]:.4f}")
    print(f"  per-task mass share (mean of |Δ_t|/Σ|Δ_t| over active coords):")
    for t in tasks:
        print(f"    {t:<12} {mass_share[t]:.4f}")

    # Concentration: max win_share - min win_share (range)
    ws = list(win_share.values())
    ms = list(mass_share.values())
    print(f"  win-share range:   {max(ws) - min(ws):.4f}  (max={max(ws):.4f} min={min(ws):.4f})")
    print(f"  mass-share range:  {max(ms) - min(ms):.4f}  (max={max(ms):.4f} min={min(ms):.4f})")

    return {
        "base": base,
        "tasks": tasks,
        "T": T,
        "n_layers": len(layers),
        "n_coords_active": n_coords_active,
        "n_coords_total": n_coords_total,
        "density": DENSITY,
        "mean_vote_margin": mean_margin,
        "vote_split_pct": {
            "unanimous": pct(unanimous),
            "split_thin": pct(split_thin),
            "split_wide": pct(split_wide),
        },
        "per_task_win_share": win_share,
        "per_task_mass_share": mass_share,
        "win_share_range": max(ws) - min(ws),
        "mass_share_range": max(ms) - min(ms),
    }


def main() -> None:
    out = {}
    for base, paths in BASES.items():
        out[base] = probe_one_base(base, paths)
    print(f"\nCross-base comparison:")
    print(f"  metric                    yi15_9b      llama31_8b   delta(Llama-Yi)")
    for key, label in [
        ("mean_vote_margin", "mean_vote_margin"),
        ("win_share_range", "win_share_range"),
        ("mass_share_range", "mass_share_range"),
    ]:
        y = out["yi15_9b"][key]
        ll = out["llama31_8b"][key]
        print(f"  {label:<22} {y:>10.4f}  {ll:>10.4f}  {ll - y:>+10.4f}")
    for split in ("unanimous", "split_thin", "split_wide"):
        y = out["yi15_9b"]["vote_split_pct"][split]
        ll = out["llama31_8b"]["vote_split_pct"][split]
        print(f"  {split:<22} {y:>9.2f}%  {ll:>9.2f}%  {ll - y:>+9.2f}%")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

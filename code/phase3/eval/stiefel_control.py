"""Tier-1 item B — synthetic Stiefel-random control panel.

Generates fake LoRA factors with CONTROLLED subspace overlap:
- Sample V_t ∈ R^{in_dim x r} as Stiefel-random subspaces (orthonormal columns)
- Vary the controlled overlap parameter via projection-mixing:
    V_t = sqrt(1 - α) · V_t_independent + sqrt(α) · V_shared
  Higher α → more shared subspace → smaller d_eff/(Tr) → smaller floor.
- For each α, generate a FakePeftModel with the controlled LoRAs, run all 5
  merging methods, observe worst_excess, compare to the bound's floor prediction.

Output: plot of `observed_worst_excess` vs `predicted_floor` across α,
showing the bound is TIGHT on Stiefel-random data (matches floor closely)
and LOOSE on real LoRAs (real data has additional slack).

This is the standard "control panel" for RD lower-bound papers and exactly
what `notes/phase3_design.md` §8 Q5 asks for.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/sanjay.g/projects/rdmerge/code/phase3")
from merging.registry import REGISTRY  # noqa: E402
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel  # noqa: E402


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
OUT_JSON = PROJECT_ROOT / "results/phase3/stiefel_control.json"
OUT_FIG = PROJECT_ROOT / "code/phase3/figures/stiefel_control.png"


def stiefel_random(in_dim: int, r: int, rng: torch.Generator) -> torch.Tensor:
    """Sample a random orthonormal matrix V ∈ R^{in_dim x r} (V^T V = I_r).

    Standard construction: QR of a random Gaussian matrix.
    """
    M = torch.randn(in_dim, r, generator=rng, dtype=torch.float32)
    Q, _ = torch.linalg.qr(M)
    return Q


def make_controlled_overlap_model(
    in_dim: int,
    out_dim: int,
    r: int,
    T: int,
    overlap_alpha: float,    # 0 = orthogonal, 1 = identical
    delta_scale: float,
    rng: torch.Generator,
) -> tuple[FakePeftModel, dict]:
    """Synthesize T LoRA adapters whose subspaces share a controlled amount of overlap."""
    LAYER = "synthetic.layer.0"
    LAYER_SPECS = {LAYER: (out_dim, in_dim, r)}

    # Shared subspace V_shared (an orthonormal in_dim × r)
    V_shared = stiefel_random(in_dim, r, rng)
    # U_shared so the deltas have a controlled row-space too
    U_shared = stiefel_random(out_dim, r, rng)

    model = FakePeftModel(layer_topology=dict(LAYER_SPECS))
    actual_norms = []
    for t in range(T):
        V_indep = stiefel_random(in_dim, r, rng)
        U_indep = stiefel_random(out_dim, r, rng)
        # Mix the row-bases of A and column-bases of B
        # A_t corresponds to V_t^T (right-singular basis); B_t to U_t (left-singular basis)
        a_mix = (np.sqrt(1.0 - overlap_alpha) * V_indep + np.sqrt(overlap_alpha) * V_shared).T  # (r, in_dim)
        b_mix = (np.sqrt(1.0 - overlap_alpha) * U_indep + np.sqrt(overlap_alpha) * U_shared)    # (out_dim, r)
        # Scale to a fixed delta norm
        delta = b_mix @ a_mix
        norm = delta.norm()
        if norm > 1e-12:
            target_scale = delta_scale / norm
            a_mix = a_mix * np.sqrt(target_scale)
            b_mix = b_mix * np.sqrt(target_scale)
        actual_norms.append(float((b_mix @ a_mix).norm()))
        model.add_adapter(f"task_{t}", {LAYER: FakeLoraLayer(A=a_mix, B=b_mix, scaling=1.0)})

    return model, {"in_dim": in_dim, "out_dim": out_dim, "r": r, "T": T,
                   "overlap_alpha": overlap_alpha, "actual_delta_norms": actual_norms}


def compute_d_eff(model: FakePeftModel, layer: str) -> int:
    """Compute d_eff for one layer from the FakePeftModel's stored A factors.

    V_t = right-singular subspace of B_t @ A_t; for rank-r LoRA, V_t = row-space of A_t.
    """
    in_dim = model.layer_topology[layer][1]
    adapters = sorted(model.adapters.keys())
    P_total = torch.zeros(in_dim, in_dim, dtype=torch.float32)
    for ad in adapters:
        A = model.adapters[ad][layer].A   # (r, in_dim)
        # row-space basis: V = A^T orthonormalized
        Q, _ = torch.linalg.qr(A.T)
        P_total += Q @ Q.T
    eigs = torch.linalg.eigvalsh(P_total)
    tol = float(eigs.max()) * 100 * torch.finfo(eigs.dtype).eps * in_dim
    return int((eigs > tol).sum().item())


def predicted_floor(B2: float, d_eff: int, Tr: int) -> float:
    if d_eff >= Tr:
        return 0.0
    return B2 * (1.0 - d_eff / Tr)


def observed_worst_excess(
    model: FakePeftModel,
    method: str,
    layer: str,
    weights: list[float],
) -> float:
    """Run the merging method, return max_t ||tau_t - merged||^2 over layer's delta."""
    adapters = sorted(model.adapters.keys())
    REGISTRY[method](model, adapters, weights, "__merged__")
    merged_delta = model.get_delta("__merged__", layer)
    excesses = []
    for t in adapters:
        tau = model.get_delta(t, layer)
        diff = (tau - merged_delta).norm() ** 2
        excesses.append(float(diff))
    # cleanup so we can reuse the model
    model.delete_adapter("__merged__")
    return max(excesses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9,1.0")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    LAYER = "synthetic.layer.0"
    Tr = args.T * args.r

    results = []   # rows: {alpha, seed, method, d_eff, B2, predicted, observed}
    methods = ["task_arithmetic", "ties", "dare", "knots"]

    for alpha in alphas:
        print(f"[stiefel] alpha={alpha:.2f}", flush=True)
        for seed in range(args.n_seeds):
            rng = torch.Generator().manual_seed(20260518 + seed * 1000 + int(alpha * 1e6))
            model, info = make_controlled_overlap_model(
                args.in_dim, args.out_dim, args.r, args.T,
                alpha, args.delta_scale, rng,
            )
            d_eff = compute_d_eff(model, LAYER)
            B2 = max(n ** 2 for n in info["actual_delta_norms"])
            pred = predicted_floor(B2, d_eff, Tr)
            weights = [1.0 / args.T] * args.T

            for method in methods:
                obs = observed_worst_excess(model, method, LAYER, weights)
                results.append({
                    "alpha": alpha, "seed": seed, "method": method,
                    "d_eff": d_eff, "Tr": Tr, "d_eff_over_Tr": d_eff / Tr,
                    "B2": B2, "predicted_floor": pred,
                    "observed_worst_excess_sq": obs,
                })

    # Aggregate by (alpha, method)
    by_alpha_method: dict = {}
    for r in results:
        k = (r["alpha"], r["method"])
        by_alpha_method.setdefault(k, []).append(r)

    summary = []
    for (alpha, method), rows in sorted(by_alpha_method.items()):
        preds = np.array([r["predicted_floor"] for r in rows])
        obss  = np.array([r["observed_worst_excess_sq"] for r in rows])
        d_eff_vals = np.array([r["d_eff"] for r in rows])
        summary.append({
            "alpha": alpha,
            "method": method,
            "n_seeds": len(rows),
            "predicted_floor_mean": float(preds.mean()),
            "predicted_floor_std":  float(preds.std()),
            "observed_mean": float(obss.mean()),
            "observed_std":  float(obss.std()),
            "d_eff_mean":   float(d_eff_vals.mean()),
            "tightness_ratio": float(obss.mean() / max(preds.mean(), 1e-12)),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"params": vars(args), "summary": summary,
                                    "raw": results}, indent=2, default=str))
    print(f"\n[stiefel] wrote {OUT_JSON}", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"task_arithmetic": "#1f77b4", "ties": "#2ca02c",
              "dare": "#d62728", "knots": "#9467bd"}

    # Panel 1: observed vs predicted across alpha (for TA)
    ax = axes[0]
    for method in methods:
        xs = [s["predicted_floor_mean"] for s in summary if s["method"] == method]
        ys = [s["observed_mean"] for s in summary if s["method"] == method]
        errs = [s["observed_std"] for s in summary if s["method"] == method]
        ax.errorbar(xs, ys, yerr=errs, marker="o", linestyle="-",
                    label=method, color=colors[method], capsize=3)
    # y = x line (bound is tight)
    lo = min(min(s["predicted_floor_mean"] for s in summary), 0)
    hi = max(max(s["observed_mean"] for s in summary),
             max(s["predicted_floor_mean"] for s in summary))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x (bound tight)")
    ax.set_xlabel("predicted floor B²(1 − d_eff/(Tr))")
    ax.set_ylabel("observed worst-task ||τ−w||²")
    ax.set_title(f"Stiefel-random: bound tightness (T={args.T}, r={args.r}, dim={args.in_dim})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: tightness ratio vs d_eff/(Tr)
    ax = axes[1]
    for method in methods:
        xs = [s["d_eff_mean"] / s.get("Tr", Tr) if "Tr" in s else s["d_eff_mean"] / Tr
              for s in summary if s["method"] == method]
        ys = [s["tightness_ratio"] for s in summary if s["method"] == method]
        # fall back: use d_eff_mean/Tr
        xs = [s["d_eff_mean"] / Tr for s in summary if s["method"] == method]
        ax.plot(xs, ys, marker="o", linestyle="-", color=colors[method], label=method)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ratio = 1 (tight)")
    ax.set_xlabel("d_eff / (T·r) (controlled via Stiefel overlap)")
    ax.set_ylabel("observed / predicted (lower = bound is tighter)")
    ax.set_title("Bound-tightness ratio across overlap regime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"[stiefel] wrote {OUT_FIG}", flush=True)

    # stdout summary table
    print(f"\n=== Stiefel control summary (T={args.T}, r={args.r}, dim={args.in_dim}, n_seeds={args.n_seeds}) ===")
    print(f"{'alpha':>6}  {'method':<18}  {'d_eff':>6}  {'predicted':>10}  {'observed':>10}  {'ratio':>8}")
    for s in summary:
        print(f"{s['alpha']:>6.2f}  {s['method']:<18}  {s['d_eff_mean']:>6.1f}  "
              f"{s['predicted_floor_mean']:>10.4f}  {s['observed_mean']:>10.4f}  {s['tightness_ratio']:>8.3f}")

    print("\nSTIEFEL_CONTROL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

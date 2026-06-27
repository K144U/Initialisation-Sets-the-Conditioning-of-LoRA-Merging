"""Td2 synthetic controlled-overlap test (closes the §6.6 "future work" line).

CLAIM (from §6.6 / Appendix Td2): the TIES inversion -- TIES going from
competitive to the WORST method, exceeding Task Arithmetic -- is caused by the
per-task sign-election WIN-SHARE RANGE crossing the Td2 threshold regime, and it
is controllable by sign/subspace overlap ALONE, with NO base model involved.
This is the controlled-dial analogue of the real-adapter result that TIES inverts
on the saturated Yi base (win-share range 0.077) but not on the balanced Llama
(0.028) or Mistral (0.0326) bases.

METHOD (uses the EXACT paper code path; no reimplementation of the mechanism):
  * Generator (single knob rho in [0,1]): T synthetic task vectors live on a
    sparse k-coordinate active support of R^d (k ~ density*d, mirroring how LoRA
    deltas concentrate in a low-dim subspace, so the TIES top-density trim is
    near-lossless and does not itself penalise TIES at baseline). On the support
    all tasks share a high-consensus sign pattern s (mirroring the ~70%
    unanimous-coordinate consensus measured on real adapters), with magnitudes
    m[j]*(1 + sigma*z) (m a shared magnitude, z mild per-task jitter). The knob
    rho is the fraction of ACTIVE coordinates on which the single MINORITY task
    flips its sign against the consensus (a fixed nested random subset, so the
    curve is a paired monotone sweep). rho=0 -> all tasks agree (win-share range
    ~0, every merge represents every task, TIES ~ TA); rho large -> the minority
    is systematically outvoted, its win-share collapses, and TIES's disjoint sign
    election drives the merged vector away from it. The (T-1)-task majority
    always follows s.
  * TIES merge: trim density 0.2 + 'total' sign election + disjoint sign-matched
    mean, using _trim_topk and _elect_sign imported VERBATIM from
    code/phase3/merging/ties.py. Only the LoRA/SVD-rank wrapper is dropped (an
    orthogonal confound); the sign-election arithmetic is identical to merge_ties.
  * TA merge: mean of the FULL (untrimmed) task vectors (Ilharco task
    arithmetic, weights 1/T) -- the paper's TA baseline.
  * Excess (Lemma-2 quadratic surrogate, isotropic Hessian H=I):
    excess_t(theta) = ||theta - tau_t||^2 ; worst-excess = max_t. Reported for
    TIES and TA.
  * Win-share range: the EXACT tally from probe_ties_sign_election.py --
    active coords = union of trimmed supports; win_share_t = (#coords where task
    t is active and its sign == elected sign) / (#active coords for task t);
    range = max_t - min_t.

OUTPUTS: results/synthetic/td2_overlap_sweep/{summary.json, table.txt}
         paper_artifacts/figures/figure_td2_synthetic_overlap.{png,pdf}

RUN (login node, CPU only, ~seconds):
  PYTHONNOUSERSITE=1 .conda/envs/rdmerge/bin/python code/synthetic/td2_overlap_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
# code/phase3 holds the 'merging' package (code/ and code/phase3/ are not packages)
sys.path.insert(0, os.path.join(REPO, "code", "phase3"))
from merging.ties import _trim_topk, _elect_sign  # the paper's exact functions

RESULTS_DIR = os.path.join(REPO, "results", "synthetic", "td2_overlap_sweep")
FIG_DIR = os.path.join(REPO, "paper_artifacts", "figures")

DENSITY = 0.2          # TIES default in code/phase3/merging/ties.py
THRESH_LO, THRESH_HI = 0.025, 0.075   # Td2 predicted ambiguity window R*
SIGMA = 0.15           # mild per-task magnitude jitter (baseline excess floor)
BASE_SEED = 20260627
# Win-share ranges MEASURED on the real bases (probe_ties_sign_election.py),
# used to read the base-free synthetic mechanism at the real operating points.
# Yi inverts on real data; Llama and Mistral do not.
REAL_ANCHORS = [("Llama-3.1", 0.028, "no"), ("Mistral-7B", 0.0326, "no"),
                ("Yi-1.5-Chat", 0.077, "YES")]


def interp_at(target: float, xs: list[float], *ys: list[float]) -> tuple[float, ...]:
    """Linear-interpolate each y-series at x=target (xs assumed increasing)."""
    if target <= xs[0]:
        i, f = 1, 0.0
    elif target >= xs[-1]:
        i, f = len(xs) - 1, 1.0
    else:
        i = 1
        while xs[i] < target:
            i += 1
        f = (target - xs[i - 1]) / (xs[i] - xs[i - 1])
    return tuple(y[i - 1] + f * (y[i] - y[i - 1]) for y in ys)


# ---------------------------------------------------------------------------
# merges (TIES disjoint-merge arithmetic copied 1:1 from merge_ties, on vectors)
# ---------------------------------------------------------------------------
def merge_ties_vec(tau: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """tau: (T, d). Returns merged vector (d,). Same ops as merge_ties()."""
    T = tau.shape[0]
    trimmed = torch.stack([_trim_topk(tau[t], DENSITY) for t in range(T)], dim=0)  # (T,d)
    elected = _elect_sign(trimmed, "total")                  # (d,)
    sign_t = torch.sign(trimmed)
    match = ((sign_t == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)).to(tau.dtype)
    w = weights.view(-1, 1)
    weighted = trimmed * match * w
    denom = (match * w).sum(dim=0).abs()
    return torch.where(denom > 1e-12, weighted.sum(dim=0) / denom.clamp_min(1e-12),
                       torch.zeros_like(denom))


def merge_ta_vec(tau: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """TA = sum_t w_t tau_t on the FULL (untrimmed) task vectors."""
    return (tau * weights.view(-1, 1)).sum(dim=0)


def win_share_range(tau: torch.Tensor) -> float:
    """Exact replica of the per-task sign-election win-share tally in
    probe_ties_sign_election.py, applied to one (T, d) cohort."""
    T = tau.shape[0]
    trimmed = torch.stack([_trim_topk(tau[t], DENSITY) for t in range(T)], dim=0)
    sign_t = torch.sign(trimmed)
    nonzero = (sign_t != 0)
    active_mask = nonzero.sum(dim=0) > 0
    elected = torch.sign(trimmed.sum(dim=0))
    ws = []
    for i in range(T):
        win = ((sign_t[i] == elected) & (sign_t[i] != 0) & active_mask).sum().item()
        act = (nonzero[i] & active_mask).sum().item()
        ws.append(win / max(1, act))
    return max(ws) - min(ws)


def worst_excess(theta: torch.Tensor, tau: torch.Tensor) -> float:
    """max_t ||theta - tau_t||^2 (isotropic-H quadratic excess)."""
    return ((theta.unsqueeze(0) - tau) ** 2).sum(dim=1).max().item()


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def make_cohort(rho: float, s: torch.Tensor, m: torch.Tensor, z: torch.Tensor,
                A: torch.Tensor, perm_active: torch.Tensor, minority: int) -> torch.Tensor:
    """Sparse high-consensus cohort on active support A: every task follows sign
    pattern s; the minority flips its sign on a nested rho-fraction of the active
    coords. Off-support coords are exactly zero. Returns (T, d)."""
    T, d = z.shape
    k = A.shape[0]
    sign = s.unsqueeze(0).repeat(T, 1).clone()           # (T,d) signs, all = s
    n_flip = int(round(rho * k))
    if n_flip > 0:
        flip = A[perm_active[:n_flip]]
        sign[minority, flip] = -sign[minority, flip]
    mag = m.unsqueeze(0) * (1.0 + SIGMA * z).abs()       # (T,d) positive magnitudes
    mask = torch.zeros(d, dtype=z.dtype)
    mask[A] = 1.0
    return sign * mag * mask.unsqueeze(0)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def run(d: int, T: int, n_seeds: int, rhos: list[float], k_active: int) -> dict:
    weights = torch.full((T,), 1.0 / T, dtype=torch.float64)
    minority = T - 1

    R = len(rhos)
    ties_w = torch.zeros(R, n_seeds, dtype=torch.float64)
    ta_w = torch.zeros(R, n_seeds, dtype=torch.float64)
    wsr = torch.zeros(R, n_seeds, dtype=torch.float64)

    for seed in range(n_seeds):
        g = torch.Generator().manual_seed(BASE_SEED + seed)
        s = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int64) * 2 - 1).to(torch.float64)
        m = torch.randn(d, generator=g, dtype=torch.float64).abs() / (k_active ** 0.5)
        z = torch.randn(T, d, generator=g, dtype=torch.float64)
        perm = torch.randperm(d, generator=g)
        A = perm[:k_active]                              # sparse active support
        perm_active = torch.randperm(k_active, generator=g)
        for ri, rho in enumerate(rhos):
            tau = make_cohort(rho, s, m, z, A, perm_active, minority)
            ties_w[ri, seed] = worst_excess(merge_ties_vec(tau, weights), tau)
            ta_w[ri, seed] = worst_excess(merge_ta_vec(tau, weights), tau)
            wsr[ri, seed] = win_share_range(tau)

    def stat(x):
        return x.mean(dim=1), x.std(dim=1), 1.96 * x.std(dim=1) / (n_seeds ** 0.5)

    ties_m, ties_s, ties_ci = stat(ties_w)
    ta_m, ta_s, ta_ci = stat(ta_w)
    wsr_m, wsr_s, wsr_ci = stat(wsr)
    diff_m, diff_s, diff_ci = stat(ties_w - ta_w)

    wsr_l, ties_l, ta_l, diff_l = (wsr_m.tolist(), ties_m.tolist(),
                                   ta_m.tolist(), diff_m.tolist())
    baseline = ta_l[0]   # = ties_l[0]; the conflict-free worst-excess floor

    # Read the BASE-FREE synthetic mechanism at the win-share ranges actually
    # measured on the real bases. (Non-circular: win-share ranges were measured
    # on real adapters independently; the synthetic predicts the TIES penalty
    # those ranges imply.) Report gap and gap-relative-to-baseline.
    anchors = []
    for name, ws, real_inv in REAL_ANCHORS:
        t_w, a_w, gp = interp_at(ws, wsr_l, ties_l, ta_l, diff_l)
        anchors.append({"base": name, "winshare_range": ws, "real_inverts": real_inv,
                        "synth_ties_worst": t_w, "synth_ta_worst": a_w,
                        "synth_gap": gp, "gap_over_baseline": gp / baseline})

    # An interpretable, unit-free inversion threshold: the win-share range at
    # which the conflict-induced TIES penalty (TIES-TA) first equals the
    # irreducible baseline excess (gap/baseline = 1).
    ws_gap_eq_base = None
    for i in range(1, R):
        if diff_l[i - 1] <= baseline < diff_l[i]:
            f = (baseline - diff_l[i - 1]) / (diff_l[i] - diff_l[i - 1])
            ws_gap_eq_base = wsr_l[i - 1] + f * (wsr_l[i] - wsr_l[i - 1])
            break

    # coupling: Pearson corr between win-share range and (TIES-TA) across the sweep
    a, b = wsr_m, diff_m
    corr = float(((a - a.mean()) * (b - b.mean())).sum()
                 / (((a - a.mean()) ** 2).sum().sqrt() * ((b - b.mean()) ** 2).sum().sqrt() + 1e-12))

    return {
        "config": {"d": d, "T": T, "k_active": k_active, "n_majority": T - 1,
                   "n_minority": 1, "n_seeds": n_seeds, "density": DENSITY,
                   "sigma": SIGMA, "base_seed": BASE_SEED,
                   "threshold_window": [THRESH_LO, THRESH_HI]},
        "rhos": rhos,
        "ties_worst_mean": ties_l, "ties_worst_ci95": ties_ci.tolist(),
        "ta_worst_mean": ta_l, "ta_worst_ci95": ta_ci.tolist(),
        "diff_mean": diff_l, "diff_ci95": diff_ci.tolist(),
        "winshare_range_mean": wsr_l, "winshare_range_ci95": wsr_ci.tolist(),
        "baseline_excess": baseline,
        "real_anchor_readout": anchors,
        "winshare_range_at_gap_eq_baseline": ws_gap_eq_base,
        "pearson_wsr_vs_diff": corr,
        "monotone_wsr": all(wsr_l[i] <= wsr_l[i + 1] + 1e-9 for i in range(R - 1)),
    }


def write_table(res: dict, path: str) -> None:
    rhos = res["rhos"]
    lines = []
    lines.append("Td2 synthetic controlled-overlap sweep (TIES vs TA, worst-task excess)")
    c = res["config"]
    lines.append(f"d={c['d']} T={c['T']} ({c['n_majority']}+{c['n_minority']}) "
                 f"density={c['density']} seeds={c['n_seeds']}")
    lines.append("")
    lines.append(f"{'rho':>6} | {'wsRange':>8} | {'TIESworst':>10} | {'TAworst':>10} | "
                 f"{'TIES-TA':>9} | {'inv?':>4}")
    lines.append("-" * 66)
    for i, rho in enumerate(rhos):
        ws = res["winshare_range_mean"][i]
        tw = res["ties_worst_mean"][i]
        aw = res["ta_worst_mean"][i]
        df = res["diff_mean"][i]
        inv = "YES" if df > 0 else ""
        lines.append(f"{rho:6.3f} | {ws:8.4f} | {tw:10.4f} | {aw:10.4f} | {df:9.4f} | {inv:>4}")
    lines.append("-" * 66)
    lines.append("")
    lines.append(f"baseline (conflict-free) worst-excess: {res['baseline_excess']:.4f}")
    lines.append(f"Pearson(win-share range, TIES-TA gap) = {res['pearson_wsr_vs_diff']:.4f}"
                 f"   (monotone win-share range: {res['monotone_wsr']})")
    wg = res["winshare_range_at_gap_eq_baseline"]
    if wg is not None:
        inside = THRESH_LO <= wg <= THRESH_HI
        lines.append(f"TIES penalty (gap) == baseline excess at win-share range = {wg:.4f} "
                     f"-> {'INSIDE' if inside else 'OUTSIDE'} Td2 window [{THRESH_LO},{THRESH_HI}]")
    lines.append("")
    lines.append("Synthetic mechanism read at the REAL bases' measured win-share ranges:")
    lines.append(f"  {'base':<13} {'wsRange':>8} {'realInv':>8} {'synGap':>8} {'gap/base':>9}")
    for a in res["real_anchor_readout"]:
        lines.append(f"  {a['base']:<13} {a['winshare_range']:>8.4f} {a['real_inverts']:>8} "
                     f"{a['synth_gap']:>8.4f} {a['gap_over_baseline']:>9.2f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_plot(res: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable, skipping plot: {e})")
        return
    rhos = res["rhos"]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    ax1.plot(rhos, res["ties_worst_mean"], "-o", color="#d62728", ms=4, label="TIES worst-excess")
    ax1.plot(rhos, res["ta_worst_mean"], "--s", color="#888888", ms=4, label="TA worst-excess")
    ax1.set_xlabel(r"minority sign-conflict fraction $\rho$")
    ax1.set_ylabel("worst-task NLL excess (quadratic surrogate)")
    ax1.grid(True, alpha=0.25, ls=":")

    ax2 = ax1.twinx()
    ax2.plot(rhos, res["winshare_range_mean"], "-^", color="#1f77b4", ms=4,
             label="win-share range")
    ax2.axhspan(THRESH_LO, THRESH_HI, color="#1f77b4", alpha=0.10)
    ax2.set_ylabel("TIES sign-election win-share range", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    # mark the real bases at the rho whose synthetic win-share range matches the
    # measured real win-share range; stagger labels vertically (the markers are
    # bunched at low rho) and colour by whether the base really inverts.
    wsr_l = res["winshare_range_mean"]
    ymax = ax1.get_ylim()[1]
    yfrac = {"Llama-3.1": 0.40, "Mistral-7B": 0.56, "Yi-1.5-Chat": 0.74}
    for a in res["real_anchor_readout"]:
        ws = a["winshare_range"]
        if ws > wsr_l[-1]:
            continue
        rho_at = interp_at(ws, wsr_l, rhos)[0]
        inv = a["real_inverts"] == "YES"
        col = "#2ca02c" if inv else "#9467bd"
        ax1.axvline(rho_at, color=col, lw=1.0, ls="--", alpha=0.8)
        ax1.text(rho_at + 0.008, ymax * yfrac[a["base"]],
                 f"{a['base'].split('-')[0]} (R={ws:.3f}, {'inverts' if inv else 'no inv.'})",
                 color=col, fontsize=7, va="center", ha="left")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, frameon=False)
    plt.title("Td2 mechanism on a controlled dial (no base model)", fontsize=10)
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIG_DIR, f"figure_td2_synthetic_overlap.{ext}")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--T", type=int, default=7)
    ap.add_argument("--k-active", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rho-max", type=float, default=0.5)
    ap.add_argument("--n-rho", type=int, default=26)
    args = ap.parse_args()

    rhos = [round(args.rho_max * i / (args.n_rho - 1), 4) for i in range(args.n_rho)]
    t0 = time.time()
    res = run(args.d, args.T, args.seeds, rhos, args.k_active)
    res["runtime_sec"] = round(time.time() - t0, 2)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_table(res, os.path.join(RESULTS_DIR, "table.txt"))
    make_plot(res)
    print(f"\n[done in {res['runtime_sec']}s] -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

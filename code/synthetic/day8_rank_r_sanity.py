"""Day 8 sanity checks for theory/theorem_v1.tex (rank-r LoRA scaffold).

Three checks paired to the scaffold's sanity-check propositions:

(a) Regression check (r = d, isotropic limit). Must reproduce the
    Phase 0 floor ratio empirical/theory in [0.998, 1.017] and the
    Phase 0 2^{-2b} excess decay for the quantized-centroid merge.
    This is the r = d cell of Prop. 3 (full-rank limit) in
    theorem_v1.tex.

(b) Shared-subspace slope (V_t = V for all t). Log-excess vs R should
    fit slope -2/r (not -2/d), confirming Prop. 5 (shared-subspace
    limit reduces Phase 0 with d -> r).

(c) Random-subspace effective dimension. Slope fit on the
    Stiefel-random V_t case should yield an empirical d_eff between r
    and d, tracking rank(sum_t P_{V_t}) (or r/gamma). This informs the
    Day-9 choice of d_eff in Conjecture 2.

Setup
-----
Task-vector tuples are generated per theorem_v1.tex §3:
  U_t in Stiefel(d, r) uniform (QR of d x r Gaussian);
  v_t in Unif(B * S^{r-1}) inside R^r;
  tau_t = U_t v_t in V_t := col(U_t).
  H_t = P_{V_t} = U_t U_t^T.

For the shared-subspace cell, a single U is reused across t.

Merging scheme (Scheme B of the scaffold): quantize bar_tau_H in an
orthonormal basis of range(bar H), b bits per effective coord, total
rate R = b * d_eff_measured where d_eff_measured = rank(bar H). This
matches the conjectured 2^{-2R/d_eff} shape with d_eff = rank(sum_t
P_{V_t}).

Outputs: PNG plots + printed tables. Saved to code/synthetic/figures/.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(20260427)
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# --- Samplers -----------------------------------------------------

def sample_stiefel(d, r, rng):
    """Uniform sample from Stiefel manifold St(d, r) via QR of Gaussian."""
    G = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]


def sample_sphere(r, B, rng):
    v = rng.standard_normal(r)
    v /= np.linalg.norm(v)
    return B * v


def sample_rank_r_tuple(T, d, r, B, overlap, rng):
    """
    Returns (taus, Us) where taus[t] is in R^d and Us[t] is d x r.

    overlap = "random": independent V_t.
    overlap = "shared": V_t = V for all t (shared random V).
    """
    if overlap == "shared":
        U_shared = sample_stiefel(d, r, rng)
        Us = [U_shared for _ in range(T)]
    elif overlap == "random":
        Us = [sample_stiefel(d, r, rng) for _ in range(T)]
    else:
        raise ValueError(f"unknown overlap: {overlap}")
    taus = [Us[t] @ sample_sphere(r, B, rng) for t in range(T)]
    return np.array(taus), Us


# --- Core quantities ---------------------------------------------

def compute_Hbar_and_tauH(taus, Us):
    """
    Hbar = (1/T) sum_t U_t U_t^T  (d x d)
    bar_H_tau = (1/T) sum_t U_t U_t^T tau_t
    tauH = Hbar^+ bar_H_tau
    Returns (Hbar, tauH, Q, d_eff) where Q is a d x d_eff basis of
    range(Hbar) and d_eff is the measured rank.
    """
    T, d = taus.shape
    Hbar = np.zeros((d, d))
    Htau_bar = np.zeros(d)
    for t in range(T):
        Ut = Us[t]
        Pt = Ut @ Ut.T
        Hbar += Pt
        Htau_bar += Pt @ taus[t]
    Hbar /= T
    Htau_bar /= T

    # Eigendecomposition; retain directions with eigenvalue > tol.
    w, V = np.linalg.eigh(Hbar)
    tol = 1e-10 * max(w.max(), 1.0)
    mask = w > tol
    d_eff = int(mask.sum())
    Q = V[:, mask]
    w_pos = w[mask]
    # Pseudoinverse: Q diag(1/w) Q^T on range.
    tauH = Q @ ((Q.T @ Htau_bar) / w_pos)
    return Hbar, tauH, Q, d_eff


def Ht_weighted_floor(taus, Us, tauH):
    """(1/T) sum_t (tau_t - tauH)^T P_{V_t} (tau_t - tauH)."""
    T = taus.shape[0]
    acc = 0.0
    for t in range(T):
        diff = taus[t] - tauH
        proj = Us[t].T @ diff  # r-vector
        acc += float(proj @ proj)
    return acc / T


def Ht_weighted_max(w, taus, Us):
    """max_t (w - tau_t)^T P_{V_t} (w - tau_t)."""
    vals = []
    for t in range(len(taus)):
        diff = w - taus[t]
        proj = Us[t].T @ diff
        vals.append(float(proj @ proj))
    return max(vals)


# --- Quantizer ---------------------------------------------------

def uniform_scalar_quantize(v, b, v_min, v_max):
    if b >= 30:
        return v.copy()
    levels = 2 ** b
    step = (v_max - v_min) / levels
    idx = np.floor((v - v_min) / step).astype(np.int64)
    idx = np.clip(idx, 0, levels - 1)
    return v_min + (idx + 0.5) * step


def quantize_tauH_in_range(tauH, Q, b, B):
    """Quantize tauH by projecting to the Q-basis of range(Hbar).

    Uses b bits per coord of the d_eff-dim representation; total
    rate R = b * d_eff. Quantization range is [-B, B] coordinate-wise,
    which is conservative since ||Q^T tauH|| <= ||tauH|| <= B.
    """
    coeffs = Q.T @ tauH
    coeffs_q = uniform_scalar_quantize(coeffs, b, -B, B)
    return Q @ coeffs_q


def quantize_tauH_adaptive(tauH, Q, b, B, T, r, c=3.0):
    """Scale-adaptive quantization in the range(Hbar) basis.

    Lemma 6 of theorem_v1.tex gives E[||Q^T tauH||^2] = B^2 d_eff / (Tr),
    so per-coord std in the Q-basis is sigma_pc = B / sqrt(T r)
    (independent of d_eff since ||Q^T tauH||^2 has d_eff coords).
    Quantize each coord in [-c*sigma_pc, c*sigma_pc] to avoid the
    range-waste that saturates low-b uniform [-B, B] quantization.
    """
    sigma_pc = B / np.sqrt(T * r)
    rng_hi = c * sigma_pc
    coeffs = Q.T @ tauH
    coeffs_q = uniform_scalar_quantize(coeffs, b, -rng_hi, rng_hi)
    return Q @ coeffs_q


# --- Sweep -------------------------------------------------------

def hbar_metric_sq(Hbar, v):
    """v^T Hbar v."""
    return float(v @ (Hbar @ v))


def run_one_cell(T, d, r, b, overlap, B, n_trials):
    """Run n_trials for (T, d, r, b, overlap); return median stats.

    Two excess metrics are reported:

    - excess_max (``D_merge - floor_emp``): what a real merger
      experiences. At T=2 shared-V this picks up a cross-term
      2|a.η| that scales as ||η||, giving slope -1 per bit and
      masking the -2 slope predicted by Theorem 7.

    - excess_hbar (``||w_merge - tauH||^2_{Hbar}``): the pure
      quantization-error component, which by Lemma 1 equals
      ``avg_t g(w_merge,t) - avg_t g(tauH, t)``. This is the
      quantity Theorem 7's converse step-5 bounds directly, and
      the one that should decay at slope -2 per bit.
    """
    floors = []
    merge_Ds = []
    excesses_max = []
    excesses_hbar = []
    d_effs = []
    for _ in range(n_trials):
        taus, Us = sample_rank_r_tuple(T, d, r, B, overlap, RNG)
        Hbar, tauH, Q, d_eff = compute_Hbar_and_tauH(taus, Us)
        floor = Ht_weighted_floor(taus, Us, tauH)
        w_merge = quantize_tauH_adaptive(tauH, Q, b, B, T, r)
        D_merge = Ht_weighted_max(w_merge, taus, Us)
        eta = w_merge - tauH
        floors.append(floor)
        merge_Ds.append(D_merge)
        excesses_max.append(max(D_merge - floor, 1e-12))
        excesses_hbar.append(max(hbar_metric_sq(Hbar, eta), 1e-12))
        d_effs.append(d_eff)
    return {
        "T": T, "d": d, "r": r, "b": b, "overlap": overlap,
        "floor_med": float(np.median(floors)),
        "merge_D_med": float(np.median(merge_Ds)),
        "excess_med": float(np.median(excesses_max)),
        "excess_hbar_med": float(np.median(excesses_hbar)),
        "d_eff_median": int(np.median(d_effs)),
    }


def run_sweep(T_list, d_list, r_list, b_list, overlap_list,
              B=1.0, n_trials=30):
    records = []
    for T in T_list:
        for d in d_list:
            for r in r_list:
                if r > d:
                    continue
                for overlap in overlap_list:
                    for b in b_list:
                        rec = run_one_cell(T, d, r, b, overlap,
                                           B, n_trials)
                        records.append(rec)
    return records


# --- Analysis ----------------------------------------------------

def phase0_floor(T, B=1.0):
    """Phase 0 isotropic floor B^2 (1 - 1/T)."""
    return (B ** 2) * (1.0 - 1.0 / T)


def print_regression_check(records, T_list):
    """Check (a): r = d cell reproduces Phase 0 ratio."""
    print("\n--- Check (a): regression vs Phase 0 (r = d, overlap=random) ---")
    print(f"{'T':>3} {'d':>6} {'b':>3} {'floor_emp':>10} "
          f"{'phase0_B2(1-1/T)':>18} {'ratio':>8}")
    for r in records:
        if r["r"] != r["d"] or r["overlap"] != "random":
            continue
        if r["b"] != max(rr["b"] for rr in records):
            continue
        thy = phase0_floor(r["T"])
        ratio = r["floor_med"] / thy
        print(f"{r['T']:>3} {r['d']:>6} {r['b']:>3} {r['floor_med']:>10.4f} "
              f"{thy:>18.4f} {ratio:>8.4f}")


def fit_slope(bs, excesses_log2):
    """Least-squares slope of log2(excess) vs b."""
    bs = np.asarray(bs, float)
    y = np.asarray(excesses_log2, float)
    A = np.vstack([bs, np.ones_like(bs)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def print_slope_fits(records):
    """Checks (b) and (c): slope fits on both excess metrics.

    The ``excess_hbar`` metric is the Hbar-weighted quantization
    error ``||w_merge - tauH||^2_{Hbar}`` that Theorem 7's converse
    bounds directly. Under scale-adaptive quantization it must
    decay at slope -2 per bit-per-coord (since the Q-basis coeffs
    are quantized with step ~ sigma_pc * 2^{-b}). The legacy
    ``excess_max`` metric is ``D_merge - floor`` which includes
    cross-terms and only decays at slope -1 in the shared-V case.
    """
    print("\n--- Checks (b) and (c): slope fits on both metrics ---")
    print(f"{'overlap':>8} {'T':>3} {'d':>6} {'r':>4} "
          f"{'d_eff_meas':>11} {'slp_hbar':>9} {'slp_max':>9} "
          f"{'slp_R_hbar':>11} {'d_eff_fit':>10}")
    keys = set((r["T"], r["d"], r["r"], r["overlap"]) for r in records)
    for T, d, r, overlap in sorted(keys):
        subset = [rr for rr in records
                  if rr["T"] == T and rr["d"] == d
                  and rr["r"] == r and rr["overlap"] == overlap]
        subset.sort(key=lambda rr: rr["b"])
        bs = [rr["b"] for rr in subset]
        log2_hbar = [np.log2(rr["excess_hbar_med"]) for rr in subset]
        log2_max = [np.log2(rr["excess_med"]) for rr in subset]
        slope_b_hbar = fit_slope(bs, log2_hbar)
        slope_b_max = fit_slope(bs, log2_max)
        d_eff_meas = subset[0]["d_eff_median"]
        Rs = [b * d_eff_meas for b in bs]
        slope_R_hbar = fit_slope(Rs, log2_hbar)
        d_eff_fit = (-2.0 / slope_R_hbar
                     if slope_R_hbar < -1e-6 else float("nan"))
        print(f"{overlap:>8} {T:>3} {d:>6} {r:>4} "
              f"{d_eff_meas:>11} {slope_b_hbar:>9.3f} "
              f"{slope_b_max:>9.3f} {slope_R_hbar:>11.4f} "
              f"{d_eff_fit:>10.2f}")


# --- Plotting ----------------------------------------------------

def plot_excess_vs_bits(records, T_list, d_list, r_list, savedir):
    """Per (T, d): two rows (Hbar metric, max metric), two cols
    (shared, random). Hbar row should show clean -2b slopes; max
    row preserves the Day-8 visualization."""
    for T in T_list:
        for d in d_list:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), sharey=False)
            for row_idx, (metric_key, metric_label) in enumerate([
                ("excess_hbar_med",
                 r"$\|w^*-\bar\tau_H\|^2_{\bar H}$"),
                ("excess_med",
                 r"max-distortion $-$ floor"),
            ]):
                for col_idx, overlap in enumerate(["shared", "random"]):
                    ax = axes[row_idx, col_idx]
                    for r in r_list:
                        if r > d:
                            continue
                        subset = [rr for rr in records
                                  if rr["T"] == T and rr["d"] == d
                                  and rr["r"] == r and rr["overlap"] == overlap]
                        subset.sort(key=lambda rr: rr["b"])
                        bs = np.array([rr["b"] for rr in subset])
                        excess = np.array([rr[metric_key] for rr in subset])
                        d_eff_meas = subset[0]["d_eff_median"]
                        label = (f"r={r}  ($d_{{\\rm eff}}$={d_eff_meas})")
                        ax.plot(bs, excess, "o-", lw=1.3, label=label)
                    ax.plot(bs, 2.0 ** (-2.0 * bs), "k--", alpha=0.5,
                            label=r"slope $2^{-2b}$")
                    ax.set_xlabel("b (bits per effective coord)")
                    ax.set_yscale("log")
                    ax.set_title(f"{overlap} V_t, {metric_label}")
                    ax.grid(True, alpha=0.3, which="both")
                    ax.legend(fontsize=7, loc="best")
                axes[row_idx, 0].set_ylabel(metric_label)
            fig.suptitle(f"Day 9 rank-r: excess decay  (T={T}, d={d})")
            fig.tight_layout()
            fp = savedir / f"day8_excess_T{T}_d{d}.png"
            fig.savefig(fp, dpi=110)
            plt.close(fig)
            print(f"saved {fp}")


def plot_deff_vs_r(records, T_list, d_list, savedir):
    """Measured d_eff = rank(bar H) vs r, for each T, d, overlap."""
    for T in T_list:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        keys = set((rr["d"], rr["r"], rr["overlap"]) for rr in records
                   if rr["T"] == T)
        by_overlap = {"shared": [], "random": []}
        for d, r, overlap in sorted(keys):
            subset = [rr for rr in records
                      if rr["T"] == T and rr["d"] == d
                      and rr["r"] == r and rr["overlap"] == overlap]
            d_eff_meas = subset[0]["d_eff_median"]
            by_overlap[overlap].append((d, r, d_eff_meas))
        for overlap, data in by_overlap.items():
            if not data:
                continue
            for d in d_list:
                pts = [(r, de) for dd, r, de in data if dd == d]
                if not pts:
                    continue
                rs = [p[0] for p in pts]
                des = [p[1] for p in pts]
                ax.plot(rs, des, "o-",
                        label=f"{overlap}, d={d}")
        ax.plot([1, max(r for _, r, _ in [p for v in by_overlap.values()
                                          for p in v])],
                [1, max(r for _, r, _ in [p for v in by_overlap.values()
                                          for p in v]) * T],
                "k:", alpha=0.4, label="$Tr$ (orthogonal limit)")
        ax.set_xlabel("r")
        ax.set_ylabel(r"measured $d_{\rm eff} = \mathrm{rank}(\bar H)$")
        ax.set_title(f"Measured d_eff vs r  (T={T})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = savedir / f"day8_deff_T{T}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        print(f"saved {fp}")


# --- Main --------------------------------------------------------

def main():
    T_list = [2, 4]
    d_list = [128, 512]
    r_list = [4, 8, 16, 32]
    b_list = [1, 2, 4, 8]
    overlap_list = ["shared", "random"]

    # For the regression check we also include r = d cells (random only).
    r_list_full = list(r_list)
    for d in d_list:
        if d not in r_list_full:
            r_list_full.append(d)

    print("Running sweep (this takes ~1-2 minutes)...")
    records = run_sweep(T_list, d_list, r_list_full, b_list,
                        overlap_list, n_trials=30)

    print_regression_check(records, T_list)
    print_slope_fits(records)

    plot_excess_vs_bits(records, T_list, d_list, r_list, FIG_DIR)
    plot_deff_vs_r(records, T_list, d_list, FIG_DIR)


if __name__ == "__main__":
    main()

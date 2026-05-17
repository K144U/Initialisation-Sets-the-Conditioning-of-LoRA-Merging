"""Day 13 — Two fixes to Day-12 achievability.

Day 12 delivered a Hadamard-incoherence quantizer that matches
Theorem 8's LB to constants (~11x) in the floor = 0 regime (Stiefel
random, d_eff = Tr), but had two weak spots:

(1) Non-power-of-2 d_eff: padding to next-pow2(d_eff) = n wastes
    bits since effective per-coord rate is R/n not R/d_eff. Fix
    here: use a d_eff-dim randomized orthogonal mixer via Gaussian
    QR (no padding).

(2) Shared-V (floor > 0): exc_max slope -1 instead of -2. The
    expand-max-around-tauH argument gives a linear cross-term
    2*(w-tauH)^T H_t (tauH-tau_t) that dominates at moderate rate.
    Fix: quantize around the H_t-Chebyshev center w_cheb instead
    of tauH. At w_cheb, the active tasks have equal distortion, so
    (w_cheb - tau_t)^T H_t (w_cheb - tau_t) = D*_cheb for each
    active t; the max-gradient KKT condition ensures the linear
    term vanishes (in expectation over zero-mean quantization
    noise) at the cheb center, leaving the quadratic 2^{-2R/d_eff}
    term to dominate.

Implementation:
    - Gaussian-QR mixer (no pad).
    - H_t-Chebyshev center by smooth-max + gradient descent
      (log-sum-exp with increasing temperature schedule).
    - Quantize w_cheb (+eta' = Hbar^{1/2}(w_cheb - tauH) rotational
      offset) instead of eta.

Wait - there is a subtlety. The Chebyshev center lives in R^d, not in
range(Hbar). For the max distortion we only need its range(Hbar)
component. And the shift w_cheb - tauH itself lives in range(Hbar).
So we can work in eta' := Hbar^{1/2} (w_cheb - tauH) in R^{d_eff},
then quantize, decode to w^star = tauH + Hbar^{+ 1/2} eta_q'.

Test layout:
    (A) Fix-1 only (Gaussian-QR mixer, still center at tauH):
        re-run Day-12 cells; expect constant ratio across rates
        even for non-pow-2 d_eff cells.
    (B) Fix-1 + Fix-2 (cheb center): re-run shared-V cells; expect
        exc_max slope -2.
"""

import math
import numpy as np
from day10_general_ht import (
    make_D0, compute_Hbar_tauH_general, ht_weighted_floor_general,
    hbar_metric_sq, uniform_scalar_quantize,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day12_achievability import ht_weighted_max, ht_weighted_avg

RNG = np.random.default_rng(20260502)


# --- Fix 1: Gaussian-QR orthogonal mixer (d_eff dim, no padding) --

def sample_orthogonal(d_eff, rng):
    """Uniform sample from O(d_eff) via QR of Gaussian."""
    G = rng.standard_normal((d_eff, d_eff))
    Q, R = np.linalg.qr(G)
    # Fix sign so the distribution is Haar (not just a coset thereof).
    sgn = np.sign(np.diag(R))
    sgn[sgn == 0] = 1.0
    return Q * sgn


def quantize_vector_gaussian_qr(v, b, sigma_pc, c, rng):
    """Quantize a d_eff-vector v with mean-0 per-coord std roughly
    sigma_pc, using a random orthogonal mixer.
    Returns the quantized v_q (same shape as v).

    Total rate spent = b * d_eff bits.
    """
    d_eff = v.shape[0]
    O = sample_orthogonal(d_eff, rng)
    v_rot = O @ v
    rng_hi = c * sigma_pc
    step = (2 * rng_hi) / (2 ** b)
    idx = np.floor((v_rot + rng_hi) / step).astype(np.int64)
    idx = np.clip(idx, 0, (2 ** b) - 1)
    v_rot_q = -rng_hi + (idx + 0.5) * step
    # Invert mixer.
    v_q = O.T @ v_rot_q
    return v_q


# --- Fix 2: H_t-Chebyshev center via smooth-max + gradient -------

def cheb_center_reduced(taus, Hts, Q, max_iter=50, tol=1e-8,
                         beta_schedule=(5.0, 50.0, 500.0)):
    """Minimize max_t (w - tau_t)^T H_t (w - tau_t) in the d_eff-
    basis (Q^T w-space). Returns w_cheb in R^d (on range(Hbar)).

    Reduced problem: let wbar = Q^T w. Then
      (w - tau_t)^T H_t (w - tau_t) = (wbar - Q^T tau_t)^T Ht_r (wbar - Q^T tau_t)
    where Ht_r = Q^T H_t Q is d_eff x d_eff and the "orthogonal-to-W"
    component of w doesn't affect any of the H_t's (range(H_t) in W).

    Uses smooth-max + weighted-linear-system iteration in d_eff-space.
    """
    T = len(taus)
    d_eff = Q.shape[1]
    # Reduced Ht's and tau_t's.
    Ht_r = [Q.T @ (Hts[t] @ Q) for t in range(T)]
    tau_r = [Q.T @ taus[t] for t in range(T)]

    # Init at tauH reduced = Q^T tauH. But since we pass Q from outside,
    # we can also init at mean of tau_r.
    wbar = np.mean(tau_r, axis=0)

    for beta in beta_schedule:
        for it in range(max_iter):
            dists = np.array([float((wbar - tau_r[t]) @ (Ht_r[t] @ (wbar - tau_r[t])))
                              for t in range(T)])
            m = dists.max()
            alphas = np.exp(beta * (dists - m))
            alphas /= alphas.sum()

            Hsum = np.zeros((d_eff, d_eff))
            rhs = np.zeros(d_eff)
            for t in range(T):
                Hsum += alphas[t] * Ht_r[t]
                rhs += alphas[t] * (Ht_r[t] @ tau_r[t])
            # Solve via pinv (Hsum may be rank-deficient in degenerate
            # cases but for shared/random-V is typically full-rank at
            # d_eff).
            try:
                wbar_new = np.linalg.solve(Hsum + 1e-12 * np.eye(d_eff), rhs)
            except np.linalg.LinAlgError:
                wbar_new = np.linalg.pinv(Hsum) @ rhs
            if np.linalg.norm(wbar_new - wbar) < tol * max(np.linalg.norm(wbar), 1.0):
                wbar = wbar_new
                break
            wbar = wbar_new
    return Q @ wbar  # Lift back to R^d.


# --- Merging pipelines -------------------------------------------

def merge_tauH_gaussianqr(taus, Hts, b, B, T, r, rng, c=5.0):
    """Fix-1 only: quantize eta = Hbar^{1/2} tauH via Gaussian-QR
    mixer on d_eff dims directly (no padding)."""
    Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
    sqrt_w = np.sqrt(w_pos)
    eta = sqrt_w * (Q.T @ tauH)
    sigma_pc = B / math.sqrt(T * r)
    eta_q = quantize_vector_gaussian_qr(eta, b, sigma_pc, c, rng)
    w_coeffs = eta_q / sqrt_w
    w_star = Q @ w_coeffs
    return w_star, d_eff


def merge_cheb_gaussianqr(taus, Hts, b, B, T, r, rng, c=5.0,
                           w_cheb_cache=None):
    """Fix-1 + Fix-2: quantize around the H_t-Chebyshev center.
    Represent the final w as tauH + Hbar^{+ 1/2} eta_shift_q,
    where eta_shift = Hbar^{1/2} (w_cheb - tauH) in range(Hbar).

    w_cheb_cache: if provided, (Hbar, tauH, Q, d_eff, w_pos, w_cheb)
    tuple that's reused (doesn't depend on b). Otherwise recomputed.
    """
    if w_cheb_cache is None:
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        w_cheb = cheb_center_reduced(taus, Hts, Q)
    else:
        Hbar, tauH, Q, d_eff, w_pos, w_cheb = w_cheb_cache

    sqrt_w = np.sqrt(w_pos)
    eta_cheb = sqrt_w * (Q.T @ w_cheb)
    sigma_pc = B / math.sqrt(T * r)
    eta_q = quantize_vector_gaussian_qr(eta_cheb, b, sigma_pc, c, rng)
    w_coeffs = eta_q / sqrt_w
    w_star = Q @ w_coeffs
    return w_star, d_eff


# --- Evaluation --------------------------------------------------

def run_cell(merge_fn, T, d, r, D_list, overlap, bits_list,
              n_trials, B=1.0, c=5.0, use_cheb_cache=False):
    results = {b: {"avg": [], "max": [], "floor": []} for b in bits_list}
    d_effs = []
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floor = ht_weighted_floor_general(taus, Hts, tauH)
        cache = None
        if use_cheb_cache:
            w_cheb = cheb_center_reduced(taus, Hts, Q)
            cache = (Hbar, tauH, Q, d_eff, w_pos, w_cheb)
        for b in bits_list:
            if use_cheb_cache:
                w_star, d_eff = merge_fn(taus, Hts, b, B, T, r, RNG,
                                         c=c, w_cheb_cache=cache)
            else:
                w_star, d_eff = merge_fn(taus, Hts, b, B, T, r, RNG, c=c)
            results[b]["avg"].append(ht_weighted_avg(w_star, taus, Hts))
            results[b]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[b]["floor"].append(floor)
        d_effs.append(d_eff)
    d_eff_mean = float(np.mean(d_effs))

    rows = []
    for b in bits_list:
        avg_emp = float(np.mean(results[b]["avg"]))
        max_emp = float(np.mean(results[b]["max"]))
        floor_emp = float(np.mean(results[b]["floor"]))
        R = b * int(round(d_eff_mean))
        floor_thy = (B ** 2) * (1 - d_eff_mean / (T * r))
        compress_thy = (B ** 2) * (d_eff_mean / (T * r)) * 2 ** (-2 * R / d_eff_mean)
        rows.append({
            "b": b, "R": R, "d_eff": d_eff_mean,
            "floor_emp": floor_emp, "floor_thy": floor_thy,
            "avg_emp": avg_emp, "max_emp": max_emp,
            "excess_avg": avg_emp - floor_emp,
            "excess_max": max_emp - floor_emp,
            "compress_thy_cTQ1": compress_thy,
        })
    return rows


def fit_slope(bits_list, vals):
    log_vals = np.log2(np.array(vals))
    b_arr = np.array(bits_list, dtype=float)
    slope, _ = np.polyfit(b_arr, log_vals, 1)
    return float(slope)


def print_cell(label, rows, bits_list):
    d_eff = rows[0]["d_eff"]
    floor_thy = rows[0]["floor_thy"]
    print(f"\n--- {label} | d_eff={d_eff:.1f} floor_thy={floor_thy:.4f} ---")
    print(f"{'b':>3} {'R':>5} {'avg_emp':>9} {'max_emp':>9} "
          f"{'exc_avg':>9} {'exc_max':>9} "
          f"{'LB(cTQ=1)':>10} {'exc_max/LB':>11}")
    for row in rows:
        ratio = (row["excess_max"] / row["compress_thy_cTQ1"]
                 if row["compress_thy_cTQ1"] > 1e-15 else float("nan"))
        print(f"{row['b']:>3} {row['R']:>5} "
              f"{row['avg_emp']:>9.5f} {row['max_emp']:>9.5f} "
              f"{row['excess_avg']:>9.5f} {row['excess_max']:>9.5f} "
              f"{row['compress_thy_cTQ1']:>10.5f} "
              f"{ratio:>11.3f}")
    s_avg = fit_slope(bits_list, [r["excess_avg"] for r in rows
                                   if r["excess_avg"] > 1e-12])
    s_max = fit_slope(bits_list, [r["excess_max"] for r in rows
                                   if r["excess_max"] > 1e-12])
    print(f"  slope_avg={s_avg:.3f}  slope_max={s_max:.3f}")


def main():
    B = 1.0
    n_trials = 200
    bits_list = [2, 4, 6, 8, 10]

    print("=" * 100)
    print("Day 13 — achievability fixes (Gaussian-QR mixer + Cheb center)")
    print("=" * 100)

    # --- Test A: Fix-1 alone (Gaussian-QR) on Day-12's problem cases
    print("\n==== TEST A: Fix-1 only (Gaussian-QR mixer, centered at tauH) ====")
    test_A_cases = [
        (2, 128, 4, ["iso", "iso"],              "random"),
        (2, 128, 4, ["iso", "geom"],             "random"),
        (3, 128, 4, ["iso", "geom", "twoscale"], "random"),
    ]
    for T, d, r, specs, overlap in test_A_cases:
        D_list = build_D_list(specs, r)
        rows = run_cell(merge_tauH_gaussianqr,
                        T, d, r, D_list, overlap, bits_list, n_trials, B=B)
        print_cell(f"T={T} r={r} {'+'.join(s[:4] for s in specs)} {overlap}",
                   rows, bits_list)

    # --- Test B: Fix-1 + Fix-2 (Cheb center) on shared-V cells
    print("\n==== TEST B: Fix-1 + Fix-2 (Cheb center) on shared-V cells ====")
    test_B_cases = [
        (2, 128, 4, ["iso", "iso"],              "shared"),
        (2, 128, 4, ["geom", "twoscale"],        "shared"),
        (4, 128, 4, ["iso", "geom", "lin", "twoscale"], "shared"),
    ]
    for T, d, r, specs, overlap in test_B_cases:
        D_list = build_D_list(specs, r)
        rows = run_cell(merge_cheb_gaussianqr,
                        T, d, r, D_list, overlap, bits_list, n_trials, B=B,
                        use_cheb_cache=True)
        print_cell(f"T={T} r={r} {'+'.join(s[:4] for s in specs)} {overlap}",
                   rows, bits_list)


if __name__ == "__main__":
    main()

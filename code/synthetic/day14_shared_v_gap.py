"""Day 14 — Shared-V achievability gap, partial fix via null-space-aware
bit allocation.

Phenomenon: in shared-V at Chebyshev center with |A| active tasks,
the active gradients g_t = w_cheb - tau_t span a (|A|-1)-dim
subspace (KKT: sum alpha_t g_t = 0). Any explicit linear encoder
that quantizes w_cheb in an arbitrary basis gives:
    max_t ||w^*-tau_t||^2 = cheb^2 + 2 max_t g_t^T dw + ||dw||^2.
The linear term (half-normal, positive) decays as 2^{-R/r} per
bit (one-scale), giving slope -1 on log2(excess) vs b=R/r.

Fix: split V into parallel (span{g_t}) and perpendicular (null).
Allocate R_parallel bits to parallel dims, R_perp bits to perp.
Balanced allocation minimizes linear + quadratic:
    - Linear (parallel-caused):    |g| * 2^{-R_parallel/(|A|-1)}
    - Quadratic (perp-caused):     k * 2^{-2R_perp/k},  k = r-|A|+1
Equate (for large R): R_parallel = 2R(|A|-1)/(r+|A|-1),
    R_perp = R k/(r+|A|-1).
Combined excess ~ 2^{-2R/(k+2(|A|-1))} = 2^{-2R/(r+|A|-1)}.

For r=4, |A|=T=2: exponent -2R/5, slope on b=R/r is -8/5=-1.6.
For r=4, |A|=T=4: exponent -2R/7, slope -8/7 ~= -1.14.

This is still weaker than LB's -2R/r, but strictly better than the
naive -R/r. The remaining gap is a conjectured information-theoretic
limit of explicit linear encoders on max-distortion in the shared-V
regime.

Implementation:
    1. Compute w_cheb via cheb_center_reduced.
    2. Identify active set by KKT alphas (from the cheb_center solver).
    3. Build parallel basis P = orthonormalized {g_1, ..., g_{|A|-1}}.
       Perp basis N = complement in V.
    4. Express w_cheb in (P, N)-basis: w_cheb_parallel, w_cheb_perp.
    5. Quantize each with its allocated rate.
    6. Decoder inverts.
"""

import math
import numpy as np
from day10_general_ht import (
    make_D0, compute_Hbar_tauH_general, ht_weighted_floor_general,
    hbar_metric_sq, uniform_scalar_quantize,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day12_achievability import ht_weighted_max, ht_weighted_avg
from day13_achievability_fixes import (
    cheb_center_reduced, sample_orthogonal,
    quantize_vector_gaussian_qr,
)

RNG = np.random.default_rng(20260503)


# --- Active set + parallel / perp basis --------------------------

def active_set_and_gradients(taus, Hts, w_cheb, thresh=0.98):
    """Return indices of active tasks (those within thresh * max
    of the Chebyshev value) and their gradients g_t = H_t (w_cheb - tau_t).
    """
    T = len(taus)
    dists = np.array([float((w_cheb - taus[t]) @ (Hts[t] @ (w_cheb - taus[t])))
                      for t in range(T)])
    max_d = dists.max()
    active = [t for t in range(T) if dists[t] >= thresh * max_d]
    gradients = [Hts[t] @ (w_cheb - taus[t]) for t in active]
    return active, gradients


def parallel_perp_basis(gradients, Q):
    """Given active gradients (list of R^d vectors) and range(Hbar)
    basis Q (d x d_eff), return:
        P: d_eff x m orthonormal basis of span(Q^T g_t) in R^{d_eff}
           (m = number of independent gradients).
        N: d_eff x (d_eff - m) orthonormal complement in R^{d_eff}.
    """
    d_eff = Q.shape[1]
    if len(gradients) == 0:
        return np.zeros((d_eff, 0)), np.eye(d_eff)
    G = np.stack([Q.T @ g for g in gradients], axis=1)  # d_eff x |A|
    # SVD: take left singular vectors for nonzero singular values.
    U, S, Vt = np.linalg.svd(G, full_matrices=True)
    m = int(np.sum(S > 1e-10 * max(S.max() if len(S) else 1.0, 1.0)))
    P = U[:, :m]
    N = U[:, m:]
    return P, N


# --- Merge pipeline with null-space-aware bit allocation ---------

def allocate_bits(R, m_parallel, k_perp, active_gnorm=1.0):
    """Return (R_parallel, R_perp) integer allocations that
    approximately equalize linear and quadratic term dominance.
    For large R: R_parallel = 2R(m)/(k+2m); R_perp = Rk/(k+2m).
    """
    total = m_parallel + k_perp  # should equal d_eff.
    if m_parallel == 0:
        return 0, R
    denom = k_perp + 2 * m_parallel
    # Continuous allocation.
    R_parallel_real = 2.0 * R * m_parallel / denom
    R_perp_real = R * k_perp / denom
    # Round to nearest integer totaling R.
    R_parallel = int(round(R_parallel_real))
    R_perp = R - R_parallel
    # Each coord needs at least 1 bit to transmit anything.
    R_parallel = max(m_parallel, min(R_parallel, R - k_perp))
    R_perp = R - R_parallel
    return R_parallel, R_perp


def quantize_subspace(v_sub, R_sub, sigma_pc, c, rng):
    """Quantize a k-dim vector with R_sub total bits (b = R_sub/k per coord).
    Uniform scalar on [-c*sigma_pc, c*sigma_pc] after random orthogonal
    mixer.
    """
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    b_per = max(1, R_sub // k)
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub
    rng_hi = c * sigma_pc
    step = (2 * rng_hi) / (2 ** b_per)
    idx = np.floor((v_rot + rng_hi) / step).astype(np.int64)
    idx = np.clip(idx, 0, (2 ** b_per) - 1)
    v_rot_q = -rng_hi + (idx + 0.5) * step
    return O.T @ v_rot_q


def merge_cheb_split(taus, Hts, R, B, T, r, rng, c=5.0):
    """Null-space-aware split quantization around Chebyshev center.
    R is TOTAL rate (bits).
    """
    Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
    w_cheb = cheb_center_reduced(taus, Hts, Q)
    active, gradients = active_set_and_gradients(taus, Hts, w_cheb)
    P, N = parallel_perp_basis(gradients, Q)
    m = P.shape[1]
    k = N.shape[1]

    sigma_pc = B / math.sqrt(T * r)
    R_par, R_perp = allocate_bits(R, m, k)

    # Represent Q^T w_cheb in (P, N) basis.
    wcheb_Qbasis = Q.T @ w_cheb  # d_eff-vector
    wpar = P.T @ wcheb_Qbasis     # m-vector
    wperp = N.T @ wcheb_Qbasis    # k-vector

    # Quantize each independently.
    wpar_q = quantize_subspace(wpar, R_par, sigma_pc, c, rng)
    wperp_q = quantize_subspace(wperp, R_perp, sigma_pc, c, rng)

    # Reconstruct.
    wcheb_Qbasis_q = P @ wpar_q + N @ wperp_q
    w_star = Q @ wcheb_Qbasis_q
    return w_star, d_eff, m, k, R_par, R_perp


# --- Evaluation --------------------------------------------------

def run_shared_V_cell(T, d, r, D_list, R_list, n_trials, B=1.0, c=5.0):
    """Measure max/avg distortion under merge_cheb_split at each R in R_list."""
    results = {R: {"avg": [], "max": [], "floor": [],
                   "m": [], "k": [], "R_par": [], "R_perp": []}
               for R in R_list}
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, "shared", RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floor = ht_weighted_floor_general(taus, Hts, tauH)
        for R in R_list:
            w_star, d_eff_out, m, k, R_par, R_perp = merge_cheb_split(
                taus, Hts, R, B, T, r, RNG, c=c)
            results[R]["avg"].append(ht_weighted_avg(w_star, taus, Hts))
            results[R]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[R]["floor"].append(floor)
            results[R]["m"].append(m)
            results[R]["k"].append(k)
            results[R]["R_par"].append(R_par)
            results[R]["R_perp"].append(R_perp)
    return results


def print_shared_V(label, results, R_list, r):
    print(f"\n--- {label} ---")
    print(f"{'R':>4} {'b=R/r':>7} {'avg_emp':>9} {'max_emp':>9} "
          f"{'exc_avg':>9} {'exc_max':>9} "
          f"{'m':>3} {'k':>3} {'R_par':>5} {'R_perp':>6} {'floor':>7}")
    rows = []
    for R in R_list:
        avg_emp = float(np.mean(results[R]["avg"]))
        max_emp = float(np.mean(results[R]["max"]))
        floor_emp = float(np.mean(results[R]["floor"]))
        m = float(np.mean(results[R]["m"]))
        k = float(np.mean(results[R]["k"]))
        R_par = float(np.mean(results[R]["R_par"]))
        R_perp = float(np.mean(results[R]["R_perp"]))
        exc_avg = avg_emp - floor_emp
        exc_max = max_emp - floor_emp
        print(f"{R:>4} {R/r:>7.2f} {avg_emp:>9.5f} {max_emp:>9.5f} "
              f"{exc_avg:>9.5f} {exc_max:>9.5f} "
              f"{m:>3.1f} {k:>3.1f} {R_par:>5.1f} {R_perp:>6.1f} "
              f"{floor_emp:>7.4f}")
        rows.append({"R": R, "b": R/r, "exc_max": exc_max,
                     "exc_avg": exc_avg})
    # Slope fits.
    max_vals = [row["exc_max"] for row in rows if row["exc_max"] > 1e-12]
    bs = [row["b"] for row in rows if row["exc_max"] > 1e-12]
    if len(max_vals) >= 2:
        slope_max = np.polyfit(bs, np.log2(max_vals), 1)[0]
    else:
        slope_max = float("nan")
    avg_vals = [row["exc_avg"] for row in rows if row["exc_avg"] > 1e-12]
    bs_a = [row["b"] for row in rows if row["exc_avg"] > 1e-12]
    if len(avg_vals) >= 2:
        slope_avg = np.polyfit(bs_a, np.log2(avg_vals), 1)[0]
    else:
        slope_avg = float("nan")
    # Theoretical split slope: -2r/(r+m_mean-1) = -2r/(r+T_active-1).
    m_mean = float(np.mean([np.mean(results[R]["m"]) for R in R_list]))
    k_mean = float(np.mean([np.mean(results[R]["k"]) for R in R_list]))
    theory_slope_split = -2.0 * r / (r + m_mean) if (r + m_mean) > 0 else float("nan")
    print(f"  slope_max(emp)={slope_max:.3f}  slope_avg(emp)={slope_avg:.3f}  "
          f"theory_split={theory_slope_split:.3f}  naive=-1.00  LB=-2.00")


def main():
    B = 1.0
    n_trials = 300

    print("=" * 100)
    print("Day 14 — Shared-V achievability with null-space-aware bit split")
    print("=" * 100)

    # Use absolute R so we can have granular allocations.
    R_list = [8, 12, 16, 20, 24, 32]

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (3, 128, 4, ["iso", "geom", "twoscale"]),
        (4, 128, 4, ["iso", "geom", "lin", "twoscale"]),
    ]
    for T, d, r, specs in cases:
        D_list = build_D_list(specs, r)
        res = run_shared_V_cell(T, d, r, D_list, R_list, n_trials, B=B)
        label = f"T={T} r={r} {'+'.join(s[:4] for s in specs)} shared"
        print_shared_V(label, res, R_list, r)


if __name__ == "__main__":
    main()

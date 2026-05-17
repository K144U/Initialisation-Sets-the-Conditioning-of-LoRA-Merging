"""Day 14b — Fix the T=2 Chebyshev solver for non-iso D_t.

Day-14 null-space-aware split worked for iso+iso (slope -1.66,
matching theory -1.60) but failed for iso+geom, geom+twos, etc.
(slope ~ 0, excess plateau). Diagnosis: the smooth-max Newton
iterator in `cheb_center_reduced` doesn't converge for anisotropic
H_t.

Fix: for T=2 there's a closed-form parameterization.
    At Chebyshev center w_cheb, KKT: alpha_1 H_1 (w-tau_1)
    + alpha_2 H_2 (w-tau_2) = 0, with alpha_1 + alpha_2 = 1.
    Parameterize: w(alpha) = (alpha H_1 + (1-alpha) H_2)^+
                             (alpha H_1 tau_1 + (1-alpha) H_2 tau_2).
    Find alpha in [0, 1] such that D_1(w(alpha)) = D_2(w(alpha)).
    This is a 1-D root finding problem (brentq).

Replaces `cheb_center_reduced` for T=2 cells; re-runs Day-14 tests.
"""

import math
import numpy as np
from scipy.optimize import brentq
from day10_general_ht import (
    compute_Hbar_tauH_general, ht_weighted_floor_general,
    hbar_metric_sq,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day12_achievability import ht_weighted_max, ht_weighted_avg
from day13_achievability_fixes import sample_orthogonal
from day14_shared_v_gap import (
    active_set_and_gradients, parallel_perp_basis,
    allocate_bits, quantize_subspace,
)

RNG = np.random.default_rng(20260504)


def cheb_center_T2_closed(taus, Hts, Q):
    """Chebyshev center for T=2 via 1-D KKT root-find.
    Works in reduced basis Q (d_eff-dim) then lifts to R^d.
    """
    assert len(taus) == 2
    d_eff = Q.shape[1]
    H1r = Q.T @ (Hts[0] @ Q)
    H2r = Q.T @ (Hts[1] @ Q)
    t1r = Q.T @ taus[0]
    t2r = Q.T @ taus[1]

    def w_of(alpha):
        Hmix = alpha * H1r + (1 - alpha) * H2r
        rhs = alpha * H1r @ t1r + (1 - alpha) * H2r @ t2r
        return np.linalg.solve(Hmix + 1e-12 * np.eye(d_eff), rhs)

    def d1_minus_d2(alpha):
        w = w_of(alpha)
        d1 = (w - t1r) @ (H1r @ (w - t1r))
        d2 = (w - t2r) @ (H2r @ (w - t2r))
        return d1 - d2

    # Boundary checks.
    # At alpha=1: w = t1r (exact), so D_1 = 0; D_2 = (t1-t2)^T H_2 (t1-t2) > 0.
    # Hence d1_minus_d2(1) < 0.
    # At alpha=0: w = t2r, so D_2 = 0; D_1 > 0; d1_minus_d2(0) > 0.
    # Root exists in (0, 1) by IVT.
    try:
        alpha_star = brentq(d1_minus_d2, 1e-6, 1 - 1e-6, xtol=1e-8)
    except ValueError:
        # Degenerate case (e.g., tau_1 == tau_2). Fall back to mean.
        return Q @ ((t1r + t2r) / 2)
    wbar_star = w_of(alpha_star)
    return Q @ wbar_star


def merge_cheb_split_T2(taus, Hts, R, B, T, r, rng, c=5.0):
    """Null-space split merging using closed-form T=2 Chebyshev."""
    assert T == 2 and len(taus) == 2
    Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
    w_cheb = cheb_center_T2_closed(taus, Hts, Q)
    active, gradients = active_set_and_gradients(taus, Hts, w_cheb)
    P, N = parallel_perp_basis(gradients, Q)
    m = P.shape[1]
    k = N.shape[1]

    sigma_pc = B / math.sqrt(T * r)
    R_par, R_perp = allocate_bits(R, m, k)

    wcheb_Qbasis = Q.T @ w_cheb
    wpar = P.T @ wcheb_Qbasis
    wperp = N.T @ wcheb_Qbasis

    wpar_q = quantize_subspace(wpar, R_par, sigma_pc, c, rng)
    wperp_q = quantize_subspace(wperp, R_perp, sigma_pc, c, rng)

    wcheb_Qbasis_q = P @ wpar_q + N @ wperp_q
    w_star = Q @ wcheb_Qbasis_q
    return w_star, d_eff, m, k, R_par, R_perp


def run_cell(T, d, r, D_list, overlap, R_list, n_trials, B=1.0, c=5.0):
    results = {R: {"avg": [], "max": [], "floor": [],
                   "m": [], "k": [], "cheb_dist_diff": []}
               for R in R_list}
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floor = ht_weighted_floor_general(taus, Hts, tauH)
        # Verify cheb solver: d1 should equal d2.
        w_cheb = cheb_center_T2_closed(taus, Hts, Q)
        d1 = (w_cheb - taus[0]) @ (Hts[0] @ (w_cheb - taus[0]))
        d2 = (w_cheb - taus[1]) @ (Hts[1] @ (w_cheb - taus[1]))
        cheb_diff = float(abs(d1 - d2))

        for R in R_list:
            w_star, d_eff_out, m, k, R_par, R_perp = merge_cheb_split_T2(
                taus, Hts, R, B, T, r, RNG, c=c)
            results[R]["avg"].append(ht_weighted_avg(w_star, taus, Hts))
            results[R]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[R]["floor"].append(floor)
            results[R]["m"].append(m)
            results[R]["k"].append(k)
            results[R]["cheb_dist_diff"].append(cheb_diff)
    return results


def print_cell(label, results, R_list, r):
    print(f"\n--- {label} ---")
    d_diff_mean = float(np.mean([np.mean(results[R]["cheb_dist_diff"])
                                  for R in R_list]))
    print(f"  cheb_solver |D1-D2| mean={d_diff_mean:.2e}")
    print(f"{'R':>4} {'b=R/r':>7} {'avg_emp':>9} {'max_emp':>9} "
          f"{'exc_avg':>9} {'exc_max':>9} {'m':>3} {'k':>3} {'floor':>7}")
    rows = []
    for R in R_list:
        avg_emp = float(np.mean(results[R]["avg"]))
        max_emp = float(np.mean(results[R]["max"]))
        floor_emp = float(np.mean(results[R]["floor"]))
        m = float(np.mean(results[R]["m"]))
        k = float(np.mean(results[R]["k"]))
        exc_avg = avg_emp - floor_emp
        exc_max = max_emp - floor_emp
        print(f"{R:>4} {R/r:>7.2f} {avg_emp:>9.5f} {max_emp:>9.5f} "
              f"{exc_avg:>9.5f} {exc_max:>9.5f} "
              f"{m:>3.1f} {k:>3.1f} {floor_emp:>7.4f}")
        rows.append({"b": R/r, "exc_max": exc_max, "exc_avg": exc_avg})
    # Slopes.
    max_vals = [r["exc_max"] for r in rows if r["exc_max"] > 1e-12]
    bs = [r["b"] for r in rows if r["exc_max"] > 1e-12]
    if len(max_vals) >= 2:
        slope_max = np.polyfit(bs, np.log2(max_vals), 1)[0]
    else:
        slope_max = float("nan")
    avg_vals = [r["exc_avg"] for r in rows if r["exc_avg"] > 1e-12]
    bs_a = [r["b"] for r in rows if r["exc_avg"] > 1e-12]
    if len(avg_vals) >= 2:
        slope_avg = np.polyfit(bs_a, np.log2(avg_vals), 1)[0]
    else:
        slope_avg = float("nan")
    m_mean = float(np.mean([np.mean(results[R]["m"]) for R in R_list]))
    theory = -2.0 * r / (r + m_mean) if (r + m_mean) > 0 else float("nan")
    print(f"  slope_max={slope_max:.3f}  slope_avg={slope_avg:.3f}  "
          f"theory_split={theory:.3f}  naive=-1.00  LB=-2.00")


def main():
    B = 1.0
    n_trials = 300
    R_list = [8, 12, 16, 20, 24, 32]

    print("=" * 100)
    print("Day 14b — Closed-form T=2 Chebyshev solver + null-space split")
    print("=" * 100)

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (2, 128, 4, ["lin", "twoscale"]),
        (2, 128, 4, ["geom", "iso"]),  # order swap
    ]
    for T, d, r, specs in cases:
        D_list = build_D_list(specs, r)
        res = run_cell(T, d, r, D_list, "shared", R_list, n_trials, B=B)
        print_cell(f"T={T} r={r} {'+'.join(s[:4] for s in specs)} shared",
                   res, R_list, r)


if __name__ == "__main__":
    main()

"""Day 15 — General-T Chebyshev center via SOCP + null-space split.

Phase 2.5 Item A. Day 14b's closed-form Chebyshev solver is T=2
only. For T >= 3, we solve the SOCP

    min_{w, s}   s
    s.t.         (w - tau_t)^T H_t (w - tau_t) <= s   for all t

directly with cvxpy (CLARABEL solver). Then the null-space split
pipeline from day14c + clip c=11.5 (locked in day14g) runs
unchanged. Theory: slope_max = -2r/(r + |A| - 1). For r=4:
  |A|=2 → -8/5 = -1.60  (T=2 baseline, confirmed at Day 14g)
  |A|=3 → -8/6 ≈ -1.33
  |A|=4 → -8/7 ≈ -1.14
"""

import math
import numpy as np
import cvxpy as cp
from day10_general_ht import (
    compute_Hbar_tauH_general, ht_weighted_floor_general,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day12_achievability import ht_weighted_max, ht_weighted_avg
from day14_shared_v_gap import (
    active_set_and_gradients, parallel_perp_basis, allocate_bits,
)
from day14c_fractional_bits import quantize_subspace_frac

RNG = np.random.default_rng(20260510)


def _refine_kkt_gauss_newton(w_r, tau_r, Ht_r_list, active,
                              n_iter=200, tol=1e-14, verbose=False):
    """Gauss-Newton on the square KKT system.

    Unknowns: z = (w, alpha)  in R^{d_eff + |A|}
    Equations (square: d_eff + |A|):
        F1_k  = sum_i alpha_i (H_i (w - tau_i))_k    for k=1..d_eff
        F2_i  = (w - tau_i)^T H_i (w - tau_i) - s    for i=1..|A|-1
                (uses i as reference; D_i - D_ref for i != ref)
        F3    = sum_i alpha_i - 1
    Here s is eliminated by using D_ref - D_i for i != ref.

    Actually cleaner: work with (w, alpha) only, |A| + d_eff unknowns,
    same count of equations:
        stationarity: d_eff eqs
        equalization: |A|-1 eqs  (D_i - D_1 = 0)
        normalization: 1 eq  (sum alpha = 1)
    """
    d_eff = w_r.shape[0]
    A = len(active)
    H_A = [Ht_r_list[t] for t in active]
    tau_A = [tau_r[t] for t in active]

    # Init
    w = w_r.copy()
    alpha = np.ones(A) / A
    n = d_eff + A

    for it in range(n_iter):
        # Build residual F(w, alpha)
        F = np.zeros(n)
        # stationarity
        Hmix = sum(alpha[i] * H_A[i] for i in range(A))
        rhs = sum(alpha[i] * (H_A[i] @ tau_A[i]) for i in range(A))
        F[:d_eff] = Hmix @ w - rhs
        # equalization (D_i - D_0 = 0 for i = 1..A-1)
        D = np.array([(w - tau_A[i]) @ (H_A[i] @ (w - tau_A[i]))
                      for i in range(A)])
        F[d_eff:d_eff + A - 1] = D[1:] - D[0]
        F[-1] = alpha.sum() - 1.0

        res = float(np.linalg.norm(F))
        if verbose and it % 5 == 0:
            print(f"    it={it} |F|={res:.3e}")
        if res < tol:
            break

        # Jacobian
        J = np.zeros((n, n))
        # d stationarity / dw = Hmix
        J[:d_eff, :d_eff] = Hmix
        # d stationarity / dalpha_i = H_i (w - tau_i)
        for i in range(A):
            J[:d_eff, d_eff + i] = H_A[i] @ (w - tau_A[i])
        # d equalization / dw: 2 (w - tau_i)^T H_i - 2 (w - tau_0)^T H_0
        for i in range(1, A):
            J[d_eff + i - 1, :d_eff] = (
                2.0 * (H_A[i] @ (w - tau_A[i]) - H_A[0] @ (w - tau_A[0])))
        # d normalization / dalpha_i = 1
        J[-1, d_eff:] = 1.0

        # Solve J * dz = -F; regularize
        try:
            dz = np.linalg.solve(J + 1e-12 * np.eye(n), -F)
        except np.linalg.LinAlgError:
            break
        # damped step
        step = 1.0
        for _ in range(10):
            w_try = w + step * dz[:d_eff]
            alpha_try = alpha + step * dz[d_eff:]
            if (alpha_try >= -1e-8).all():
                break
            step *= 0.5
        w = w_try
        alpha = np.maximum(alpha_try, 0.0)
        if alpha.sum() > 0:
            alpha /= alpha.sum()

    return w


def cheb_center_general_T_socp(taus, Hts, Q, solver="CLARABEL",
                                refine=True, tol_socp=1e-12):
    """Chebyshev center in the reduced Q-basis (d_eff-dim) via SOCP.

    In the Q-basis: tau_t_r = Q^T tau_t (d_eff-vec),
    H_t_r = Q^T H_t Q (d_eff x d_eff psd).

    Problem:
        min  s
        s.t. quad_form(w - tau_t_r, H_t_r) <= s    for all t

    cvxpy handles the positive-semidefinite quadratic form natively
    when H_t_r is declared PSD. For possibly rank-deficient H_t_r,
    we factor H_t_r = L_t L_t^T via eigh and rewrite the quadratic
    as ||L_t^T (w - tau_t_r)||^2 <= s, which is an SOC constraint.
    """
    T = len(taus)
    d_eff = Q.shape[1]

    # Reduce to Q-basis and factor each H_t_r.
    tau_r = [Q.T @ taus[t] for t in range(T)]
    L_list = []
    for t in range(T):
        Ht_r = Q.T @ (Hts[t] @ Q)
        # Symmetrize to kill numerical noise.
        Ht_r = 0.5 * (Ht_r + Ht_r.T)
        w_eig, V = np.linalg.eigh(Ht_r)
        mask = w_eig > 1e-10 * max(w_eig.max(), 1.0)
        # L = V[:, mask] · diag(sqrt(w[mask])); so L L^T = H_t_r.
        L = V[:, mask] * np.sqrt(w_eig[mask])
        L_list.append(L)  # d_eff x rank_t

    w = cp.Variable(d_eff)
    s = cp.Variable()
    constraints = []
    for t in range(T):
        # ||L_t^T (w - tau_t_r)||^2 <= s  ⇔  ||L_t^T (w-tau)|| <= sqrt(s).
        # cvxpy's cp.SOC handles this via:
        #   cp.norm(L_t^T (w - tau_t_r), 2) <= cp.sqrt(s) is disallowed
        # (not DCP). Instead: use cp.quad_form(w - tau_t_r, Ht_r) <= s.
        # But Ht_r might be rank-deficient; use psd_wrap.
        # Cleanest: use SOC via auxiliary variable.
        #   ||L_t^T(w-tau_t_r)||^2 <= s  is equivalent to the SOC
        #   constraint on (s, L_t^T(w-tau_t_r)) with the quad-form
        #   rewrite: cp.sum_squares(L_t^T (w - tau_t_r)) <= s.
        resid = L_list[t].T @ (w - tau_r[t])
        constraints.append(cp.sum_squares(resid) <= s)

    prob = cp.Problem(cp.Minimize(s), constraints)
    # Solver default tolerance is fine; we'll Newton-refine after.
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SOCP failed: {prob.status}")

    w_cheb_r = np.asarray(w.value)

    if refine:
        # Identify active set based on SOCP solution, then Newton-refine.
        H_r_list = [0.5 * (Q.T @ (Hts[t] @ Q) + (Q.T @ (Hts[t] @ Q)).T)
                    for t in range(T)]
        dists = np.array([float((w_cheb_r - tau_r[t]) @
                                (H_r_list[t] @ (w_cheb_r - tau_r[t])))
                          for t in range(T)])
        max_d = dists.max()
        # Active = tasks within 1% of max (conservative; catches near-ties).
        active = [t for t in range(T) if dists[t] >= max_d - 1e-2 * max(max_d, 1.0)]
        if len(active) >= 2:
            w_cheb_r = _refine_kkt_gauss_newton(
                w_cheb_r, tau_r, H_r_list, active)

    return Q @ w_cheb_r


def merge_general_T(taus, Hts, R, B, T, r, rng, c=11.5):
    """General-T null-space split merge with SOCP Chebyshev solver
    + fractional bits + locked clip c=11.5."""
    Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
    w_cheb = cheb_center_general_T_socp(taus, Hts, Q)
    active, gradients = active_set_and_gradients(taus, Hts, w_cheb)
    P, N = parallel_perp_basis(gradients, Q)
    m = P.shape[1]
    k = N.shape[1]

    sigma_pc = B / math.sqrt(T * r)
    R_par, R_perp = allocate_bits(R, m, k)

    wcheb_Qbasis = Q.T @ w_cheb
    wpar = P.T @ wcheb_Qbasis
    wperp = N.T @ wcheb_Qbasis

    wpar_q = quantize_subspace_frac(wpar, R_par, sigma_pc, c, rng)
    wperp_q = quantize_subspace_frac(wperp, R_perp, sigma_pc, c, rng)

    wcheb_Qbasis_q = P @ wpar_q + N @ wperp_q
    w_star = Q @ wcheb_Qbasis_q
    return w_star, d_eff, m, k, R_par, R_perp


def cheb_diff_metric(taus, Hts, w_cheb):
    """Max_t D_t minus min_t D_t at Chebyshev center.
    Should be ~0 to numerical tolerance if all tasks are active."""
    T = len(taus)
    ds = np.array([float((w_cheb - taus[t]) @ (Hts[t] @ (w_cheb - taus[t])))
                   for t in range(T)])
    return float(ds.max() - ds.min())


def run_cell(T, d, r, D_list, R_list, n_trials, B=1.0, c=11.5):
    results = {R: {"avg": [], "max": [], "floor_avg": [], "cheb_sq": [],
                   "m": [], "k": [], "cheb_diff": [], "n_active": []}
               for R in R_list}
    for trial in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, "shared", RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floor_avg = ht_weighted_floor_general(taus, Hts, tauH)
        try:
            w_cheb = cheb_center_general_T_socp(taus, Hts, Q)
        except Exception as e:
            print(f"  trial {trial}: SOCP failed ({e}), skipping")
            continue
        cheb_sq = ht_weighted_max(w_cheb, taus, Hts)
        cheb_diff = cheb_diff_metric(taus, Hts, w_cheb)
        active, _ = active_set_and_gradients(taus, Hts, w_cheb)

        for R in R_list:
            w_star, d_eff_out, m, k, R_par, R_perp = merge_general_T(
                taus, Hts, R, B, T, r, RNG, c=c)
            results[R]["avg"].append(ht_weighted_avg(w_star, taus, Hts))
            results[R]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[R]["floor_avg"].append(floor_avg)
            results[R]["cheb_sq"].append(cheb_sq)
            results[R]["m"].append(m)
            results[R]["k"].append(k)
            results[R]["cheb_diff"].append(cheb_diff)
            results[R]["n_active"].append(len(active))
    return results


def print_cell(label, results, R_list, r, T):
    """Report BOTH exc vs avg-floor (classical; flat for shared-V T>=3
    due to cheb != tauH) AND exc vs cheb^2 (per-trial Chebyshev value)
    which is what the null-space-split theory predicts should decay at
    slope -2r/(r+|A|-1)."""
    print(f"\n--- {label} ---", flush=True)
    cheb_mean = float(np.mean([np.mean(results[R]["cheb_diff"])
                                 for R in R_list]))
    n_active_mean = float(np.mean([np.mean(results[R]["n_active"])
                                     for R in R_list]))
    m_mean = float(np.mean([np.mean(results[R]["m"]) for R in R_list]))
    print(f"  KKT |D_max - D_min on active| mean={cheb_mean:.2e}  "
          f"|A|_mean={n_active_mean:.2f}  m_mean={m_mean:.2f}", flush=True)
    print(f"{'R':>4} {'b=R/r':>7} {'max_emp':>9} {'cheb_sq':>9} "
          f"{'floor_avg':>10} {'exc_vs_floor':>13} {'exc_vs_cheb':>12} "
          f"{'m':>3} {'k':>3}", flush=True)
    rows = []
    for R in R_list:
        avg_emp = float(np.mean(results[R]["avg"]))
        max_emp = float(np.mean(results[R]["max"]))
        floor_avg = float(np.mean(results[R]["floor_avg"]))
        cheb_sq = float(np.mean(results[R]["cheb_sq"]))
        m = float(np.mean(results[R]["m"]))
        k = float(np.mean(results[R]["k"]))
        exc_vs_floor = max_emp - floor_avg
        exc_vs_cheb = max_emp - cheb_sq
        print(f"{R:>4} {R/r:>7.2f} {max_emp:>9.5f} {cheb_sq:>9.5f} "
              f"{floor_avg:>10.5f} {exc_vs_floor:>13.5f} "
              f"{exc_vs_cheb:>12.5f} {m:>3.1f} {k:>3.1f}", flush=True)
        rows.append({"b": R/r, "exc_vs_cheb": exc_vs_cheb,
                     "exc_vs_floor": exc_vs_floor,
                     "exc_avg": avg_emp - floor_avg})

    def _slope(rows, key):
        vals = [row[key] for row in rows if row[key] > 1e-12]
        bs = [row["b"] for row in rows if row[key] > 1e-12]
        if len(vals) < 2:
            return float("nan")
        return float(np.polyfit(bs, np.log2(vals), 1)[0])

    slope_vs_cheb = _slope(rows, "exc_vs_cheb")
    slope_vs_floor = _slope(rows, "exc_vs_floor")
    slope_avg = _slope(rows, "exc_avg")
    theory = -2.0 * r / (r + n_active_mean - 1) if (r + n_active_mean - 1) > 0 else float("nan")
    print(f"  slope(max - cheb^2)={slope_vs_cheb:.3f}  "
          f"slope(max - floor_avg)={slope_vs_floor:.3f}  "
          f"slope(avg - floor_avg)={slope_avg:.3f}  "
          f"theory for |A|={n_active_mean:.1f}: {theory:.3f}",
          flush=True)
    return slope_vs_cheb, slope_avg


def main():
    B = 1.0
    n_trials = 250     # cvxpy is slower; 250 is plenty w/ our variance
    R_list = [8, 12, 16, 20, 24, 32]
    c = 11.5           # locked in day14g

    print("=" * 110, flush=True)
    print(f"Day 15 — General-T Chebyshev via SOCP, clip c={c}, n_trials={n_trials}",
          flush=True)
    print("=" * 110, flush=True)

    cases = [
        # T=2 sanity (regression from day14g, should reproduce -1.60-ish).
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        # T=3: expected slope -8/6 ≈ -1.33.
        (3, 128, 4, ["iso", "geom", "twoscale"]),
        (3, 128, 4, ["iso", "iso", "iso"]),
        (3, 128, 4, ["geom", "twoscale", "lin"]),
        # T=4: expected slope -8/7 ≈ -1.14.
        (4, 128, 4, ["iso", "geom", "lin", "twoscale"]),
        (4, 128, 4, ["iso", "iso", "iso", "iso"]),
    ]

    summary = []
    for T, d, r, specs in cases:
        D_list = build_D_list(specs, r)
        label = f"T={T} r={r} {'+'.join(s[:4] for s in specs)} shared"
        res = run_cell(T, d, r, D_list, R_list, n_trials, B=B, c=c)
        slope_vs_cheb, slope_avg = print_cell(label, res, R_list, r, T)
        n_active_mean = float(np.mean([np.mean(res[R]["n_active"])
                                         for R in R_list]))
        theory = -2.0 * r / (r + n_active_mean - 1)
        summary.append((label, T, slope_vs_cheb, slope_avg, theory,
                        n_active_mean))

    print("\n" + "=" * 110, flush=True)
    print("Summary: slope(max_emp - cheb_sq) vs theory -2r/(r+|A|-1):",
          flush=True)
    print(f"{'case':<50} {'T':>3} {'|A|':>5} {'theory':>8} "
          f"{'emp_slope':>10} {'gap':>7}", flush=True)
    for label, T, slope_vs_cheb, slope_avg, theory, n_a in summary:
        gap = slope_vs_cheb - theory
        print(f"{label:<50} {T:>3} {n_a:>5.2f} {theory:>+8.3f} "
              f"{slope_vs_cheb:>+10.3f} {gap:>+7.3f}", flush=True)


if __name__ == "__main__":
    main()

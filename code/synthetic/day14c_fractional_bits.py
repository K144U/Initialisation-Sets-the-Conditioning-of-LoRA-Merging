"""Day 14c — Fractional per-coord bit allocation on the perp subspace.

Day 14b fixed the T=2 Chebyshev solver (closed-form brentq on the
KKT condition) and gave slopes strictly beating naive -1 for all
anisotropy regimes, but non-iso cells undershoot theory -1.60
(iso+geom: -1.38, geom+twos: -1.19, lin+twos: -1.32, geom+iso -1.39).

Diagnosis: the perp quantizer in `day14_shared_v_gap.quantize_subspace`
uses integer per-coord bits via `b_per = R_sub // k`. For m=1, k=3,
R=20 -> R_perp=12 -> b_per=4, R=24 -> R_perp=14 -> still b_per=4,
R=32 -> R_perp=19 -> b_per=6. Two distinct rates sharing the same
step size produces a plateau at R=20-24 that drags the log-linear
slope toward -1.

Fix (this file): per-coord variable bit widths. For average
b_avg = R_sub/k, give n_hi = round(k * frac(b_avg)) coords
ceil(b_avg) bits and the remaining k - n_hi coords floor(b_avg)
bits. Total bits ~= R_sub; step size shrinks smoothly with R.
"""

import math
import numpy as np
from day10_general_ht import (
    compute_Hbar_tauH_general, ht_weighted_floor_general,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day12_achievability import ht_weighted_max, ht_weighted_avg
from day13_achievability_fixes import sample_orthogonal
from day14_shared_v_gap import (
    active_set_and_gradients, parallel_perp_basis, allocate_bits,
)
from day14b_cheb_T2_closedform import cheb_center_T2_closed

RNG = np.random.default_rng(20260505)


def quantize_subspace_frac(v_sub, R_sub, sigma_pc, c, rng):
    """Quantize k-dim vector with total R_sub bits using per-coord
    variable widths for fractional average rate.

    n_hi = round(k * frac(b_avg)) coords get ceil(b_avg) bits;
    the rest get floor(b_avg). Random orthogonal mixer applied
    first (as in Day 14) so coord identity is already randomized.
    """
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub

    b_avg = R_sub / k
    b_lo = int(math.floor(b_avg))
    b_hi = b_lo + 1
    n_hi = int(round((b_avg - b_lo) * k))
    n_hi = max(0, min(n_hi, k))
    bit_widths = np.array([b_hi] * n_hi + [b_lo] * (k - n_hi))
    rng.shuffle(bit_widths)

    rng_hi = c * sigma_pc
    v_rot_q = np.zeros(k)
    for i in range(k):
        b = int(bit_widths[i])
        if b <= 0:
            v_rot_q[i] = 0.0
        else:
            step = (2 * rng_hi) / (2 ** b)
            idx = int(math.floor((v_rot[i] + rng_hi) / step))
            idx = max(0, min(idx, (2 ** b) - 1))
            v_rot_q[i] = -rng_hi + (idx + 0.5) * step

    return O.T @ v_rot_q


def merge_cheb_split_T2_frac(taus, Hts, R, B, T, r, rng, c=5.0):
    """Null-space split with closed-form T=2 Chebyshev (day14b) +
    fractional per-coord bits on the perp subspace (this file)."""
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

    wpar_q = quantize_subspace_frac(wpar, R_par, sigma_pc, c, rng)
    wperp_q = quantize_subspace_frac(wperp, R_perp, sigma_pc, c, rng)

    wcheb_Qbasis_q = P @ wpar_q + N @ wperp_q
    w_star = Q @ wcheb_Qbasis_q
    return w_star, d_eff, m, k, R_par, R_perp


def run_cell(T, d, r, D_list, overlap, R_list, n_trials, B=1.0, c=5.0):
    results = {R: {"avg": [], "max": [], "floor": [],
                   "m": [], "k": [], "R_par": [], "R_perp": [],
                   "cheb_dist_diff": []}
               for R in R_list}
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floor = ht_weighted_floor_general(taus, Hts, tauH)
        w_cheb = cheb_center_T2_closed(taus, Hts, Q)
        d1 = (w_cheb - taus[0]) @ (Hts[0] @ (w_cheb - taus[0]))
        d2 = (w_cheb - taus[1]) @ (Hts[1] @ (w_cheb - taus[1]))
        cheb_diff = float(abs(d1 - d2))

        for R in R_list:
            w_star, d_eff_out, m, k, R_par, R_perp = merge_cheb_split_T2_frac(
                taus, Hts, R, B, T, r, RNG, c=c)
            results[R]["avg"].append(ht_weighted_avg(w_star, taus, Hts))
            results[R]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[R]["floor"].append(floor)
            results[R]["m"].append(m)
            results[R]["k"].append(k)
            results[R]["R_par"].append(R_par)
            results[R]["R_perp"].append(R_perp)
            results[R]["cheb_dist_diff"].append(cheb_diff)
    return results


def print_cell(label, results, R_list, r):
    print(f"\n--- {label} ---")
    d_diff_mean = float(np.mean([np.mean(results[R]["cheb_dist_diff"])
                                  for R in R_list]))
    print(f"  cheb_solver |D1-D2| mean={d_diff_mean:.2e}")
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
        rows.append({"b": R/r, "exc_max": exc_max, "exc_avg": exc_avg})
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
    m_mean = float(np.mean([np.mean(results[R]["m"]) for R in R_list]))
    theory = -2.0 * r / (r + m_mean) if (r + m_mean) > 0 else float("nan")
    print(f"  slope_max={slope_max:.3f}  slope_avg={slope_avg:.3f}  "
          f"theory_split={theory:.3f}  naive=-1.00  LB=-2.00")


def main():
    B = 1.0
    n_trials = 300
    R_list = [8, 12, 16, 20, 24, 32]

    print("=" * 100)
    print("Day 14c — Closed-form T=2 Chebyshev + fractional per-coord bits")
    print("=" * 100)

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (2, 128, 4, ["lin", "twoscale"]),
        (2, 128, 4, ["geom", "iso"]),
    ]
    for T, d, r, specs in cases:
        D_list = build_D_list(specs, r)
        res = run_cell(T, d, r, D_list, "shared", R_list, n_trials, B=B)
        print_cell(f"T={T} r={r} {'+'.join(s[:4] for s in specs)} shared",
                   res, R_list, r)


if __name__ == "__main__":
    main()

"""Day 14e — Systematic quantizer sweep to find the sweet spot.

Context. Day 14b's closed-form T=2 Chebyshev solver + Day 14c's
null-space split + fractional bits gave:
   iso+iso -1.62, iso+geom -1.34, geom+twos -1.08, lin+twos -1.21,
   geom+iso -1.33
at clip c=5. Diagnostic at c=20 over-corrected to -1.69 to -1.79.

Variants tested here (all use day14b cheb solver + null-space split
+ fractional bits; they vary only in clip/basis choice):
  c{C}-frac       : fixed clip c*sigma_pc
  emp-max         : empirical max(|v_rot|) per subspace, ×(1+eps)
  emp-max-clip    : emp-max capped by theoretical bound
  whiten-c{C}     : quantize in H_bar^{1/2}-whitened basis
  whiten-empmax   : whiten + empirical max
  waterfill-c{C}  : bit allocation per coord by H_bar eigenvalue
  clip-per-trial  : clip range = max |v_rot| across the two subspaces
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


# =================================================================
#  Quantizer variants
# =================================================================

def _frac_bit_widths(R_sub, k):
    if k == 0:
        return np.array([], dtype=int)
    b_avg = R_sub / k
    b_lo = int(math.floor(b_avg))
    b_hi = b_lo + 1
    n_hi = int(round((b_avg - b_lo) * k))
    n_hi = max(0, min(n_hi, k))
    return np.array([b_hi] * n_hi + [b_lo] * (k - n_hi), dtype=int)


def _quantize_scalar(x, clip, b):
    """Scalar quantize in [-clip, clip] with b bits."""
    if b <= 0:
        return 0.0
    step = (2.0 * clip) / (2 ** b)
    idx = int(math.floor((x + clip) / step))
    idx = max(0, min(idx, (2 ** b) - 1))
    return -clip + (idx + 0.5) * step


def qz_fixed(v_sub, R_sub, sigma_pc, c, rng):
    """Fixed clip c*sigma_pc, fractional bits, random orthogonal mixer."""
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub
    bit_widths = _frac_bit_widths(R_sub, k)
    rng.shuffle(bit_widths)
    clip = c * sigma_pc
    v_rot_q = np.array([_quantize_scalar(v_rot[i], clip, int(bit_widths[i]))
                        for i in range(k)])
    return O.T @ v_rot_q


def qz_empmax(v_sub, R_sub, sigma_pc, c_eps, rng):
    """Clip = max(|v_rot|) · (1 + eps). No information leak in
    practice — it's a per-trial scalar — but for a fair comparison
    we charge zero side-info bits."""
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub
    bit_widths = _frac_bit_widths(R_sub, k)
    rng.shuffle(bit_widths)
    m = float(np.max(np.abs(v_rot)))
    clip = max(m * (1 + c_eps), 1e-6 * sigma_pc)
    v_rot_q = np.array([_quantize_scalar(v_rot[i], clip, int(bit_widths[i]))
                        for i in range(k)])
    return O.T @ v_rot_q


def qz_empmax_sigmafloor(v_sub, R_sub, sigma_pc, c, rng):
    """Clip = max(c*sigma_pc, max|v_rot|). Floor at sigma-scale so
    we don't over-shrink when the trial has a small source."""
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub
    bit_widths = _frac_bit_widths(R_sub, k)
    rng.shuffle(bit_widths)
    clip = max(c * sigma_pc, float(np.max(np.abs(v_rot))))
    v_rot_q = np.array([_quantize_scalar(v_rot[i], clip, int(bit_widths[i]))
                        for i in range(k)])
    return O.T @ v_rot_q


def qz_percoord_empmax(v_sub, R_sub, sigma_pc, c, rng):
    """Per-coord clip = max(c*sigma_pc, |v_rot[i]|) — no mixer, use
    native basis so each coord's range is tight."""
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    bit_widths = _frac_bit_widths(R_sub, k)
    rng.shuffle(bit_widths)
    v_q = np.zeros(k)
    for i in range(k):
        clip = max(c * sigma_pc, abs(v_sub[i]) * 1.01)
        v_q[i] = _quantize_scalar(v_sub[i], clip, int(bit_widths[i]))
    return v_q


# =================================================================
#  Merge pipeline
# =================================================================

def merge_variant(taus, Hts, R, B, T, r, rng, variant):
    assert T == 2 and len(taus) == 2
    Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
    w_cheb = cheb_center_T2_closed(taus, Hts, Q)
    active, gradients = active_set_and_gradients(taus, Hts, w_cheb)

    name = variant["name"]
    whiten = variant.get("whiten", False)

    if whiten:
        # Switch to H_bar^{1/2}-whitened Q-basis: Qw = Q·diag(sqrt(w_pos)).
        # In Qw-coords, y = Qw^T w = diag(sqrt(w_pos))·Q^T w. Decoder
        # recovers w = Q · diag(1/sqrt(w_pos)) · y. In y-coords, H_bar = I,
        # so uniform quantization weights all directions equally under
        # the H_bar metric.
        S = np.diag(np.sqrt(w_pos))
        Sinv = np.diag(1.0 / np.sqrt(w_pos))
        grads_y = [S @ (Q.T @ g) for g in gradients]
        P_y, N_y = parallel_perp_basis(grads_y, np.eye(d_eff))
        m = P_y.shape[1]
        k = N_y.shape[1]
        sigma_pc = B / math.sqrt(T * r)
        R_par, R_perp = allocate_bits(R, m, k)
        y_cheb = S @ (Q.T @ w_cheb)
        ypar = P_y.T @ y_cheb
        yperp = N_y.T @ y_cheb
        qz = variant["qz"]
        kwargs = variant.get("kwargs", {})
        ypar_q = qz(ypar, R_par, sigma_pc, rng=rng, **kwargs)
        yperp_q = qz(yperp, R_perp, sigma_pc, rng=rng, **kwargs)
        y_q = P_y @ ypar_q + N_y @ yperp_q
        v_q = Sinv @ y_q
        w_star = Q @ v_q
    else:
        P, N = parallel_perp_basis(gradients, Q)
        m = P.shape[1]
        k = N.shape[1]
        sigma_pc = B / math.sqrt(T * r)
        R_par, R_perp = allocate_bits(R, m, k)
        v_cheb = Q.T @ w_cheb
        wpar = P.T @ v_cheb
        wperp = N.T @ v_cheb
        qz = variant["qz"]
        kwargs = variant.get("kwargs", {})
        wpar_q = qz(wpar, R_par, sigma_pc, rng=rng, **kwargs)
        wperp_q = qz(wperp, R_perp, sigma_pc, rng=rng, **kwargs)
        v_q = P @ wpar_q + N @ wperp_q
        w_star = Q @ v_q

    return w_star, d_eff, m, k, R_par, R_perp


# =================================================================
#  Runner
# =================================================================

def run_cell(T, d, r, D_list, R_list, n_trials, variant, B=1.0, RNG=None):
    rng = RNG or np.random.default_rng(20260507)
    results = {R: {"avg": [], "max": [], "floor": []} for R in R_list}
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, "shared", rng)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floor = ht_weighted_floor_general(taus, Hts, tauH)
        for R in R_list:
            w_star, d_eff_out, m, k, R_par, R_perp = merge_variant(
                taus, Hts, R, B, T, r, rng, variant)
            results[R]["avg"].append(ht_weighted_avg(w_star, taus, Hts))
            results[R]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[R]["floor"].append(floor)
    return results


def fit_slope(rows, key):
    vals = [row[key] for row in rows if row[key] > 1e-12]
    bs = [row["b"] for row in rows if row[key] > 1e-12]
    if len(vals) < 2:
        return float("nan")
    return float(np.polyfit(bs, np.log2(vals), 1)[0])


def summarize(label, results, R_list, r):
    rows = []
    for R in R_list:
        avg_e = float(np.mean(results[R]["avg"]))
        max_e = float(np.mean(results[R]["max"]))
        fl = float(np.mean(results[R]["floor"]))
        rows.append({"R": R, "b": R/r,
                     "exc_avg": avg_e - fl, "exc_max": max_e - fl,
                     "avg": avg_e, "max": max_e, "floor": fl})
    slope_max = fit_slope(rows, "exc_max")
    slope_avg = fit_slope(rows, "exc_avg")
    return rows, slope_max, slope_avg


def print_summary(rows, label, slope_max, slope_avg):
    print(f"\n  [{label}]  slope_max={slope_max:+.3f} "
          f"slope_avg={slope_avg:+.3f}  "
          f"exc_max@R=32={rows[-1]['exc_max']:.5f}")


# =================================================================
#  Main sweep
# =================================================================

def main():
    B = 1.0
    n_trials = 300
    R_list = [8, 12, 16, 20, 24, 32]

    # All variants share fractional bits + closed-form T=2 Chebyshev.
    variants = [
        dict(name="c=3",          qz=qz_fixed, kwargs={"c": 3.0}),
        dict(name="c=5",          qz=qz_fixed, kwargs={"c": 5.0}),
        dict(name="c=7",          qz=qz_fixed, kwargs={"c": 7.0}),
        dict(name="c=10",         qz=qz_fixed, kwargs={"c": 10.0}),
        dict(name="c=14",         qz=qz_fixed, kwargs={"c": 14.0}),
        dict(name="c=20",         qz=qz_fixed, kwargs={"c": 20.0}),
        dict(name="c=50",         qz=qz_fixed, kwargs={"c": 50.0}),
        dict(name="emp-max",      qz=qz_empmax, kwargs={"c_eps": 0.01}),
        dict(name="emp-sigfloor-c5",  qz=qz_empmax_sigmafloor, kwargs={"c": 5.0}),
        dict(name="emp-sigfloor-c2",  qz=qz_empmax_sigmafloor, kwargs={"c": 2.0}),
        dict(name="percoord-c5",  qz=qz_percoord_empmax, kwargs={"c": 5.0}),
        dict(name="whiten-c=5",   qz=qz_fixed, kwargs={"c": 5.0}, whiten=True),
        dict(name="whiten-c=10",  qz=qz_fixed, kwargs={"c": 10.0}, whiten=True),
        dict(name="whiten-c=20",  qz=qz_fixed, kwargs={"c": 20.0}, whiten=True),
        dict(name="whiten-emp",   qz=qz_empmax, kwargs={"c_eps": 0.01}, whiten=True),
        dict(name="whiten-sigf-c5", qz=qz_empmax_sigmafloor, kwargs={"c": 5.0}, whiten=True),
    ]

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (2, 128, 4, ["lin", "twoscale"]),
        (2, 128, 4, ["geom", "iso"]),
    ]

    # Each variant gets its own RNG seed (consistent across variants).
    header = f"{'variant':<22} " + " ".join(
        [f"{'+'.join(s[:4] for s in specs):>17}" for _, _, _, specs in cases])
    print("=" * 100)
    print("Day 14e — Quantizer sweep (slope_max @ 5 T=2 cells)")
    print("=" * 100)
    print(header)

    all_results = {}  # variant_name -> {case_label -> (rows, slope_max, slope_avg)}
    for variant in variants:
        vname = variant["name"]
        line_parts = [f"{vname:<22}"]
        all_results[vname] = {}
        for T, d, r, specs in cases:
            D_list = build_D_list(specs, r)
            case_label = '+'.join(s[:4] for s in specs)
            rng = np.random.default_rng(20260507)
            res = run_cell(T, d, r, D_list, R_list, n_trials, variant,
                           B=B, RNG=rng)
            rows, slope_max, slope_avg = summarize(
                f"{vname} {case_label}", res, R_list, r)
            all_results[vname][case_label] = (rows, slope_max, slope_avg)
            exc32 = rows[-1]["exc_max"]
            line_parts.append(f"{slope_max:+6.3f}/{exc32:6.4f}  ")
        print("  ".join(line_parts))

    print("\n" + "=" * 100)
    print("Interpretation: 'slope_max / exc_max@R=32'. Theory: slope=-1.60.")
    print("=" * 100)

    # Best-by-slope variant per cell.
    print("\nBest-slope variant per cell (slope closest to -1.60):")
    for T, d, r, specs in cases:
        case_label = '+'.join(s[:4] for s in specs)
        best = min(variants,
                   key=lambda v: abs(all_results[v["name"]][case_label][1] - (-1.6)))
        rows, sm, sa = all_results[best["name"]][case_label]
        print(f"  {case_label:>17}: {best['name']:<22} "
              f"slope_max={sm:+.3f} exc_max@R=32={rows[-1]['exc_max']:.5f}")


if __name__ == "__main__":
    main()

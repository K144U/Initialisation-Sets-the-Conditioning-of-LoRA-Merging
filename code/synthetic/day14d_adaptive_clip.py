"""Day 14d — Rate-adaptive clip range on top of fractional bits.

Day 14c's fractional-bit fix closed the R=20-24 plateau but left a
floor-like tail at high R in non-iso cells. Day 14c diagnostic with
c=20 showed slopes recover to theory ≈ -1.60 / -1.80 when clip is
widened, confirming clipping was the cause. But c=20 is too loose
at low R (step size inflates excess there).

Fix: rate-adaptive clip per coord: c_eff = c0 * sqrt(2 * b * ln 2) *
sigma_pc. This is the standard Gaussian scalar-quantizer design --
at b bits, the clip scales as sqrt(b), so step (2 * clip / 2^b)
shrinks as sqrt(b) * 2^{-b}, and no clipping at high R.
"""

import math
import numpy as np
from day10_general_ht import compute_Hbar_tauH_general
from day11_task_dep_Dt import build_D_list
from day13_achievability_fixes import sample_orthogonal
from day14_shared_v_gap import (
    active_set_and_gradients, parallel_perp_basis, allocate_bits,
)
from day14b_cheb_T2_closedform import cheb_center_T2_closed
from day14c_fractional_bits import run_cell, print_cell

RNG = np.random.default_rng(20260506)


def quantize_subspace_adaptive(v_sub, R_sub, sigma_pc, c, rng):
    """Per-coord variable bits + rate-adaptive clip.
    clip_i = c * sqrt(max(b_i, 1)) * sigma_pc.
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

    v_rot_q = np.zeros(k)
    for i in range(k):
        b = int(bit_widths[i])
        if b <= 0:
            v_rot_q[i] = 0.0
        else:
            rng_hi = c * math.sqrt(b) * sigma_pc
            step = (2 * rng_hi) / (2 ** b)
            idx = int(math.floor((v_rot[i] + rng_hi) / step))
            idx = max(0, min(idx, (2 ** b) - 1))
            v_rot_q[i] = -rng_hi + (idx + 0.5) * step

    return O.T @ v_rot_q


def merge_cheb_split_T2_adaptive(taus, Hts, R, B, T, r, rng, c=2.0):
    """Day 14d merge: closed-form T=2 Chebyshev + fractional bits +
    rate-adaptive clip."""
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

    wpar_q = quantize_subspace_adaptive(wpar, R_par, sigma_pc, c, rng)
    wperp_q = quantize_subspace_adaptive(wperp, R_perp, sigma_pc, c, rng)

    wcheb_Qbasis_q = P @ wpar_q + N @ wperp_q
    w_star = Q @ wcheb_Qbasis_q
    return w_star, d_eff, m, k, R_par, R_perp


# Patch day14c's merge function to route through adaptive variant,
# then reuse run_cell / print_cell unchanged.
import day14c_fractional_bits as _d14c
_d14c.merge_cheb_split_T2_frac = merge_cheb_split_T2_adaptive


def main():
    B = 1.0
    n_trials = 300
    R_list = [8, 12, 16, 20, 24, 32]

    print("=" * 100)
    print("Day 14d — Fractional bits + rate-adaptive clip (c0 * sqrt(b) * sigma_pc)")
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
        res = _d14c.run_cell(T, d, r, D_list, "shared", R_list,
                             n_trials, B=B, c=2.0)
        print_cell(f"T={T} r={r} {'+'.join(s[:4] for s in specs)} shared c0=2",
                   res, R_list, r)


if __name__ == "__main__":
    main()

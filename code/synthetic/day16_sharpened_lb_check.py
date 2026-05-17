"""Day 16 — Numerical sanity check for the sharpened LB conjecture.

Conjecture: in the shared-V regime with floor > 0 at |A| active
tasks, the *true* RD function scales as 2^{-2R/(r+|A|-1)}, matching
the linear-encoder upper bound. Equivalently, the LB should sharpen
from -2R/r (current Thm 8) to -2R/(r+|A|-1).

Test: if this is right, then *any* encoder (not just linear) at rate
R should have excess >= c * 2^{-2R/(r+|A|-1)}. We can't test arbitrary
encoders, but we CAN test a near-optimal nonlinear one: nearest
neighbor in a large random codebook. If slope of excess matches
-2R/(r+|A|-1) (not -2R/r), sharpened LB is plausible. If slope
matches -2R/r, random coding beats linear encoders and the LB is
actually -2R/r (linear-encoder best is not tight).

Protocol: For each (T, r, D_list) cell, we do 3 experiments:
  (1) Linear encoder (null-space split, from day15).
  (2) Random codebook: 2^R iid codewords in R^{d_eff},
      decoder returns nearest by max-over-t D_t metric. At
      modest R this is tractable; limits us to R <= ~18.
  (3) "Oracle" lattice: quantize (w - w_cheb) on a fine enough
      lattice that error is negligible; estimate achievable rate
      by counting distinct codewords used. This is not rigorous
      but gives an empirical upper envelope.

We focus on (1) and (2) only; (3) is sketchy.
"""

import math
import numpy as np
from day10_general_ht import (
    compute_Hbar_tauH_general,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day12_achievability import ht_weighted_max
from day15_cheb_general_T import cheb_center_general_T_socp, merge_general_T

RNG = np.random.default_rng(20260512)


def random_codebook_encoder(taus, Hts, R, Q, c_range, rng, n_codewords=None,
                             shift_cheb=False):
    """Random-codebook encoder: 2^R iid Gaussian codewords in R^{d_eff}
    at std c_range. Encoder picks the one with lowest max_t D_t(w);
    decoder looks up the index. Returns w_star in R^d.

    WARNING: shift_cheb=True gives the encoder data-dependent
    side info (w_cheb is a function of tau), which is NOT a valid
    rate-R code. Only use for upper-envelope comparisons. Default
    is False, i.e., codebook is centered at origin and must cover
    the full tau-range.
    """
    d_eff = Q.shape[1]
    N = int(n_codewords or 2 ** min(R, 18))
    codebook = rng.standard_normal((N, d_eff)) * (c_range / math.sqrt(d_eff))
    if shift_cheb:
        w_cheb = cheb_center_general_T_socp(taus, Hts, Q)
        w_cheb_r = Q.T @ w_cheb
        codebook_r = codebook + w_cheb_r[None, :]
    else:
        codebook_r = codebook
    codewords_d = codebook_r @ Q.T  # N x d (lift into original space)

    # Evaluate max D_t for each codeword (vectorized).
    T = len(taus)
    # codewords_d: N x d. For each task t, compute D_t(w) for all codewords.
    all_Dt = np.zeros((T, N))
    for t in range(T):
        diff = codewords_d - taus[t][None, :]     # N x d
        Hdiff = diff @ Hts[t]                      # N x d
        all_Dt[t] = np.einsum('ij,ij->i', diff, Hdiff)  # N
    max_ds = all_Dt.max(axis=0)                    # N
    best_idx = int(np.argmin(max_ds))
    return codewords_d[best_idx]


def run_random_codebook(T, d, r, D_list, R_list, n_trials,
                         B=1.0, shift_cheb=False, c_range_factor=1.0):
    """Run random codebook encoder at each R in R_list.
    R_list must stay <= 18 (2^18 = 262144 codewords).

    If shift_cheb=False (default, valid rate-R code): codebook is
    at origin, must cover B-ball in R^{d_eff}. c_range = c_range_factor * B.

    If shift_cheb=True (invalid rate-R code; free side info): codebook
    centered at w_cheb(tau), c_range scales with cheb.
    """
    results = {R: {"max": [], "cheb_sq": []} for R in R_list}
    for trial in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, "shared", RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        try:
            w_cheb = cheb_center_general_T_socp(taus, Hts, Q)
        except Exception:
            continue
        cheb_sq = ht_weighted_max(w_cheb, taus, Hts)
        if shift_cheb:
            c_range = 3.0 * math.sqrt(max(cheb_sq, 1e-8))
        else:
            c_range = c_range_factor * B
        for R in R_list:
            w_star = random_codebook_encoder(
                taus, Hts, R, Q, c_range, RNG, shift_cheb=shift_cheb)
            results[R]["max"].append(ht_weighted_max(w_star, taus, Hts))
            results[R]["cheb_sq"].append(cheb_sq)
    return results


def fit_slope(rows, key):
    vals = [row[key] for row in rows if row[key] > 1e-12]
    bs = [row["b"] for row in rows if row[key] > 1e-12]
    if len(vals) < 2:
        return float("nan")
    return float(np.polyfit(bs, np.log2(vals), 1)[0])


def main():
    B = 1.0
    n_trials = 30   # expensive: 2^18 codewords per trial per R
    R_list = [6, 8, 10, 12, 14, 16, 18]  # small rates so codebook tractable
    # r = 3 (small) so the regime is saturating within this R range.

    cases = [
        (2, 64, 3, ["iso", "iso"]),          # theory-sharp -1.5, Thm8 -2.0
        (3, 64, 3, ["iso", "iso", "iso"]),   # theory-sharp -1.2, Thm8 -2.0
    ]
    # Two variants to separate valid-code from side-info versions.
    variants = [
        dict(name="valid (no shift)", shift_cheb=False, c_range_factor=1.0),
        dict(name="valid (no shift, wider)", shift_cheb=False, c_range_factor=2.0),
        dict(name="INVALID (cheb side info)", shift_cheb=True,
             c_range_factor=None),
    ]
    for T, d, r, specs in cases:
        D_list = build_D_list(specs, r)
        label = f"T={T} r={r} {'+'.join(s[:4] for s in specs)}"
        print(f"\n=== {label} ===", flush=True)
        theory_sharp = -2.0 * r / (r + T - 1)
        theory_thm8 = -2.0
        print(f"  theory-linear-UB={theory_sharp:.3f}  "
              f"theory-Thm8-LB={theory_thm8:.3f}", flush=True)
        for variant in variants:
            print(f"\n  [{variant['name']}]", flush=True)
            kwargs = {"shift_cheb": variant["shift_cheb"]}
            if variant.get("c_range_factor") is not None:
                kwargs["c_range_factor"] = variant["c_range_factor"]
            res = run_random_codebook(T, d, r, D_list, R_list, n_trials,
                                       B=B, **kwargs)
            rows = []
            print(f"  {'R':>4} {'b=R/r':>7} {'max_emp':>10} "
                  f"{'cheb_sq':>10} {'exc':>10}", flush=True)
            for R in R_list:
                max_e = float(np.mean(res[R]["max"]))
                cs = float(np.mean(res[R]["cheb_sq"]))
                exc = max_e - cs
                rows.append({"b": R/r, "exc": exc})
                print(f"  {R:>4} {R/r:>7.2f} {max_e:>10.5f} "
                      f"{cs:>10.5f} {exc:>10.5f}", flush=True)
            slope = fit_slope(rows, "exc")
            print(f"  >> slope={slope:.3f}", flush=True)


if __name__ == "__main__":
    main()

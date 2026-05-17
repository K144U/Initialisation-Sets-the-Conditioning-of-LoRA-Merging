"""Day 12 — Phase 2 Achievability via Hadamard-incoherence quantization.

An explicit encoder-decoder pair that achieves Theorem 8's lower
bound up to constants:

    algorithm_distortion(R) <= floor + C * (B^2 d_eff/(Tr)) * 2^{-2R/d_eff}

for a universal constant C (empirically ~= 3 with c=5 clipping).
Combined with Theorem 8's LB

    D_star(R) >= floor + c_TQ * (B^2 d_eff/(Tr)) * 2^{-2R/d_eff},

this pins down the RD function up to a Theta-factor.

Algorithm:
    Encoder (shared random seed s for the Hadamard diagonal signs):
        1. Compute Hbar, tauH, eta = Hbar^{1/2} tauH in the Q-basis
           of range(Hbar). eta has d_eff coordinates.
        2. Pad eta to length 2^{ceil(log2 d_eff)} with zeros.
        3. Apply a randomized Walsh-Hadamard transform:
               eta_hat = (1/sqrt(n)) H diag(+/-1)^s eta_padded.
           This equidistributes energy across the n coords of eta_hat.
        4. Scalar-quantize each coord of eta_hat uniformly in
           [-c sigma_pc, c sigma_pc] with b = R/n bits per coord,
           sigma_pc = B/sqrt(Tr) the theorem-predicted per-coord std.
        5. Transmit the quantization indices. Total rate R = n*b bits.

    Decoder (same seed s):
        1. Unquantize to get eta_hat_q.
        2. Invert Hadamard + diagonal signs to get eta_q_padded;
           truncate to first d_eff coords to get eta_q.
        3. Return w^star = Hbar^{+ 1/2} eta_q
           = Q diag(1/sqrt(w_pos)) eta_q  (on range(Hbar)).

Tests:
    (A) Empirical distortion vs rate: measure
            avg_t (w^star - tau_t)^T H_t (w^star - tau_t)  (soft metric)
            max_t ...  (hard metric, what the theorem is about)
        across rate budgets R in {d_eff*b : b in [2..12]}.
    (B) Constant-factor gap: compare empirical-excess / LB-excess.
        Expected: a constant O(1) across rates and cells.
    (C) Slope check: log2(empirical - floor) vs R/d_eff should be ~-2.
"""

import math
import numpy as np
from day10_general_ht import (
    make_D0, sample_stiefel, compute_Hbar_tauH_general,
    ht_weighted_floor_general, hbar_metric_sq,
    uniform_scalar_quantize,
)
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list

RNG = np.random.default_rng(20260501)


# --- Hadamard helpers --------------------------------------------

def hadamard_matrix(n):
    """Walsh-Hadamard matrix of size n (must be power of 2).
    Returns orthogonal matrix H with H H^T = n I (use 1/sqrt(n) to normalize).
    """
    assert n & (n - 1) == 0 and n > 0, f"n must be a power of 2: {n}"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


# --- Encoder / decoder --------------------------------------------

def encode_merge(tauH, Q, w_pos, b, B, T, r, rng, c=5.0):
    """Hadamard-incoherence quantization of eta = Hbar^{1/2} tauH.

    b: bits per coord (so total rate R = b * n, where n = next_pow2(d_eff)).
    Returns (indices, n, signs) for the decoder.
    """
    d_eff = Q.shape[1]
    n = next_pow2(d_eff)
    sqrt_w = np.sqrt(w_pos)
    eta = sqrt_w * (Q.T @ tauH)  # d_eff-vector
    eta_padded = np.zeros(n)
    eta_padded[:d_eff] = eta

    signs = rng.choice([-1.0, 1.0], size=n)
    Hmat = hadamard_matrix(n)
    eta_hat = (Hmat @ (signs * eta_padded)) / math.sqrt(n)

    sigma_pc = B / math.sqrt(T * r)
    rng_hi = c * sigma_pc
    step = (2 * rng_hi) / (2 ** b)
    # Save the floor indices (so decoder can reconstruct).
    idx = np.floor((eta_hat + rng_hi) / step).astype(np.int64)
    idx = np.clip(idx, 0, (2 ** b) - 1)
    return idx, n, signs, step, rng_hi


def decode_merge(idx, n, signs, step, rng_hi, Q, w_pos):
    """Invert the Hadamard transform and embed back to w^star in R^d."""
    d_eff = Q.shape[1]
    sqrt_w = np.sqrt(w_pos)
    eta_hat_q = -rng_hi + (idx + 0.5) * step

    Hmat = hadamard_matrix(n)
    # Invert: eta_padded = (1/sqrt(n)) signs * (H eta_hat_q), since H H = n I.
    eta_q_padded = signs * (Hmat @ eta_hat_q) / math.sqrt(n)
    eta_q = eta_q_padded[:d_eff]

    # Decode: w^star = Q diag(1/sqrt(w_pos)) eta_q.
    w_coeffs = eta_q / sqrt_w
    return Q @ w_coeffs


def merge_ht(taus, Hts, b, B, T, r, rng, c=5.0):
    """Full encode-decode pipeline. Returns w^star."""
    Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
    idx, n, signs, step, rng_hi = encode_merge(
        tauH, Q, w_pos, b, B, T, r, rng, c=c)
    w_star = decode_merge(idx, n, signs, step, rng_hi, Q, w_pos)
    return w_star, Hbar, tauH, d_eff


# --- Distortion measurement ---------------------------------------

def ht_weighted_max(w, taus, Hts):
    """max_t (w - tau_t)^T H_t (w - tau_t)."""
    vals = []
    for t in range(len(taus)):
        diff = w - taus[t]
        vals.append(float(diff @ (Hts[t] @ diff)))
    return max(vals)


def ht_weighted_avg(w, taus, Hts):
    """avg_t (w - tau_t)^T H_t (w - tau_t)."""
    T = len(taus)
    acc = 0.0
    for t in range(T):
        diff = w - taus[t]
        acc += float(diff @ (Hts[t] @ diff))
    return acc / T


# --- Tests --------------------------------------------------------

def run_achievability_cell(T, d, r, D_list, overlap, bits_list,
                            n_trials, B=1.0, c=5.0):
    """For each b, measure empirical avg_t and max_t distortion over
    n_trials tuples. Return table keyed by b.
    """
    results = {b: {"avg": [], "max": [], "floor": []}
               for b in bits_list}
    d_effs = []
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, overlap, RNG)
        _, tauH_only, _, d_eff_only, _ = compute_Hbar_tauH_general(taus, Hts)
        floor = ht_weighted_floor_general(taus, Hts, tauH_only)
        for b in bits_list:
            w_star, Hbar, tauH, d_eff = merge_ht(
                taus, Hts, b, B, T, r, RNG, c=c)
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
        R = b * next_pow2(int(round(d_eff_mean)))
        # LB floor + compression term (Theorem 8, using d_eff).
        floor_thy = (B ** 2) * (1 - d_eff_mean / (T * r))
        compress_thy = (B ** 2) * (d_eff_mean / (T * r)) * 2 ** (-2 * R / d_eff_mean)
        excess_avg = avg_emp - floor_emp
        excess_max = max_emp - floor_emp
        rows.append({
            "b": b, "R": R, "d_eff": d_eff_mean,
            "floor_emp": floor_emp, "floor_thy": floor_thy,
            "avg_emp": avg_emp, "max_emp": max_emp,
            "excess_avg": excess_avg, "excess_max": excess_max,
            "compress_thy_cTQ1": compress_thy,  # i.e., LB with c_TQ=1
        })
    return rows


def fit_slope(bits_list, vals):
    log_vals = np.log2(np.array(vals))
    b_arr = np.array(bits_list, dtype=float)
    slope, _ = np.polyfit(b_arr, log_vals, 1)
    return float(slope)


def main():
    B = 1.0
    n_trials = 200
    bits_list = [2, 4, 6, 8, 10]

    print("=" * 100)
    print("Day 12 achievability: Hadamard-incoherence quantization")
    print("=" * 100)

    cases = [
        # (T, d, r, D_list_specs, overlap)
        (2, 128, 4, ["iso", "iso"],              "random"),
        (2, 128, 4, ["iso", "geom"],             "random"),
        (3, 128, 4, ["iso", "geom", "twoscale"], "random"),
        (2, 128, 4, ["iso", "iso"],              "shared"),
        (2, 128, 4, ["geom", "twoscale"],        "shared"),
        (4, 128, 4, ["iso", "geom", "lin", "twoscale"], "shared"),
    ]

    for T, d, r, specs, overlap in cases:
        D_list = build_D_list(specs, r)
        rows = run_achievability_cell(
            T, d, r, D_list, overlap, bits_list, n_trials, B=B)
        d_eff = rows[0]["d_eff"]
        floor_thy = rows[0]["floor_thy"]
        label = f"T={T} r={r} {'+'.join(s[:4] for s in specs)} {overlap}"
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
        # Slope on excess_max vs b.
        vals = [row["excess_max"] for row in rows]
        slope = fit_slope(bits_list, vals)
        print(f"  slope(log2 excess_max vs b) = {slope:.3f}  "
              f"(theory: -2 * n/d_eff * 1)")


if __name__ == "__main__":
    main()

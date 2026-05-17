"""Day 10 numerical verification: extend Theorem 7 to general H_t >= 0.

Theorem 7 is proved for H_t = P_{V_t}. Here we verify that the same
formula
    floor = B^2 (1 - E[d_eff] / (T r)),
    E[||eta||^2] = B^2 d_eff / (T r),
    conditional law of eta = Hbar^{1/2} tauH in W is O(W)-invariant,
holds under the generalized hard distribution

    U_t in Stiefel(d, r) iid,
    H_t = U_t D0 U_t^T  (D0 fixed PSD r x r, not necessarily I_r),
    v_t = B D0^{-1/2} u_t,  u_t ~ Unif(S^{r-1}) iid,
    tau_t = U_t v_t.

The scale normalization tau_t^T H_t tau_t = v_t^T D0 v_t = B^2 holds
by construction. For D0 = I_r this reduces to Day 9's setup.

Tests:
    (A) Floor formula under several D0 choices.
    (B) Trace of conditional eta-covariance matches B^2 d_eff / (T r).
    (C) MP-aware rotational-invariance check (as in day9_verify_lemma6).
    (D) Slope of log(excess_hbar) vs bits is ~ -2 under adaptive
        quantization, same as Day 9.
"""

import numpy as np
from day8_rank_r_sanity import (
    sample_stiefel, compute_Hbar_and_tauH,
    uniform_scalar_quantize,
)

RNG = np.random.default_rng(20260429)


# --- Samplers for general H_t ------------------------------------

def make_D0(r, spec):
    """Fixed r x r diagonal PSD matrix (eigenvalues only)."""
    if spec == "iso":
        return np.ones(r)
    if spec == "geom":
        return 2.0 ** np.linspace(1.0, -1.0, r)  # [2, ..., 0.5]
    if spec == "lin":
        return np.linspace(1.0, 3.0, r)
    if spec == "twoscale":
        d0 = np.ones(r)
        d0[: r // 2] = 4.0
        return d0
    raise ValueError(spec)


def sample_general_ht_tuple(T, d, r, B, D0_diag, overlap, rng):
    """
    Returns (taus, Us, Hts) with:
        Us[t] in Stiefel(d, r)
        Hts[t] = Us[t] diag(D0) Us[t]^T
        tau_t = Us[t] v_t, v_t = B D0^{-1/2} u_t, u_t ~ Unif(S^{r-1})
    so tau_t^T H_t tau_t = B^2 exactly.
    """
    if overlap == "random":
        Us = [sample_stiefel(d, r, rng) for _ in range(T)]
    elif overlap == "shared":
        U0 = sample_stiefel(d, r, rng)
        Us = [U0 for _ in range(T)]
    else:
        raise ValueError(overlap)

    D0_inv_sqrt = 1.0 / np.sqrt(D0_diag)
    taus = []
    Hts = []
    for t in range(T):
        u = rng.standard_normal(r)
        u /= np.linalg.norm(u)
        v = B * D0_inv_sqrt * u
        taus.append(Us[t] @ v)
        Hts.append(Us[t] @ np.diag(D0_diag) @ Us[t].T)
    return np.array(taus), Us, Hts


def compute_Hbar_tauH_general(taus, Hts):
    """General-H_t analog of compute_Hbar_and_tauH.

    Hbar = (1/T) sum H_t
    Htau_bar = (1/T) sum H_t tau_t
    tauH = Hbar^+ Htau_bar
    Returns (Hbar, tauH, Q, d_eff) with Q orthonormal basis of range(Hbar).
    """
    T = taus.shape[0]
    d = taus.shape[1]
    Hbar = np.zeros((d, d))
    Htau_bar = np.zeros(d)
    for t in range(T):
        Hbar += Hts[t]
        Htau_bar += Hts[t] @ taus[t]
    Hbar /= T
    Htau_bar /= T

    w, V = np.linalg.eigh(Hbar)
    tol = 1e-10 * max(w.max(), 1.0)
    mask = w > tol
    d_eff = int(mask.sum())
    Q = V[:, mask]
    w_pos = w[mask]
    tauH = Q @ ((Q.T @ Htau_bar) / w_pos)
    return Hbar, tauH, Q, d_eff, w_pos


def ht_weighted_floor_general(taus, Hts, tauH):
    """(1/T) sum_t (tau_t - tauH)^T H_t (tau_t - tauH)."""
    T = taus.shape[0]
    acc = 0.0
    for t in range(T):
        diff = taus[t] - tauH
        acc += float(diff @ (Hts[t] @ diff))
    return acc / T


def hbar_metric_sq(Hbar, v):
    return float(v @ (Hbar @ v))


# --- Test A: floor formula ---------------------------------------

def test_floor_formula(T, d, r, B, D0_spec, overlap, n_trials):
    D0 = make_D0(r, D0_spec)
    floors = []
    d_effs = []
    eta_sq = []
    for _ in range(n_trials):
        taus, Us, Hts = sample_general_ht_tuple(
            T, d, r, B, D0, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floors.append(ht_weighted_floor_general(taus, Hts, tauH))
        d_effs.append(d_eff)
        # ||eta||^2 = tauH^T Hbar tauH
        eta_sq.append(hbar_metric_sq(Hbar, tauH))
    return (float(np.mean(floors)), float(np.mean(d_effs)),
            float(np.mean(eta_sq)))


# --- Test C: rotational invariance (MP-aware) --------------------

def test_rotational_invariance(T, d, r, B, D0_spec, n_trials):
    """Fix W = random d_eff-dim subspace of R^d. Sample tuples with
    V_t uniformly in W; compute eta in W-basis; measure covariance
    eigenvalue spread vs Marchenko-Pastur upper bound.
    """
    D0 = make_D0(r, D0_spec)
    d_eff_target = T * r
    assert d_eff_target <= d

    W = np.linalg.qr(RNG.standard_normal((d, d_eff_target)))[0][:, :d_eff_target]

    etas_W = np.zeros((n_trials, d_eff_target))
    D0_inv_sqrt = 1.0 / np.sqrt(D0)
    for i in range(n_trials):
        # Sample V_t's inside W: pick random r-dim subspaces of W.
        Us_in_W = [np.linalg.qr(RNG.standard_normal((d_eff_target, r)))[0][:, :r]
                   for _ in range(T)]
        Us = [W @ U_inW for U_inW in Us_in_W]
        taus = []
        Hts = []
        for t in range(T):
            u = RNG.standard_normal(r); u /= np.linalg.norm(u)
            v = B * D0_inv_sqrt * u
            taus.append(Us[t] @ v)
            Hts.append(Us[t] @ np.diag(D0) @ Us[t].T)
        taus = np.array(taus)

        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        # eta = Hbar^{1/2} tauH, expressed in W-basis.
        eta_d = Q @ (np.sqrt(w_pos) * (Q.T @ tauH))
        etas_W[i] = W.T @ eta_d

    M = (etas_W.T @ etas_W) / n_trials
    eig = np.linalg.eigvalsh(M)
    spread = eig.max() / max(eig.min(), 1e-12)
    c = d_eff_target / n_trials
    sqc = np.sqrt(c)
    mp_spread = ((1 + sqc) / (1 - sqc)) ** 2

    return {
        "spread_obs": float(spread),
        "mp_UB": float(mp_spread),
        "trace_emp": float(M.trace()),
        "trace_thy": (B ** 2) * d_eff_target / (T * r),
    }


# --- Test D: slope of excess_hbar vs bits ------------------------

def quantize_tauH_adaptive_general(tauH, Q, b, B, T, r, c=3.0):
    """Per-coord range +- c sigma_pc with sigma_pc = B/sqrt(Tr).
    Quantizes Q^T tauH. WARNING: per-coord std of Q^T tauH depends on
    the eigenvalues of Hbar; for non-uniform spectra this miscalibrates.
    Use quantize_eta_adaptive instead for general H_t.
    """
    sigma_pc = B / np.sqrt(T * r)
    rng_hi = c * sigma_pc
    coeffs = Q.T @ tauH
    coeffs_q = uniform_scalar_quantize(coeffs, b, -rng_hi, rng_hi)
    return Q @ coeffs_q


def quantize_eta_adaptive(tauH, Q, w_pos, b, B, T, r, c=3.0):
    """Quantize eta = Hbar^{1/2} tauH in the Q-basis of range(Hbar).

    Theorem 7/Lemma 6 give E[||eta||^2] = B^2 d_eff/(Tr), with eta
    conditionally O(W)-isotropic, so per-coord std in any basis of W
    is sigma_pc = B/sqrt(Tr) uniformly. Returns w such that
        (w - tauH)^T Hbar (w - tauH) = ||eta_q - eta||^2
    where eta_q is the quantized eta. Concretely, decode by
        w = Hbar^{+1/2} eta_q  = Q diag(1/sqrt(w_pos)) Q^T eta_q.
    """
    sigma_pc = B / np.sqrt(T * r)
    rng_hi = c * sigma_pc
    sqrt_w = np.sqrt(w_pos)
    eta_coeffs = sqrt_w * (Q.T @ tauH)  # in Q-basis of W
    eta_q_coeffs = uniform_scalar_quantize(
        eta_coeffs, b, -rng_hi, rng_hi)
    # w = Hbar^{+1/2} eta_q = Q diag(1/sqrt(w_pos)) eta_q_coeffs.
    w_coeffs = eta_q_coeffs / sqrt_w
    return Q @ w_coeffs


def test_slope(T, d, r, B, D0_spec, overlap, bits_list, n_trials, c=5.0):
    """Measure E[(w - tauH)^T Hbar (w - tauH)] vs b; fit slope on log2.

    Uses eta = Hbar^{1/2} tauH quantization (the right metric-aware
    quantizer for general H_t). c controls the clipping range in units
    of per-coord std sigma_pc = B/sqrt(Tr).
    """
    D0 = make_D0(r, D0_spec)
    excess_by_b = {b: [] for b in bits_list}
    d_effs = []
    for _ in range(n_trials):
        taus, Us, Hts = sample_general_ht_tuple(
            T, d, r, B, D0, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        d_effs.append(d_eff)
        for b in bits_list:
            w = quantize_eta_adaptive(tauH, Q, w_pos, b, B, T, r, c=c)
            diff = w - tauH
            excess_by_b[b].append(hbar_metric_sq(Hbar, diff))
    log_vals = np.array([np.log2(np.mean(excess_by_b[b]))
                         for b in bits_list])
    b_arr = np.array(bits_list, dtype=float)
    slope, intercept = np.polyfit(b_arr, log_vals, 1)
    return {
        "slope": float(slope),
        "d_eff_mean": float(np.mean(d_effs)),
        "log_vals": log_vals.tolist(),
    }


def main():
    B = 1.0

    print("=" * 72)
    print("Day 10 verification: general H_t = U_t D0 U_t^T")
    print("=" * 72)

    # --- Test A + B: floor and trace -----------------------------
    print("\n=== Test A/B: floor formula under several D0 specs ===")
    print("Theory: floor = B^2 (1 - d_eff / (Tr));  ||eta||^2 = B^2 d_eff/(Tr)")
    print(f"{'T':>3} {'d':>4} {'r':>3} {'D0':>8} {'overlap':>8} "
          f"{'d_eff':>6} {'floor_emp':>10} {'floor_thy':>10} "
          f"{'eta2_emp':>10} {'eta2_thy':>10}")
    n_trials = 2000
    cases = [
        # (T, d, r, D0_spec, overlap)
        (2, 64, 4, "iso",      "random"),
        (2, 64, 4, "geom",     "random"),
        (2, 64, 4, "twoscale", "random"),
        (2, 64, 4, "lin",      "random"),
        (3, 64, 4, "geom",     "random"),
        (4, 128, 8, "twoscale","random"),
        (2, 64, 4, "geom",     "shared"),
        (4, 64, 4, "geom",     "shared"),
    ]
    for T, d, r, spec, overlap in cases:
        floor_emp, d_eff_emp, eta2_emp = test_floor_formula(
            T, d, r, B, spec, overlap, n_trials)
        floor_thy = (B ** 2) * (1 - d_eff_emp / (T * r))
        eta2_thy = (B ** 2) * d_eff_emp / (T * r)
        print(f"{T:>3} {d:>4} {r:>3} {spec:>8} {overlap:>8} "
              f"{d_eff_emp:>6.2f} {floor_emp:>10.5f} {floor_thy:>10.5f} "
              f"{eta2_emp:>10.5f} {eta2_thy:>10.5f}")

    # --- Test C: rotational invariance ---------------------------
    print("\n=== Test C: rotational invariance (MP-aware) ===")
    print("Under O(W)-invariance of eta|W, cov spread <= MP_UB at sample size n.")
    print(f"{'T':>3} {'d':>4} {'r':>3} {'D0':>8} {'n':>7} "
          f"{'spread':>8} {'MP_UB':>8} {'tr_emp':>8} {'tr_thy':>8}")
    for T, d, r, spec, n in [
        (2, 32, 4, "iso",      30000),
        (2, 32, 4, "geom",     30000),
        (2, 32, 4, "twoscale", 30000),
        (3, 24, 4, "geom",     30000),
        (2, 32, 4, "geom",    100000),
    ]:
        out = test_rotational_invariance(T, d, r, B, spec, n)
        print(f"{T:>3} {d:>4} {r:>3} {spec:>8} {n:>7} "
              f"{out['spread_obs']:>8.4f} {out['mp_UB']:>8.4f} "
              f"{out['trace_emp']:>8.5f} {out['trace_thy']:>8.5f}")

    # --- Test D: slope ------------------------------------------
    print("\n=== Test D: slope of log2(excess_hbar) vs bits ===")
    print("Theory: slope = -2 per bit per eff-dim; d_eff_fit = -2/slope.")
    print(f"{'T':>3} {'d':>4} {'r':>3} {'D0':>8} {'overlap':>8} "
          f"{'slope':>8} {'d_eff':>7}")
    bits_list = [4, 6, 8, 10, 12]
    for T, d, r, spec, overlap in [
        (2, 128, 4, "iso",      "shared"),
        (2, 128, 4, "geom",     "shared"),
        (2, 128, 4, "twoscale", "shared"),
        (4, 128, 4, "geom",     "shared"),
        (2, 128, 4, "iso",      "random"),
        (2, 128, 4, "geom",     "random"),
    ]:
        out = test_slope(T, d, r, B, spec, overlap, bits_list, n_trials=400)
        print(f"{T:>3} {d:>4} {r:>3} {spec:>8} {overlap:>8} "
              f"{out['slope']:>8.3f} {out['d_eff_mean']:>7.2f}")


if __name__ == "__main__":
    main()

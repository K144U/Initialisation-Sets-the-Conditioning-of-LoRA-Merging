"""Direct verification of Lemma 6 (closed-form floor) and the
rotational-invariance claim from Theorem 7's Step 5.

Lemma 6:
    E_P[(1/T) sum_t (tau_t - tauH)^T P_{V_t} (tau_t - tauH)]
      = B^2 (1 - E[d_eff] / (Tr))

Step 5 claim (for the bug flagged in theorem_v1.tex):
    Conditional on the span W = range(Hbar), the law of
    eta = Hbar^{1/2} tauH inside W is O(W)-rotationally invariant.

Both checked at four (T, r, d) points.
"""

import numpy as np
from day8_rank_r_sanity import (
    sample_rank_r_tuple, compute_Hbar_and_tauH, Ht_weighted_floor,
)

RNG = np.random.default_rng(20260428)


def empirical_floor_and_deff(T, d, r, B, overlap, n_trials):
    floors, d_effs = [], []
    for _ in range(n_trials):
        taus, Us = sample_rank_r_tuple(T, d, r, B, overlap, RNG)
        _, tauH, _, d_eff = compute_Hbar_and_tauH(taus, Us)
        floors.append(Ht_weighted_floor(taus, Us, tauH))
        d_effs.append(d_eff)
    return float(np.mean(floors)), float(np.mean(d_effs))


def lemma6_theory(T, r, d_eff, B):
    return (B ** 2) * (1.0 - d_eff / (T * r))


def check_rotational_invariance(T, d, r, B, n_trials):
    """Fix W = span(V_1, ..., V_T) and test whether eta = Hbar^{1/2} tauH
    is O(W)-rotationally invariant inside W.

    Test (Marchenko-Pastur-aware): let M = E[eta_W eta_W^T] in W-basis.
    Under O(W)-invariance, M must be scalar: M = (||eta||^2/d_eff) I.
    Empirical M_n with n iid samples in dim k has MP eigenvalue spread
    bounded by (1+sqrt(k/n))^2 / (1-sqrt(k/n))^2 + finite-n noise. So
    we report the observed spread vs. the MP upper limit, plus a
    direction-by-direction check: var(<eta, e_i>) should be constant
    across i, with relative spread O(sqrt(2/n)) under H_0.
    """
    d_eff_target = T * r
    assert d_eff_target <= d
    W_basis = np.linalg.qr(RNG.standard_normal((d, d_eff_target)))[0][:, :d_eff_target]

    etas_W = np.zeros((n_trials, d_eff_target))
    for i in range(n_trials):
        Us_in_W = [np.linalg.qr(RNG.standard_normal((d_eff_target, r)))[0][:, :r]
                   for _ in range(T)]
        Us = [W_basis @ U_inW for U_inW in Us_in_W]
        taus = []
        for t in range(T):
            v = RNG.standard_normal(r); v /= np.linalg.norm(v); v *= B
            taus.append(Us[t] @ v)
        taus = np.array(taus)

        Hbar = sum(Us[t] @ Us[t].T for t in range(T)) / T
        w, V = np.linalg.eigh(Hbar)
        tol = 1e-10 * max(w.max(), 1.0)
        mask = w > tol
        Q = V[:, mask]
        w_pos = w[mask]

        bar_tau = taus.mean(axis=0)
        tauH = Q @ ((Q.T @ bar_tau) / w_pos)
        # eta = Hbar^{1/2} tauH, expressed in the W-basis.
        eta_d = Q @ (np.sqrt(w_pos) * (Q.T @ tauH))
        etas_W[i] = W_basis.T @ eta_d

    M = (etas_W.T @ etas_W) / n_trials
    eig = np.linalg.eigvalsh(M)
    spread = eig.max() / max(eig.min(), 1e-12)

    # MP upper-bound spread for isotropic source.
    c = d_eff_target / n_trials
    sqc = np.sqrt(c)
    mp_spread = ((1 + sqc) / (1 - sqc)) ** 2

    # Per-direction variance: should all equal trace(M)/d_eff if O(W)-inv.
    diag_var = np.diag(M)
    diag_spread = diag_var.max() / max(diag_var.min(), 1e-12)
    diag_expected_spread = 1 + 4 * np.sqrt(2 / n_trials)  # ~ +- 2 std on var

    # Trace and theoretical: E[||eta||^2] = B^2 d_eff / (Tr).
    trace_emp = float(M.trace())
    trace_thy = (B ** 2) * d_eff_target / (T * r)

    return {
        "spread_obs": float(spread),
        "spread_MP_isotropic_UB": float(mp_spread),
        "diag_spread_obs": float(diag_spread),
        "diag_spread_isotropic_band": float(diag_expected_spread),
        "trace_emp": trace_emp,
        "trace_thy": trace_thy,
    }


def main():
    B = 1.0
    n_trials = 2000
    cases = [
        # (T, d, r, overlap)
        (2, 128, 4, "random"),   # Tr = 8,  expect d_eff = 8, floor ~ 0
        (2, 128, 16, "random"),  # Tr = 32, expect d_eff = 32, floor ~ 0
        (4, 128, 4, "random"),   # Tr = 16, expect d_eff = 16, floor ~ 0
        (2, 128, 8, "shared"),   # d_eff = r = 8, floor = 1/2 exactly
        (4, 128, 8, "shared"),   # d_eff = r = 8, floor = 3/4 exactly
        (2, 128, 128, "random"), # r=d, d_eff = 128, floor = B^2(1-1/2) = 1/2
    ]
    print("=== Lemma 6 verification (floor vs closed form) ===")
    print(f"{'T':>3} {'d':>5} {'r':>4} {'overlap':>8} "
          f"{'d_eff_emp':>10} {'floor_emp':>10} "
          f"{'floor_thy':>10} {'rel_err':>10}")
    for T, d, r, overlap in cases:
        floor_emp, d_eff_emp = empirical_floor_and_deff(
            T, d, r, B, overlap, n_trials)
        floor_thy = lemma6_theory(T, r, d_eff_emp, B)
        rel = abs(floor_emp - floor_thy) / max(abs(floor_thy), 1e-6)
        print(f"{T:>3} {d:>5} {r:>4} {overlap:>8} "
              f"{d_eff_emp:>10.3f} {floor_emp:>10.5f} "
              f"{floor_thy:>10.5f} {rel:>10.2e}")

    print("\n=== Step-5 rotational-invariance check (MP-aware) ===")
    print("  Under O(W)-invariance: M = E[eta_W eta_W^T] is scalar.")
    print("  Empirical spread bounded above by MP isotropic UB. If")
    print("  observed spread <= MP UB, anisotropy is consistent with")
    print("  finite-sample noise, NOT a real symmetry break.")
    print(f"{'T':>3} {'d':>5} {'r':>4} {'n':>6} "
          f"{'spread_obs':>11} {'MP_UB':>9} "
          f"{'diag_spr':>9} {'iso_band':>9} "
          f"{'tr_emp':>8} {'tr_thy':>8}")
    for T, d, r, n in [(2, 32, 4, 30000), (3, 24, 4, 30000),
                       (4, 64, 4, 30000), (2, 32, 4, 100000)]:
        out = check_rotational_invariance(T, d, r, B, n_trials=n)
        print(f"{T:>3} {d:>5} {r:>4} {n:>6} "
              f"{out['spread_obs']:>11.3f} "
              f"{out['spread_MP_isotropic_UB']:>9.3f} "
              f"{out['diag_spread_obs']:>9.3f} "
              f"{out['diag_spread_isotropic_band']:>9.3f} "
              f"{out['trace_emp']:>8.4f} {out['trace_thy']:>8.4f}")


if __name__ == "__main__":
    main()

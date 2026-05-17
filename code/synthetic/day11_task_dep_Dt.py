"""Day 11 numerical verification: task-dependent D_t.

Day 10 closed Theorem 8 under a common-D_0 hard distribution
(H_t = U_t D_0 U_t^T). Today's conjecture: the proof extends
verbatim to arbitrary deterministic per-task spectra D_t, i.e.
H_t = U_t D_t U_t^T with D_t possibly different across t.

Hard distribution P*_D (per-task):
    U_t in Stiefel(d, r) iid,
    H_t = U_t D_t U_t^T,
    v_t = B D_t^{-1/2} u_t,  u_t ~ Unif(S^{r-1}) iid,
    tau_t = U_t v_t.

By construction tau_t^T H_t tau_t = v_t^T D_t v_t = B^2 per task.

Tests (mirror Day 10):
    (A) floor formula Lemma 6' extends: floor = B^2 (1 - d_eff/(Tr))
    (B) trace identity: E[||eta||^2] = B^2 d_eff/(Tr)
    (C) MP-aware rotational invariance of eta|W
    (D) slope -2 on log2(excess_hbar) vs b
"""

import numpy as np
from day10_general_ht import (
    make_D0, sample_stiefel, compute_Hbar_tauH_general,
    ht_weighted_floor_general, hbar_metric_sq,
    quantize_eta_adaptive,
)

RNG = np.random.default_rng(20260430)


def sample_task_dep_ht_tuple(T, d, r, B, D_list, overlap, rng):
    """
    D_list: list of T diagonal vectors (each length r), one per task.
    Returns (taus, Us, Hts) with:
        Us[t] in Stiefel(d, r)
        Hts[t] = Us[t] diag(D_list[t]) Us[t]^T
        tau_t = Us[t] v_t, v_t = B D_t^{-1/2} u_t, u_t ~ Unif(S^{r-1})
    """
    assert len(D_list) == T
    if overlap == "random":
        Us = [sample_stiefel(d, r, rng) for _ in range(T)]
    elif overlap == "shared":
        U0 = sample_stiefel(d, r, rng)
        Us = [U0 for _ in range(T)]
    else:
        raise ValueError(overlap)

    taus = []
    Hts = []
    for t in range(T):
        Dt = D_list[t]
        Dt_inv_sqrt = 1.0 / np.sqrt(Dt)
        u = rng.standard_normal(r); u /= np.linalg.norm(u)
        v = B * Dt_inv_sqrt * u
        taus.append(Us[t] @ v)
        Hts.append(Us[t] @ np.diag(Dt) @ Us[t].T)
    return np.array(taus), Us, Hts


def test_floor_trace(T, d, r, B, D_list, overlap, n_trials):
    floors, d_effs, eta_sq = [], [], []
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        floors.append(ht_weighted_floor_general(taus, Hts, tauH))
        d_effs.append(d_eff)
        eta_sq.append(hbar_metric_sq(Hbar, tauH))
    return (float(np.mean(floors)), float(np.mean(d_effs)),
            float(np.mean(eta_sq)))


def test_rotational_invariance(T, d, r, B, D_list, n_trials):
    """Fix W (random d_eff-dim subspace of R^d); sample V_t uniformly
    in W; measure spread of empirical cov(eta) in W-basis vs MP UB.
    """
    d_eff_target = T * r
    assert d_eff_target <= d
    W = np.linalg.qr(RNG.standard_normal((d, d_eff_target)))[0][:, :d_eff_target]
    etas_W = np.zeros((n_trials, d_eff_target))
    for i in range(n_trials):
        Us_in_W = [np.linalg.qr(RNG.standard_normal((d_eff_target, r)))[0][:, :r]
                   for _ in range(T)]
        Us = [W @ U_inW for U_inW in Us_in_W]
        taus, Hts = [], []
        for t in range(T):
            Dt = D_list[t]
            Dt_inv_sqrt = 1.0 / np.sqrt(Dt)
            u = RNG.standard_normal(r); u /= np.linalg.norm(u)
            v = B * Dt_inv_sqrt * u
            taus.append(Us[t] @ v)
            Hts.append(Us[t] @ np.diag(Dt) @ Us[t].T)
        taus = np.array(taus)

        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
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


def test_slope(T, d, r, B, D_list, overlap, bits_list, n_trials, c=5.0):
    excess_by_b = {b: [] for b in bits_list}
    d_effs = []
    for _ in range(n_trials):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, d, r, B, D_list, overlap, RNG)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        d_effs.append(d_eff)
        for b in bits_list:
            w = quantize_eta_adaptive(tauH, Q, w_pos, b, B, T, r, c=c)
            diff = w - tauH
            excess_by_b[b].append(hbar_metric_sq(Hbar, diff))
    log_vals = np.array([np.log2(np.mean(excess_by_b[b])) for b in bits_list])
    b_arr = np.array(bits_list, dtype=float)
    slope, _ = np.polyfit(b_arr, log_vals, 1)
    return {"slope": float(slope), "d_eff_mean": float(np.mean(d_effs))}


def build_D_list(specs, r):
    return [make_D0(r, s) for s in specs]


def spec_label(specs):
    return "/".join(s[:4] for s in specs)


def main():
    B = 1.0

    print("=" * 72)
    print("Day 11 verification: task-dependent D_t (H_t = U_t D_t U_t^T)")
    print("=" * 72)

    # --- Test A/B: floor + trace under task-mismatched D_t --------
    print("\n=== Test A/B: floor + trace, D_t varies across tasks ===")
    print("Theory (D_t-free): floor = B^2(1 - d_eff/Tr); ||eta||^2 = B^2 d_eff/Tr")
    print(f"{'T':>3} {'d':>4} {'r':>3} {'spec':>25} {'overlap':>8} "
          f"{'d_eff':>6} {'floor_emp':>10} {'floor_thy':>10} "
          f"{'eta2_emp':>10} {'eta2_thy':>10}")
    n_trials = 2000
    cases = [
        # (T, d, r, [D_t specs, one per task], overlap)
        (2, 64, 4, ["iso", "geom"],              "random"),
        (2, 64, 4, ["geom", "twoscale"],         "random"),
        (3, 64, 4, ["iso", "geom", "twoscale"],  "random"),
        (3, 64, 4, ["geom", "lin", "twoscale"],  "random"),
        (4, 128, 4, ["iso", "geom", "lin", "twoscale"], "random"),
        (2, 64, 4, ["iso", "geom"],              "shared"),
        (3, 64, 4, ["iso", "geom", "twoscale"],  "shared"),
        (4, 64, 4, ["iso", "geom", "lin", "twoscale"], "shared"),
    ]
    for T, d, r, specs, overlap in cases:
        D_list = build_D_list(specs, r)
        floor_emp, d_eff_emp, eta2_emp = test_floor_trace(
            T, d, r, B, D_list, overlap, n_trials)
        floor_thy = (B ** 2) * (1 - d_eff_emp / (T * r))
        eta2_thy = (B ** 2) * d_eff_emp / (T * r)
        print(f"{T:>3} {d:>4} {r:>3} {spec_label(specs):>25} {overlap:>8} "
              f"{d_eff_emp:>6.2f} {floor_emp:>10.5f} {floor_thy:>10.5f} "
              f"{eta2_emp:>10.5f} {eta2_thy:>10.5f}")

    # --- Test C: rotational invariance ---------------------------
    print("\n=== Test C: rotational invariance (MP-aware) ===")
    print(f"{'T':>3} {'d':>4} {'r':>3} {'spec':>25} {'n':>7} "
          f"{'spread':>8} {'MP_UB':>8} {'tr_emp':>8} {'tr_thy':>8}")
    for T, d, r, specs, n in [
        (2, 32, 4, ["iso", "geom"],              30000),
        (2, 32, 4, ["geom", "twoscale"],         30000),
        (3, 24, 4, ["iso", "geom", "twoscale"],  30000),
        (2, 32, 4, ["geom", "twoscale"],        100000),
    ]:
        D_list = build_D_list(specs, r)
        out = test_rotational_invariance(T, d, r, B, D_list, n)
        print(f"{T:>3} {d:>4} {r:>3} {spec_label(specs):>25} {n:>7} "
              f"{out['spread_obs']:>8.4f} {out['mp_UB']:>8.4f} "
              f"{out['trace_emp']:>8.5f} {out['trace_thy']:>8.5f}")

    # --- Test D: slope ------------------------------------------
    print("\n=== Test D: slope of log2(excess_hbar) vs bits ===")
    print(f"{'T':>3} {'d':>4} {'r':>3} {'spec':>25} {'overlap':>8} "
          f"{'slope':>8} {'d_eff':>7}")
    bits_list = [4, 6, 8, 10, 12]
    for T, d, r, specs, overlap in [
        (2, 128, 4, ["iso", "geom"],              "shared"),
        (2, 128, 4, ["geom", "twoscale"],         "shared"),
        (3, 128, 4, ["iso", "geom", "twoscale"],  "shared"),
        (4, 128, 4, ["iso", "geom", "lin", "twoscale"], "shared"),
        (2, 128, 4, ["iso", "geom"],              "random"),
        (3, 128, 4, ["iso", "geom", "twoscale"],  "random"),
    ]:
        D_list = build_D_list(specs, r)
        out = test_slope(T, d, r, B, D_list, overlap, bits_list, n_trials=400)
        print(f"{T:>3} {d:>4} {r:>3} {spec_label(specs):>25} {overlap:>8} "
              f"{out['slope']:>8.3f} {out['d_eff_mean']:>7.2f}")


if __name__ == "__main__":
    main()

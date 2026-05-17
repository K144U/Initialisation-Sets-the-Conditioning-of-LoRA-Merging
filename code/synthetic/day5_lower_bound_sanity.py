"""Day 5 sanity checks for theory/toy_theorem_v0.tex.

Three checks:
(1) Empirical Chebyshev radius^2 of iid uniform-sphere tuples vs the
    analytical floor B^2(1-1/T) used in Lemma 1.
(2) Does the quantized-mean merge respect the theorem's lower bound
    shape B^2(1-1/T) + c * (B^2/T) * 2^{-2R/d} for some universal c > 0?
    (It must, since the LB is valid; the point is to see how tight.)
(3) Does a quantized-Chebyshev-center merge close the 2^{-R/d} gap
    between the theorem's LB and the quantized-mean merge's UB, or
    does it still exhibit a linear regime at high rate? This is the
    open question flagged at end of toy_theorem_v0.tex.

Setup
-----
Same sampling as day1_distortion_measures.py: tau_t iid
Unif(B * S^{d-1}), B = 1. Sweep T in {2,4,8}, d in {128,512,2048},
b in {1,2,3,4,6,8,12}, 30 trials per cell.

Chebyshev center computed by Frank-Wolfe / Badoiu-Clarkson iteration
(converges at rate O(1/k) on smallest-enclosing-ball).

Outputs: PNG plots + printed tables.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RNG = np.random.default_rng(20260424)
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def sample_task_vectors(T, d, B=1.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    X = rng.standard_normal((T, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return B * X


def uniform_scalar_quantize(v, b, v_min, v_max):
    if b >= 30:
        return v.copy()
    levels = 2 ** b
    step = (v_max - v_min) / levels
    idx = np.floor((v - v_min) / step).astype(np.int64)
    idx = np.clip(idx, 0, levels - 1)
    return v_min + (idx + 0.5) * step


def chebyshev_center(tau, n_iter=400):
    """Frank-Wolfe iteration for smallest-enclosing-ball center.
    Converges at rate O(1/k) with the 2/(k+3) step schedule."""
    T, d = tau.shape
    c = tau.mean(axis=0).copy()
    for k in range(n_iter):
        sq = np.sum((tau - c) ** 2, axis=1)
        t_star = int(np.argmax(sq))
        step = 2.0 / (k + 3)
        c = (1.0 - step) * c + step * tau[t_star]
    return c


def cheb_radius_sq(tau):
    c = chebyshev_center(tau)
    return float(np.max(np.sum((tau - c) ** 2, axis=1)))


def merge_quantized_mean(tau, b, B):
    m = tau.mean(axis=0)
    return uniform_scalar_quantize(m, b, -B, B)


def merge_quantized_cheb(tau, b, B):
    c = chebyshev_center(tau)
    return uniform_scalar_quantize(c, b, -B, B)


def run_sweep(T_list, d_list, b_list, B=1.0, n_trials=30):
    records = []
    for T in T_list:
        for d in d_list:
            trials = [sample_task_vectors(T, d, B=B, rng=RNG)
                      for _ in range(n_trials)]
            cheb_floor_theory = (B ** 2) * (1.0 - 1.0 / T)
            cheb_sq_emp = [cheb_radius_sq(tau) for tau in trials]
            cheb_sq_emp_med = float(np.median(cheb_sq_emp))
            for b in b_list:
                d_mean, d_cheb = [], []
                for tau in trials:
                    w_m = merge_quantized_mean(tau, b, B)
                    w_c = merge_quantized_cheb(tau, b, B)
                    d_mean.append(float(np.max(
                        np.sum((tau - w_m) ** 2, axis=1))))
                    d_cheb.append(float(np.max(
                        np.sum((tau - w_c) ** 2, axis=1))))
                records.append({
                    "T": T, "d": d, "b": b,
                    "cheb_floor_theory": cheb_floor_theory,
                    "cheb_radius_sq_emp": cheb_sq_emp_med,
                    "mean_merge_D": float(np.median(d_mean)),
                    "cheb_merge_D": float(np.median(d_cheb)),
                })
    return records


def print_cheb_table(records):
    print("\n--- Check (1): empirical Chebyshev radius^2 "
          "vs theory B^2(1-1/T) ---")
    print(f"{'T':>3} {'d':>6} {'theory':>10} {'empirical':>12} "
          f"{'emp/thy':>8}")
    seen = set()
    for r in records:
        key = (r["T"], r["d"])
        if key in seen:
            continue
        seen.add(key)
        ratio = r["cheb_radius_sq_emp"] / r["cheb_floor_theory"]
        print(f"{r['T']:>3} {r['d']:>6} {r['cheb_floor_theory']:>10.4f} "
              f"{r['cheb_radius_sq_emp']:>12.4f} {ratio:>8.4f}")


def print_merge_table(records):
    print("\n--- Checks (2) and (3): merge comparisons "
          "(max-distortion, medians) ---")
    print(f"{'T':>3} {'d':>6} {'b':>3} {'mean_D':>10} {'cheb_D':>10} "
          f"{'mean-floor':>12} {'cheb-floor':>12}")
    for r in records:
        floor = r["cheb_floor_theory"]
        print(f"{r['T']:>3} {r['d']:>6} {r['b']:>3} "
              f"{r['mean_merge_D']:>10.4g} {r['cheb_merge_D']:>10.4g} "
              f"{r['mean_merge_D'] - floor:>12.4g} "
              f"{r['cheb_merge_D'] - floor:>12.4g}")


def plot_comparison(records, T_list, d_list, savedir):
    for d in d_list:
        fig, axes = plt.subplots(1, len(T_list),
                                 figsize=(5 * len(T_list), 4.3),
                                 sharey=False)
        if len(T_list) == 1:
            axes = [axes]
        for ax, T in zip(axes, T_list):
            subset = [r for r in records
                      if r["T"] == T and r["d"] == d]
            bs = np.array([r["b"] for r in subset])
            d_mean = np.array([r["mean_merge_D"] for r in subset])
            d_cheb = np.array([r["cheb_merge_D"] for r in subset])
            floor_thy = subset[0]["cheb_floor_theory"]
            cheb_emp = subset[0]["cheb_radius_sq_emp"]
            # LB family: floor + c (B^2/T) 2^{-2b}; show c = 0.05, 0.2, 1
            for c_const, style in [(0.05, ":"), (0.2, "--"), (1.0, "-")]:
                lb = floor_thy + c_const * (1.0 / T) * 2.0 ** (-2 * bs)
                ax.plot(bs, lb, "k" + style, alpha=0.45,
                        label=f"LB (c={c_const})")
            ax.plot(bs, d_mean, "o-", color="C0", lw=1.5,
                    label="quantized-mean merge")
            ax.plot(bs, d_cheb, "s-", color="C1", lw=1.5,
                    label="quantized-Chebyshev merge")
            ax.axhline(floor_thy, color="gray", lw=0.8, ls="-",
                       alpha=0.5, label=r"theory floor $B^2(1-1/T)$")
            ax.axhline(cheb_emp, color="red", lw=0.8, ls=":",
                       alpha=0.5, label=r"empirical $R_c^2$")
            ax.set_xlabel("b (bits per coord)")
            if T == T_list[0]:
                ax.set_ylabel("max-distortion (median)")
            ax.set_yscale("log")
            ax.set_title(f"T={T}, d={d}")
            ax.grid(True, alpha=0.3, which="both")
            if T == T_list[-1]:
                ax.legend(fontsize=7, loc="lower left")
        fig.suptitle(
            f"Day 5 sanity: LB shape vs empirical merges  (d={d})")
        fig.tight_layout()
        fp = savedir / f"day5_sanity_d{d}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        print(f"saved {fp}")


def plot_excess_over_floor(records, T_list, d_list, savedir):
    """Focus plot: (D - floor) on log scale, to see decay regime."""
    for d in d_list:
        fig, axes = plt.subplots(1, len(T_list),
                                 figsize=(5 * len(T_list), 4.0))
        if len(T_list) == 1:
            axes = [axes]
        for ax, T in zip(axes, T_list):
            subset = [r for r in records
                      if r["T"] == T and r["d"] == d]
            bs = np.array([r["b"] for r in subset])
            floor_emp = np.array([r["cheb_radius_sq_emp"]
                                  for r in subset])
            # subtract empirical Chebyshev radius^2 (tightest floor)
            excess_mean = np.maximum(
                np.array([r["mean_merge_D"] for r in subset])
                - floor_emp, 1e-12)
            excess_cheb = np.maximum(
                np.array([r["cheb_merge_D"] for r in subset])
                - floor_emp, 1e-12)
            ax.plot(bs, excess_mean, "o-", color="C0",
                    label="mean merge: $D - R_c^2$")
            ax.plot(bs, excess_cheb, "s-", color="C1",
                    label="Cheb merge: $D - R_c^2$")
            # reference slopes
            ref_2 = 2.0 ** (-2.0 * bs)
            ref_1 = 2.0 ** (-1.0 * bs)
            ax.plot(bs, ref_2 * excess_mean[0] / ref_2[0], "k:",
                    alpha=0.5, label=r"slope $2^{-2b}$")
            ax.plot(bs, ref_1 * excess_mean[0] / ref_1[0], "k--",
                    alpha=0.5, label=r"slope $2^{-b}$")
            ax.set_xlabel("b (bits per coord)")
            if T == T_list[0]:
                ax.set_ylabel("excess over empirical Chebyshev floor")
            ax.set_yscale("log")
            ax.set_title(f"T={T}, d={d}")
            ax.grid(True, alpha=0.3, which="both")
            if T == T_list[-1]:
                ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Day 5: excess distortion decay regime  (d={d})")
        fig.tight_layout()
        fp = savedir / f"day5_excess_d{d}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        print(f"saved {fp}")


def main():
    T_list = [2, 4, 8]
    d_list = [128, 512, 2048]
    b_list = [1, 2, 3, 4, 6, 8, 12]
    records = run_sweep(T_list, d_list, b_list)
    print_cheb_table(records)
    print_merge_table(records)
    plot_comparison(records, T_list, d_list, FIG_DIR)
    plot_excess_over_floor(records, T_list, d_list, FIG_DIR)


if __name__ == "__main__":
    main()

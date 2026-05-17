"""Day 1 sanity check: three candidate distortion measures for LoRA merging.

Setup
-----
Task vectors tau_t iid uniform on B*S^{d-1}. Base w_0 = 0 (WLOG).
Merge: quantized mean of task vectors. Uniform scalar quantization
per coordinate, b bits per coord, total rate R = b*d bits.

The merged model w* = Q_b(mean_t tau_t) is intentionally simple.
It is NOT the optimal merge for max-distortion (Chebyshev center is),
but it is the mean-field natural choice and reveals each metric's shape.

Three metrics per trial
-----------------------
- raw_max:  max_t ||w* - tau_t||^2
- excess:   raw_max - cheb_sq   where cheb_sq is a Chebyshev-radius^2 estimate.
            For iid unit-sphere tau_t in high d, cheb_sq is tightly concentrated
            around B^2 * (1 - 1/T) because the mean is (approximately) the
            Chebyshev center: E[||mean||^2] = B^2/T and ||mean - tau_i||^2
            concentrates at B^2 * (1 - 1/T) uniformly over i.
            Under this (valid in the regime we care about) approximation
            the excess should -> 0 as b -> infinity.
- sum:      sum_t ||w* - tau_t||^2

Sweep
-----
T in {2, 4, 8}, d in {128, 512, 2048}, b in {1, 2, 3, 4, 6, 8, 12}.
30 trials per (T, d, b); report median. B = 1 throughout.

Expected shapes (from back-of-envelope)
---------------------------------------
At high b, w* -> mean(tau), and:
- raw_max plateaus near B^2 * (1 + 1/T)      [mean is not Chebyshev center]
- excess  plateaus near B^2 / T              [cost of mean vs origin]
- sum     plateaus near T * B^2 * (1 - 1/T)  [mean IS optimal for sum]

At low b, all three are dominated by quantization noise
~ d * B^2 / (3 * 4^b) on the mean, which transfers to the metrics.

If these predictions bear out numerically, it tells us:
- The "excess" metric has the cleanest 0-at-inf-rate shape.
- The "sum" metric is the natural mean-optimal RD setup.
- The "raw" metric has a data-dependent floor we'd need to handle.

Outputs
-------
PNG plots to code/synthetic/figures/ and a printed table.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RNG = np.random.default_rng(20260420)
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def sample_task_vectors(T: int, d: int, B: float = 1.0, rng=None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    X = rng.standard_normal((T, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return B * X


def uniform_scalar_quantize(v: np.ndarray, b: int, v_min: float, v_max: float) -> np.ndarray:
    if b >= 30:
        return v.copy()
    levels = 2 ** b
    step = (v_max - v_min) / levels
    idx = np.floor((v - v_min) / step).astype(np.int64)
    idx = np.clip(idx, 0, levels - 1)
    return v_min + (idx + 0.5) * step


def merge_quantized_mean(tau: np.ndarray, b: int, B: float) -> np.ndarray:
    m = tau.mean(axis=0)
    return uniform_scalar_quantize(m, b, -B, B)


def run_sweep(T_list, d_list, b_list, B: float = 1.0, n_trials: int = 30):
    records = []
    for T in T_list:
        for d in d_list:
            trials = [sample_task_vectors(T, d, B=B, rng=RNG) for _ in range(n_trials)]
            cheb_sq_floor = (B ** 2) * (1.0 - 1.0 / T)
            for b in b_list:
                raws, excesses, sums = [], [], []
                for tau in trials:
                    w = merge_quantized_mean(tau, b, B=B)
                    sq = np.sum((tau - w) ** 2, axis=1)
                    raw_max = float(np.max(sq))
                    raws.append(raw_max)
                    excesses.append(max(0.0, raw_max - cheb_sq_floor))
                    sums.append(float(np.sum(sq)))
                records.append({
                    "T": T, "d": d, "b": b,
                    "raw_max": float(np.median(raws)),
                    "excess": float(np.median(excesses)),
                    "sum": float(np.median(sums)),
                })
    return records


def print_table(records):
    print(f"{'T':>3} {'d':>6} {'b':>3} {'raw_max':>10} {'excess':>10} {'sum':>10}")
    for r in records:
        print(f"{r['T']:>3} {r['d']:>6} {r['b']:>3} "
              f"{r['raw_max']:>10.4g} {r['excess']:>10.4g} {r['sum']:>10.4g}")


def plot_d(records, d, T_list, b_list, savedir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ["raw_max", "excess", "sum"]
    titles = [
        r"$\max_t \|w^\star-\tau_t\|^2$  (raw)",
        r"$\max_t \|w^\star-\tau_t\|^2 - B^2(1-\frac{1}{T})$  (excess)",
        r"$\sum_t \|w^\star-\tau_t\|^2$  (sum)",
    ]
    # reference line slope: quant MSE contribution is d * B^2 / (3 * 4^b);
    # plot a dashed guide at rate slope 4^{-b}
    for ax, metric, title in zip(axes, metrics, titles):
        for T in T_list:
            xs = [r['b'] for r in records if r['T'] == T and r['d'] == d]
            ys = [r[metric] for r in records if r['T'] == T and r['d'] == d]
            ax.plot(xs, ys, marker='o', label=f"T={T}")
        ax.set_xlabel('b  (bits per coord,  total rate R = b*d)')
        ax.set_ylabel('distortion')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        ax.set_yscale('log')
        # ideal 4^{-b} reference slope
        bs = np.array(sorted(set([r['b'] for r in records if r['d'] == d])))
        ref = d / (3.0 * (4.0 ** bs))
        ax.plot(bs, ref, 'k--', alpha=0.4, label=r'$d/(3\cdot 4^b)$')
    fig.suptitle(f'd = {d}, B = 1, quantized-mean merge, 30 trials (median). '
                 r'Dashed: quant MSE reference $d/(3\cdot 4^b)$.')
    fig.tight_layout()
    fp = savedir / f"day1_distortion_measures_d{d}.png"
    fig.savefig(fp, dpi=110)
    plt.close(fig)
    return fp


def main():
    T_list = [2, 4, 8]
    d_list = [128, 512, 2048]
    b_list = [1, 2, 3, 4, 6, 8, 12]
    records = run_sweep(T_list, d_list, b_list)
    print_table(records)
    for d in d_list:
        fp = plot_d(records, d, T_list, b_list, FIG_DIR)
        print(f"saved {fp}")


if __name__ == "__main__":
    main()

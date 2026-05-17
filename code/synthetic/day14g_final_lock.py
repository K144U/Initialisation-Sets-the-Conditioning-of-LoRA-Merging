"""Day 14g — Final lock on the c∈[10.5, 12.5] sweet spot with
n_trials=1000 for tight slope confidence intervals.

Also bootstraps 95% CI on the slope estimates so we can write
"slope = -1.60 ± 0.XX" in the paper.
"""

import math
import numpy as np
from day11_task_dep_Dt import build_D_list
from day14e_quantizer_sweep import (
    qz_fixed, merge_variant, run_cell, summarize,
)


def bootstrap_slope(results_per_R, R_list, r, n_boot=300, rng=None):
    """Bootstrap 95% CI on slope of log2(exc_max) vs b=R/r.
    results_per_R[R] has a list of 'max' and 'floor' per trial."""
    if rng is None:
        rng = np.random.default_rng(7777)
    n_trials = len(results_per_R[R_list[0]]["max"])
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        rows = []
        for R in R_list:
            mx = np.array(results_per_R[R]["max"])[idx].mean()
            fl = np.array(results_per_R[R]["floor"])[idx].mean()
            exc = mx - fl
            if exc > 1e-12:
                rows.append({"b": R/r, "log_exc": math.log2(exc)})
        if len(rows) >= 2:
            slope = np.polyfit([r_["b"] for r_ in rows],
                               [r_["log_exc"] for r_ in rows], 1)[0]
            slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan"), float("nan")
    slopes = np.array(sorted(slopes))
    return (float(np.mean(slopes)),
            float(slopes[int(0.025 * len(slopes))]),
            float(slopes[int(0.975 * len(slopes))]))


def main():
    B = 1.0
    n_trials = 1000
    R_list = [8, 12, 16, 20, 24, 32]

    variants = [
        dict(name=f"c={c}", qz=qz_fixed, kwargs={"c": float(c)})
        for c in [10.5, 11.0, 11.5, 12.0, 12.5]
    ]

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (2, 128, 4, ["lin", "twoscale"]),
        (2, 128, 4, ["geom", "iso"]),
    ]

    print("=" * 120)
    print(f"Day 14g — Final lock, n_trials={n_trials}, bootstrap 95% CI")
    print("=" * 120)
    header = f"{'variant':<10} {'cell':<18} {'slope_max':>10} {'CI_lo':>8} "\
             f"{'CI_hi':>8} {'exc32':>10} {'slope_avg':>10}"
    print(header, flush=True)

    all_mean_dev = {}
    all_max_dev = {}
    for variant in variants:
        vname = variant["name"]
        slopes_for_variant = []
        for T, d, r, specs in cases:
            D_list = build_D_list(specs, r)
            case_label = '+'.join(s[:4] for s in specs)
            rng = np.random.default_rng(20260509)
            res = run_cell(T, d, r, D_list, R_list, n_trials, variant,
                           B=B, RNG=rng)
            rows, slope_max, slope_avg = summarize(
                f"{vname}", res, R_list, r)
            mean_s, ci_lo, ci_hi = bootstrap_slope(
                res, R_list, r, n_boot=500,
                rng=np.random.default_rng(hash(vname+case_label) & 0xffff))
            exc32 = rows[-1]["exc_max"]
            slopes_for_variant.append(slope_max)
            print(f"{vname:<10} {case_label:<18} {slope_max:>+10.4f} "
                  f"{ci_lo:>+8.3f} {ci_hi:>+8.3f} {exc32:>10.5f} "
                  f"{slope_avg:>+10.4f}", flush=True)
        devs = [abs(s + 1.6) for s in slopes_for_variant]
        all_mean_dev[vname] = float(np.mean(devs))
        all_max_dev[vname] = float(max(devs))
        print(f"  >> {vname} summary: mean_dev={all_mean_dev[vname]:.4f}  "
              f"max_dev={all_max_dev[vname]:.4f}  "
              f"slopes={slopes_for_variant}", flush=True)
        print("", flush=True)

    print("=" * 120)
    print("Overall ranking:")
    print("=" * 120)
    for vname in sorted(all_mean_dev, key=lambda v: all_mean_dev[v]):
        print(f"  {vname:<10} mean_dev={all_mean_dev[vname]:.4f}  "
              f"max_dev={all_max_dev[vname]:.4f}")


if __name__ == "__main__":
    main()

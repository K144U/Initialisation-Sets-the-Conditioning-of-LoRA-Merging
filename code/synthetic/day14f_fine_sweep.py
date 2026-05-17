"""Day 14f — Fine c-sweep around the c=14 sweet spot found in Day 14e.

Adds:
  - 500 trials (vs 300) for lower noise on slope fits.
  - c in {11, 12, 13, 14, 15, 16, 17, 18}.
  - Gaussian-optimal rate-adaptive clip: c_eff = sqrt(2 * ln(2) * b_i)
    per coord (b_i is the coord's bit width). This is the clip that
    matches the tail mass of a standard Gaussian to the quantizer
    granularity, giving asymptotic scalar-quantizer optimality.
  - Uniformly-optimal scalar clip: c_eff = 2.0 * sqrt(b_i + 1) which
    is a common heuristic.
"""

import math
import numpy as np
from day11_task_dep_Dt import build_D_list
from day14e_quantizer_sweep import (
    qz_fixed, qz_empmax, qz_empmax_sigmafloor, qz_percoord_empmax,
    merge_variant, run_cell, summarize, fit_slope,
    _frac_bit_widths, _quantize_scalar,
)
from day13_achievability_fixes import sample_orthogonal


def qz_gaussopt(v_sub, R_sub, sigma_pc, c0, rng):
    """Rate-adaptive per-coord: c_eff = c0 * sqrt(b_i * 2 ln 2) * sigma_pc.
    For Gaussian source with sigma = sigma_pc, the optimal symmetric
    clip grows as sqrt(b) up to log-factors; c0 is a tuning knob.
    """
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub
    bit_widths = _frac_bit_widths(R_sub, k)
    rng.shuffle(bit_widths)
    v_rot_q = np.zeros(k)
    for i in range(k):
        b = int(bit_widths[i])
        if b <= 0:
            v_rot_q[i] = 0.0
        else:
            clip = c0 * math.sqrt(b * 2 * math.log(2)) * sigma_pc
            v_rot_q[i] = _quantize_scalar(v_rot[i], clip, b)
    return O.T @ v_rot_q


def qz_linear_adaptive(v_sub, R_sub, sigma_pc, c0, rng):
    """c_eff = c0 * sqrt(b_i + 1) * sigma_pc. Linear-in-bit adaptive."""
    k = v_sub.shape[0]
    if k == 0:
        return v_sub
    O = sample_orthogonal(k, rng)
    v_rot = O @ v_sub
    bit_widths = _frac_bit_widths(R_sub, k)
    rng.shuffle(bit_widths)
    v_rot_q = np.zeros(k)
    for i in range(k):
        b = int(bit_widths[i])
        if b <= 0:
            v_rot_q[i] = 0.0
        else:
            clip = c0 * math.sqrt(b + 1) * sigma_pc
            v_rot_q[i] = _quantize_scalar(v_rot[i], clip, b)
    return O.T @ v_rot_q


def main():
    B = 1.0
    n_trials = 500
    R_list = [8, 12, 16, 20, 24, 32]

    variants = [
        dict(name=f"c={int(c)}", qz=qz_fixed, kwargs={"c": float(c)})
        for c in [11, 12, 13, 14, 15, 16, 17, 18]
    ] + [
        dict(name="gaussopt-c0=1.0", qz=qz_gaussopt, kwargs={"c0": 1.0}),
        dict(name="gaussopt-c0=1.5", qz=qz_gaussopt, kwargs={"c0": 1.5}),
        dict(name="gaussopt-c0=2.0", qz=qz_gaussopt, kwargs={"c0": 2.0}),
        dict(name="gaussopt-c0=3.0", qz=qz_gaussopt, kwargs={"c0": 3.0}),
        dict(name="linadapt-c0=3",   qz=qz_linear_adaptive, kwargs={"c0": 3.0}),
        dict(name="linadapt-c0=5",   qz=qz_linear_adaptive, kwargs={"c0": 5.0}),
        dict(name="linadapt-c0=7",   qz=qz_linear_adaptive, kwargs={"c0": 7.0}),
    ]

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (2, 128, 4, ["lin", "twoscale"]),
        (2, 128, 4, ["geom", "iso"]),
    ]

    header = f"{'variant':<22} " + " ".join(
        [f"{'+'.join(s[:4] for s in specs):>18}" for _, _, _, specs in cases])
    print("=" * 110)
    print(f"Day 14f — Fine c-sweep, n_trials={n_trials}")
    print("=" * 110)
    print(header, flush=True)

    best_slopes = {}  # variant_name -> list of |slope - (-1.6)| across cells
    for variant in variants:
        vname = variant["name"]
        line_parts = [f"{vname:<22}"]
        slopes = []
        for T, d, r, specs in cases:
            D_list = build_D_list(specs, r)
            rng = np.random.default_rng(20260508)
            res = run_cell(T, d, r, D_list, R_list, n_trials, variant,
                           B=B, RNG=rng)
            rows, slope_max, slope_avg = summarize(
                f"{vname}", res, R_list, r)
            slopes.append(slope_max)
            exc32 = rows[-1]["exc_max"]
            line_parts.append(f" {slope_max:+6.3f}/{exc32:7.5f} ")
        best_slopes[vname] = slopes
        line_str = "  ".join(line_parts)
        print(line_str, flush=True)

    # Aggregate score: mean |slope - (-1.6)| across cells + max deviation.
    print("\n" + "=" * 110)
    print("Ranked by (mean |slope - (-1.6)|) across all 5 cells:")
    print("=" * 110)
    ranked = sorted(best_slopes.items(),
                    key=lambda kv: np.mean([abs(s + 1.6) for s in kv[1]]))
    print(f"{'rank':<6}{'variant':<22}{'mean_|dev|':>12}{'max_|dev|':>12}"
          f"{'slopes':>60}")
    for i, (vname, slopes) in enumerate(ranked, 1):
        devs = [abs(s + 1.6) for s in slopes]
        mean_dev = np.mean(devs)
        max_dev = max(devs)
        slope_str = " ".join([f"{s:+.3f}" for s in slopes])
        print(f"{i:<6}{vname:<22}{mean_dev:>12.4f}{max_dev:>12.4f}"
              f"   {slope_str}", flush=True)

    print("\nMin-max rule (minimize worst-cell deviation from -1.6):")
    minmax = sorted(best_slopes.items(),
                    key=lambda kv: max(abs(s + 1.6) for s in kv[1]))
    for i, (vname, slopes) in enumerate(minmax[:5], 1):
        devs = [abs(s + 1.6) for s in slopes]
        slope_str = " ".join([f"{s:+.3f}" for s in slopes])
        print(f"  {i}. {vname:<22} max_dev={max(devs):.4f}  slopes=[{slope_str}]",
              flush=True)


if __name__ == "__main__":
    main()

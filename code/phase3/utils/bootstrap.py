"""Bootstrap CI helpers, ported from day14g_final_lock.py.

Two flavors:
  - bootstrap_slope: 95% CI on slope of log2(excess) vs bits/param,
    for the TVQ rate-decay validation criterion.
  - bootstrap_mean_ci: 95% CI on the mean of a sample,
    for per-token NLL aggregation.
"""

import math
from typing import Sequence

import numpy as np


def bootstrap_slope(
    results_per_R: dict,
    R_list: Sequence[int],
    r: int,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on slope of log2(max - floor) vs b = R / r.

    results_per_R[R] must contain lists "max" and "floor" with one
    entry per trial (same length across all R).
    """
    if rng is None:
        rng = np.random.default_rng(7777)
    n_trials = len(results_per_R[R_list[0]]["max"])
    slopes: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        bs, log_excs = [], []
        for R in R_list:
            mx = np.array(results_per_R[R]["max"])[idx].mean()
            fl = np.array(results_per_R[R]["floor"])[idx].mean()
            exc = mx - fl
            if exc > 1e-12:
                bs.append(R / r)
                log_excs.append(math.log2(exc))
        if len(bs) >= 2:
            slopes.append(float(np.polyfit(bs, log_excs, 1)[0]))
    if not slopes:
        return float("nan"), float("nan"), float("nan")
    s = np.sort(np.array(slopes))
    return float(s.mean()), float(s[int(0.025 * len(s))]), float(s[int(0.975 * len(s))])


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the mean of a 1-D sample."""
    if rng is None:
        rng = np.random.default_rng(7777)
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, v.size, size=v.size)
        means[i] = v[idx].mean()
    means.sort()
    return (
        float(means.mean()),
        float(means[int(0.025 * n_boot)]),
        float(means[int(0.975 * n_boot)]),
    )

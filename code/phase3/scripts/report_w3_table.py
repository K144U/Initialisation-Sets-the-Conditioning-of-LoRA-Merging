#!/usr/bin/env python3
"""Emit the paper's rate-sweep table from the data, at checkable precision.

analyze_w3_rate_fixed.py owns the VERDICT and its registered rule is not
touched here. This script owns the TABLE, and exists because the published
version could not be checked: at three decimals a reader cannot tell whether
a value sits above or below its own asymptote, and the fitted quantity was
never stated.

What it adds, none of which changes any rule:

  * four decimals, so the below-asymptote cases are visible;
  * the residual excess - asymptote, which is what is actually fitted;
  * a mark on every residual below the metric's resolution, which the paper
    estimates at about 0.001 nats (discussion, limitation 1);
  * the points each fit used, so the fit can be reproduced by hand.

The fit is ordinary least squares of log2(excess - asymptote) on b, over
b in {2,3,4,8}, keeping points whose residual exceeds 1e-9. That filter is
the one the committed analyzer used and it is deliberately NOT changed here:
moving it after seeing which points it admits is exactly the practice the
pre-registrations exist to prevent. The consequence of leaving it alone is
that two fits survive on points the paper's own resolution says are
indistinguishable from zero, and the table now shows that rather than hiding
it behind rounding.

Usage:
  python code/phase3/scripts/report_w3_table.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

BASES = [("llama31_8b", "Llama-3.1-8B"), ("mistral_7b", "Mistral-7B"),
         ("qwen25_7b", "Qwen2.5-7B"), ("yi15_9b", "Yi-1.5-9B")]
BITS = [1, 2, 3, 4, 8, 16]
FIT_BITS = [2, 3, 4, 8]
FIT_FLOOR = 1e-9        # the committed analyzer's filter, unchanged
RESOLUTION = 0.001      # nats; the paper's own estimate, discussion limitation 1

ASYM = {
    "llama31_8b": ("eval_seed_rdridge_regmean", "llama31_8b__rd_ridge__seed1"),
    "mistral_7b": ("eval_w3_asymptote", "mistral_7b__rd_ridge_binf__seed1"),
    "qwen25_7b": ("eval_w3_asymptote", "qwen25_7b__rd_ridge_binf__seed1"),
    "yi15_9b": ("eval_w3_asymptote", "yi15_9b__rd_ridge_binf__seed1"),
}


def worst(p: Path):
    return json.loads(p.read_text()).get("worst_task_excess") if p.exists() else None


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = sxy / sxx
    inter = my - slope * mx
    ss_res = sum((b - (slope * a + inter)) ** 2 for a, b in zip(xs, ys))
    ss_tot = sum((b - my) ** 2 for b in ys)
    return slope, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def main() -> int:
    print("=" * 100)
    print("Rate sweep, at the precision the claim needs")
    print("=" * 100)
    print(f"fit target: log2(excess - asymptote) on b, over b in {FIT_BITS}")
    print(f"fit filter: residual > {FIT_FLOOR:g}  (the committed analyzer's, unchanged)")
    print(f"metric resolution, paper's own estimate: {RESOLUTION} nats")
    print()

    n_below = n_noise_fits = 0
    latex = []
    for key, label in BASES:
        vals = {b: worst(RES / "eval_w3_rate" / f"{key}__rd_ridge_b{b}__seed1.json")
                for b in BITS}
        d, nm = ASYM[key]
        einf = worst(RES / d / f"{nm}.json")
        if einf is None or any(v is None for v in vals.values()):
            print(f"{label}: incomplete")
            continue

        print(f"{label}   asymptote {einf:.6f}")
        for b in BITS:
            r = vals[b] - einf
            mark = ""
            if r < 0:
                mark = "  BELOW ASYMPTOTE"
                n_below += 1
            elif abs(r) < RESOLUTION:
                mark = "  below resolution"
            print(f"   b={b:<3} {vals[b]:.6f}   residual {r:+.8f}{mark}")

        pts = [(b, vals[b] - einf) for b in FIT_BITS
               if vals[b] - einf > FIT_FLOOR]
        used = [b for b, _ in pts]
        if len(pts) >= 3:
            slope, r2 = ols([float(b) for b, _ in pts],
                            [math.log2(q) for _, q in pts])
            sub = [b for b, r in pts if r < RESOLUTION]
            if sub:
                n_noise_fits += 1
            print(f"   fit on b={used}: slope {slope:.4f}, R2 {r2:.4f}")
            if sub:
                print(f"   *** {len(sub)} of {len(pts)} fitted residuals are below "
                      f"the metric's resolution: b={sub}")
            slope_s, r2_s = f"${slope:.2f}$", f"{r2:.4f}"
        else:
            print(f"   unfittable: only {len(pts)} residual(s) above the filter "
                  f"(b={used})")
            slope_s, r2_s = "---", "---"
        print()

        cells = " & ".join(f"{vals[b]:.4f}" for b in BITS)
        latex.append(f"{label} & {cells} & {einf:.4f} & "
                     f"{','.join(str(b) for b in used) if used else '---'} & "
                     f"{slope_s} & {r2_s} \\\\")

    print("=" * 100)
    print(f"values below their own asymptote: {n_below}")
    print(f"fits containing a residual below the metric's resolution: {n_noise_fits}")
    print()
    print("LaTeX table body:")
    for row in latex:
        print("  " + row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""W3 verdict: does the achievability exponent show up on real adapters?

The encoder quantizes eta, which is (out x d_eff), at b bits per entry, so the
rate is R = b * out * d_eff over n = out * d_eff dimensions and the theory's
2^{-2R/n} reduces to 2^{-2b}: excess should fall 4x per added bit, a slope of
-2 in log2(excess) against b. The synthetic validation reports -2.00 +/- 0.01
(App. A); this is the same fit on real adapters.

What gets fitted is the QUANTIZATION contribution,

    q(b) = excess(b) - excess(inf)

not raw excess, because excess(inf) is the merge error itself and does not
vanish with rate. Fitting raw excess would flatten the slope toward 0 for free
and would not test anything.

Decision rule, fixed before the cells run:

  slope in [-2.4, -1.6] on >= 3 of 4 bases
      the rate axis behaves as Theorem 4 predicts on real adapters. This is the
      paper's first real-data confirmation of the achievability exponent and
      belongs in the main text.

  otherwise
      the quadratic surrogate does not describe real merging in the rate regime
      where the theory has content. The abstract's implication that the
      rate-distortion machinery does empirical work must be withdrawn, not just
      qualified in 6.1.

Also prints the published lambda = 0 sweep from eval_e1/ for contrast: that one
is non-monotone on 4 of 4 bases, but it conflates the sliver blow-up with the
rate axis, which is exactly why the ridge-on sweep was needed.

Usage:  python code/phase3/scripts/analyze_w3_rate.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BITS = [1, 2, 3, 4, 8, 16]
SEEDS = ["seed1", "seed2", "seed3"]
SLOPE_LO, SLOPE_HI = -2.4, -1.6
# Fit over the rates where quantization actually dominates. b=1 is the
# clipping-dominated regime (the published lambda=0 sweep shows 9-13 nats
# there) and is reported but excluded from the fit.
FIT_BITS = [2, 3, 4, 8]


def worst(p: Path) -> float | None:
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["worst_task_excess"])
    except Exception:
        return None


def excess_inf(base: str) -> float | None:
    """b -> inf reference: the published 3-seed rd-ridge mean."""
    for d, pat in [("eval_seed_rdridge_regmean", "{b}__rd_ridge__{s}.json"),
                   ("eval_ridge_seed", "{b}__ridge_l0p05__{s}.json")]:
        v = [worst(RES / d / pat.format(b=base, s=s)) for s in SEEDS]
        v = [x for x in v if x is not None]
        if v:
            return statistics.mean(v)
    return None


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = sxy / sxx
    inter = my - slope * mx
    ss_res = sum((b - (slope * a + inter)) ** 2 for a, b in zip(xs, ys))
    ss_tot = sum((b - my) ** 2 for b in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, inter, r2


def main() -> int:
    print("=" * 88)
    print("W3  rd-encoder ridge at finite rate (lambda = lambda*, realize=rank_deff, seed1)")
    print("=" * 88)
    print(f"{'base':<13}" + "".join(f"{('b=' + str(b)):>9}" for b in BITS)
          + f"{'b=inf':>9}{'slope':>9}{'R2':>7}  verdict")

    n_ok, n_have = 0, 0
    summary = {}
    for base in BASES:
        vals = {b: worst(RES / "eval_w3_rate"
                         / f"{base}__rd_ridge_b{b}__seed1.json") for b in BITS}
        einf = excess_inf(base)
        row = f"{base:<13}"
        for b in BITS:
            row += f"{vals[b]:>9.3f}" if vals[b] is not None else f"{'--':>9}"
        row += f"{einf:>9.3f}" if einf is not None else f"{'--':>9}"

        pts = [(b, vals[b] - einf) for b in FIT_BITS
               if vals.get(b) is not None and einf is not None
               and vals[b] - einf > 1e-9]
        if len(pts) >= 3:
            n_have += 1
            slope, _, r2 = ols([float(b) for b, _ in pts],
                               [math.log2(q) for _, q in pts])
            ok = SLOPE_LO <= slope <= SLOPE_HI
            n_ok += ok
            row += f"{slope:>9.2f}{r2:>7.3f}  {'matches -2' if ok else 'OFF'}"
            summary[base] = {"slope": slope, "r2": r2, "n_fit": len(pts)}
        else:
            row += f"{'--':>9}{'--':>7}  pending"
        print(row)

    print(f"\n  fit: log2(excess(b) - excess(inf)) vs b over b in {FIT_BITS}; "
          f"b=1 shown but excluded (clipping-dominated).")
    if n_have:
        print(f"  VERDICT: slope in [{SLOPE_LO}, {SLOPE_HI}] on {n_ok}/{n_have} "
              f"bases fitted (>=3 of 4 => the exponent holds on real adapters).")
    else:
        print("  VERDICT: pending, no eval_w3_rate cells yet.")

    # Contrast: the published lambda = 0 sweep, never shown as a rate curve.
    print()
    print("=" * 88)
    print("For contrast: the published lambda = 0 sweep (eval_e1/, rank_r)")
    print("=" * 88)
    e1 = {1: "rd_b1", 2: "rd_b2", 3: "rd_b3", 4: "rd_b4",
          8: "rd_b8", 16: "rd_b16", 32: "rd_b32"}
    print(f"{'base':<13}" + "".join(f"{('b=' + str(b)):>9}" for b in e1)
          + "  monotone?")
    for base in BASES:
        vals = {b: worst(RES / "eval_e1" / f"{base}__{tag}.json")
                for b, tag in e1.items()}
        row = f"{base:<13}"
        for b in e1:
            row += f"{vals[b]:>9.3f}" if vals[b] is not None else f"{'--':>9}"
        got = [vals[b] for b in e1 if vals[b] is not None]
        mono = all(a >= b for a, b in zip(got, got[1:])) if len(got) > 1 else None
        row += f"  {'yes' if mono else 'NO'}"
        print(row)
    print("\n  lambda = 0 conflates the sliver blow-up with the rate axis, which"
          "\n  is why the ridge-on sweep above is the one that tests Theorem 4.")

    if summary:
        p = RES / "w3_rate_summary.json"
        p.write_text(json.dumps(summary, indent=2))
        print(f"\n[analyze] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

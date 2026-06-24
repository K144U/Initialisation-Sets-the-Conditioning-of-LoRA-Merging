"""E11 quadratic-bridge analysis.

Tests the prediction: as delta_scale alpha -> 0, the rd-encoder ridge
solution approaches the Fisher-quadratic diagonal optimum (which our
fisher_avg method instantiates as F_t[i] = Delta_t[i]^2 weighted avg).

For 2 bases (Llama-3.1, Yi-1.5-Chat) x 4 scales {0.1, 0.25, 0.5, 1.0},
loads the worst-task NLL excess of rd_encoder and fisher_avg and
reports:
  - Phi^ridge(alpha) and Phi^fisher(alpha) at each scale.
  - Ratio Phi^ridge(alpha) / Phi^fisher(alpha) at each scale.
  - Trend of the ratio as alpha -> 0 (should approach 1 if prediction
    holds).

Decision rule (Td2 / §6.8 prediction):
  alpha=0.1 ratio in [0.8, 1.2]  -> PREDICTION HOLDS (ridge ~ Fisher
    in small-perturbation limit).
  alpha=0.1 ratio outside [0.8, 1.2] AND monotone toward 1 from
    alpha=1.0 -> PARTIAL (right direction, asymptote not reached at
    alpha=0.1; would need smaller alpha or finer analysis).
  alpha=0.1 ratio NOT trending toward 1 -> PREDICTION FAILS, paper
    reports honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
E11_DIR = PROJECT_ROOT / "results/phase3/eval_e11_quadbridge"
OUT_JSON = PROJECT_ROOT / "results/phase3/e11_quadbridge_summary.json"

BASES = ["llama31_8b", "yi15_9b"]
SCALES = [0.1, 0.25, 0.5, 1.0]
METHODS = ["rd_encoder", "fisher_avg"]


def tag(scale: float) -> str:
    return f"a{int(scale * 100):03d}"


def load_excess(base: str, method: str, scale: float) -> float | None:
    p = E11_DIR / f"{base}__{method}__{tag(scale)}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("worst_task_excess")


def main() -> int:
    table: dict[str, dict[float, dict[str, float | None]]] = {}
    for base in BASES:
        table[base] = {}
        for s in SCALES:
            table[base][s] = {m: load_excess(base, m, s) for m in METHODS}

    print("=" * 88)
    print("E11 quadratic-bridge — worst-task NLL excess at varying delta_scale alpha")
    print("=" * 88)
    print(f"{'base':<14}{'alpha':<8}{'rd_encoder':<14}{'fisher_avg':<14}{'ratio R':<10}")
    summary = {}
    for base in BASES:
        summary[base] = {}
        for s in SCALES:
            r = table[base][s]["rd_encoder"]
            f = table[base][s]["fisher_avg"]
            ratio = r / f if (r is not None and f is not None and f > 0) else None
            summary[base][str(s)] = {"rd_encoder": r, "fisher_avg": f,
                                     "ratio": ratio}
            rstr = f"{r:<14.6f}" if r is not None else f"{'--':<14}"
            fstr = f"{f:<14.6f}" if f is not None else f"{'--':<14}"
            ratstr = f"{ratio:<10.3f}" if ratio is not None else f"{'--':<10}"
            print(f"{base:<14}{s:<8.2f}{rstr}{fstr}{ratstr}")
        print()

    # Quadratic-bridge verdict
    print("=" * 88)
    print("Quadratic-bridge verdict per base")
    print("=" * 88)
    for base in BASES:
        ratios = {s: summary[base][str(s)]["ratio"] for s in SCALES}
        r_01 = ratios.get(0.1)
        r_10 = ratios.get(1.0)
        if r_01 is None or r_10 is None:
            print(f"  {base}: incomplete")
            continue
        # Monotonicity check: ratios should move toward 1 as alpha shrinks
        ordered_ratios = [ratios[s] for s in sorted(SCALES, reverse=True)
                          if ratios[s] is not None]
        deltas_from_one = [abs(r - 1.0) for r in ordered_ratios]
        monotone_toward_one = all(deltas_from_one[i] >= deltas_from_one[i+1]
                                   for i in range(len(deltas_from_one)-1))
        in_window_at_small = (0.8 <= r_01 <= 1.2)

        print(f"  {base}:")
        print(f"    R(alpha=1.0) = {r_10:+.3f}  R(alpha=0.5) = {ratios[0.5]:+.3f}  "
              f"R(alpha=0.25) = {ratios[0.25]:+.3f}  R(alpha=0.1) = {r_01:+.3f}")
        if in_window_at_small:
            print(f"    VERDICT: PREDICTION HOLDS (ratio approaches 1 in small-alpha limit).")
        elif monotone_toward_one:
            print(f"    VERDICT: PARTIAL — ratio trends toward 1 but doesn't reach "
                  f"the [0.8, 1.2] window at alpha=0.1.")
        else:
            print(f"    VERDICT: PREDICTION DOES NOT HOLD — ratio does not consistently "
                  f"approach 1 as alpha shrinks.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

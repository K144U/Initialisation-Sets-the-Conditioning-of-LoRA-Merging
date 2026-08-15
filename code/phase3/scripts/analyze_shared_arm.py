#!/usr/bin/env python3
"""Verdict for the ridge sweep's shared arm at n = 3.

Rules: notes/prereg_shared_arm_2026-08-15.md (603e013). Committed before any
cell of this arm landed, which is the only reason its output is worth anything.

Implements exactly the registered rules and nothing else:

  lambda* is the arg-min over the grid, selected PER COHORT. The ridge gain
  G = L(0) - L(lambda*) is averaged across the three cohorts within an arm.

  Both arms are n = 3, so the difference between them carries
  SE = sqrt(SE_s^2 + SE_i^2), with SE = sd/sqrt(3) per arm. R8's gate used the
  independent term alone because the shared sd did not exist. That is the
  asymmetry this arm removes and limitation 6 records.

  Gate: a directional call is downgraded to a tie unless
  |mean difference| > max(0.005, 2 x SE). One-directional, may only downgrade.

  P1 (ridge gain)             G_shared > G_indep, gate cleared, >= 3 of 4 bases.
  P2 (unregularised penalty)  L_shared(0) >= 2 x L_indep(0), gate cleared on the
                              difference, >= 3 of 4 bases.

  CONFIRMED if both hold on >= 3 of 4. REFUTED if neither does. PARTIAL else.

Also reported regardless of outcome, as registered: the shared arm's
cross-cohort spread, where seed1 sits relative to the three-cohort shared mean
in sd units, and the minimum detectable effect under the two-sided gate.

The script REFUSES to print a verdict unless every cell of both arms is
present, on the same terms as analyze_dare_ties.

Usage:
  python code/phase3/scripts/analyze_shared_arm.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TAGS = ["0", "0p01", "0p03", "0p05", "0p13", "0p30", "1p00"]
LAM = {"0": 0.0, "0p01": 0.01, "0p03": 0.03, "0p05": 0.05,
       "0p13": 0.13, "0p30": 0.30, "1p00": 1.00}

TIE = 0.005          # nats, fixed 2026-08-03 and unchanged since
P2_FACTOR = 2.0      # "worse by a factor of at least 2"
N_OF_4 = 3           # "on at least 3 of 4 bases"

# cohort -> (results dir, filename prefix). seed1/indep1 are the published E2
# cells and are reused, not re-run.
SOURCES = {
    "seed1":  ("eval_ridge_cond",    "rd"),
    "seed2":  ("eval_ridge_shared",  "rds"),
    "seed3":  ("eval_ridge_shared",  "rds"),
    "indep1": ("eval_ridge_cond",    "rd"),
    "indep2": ("eval_ridge_cohorts", "rdc"),
    "indep3": ("eval_ridge_cohorts", "rdc"),
}
SHARED = ["seed1", "seed2", "seed3"]
INDEP = ["indep1", "indep2", "indep3"]


def load(base: str, cohort: str, tag: str) -> float | None:
    d, prefix = SOURCES[cohort]
    p = RES / d / f"{base}__{prefix}_l{tag}__{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["worst_task_excess"]


def se(xs: list[float]) -> float:
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def arm_stats(base: str, cohorts: list[str], grid: dict) -> dict | None:
    """Per-cohort lambda*, gain and L(0); then the arm mean and SE."""
    per = {}
    for c in cohorts:
        vals = {t: grid[(c, t)] for t in TAGS}
        if any(v is None for v in vals.values()):
            return None
        star = min(TAGS, key=lambda t: vals[t])
        per[c] = {"lstar": LAM[star], "lstar_tag": star,
                  "L0": vals["0"], "Lstar": vals[star],
                  "gain": vals["0"] - vals[star]}
    gains = [per[c]["gain"] for c in cohorts]
    l0s = [per[c]["L0"] for c in cohorts]
    return {
        "per": per,
        "gain_mean": st.mean(gains), "gain_se": se(gains),
        "l0_mean": st.mean(l0s), "l0_sd": st.stdev(l0s), "l0_se": se(l0s),
    }


def gate(diff: float, se_diff: float) -> tuple[bool, float]:
    """One-directional gate. Returns (passes, the threshold it had to clear)."""
    thr = max(TIE, 2.0 * se_diff)
    return abs(diff) > thr, thr


def main() -> int:
    grid = {}
    missing = []
    for base in BASES:
        for cohort in SOURCES:
            for tag in TAGS:
                v = load(base, cohort, tag)
                grid[(base, cohort, tag)] = v
                if v is None:
                    missing.append(f"{base} {cohort} l{tag}")

    n_expected = len(BASES) * len(SOURCES) * len(TAGS)
    print(f"cells present: {n_expected - len(missing)}/{n_expected}")
    if missing:
        print(f"MISSING {len(missing)} cells, first ten:")
        for m in missing[:10]:
            print("   ", m)

    print("\n=== shared arm, cross-cohort spread (registered report 1) ===")
    print(f"{'base':<12}{'lambda':>8}{'seed1':>10}{'seed2':>10}{'seed3':>10}"
          f"{'mean':>10}{'sd':>10}")
    for base in BASES:
        for tag in TAGS:
            vs = [grid[(base, c, tag)] for c in SHARED]
            if any(v is None for v in vs):
                continue
            print(f"{base:<12}{LAM[tag]:>8.2f}" + "".join(f"{v:>10.4f}" for v in vs)
                  + f"{st.mean(vs):>10.4f}{st.stdev(vs):>10.4f}")

    results = {}
    p1_hits, p2_hits = 0, 0
    print("\n=== per base ===")
    for base in BASES:
        g = {(c, t): grid[(base, c, t)] for c in SOURCES for t in TAGS}
        s = arm_stats(base, SHARED, g)
        i = arm_stats(base, INDEP, g)
        if s is None or i is None:
            print(f"{base}: incomplete, skipped")
            continue

        gain_diff = s["gain_mean"] - i["gain_mean"]
        gain_se = math.sqrt(s["gain_se"] ** 2 + i["gain_se"] ** 2)
        p1_gate, p1_thr = gate(gain_diff, gain_se)
        p1 = gain_diff > 0 and p1_gate

        l0_diff = s["l0_mean"] - i["l0_mean"]
        l0_se = math.sqrt(s["l0_se"] ** 2 + i["l0_se"] ** 2)
        p2_gate, p2_thr = gate(l0_diff, l0_se)
        ratio = s["l0_mean"] / i["l0_mean"] if i["l0_mean"] > 0 else float("inf")
        p2 = ratio >= P2_FACTOR and l0_diff > 0 and p2_gate

        p1_hits += p1
        p2_hits += p2

        # Registered report 2: where seed1 sits in its own arm, in sd units.
        s1_l0 = s["per"]["seed1"]["L0"]
        z_l0 = (abs(s1_l0 - s["l0_mean"]) / s["l0_sd"]) if s["l0_sd"] > 0 else 0.0

        results[base] = {
            "gain_shared": s["gain_mean"], "gain_indep": i["gain_mean"],
            "gain_diff": gain_diff, "gain_gate": p1_thr, "P1": p1,
            "l0_shared": s["l0_mean"], "l0_indep": i["l0_mean"],
            "l0_ratio": ratio, "l0_gate": p2_thr, "P2": p2,
            "lstar_shared": [s["per"][c]["lstar"] for c in SHARED],
            "lstar_indep": [i["per"][c]["lstar"] for c in INDEP],
            "seed1_z_at_l0": z_l0,
            "mde_gain": p1_thr, "mde_l0": p2_thr,
        }

        print(f"\n{base}")
        print(f"  lambda* shared     {[s['per'][c]['lstar'] for c in SHARED]}")
        print(f"  lambda* indep      {[i['per'][c]['lstar'] for c in INDEP]}")
        print(f"  ridge gain         shared {s['gain_mean']:.4f} "
              f"(SE {s['gain_se']:.4f})   indep {i['gain_mean']:.4f} "
              f"(SE {i['gain_se']:.4f})")
        print(f"  P1  diff {gain_diff:+.4f}  gate {p1_thr:.4f}  -> "
              f"{'HOLDS' if p1 else 'does not hold'}")
        print(f"  L(0)               shared {s['l0_mean']:.4f} "
              f"(SE {s['l0_se']:.4f})   indep {i['l0_mean']:.4f} "
              f"(SE {i['l0_se']:.4f})")
        print(f"  P2  ratio {ratio:.2f}x  diff {l0_diff:+.4f}  gate {p2_thr:.4f}"
              f"  -> {'HOLDS' if p2 else 'does not hold'}")
        print(f"  seed1 at L(0) sits {z_l0:.2f} sd from its own arm's mean")

    if missing:
        print("\nNO VERDICT: cells are missing. The registration requires the "
              "full design before any verdict is read.")
        return 1

    p1_ok = p1_hits >= N_OF_4
    p2_ok = p2_hits >= N_OF_4
    if p1_ok and p2_ok:
        verdict = "CONFIRMED"
    elif not p1_ok and not p2_ok:
        verdict = "REFUTED"
    else:
        verdict = "PARTIAL"

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  P1 holds on {p1_hits}/4 bases (needs {N_OF_4})")
    print(f"  P2 holds on {p2_hits}/4 bases (needs {N_OF_4})")

    # Registered report 2, the consequence clause.
    z_flags = sum(1 for b in results if results[b]["seed1_z_at_l0"] > 2.0)
    if z_flags >= 2:
        print(f"\n  NOTE, registered in advance: seed1 sits more than 2 sd from "
              f"its own arm's mean at lambda = 0 on {z_flags} of 4 bases. The "
              f"published single-cohort value was not representative of its own "
              f"arm, and the paper says so in those words.")

    out = RES / "shared_arm_summary.json"
    out.write_text(json.dumps(
        {"verdict": verdict, "p1_bases": p1_hits, "p2_bases": p2_hits,
         "per_base": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

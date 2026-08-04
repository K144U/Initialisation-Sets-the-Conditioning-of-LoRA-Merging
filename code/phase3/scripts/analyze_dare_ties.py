#!/usr/bin/env python3
"""Does DARE help when composed with TIES?

Implements notes/prereg_dare_ties_2026-08-04.md, committed at efb593f before the
dare_ties method existed and before any cell ran. Do not change a threshold in
this file. If a rule is badly specified, report the result under the rule as
written and record the problem as a limitation.

  arms       dare_ties (new) vs the EXISTING eval_a1_indep ties cells
  bases      4        cohorts  indep1/2/3        densities 0.5, 0.2, 0.1
  ties_density pinned 0.2 in both arms, so the only difference is the DARE mask

Q1  PRIMARY, at dare_density 0.2 ONLY. Naming one density in advance is what
    stops this being a three-shot test reported as one.
Q2  SECONDARY, descriptive: is the penalty monotone in density, as it was for
    DARE + task arithmetic on 3 of 4 bases?
Q3  free readout: at dare_density 0.1 the TIES trim is inert (fewer entries are
    nonzero than it wants to keep), so whatever advantage over TA survives there
    is the sign election's doing, not the trim's.

The noise gate is one-directional: it can only downgrade HELPS/HURTS to NEUTRAL,
never promote. n = 3, so the sd carries 2 degrees of freedom.

Usage:  python code/phase3/scripts/analyze_dare_ties.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
COHORTS = ["indep1", "indep2", "indep3"]
DENS = [(0.5, "0p5"), (0.2, "0p2"), (0.1, "0p1")]
PRIMARY_TAG = "0p2"        # named in the pre-registration, not chosen here

TIE = 0.005
GATE_SD = 2.0


def read_excess(p: Path) -> float | None:
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["worst_task_excess"])
    except Exception:
        return None


def ties_ref(base: str, cohort: str) -> float | None:
    return read_excess(RES / "eval_a1_indep" / f"{base}__ties__{cohort}.json")


def ta_ref(base: str, cohort: str) -> float | None:
    return read_excess(RES / "eval_a1_indep" / f"{base}__task_arithmetic__{cohort}.json")


def dt(base: str, tag: str, cohort: str) -> float | None:
    return read_excess(RES / "eval_dare_ties" /
                       f"{base}__dare_ties_d{tag}__{cohort}.json")


def fmt(x: float | None) -> str:
    return "   n/a" if x is None else f"{x:6.4f}"


def stats(d: list[float]) -> tuple[float, float, float]:
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return m, sd, sd / (len(d) ** 0.5)


def main() -> int:
    missing = [f"{b}/{t}/{c}" for b in BASES for _, t in DENS for c in COHORTS
               if dt(b, t, c) is None]
    if missing:
        print(f"[warn] {len(missing)} dare_ties cells missing: "
              f"{', '.join(missing[:8])}")

    print("=" * 100)
    print("DARE-TIES vs TIES, worst-task NLL excess (nats)")
    print("cell = mean over indep1/2/3, sd in brackets; ties_density 0.2 in both arms")
    print("=" * 100)
    hdr = f"{'base':<13}{'TIES':>17}" + "".join(
        f"{'dare_ties d=' + str(d):>19}" for d, _ in DENS)
    print(hdr)
    for b in BASES:
        tv = [ties_ref(b, c) for c in COHORTS if ties_ref(b, c) is not None]
        row = f"{b:<13}"
        row += (f"{statistics.fmean(tv):9.4f} [{statistics.stdev(tv):.4f}]"
                if len(tv) > 1 else f"{'n/a':>17}")
        for _, tag in DENS:
            v = [dt(b, tag, c) for c in COHORTS if dt(b, tag, c) is not None]
            row += (f"{statistics.fmean(v):11.4f} [{statistics.stdev(v):.4f}]"
                    if len(v) > 1 else f"{'n/a':>19}")
        print(row)

    # ---------------- Q1 ----------------
    print()
    print("=" * 100)
    print(f"Q1  PRIMARY, dare_density 0.2 only (named in the pre-registration)")
    print("    mean d = mean(TIES) - mean(DARE-TIES); positive means DARE HELPS")
    print("=" * 100)
    print(f"{'base':<13}{'TIES':>10}{'DARE-TIES':>12}{'mean d':>10}"
          f"{'sd(d)':>9}{'2xSE':>9}  result")
    helps = hurts = neutral = 0
    rows = []
    for b in BASES:
        pairs = [(ties_ref(b, c), dt(b, PRIMARY_TAG, c)) for c in COHORTS]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            print(f"{b:<13}{'n/a':>10}")
            continue
        dvals = [x - y for x, y in pairs]
        md, sd, se = stats(dvals)
        if md > TIE and abs(md) > GATE_SD * se:
            res, helps = "HELPS", helps + 1
        elif md < -TIE and abs(md) > GATE_SD * se:
            res, hurts = "HURTS", hurts + 1
        else:
            res, neutral = "NEUTRAL", neutral + 1
            if abs(md) > TIE:
                res += "  (downgraded, inside noise)"
        print(f"{b:<13}{statistics.fmean([x for x, _ in pairs]):10.4f}"
              f"{statistics.fmean([y for _, y in pairs]):12.4f}"
              f"{md:+10.4f}{sd:9.4f}{GATE_SD * se:9.4f}  {res}")
        rows.append({"base": b, "mean_d": md, "sd": sd, "two_se": GATE_SD * se,
                     "per_cohort_d": dvals, "result": res.split()[0]})

    if helps >= 2 and hurts == 0:
        v1 = "DARE HELPS TIES"
    elif hurts >= 2 and helps == 0:
        v1 = "DARE HURTS TIES"
    elif neutral >= 3:
        v1 = "DARE IS NEUTRAL"
    else:
        v1 = "MIXED (neither direction claimed)"
    print(f"\n  helps {helps}, neutral {neutral}, hurts {hurts}  ->  Q1 VERDICT: {v1}")
    if v1 == "DARE HELPS TIES":
        print("  CONSEQUENCE (fixed in advance): the DARE negative result must be")
        print("  rescoped to 'composed with task arithmetic'. notes/audit_dare_")
        print("  2026-08-04.md and the decisions.md entry are amended in place.")
    else:
        print("  CONSEQUENCE (fixed in advance): the negative result generalises to")
        print("  both compositions. NOTE that the unbiasedness mechanism does NOT")
        print("  explain this one, since it does not apply here. Report it as an")
        print("  unexplained empirical regularity. Do not invent a mechanism.")

    # ---------------- Q2 ----------------
    print()
    print("=" * 100)
    print("Q2  SECONDARY, descriptive: is the penalty monotone as density falls?")
    print("    (DARE + task arithmetic was monotone on 3 of 4 bases)")
    print("=" * 100)
    print(f"{'base':<13}" + "".join(f"{'d=' + str(d):>12}" for d, _ in DENS)
          + "   monotone?")
    mono = 0
    q2rows = []
    for b in BASES:
        pens = []
        for _, tag in DENS:
            pairs = [(ties_ref(b, c), dt(b, tag, c)) for c in COHORTS]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            pens.append(statistics.fmean([y - x for x, y in pairs]) if pairs else None)
        if any(p is None for p in pens):
            print(f"{b:<13} incomplete")
            continue
        # DENS is ordered 0.5, 0.2, 0.1, i.e. density FALLING
        is_mono = all(pens[i + 1] >= pens[i] - 1e-9 for i in range(len(pens) - 1))
        mono += int(is_mono)
        print(f"{b:<13}" + "".join(f"{p:+12.4f}" for p in pens)
              + f"   {'yes' if is_mono else 'NO'}")
        q2rows.append({"base": b, "penalties": pens, "monotone": is_mono})
    v2 = "MONOTONE" if mono >= 3 else "NOT MONOTONE"
    print(f"\n  monotone on {mono}/4  ->  Q2: {v2}")
    print("  (penalty = DARE-TIES minus TIES; positive means the mask hurt)")
    beats = [(r["base"], r["penalties"]) for r in q2rows if min(r["penalties"]) < -TIE]
    if beats:
        print("  bases where DARE-TIES beats TIES at SOME density, reported per the")
        print("  pre-registration even though Q1 is primary:")
        for b, p in beats:
            print(f"    {b}: {['%+.4f' % x for x in p]}")

    # ---------------- Q3 ----------------
    print()
    print("=" * 100)
    print("Q3  free readout: at dare_density 0.1 the TIES trim is inert, so any")
    print("    advantage over TA that survives is the SIGN ELECTION's doing")
    print("=" * 100)
    print(f"{'base':<13}{'TA':>10}{'TIES':>10}{'DT d=0.1':>11}"
          f"{'TIES gain':>11}{'DT01 gain':>11}{'retained':>10}")
    q3rows = []
    for b in BASES:
        ta = [ta_ref(b, c) for c in COHORTS if ta_ref(b, c) is not None]
        ti = [ties_ref(b, c) for c in COHORTS if ties_ref(b, c) is not None]
        d1 = [dt(b, "0p1", c) for c in COHORTS if dt(b, "0p1", c) is not None]
        if not (ta and ti and d1):
            print(f"{b:<13} incomplete")
            continue
        mta, mti, md1 = statistics.fmean(ta), statistics.fmean(ti), statistics.fmean(d1)
        g_ties, g_d1 = mta - mti, mta - md1
        ret = (g_d1 / g_ties) if abs(g_ties) > 1e-9 else float("nan")
        print(f"{b:<13}{mta:10.4f}{mti:10.4f}{md1:11.4f}"
              f"{g_ties:+11.4f}{g_d1:+11.4f}{ret:10.2f}")
        q3rows.append({"base": b, "ta": mta, "ties": mti, "dt_d01": md1,
                       "ties_gain_over_ta": g_ties, "dt01_gain_over_ta": g_d1,
                       "fraction_retained": ret})
    print("  retained = (TA - DARE-TIES@0.1) / (TA - TIES); near 1 means the trim")
    print("  was never the point, near 0 means the trim was doing the work.")

    print()
    print("=" * 100)
    print(f"Q1 {v1}   |   Q2 {v2}")
    print("n = 3 cohorts, init draw only. ties_density pinned 0.2 in both arms.")
    print("=" * 100)

    out = RES / "dare_ties_summary.json"
    out.write_text(json.dumps(
        {"q1": {"helps": helps, "neutral": neutral, "hurts": hurts,
                "verdict": v1, "primary_density": 0.2, "rows": rows},
         "q2": {"verdict": v2, "monotone_bases": mono, "rows": q2rows},
         "q3": {"rows": q3rows},
         "thresholds": {"tie": TIE, "gate_sd": GATE_SD},
         "prereg": "notes/prereg_dare_ties_2026-08-04.md"}, indent=2))
    print(f"[dt] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

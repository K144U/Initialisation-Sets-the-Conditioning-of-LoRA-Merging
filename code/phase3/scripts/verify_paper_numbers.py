#!/usr/bin/env python3
"""Check the numbers in the paper's tables against the result files.

A referee found two entries that looked like transcription errors by eye. Both
turned out to be genuine, but eye-checking a paper with several hundred printed
numbers does not scale and is not the standard this paper asks to be held to.

This checks the tables whose entries map onto a single result file each. It
deliberately does not try to cover every table: several are aggregates whose
recomputation belongs in their own analyzer, and a check that silently skips
those while appearing comprehensive would be worse than none. What it does not
cover, it names.

Usage:
  python code/phase3/scripts/verify_paper_numbers.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT",
                           Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
SEC = ROOT / "paper" / "sections"

TOL = 5e-5          # printed to four decimals, so half a unit in the last place


def cell(d: str, name: str, key: str = "worst_task_excess"):
    p = RES / d / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(key)


def geom(cohort: str, base: str, key: str):
    p = RES / f"subspace_geometry_{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(base, {}).get(key)


CHECKS: list[tuple[str, float, object]] = []


def add(label: str, printed: float, actual):
    CHECKS.append((label, printed, actual))


# --- Table: subspace geometry (tab:geometry), shared and indep1.
for base, cos_s, cos_i in [
    ("llama31_8b", 0.9948, 0.0473),
    ("mistral_7b", 0.9947, 0.0468),
    ("qwen25_7b", 0.9964, 0.0500),
    ("yi15_9b", 0.9973, 0.0467),
]:
    add(f"tab:geometry {base} shared cos", cos_s,
        geom("seed1", base, "median_principal_cosine"))
    add(f"tab:geometry {base} indep cos", cos_i,
        geom("indep1", base, "median_principal_cosine"))

# --- Appendix table: between-adapter distance on the shared cohort.
for base, d in [("llama31_8b", 0.1677), ("mistral_7b", 0.1585),
                ("qwen25_7b", 0.1910), ("yi15_9b", 0.1626)]:
    add(f"C.3 {base} shared |Ai-Aj|/|Ai|", d,
        geom("seed1", base, "A_rel_distance"))

# --- Table: the untruncated sweep (tab:untruncated), lambda = 0.
for base, sh, ind in [
    ("llama31_8b", 7.4266, 0.0635),
    ("mistral_7b", 11.6283, 0.1582),
    ("qwen25_7b", 0.8993, 0.0256),
    ("yi15_9b", 3.1294, 0.1227),
]:
    add(f"tab:untruncated {base} shared", sh,
        cell("eval_ridge_untrunc", f"{base}__rdu_l0__seed1"))
    add(f"tab:untruncated {base} indep", ind,
        cell("eval_ridge_untrunc", f"{base}__rdu_l0__indep1"))

# --- Table: RegMean (tab:regmean), the minimally regularised reference.
for base, sh, ind in [
    ("llama31_8b", 1.9789, 0.0708),
    ("mistral_7b", 10.6283, 0.1499),
    ("qwen25_7b", 0.2376, 0.0256),
    ("yi15_9b", 0.5700, 0.1203),
]:
    add(f"tab:regmean {base} shared 1e-6", sh,
        cell("eval_regmean_cond", f"{base}__rm_l1em6__seed1"))
    add(f"tab:regmean {base} indep 1e-6", ind,
        cell("eval_regmean_cond", f"{base}__rm_l1em6__indep1"))

# --- Table: heuristics in accuracy (tab:heur-downstream), a sample of rows.
p = RES / "heur_downstream_summary.json"
if p.exists():
    rows = {(r["base"], r["method"], r["bench"]): r
            for r in json.loads(p.read_text())["rows"]}
    for (b, m, bench), printed in [
        (("llama31_8b", "ties", "humaneval"), -0.104),
        (("mistral_7b", "tvq_b2", "humaneval"), +0.067),
        (("qwen25_7b", "knots_ties", "humaneval"), +0.055),
        (("yi15_9b", "knots_ties", "gsm8k"), -0.016),
    ]:
        r = rows.get((b, m, bench))
        add(f"tab:heur-downstream {b} {m} {bench} d", printed,
            None if r is None else round(r["d"], 3))


def main() -> int:
    bad = skipped = 0
    for label, printed, actual in CHECKS:
        if actual is None:
            print(f"  SKIP  {label:<48} (result file not present locally)")
            skipped += 1
            continue
        if abs(float(actual) - printed) > TOL:
            print(f"  FAIL  {label:<48} paper {printed}  file {float(actual):.6f}")
            bad += 1
        else:
            print(f"  ok    {label:<48} {printed}")

    print(f"\n{len(CHECKS) - bad - skipped} checked, {bad} mismatched, "
          f"{skipped} skipped")
    print("\nNot covered here, and why:")
    print("  tab:q1, tab:ridge      paired differences and gates; recomputed by"
          " their own analyzers")
    print("  tab:conditioning       aggregates over layers, not one file per"
          " cell")
    print("  tab:prevalence         the audit's own summary is the source")
    print("  tab:w3                 asymptote-matched residuals; see"
          " Appendix E")
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

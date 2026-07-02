#!/usr/bin/env python3
"""Analyze Experiment B: RegMean ridge_lambda sweep.

For each base, find the lambda that minimizes RegMean worst-task NLL
excess, and compare RegMean's *best* lambda against rd-encoder ridge at
its base-optimal lambda. Directly tests the paper's claim that the
RegMean comparison conflated the rate-distortion centroid with the ridge
regularizer: if RegMean's best lambda still loses to rd-ridge on all 4
bases, the claim is DEFENDED; if RegMean's best lambda beats rd-ridge on
any base, the claim must be RETRACTED / softened.

Reads (all seed1, C4 cohort):
  lambda=1e-3 -> results/phase3/eval_e12_regmean_adamerging/{base}__regmean__seed1.json  (reused)
  other lam   -> results/phase3/eval_regmean_lambda/{base}__regmean__{ltag}__seed1.json
  rd-ridge    -> base-optimal lambda files (eval_ridge / eval_ridge_xmodel)

Writes results/phase3/regmean_lambda_summary.json + prints a verdict table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("/home/sanjay.g/projects/rdmerge")
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
# (lambda, source); 1e-3 reused from the e12 RegMean cells, not re-run.
LAMS = [(1e-4, "sweep"), (1e-3, "e12"), (1e-2, "sweep"),
        (1e-1, "sweep"), (1.0, "sweep"), (10.0, "sweep")]

RD_RIDGE_FILES = {
    "llama31_8b": RES / "eval_ridge/llama31_8b__ridge_l0p05.json",
    "mistral_7b": RES / "eval_ridge_xmodel/mistral_7b__ridge_l0p13.json",
    "qwen25_7b":  RES / "eval_ridge_xmodel/qwen25_7b__ridge_l0p13.json",
    "yi15_9b":    RES / "eval_ridge_xmodel/yi15_9b__ridge_l0p13.json",
}


def ltag(l: float) -> str:
    e = round(math.log10(l))
    return f"lam1e{'m' if e < 0 else ''}{abs(e)}"


def load_worst(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("worst_task_excess")
    except Exception:
        return None


def lam_path(base: str, l: float, src: str) -> Path:
    if src == "e12":
        return RES / f"eval_e12_regmean_adamerging/{base}__regmean__seed1.json"
    return RES / f"eval_regmean_lambda/{base}__regmean__{ltag(l)}__seed1.json"


def main() -> int:
    summary: dict[str, dict] = {}
    header = f"{'base':12s} " + " ".join(f"{('l=%g' % l):>8}" for l, _ in LAMS)
    print(header + "   best(l)      rd-ridge  verdict")
    all_present = True
    retract_bases = []
    for base in BASES:
        rd = load_worst(RD_RIDGE_FILES[base])
        vals = {l: load_worst(lam_path(base, l, src)) for l, src in LAMS}
        present = {l: v for l, v in vals.items() if v is not None}
        if len(present) < len(LAMS):
            all_present = False
        best_l = min(present, key=present.get) if present else None
        best_v = present[best_l] if best_l is not None else None
        if best_v is not None and rd is not None:
            if rd < best_v:
                verdict = "rd-ridge wins"
            else:
                verdict = f"** RegMean@l={best_l:g} BEATS rd-ridge **"
                retract_bases.append(base)
        else:
            verdict = "pending"
        cells = " ".join((f"{vals[l]:.4f}" if vals[l] is not None else " ....  ") for l, _ in LAMS)
        bs = f"{best_v:.4f}@{best_l:g}" if best_v is not None else "  ...."
        rds = f"{rd:.4f}" if rd is not None else "...."
        print(f"{base:12s} {cells}  {bs:>11s}  {rds:>7s}   {verdict}")
        summary[base] = {
            "by_lambda": {str(k): v for k, v in vals.items()},
            "best_lambda": best_l,
            "best_worst_excess": best_v,
            "rd_ridge": rd,
            "rd_ridge_wins": (rd < best_v) if (rd is not None and best_v is not None) else None,
        }
    out = RES / "regmean_lambda_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[analyze] wrote {out}  (all_cells_present={all_present})")
    if all_present:
        if not retract_bases:
            print("[analyze] VERDICT: centroid claim DEFENDED -- rd-ridge beats "
                  "best-lambda RegMean on all 4 bases.")
        else:
            print(f"[analyze] VERDICT: RETRACT/SOFTEN -- best-lambda RegMean beats "
                  f"rd-ridge on: {retract_bases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze Experiment A: TIES/DARE density sweep.

For each base x method, find the density that minimizes worst-task NLL
excess, and compare that *best* density against rd-encoder ridge at its
base-optimal lambda. Tests whether rd-ridge's win over TIES/DARE is an
artifact of the fixed density=0.2 used in the main matrix (seed1).

Reads (all seed1, C4 cohort, identical eval pipeline):
  density=0.2  -> results/phase3/eval_matrix_seeds/{base}__{method}__seed1.json   (reused)
  other dens   -> results/phase3/eval_density_sweep/{base}__{method}__{dtag}__seed1.json
  rd-ridge     -> base-optimal lambda files (eval_ridge / eval_ridge_xmodel)

Writes results/phase3/density_sweep_summary.json + prints a verdict table.
Safe to run mid-job: missing cells show as '....' and the verdict is
'pending' until every density is present.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/sanjay.g/projects/rdmerge")
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
METHODS = ["ties", "dare"]
# (density, source); 0.2 is reused from the main matrix, not re-run.
DENS = [(0.05, "sweep"), (0.1, "sweep"), (0.2, "matrix"), (0.3, "sweep"), (0.5, "sweep")]

RD_RIDGE_FILES = {
    "llama31_8b": RES / "eval_ridge/llama31_8b__ridge_l0p05.json",
    "mistral_7b": RES / "eval_ridge_xmodel/mistral_7b__ridge_l0p13.json",
    "qwen25_7b":  RES / "eval_ridge_xmodel/qwen25_7b__ridge_l0p13.json",
    "yi15_9b":    RES / "eval_ridge_xmodel/yi15_9b__ridge_l0p13.json",
}


def dtag(d: float) -> str:
    return "d" + str(d).replace(".", "p")


def load_worst(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("worst_task_excess")
    except Exception:
        return None


def density_path(base: str, method: str, d: float, src: str) -> Path:
    if src == "matrix":
        return RES / f"eval_matrix_seeds/{base}__{method}__seed1.json"
    return RES / f"eval_density_sweep/{base}__{method}__{dtag(d)}__seed1.json"


def main() -> int:
    summary: dict[str, dict] = {}
    header = f"{'base':12s} {'method':6s} " + " ".join(f"{d:>6}" for d, _ in DENS)
    print(header + "    best(d)     rd-ridge  verdict")
    all_present = True
    for base in BASES:
        rd = load_worst(RD_RIDGE_FILES[base])
        for method in METHODS:
            vals = {d: load_worst(density_path(base, method, d, src)) for d, src in DENS}
            present = {d: v for d, v in vals.items() if v is not None}
            if len(present) < len(DENS):
                all_present = False
            best_d = min(present, key=present.get) if present else None
            best_v = present[best_d] if best_d is not None else None
            if best_v is not None and rd is not None:
                verdict = "rd-ridge wins" if rd < best_v else f"** {method}@d={best_d} BEATS rd-ridge **"
            else:
                verdict = "pending"
            cells = " ".join((f"{vals[d]:.4f}" if vals[d] is not None else " ....  ") for d, _ in DENS)
            bs = f"{best_v:.4f}@{best_d}" if best_v is not None else "  ...."
            rds = f"{rd:.4f}" if rd is not None else "...."
            print(f"{base:12s} {method:6s} {cells}  {bs:>11s}  {rds:>7s}   {verdict}")
            summary[f"{base}__{method}"] = {
                "by_density": vals,
                "best_density": best_d,
                "best_worst_excess": best_v,
                "rd_ridge": rd,
                "rd_ridge_wins": (rd < best_v) if (rd is not None and best_v is not None) else None,
            }
    out = RES / "density_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[analyze] wrote {out}  (all_cells_present={all_present})")
    if all_present:
        wins = sum(1 for v in summary.values() if v["rd_ridge_wins"])
        print(f"[analyze] rd-ridge beats best-density baseline in {wins}/{len(summary)} (base x method) cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

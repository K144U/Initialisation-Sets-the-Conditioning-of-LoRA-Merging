"""E6 pilot — T-scaling analysis on Yi-1.5-9B real adapters.

Inputs: results/phase3/eval_e6_pilot/yi15_9b__T{2,4,7}_{nested,rand0,rand1,rand2,all}__{ta,ties,dare,knots,tvq_b2,rd_ridge}.json
        (54 cells = 9 subsets x 6 methods)

Outputs:
  results/phase3/e6_T_scaling_summary.csv  — tidy long-form table (one row per cell)
  results/phase3/e6_T_scaling_summary.json — machine-readable verdict
  prints summary tables + log-T slope per method to stdout

Pre-registered claims from master_plan §E6 + decisions.md:
  (V1) Worst-task NLL excess vs T grows with T for every method (no method
       is T-flat on real Yi adapters).
  (V2) Method ordering at T=7: TIES <= rd_ridge < TVQ_b2 < TA ~ DARE < KnOTS
       (the T=4 ordering of §6.1 persists at T=7).
  (V3) Log-T slope: per-method excess fits y = a + b * log(T). E4's synthetic
       prediction is b ~ 1/sqrt(r) ~ 0.25 at r=16. We report measured b per
       method and flag any method whose slope departs sharply.
  (V4) rd_ridge achievability ratio: rd_ridge worst-excess at each T,
       relative to TA worst-excess at that T. Confirms or contradicts §6.2's
       ridge salvage at T > 4.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
EVAL_DIR = PROJECT_ROOT / "results/phase3/eval_e6_pilot"
OUT_CSV = PROJECT_ROOT / "results/phase3/e6_T_scaling_summary.csv"
OUT_JSON = PROJECT_ROOT / "results/phase3/e6_T_scaling_summary.json"

METHODS = ["ta", "ties", "dare", "knots", "tvq_b2", "rd_ridge"]
T_VALUES = [2, 4, 7]
# Subsets at each T (matches manifest):
SUBSETS = {
    2: ["nested", "rand0", "rand1", "rand2"],
    4: ["nested", "rand0", "rand1", "rand2"],
    7: ["all"],
}


def parse_filename(p: Path) -> tuple[int, str, str] | None:
    """Parse `yi15_9b__T<N>_<subset>__<method>.json` into (T, subset, method)."""
    stem = p.stem
    parts = stem.split("__")
    if len(parts) != 3 or parts[0] != "yi15_9b":
        return None
    t_subset = parts[1]
    method = parts[2]
    if not t_subset.startswith("T"):
        return None
    # split T<N>_<subset>
    try:
        i = 1
        while i < len(t_subset) and t_subset[i].isdigit():
            i += 1
        T = int(t_subset[1:i])
        if i >= len(t_subset) or t_subset[i] != "_":
            return None
        subset = t_subset[i + 1:]
    except (ValueError, IndexError):
        return None
    return T, subset, method


def load_cell(p: Path) -> dict:
    d = json.load(open(p))
    return {
        "worst_task_excess": d["worst_task_excess"],
        "avg_task_excess": d["avg_task_excess"],
        "per_task_excess": d["excess_per_task"],
        "tasks": list(d["excess_per_task"].keys()),
        "method": d["method"],
    }


def build_table() -> list[dict]:
    rows: list[dict] = []
    files = sorted(EVAL_DIR.glob("yi15_9b__T*.json"))
    for p in files:
        info = parse_filename(p)
        if info is None:
            print(f"[warn] could not parse {p.name}", file=sys.stderr)
            continue
        T, subset, method = info
        cell = load_cell(p)
        rows.append({
            "T": T,
            "subset": subset,
            "method": method,
            "worst_task_excess": cell["worst_task_excess"],
            "avg_task_excess": cell["avg_task_excess"],
            "n_tasks_seen": len(cell["tasks"]),
        })
    return rows


def aggregate_by_T_method(rows: list[dict]
                          ) -> dict[tuple[int, str], dict]:
    """Mean / min / max worst-excess across subsets at each (T, method)."""
    bucket: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        bucket.setdefault((r["T"], r["method"]), []).append(r)
    out: dict[tuple[int, str], dict] = {}
    for key, group in bucket.items():
        ws = [g["worst_task_excess"] for g in group]
        avs = [g["avg_task_excess"] for g in group]
        out[key] = {
            "n_subsets": len(group),
            "worst_mean": sum(ws) / len(ws),
            "worst_min": min(ws),
            "worst_max": max(ws),
            "worst_range": max(ws) - min(ws),
            "avg_mean": sum(avs) / len(avs),
            "subsets": sorted(g["subset"] for g in group),
        }
    return out


def fit_log_t_slope(agg: dict[tuple[int, str], dict],
                    method: str) -> dict:
    """Linear regression of worst_mean against log(T) for `method`.
    Returns slope (b), intercept (a), R^2, and per-T points."""
    xs: list[float] = []
    ys: list[float] = []
    points: list[dict] = []
    for T in T_VALUES:
        s = agg.get((T, method))
        if s is None:
            continue
        x = math.log(T)
        y = s["worst_mean"]
        xs.append(x)
        ys.append(y)
        points.append({"T": T, "log_T": x, "worst_mean": y})
    if len(xs) < 2:
        return {"slope": None, "intercept": None, "r_squared": None,
                "points": points}
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if den == 0:
        return {"slope": None, "intercept": None, "r_squared": None,
                "points": points}
    b = num / den
    a = my - b * mx
    # R^2
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": b, "intercept": a, "r_squared": r2, "points": points}


def check_grows_with_T(agg: dict[tuple[int, str], dict]) -> dict:
    """V1: for each method, does worst_mean(T=2) < worst_mean(T=4) < worst_mean(T=7)?"""
    out: dict[str, dict] = {}
    for m in METHODS:
        seq = [agg.get((T, m), {}).get("worst_mean") for T in T_VALUES]
        monotone = all(seq[i] is not None and seq[i + 1] is not None
                       and seq[i] < seq[i + 1]
                       for i in range(len(seq) - 1))
        out[m] = {"T2": seq[0], "T4": seq[1], "T7": seq[2],
                  "monotone_up": monotone}
    n_hold = sum(v["monotone_up"] for v in out.values())
    return {"per_method": out, "n_hold": n_hold, "n_total": len(METHODS)}


def check_T7_ordering(agg: dict[tuple[int, str], dict]) -> dict:
    """V2: at T=7, methods sorted by worst_mean ascending."""
    items = [(m, agg.get((7, m), {}).get("worst_mean")) for m in METHODS]
    items = [(m, v) for m, v in items if v is not None]
    items.sort(key=lambda x: x[1])
    return {"sorted_ascending": [{"method": m, "worst_mean": v} for m, v in items]}


def check_ridge_salvage(agg: dict[tuple[int, str], dict]) -> dict:
    """V4: rd_ridge worst-excess vs TA worst-excess at each T."""
    out: dict[int, dict] = {}
    for T in T_VALUES:
        ridge = agg.get((T, "rd_ridge"), {}).get("worst_mean")
        ta = agg.get((T, "ta"), {}).get("worst_mean")
        if ridge is None or ta is None:
            continue
        out[T] = {
            "rd_ridge": ridge,
            "ta": ta,
            "ratio_ridge_over_ta": ridge / ta if ta > 0 else None,
            "salvage": ridge < ta,
        }
    return {"per_T": out}


def write_csv(rows: list[dict], agg: dict[tuple[int, str], dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["T", "method", "n_subsets",
              "worst_mean", "worst_min", "worst_max", "worst_range",
              "avg_mean"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (T, method) in sorted(agg.keys()):
            s = agg[(T, method)]
            w.writerow({
                "T": T, "method": method, "n_subsets": s["n_subsets"],
                "worst_mean": round(s["worst_mean"], 4),
                "worst_min": round(s["worst_min"], 4),
                "worst_max": round(s["worst_max"], 4),
                "worst_range": round(s["worst_range"], 4),
                "avg_mean": round(s["avg_mean"], 4),
            })


def print_summary(rows: list[dict],
                  agg: dict[tuple[int, str], dict],
                  slopes: dict[str, dict],
                  v1: dict, v2: dict, v4: dict) -> None:
    print(f"\nLoaded {len(rows)} cells.")
    print(f"T values: {sorted({r['T'] for r in rows})}")
    print(f"Methods: {sorted({r['method'] for r in rows})}\n")

    # main table
    print("=" * 84)
    print(f"{'T':<4}{'method':<14}{'n_sub':<6}{'worst_mean':<13}"
          f"{'worst_range':<13}{'avg_mean':<10}")
    print("-" * 84)
    for (T, method) in sorted(agg.keys()):
        s = agg[(T, method)]
        print(f"{T:<4}{method:<14}{s['n_subsets']:<6}"
              f"{s['worst_mean']:<13.4f}"
              f"{s['worst_range']:<13.4f}"
              f"{s['avg_mean']:<10.4f}")
    print("=" * 84)

    # V1
    print("\n[V1] Worst-excess monotone increasing in T:")
    for m, v in v1["per_method"].items():
        print(f"  {m:<10} T2={v['T2']!s:<10} T4={v['T4']!s:<10} "
              f"T7={v['T7']!s:<10} monotone={v['monotone_up']}")
    print(f"  HOLDS in {v1['n_hold']}/{v1['n_total']} methods")

    # V2
    print("\n[V2] T=7 method ordering (worst-excess ascending):")
    for i, item in enumerate(v2["sorted_ascending"]):
        print(f"  {i + 1}. {item['method']:<10} {item['worst_mean']:.4f}")

    # V3 — slopes
    print("\n[V3] Log-T slopes (y = a + b * log(T)):")
    print(f"  E4 synthetic prediction: b ~ 1/sqrt(r=16) ~ 0.25")
    print(f"  {'method':<10}{'slope b':<12}{'intercept':<12}{'R^2':<8}")
    for m in METHODS:
        s = slopes.get(m, {})
        if s.get("slope") is None:
            print(f"  {m:<10}{'NA':<12}{'NA':<12}{'NA':<8}")
        else:
            print(f"  {m:<10}{s['slope']:<12.4f}{s['intercept']:<12.4f}"
                  f"{s['r_squared']:<8.4f}")

    # V4 — ridge salvage
    print("\n[V4] rd_ridge vs TA at each T (ratio<1 = ridge wins):")
    for T, v in v4["per_T"].items():
        r = v["ratio_ridge_over_ta"]
        r_str = f"{r:.3f}" if r is not None else "NA"
        print(f"  T={T}  rd_ridge={v['rd_ridge']:.4f}  "
              f"ta={v['ta']:.4f}  ratio={r_str}  salvage={v['salvage']}")

    print(f"\nWrote summary to:\n  {OUT_CSV}\n  {OUT_JSON}")


def main() -> int:
    rows = build_table()
    if not rows:
        print(f"No cells found in {EVAL_DIR}", file=sys.stderr)
        return 1
    agg = aggregate_by_T_method(rows)
    slopes = {m: fit_log_t_slope(agg, m) for m in METHODS}
    v1 = check_grows_with_T(agg)
    v2 = check_T7_ordering(agg)
    v4 = check_ridge_salvage(agg)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cells": len(rows),
        "T_values": T_VALUES,
        "methods": METHODS,
        "agg": {f"T{T}__{m}": s for (T, m), s in agg.items()},
        "slopes": slopes,
        "claims": {
            "V1_grows_with_T": v1,
            "V2_T7_ordering": v2,
            "V4_ridge_salvage": v4,
        },
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    write_csv(rows, agg)
    print_summary(rows, agg, slopes, v1, v2, v4)
    return 0


if __name__ == "__main__":
    sys.exit(main())

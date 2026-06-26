"""Mistral-7B T-scaling analysis — Td2 pre-registration verdict.

Inputs: results/phase3/eval_mistral_t7/mistral_7b__T{2,4,7}_{nested,rand0,rand1,rand2,all}__{ta,ties,dare,knots,tvq_b2,rd_ridge}.json
        (54 cells = 9 subsets x 6 methods)

Outputs:
  results/phase3/mistral_t7_summary.csv  — tidy long-form table (one row per (T, method))
  results/phase3/mistral_t7_summary.json — machine-readable verdict bundle
  prints summary tables + log-T slope per method + Td2 verdict to stdout

Mirrors analyze_e6_T_scaling.py V1-V4 claims; adds V5 Td2.

V5 Td2 PRE-REGISTRATION (commit 3582799, 2026-06-25):
  Prediction (locked, not to be edited): "Mistral-7B-v0.3 at T=7 will show
  TIES neither clearly inverting to worst-method (as on Yi-Chat at R=0.077)
  nor clearly winning (as on Llama-Instruct at R=0.028)."

  Operationalization:
    - "Clearly worst" = TIES last in T=7 ranking AND mean > TA + 0.04 nats
    - "Clearly best"  = TIES first in T=7 ranking AND
                        mean < second-best - 0.02 nats
    - "Ambiguous"     = neither

  Falsification: any "clearly worst" or "clearly best" verdict falsifies
  the Td2 threshold model as written and forces a revision. Report honestly
  --- do not retrofit thresholds to data.

Empirical anchor (verified at commit 537796a): Mistral T=4 sign-election
win-share range R = 0.0326, sitting inside the Td2 perturbation analysis's
predicted ambiguity window R* in [0.025, 0.075]; distance 0.008 to lower
edge, 0.043 to upper.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
EVAL_DIR = PROJECT_ROOT / "results/phase3/eval_mistral_t7"
OUT_CSV = PROJECT_ROOT / "results/phase3/mistral_t7_summary.csv"
OUT_JSON = PROJECT_ROOT / "results/phase3/mistral_t7_summary.json"
BASE_PREFIX = "mistral_7b"

METHODS = ["ta", "ties", "dare", "knots", "tvq_b2", "rd_ridge"]
T_VALUES = [2, 4, 7]
SUBSETS = {
    2: ["nested", "rand0", "rand1", "rand2"],
    4: ["nested", "rand0", "rand1", "rand2"],
    7: ["all"],
}

TD2_TIES_GAP_WORST = 0.04   # nats; TIES - TA > this => "clearly worst"
TD2_TIES_GAP_BEST = 0.02    # nats; second_best - TIES > this => "clearly best"


def parse_filename(p: Path) -> tuple[int, str, str] | None:
    """Parse `<base>__T<N>_<subset>__<method>.json` into (T, subset, method)."""
    stem = p.stem
    parts = stem.split("__")
    if len(parts) != 3 or parts[0] != BASE_PREFIX:
        return None
    t_subset = parts[1]
    method = parts[2]
    if not t_subset.startswith("T"):
        return None
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
    files = sorted(EVAL_DIR.glob(f"{BASE_PREFIX}__T*.json"))
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
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": b, "intercept": a, "r_squared": r2, "points": points}


def check_grows_with_T(agg: dict[tuple[int, str], dict]) -> dict:
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
    items = [(m, agg.get((7, m), {}).get("worst_mean")) for m in METHODS]
    items = [(m, v) for m, v in items if v is not None]
    items.sort(key=lambda x: x[1])
    return {"sorted_ascending": [{"method": m, "worst_mean": v}
                                 for m, v in items]}


def check_ridge_salvage(agg: dict[tuple[int, str], dict]) -> dict:
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


def check_td2_verdict(agg: dict[tuple[int, str], dict]) -> dict:
    """V5 Td2 pre-registered three-way bucket at T=7.

    Returns a verdict dict including the bucket
    ('clearly_worst' | 'clearly_best' | 'ambiguous'), the TIES rank,
    measured gaps, and a falsifies-Td2 flag.
    """
    ranking = [(m, agg.get((7, m), {}).get("worst_mean")) for m in METHODS]
    ranking = [(m, v) for m, v in ranking if v is not None]
    if not ranking or "ties" not in {m for m, _ in ranking}:
        return {"verdict": None, "reason": "missing T=7 TIES cell"}

    ranking.sort(key=lambda x: x[1])  # ascending: best=lowest excess
    methods_sorted = [m for m, _ in ranking]
    means = {m: v for m, v in ranking}

    ties_rank = methods_sorted.index("ties") + 1  # 1-based
    ties_first = methods_sorted[0] == "ties"
    ties_last = methods_sorted[-1] == "ties"

    ta_gap = means["ties"] - means.get("ta", float("nan"))
    second_best_gap = (
        means[methods_sorted[1]] - means["ties"] if ties_first else None
    )

    clearly_worst = ties_last and ta_gap > TD2_TIES_GAP_WORST
    clearly_best = (
        ties_first
        and second_best_gap is not None
        and second_best_gap > TD2_TIES_GAP_BEST
    )

    if clearly_worst:
        bucket = "clearly_worst"
    elif clearly_best:
        bucket = "clearly_best"
    else:
        bucket = "ambiguous"

    # Pre-registered prediction = "ambiguous". Anything else falsifies.
    falsifies_td2 = bucket != "ambiguous"

    return {
        "verdict": bucket,
        "ties_rank": ties_rank,
        "ranking_T7": [{"method": m, "worst_mean": v} for m, v in ranking],
        "ties_minus_ta_nats": ta_gap,
        "second_best_minus_ties_nats": second_best_gap,
        "thresholds": {
            "ties_worst_if_ties_minus_ta_gt": TD2_TIES_GAP_WORST,
            "ties_best_if_second_minus_ties_gt": TD2_TIES_GAP_BEST,
        },
        "prediction": "ambiguous (commit 3582799)",
        "falsifies_td2": falsifies_td2,
        "anchor_T4_R": 0.0326,  # from commit 537796a
        "anchor_window_R_star": [0.025, 0.075],
    }


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
                  v1: dict, v2: dict, v4: dict, v5: dict) -> None:
    print(f"\nLoaded {len(rows)} cells from {EVAL_DIR.name}.")
    print(f"T values: {sorted({r['T'] for r in rows})}")
    print(f"Methods: {sorted({r['method'] for r in rows})}\n")

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

    print("\n[V1] Worst-excess monotone increasing in T:")
    for m, v in v1["per_method"].items():
        print(f"  {m:<10} T2={v['T2']!s:<10} T4={v['T4']!s:<10} "
              f"T7={v['T7']!s:<10} monotone={v['monotone_up']}")
    print(f"  HOLDS in {v1['n_hold']}/{v1['n_total']} methods")

    print("\n[V2] T=7 method ordering (worst-excess ascending):")
    for i, item in enumerate(v2["sorted_ascending"]):
        print(f"  {i + 1}. {item['method']:<10} {item['worst_mean']:.4f}")

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

    print("\n[V4] rd_ridge vs TA at each T (ratio<1 = ridge wins):")
    for T, v in v4["per_T"].items():
        r = v["ratio_ridge_over_ta"]
        r_str = f"{r:.3f}" if r is not None else "NA"
        print(f"  T={T}  rd_ridge={v['rd_ridge']:.4f}  "
              f"ta={v['ta']:.4f}  ratio={r_str}  salvage={v['salvage']}")

    # V5 Td2 verdict --- the headline result
    print("\n" + "#" * 84)
    print("[V5] Td2 PRE-REGISTERED VERDICT (commit 3582799)")
    print("#" * 84)
    if v5.get("verdict") is None:
        print(f"  CANNOT DECIDE: {v5.get('reason', 'unknown')}")
    else:
        print(f"  Prediction (locked): {v5['prediction']}")
        print(f"  T=7 ranking (worst-excess ascending):")
        for i, item in enumerate(v5["ranking_T7"]):
            marker = " <-- TIES" if item["method"] == "ties" else ""
            print(f"    {i + 1}. {item['method']:<10} "
                  f"{item['worst_mean']:.4f}{marker}")
        print(f"  TIES rank: {v5['ties_rank']}/{len(v5['ranking_T7'])}")
        gap_ta = v5["ties_minus_ta_nats"]
        print(f"  TIES - TA gap: {gap_ta:+.4f} nats "
              f"(clearly-worst threshold: >{TD2_TIES_GAP_WORST})")
        sb = v5["second_best_minus_ties_nats"]
        if sb is not None:
            print(f"  Second-best - TIES gap: {sb:+.4f} nats "
                  f"(clearly-best threshold: >{TD2_TIES_GAP_BEST})")
        print(f"  VERDICT: {v5['verdict'].upper()}")
        if v5["falsifies_td2"]:
            print(f"  ==> FALSIFIES Td2 prediction. Honest revision required.")
        else:
            print(f"  ==> CONFIRMS Td2 prediction. Promote bracket fit to "
                  f"confirmed prediction in §6.6/§6.7/Td2 appendix.")

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
    v5 = check_td2_verdict(agg)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cells": len(rows),
        "T_values": T_VALUES,
        "methods": METHODS,
        "base": BASE_PREFIX,
        "agg": {f"T{T}__{m}": s for (T, m), s in agg.items()},
        "slopes": slopes,
        "claims": {
            "V1_grows_with_T": v1,
            "V2_T7_ordering": v2,
            "V4_ridge_salvage": v4,
            "V5_td2_verdict": v5,
        },
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    write_csv(rows, agg)
    print_summary(rows, agg, slopes, v1, v2, v4, v5)
    return 0


if __name__ == "__main__":
    sys.exit(main())

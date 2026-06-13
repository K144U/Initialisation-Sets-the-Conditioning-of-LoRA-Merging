"""E2 multi-seed merge-matrix analysis (Phase 3, §6.1 calibration data).

Inputs: results/phase3/eval_matrix_seeds/*.json (80 cells = 4 models x
10 methods x 2 seeds). The seed-0 reference matrix lives in
results/phase3/eval_matrix_n1k_v3_perexample/ (loaded for cross-seed
comparison if present).

Outputs:
  results/phase3/e2_analysis_summary.csv  — tidy long-form table
  results/phase3/e2_analysis_summary.json — same as JSON for plotting
  prints summary tables to stdout

Pre-registered claims verified (per master_plan §E2 + decisions.md):
  (C1) Cross-model ordering on worst-task excess for TA:
       Llama > Mistral > Qwen > Yi   (Yi merges best, Llama worst)
       Test: does the ordering hold for all available seeds and on the mean?
  (C2) b=2 TVQ "less is more" dip: worst-excess(TVQ b=2) < worst-excess(TVQ b=4)
       across models and seeds.
  (C3) DARE tracks TA per-seed to 3 decimals (anchor sanity).
  (C4) Seed range for any (model, method) is small relative to method spread.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
MATRIX_DIR = PROJECT_ROOT / "results/phase3/eval_matrix_seeds"
SEED0_DIR = PROJECT_ROOT / "results/phase3/eval_matrix_n1k_v3_perexample"
OUT_CSV = PROJECT_ROOT / "results/phase3/e2_analysis_summary.csv"
OUT_JSON = PROJECT_ROOT / "results/phase3/e2_analysis_summary.json"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
# Canonical method order (matches manifest)
METHODS = ["task_arithmetic", "ties", "dare", "knots",
           "tvq_b1", "tvq_b2", "tvq_b4", "tvq_b8", "tvq_b16", "tvq_b32"]


def parse_filename(p: Path) -> tuple[str, str, int] | None:
    """Parse `<model>__<method>__seed<N>.json` into (model, method, seed)."""
    stem = p.stem  # strip .json
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    model, method, seed_tok = parts
    if not seed_tok.startswith("seed"):
        return None
    try:
        seed = int(seed_tok[4:])
    except ValueError:
        return None
    return model, method, seed


def load_cell(p: Path) -> dict:
    d = json.load(open(p))
    excess = d["excess_per_task"]
    return {
        "worst_task_excess": d["worst_task_excess"],
        "avg_task_excess": d["avg_task_excess"],
        "per_task_excess": excess,
        "tasks": list(excess.keys()),
    }


def load_seed0_reference() -> dict[tuple[str, str], float]:
    """If a seed-0 reference matrix exists, load worst-task excess per
    (model, method). Otherwise return empty dict."""
    out: dict[tuple[str, str], float] = {}
    if not SEED0_DIR.exists():
        return out
    for p in sorted(SEED0_DIR.glob("*.json")):
        info = parse_filename(p)
        if info is None:
            # seed-0 reference may use a different naming convention; skip
            continue
        model, method, seed = info
        if seed != 0:
            continue
        try:
            cell = load_cell(p)
        except (KeyError, json.JSONDecodeError):
            continue
        out[(model, method)] = cell["worst_task_excess"]
    return out


def build_table() -> list[dict]:
    """Walk MATRIX_DIR and emit one row per (model, method, seed) cell."""
    rows: list[dict] = []
    files = sorted(MATRIX_DIR.glob("*.json"))
    for p in files:
        info = parse_filename(p)
        if info is None:
            print(f"[warn] could not parse {p.name}", file=sys.stderr)
            continue
        model, method, seed = info
        cell = load_cell(p)
        row = {
            "model": model,
            "method": method,
            "seed": seed,
            "worst_task_excess": cell["worst_task_excess"],
            "avg_task_excess": cell["avg_task_excess"],
        }
        for task, v in cell["per_task_excess"].items():
            row[f"excess_{task}"] = v
        rows.append(row)
    return rows


def aggregate_by_method(rows: list[dict]
                        ) -> dict[tuple[str, str], dict]:
    """Compute mean and seed-range for each (model, method) pair."""
    bucket: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        bucket.setdefault((r["model"], r["method"]), []).append(r)
    out: dict[tuple[str, str], dict] = {}
    for key, group in bucket.items():
        vs = [g["worst_task_excess"] for g in group]
        avgs = [g["avg_task_excess"] for g in group]
        out[key] = {
            "n_seeds": len(group),
            "worst_mean": sum(vs) / len(vs),
            "worst_min": min(vs),
            "worst_max": max(vs),
            "worst_range": max(vs) - min(vs),
            "avg_mean": sum(avgs) / len(avgs),
            "seeds": sorted(g["seed"] for g in group),
        }
    return out


def check_ordering(agg: dict[tuple[str, str], dict],
                   method: str = "task_arithmetic") -> dict:
    """C1: does worst-mean follow Llama > Mistral > Qwen > Yi for `method`?"""
    seq = [agg.get((m, method), {}).get("worst_mean") for m in MODELS]
    holds = all(seq[i] is not None and seq[i + 1] is not None
                and seq[i] > seq[i + 1]
                for i in range(len(seq) - 1))
    return {
        "method": method,
        "expected_order": " > ".join(MODELS),
        "measured": dict(zip(MODELS, seq)),
        "holds": holds,
    }


def check_b2_dip(agg: dict[tuple[str, str], dict]) -> dict:
    """C2: worst(TVQ b=2) < worst(TVQ b=4) per model."""
    out: dict[str, dict] = {}
    for m in MODELS:
        b2 = agg.get((m, "tvq_b2"), {}).get("worst_mean")
        b4 = agg.get((m, "tvq_b4"), {}).get("worst_mean")
        out[m] = {"b2": b2, "b4": b4,
                  "dip_holds": (b2 is not None and b4 is not None
                                and b2 < b4)}
    n_hold = sum(v["dip_holds"] for v in out.values())
    return {"per_model": out, "n_hold": n_hold, "n_total": len(MODELS)}


def check_dare_tracks_ta(rows: list[dict]) -> dict:
    """C3: DARE worst-excess tracks TA worst-excess per (model, seed) to 3 decimals."""
    by_key: dict[tuple[str, int, str], float] = {}
    for r in rows:
        if r["method"] in ("dare", "task_arithmetic"):
            by_key[(r["model"], r["seed"], r["method"])] = r["worst_task_excess"]
    deltas = []
    for m in MODELS:
        for s in (1, 2):
            ta = by_key.get((m, s, "task_arithmetic"))
            dr = by_key.get((m, s, "dare"))
            if ta is None or dr is None:
                continue
            deltas.append({"model": m, "seed": s, "ta": ta, "dare": dr,
                           "abs_diff": abs(ta - dr)})
    max_diff = max((d["abs_diff"] for d in deltas), default=float("inf"))
    return {
        "per_cell": deltas,
        "max_abs_diff": max_diff,
        "holds_to_3_decimals": max_diff < 5e-4,
    }


def check_seed_range_small(agg: dict[tuple[str, str], dict]) -> dict:
    """C4: median seed-range is small relative to median method spread."""
    ranges = [v["worst_range"] for v in agg.values() if v["n_seeds"] >= 2]
    method_means = [v["worst_mean"] for v in agg.values()]
    if not ranges or not method_means:
        return {}
    sorted_ranges = sorted(ranges)
    median_range = sorted_ranges[len(sorted_ranges) // 2]
    spread = max(method_means) - min(method_means)
    return {
        "median_seed_range": median_range,
        "max_seed_range": max(ranges),
        "method_spread": spread,
        "seed_range_over_spread": (median_range / spread
                                   if spread > 0 else float("inf")),
    }


def write_csv(rows: list[dict], agg: dict[tuple[str, str], dict]) -> None:
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "method", "n_seeds",
              "worst_mean", "worst_min", "worst_max", "worst_range",
              "avg_mean"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, method), s in sorted(agg.items()):
            w.writerow({
                "model": model, "method": method, "n_seeds": s["n_seeds"],
                "worst_mean": round(s["worst_mean"], 4),
                "worst_min": round(s["worst_min"], 4),
                "worst_max": round(s["worst_max"], 4),
                "worst_range": round(s["worst_range"], 4),
                "avg_mean": round(s["avg_mean"], 4),
            })


def print_summary(rows: list[dict],
                  agg: dict[tuple[str, str], dict],
                  ordering: dict, dip: dict, dare: dict, srange: dict) -> None:
    print(f"\nLoaded {len(rows)} cells across "
          f"{len({r['model'] for r in rows})} models x "
          f"{len({r['method'] for r in rows})} methods x "
          f"{len({r['seed'] for r in rows})} seeds.\n")
    # ---- main table
    print("=" * 88)
    print(f"{'model':<14}{'method':<22}{'n':<3}{'worst_mean':<13}"
          f"{'worst_range':<13}{'avg_mean':<10}")
    print("-" * 88)
    for (model, method) in sorted(agg.keys()):
        s = agg[(model, method)]
        print(f"{model:<14}{method:<22}{s['n_seeds']:<3}"
              f"{s['worst_mean']:<13.4f}"
              f"{s['worst_range']:<13.4f}"
              f"{s['avg_mean']:<10.4f}")
    print("=" * 88)

    # ---- C1
    print("\n[C1] Cross-model ordering for task_arithmetic worst-excess:")
    measured = ordering["measured"]
    for m in MODELS:
        v = measured.get(m)
        print(f"  {m:<14}{v if v is None else f'{v:.4f}'}")
    print(f"  expected: {ordering['expected_order']}")
    print(f"  HOLDS: {ordering['holds']}")

    # ---- C2
    print("\n[C2] b=2 TVQ less-is-more dip (b2 < b4 worst-excess):")
    for m, v in dip["per_model"].items():
        print(f"  {m:<14}b2={v['b2']!s:<10}  b4={v['b4']!s:<10}  "
              f"dip={v['dip_holds']}")
    print(f"  HOLDS in {dip['n_hold']}/{dip['n_total']} models")

    # ---- C3
    print("\n[C3] DARE tracks TA per-(model, seed) to 3 decimals:")
    for d in dare["per_cell"]:
        print(f"  {d['model']:<14}seed{d['seed']}  ta={d['ta']:.4f}  "
              f"dare={d['dare']:.4f}  |Δ|={d['abs_diff']:.5f}")
    print(f"  max |Δ| = {dare['max_abs_diff']:.5f}  "
          f"(holds: {dare['holds_to_3_decimals']})")

    # ---- C4
    if srange:
        print("\n[C4] Seed-range scale:")
        print(f"  median seed range  = {srange['median_seed_range']:.4f}")
        print(f"  max seed range     = {srange['max_seed_range']:.4f}")
        print(f"  method spread      = {srange['method_spread']:.4f}")
        print(f"  ratio (median/spread) = {srange['seed_range_over_spread']:.3f}")

    print("\nWrote summary to:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")


def main() -> int:
    rows = build_table()
    if not rows:
        print(f"No cells found in {MATRIX_DIR}", file=sys.stderr)
        return 1
    agg = aggregate_by_method(rows)
    ordering = check_ordering(agg, method="task_arithmetic")
    dip = check_b2_dip(agg)
    dare = check_dare_tracks_ta(rows)
    srange = check_seed_range_small(agg)
    seed0_ref = load_seed0_reference()

    # ---- write JSON summary (machine-readable)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cells": len(rows),
        "models": MODELS,
        "methods": METHODS,
        "agg": {f"{m}__{meth}": s for (m, meth), s in agg.items()},
        "claims": {
            "C1_cross_model_ordering": ordering,
            "C2_b2_dip": dip,
            "C3_dare_tracks_ta": dare,
            "C4_seed_range_scale": srange,
        },
        "seed0_reference": {f"{m}__{meth}": v
                            for (m, meth), v in seed0_ref.items()},
    }
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    write_csv(rows, agg)
    print_summary(rows, agg, ordering, dip, dare, srange)
    return 0


if __name__ == "__main__":
    sys.exit(main())

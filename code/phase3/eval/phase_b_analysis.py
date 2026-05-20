"""Phase B analysis: aggregate the 20 eval cells, generate the headline
rate-distortion figure, run the 3 validation-criteria checks.

Default outputs (n=200 / Day-4 paths):
  - code/phase3/figures/headline_rd.png   — the headline plot
  - code/phase3/figures/method_compare.png — TA/TIES/DARE/KnOTS bar chart
  - results/phase3/phase_b_summary.json    — criteria pass/fail + numbers
  - stdout: human-readable summary

Per notes/phase3_design.md §6 validation criteria:
  (1) Floor exists at b=32 — non-zero `worst_excess` for at least one method.
  (2) TVQ slope ≈ -2  — slope of log2(worst_excess - floor) vs b for b in {1,2,4,8}.
  (3) KnOTS > Task Arithmetic on at least some tasks.

CLI (added 2026-05-19):
  --eval-dir PATH    directory of {model}__{method}.json cell files
                     (default: results/phase3/eval_matrix)
  --fig-dir PATH     output directory for figures (default: code/phase3/figures)
  --summary PATH     output path for summary JSON
                     (default: results/phase3/phase_b_summary.json)
  --models LIST      comma-separated model keys (default: auto-detect from eval_dir)
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
DEFAULT_EVAL_DIR = PROJECT_ROOT / "results/phase3/eval_matrix"
DEFAULT_FIG_DIR = PROJECT_ROOT / "code/phase3/figures"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/phase3/phase_b_summary.json"

NON_TVQ = ["task_arithmetic", "ties", "dare", "knots"]
TVQ_RATES = [1, 2, 4, 8, 16, 32]
# Stable display order; models not in this list go to the end of `models`.
PREFERRED_MODEL_ORDER = ["llama31_8b", "qwen25_7b", "mistral_7b", "yi15_9b"]


def discover_models(eval_dir: Path) -> list[str]:
    """Auto-detect model keys from cell JSON filenames `{model}__{method}.json`."""
    found = set()
    for p in eval_dir.glob("*__*.json"):
        m = re.match(r"^(.+?)__", p.name)
        if m:
            found.add(m.group(1))
    ordered = [m for m in PREFERRED_MODEL_ORDER if m in found]
    ordered += sorted(found - set(ordered))
    return ordered


def load_all(eval_dir: Path, models: list[str]) -> dict:
    """Returns {model: {method or ('tvq', b): cell_dict}}."""
    out = {m: {} for m in models}
    for m in models:
        for meth in NON_TVQ:
            p = eval_dir / f"{m}__{meth}.json"
            if p.exists():
                out[m][meth] = json.loads(p.read_text())
        for b in TVQ_RATES:
            p = eval_dir / f"{m}__tvq_b{b}.json"
            if p.exists():
                out[m][("tvq", b)] = json.loads(p.read_text())
    return out


def _subplot_grid(n_models: int) -> tuple[int, int, tuple[float, float]]:
    """Return (rows, cols, figsize) for a model-panel grid."""
    if n_models == 1:
        return 1, 1, (7, 5)
    if n_models == 2:
        return 1, 2, (12, 5)
    if n_models <= 4:
        return 2, 2, (12, 9)
    # >4: 2 rows, ceil(n/2) cols
    cols = (n_models + 1) // 2
    return 2, cols, (5 * cols, 9)


def headline_plot(data: dict, models: list[str], out_path: Path) -> None:
    rows, cols, figsize = _subplot_grid(len(models))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=False)
    # Flatten to a uniform iterable regardless of grid shape
    if rows * cols == 1:
        ax_list = [axes]
    else:
        ax_list = np.array(axes).flatten().tolist()
    colors_method = {"task_arithmetic": "#1f77b4", "ties": "#2ca02c",
                     "dare": "#d62728", "knots": "#9467bd"}
    for i, m in enumerate(models):
        ax = ax_list[i]
        # TVQ curve
        b_vals, w_vals = [], []
        for b in TVQ_RATES:
            cell = data[m].get(("tvq", b))
            if cell is None:
                continue
            b_vals.append(b)
            w_vals.append(cell["worst_task_excess"])
        ax.plot(b_vals, w_vals, marker="o", linestyle="-", color="black", label="TVQ")
        # Non-TVQ as horizontal lines at b=32 (their implicit rate)
        for meth in NON_TVQ:
            cell = data[m].get(meth)
            if cell is None:
                continue
            ax.axhline(cell["worst_task_excess"], color=colors_method[meth],
                       linestyle="--", alpha=0.7, label=meth)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bits per LoRA parameter (TVQ rate)")
        ax.set_ylabel("worst-task excess NLL (nats/token)")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    # Hide unused panels in the grid
    for j in range(len(models), len(ax_list)):
        ax_list[j].set_visible(False)
    fig.suptitle("Headline rate-distortion: worst-task excess vs TVQ rate, with non-TVQ baselines")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[phase_b] wrote {out_path}", flush=True)


def method_compare_plot(data: dict, models: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(models) + 4), 5))
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(len(NON_TVQ))
    for i, m in enumerate(models):
        ys = []
        for meth in NON_TVQ:
            cell = data[m].get(meth)
            ys.append(cell["worst_task_excess"] if cell else float("nan"))
        ax.bar(x + i * width, ys, width, label=m)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(NON_TVQ)
    ax.set_ylabel("worst-task excess NLL")
    ax.set_title("Non-TVQ merging methods: worst-task excess per model")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[phase_b] wrote {out_path}", flush=True)


def check_criterion_1(data: dict, models: list[str]) -> dict:
    """Floor exists at b=32: non-zero worst_excess for each model."""
    out = {}
    for m in models:
        cell = data[m].get(("tvq", 32))
        if cell is None:
            out[m] = {"pass": False, "reason": "tvq b=32 cell missing"}
            continue
        we = cell["worst_task_excess"]
        out[m] = {"pass": we > 1e-3, "worst_excess_at_b32": we}
    return out


def check_criterion_2(data: dict, models: list[str]) -> dict:
    """TVQ slope ≈ -2 of log2(worst_excess - floor) vs b, for b in {1, 2, 4, 8}."""
    out = {}
    for m in models:
        floor = data[m].get(("tvq", 32), {}).get("worst_task_excess")
        if floor is None:
            out[m] = {"pass": False, "reason": "floor (b=32) missing"}
            continue
        pts = []
        for b in [1, 2, 4, 8]:
            cell = data[m].get(("tvq", b))
            if cell is None:
                continue
            diff = cell["worst_task_excess"] - floor
            # only positive (above floor) points contribute to a meaningful slope
            if diff > 1e-4:
                pts.append((b, math.log2(diff)))
        if len(pts) < 3:
            out[m] = {"pass": False, "reason": f"only {len(pts)} usable points above floor",
                       "points": pts}
            continue
        bs, logs = zip(*pts)
        slope, intercept = np.polyfit(bs, logs, 1)
        # accept |slope - (-2)| <= 0.5 as "approximately -2"
        ok = abs(slope - (-2.0)) <= 0.5
        out[m] = {"pass": bool(ok), "slope_estimate": float(slope),
                  "n_points": len(pts), "points": pts}
    return out


def check_criterion_3(data: dict, models: list[str]) -> dict:
    """KnOTS > Task Arithmetic on per-task excess for at least one (model, task)."""
    out = {}
    wins = []
    for m in models:
        ta = data[m].get("task_arithmetic")
        kn = data[m].get("knots")
        if ta is None or kn is None:
            out[m] = {"pass": False, "reason": "TA or KnOTS missing"}
            continue
        per_task = {}
        for task in ta["excess_per_task"]:
            ta_e = ta["excess_per_task"][task]
            kn_e = kn["excess_per_task"].get(task, float("nan"))
            per_task[task] = {"task_arithmetic": ta_e, "knots": kn_e,
                              "knots_better": kn_e < ta_e}
            if kn_e < ta_e:
                wins.append((m, task))
        n_wins = sum(1 for v in per_task.values() if v["knots_better"])
        out[m] = {"per_task": per_task, "n_knots_wins": n_wins, "n_tasks": len(per_task)}
    overall_pass = len(wins) > 0
    return {"per_model": out, "any_knots_win": overall_pass, "wins": wins}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR,
                   help="dir of {model}__{method}.json cells")
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR,
                   help="output dir for figures")
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                   help="output path for summary JSON")
    p.add_argument("--models", default=None,
                   help="comma-separated model keys (default: auto-detect)")
    args = p.parse_args()

    eval_dir = args.eval_dir
    fig_dir = args.fig_dir
    summary_path = args.summary

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models(eval_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[phase_b] eval_dir = {eval_dir}", flush=True)
    print(f"[phase_b] fig_dir  = {fig_dir}", flush=True)
    print(f"[phase_b] summary  = {summary_path}", flush=True)
    print(f"[phase_b] models   = {models}", flush=True)

    if not models:
        print("ERROR: no models found / specified", flush=True)
        return 2

    data = load_all(eval_dir, models)
    n_loaded = sum(len(v) for v in data.values())
    print(f"[phase_b] loaded {n_loaded} cells across {len(models)} models", flush=True)

    if n_loaded == 0:
        print("ERROR: no cells found", flush=True)
        return 2

    headline_plot(data, models, fig_dir / "headline_rd.png")
    method_compare_plot(data, models, fig_dir / "method_compare.png")

    c1 = check_criterion_1(data, models)
    c2 = check_criterion_2(data, models)
    c3 = check_criterion_3(data, models)

    summary = {
        "n_cells_loaded": n_loaded,
        "eval_dir": str(eval_dir),
        "models": models,
        "criterion_1_floor_at_b32": c1,
        "criterion_2_tvq_slope_near_-2": c2,
        "criterion_3_knots_beats_ta_per_task": c3,
        "per_model_summary": {
            m: {
                (f"tvq_b{k[1]}" if isinstance(k, tuple) else k):
                    {"worst_excess": cell["worst_task_excess"],
                     "avg_excess": cell["avg_task_excess"]}
                for k, cell in cells.items()
            }
            for m, cells in data.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[phase_b] wrote {summary_path}", flush=True)

    # Human-readable stdout
    print()
    print("=" * 70)
    print(f"PHASE B SUMMARY — Validation criteria per phase3_design §6  ({len(models)} models)")
    print("=" * 70)
    print()
    print("(1) Floor exists at b=32:")
    for m in models:
        r = c1[m]
        we = r.get("worst_excess_at_b32", float("nan"))
        print(f"    {m:<14}: worst_excess = {we:+.4f}  {'PASS' if r['pass'] else 'FAIL'}")
    print()
    print("(2) TVQ slope ≈ -2 (for b ∈ {1,2,4,8}, above floor):")
    for m in models:
        r = c2[m]
        if "slope_estimate" in r:
            print(f"    {m:<14}: slope = {r['slope_estimate']:+.3f}  (n_pts={r['n_points']})  {'PASS' if r['pass'] else 'FAIL'}")
        else:
            print(f"    {m:<14}: SKIP — {r.get('reason')}")
    print()
    print("(3) KnOTS beats TA per-task:")
    for m in models:
        r = c3["per_model"][m]
        n_wins = r.get("n_knots_wins", 0)
        n_tasks = r.get("n_tasks", 0)
        print(f"    {m:<14}: KnOTS beat TA on {n_wins}/{n_tasks} tasks")
    print(f"    Overall any-KnOTS-win: {'PASS' if c3['any_knots_win'] else 'FAIL'}")
    print()
    print("CRITERIA_CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

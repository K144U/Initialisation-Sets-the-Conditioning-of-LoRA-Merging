"""Headline figure with bootstrap error bars.

Two panels per the paper's needs:
  (left) worst_task_excess per non-TVQ method × 4 models, with 95% bootstrap CI
  (right) TVQ rate sweep per model, with 95% CI, showing the universal b=2 dip

Reads bootstrap_ci_v3.json (per-cell worst_task_excess CIs).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/sanjay.g/projects/rdmerge")
CI = json.loads((ROOT / "results/phase3/bootstrap_ci_v3.json").read_text())["cells"]

MODELS = ["llama31_8b", "qwen25_7b", "mistral_7b", "yi15_9b"]
NON_TVQ = ["task_arithmetic", "ties", "dare", "knots"]
TVQ_B = [1, 2, 4, 8, 16, 32]
COLORS = {"llama31_8b": "#1f77b4", "qwen25_7b": "#ff7f0e",
          "mistral_7b": "#2ca02c", "yi15_9b": "#d62728"}


def pt_ci(cell_name):
    c = CI[cell_name]
    p = c["point_worst_task_excess"]
    lo, hi = c["boot_worst_task_excess_ci95"]
    return p, p - lo, hi - p


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: non-TVQ methods, grouped bars by model, with CI
ax = axes[0]
x = np.arange(len(NON_TVQ))
w = 0.2
for i, m in enumerate(MODELS):
    pts, los, his = [], [], []
    for meth in NON_TVQ:
        p, l, h = pt_ci(f"{m}__{meth}")
        pts.append(p); los.append(l); his.append(h)
    ax.bar(x + i * w, pts, w, yerr=[los, his], capsize=3,
           label=m, color=COLORS[m], edgecolor="black", linewidth=0.5)
ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(["Task Arith", "TIES", "DARE", "KnOTS"])
ax.set_ylabel("worst-task excess NLL (nats/token)")
ax.set_title("Non-TVQ methods (95% bootstrap CI)\nOnly TIES separates from the TA/DARE/KnOTS cluster")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Panel 2: TVQ rate sweep with CI, log-x
ax = axes[1]
for m in MODELS:
    pts, los, his = [], [], []
    for b in TVQ_B:
        p, l, h = pt_ci(f"{m}__tvq_b{b}")
        pts.append(p); los.append(l); his.append(h)
    ax.errorbar(TVQ_B, pts, yerr=[los, his], marker="o", capsize=3,
                label=m, color=COLORS[m])
ax.set_xscale("log", base=2)
ax.set_xlabel("bits per LoRA parameter (TVQ rate)")
ax.set_ylabel("worst-task excess NLL (nats/token)")
ax.set_title("TVQ rate sweep (95% bootstrap CI)\nUniversal b=2 local minimum across 4 architectures")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = ROOT / "code/phase3/figures/v3/headline_with_ci.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Split versions (one figure per panel) so each gets full real estate.
# ---------------------------------------------------------------------------
def _methods_panel(ax):
    x = np.arange(len(NON_TVQ))
    w = 0.2
    for i, m in enumerate(MODELS):
        pts, los, his = [], [], []
        for meth in NON_TVQ:
            p, l, h = pt_ci(f"{m}__{meth}")
            pts.append(p); los.append(l); his.append(h)
        ax.bar(x + i * w, pts, w, yerr=[los, his], capsize=3,
               label=m, color=COLORS[m], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(["Task Arith", "TIES", "DARE", "KnOTS"])
    ax.set_ylabel("worst-task excess NLL (nats/token)")
    ax.set_title("Non-TVQ merging methods (95% bootstrap CI)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def _tvq_panel(ax):
    for m in MODELS:
        pts, los, his = [], [], []
        for b in TVQ_B:
            p, l, h = pt_ci(f"{m}__tvq_b{b}")
            pts.append(p); los.append(l); his.append(h)
        ax.errorbar(TVQ_B, pts, yerr=[los, his], marker="o", capsize=3,
                    label=m, color=COLORS[m])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bits per LoRA parameter (TVQ rate)")
    ax.set_ylabel("worst-task excess NLL (nats/token)")
    ax.set_title("TVQ rate sweep (95% bootstrap CI)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


f1, a1 = plt.subplots(figsize=(8, 5)); _methods_panel(a1); f1.tight_layout()
out1 = ROOT / "code/phase3/figures/v3/headline_methods_ci.png"
f1.savefig(out1, dpi=300, bbox_inches="tight"); print(f"wrote {out1}")

f2, a2 = plt.subplots(figsize=(8, 5)); _tvq_panel(a2); f2.tight_layout()
out2 = ROOT / "code/phase3/figures/v3/headline_tvq_ci.png"
f2.savefig(out2, dpi=300, bbox_inches="tight"); print(f"wrote {out2}")

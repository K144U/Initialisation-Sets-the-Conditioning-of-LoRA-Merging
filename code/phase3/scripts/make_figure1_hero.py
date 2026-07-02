"""Generates Figure 1: cross-model T-scaling hero plot.

A 1x3 panel showing worst-task NLL excess vs T for the matrix methods,
ordered left-to-right by per-task sign-election win-share range R:
Yi-1.5-9B-Chat (left, saturated base, R=0.077 — TIES inverts at T=7),
Mistral-7B-Instruct-v0.3 (middle, the pre-registered third base,
R=0.0326 — TIES ambiguous, as pre-registered), and
Llama-3.1-8B-Instruct (right, non-saturated, R=0.028 — TIES stays low,
rd_ridge widens its lead).

Output: paper_artifacts/figures/figure1_cross_model_T_scaling.{png,pdf}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Portable root: .../code/phase3/scripts/make_figure1_hero.py -> repo root.
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "paper_artifacts/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY = {
    "yi15_9b": ROOT / "results/phase3/e6_T_scaling_summary.json",
    "mistral_7b": ROOT / "results/phase3/mistral_t7_summary.json",
    "llama31_8b": ROOT / "results/phase3/e6_T_scaling_summary_llama31_8b.json",
}
TITLE = {
    "yi15_9b": "Yi-1.5-9B-Chat (saturated, $R{=}0.077$)",
    "mistral_7b": "Mistral-7B-v0.3 (pre-registered, $R{=}0.0326$)",
    "llama31_8b": "Llama-3.1-8B-Instruct (non-saturated, $R{=}0.028$)",
}
METHODS_PLOT = ["ta", "ties", "tvq_b2", "rd_ridge"]
METHOD_LABEL = {
    "ta": "Task Arithmetic",
    "ties": "TIES",
    "tvq_b2": "TVQ (b=2)",
    "rd_ridge": "rd-encoder ridge (ours)",
}
METHOD_COLOR = {
    "ta":       "#888888",
    "ties":     "#d62728",
    "tvq_b2":   "#ff7f0e",
    "rd_ridge": "#1f77b4",
}
METHOD_LS = {
    "ta": ":",
    "ties": "--",
    "tvq_b2": "-.",
    "rd_ridge": "-",
}
T_VALUES = [2, 4, 7]


def load_arms(base: str) -> dict[int, dict[str, dict[str, float]]]:
    p = SUMMARY[base]
    s = json.load(open(p))
    agg = s["agg"]
    out: dict[int, dict[str, dict[str, float]]] = {T: {} for T in T_VALUES}
    for key, val in agg.items():
        # Key format: "T{n}__{method}"
        T_str, m = key.split("__", 1)
        T = int(T_str[1:])
        if T in out:
            out[T][m] = val
    return out


def main() -> int:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=False)

    for ax, base in zip(axes, ["yi15_9b", "mistral_7b", "llama31_8b"]):
        arms = load_arms(base)
        for m in METHODS_PLOT:
            xs, means, mins, maxs = [], [], [], []
            for T in T_VALUES:
                if m in arms[T]:
                    xs.append(T)
                    means.append(arms[T][m]["worst_mean"])
                    mins.append(arms[T][m]["worst_min"])
                    maxs.append(arms[T][m]["worst_max"])
            xs = np.array(xs)
            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)
            ax.fill_between(xs, mins, maxs,
                            color=METHOD_COLOR[m], alpha=0.13)
            lw = 2.8 if m == "rd_ridge" else 1.9
            ms = 6.0 if m == "rd_ridge" else 4.5
            ax.plot(xs, means, color=METHOD_COLOR[m], ls=METHOD_LS[m],
                    lw=lw, marker="o", ms=ms,
                    label=METHOD_LABEL[m])
        ax.set_xticks(T_VALUES)
        ax.set_xlabel(r"task count $T$")
        ax.set_title(TITLE[base])
        ax.grid(True, alpha=0.25, ls=":")
        ax.set_xlim(1.6, 7.4)

    axes[0].set_ylabel("worst-task NLL excess")

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS_PLOT),
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    out_png = OUT_DIR / "figure1_cross_model_T_scaling.png"
    out_pdf = OUT_DIR / "figure1_cross_model_T_scaling.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

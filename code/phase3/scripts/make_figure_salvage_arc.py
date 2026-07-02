"""Salvage-arc Figure: exact construction (catastrophic) -> ridge correction
(state-of-the-art) on Llama-3.1-8B-Instruct.

A 1x2 panel:
  (left) bar chart: TA / TIES / rd-encoder (exact) / rd-encoder ridge λ*
         worst-task NLL excess at T=4. The "exact" bar is 0.497,
         dwarfing the others; "ridge λ*" sits at 0.091.
  (right) ridge λ sweep: worst-task excess vs λ on a log axis,
          with TA and TIES baselines as horizontal reference lines.

The figure visually tells the §6.2 salvage story in 3 seconds:
the theorem's exact encoder fails for a measured spectral reason,
one line of Tikhonov ridge rescues it to state-of-the-art.

Output: paper_artifacts/figures/figure_salvage_arc.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/sanjay.g/projects/rdmerge")
OUT = ROOT / "paper_artifacts/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Numbers from §6.2 Table tab:rd-ridge-sweep on Llama-3.1-8B-Instruct, T=4.
# MATCHED seed1/2/3 3-seed means (2026-07-01); replaces the legacy v1 numbers.
LAMBDAS = [0.001, 0.01, 0.05, 0.07, 0.10, 0.13, 0.17, 0.20, 0.30, 1.00]
WORST_EXCESS = [2.879, 0.389, 0.094, 0.095, 0.110, 0.125, 0.143, 0.155, 0.191, 0.289]
TA_EXCESS = 0.220   # 3-seed mean (was single-seed 0.219)
TIES_EXCESS = 0.154  # 3-seed mean (was single-seed 0.161)
EXACT_EXCESS = 0.340  # rd-encoder λ=0 exact construction, matched 3-seed mean (v1 was 0.497)
LAMBDA_STAR = 0.05
RIDGE_STAR = 0.094

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

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 3.8),
                                  gridspec_kw={"width_ratios": [1.0, 1.4]})

# Left: bar chart with 4 methods on Llama-3.1
methods = ["TA", "TIES", "rd-enc.\nλ=0", "rd-enc.\nridge λ★"]
values = [TA_EXCESS, TIES_EXCESS, EXACT_EXCESS, RIDGE_STAR]
colors = ["#888888", "#d62728", "#aa3333", "#1f77b4"]
bars = ax_l.bar(methods, values, color=colors, edgecolor="black", lw=0.5)
ax_l.set_ylabel("worst-task NLL excess")
ax_l.set_title("(a) Salvage arc on Llama-3.1-8B-Instruct")
ax_l.grid(True, alpha=0.25, ls=":", axis="y")
ax_l.set_ylim(0, 0.62)
# Annotate each bar with the value (above the bar)
for bar, v in zip(bars, values):
    ax_l.text(bar.get_x() + bar.get_width()/2, v + 0.018,
              f"{v:.3f}", ha="center", va="bottom", fontsize=10,
              fontweight="bold" if v == EXACT_EXCESS or v == RIDGE_STAR else "normal")
# Annotation arrow: ridge "rescues" the encoder, drawn ABOVE the bars
ax_l.annotate(
    "", xy=(3.15, RIDGE_STAR + 0.05),
    xytext=(2.35, EXACT_EXCESS + 0.03),
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.6,
                    connectionstyle="arc3,rad=-0.4"))
ax_l.text(2.7, 0.42, "ridge\nrescues", color="#1f77b4",
          ha="center", va="center", fontsize=10, fontweight="bold")
ax_l.tick_params(axis="x", labelsize=9.5)

# Right: λ sweep
ax_r.plot(LAMBDAS, WORST_EXCESS, "o-", color="#1f77b4", lw=2.0, ms=5.5,
          label="rd-encoder ridge")
ax_r.axhline(TA_EXCESS, color="#888888", ls=":", lw=1.6,
             label=f"Task Arithmetic ({TA_EXCESS:.3f})")
ax_r.axhline(TIES_EXCESS, color="#d62728", ls="--", lw=1.6,
             label=f"TIES ({TIES_EXCESS:.3f})")
# Star the optimum
ax_r.plot([LAMBDA_STAR], [RIDGE_STAR], marker="*", color="gold",
          ms=18, markeredgecolor="black", markeredgewidth=1.0,
          zorder=10)
ax_r.annotate(f"λ★ = {LAMBDA_STAR},  Φ★ = {RIDGE_STAR:.3f}",
              xy=(LAMBDA_STAR, RIDGE_STAR),
              xytext=(0.15, 0.40),
              fontsize=10, color="black",
              arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
ax_r.set_xscale("log")
ax_r.set_xlabel(r"Tikhonov $\lambda$ (log scale)")
ax_r.set_ylabel("worst-task NLL excess")
ax_r.set_title("(b) Ridge sweep: λ* recovers state-of-the-art")
ax_r.grid(True, alpha=0.25, ls=":")
ax_r.set_ylim(0.0, 0.65)
ax_r.legend(loc="upper left", framealpha=0.95)

plt.tight_layout()
out_png = OUT / "figure_salvage_arc.png"
out_pdf = OUT / "figure_salvage_arc.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")
plt.close()

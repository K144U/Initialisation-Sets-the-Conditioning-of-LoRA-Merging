"""Hero figure for the two-regime result.

Left panel: hard effective dimension against the rank tolerance eps. The
shared-initialisation cohort only reaches d_eff = Tr at numerically loose
tolerances and collapses as soon as the tolerance is realistic; the three
independently initialised cohorts hold d_eff = Tr = 64 across four decades.

axR.set_ylabel(r"condition number $\kappa(\bar H)$")
floor panel: the exact floor is zero in both regimes (see app_proofs), so the
floor difference that panel showed was an artifact of substituting a soft
participation ratio into a rank formula.

Writes paper/figures/figure_two_regimes.pdf. Run from the repo root.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "phase3"
OUT = ROOT / "paper" / "figures" / "figure_two_regimes.pdf"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
LABEL = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B",
         "qwen25_7b": "Qwen2.5-7B", "yi15_9b": "Yi-1.5-9B"}
COLOR = {"llama31_8b": "#1b6ca8", "mistral_7b": "#c8552a",
         "qwen25_7b": "#2e7d51", "yi15_9b": "#7d4b9e"}

# JAIR asks that every figure be understandable on a monochrome device, and
# hue is the first thing a greyscale render throws away: these four colours
# convert to nearly the same grey, which left the legend with four identical
# swatches. Marker shape and dash pattern survive the conversion, so the
# series are separated by those and the colour is now decoration rather than
# information.
MARKER = {"llama31_8b": "o", "mistral_7b": "s",
          "qwen25_7b": "^", "yi15_9b": "D"}
DASH = {"llama31_8b": (4, 1.6), "mistral_7b": (1.4, 1.4),
        "qwen25_7b": (6, 1.6, 1.4, 1.6), "yi15_9b": (3, 1.4, 1.4, 1.4)}

shared = json.loads((RES / "subspace_geometry_seed1.json").read_text())
indep = {c: json.loads((RES / f"subspace_geometry_{c}.json").read_text())
         for c in ("indep1", "indep2", "indep3")}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.0))

# ---------------- left: hard d_eff vs eps ----------------
eps_keys = ["1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.03", "0.1"]
eps_val = [float(e) for e in eps_keys]

# Every independent cell (4 bases x 3 cohorts) sits exactly on d_eff = 64 at
# every tolerance, so plotting twelve overlapping lines hides the point. Draw
# them as one band and assert the coincidence rather than implying it.
flat = [indep[c][b]["hard_d_eff"][e]
        for c in indep for b in BASES for e in eps_keys]
assert set(flat) == {64.0}, sorted(set(flat))
# Drawn in black rather than in llama's blue, which it used to share: in
# greyscale the reference line and one of the four falling curves came out as
# the same ink. Thick and solid against thin and dashed reads in both.
axL.plot(eps_val, [64] * len(eps_val), lw=4.0, color="#111111", alpha=0.85,
         solid_capstyle="round", zorder=3,
         label="independent init (4 bases $\\times$ 3 cohorts, all identical)")

for b in BASES:
    axL.plot(eps_val, [shared[b]["hard_d_eff"][e] for e in eps_keys],
             marker=MARKER[b], ms=4.5, lw=1.6, color=COLOR[b],
             ls=(0, DASH[b]), mfc="white", mew=1.1, zorder=4,
             label=f"shared init, {LABEL[b]}")

axL.set_xscale("log")
axL.set_xlabel(r"rank tolerance $\epsilon$ (relative to $\sigma_1$)")
axL.set_ylabel(r"hard effective dimension $d_{\mathrm{eff}}$")
axL.set_ylim(0, 78)
axL.set_title("Effective dimension is tolerance-fragile\nonly under shared initialisation",
              fontsize=10)
axL.legend(fontsize=7.0, loc="lower left", frameon=False)

# ---------------- right: conditioning ----------------
# The floor panel this replaces asserted a shared-vs-independent difference
# of ~60x in B^2(1 - d_eff/(Tr)). That was computed by substituting the SOFT
# participation ratio into a formula derived for a RANK. The exact floor is
# zero in both regimes, so the panel was showing an artifact. What actually
# separates the regimes is the conditioning of Hbar.
cond = json.loads((RES / "floor_conditioning.json").read_text())
KS, KI = "seed1 | T=4 (all)", "indep1 | T=4 (all)"

x = range(len(BASES))
sh = [cond[KS][b]["cond_Hbar"] for b in BASES]
ind = [cond[KI][b]["cond_Hbar"] for b in BASES]

w = 0.36
axR.bar([i - w / 2 for i in x], sh, w, color="#b03a2e",
        edgecolor="black", linewidth=0.8, hatch="////",
        label="shared init (one $A$ per cohort)")
axR.bar([i + w / 2 for i in x], ind, w, color="#1b6ca8",
        edgecolor="black", linewidth=0.8, hatch="..",
        label="independent init")
for i, (a, b) in enumerate(zip(sh, ind)):
    axR.text(i - w / 2, a * 1.35, f"{a/1e4:.1f}e4", ha="center", fontsize=7.5)
    axR.text(i + w / 2, b * 1.35, f"{b:.1f}", ha="center", fontsize=7.5)

axR.set_yscale("log")
axR.set_xticks(list(x))
axR.set_xticklabels([LABEL[b] for b in BASES], fontsize=8, rotation=12)
axR.set_ylabel(r"condition number $\kappa(\bar H)$")
axR.set_ylim(1, 3e6)
axR.set_title("Conditioning, not the floor, is what\n"
              "initialisation controls", fontsize=10)
axR.legend(fontsize=7.5, frameon=False, loc="upper center")

for ax in (axL, axR):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
print("cond shared :", ["%.0f" % v for v in sh])
print("cond indep  :", ["%.1f" % v for v in ind])
print("ratio       :", ["%.0fx" % (a / b) for a, b in zip(sh, ind)])

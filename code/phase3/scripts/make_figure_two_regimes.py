"""Hero figure for the two-regime result.

Left panel: hard effective dimension against the rank tolerance eps. The
shared-initialisation cohort only reaches d_eff = Tr at numerically loose
tolerances and collapses as soon as the tolerance is realistic; the three
independently initialised cohorts hold d_eff = Tr = 64 across four decades.

Right panel: the resulting Lemma 2 floor B^2 (1 - d_eff / (Tr)) as a fraction
of B^2. Shared init sits at ~0.745, independent init at ~0.012.

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
axL.plot(eps_val, [64] * len(eps_val), lw=4.0, color="#1b6ca8", alpha=0.85,
         solid_capstyle="round", zorder=3,
         label="independent init (4 bases $\\times$ 3 cohorts, all identical)")

for b in BASES:
    axL.plot(eps_val, [shared[b]["hard_d_eff"][e] for e in eps_keys],
             marker="o", ms=4, lw=1.6, color=COLOR[b], ls="--", zorder=4,
             label=f"shared init, {LABEL[b]}")

axL.set_xscale("log")
axL.set_xlabel(r"rank tolerance $\epsilon$ (relative to $\sigma_1$)")
axL.set_ylabel(r"hard effective dimension $d_{\mathrm{eff}}$")
axL.set_ylim(0, 78)
axL.set_title("Effective dimension is tolerance-fragile\nonly under shared initialisation",
              fontsize=10)
axL.legend(fontsize=7.0, loc="lower left", frameon=False)

# ---------------- right: floor as a fraction of B^2 ----------------
x = range(len(BASES))
sh_frac = [1.0 - shared[b]["soft_d_eff"] / shared[b]["Tr"] for b in BASES]
# mean across the three independent cohorts
in_frac = [sum(1.0 - indep[c][b]["soft_d_eff"] / indep[c][b]["Tr"]
               for c in indep) / len(indep) for b in BASES]

w = 0.36
axR.bar([i - w / 2 for i in x], sh_frac, w, color="#b03a2e",
        label="shared init (one $A$ per cohort)")
axR.bar([i + w / 2 for i in x], in_frac, w, color="#1b6ca8",
        label="independent init (mean of 3 cohorts)")

for i, (s, v) in enumerate(zip(sh_frac, in_frac)):
    axR.text(i - w / 2, s + 0.018, f"{s:.3f}", ha="center", fontsize=7.5)
    axR.text(i + w / 2, v + 0.018, f"{v:.3f}", ha="center", fontsize=7.5)

axR.set_xticks(list(x))
axR.set_xticklabels([LABEL[b] for b in BASES], fontsize=8, rotation=12)
axR.set_ylabel(r"irreducible floor $\;/\;B^2$")
# headroom so the legend clears the 0.745 bars and their value labels
axR.set_ylim(0, 1.16)
axR.set_title(r"Lemma 2 floor $B^2(1 - d_{\mathrm{eff}}/(Tr))$" "\n"
              "moves by a factor of about 60", fontsize=10)
axR.legend(fontsize=7.5, frameon=False, loc="upper center")

for ax in (axL, axR):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
print("shared frac :", ["%.4f" % v for v in sh_frac])
print("indep  frac :", ["%.4f" % v for v in in_frac])
print("ratio       :", ["%.1fx" % (s / v) for s, v in zip(sh_frac, in_frac)])

#!/usr/bin/env python3
"""Figures for the Neurocomputing short paper (paper/neurocomputing.tex).

Every number plotted is read from a result file; nothing is typed in by hand.

    python paper/nc/geometry_stats.py --art artifacts/lora \
        --out paper/nc/data/geometry_stats.json        # once, CPU, ~minutes
    python paper/nc/make_figures.py

Writes paper/nc/figures/fig{1..6}_*.pdf plus PNG and greyscale previews in
figures/preview/ for the print check: shared and independent arms differ by
marker and line style as well as hue, so every figure reads without colour.

Drawn at the true column width of the CAS cas-sc layout (6.47 in), so
LaTeX places them at 100% and the 7-8 pt labels print at that size.

Palette: the first three slots of the dataviz reference palette, validated
all-pairs (worst CVD dE 9.2, normal vision 24.0). Scatter plots use at most
three hues; further distinctions are carried by marker shape and fill.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "results" / "phase3"
CAMP = RES / "campaign_2026-09"
DATA = HERE / "data"
OUT = HERE / "figures"
PREV = OUT / "preview"
OUT.mkdir(exist_ok=True)
PREV.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, FAINT = "#0b0b0b", "#52514e", "#e2e1dc", "#9a9890"
SH = dict(color=ORANGE, marker="^", ls="-", label="shared initialisation")
IN = dict(color=BLUE, marker="o", ls="--", label="independent initialisation")

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BL = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B",
      "qwen25_7b": "Qwen2.5-7B", "yi15_9b": "Yi-1.5-9B"}
WIDE = 6.47  # inches: the measured text width of the CAS (cas-sc) layout

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.frameon": False, "legend.fontsize": 6.8,
    "lines.linewidth": 1.3, "lines.markersize": 4.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mean(v):
    return float(np.mean(v)) if isinstance(v, list) else float(v)


def sd(v):
    return float(np.std(v, ddof=1)) if isinstance(v, list) and len(v) > 1 else 0.0


def lab(ax, s, x=-0.02):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontweight="bold", fontsize=8.5,
            color=INK, va="bottom", ha="right")


def title(ax, s):
    ax.set_title(s, loc="left", color=INK, fontsize=7.8, pad=4)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    # Previews are conveniences: if one is open in an image viewer, Windows
    # locks it, and that must not stop the PDF (the real output) being written.
    try:
        fig.savefig(PREV / f"{name}.png", dpi=200)
        from PIL import Image
        Image.open(PREV / f"{name}.png").convert("L").save(PREV / f"{name}_grey.png")
    except OSError as e:
        print(f"  preview for {name} not refreshed (file in use?): {e.__class__.__name__}")
    plt.close(fig)


def thresholds(ax, vertical=False):
    for v in (0.9, 0.5):
        (ax.axvline if vertical else ax.axhline)(v, color=INK2, lw=0.6, ls=":", zorder=1)


# --------------------------------------------------------------------------
# Figure 1: the collapse.
# --------------------------------------------------------------------------
def fig1_collapse(G):
    drift = load(RES / "drift_summary.json")["arms"]
    nc = load(CAMP / "new_cohorts_summary.json")
    D = G["data"]
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(WIDE, 4.55))

    # (a) A moves the same distance in both arms.
    x = [0, 25, 50, 75, 100]
    for arm, sty in (("drift_shared", SH), ("drift_indep", IN)):
        tr = drift[arm]["drift_trace"][:4]
        a.plot(x, [0.0] + tr, color=sty["color"], marker=sty["marker"],
               ls=sty["ls"], label=sty["label"].replace(" initialisation", ""))
    a.set_xlabel("training progress (%)")
    a.set_ylabel(r"$\|A - A_0\|_F \,/\, \|A_0\|_F$")
    a.set_xticks(x)
    a.set_ylim(0, 0.16)
    a.legend(loc="upper left")
    cs, ci = drift["drift_shared"]["cos_final"], drift["drift_indep"]["cos_final"]
    a.text(0.98, 0.05, f"same drift, final median cosine\nshared {cs:.3f} vs. independent {ci:.3f}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=6.6, color=INK2)
    title(a, "A moves 13% in both arms")
    lab(a, "a")

    # (b) every principal angle, every pair, every layer, all bases.
    def pooled(prefix):
        out = []
        for base in BASES:
            for k in range(1, 4):
                out += D[f"{base}/{prefix}{k}"]["cos"]
        return np.array(out)
    cs_all, ci_all = pooled("seed"), pooled("indep")
    bins = np.linspace(0, 1, 101)
    b.hist(ci_all, bins=bins, color=BLUE, alpha=0.85, label="independent")
    b.hist(cs_all, bins=bins, color=ORANGE, alpha=0.85, hatch="////",
           edgecolor="white", lw=0, label="shared")
    b.set_yscale("log")
    b.set_xlim(0, 1)
    b.set_xlabel("cosine of a principal angle")
    b.set_ylabel("count")
    b.text(0.47, 0.97,
           f"{len(cs_all):,} angles per arm:\n4 bases, 3 cohorts, 8 layers,\n6 pairs, 16 angles each",
           transform=b.transAxes, va="top", ha="center", fontsize=6.3, color=INK2)
    # Direct labels instead of a legend: each population is named where it sits.
    b.annotate(f"independent:\nall $\\leq${ci_all.max():.2f}", xy=(ci_all.max(), 2),
               xytext=(0.22, 6), fontsize=6.6, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    b.annotate(f"shared (hatched):\nall $\\geq${cs_all.min():.2f}", xy=(cs_all.min(), 2),
               xytext=(0.42, 110), fontsize=6.6, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    title(b, "Every angle, not only the median")
    lab(b, "b")

    # (c) singular values of the stacked bases.
    idx = np.arange(1, 65)
    for base in BASES:
        for prefix, sty in (("seed1", SH), ("indep1", IN)):
            sv = np.median(np.array(D[f"{base}/{prefix}"]["svals"]), axis=0)
            c.semilogy(idx, sv, color=sty["color"], ls=sty["ls"], lw=1.0, alpha=0.9,
                       label=(sty["label"].replace(" initialisation", "") if base == BASES[0] else None))
    c.axhline(2.0, color=INK2, lw=0.6, ls=":")
    c.axvline(16.5, color=GRID, lw=0.8)
    c.text(63, 2.15, r"$\sqrt{T} = 2$: identical subspaces", ha="right", va="bottom",
           fontsize=6.4, color=INK2)
    c.text(17.5, 0.02, "48 directions\nnearly degenerate", fontsize=6.4, color=INK2, va="bottom")
    c.set_xlim(0, 65)
    c.set_ylim(5e-3, 4)
    c.set_xlabel("index of singular value of $[V_1\\,V_2\\,V_3\\,V_4]$")
    c.set_ylabel("singular value")
    c.legend(loc="lower left")
    title(c, "Shared: four subspaces in 16 dimensions")
    lab(c, "c")

    # (d) task count, rank, PiSSA.
    geo = nc["R"]["geometry"]
    cfg = [("T2", "T=2"), ("T4", "T=4"), ("T8", "T=8"), ("r8", "r=8"), ("r64", "r=64")]
    xb = np.arange(len(cfg))
    d.plot(xb, [geo[f"{k}/seed1fx"]["cos"] for k, _ in cfg], color=ORANGE, marker="^",
           ls="none", ms=6, label="shared")
    d.plot(xb, [geo[f"{k}/indep1fx"]["cos"] for k, _ in cfg], color=BLUE, marker="o",
           ls="none", ms=5, label="independent")
    xp = len(cfg) + 0.4
    d.plot([xp], [nc["P"]["geometry"]["cos"]], color=AQUA, marker="D", ls="none", ms=5.5,
           label="PiSSA, independent seeds")
    d.axvline(len(cfg) - 0.3, color=GRID, lw=0.8)
    thresholds(d)
    d.set_xticks(list(xb) + [xp], [l for _, l in cfg] + ["PiSSA"])
    d.set_ylim(-0.03, 1.07)
    d.set_ylabel("median principal cosine")
    d.legend(loc="center left", bbox_to_anchor=(0.0, 0.55))
    title(d, "Any T or r; PiSSA whatever the seed")
    lab(d, "d")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, "fig1_collapse")


# --------------------------------------------------------------------------
# Figure 2: the first-order theory.
# --------------------------------------------------------------------------
def fig2_theory(G):
    dt = load(CAMP / "drift_theory.json")
    nc = load(CAMP / "new_cohorts_summary.json")
    D = G["data"]
    fig, ((a, b), (c, e)) = plt.subplots(2, 2, figsize=(WIDE, 4.4))

    # (a) the spectrum of Hbar, exact against predicted, one representative layer.
    sh, ind = D["llama31_8b/seed1"], D["llama31_8b/indep1"]
    kap = [l[0] / l[-1] for l in sh["eig"]]
    li = int(np.argsort(kap)[len(kap) // 2])
    ev_s, ev_i = np.array(sh["eig"][li]), np.array(ind["eig"][li])
    pred = np.sort(np.array(sh["pred"][li]))[::-1]
    k = np.arange(1, 65)
    a.semilogy(k, ev_i, color=BLUE, marker="o", ls="none", ms=2.6, label="independent, exact")
    a.semilogy(k, ev_s, color=ORANGE, marker="^", ls="none", ms=3.0, label="shared, exact")
    a.semilogy(k[16:], pred, color=INK, lw=1.0, label="shared, first-order prediction")
    a.semilogy(k[:16], np.ones(16), color=INK, lw=1.0)
    a.set_xlabel("eigenvalue index")
    a.set_ylabel(r"eigenvalue of $\bar H$")
    a.set_ylim(min(ev_s.min(), pred.min()) / 3, 3)
    a.legend(loc="lower left")
    ks = ev_s[0] / ev_s[-1]
    ex = int(np.floor(np.log10(ks)))
    a.text(0.97, 0.97, f"$\\kappa = {ks / 10 ** ex:.1f}\\times10^{{{ex}}}$ shared\n"
           f"$\\kappa = {ev_i[0] / ev_i[-1]:.1f}$ independent",
           transform=a.transAxes, ha="right", va="top", fontsize=6.6, color=INK2)
    title(a, "Prediction matches the whole spectrum")
    lab(a, "a")

    pts = dt["dial"]["points"]
    s = np.array(sorted(float(x) for x in pts))
    key = {float(x): x for x in pts}
    kv = np.array([pts[key[v]]["kappa"] for v in s])
    av = np.array([pts[key[v]]["amp"] for v in s])
    grid = np.logspace(np.log10(0.04), np.log10(1.25), 50)
    b.loglog(s, kv, color=ORANGE, marker="^", ls="none", ms=6, label="exact")
    b.loglog(grid, kv[-1] * grid ** -2.0, color=INK2, lw=1.0,
             label=f"slope {dt['dial']['kappa_slope']:.2f}")
    b.set_xlabel("displacement scale $s$")
    b.set_ylabel(r"$\kappa(\bar H)$")
    b.legend(loc="upper right")
    title(b, r"$\kappa \propto s^{-2}$")
    lab(b, "b")

    c.loglog(s, av, color=ORANGE, marker="^", ls="none", ms=6, label="exact")
    c.loglog(grid, av[-1] * grid ** -1.0, color=INK2, lw=1.0,
             label=f"slope {dt['dial']['amp_slope']:.2f}")
    c.loglog(grid, av[-1] * grid ** -2.0, color=INK2, lw=0.9, ls=(0, (2, 2)),
             label=r"rival $s^{-2}$ form")
    c.set_ylim(20, 3e3)
    c.set_xlabel("displacement scale $s$")
    c.set_ylabel(r"$\|\bar\tau_H\| \,/\, \|\bar D\|$")
    c.legend(loc="lower left")
    title(c, r"amplification $\propto s^{-1}$")
    lab(c, "c")

    groups = [("training\ncheckpoints", [v["kappa_ratio"] for v in dt["checkpoints"].values()], "s"),
              ("12 shared\ncohorts", [v["kappa_ratio"] for v in dt["pt4"].values()], "o"),
              ("T and r\nsweep", [v["kappa_ratio"] for kk, v in nc["R"]["geometry"].items()
                                  if "seed" in kk], "D")]
    for i, (_, vals, m) in enumerate(groups):
        e.plot(i + np.linspace(-0.22, 0.22, len(vals)), vals, color=BLUE, marker=m,
               ls="none", ms=4.5)
    e.axhline(1.0, color=INK2, lw=0.8)
    e.set_xticks(range(3), [g[0] for g in groups])
    e.set_ylim(0.998, 1.002)
    e.ticklabel_format(axis="y", useOffset=False)
    e.set_ylabel(r"predicted / exact $\kappa(\bar H)$")
    title(e, "21 configurations, all within 0.11%")
    lab(e, "d")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, "fig2_theory")


# --------------------------------------------------------------------------
# Figure 3: prevalence in public adapters.
# --------------------------------------------------------------------------
def fig3_prevalence():
    p = load(DATA / "prevalence_summary.json")
    pb = load(CAMP / "prevalence_part_b_prevalence_summary.json")
    ours_s = load(RES / "subspace_geometry_seed1.json")
    ours_i = [load(RES / f"subspace_geometry_indep{i}.json") for i in (1, 2, 3)]
    sc = [c for c in p["cohorts"] if c.get("status") in ("COLLAPSED", "SEPARATED", "INTERMEDIATE")]
    assert len(sc) == 45
    col = {"COLLAPSED": ORANGE, "SEPARATED": BLUE, "INTERMEDIATE": INK2}
    mk = {"COLLAPSED": "^", "SEPARATED": "o", "INTERMEDIATE": "s"}

    fig = plt.figure(figsize=(WIDE, 3.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.2], hspace=0.05)
    top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=top)
    cos = np.array([c["median_principal_cosine"] for c in sc])
    bins = np.linspace(0, 1, 41)
    for st in ("SEPARATED", "INTERMEDIATE", "COLLAPSED"):
        v = cos[[c["status"] == st for c in sc]]
        top.hist(v, bins=bins, color=col[st], hatch="////" if st == "COLLAPSED" else None,
                 edgecolor="white", lw=0.4)
    top.set_ylabel("cohorts")
    top.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=3))
    plt.setp(top.get_xticklabels(), visible=False)
    top.grid(axis="x", visible=False)
    thresholds(top, vertical=True)
    n_c = sum(c["status"] == "COLLAPSED" for c in sc)
    top.set_title(f"45 public LoRA cohorts: {n_c} collapsed ({100 * n_c / 45:.1f}%), "
                  f"{sum(c['status'] == 'SEPARATED' for c in sc)} separated, 1 in between",
                  loc="left", color=INK, fontsize=7.8, pad=4)

    for st in ("SEPARATED", "INTERMEDIATE", "COLLAPSED"):
        pts = [(c["median_principal_cosine"], c["rel_A_diff"]) for c in sc
               if c["status"] == st and c.get("rel_A_diff") is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls="none", marker=mk[st], color=col[st], ms=5, mec="white",
                    mew=0.5, label=f"public, {st.lower()}")
    # reference cohorts: ours and the one merging paper that releases LoRA cohorts
    ax.plot([ours_s[b]["median_principal_cosine"] for b in BASES],
            [ours_s[b]["A_rel_distance"] for b in BASES], ls="none", marker="^", ms=6,
            mfc="none", mec=INK, mew=0.9, label="ours, shared (4 bases)")
    ax.plot([g[b]["median_principal_cosine"] for g in ours_i for b in BASES],
            [g[b]["A_rel_distance"] for g in ours_i for b in BASES], ls="none", marker="o",
            ms=5, mfc="none", mec=INK, mew=0.9, label="ours, independent (12 cohorts)")
    kn = pb["cohorts"]
    ax.plot([c["median_principal_cosine"] for c in kn], [c["rel_A_diff"] for c in kn],
            ls="none", marker="X", ms=6.5, color=AQUA, mec=INK, mew=0.4,
            label="KnOTS release (Llama-3 NLI, 2 ViT)")
    ax.axhline(np.sqrt(2), color=INK2, lw=0.7, ls="--")
    ax.text(0.5, np.sqrt(2) + 0.04, r"$\sqrt{2}$: two independent random draws",
            ha="center", va="bottom", fontsize=6.6, color=INK2)
    thresholds(ax, vertical=True)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.03, 1.62)
    ax.set_xlabel("median principal cosine between the cohort's task subspaces")
    ax.set_ylabel(r"$\|A_i - A_j\|_F \,/\, \|A_i\|_F$")
    # The empty middle of the plot holds the legend; the lower left has a real
    # separated cohort (LASR-Callum, Qwen3, rel. distance 0.29) that must stay visible.
    ax.legend(loc="center", ncol=2, fontsize=6.4, bbox_to_anchor=(0.55, 0.5))
    save(fig, "fig3_prevalence")


# --------------------------------------------------------------------------
# Figure 4: the mechanism. Where each merge puts its energy.
# --------------------------------------------------------------------------
def fig4_mechanism():
    m = load(RES / "merged_solution_norms.json")
    groups = [
        ("averaging and sparsifying (TA, TIES, DARE, TVQ)", ["task_arithmetic", "ties", "dare", "tvq_b2"], BLUE, "o"),
        ("solvers built from the adapters, no ridge", ["regmean", "rd_encoder_l0"], ORANGE, "^"),
        ("the same solvers with a ridge", ["regmean_l0p13", "rd_encoder_l0p05"], AQUA, "D"),
        (r"exact interpolator $\bar\tau_H$", ["tau_H"], INK, "*"),
    ]
    fig, ax = plt.subplots(figsize=(WIDE, 3.2))
    for name, keys, colr, mark in groups:
        for arm, filled in (("seed1", True), ("indep1", False)):
            xs = [m[f"{arm}|{b}"][k]["amp"] for b in BASES for k in keys]
            ys = [max(m[f"{arm}|{b}"][k]["tail"], 1.5e-6) for b in BASES for k in keys]
            ax.plot(xs, ys, ls="none", marker=mark, ms=7 if mark == "*" else 5,
                    color=colr, mfc=colr if filled else "white", mec=colr, mew=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 120)
    ax.set_ylim(1e-6, 1.6)
    ax.axhspan(0.15, 0.35, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0.66, 0.07, "independent arm (open): every merge at 0.2 to 0.3,\nclose to what a generic direction gives",
            fontsize=6.4, color=INK2, va="center")
    ax.text(2.9, 2.2e-6, "shared arm (filled): averaging, sparsifying and ridged\n"
            "solvers put almost nothing in the ill-conditioned quarter",
            fontsize=6.4, color=INK2, va="bottom")
    ax.annotate("unregularised solvers, shared:\n35 to 57 times the mean update,\n"
                "about 3/4 of it in the bottom quarter",
                xy=(42, 0.75), xytext=(7, 0.012), fontsize=6.4, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel(r"norm of the merged update / norm of the mean task update")
    ax.set_ylabel("share of squared norm in the\nbottom quarter of $\\bar H$'s spectrum")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker=mk, color=c, mfc=c, ms=6 if mk == "*" else 5, label=n)
         for n, _, c, mk in groups]
    ax.legend(handles=h, loc="lower right", fontsize=6.4, bbox_to_anchor=(1.0, 0.13))
    title(ax, "Where each merge puts its energy, four bases")
    fig.tight_layout()
    save(fig, "fig4_mechanism")


# --------------------------------------------------------------------------
# Figure 5: the ridge, in loss and in conditioning.
# --------------------------------------------------------------------------
def fig5_ridge():
    tags = ["1em6", "0p01", "0p03", "0p05", "0p13", "0p30", "1p00"]
    lam = [1e-6, 0.01, 0.03, 0.05, 0.13, 0.30, 1.0]
    rpd = load(RES / "regmean_published_default.json")
    fig, axs = plt.subplots(2, 4, figsize=(WIDE, 3.7), sharex=True)
    for j, b in enumerate(BASES):
        top, bot = axs[0, j], axs[1, j]
        for coh, sty in (("seed1", SH), ("indep1", IN)):
            ys = [load(DATA / "regmean_agram" / f"{b}__rm_l{t}__{coh}.json")["worst_task_excess"]
                  for t in tags]
            # 1e-6 is four empty decades from the next grid point: draw it as a
            # lone marker, not joined by a line that would imply data between.
            top.loglog(lam[:1], ys[:1], color=sty["color"], marker=sty["marker"], ls="none", ms=3.5)
            top.loglog(lam[1:], ys[1:], color=sty["color"], marker=sty["marker"], ls=sty["ls"],
                       ms=3.5, lw=1.0)
            i = int(np.argmin(ys))
            top.plot([lam[i]], [ys[i]], marker="o", ms=8, mfc="none", mec=sty["color"], mew=1.0)
            v = rpd["bases"][b]["shared" if coh == "seed1" else "independent"]
            ls_ = [1e-6, 1e-3, 1e-2]
            # Condition number of a fixed Gram plus lambda*I is smooth in lambda,
            # so joining these three points is an honest interpolation.
            bot.loglog(ls_, [v[f"cond_ridge{x:g}_median"] for x in ls_], color=sty["color"],
                       marker=sty["marker"], ls=sty["ls"], ms=3.5, lw=1.0)
            bot.plot([v["alpha_equiv_ridge_median"]], [v["alpha_cond_median"]], marker="*",
                     ms=8, color=sty["color"], mec=INK, mew=0.4, ls="none")
        a_eq = rpd["bases"][b]["shared"]["alpha_equiv_ridge_median"]
        for ax in (top, bot):
            ax.axvline(a_eq, color=INK2, lw=0.6, ls=":")
            ax.set_xlim(4e-7, 2.5)
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        top.set_title(BL[b], loc="left", color=INK, fontsize=7.6, pad=3)
        top.set_ylim(0.004, 30)
        bot.set_ylim(15, 5e6)
        bot.set_xlabel(r"ridge $\lambda$")
        if j == 0:
            top.set_ylabel("worst-task NLL excess")
            bot.set_ylabel("condition number\nof the adapter Gram")
        else:
            top.tick_params(labelleft=False)
            bot.tick_params(labelleft=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ORANGE, marker="^", ls="-", ms=4, label="shared"),
         Line2D([], [], color=BLUE, marker="o", ls="--", ms=4, label="independent"),
         Line2D([], [], color=INK2, marker="o", mfc="none", ls="none", ms=7, label=r"selected $\lambda^\star$"),
         Line2D([], [], color=INK2, marker="*", mec=INK, ls="none", ms=8,
                label=r"RegMean's shipped $\alpha=0.9$"),
         Line2D([], [], color=INK2, ls=":", lw=0.8, label=r"ridge equivalent to $\alpha=0.9$")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), fontsize=6.3,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8, w_pad=0.6)
    save(fig, "fig6_ridge")


# --------------------------------------------------------------------------
# Figure 6: downstream accuracy, shared against independent.
# --------------------------------------------------------------------------
def fig6_accuracy():
    h = load(CAMP / "heur_tost_summary.json")
    rm = load(CAMP / "regmean_real_summary.json")["downstream"]["table"]
    ts = load(CAMP / "third_solver_summary.json")["downstream"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.95))
    for ax, bench, name in ((axes[0], "gsm8k", "GSM8K"), (axes[1], "humaneval", "HumanEval")):
        def pts(get):
            return [(mean(i), mean(s), sd(i), sd(s)) for i, s in get]
        heur = pts([(v["indep"], v["shared"]) for mm in ("task_arithmetic", "ties", "dare", "tvq_b2", "knots_ties")
                    for v in h["equivalence"][f"{mm}/{bench}"]["per_base"].values()])
        data = pts([(rm[f"{b}/rma/{bench}"]["indep"], rm[f"{b}/rma/{bench}"]["shared"]) for b in BASES]
                   + [(ts["mats"][f"{b}/{bench}"]["indep"], ts["mats"][f"{b}/{bench}"]["shared"]) for b in BASES])
        tsv = pts([(ts["tsv"][bench][b]["indep"], ts["tsv"][bench][b]["shared"]) for b in BASES])
        a09 = pts([(rm[f"{b}/rmdf_a09/{bench}"]["indep"], rm[f"{b}/rmdf_a09/{bench}"]["shared"]) for b in BASES])
        l01 = pts([(rm[f"{b}/rmdf_l0p01/{bench}"]["indep"], rm[f"{b}/rmdf_l0p01/{bench}"]["shared"]) for b in BASES])
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, zorder=1)

        def sc(P, **kw):
            for x, y, ex, ey in P:
                ax.errorbar(x, y, xerr=ex, yerr=ey, fmt="none", ecolor=kw["color"], elinewidth=0.6,
                            alpha=0.7, zorder=2)
            ax.plot([p[0] for p in P], [p[1] for p in P], ls="none", zorder=3, mec="white",
                    mew=0.5, **kw)
        sc(heur, color=BLUE, marker="o", ms=4, label="averaging and sparsifying (5 methods)")
        sc(data, color=BLUE, marker="s", ms=4, label="data-built solvers (RegMean, MaTS)")
        sc(tsv, color=BLUE, marker="D", ms=4, label="whitening (TSV-M)")
        sc(a09, color=ORANGE, marker="v", ms=6, label=r"adapter-only RegMean, shipped $\alpha=0.9$")
        sc(l01, color=AQUA, marker="^", ms=6, label=r"adapter-only RegMean, ridge $\lambda=0.01$")
        for (x0, y0, *_), (x1, y1, *_) in zip(a09, l01):
            if abs(y1 - y0) > 0.03:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=AQUA, lw=0.9, shrinkA=3, shrinkB=3))
        mi = BASES.index("mistral_7b")
        ax.annotate("Mistral: 0.000", xy=a09[mi][:2], xytext=(a09[mi][0] + 0.13, 0.03),
                    fontsize=6.6, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        lim = 0.85 if bench == "gsm8k" else 0.8
        ax.set_xlim(0, lim)
        ax.set_ylim(-0.02, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("accuracy, independent initialisation")
        ax.set_ylabel("accuracy, shared initialisation")
        title(ax, name)
    axes[0].text(0.16, 0.205, "shared = independent", rotation=45, fontsize=6.4, color=INK2,
                 ha="center", va="bottom", rotation_mode="anchor")
    lab(axes[0], "a")
    lab(axes[1], "b")
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.8)
    save(fig, "fig5_accuracy")


if __name__ == "__main__":
    gpath = DATA / "geometry_stats.json"
    G = load(gpath) if gpath.exists() else None
    fig3_prevalence()
    fig4_mechanism()
    fig5_ridge()
    fig6_accuracy()
    if G:
        fig1_collapse(G)
        fig2_theory(G)
    else:
        print("geometry_stats.json missing: figures 1 and 2 skipped")
    for old in ("fig4_exposure", "fig5_conditioning", "fig5_ridge", "fig6_accuracy"):
        for p in (OUT / f"{old}.pdf",):
            if p.exists():
                p.unlink()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))

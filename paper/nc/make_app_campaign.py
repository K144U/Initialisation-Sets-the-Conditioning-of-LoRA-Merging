#!/usr/bin/env python3
"""Write paper/nc/app_campaign.tex: Appendix C, the campaign detail tables.

Every number in the output is read from a result file under results/phase3
and formatted here; none is typed in. Run from anywhere:

    python paper/nc/make_app_campaign.py

Inputs (all relative to the repository root):
  results/phase3/campaign_2026-10/training_regimes_summary.json   (C.1)
  results/phase3/campaign_2026-10/regimes_geometry.json           (C.1)
  results/phase3/regimes_calibration.json                         (C.1, C_k and C_a)
  results/phase3/eval_regimes/llama31_8b__rm1em6__<cohort>.json   (C.1, optional:
      the RegMean column is written only when every regime cell is present)
  results/phase3/campaign_2026-10/heterogeneity_summary.json      (C.2)
  results/phase3/campaign_2026-10/lambda_selection_summary.json   (C.3)
  results/phase3/campaign_2026-09/regmean_agram_summary.json      (C.3, sweep)
  results/phase3/campaign_2026-10/vision_summary.json             (C.4)
  results/phase3/campaign_2026-10/all_tasks_summary.json          (C.5, contrasts)
  results/phase3/eval_all_tasks/*.json                            (C.5, scores; the
      registered differences, IFEval gates and reference rows are asserted equal)
  results/phase3/eval_ridge_cond/<base>__rd_l*__<cohort>.json     (C.5, lambda* check)
  results/phase3/campaign_2026-10/solve_precision.json            (C.6)
  results/phase3/eval_ridge_cond/<base>__rd_l0__{seed1,indep1}.json and
  results/phase3/eval_precision_f64/*.json                        (C.7)

Set RDMERGE_ROOT to read from (and write into) another checkout; this is only
for testing.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results" / "phase3"
CAMP = RES / "campaign_2026-10"
OUT = ROOT / "paper" / "nc" / "app_campaign.tex"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BASE_NAME = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B",
             "qwen25_7b": "Qwen2.5-7B", "yi15_9b": "Yi-1.5-9B"}

WARNINGS: list[str] = []


# ---------------------------------------------------------------- helpers

def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def fx(x: float, n: int = 3) -> str:
    """Fixed decimals; negative values go in math mode for a proper minus."""
    s = f"{x:.{n}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return f"${s}$" if s.startswith("-") else s


def sci(x: float, sig: int = 3) -> str:
    e = math.floor(math.log10(abs(x)))
    mant = f"{x / 10 ** e:.{sig - 1}f}"
    if abs(float(mant)) >= 10:
        e += 1
        mant = f"{x / 10 ** e:.{sig - 1}f}"
    return f"${mant}\\times10^{{{e}}}$"


def kap(x: float) -> str:
    """Condition numbers: scientific from 1000 up, two decimals below."""
    return sci(x, 3) if x >= 1000 else f"{x:.2f}"


def g3(x: float) -> str:
    return f"{x:#.3g}"


def tag_value(tag: str) -> float:
    """Sweep tags of eval_regmean_agram: '1em6' -> 1e-6, '0p03' -> 0.03."""
    if "em" in tag:
        m, e = tag.split("em")
        return float(m) * 10 ** (-int(e))
    return float(tag.replace("p", "."))


def lam(x: float) -> str:
    if x < 1e-3:
        m = x / 10 ** math.floor(math.log10(x))
        if close(m, 1.0, 1e-12):
            return f"$10^{{{math.floor(math.log10(x))}}}$"
        return sci(x, 2)
    return f"{x:g}"


def table(label: str, caption: str, body: list[str], pre: str = "") -> str:
    out = ["\\begin{table}[pos=tbp]", "\\centering", "\\small",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if pre:
        out.append(pre)
    out += body
    out.append("\\end{table}")
    return "\n".join(out)


# ---------------------------------------------------------------- C.1

REGIMES_SHARED = [("baseline", "s_sast"),
                  ("learning rate $/3$", "k_lr033"),
                  ("learning rate $\\times3$", "k_lr3"),
                  ("1 epoch", "k_ep1"),
                  ("4 epochs", "k_ep4"),
                  ("$A$ learning rate $\\times4$", "k_amult4"),
                  ("LoRA+, ratio 16", "k_loraplus16"),
                  ("$A_0\\times0.25$", "k_scale025")]
REGIMES_INDEP = [("learning rate $\\times3$", "ki_lr3"),
                 ("$A$ learning rate $\\times4$", "ki_amult4"),
                 ("unchanged", "indep1")]
ROUTES = [("continued training", "c_cont"),
          ("DoRA, shared", "d_shared"),
          ("DoRA, independent", "d_indep")]


def regime_regmean(tr: dict) -> dict | None:
    """Adapter-only RegMean(1e-6) excess per regime cohort, or None if any of
    the regime cells is absent (the summary file stores only Parts S and C)."""
    vals = {"s_sast": tr["S"]["rm1em6"]["s_sast"], "indep1": tr["S"]["rm1em6"]["indep1"],
            "c_cont": tr["C"]["rm1em6"]}
    missing = []
    for _, c in REGIMES_SHARED[1:] + REGIMES_INDEP[:2]:
        p = RES / "eval_regimes" / f"llama31_8b__rm1em6__{c}.json"
        if p.exists():
            vals[c] = load(p)["worst_task_excess"]
        else:
            missing.append(str(p.relative_to(ROOT)))
    for c in ("s_sast", "c_cont"):   # summary values must match their cells when present
        p = RES / "eval_regimes" / f"llama31_8b__rm1em6__{c}.json"
        if p.exists():
            assert close(load(p)["worst_task_excess"], vals[c]), f"{c} RegMean mismatch"
    if missing:
        WARNINGS.append("C.1: RegMean column omitted; missing " + ", ".join(missing))
        return None
    return vals


def c1() -> str:
    tr = load(CAMP / "training_regimes_summary.json")
    geo = load(CAMP / "regimes_geometry.json")
    cal = load(RES / "regimes_calibration.json")
    K = tr["K"]
    ck, ca = cal["C_k"], cal["C_a"]
    kok = aok = 0
    for _, c in REGIMES_SHARED:
        g = geo[c]
        assert g == K["geometry"][c], c
        p = K["pred"][c]
        assert close(p["kappa_pred"], ck / g["t"] ** 2) and close(p["amp_pred"], ca / g["t"]), c
        kok += 0.5 <= g["kappa"] / p["kappa_pred"] <= 2
        aok += 1 / 1.25 <= g["amp"] / p["amp_pred"] <= 1.25
    assert kok == K["kappa_within_2x"] and aok == K["amp_within_1.25x"]
    for c in ("ki_lr3", "ki_amult4"):
        assert geo[c] == K["indep"][c], c
    assert geo["indep1"] == tr["S"]["geometry"]["indep1"]
    assert geo["c_cont"] == tr["C"]["geometry"]
    assert geo["d_shared"] == tr["D"]["shared"] and geo["d_indep"] == tr["D"]["indep"]

    rm = regime_regmean(tr)
    ncol = 10 if rm else 9
    dash = "--"

    def row(label, c, pred):
        g = geo[c]
        cells = [label, f"{g['median_cos']:.3f}", g3(g["t"]), kap(g["kappa"])]
        if pred:
            p = K["pred"][c]
            cells += [kap(p["kappa_pred"]), f"{g['kappa'] / p['kappa_pred']:.2f}",
                      f"{g['amp']:.1f}", f"{p['amp_pred']:.1f}", f"{g['amp'] / p['amp_pred']:.2f}"]
        else:
            cells += [dash, dash, f"{g['amp']:.1f}", dash, dash]
        if rm:
            cells.append(f"{rm[c]:.3f}" if c in rm else dash)
        return " & ".join(cells) + " \\\\"

    colspec = "l" + "c" * (ncol - 1)
    head2 = ("regime & cosine & $t$ & exact & law & ratio & exact & law & ratio"
             + (" & excess" if rm else "") + " \\\\")
    body = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
            " & median & & \\multicolumn{3}{c}{$\\kappa(\\bar H)$} & \\multicolumn{3}{c}{amplification}"
            + (" & RegMean" if rm else "") + " \\\\",
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}", head2, "\\midrule",
            f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{shared initialisation}}}} \\\\"]
    body += [row(lbl, c, True) for lbl, c in REGIMES_SHARED]
    body += ["\\addlinespace", f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{independent initialisation}}}} \\\\"]
    body += [row(lbl, c, False) for lbl, c in REGIMES_INDEP]
    body += ["\\addlinespace",
             f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{other routes (\\S\\ref{{subsec:routes}})}}}} \\\\"]
    body += [row(lbl, c, False) for lbl, c in ROUTES]
    body += ["\\bottomrule", "\\end{tabular}"]

    rm_txt = (" RegMean excess is the worst-task excess of adapter-only RegMean at "
              "$\\lambda = 10^{-6}$; DoRA was not merged.") if rm else ""
    cap = ("Training regimes on Llama-3.1-8B, one cohort of four adapters each "
           "(\\S\\ref{subsec:tr}). $t$ is the median over tasks and layers of the spectral "
           "norm of the tilt under the consensus $W_0$. \\emph{Law} is the scalar law of "
           f"\\S\\ref{{sec:theory}}, $C_\\kappa/t^2$ and $C_a/t$ with $C_\\kappa = {ck:.1f}$ and "
           f"$C_a = {ca:.2f}$, fixed on four earlier shared cohorts before these were trained; "
           "\\emph{ratio} is exact over law, and the registered bands for it were $[0.5, 2]$ "
           "for $\\kappa$ and $[0.8, 1.25]$ for the amplification. The law covers shared "
           "cohorts only. The unchanged independent cohort is the first independent cohort of "
           "the main grid. Continued training starts four task adapters from one existing "
           "adapter, each with its own seed (\\S\\ref{subsec:routes})."
           + rm_txt)
    return table("tab:appC-regimes", cap, body)


# ---------------------------------------------------------------- C.2

PERTS = [("gauge", "gauge"), ("canonical form", "canon"), ("scale", "scale"),
         ("rank", "rank"), ("module", "module")]


def c2() -> str:
    h = load(CAMP / "heterogeneity_summary.json")
    G, M = h["G"], h["M"]
    a = ["\\begin{tabular}{llcccccc}", "\\toprule",
         " & & \\multicolumn{2}{c}{RegMean excess} & & & & \\\\",
         "\\cmidrule(lr){3-4}",
         "base & perturbation & perturbed & unperturbed & ratio & $d$ & gate"
         " & task arithmetic ratio \\\\", "\\midrule"]
    for i, b in enumerate(["llama31_8b", "qwen25_7b"]):
        if i:
            a.append("\\addlinespace")
        for j, (lbl, k) in enumerate(PERTS):
            e = G[f"{b}/{k}"]
            mr, mf = st.mean(e["rm"]), st.mean(e["ref"])
            assert close(e["ratio"], mr / mf), (b, k)
            assert close(e["d"], st.mean(x - y for x, y in zip(e["rm"], e["ref"]))), (b, k)
            ta = f"{e['ta_ratio']:.2f}" if "ta_ratio" in e else "--"
            a.append(" & ".join([BASE_NAME[b] if j == 0 else "", lbl, f"{mr:.3f}", f"{mf:.3f}",
                                 f"{e['ratio']:.2f}", fx(e["d"], 4), f"{e['gate']:.4f}", ta]) + " \\\\")
    a += ["\\bottomrule", "\\end{tabular}"]

    bpanel = ["\\begin{tabular}{llccccc}", "\\toprule",
              " & & \\multicolumn{2}{c}{principal cosine} & & \\multicolumn{2}{c}{excess} \\\\",
              "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}",
              "base & cohort & median & max.\\ pair & $\\kappa(\\bar H)$"
              " & RegMean & task arithmetic \\\\", "\\midrule"]
    lbls = [("M1, one pair", "M1"), ("M2, three pairs", "M2"), ("independent", "indep1")]
    for i, b in enumerate(BASES):
        if i:
            bpanel.append("\\addlinespace")
        assert M[f"{b}/M1"]["rm_indep1"] == M[f"{b}/M2"]["rm_indep1"], b
        for j, (lbl, k) in enumerate(lbls):
            e = M[f"{b}/{k}"]
            rmv = e["rm"] if k != "indep1" else M[f"{b}/M1"]["rm_indep1"]
            tav = f"{e['ta']:.3f}" if "ta" in e else "--"
            bpanel.append(" & ".join([BASE_NAME[b] if j == 0 else "", lbl, f"{e['median_cos']:.3f}",
                                      f"{e['max_pair_cos']:.3f}", kap(e["kappa_hbar_median"]),
                                      f"{rmv:.3f}", tav]) + " \\\\")
    bpanel += ["\\bottomrule", "\\end{tabular}"]

    body = ["\\textbf{(a)} perturbations of separated cohorts\\par\\smallskip"] + a + \
           ["\\par\\bigskip", "\\textbf{(b)} partial collapse\\par\\smallskip"] + bpanel
    cap = ("What breaks adapter-only RegMean at $\\lambda = 10^{-6}$ (\\emph{RegMean}). "
           "\\textbf{(a)} Perturbations of the three independent cohorts of two bases "
           "(\\S\\ref{subsec:against}). Gauge: the GSM8K adapter's $A$ multiplied by $4$ and "
           "its $B$ divided by $4$, which leaves its update unchanged. Canonical form: every "
           "adapter refactorised into balanced SVD form, also update-preserving. Scale: the "
           "GSM8K adapter's $B$ multiplied by $4$. Rank: canonical form, then the GSM8K and "
           "code adapters truncated to rank $4$. Module: the GSM8K adapter's \\texttt{k\\_proj} "
           "and \\texttt{v\\_proj} updates removed. Excesses are means over the three cohorts, "
           "the ratio is perturbed over unperturbed, $d$ is the mean paired difference, and the "
           "gate is the larger of $0.005$ and two standard errors of $d$. Task arithmetic was "
           "not run for the two update-preserving perturbations. \\textbf{(b)} Cohorts built "
           "from existing adapters, one per base (\\S\\ref{subsec:audit}). M1: the GSM8K and "
           "code adapters of the seed-1 shared cohort with the instruction and translation "
           "adapters of the first independent cohort, so one of the six pairs shares an "
           "initialisation. M2: three adapters from the shared cohort and the translation "
           "adapter from the independent one, three pairs. Median cosine pools every principal "
           "cosine of every pair; the maximum is over pairs of the per-pair median. "
           "$\\kappa(\\bar H)$ is the median over the \\texttt{q\\_proj} modules. Task "
           "arithmetic was not run on the unmodified independent cohort in this experiment.")
    return table("tab:appC-hetero", cap, body)


# ---------------------------------------------------------------- C.3

def c3() -> str:
    ls = load(CAMP / "lambda_selection_summary.json")["per_rule"]
    ag = load(RES / "campaign_2026-09" / "regmean_agram_summary.json")["described_gram_AtA"]
    body = ["\\begin{tabular}{llccccccc}", "\\toprule",
            " & & \\multicolumn{2}{c}{GCV} & \\multicolumn{2}{c}{L-curve}"
            " & $\\lambda = 10^{-6}$ & \\multicolumn{2}{c}{best of sweep} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-9}",
            "base & arm & $\\lambda$ & excess & $\\lambda$ & excess & excess & $\\lambda$ & excess \\\\",
            "\\midrule"]
    for i, b in enumerate(BASES):
        g, lc, a = ls["gcv"]["per_base"][b], ls["lcurve"]["per_base"][b], ag[b]
        for r in (g, lc):
            assert close(r["min_ridge_shared"], a["seed1"]["l_min"]), b
            assert close(r["oracle_shared"], a["seed1"]["l_min"] - a["seed1"]["gain"], 1e-12), b
        if i:
            body.append("\\addlinespace")
        # rule_shared / rule_indep are ordered seed1-3 / indep1-3 by the analyzer,
        # so element 0 is the seed1 / indep1 cohort the sweep columns refer to.
        for j, (arm, coh, k) in enumerate([("shared", "seed1", "rule_shared"),
                                            ("indep.", "indep1", "rule_indep")]):
            x = a[coh]
            body.append(" & ".join([
                BASE_NAME[b] if j == 0 else "", arm,
                sci(g[f"lambda_median_{coh}"], 2), f"{g[k][0]:.3f}",
                sci(lc[f"lambda_median_{coh}"], 2), f"{lc[k][0]:.3f}",
                f"{x['l_min']:.3f}", lam(tag_value(x["lstar"])),
                f"{x['l_min'] - x['gain']:.3f}"]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    cap = ("Data-free choice of the ridge for adapter-only RegMean (\\S\\ref{subsec:lambda}), "
           "on the seed-1 shared cohort and the first independent cohort of each base. GCV and "
           "the L-curve choose $\\lambda$ per layer on a logarithmic grid from $10^{-8}$ to "
           "$10^{2}$; the table gives the median over layers and the worst-task excess of the "
           "resulting merge. The sweep is the seven-point grid of Figure~\\ref{fig:ridge}, and "
           "\\emph{best of sweep} is its lowest excess, which needs data to find.")
    return table("tab:appC-lambda", cap, body)


# ---------------------------------------------------------------- C.4

def vision_label(k: str) -> str:
    fixed = {"ta": "task arithmetic", "ties": "TIES", "tsv": "TSV-M",
             "rma": "RegMean, activation Grams", "rm_a09": "adapter-only RegMean, $\\alpha = 0.9$",
             "wudi_pub": "WUDI, as published", "wudi_cf": "WUDI, closed form"}
    if k in fixed:
        return fixed[k]
    return f"adapter-only RegMean, $\\lambda = {lam(vision_lambda(k)).strip('$')}$"


def vision_lambda(k: str) -> float:
    """rm_1em6, rm_1em3, rm_1em2, rm_3em2, rm_1em1, rm_3em1, rm_1 -> lambda."""
    return tag_value(k[len("rm_"):])


def c4() -> str:
    v = load(CAMP / "vision_summary.json")
    C = v["contrasts"]
    order = [["ta", "ties"], ["tsv", "rma"],
             ["rm_a09"] + [k for k in C if k.startswith("rm_") and k != "rm_a09"],
             ["wudi_pub", "wudi_cf"]]
    assert sorted(sum(order, [])) == sorted(C), "vision methods not all placed"

    def ms(xs):
        return f"${st.mean(xs):.3f} \\pm {st.stdev(xs):.3f}$"

    body = ["\\begin{tabular}{lcccc}", "\\toprule",
            "method & shared & independent & $d$ & gate \\\\", "\\midrule"]
    for gi, grp in enumerate(order):
        if gi:
            body.append("\\addlinespace")
        for k in grp:
            e = C[k]
            assert close(e["d"], st.mean(e["indep"]) - st.mean(e["shared"]), 1e-9), k
            body.append(" & ".join([vision_label(k), ms(e["shared"]), ms(e["indep"]),
                                    fx(e["d"]), f"{e['gate']:.3f}"]) + " \\\\")
    geo = v["geometry"]
    sh = ", ".join(f"{geo[f'shared{i}']:.3f}" for i in (1, 2, 3))
    ind = ", ".join(f"{geo[f'indep{i}']:.3f}" for i in (1, 2, 3))
    body += ["\\midrule", f"median cosine, cohorts 1, 2, 3 & {sh} & {ind} & & \\\\",
             "\\bottomrule", "\\end{tabular}"]
    best = v["best_lambda"]
    assert best["shared"] == best["indep"]
    cap = ("CLIP ViT-B/32 on eight image classification tasks, three cohorts per arm "
           "(\\S\\ref{subsec:beyond}). Mean top-1 accuracy over the eight tasks, given as the "
           "mean $\\pm$ one standard deviation across cohorts; merges are applied as dense "
           "updates without truncation. $d$ is independent minus shared, and the gate is the "
           "larger of $0.01$ and twice the standard error of $d$. The best ridge on the grid is "
           f"$\\lambda = {lam(vision_lambda(best['shared'])).strip('$')}$ on both arms.")
    return table("tab:appC-vision", cap, body)


# ---------------------------------------------------------------- C.5

AT_METHODS = [("enc_l0", "rd_encoder"), ("enc_lstar", "rd_encoder"),
              ("task_arithmetic", "task_arithmetic"), ("ties", "ties")]
AT_METRICS = [("WMT19 chrF", "wmt_chrf", 1), ("IFEval, strict", "ifeval_strict", 3)]
AT_REF = [("GSM8K", "gsm8k", "ind_gsm8k", "GSM8K adapter", 3),
          ("HumanEval", "humaneval", "ind_magicoder", "code adapter", 3),
          ("WMT19 chrF", "wmt_chrf", "ind_translation", "translation adapter", 1),
          ("IFEval, strict", "ifeval_strict", "ind_alpaca", "instruction adapter", 3)]
# adapter directory of each single-adapter reference row
AT_OWN_DIR = {"ind_gsm8k": "gsm8k", "ind_alpaca": "alpaca", "ind_magicoder": "magicoder",
              "ind_translation": "flores"}
RIDGE_TAGS = ["l0", "l0p01", "l0p03", "l0p05", "l0p13", "l0p30", "l1p00"]


def lamz(x: float) -> str:
    return "0" if x == 0 else lam(x)


def all_task_cells() -> tuple[dict, dict, dict, dict]:
    """Raw scores from results/phase3/eval_all_tasks, checked against the
    registered summary (differences, IFEval gates, reference rows) and against
    the NLL ridge sweep that chose lambda*."""
    a = load(CAMP / "all_tasks_summary.json")
    cells, refsum = a["cells"], a["reference_rows"]
    D = RES / "eval_all_tasks"
    raw, lstar, n_eval = {}, {}, {}
    for b in BASES:
        for _, m, _ in AT_METRICS:
            for meth, mname in AT_METHODS:
                for coh in ("seed1", "indep1"):
                    d = load(D / f"{b}__{meth}_{m}__{coh}.json")
                    assert d["metric"] == m and d["method"] == mname, (b, m, meth, coh)
                    assert {Path(x).name for x in d["adapter_dirs"]} == {coh}, (b, m, meth, coh)
                    assert BASE_NAME[b].split("-")[0] in d["base_model"], (b, d["base_model"])
                    if meth == "enc_l0":
                        assert d["method_kwargs"]["ridge_lambda"] == 0.0
                    if meth == "enc_lstar":
                        lstar.setdefault((b, coh), set()).add(d["method_kwargs"]["ridge_lambda"])
                    raw[(b, m, meth, coh)] = d["metric_score"]
                    n_eval.setdefault(m, set()).add(d["n_eval_metric"])
            # the registered differences must be reproduced from the raw cells
            e = cells[f"{b}/{m}/enc_gain_seed1"]
            dd = raw[(b, m, "enc_lstar", "seed1")] - raw[(b, m, "enc_l0", "seed1")]
            assert abs(dd - e["d"]) <= 1e-9, (b, m, "enc", dd, e["d"])
            for h in ("task_arithmetic", "ties"):
                e = cells[f"{b}/{m}/{h}_shared_minus_indep"]
                pa, pb = raw[(b, m, h, "seed1")], raw[(b, m, h, "indep1")]
                assert abs((pa - pb) - e["d"]) <= 1e-9, (b, m, h, pa - pb, e["d"])
            if m == "ifeval_strict":   # binomial gate is reproducible without resampling
                n = next(iter(n_eval[m]))
                for key, (pa, pb) in {
                        "enc_gain_seed1": (raw[(b, m, "enc_lstar", "seed1")], raw[(b, m, "enc_l0", "seed1")]),
                        "task_arithmetic_shared_minus_indep": (raw[(b, m, "task_arithmetic", "seed1")],
                                                               raw[(b, m, "task_arithmetic", "indep1")]),
                        "ties_shared_minus_indep": (raw[(b, m, "ties", "seed1")], raw[(b, m, "ties", "indep1")])
                }.items():
                    g = max(0.05, 2 * math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n))
                    assert abs(g - cells[f"{b}/{m}/{key}"]["gate"]) <= 1e-12, (b, key)
    for k, s in n_eval.items():
        assert len(s) == 1, (k, s)
    # one lambda* per base and arm, shared by both benchmarks, equal to the
    # argmin of the worst-task NLL excess over the encoder's ridge sweep
    ls = {}
    for (b, coh), s in lstar.items():
        assert len(s) == 1, (b, coh, s)
        sweep = {}
        for t in RIDGE_TAGS:
            d = load(RES / "eval_ridge_cond" / f"{b}__rd_{t}__{coh}.json")
            sweep[d["method_kwargs"]["ridge_lambda"]] = d["worst_task_excess"]
        ls[(b, coh)] = next(iter(s))
        assert ls[(b, coh)] == min(sweep, key=sweep.get), (b, coh)
    ref = {}
    for b in BASES:
        for _, m, own, _, _ in AT_REF:
            for tag in ("base", own):
                d = load(D / f"{b}__{tag}_{m}.json")
                dirs = [Path(x) for x in d["adapter_dirs"]]
                assert all(p.name == "seed1" for p in dirs), (b, tag, m)
                want = [1.0 if (tag != "base" and p.parent.name == AT_OWN_DIR[tag]) else 0.0 for p in dirs]
                assert d["weights"] == want, (b, tag, m, d["weights"])
                assert d["metric_score"] == refsum[f"{b}/{m}/{tag}"], (b, tag, m)
                ref[(b, m, tag)] = d["metric_score"]
    return raw, ls, {k: next(iter(v)) for k, v in n_eval.items()}, ref


def c5() -> str:
    a = load(CAMP / "all_tasks_summary.json")
    cells = a["cells"]
    raw, ls, n_eval, ref = all_task_cells()

    def dg(b, metric, key, n):
        e = cells[f"{b}/{metric}/{key}"]
        return f"{fx(e['d'], n)} ({e['gate']:.{n}f})"

    body = ["\\textbf{(a)} scores\\par\\smallskip",
            "\\begin{tabular}{lccccccccc}", "\\toprule",
            " & & \\multicolumn{2}{c}{encoder, $\\lambda = 0$}"
            " & \\multicolumn{2}{c}{encoder, $\\lambda^\\star$}"
            " & \\multicolumn{2}{c}{task arithmetic} & \\multicolumn{2}{c}{TIES} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
            "base & $\\lambda^\\star$ & shared & indep. & shared & indep. & shared & indep."
            " & shared & indep. \\\\", "\\midrule"]
    for i, (name, m, n) in enumerate(AT_METRICS):
        if i:
            body.append("\\addlinespace")
        body.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{name}}}}} \\\\")
        for b in BASES:
            lst = f"{lamz(ls[(b, 'seed1')])} / {lamz(ls[(b, 'indep1')])}"
            body.append(" & ".join([BASE_NAME[b], lst]
                                   + [f"{raw[(b, m, meth, coh)]:.{n}f}"
                                      for meth, _ in AT_METHODS for coh in ("seed1", "indep1")])
                        + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\par\\bigskip",
             "\\textbf{(b)} registered contrasts\\par\\smallskip",
             "\\begin{tabular}{lcccccc}", "\\toprule",
            " & \\multicolumn{3}{c}{WMT19 chrF} & \\multicolumn{3}{c}{IFEval, strict} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "base & encoder & TA & TIES & encoder & TA & TIES \\\\", "\\midrule"]
    for b in BASES:
        body.append(" & ".join([BASE_NAME[b]]
                               + [dg(b, "wmt_chrf", k, 1) for k in
                                  ("enc_gain_seed1", "task_arithmetic_shared_minus_indep",
                                   "ties_shared_minus_indep")]
                               + [dg(b, "ifeval_strict", k, 3) for k in
                                  ("enc_gain_seed1", "task_arithmetic_shared_minus_indep",
                                   "ties_shared_minus_indep")]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\par\\bigskip",
             "\\textbf{(c)} reference rows\\par\\smallskip",
             "\\begin{tabular}{llcccc}", "\\toprule",
             "benchmark & model & " + " & ".join(BASE_NAME[b] for b in BASES) + " \\\\",
             "\\midrule"]
    for i, (name, m, own, own_lbl, n) in enumerate(AT_REF):
        if i:
            body.append("\\addlinespace")
        body.append(" & ".join([name, "base model"]
                               + [f"{ref[(b, m, 'base')]:.{n}f}" for b in BASES]) + " \\\\")
        body.append(" & ".join(["", own_lbl]
                               + [f"{ref[(b, m, own)]:.{n}f}" for b in BASES]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]

    # cohorts whose selected ridge is 0: their lambda = 0 and lambda* cells share
    # one configuration but were evaluated separately, so say how far apart they are
    same = []
    for b in BASES:
        for coh, arm in (("seed1", "shared"), ("indep1", "independent")):
            if ls[(b, coh)] == 0:
                gaps = [f"{abs(raw[(b, m, 'enc_lstar', coh)] - raw[(b, m, 'enc_l0', coh)]):.{n}f}"
                        for _, m, n in AT_METRICS]
                same.append(f"On {BASE_NAME[b]} the selected ridge of the {arm} cohort is $0$, so "
                            "its two encoder columns have one configuration; they are separate "
                            f"evaluations and differ by {gaps[0]} chrF points and {gaps[1]} on "
                            "IFEval. ")
    cap = ("The two tasks not scored elsewhere (\\S\\ref{subsec:beyond}), on the seed-1 shared "
           f"and first independent cohorts: chrF on {n_eval['wmt_chrf']} WMT19 English-to-German "
           f"pairs and IFEval strict prompt-level accuracy on {n_eval['ifeval_strict']} prompts. "
           "\\textbf{(a)} Scores of our encoder at $\\lambda = 0$ and at its selected ridge "
           "$\\lambda^\\star$, and of task arithmetic and TIES. $\\lambda^\\star$ (shared / "
           "independent) is the ridge with the lowest worst-task NLL excess in the encoder's "
           "ridge sweep on that cohort. " + "".join(same) +
           "\\textbf{(b)} The registered contrasts, each with its gate in parentheses. "
           "\\emph{Encoder}: $\\lambda^\\star$ minus $\\lambda = 0$ on the shared cohort. "
           "\\emph{TA}, \\emph{TIES}: shared minus independent. The chrF gate is the larger of "
           "$2$ points and two bootstrap standard errors, the IFEval gate the larger of $0.05$ "
           "and two binomial standard errors. \\textbf{(c)} The base model without an adapter "
           "and the task's own seed-1 adapter alone, on all four benchmarks (GSM8K exact match, "
           "HumanEval pass@1, chrF, IFEval strict).")
    return table("tab:appC-alltasks", cap, body)


# ---------------------------------------------------------------- C.6

def c6() -> str:
    sp = load(CAMP / "solve_precision.json")
    thr = sp["thresholds"]
    rows = {(r["base"], r["arm"], r["method"]): r for r in sp["rows"]}
    assert len(rows) == 16
    for (b, arm, m), r in rows.items():
        c = sp["cases"][f"{b}/{arm}"][m]
        assert close(r["rel_trunc_median"], c["rel_trunc"]["median"])
        assert close(r["rel_trunc_max"], c["rel_trunc"]["max"])
        assert close(r["prod_gap_median"], c["prod_gap"]["median"])
        assert r["pass"] == (r["rel_trunc_median"] < thr["rel_trunc_median_below"]
                             and thr["ratio_band"][0] <= r["amp_ratio_64"] / r["amp_ratio_32"]
                             <= thr["ratio_band"][1])

    def rel(r):
        s = sci(r["rel_trunc_median"], 2)
        return s[:-1] + "{}^{\\dagger}$" if not r["pass"] else s

    body = ["\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llcccccccc}", "\\toprule",
            " & & \\multicolumn{2}{c}{shared, rel.} & \\multicolumn{2}{c}{independent, rel.}"
            " & \\multicolumn{2}{c}{amp.\\ ratio} & \\multicolumn{2}{c}{production gap} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
            "base & solver & median & max & median & max & float32 & 64/32 & shared & indep. \\\\",
            "\\midrule"]
    meth = [("encoder_l0", "encoder"), ("regmean_1e-6", "RegMean")]
    for i, b in enumerate(BASES):
        if i:
            body.append("\\addlinespace")
        for j, (m, lbl) in enumerate(meth):
            s, d = rows[(b, "seed1", m)], rows[(b, "indep1", m)]
            assert s["amp_ratio_32"] == d["amp_ratio_32"] and s["amp_ratio_64"] == d["amp_ratio_64"]
            body.append(" & ".join([
                BASE_NAME[b] if j == 0 else "", lbl,
                rel(s), sci(s["rel_trunc_max"], 2), rel(d), sci(d["rel_trunc_max"], 2),
                f"{s['amp_ratio_32']:.2f}", f"{s['amp_ratio_64'] / s['amp_ratio_32']:.4f}",
                f"{s['prod_gap_median']:.2f}", f"{d['prod_gap_median']:.2f}"]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    lo, hi = thr["ratio_band"]
    below = lam(thr["rel_trunc_median_below"])
    cap = ("The registered precision check (\\S\\ref{subsec:results}) for our encoder at "
           "$\\lambda = 0$ and adapter-only RegMean at $\\lambda = 10^{-6}$. Each merge on the "
           "seed-1 shared and first independent cohorts was recomputed in float64 from the same "
           "adapter factors. \\emph{Rel.}: relative Frobenius difference between the "
           "float32 and float64 merges after exact truncation to rank $r$, median and maximum "
           "over modules. \\emph{Amp.\\ ratio}: the median over modules of the amplification "
           "(norm of the merge over that of task arithmetic) on the shared cohort, over the same "
           "on the independent one, in float32; and its float64 value divided by the float32 "
           "one. \\emph{Production gap}: median relative distance between the randomised "
           "truncation every merge in the paper uses and the exact rank-$r$ truncation. The "
           "rule excluded precision as the cause if, for every base, arm and solver, the "
           f"median was below {below} and the ratio of the two amplification ratios lay in "
           f"$[{lo:g}, {hi:g}]$; otherwise the affected merges were to be rebuilt in float64. "
           "$^\\dagger$Above the threshold.")
    return table("tab:appC-precision", cap, body)


# ---------------------------------------------------------------- C.7

def c7() -> str:
    rc, pf = RES / "eval_ridge_cond", RES / "eval_precision_f64"
    body = ["\\textbf{(a)} float32 encoder, production truncation\\par\\smallskip",
            "\\begin{tabular}{lcccccc}", "\\toprule",
            "base & published & \\texttt{rng\\_seed} 1 & \\texttt{rng\\_seed} 2 & range"
            " & independent & gap \\\\", "\\midrule"]
    indep = {}
    for b in BASES:
        pub = load(rc / f"{b}__rd_l0__seed1.json")
        ind = load(rc / f"{b}__rd_l0__indep1.json")
        draws = [pub] + [load(pf / f"{b}__rd_l0_rerun_s{k}__seed1.json") for k in (1, 2)]
        for k, d in enumerate(draws):
            assert d["nll_tau"] == pub["nll_tau"], (b, k)
            assert d["method_kwargs"].get("rng_seed") == (None if k == 0 else k), (b, k)
            assert d["method_kwargs"].get("ridge_lambda") == 0.0 and d["method_kwargs"].get("bits") == 32
            assert "truncation" not in d["method_kwargs"] and "solve_dtype" not in d["method_kwargs"]
        w = [d["worst_task_excess"] for d in draws]
        iw = ind["worst_task_excess"]
        assert all(x > iw for x in w), b
        indep[b] = iw
        body.append(" & ".join([BASE_NAME[b]] + [f"{x:.3f}" for x in w]
                               + [f"{max(w) - min(w):.3f}", f"{iw:.3f}", f"{min(w) - iw:.3f}"]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\par\\bigskip",
             "\\textbf{(b)} Yi-1.5-9B shared cohort, encoder at $\\lambda = 0$, all realisations"
             "\\par\\smallskip",
             "\\begin{tabular}{lccc}", "\\toprule",
             "realisation & default stream & \\texttt{rng\\_seed} 1 & \\texttt{rng\\_seed} 2 \\\\",
             "\\midrule"]
    b = "yi15_9b"
    pub = load(rc / f"{b}__rd_l0__seed1.json")

    def yi(name, kw):
        d = load(pf / f"{b}__{name}__seed1.json")
        assert d["nll_tau"] == pub["nll_tau"], name
        for k, v in kw.items():
            assert d["method_kwargs"].get(k) == v, (name, k)
        return d["worst_task_excess"]

    rerun = yi("rd_l0_rerun", {"rng_seed": None, "truncation": None, "solve_dtype": None})
    prod = [yi(f"rd_l0_rerun_s{k}", {"rng_seed": k, "truncation": None}) for k in (1, 2)]
    exact = [yi("rd_l0_f32exact", {"rng_seed": None, "truncation": "exact", "solve_dtype": None})] + \
            [yi(f"rd_l0_f32exact_s{k}", {"rng_seed": k, "truncation": "exact"}) for k in (1, 2)]
    f64 = yi("rd_l0_f64", {"rng_seed": None, "truncation": "exact", "solve_dtype": "float64"})
    body += [f"float32, production truncation & {pub['worst_task_excess']:.3f} / {rerun:.3f}"
             f" & {prod[0]:.3f} & {prod[1]:.3f} \\\\",
             f"float32, exact truncation & {exact[0]:.3f} & {exact[1]:.3f} & {exact[2]:.3f} \\\\",
             f"float64, exact truncation (registered) & {f64:.3f} & -- & -- \\\\",
             "\\bottomrule", "\\end{tabular}"]
    cap = ("Realisations of our encoder at $\\lambda = 0$; worst-task excess. Apart from the "
           "published cells and the registered float64 rebuild, these runs were added after "
           "the precision check and were not registered. \\textbf{(a)} The seed-1 shared cohort of each base under three "
           "random streams with the production truncation: the published cell and the same "
           "merge with the generator seeded by \\texttt{rng\\_seed} $1$ and $2$. "
           "\\emph{Independent}: the first independent cohort's float32 cell. \\emph{Gap}: the "
           "smallest shared draw minus the independent value. \\textbf{(b)} The cohort the "
           "rule flagged. The default stream with production truncation has two cells, the "
           "published one and a rerun on the same setup. The float64 merge with exact "
           "truncation is the registered rebuild and does not depend on the stream. The "
           f"independent cohort is at {indep[b]:.3f}.")
    return table("tab:appC-sweep", cap, body)


# ---------------------------------------------------------------- text

INTRO = (
    "This appendix gives the per-base and per-cohort values behind the campaign results "
    "that the main text summarises. Every result in it is written by a script directly "
    "from the result files of the campaign; no result was entered by hand. \\emph{Excess} is "
    "the worst-task NLL excess in nats (\\S\\ref{subsec:setup}), and \\emph{shared} and "
    "\\emph{independent} are the two arms. Where one cohort per arm was evaluated, these "
    "are the seed-1 shared cohort and the first independent cohort of the main grid. "
    "Tables~\\ref{tab:seeds}, \\ref{tab:regimes}, \\ref{tab:wudi} and \\ref{tab:exposure} "
    "are not repeated here.")

LEADS = {
    "C1": ("Table~\\ref{tab:appC-regimes} gives each training regime of \\S\\ref{subsec:tr} "
           "with the predictions of the scalar law tested in \\S\\ref{sec:theory}, the "
           "independent counterparts of two regimes, and the geometry of the two other routes "
           "of \\S\\ref{subsec:routes}."),
    "C2": ("Table~\\ref{tab:appC-hetero} gives the perturbations of separated cohorts "
           "summarised in \\S\\ref{subsec:against} and the partial-collapse cohorts of "
           "\\S\\ref{subsec:audit}, on every base."),
    "C3": ("Table~\\ref{tab:appC-lambda} gives, for each base and arm, the ridge that GCV and "
           "the L-curve choose and the excess it leaves, beside the minimal ridge and the best "
           "point of the sweep (\\S\\ref{subsec:lambda})."),
    "C4": ("Table~\\ref{tab:appC-vision} gives every merge of the vision experiment of "
           "\\S\\ref{subsec:beyond}, with its spread across the three cohorts of each arm and "
           "the median principal cosine of every cohort."),
    "C5": ("Table~\\ref{tab:appC-alltasks} gives the scores behind the registered contrasts on "
           "translation and IFEval (\\S\\ref{subsec:beyond}), the contrasts themselves, and the "
           "reference rows: the base model and each task's own adapter on all four benchmarks."),
    "C6": ("Table~\\ref{tab:appC-precision} gives the precision check of "
           "\\S\\ref{subsec:results} for every base, arm and solver. Table~\\ref{tab:appC-sweep} "
           "gives the float32 draws of the unregularised encoder on every base, and for the one "
           "cohort the rule flagged, the float64 rebuild and the draws with exact truncation."),
}


def main() -> int:
    parts = ["% Generated by paper/nc/make_app_campaign.py from results/phase3; do not edit by hand.",
             "% Rerun the script after any change to the result files.", "", INTRO, ""]
    for key, fn in [("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5), ("C6", None)]:
        parts += [LEADS[key], ""]
        if fn is None:
            parts += [c6(), "", c7(), ""]
        else:
            parts += [fn(), ""]
    OUT.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    for w in WARNINGS:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

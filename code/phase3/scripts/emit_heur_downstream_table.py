#!/usr/bin/env python3
"""Turn the heuristics-downstream summary into the LaTeX table, mechanically.

Forty rows transcribed by hand is forty chances to put a number in the wrong
place, and this is a table whose whole point is that a reader can check it.
So the paper's table is generated from the analyzer's own output file rather
than typed.

This script is deliberately separate from the analyzer and reads only
`heur_downstream_summary.json`. It applies no threshold and makes no decision:
every verdict in the output was computed by
`analyze_heuristics_downstream.py` under the rules registered at 9cc2394, and
this only formats them. Keeping the two apart means editing the presentation
can never move a rule.

Usage:
  python code/phase3/scripts/emit_heur_downstream_table.py > table.tex
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
SUMMARY = ROOT / "results/phase3/heur_downstream_summary.json"

PRETTY = {
    "task_arithmetic": "task arithmetic",
    "ties": "TIES",
    "dare": "DARE",
    "tvq_b2": "TVQ$_2$",
    "knots_ties": "KnOTS",
}
BASE = {
    "llama31_8b": "Llama-3.1-8B",
    "mistral_7b": "Mistral-7B",
    "qwen25_7b": "Qwen2.5-7B",
    "yi15_9b": "Yi-1.5-9B",
}


def main() -> int:
    if not SUMMARY.exists():
        print(f"no summary at {SUMMARY}; run the analyzer first",
              file=sys.stderr)
        return 2
    d = json.loads(SUMMARY.read_text())
    rows = d["rows"]

    n_eff = d["n_effects"]
    mde = d["mde"]

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\caption{The heuristics null, measured in downstream accuracy "
               "rather than in NLL. Each entry is $d = \\mathrm{acc}_{"
               "\\mathrm{indep}} - \\mathrm{acc}_{\\mathrm{shared}}$ for one "
               "method on one base; a cell counts as an effect only if $|d|$ "
               "exceeds both $0.05$ and $2\\,\\mathrm{SE}_{\\mathrm{binom}}$, "
               "both fixed in advance. Cells marked $\\ast$ clear both. "
               "\\textbf{This design has one cohort per arm}, so its gate is "
               "binomial and its minimum detectable effect is about "
               f"${mde['gsm8k']:.2f}$ on GSM8K and ${mde['humaneval']:.2f}$ on "
               "HumanEval: a null here means no effect detectable at that "
               "size, not no effect.}")
    out.append("\\label{tab:heur-downstream}")
    out.append("\\begin{tabular}{ll" + "cc" * 2 + "}")
    out.append("\\toprule")
    out.append(" & & \\multicolumn{2}{c}{GSM8K} & "
               "\\multicolumn{2}{c}{HumanEval} \\\\")
    out.append("base & method & shared & $d$ & shared & $d$ \\\\")
    out.append("\\midrule")

    by = {(r["base"], r["method"], r["bench"]): r for r in rows}
    bases = [b for b in BASE if any(r["base"] == b for r in rows)]
    for bi, base in enumerate(bases):
        if bi:
            out.append("\\addlinespace")
        for mi, meth in enumerate(PRETTY):
            g = by.get((base, meth, "gsm8k"))
            h = by.get((base, meth, "humaneval"))
            if not g or not h:
                continue
            label = BASE[base] if mi == 0 else ""
            def cell(r):
                star = "$^\\ast$" if r["effect"] else ""
                return f"{r['acc_shared']:.3f} & ${r['d']:+.3f}${star}"
            out.append(f"{label} & {PRETTY[meth]} & {cell(g)} & {cell(h)} \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")

    print("\n".join(out))
    print(f"\n% {n_eff}/40 cells clear both criteria; "
          f"P1 {'holds' if d['P1'] else 'fails'}, "
          f"P2 {'holds' if d['P2'] else 'fails'}, "
          f"P3 {'holds' if d['P3'] else 'fails'}; "
          f"registered branch {d['branch']}", file=sys.stderr)
    print(f"% action: {d['action']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Paper integration punch list (A4 output)
*Generated 2026-06-25 during A4 cross-read pass.*

For Phase 8 P1 (integrate sections into `paper/main.tex`). Each item is
a one-line fix at integration time, not a content fix.

## Section anchors to add in `main.tex`

The §6 section drafts (`6_2`...`6_7` + appendix) reference
section-level anchors that live in the parent file, not in the
fragments. When wiring fragments into `main.tex`, add these `\label{}`s
to the surrounding `\section{...}` heads:

| Anchor needed | Wraps fragment | Referenced from |
|---|---|---|
| `\label{sec:lower_bound}` | `lower_bound.tex` | 6_2 (×1), 6_3 (×1), 6_3b (×1), 6_5 (×2), 6_7 (×1), Td2 (×1) |
| `\label{sec:experiments}` | `experiments.tex` (or wrapping §6) | 6_7 (×1) |
| `\label{sec:discussion}` | `discussion.tex` | 6_4 (×1) |
| `\label{sec:future}` | last subsection of discussion | 6_4 (×1), 6_5 (×1) |
| `\label{sec:practical-recommendations}` | 6_7 wraps its own ✓ | n/a |

## Subsection anchors that need to be added in `experiments.tex` (or wherever)

These are referenced multiple times but never defined in the current set:

| Anchor needed | Concept | Referenced from |
|---|---|---|
| `\label{subsec:exp-multiseed}` | the §6.1 multi-seed T=4 matrix subsection | 6_2 (×1), 6_3 (×1), 6_4 (×1), 6_5 (×3), 6_6 (×0), 6_7 (×2) |
| `\label{subsec:exp-T-synthetic}` | synthetic T-scaling validation | 6_6 (×1) |
| `\label{subsec:rd-encoder-impl}` | rd-encoder algorithm subsection | 6_2 (×1) |

## Tables referenced but not yet existing

| Label | Reference site | Action |
|---|---|---|
| `tab:exp-lora-full` | 6_2:35 | Should be the §6.1 T=4 matrix table; add to experiments.tex |
| `tab:e5-arm2-floor-numbers` | 6_3b:91 | Sidebar references this; needs to be added to 6_3 main section, or change 6_3b ref to "Table~\ref{tab:e5-null}" which exists |

## References within §6 cluster (verified during A4)

✓ `subsec:exp-rd-encoder` (6_2), `subsec:exp-e5-null` (6_3), `subsec:floor-recipe` (6_3b),
  `subsec:exp-b2-mechanism` (6_4), `subsec:exp-downstream-gsm8k` (6_5),
  `subsec:exp-T-scaling` (6_6), `sec:practical-recommendations` (6_7),
  `app:sign-election-threshold` + `app:td2-setup` + `app:td2-comparison`
  (appendix Td2) — all resolve correctly.

✓ Tables in §6 cluster: `tab:e6-worst`, `tab:e6-slopes`, `tab:ties-probe`,
  `tab:e3-gsm8k`, `tab:b4-humaneval`, `tab:e5-null`, `tab:rd-ridge-sweep`,
  `tab:e7-prune-vs-tvq`, `tab:practical-tree` — all defined within the cluster.

## Notation consistency (verified during A4)

✓ $\Phi^\star$ family used consistently:
  - $\Phi^\star$ (worst-task floor, generic) — §3, §6.3, §6.3b
  - $\Phi^\star_{\mathrm{lower}}$ (specific floor lower bound) — §6.3b, §6.7
  - $\Phi^\star_{\mathrm{meas}}$ (measured worst-task excess) — §6.3b, §6.7
  - $\Phi_t^{\mathrm{TIES}}$, $\Phi_t^{\mathrm{TA}}$ — Td2 only, local
  
✓ $d_{\mathrm{eff}}$, $d_{\mathrm{eff}}^{\mathrm{hard}}$, $d_{\mathrm{eff}}^{\mathrm{soft}}$ —
  consistent across §3, §6.3, §6.3b, §6.7

✓ $B$ (per-coord radius), $B^2$ in lower bound — consistent

✓ $R$ (win-share range), $R^\star$ (threshold) — consistent across §6.6, Td2, §6.7

✓ $\rho$ (Spearman) — consistent across §6.5, §6.6, §6.7

✓ $T$ (task count) — consistent everywhere

## Content fixes made during A4

1. `paper/sections/6_2_e1_real_adapters_draft.tex` — `thm:achievability` →
   `thm:achv` (4 occurrences; correct label is `thm:achv` per
   `achievability.tex:60`).

2. `paper/sections/6_7_practical_recommendations_draft.tex` R3 paragraph
   updated to incorporate B4 HumanEval pass@1 evidence:
   - TVQ-$b{=}2$ noted as best on 3/4 bases on pass@1 (was
     incorrectly described as "meaningfully worse than rd-encoder ridge")
   - TA noted as worst on every base on pass@1 by factor 2-100x
   - Cross-metric backing for "TIES or TVQ-b=2, not TA" ordering

## Open items not within A4 scope

- `prop:quadratic-surrogate` (6_2:53) referenced but not yet
  defined anywhere. Either (a) add the proposition body to
  `achievability.tex` near `thm:achv`, or (b) drop the reference and
  reword the sentence. Action item for P1 integration.

- `figs/methods` and `figs/tvq` (figures) — referenced in
  `experiments.tex:247,257` but the actual figure assets are TODO.

- Cross-section abstract / intro polish (Block E P2) will be the
  natural moment to push the "7/8 cross-metric robust agreement"
  narrative as the §6.5 headline.

%%% END A4 punch list — fixes applied + integration to-do for P1 %%%

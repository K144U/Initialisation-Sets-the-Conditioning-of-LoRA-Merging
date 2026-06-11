# Decision log (per master_plan_iclr2027.md Part IX)

Dated record of gate outcomes, decision-rule branches, and cuts.

## 2026-06-10 — TMLR desk rejection
Desk-rejected by TMLR Editors-in-Chief (Kamath, Murray, Shah, Charlin),
no reviews. Verbatim grounds: "does not meet our editorial standards or
allow us to assess claims and evidence. In particular, the argumentation
and motivation for the work lack clarity. The authors are advised that
TMLR is not a suitable venue for this work."

Interpretation (2026-06-11): a triage/clarity failure, not a scientific
verdict — no reviewer assessed the theory or experiments. Likely causes:
formula-dense abstract (4 inline formulas, "Stiefel-random" unglossed),
and the floor-zero finding framed as the headline (reads as "the
interesting quantity vanishes on all real data"), plus IT-theory register
for a general-ML editorial board.

## 2026-06-11 — Recalibrated response
- master_plan_iclr2027.md adopted as the ICLR 2027 campaign, with
  reordering: the clarity rewrite (W items, 60-second cold read) is the
  IMMEDIATE action, not a parallel workstream; experiments proceed on
  their own merit, not as a response to the desk reject.
- E5 (floor-positive regime) doubles as the fix for the motivation
  critique: it makes the floor non-vacuous on real data.
- No resubmission anywhere within a month; target remains ICLR 2027 as
  originally planned (TMLR was an interim attempt).
- I0 note: the "all PBS jobs failed" premise predates the keeper +
  checkpoint + idempotent-resume patterns since proven on this cluster
  (LMCA 1,020-config grid; MOOLoRa pilot). Port those, not a rebuild.
- E4 launched 2026-06-11 (CPU, synthetic T sweep) — outcome schedules T1.

## 2026-06-11 — E4 outcome: decision rule FIRED -> T1 triggered
1000 trials/cell, T in {2,4,8,16} x r in {4,8}, d=256, seed 20260611
(results/e4_t_sweep/). Three findings:
1. Linear-T DECISIVELY FALSIFIED: median T16/T2 ratio growth = 1.77 vs
   the 8x that C = Tc^2/3 predicts.
2. Growth is cleanly LOGARITHMIC in T: ratio vs log2(T) fits with
   R^2 = 0.998-1.000 in all 10 (r,b) cells; ratios flat in b within each
   cell (constant independent of rate, as theory expects).
   r=4: 10.8 -> 15.0 -> 18.6 -> 22.4; r=8: 10.8 -> 13.4 -> 15.9 -> 18.2.
3. The log-T slope shrinks with r (~3.9/doubling at r=4 vs ~2.5 at r=8,
   ratio ~ sqrt(2)) -> consistent with a max-of-T concentration mechanism
   with deviation ~ sqrt(.../r), the exact route master_plan T1 sketches.
ACTION: T1 (theory week with Prof. Garg) is ON — target a
C = O(c^2 (1 + f(log T, r))) bound; Remark 5 to be rewritten around this
measurement either way. T=3 excluded from sweep (Hadamard-padding rate-
accounting artifact, documented in e4_t_sweep.py).

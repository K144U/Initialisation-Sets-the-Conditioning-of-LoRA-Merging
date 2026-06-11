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

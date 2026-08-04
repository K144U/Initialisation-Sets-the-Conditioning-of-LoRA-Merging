# Pre-registration: does DARE help when composed with TIES?

**Written 2026-08-04, BEFORE the `dare_ties` method exists in the codebase and
before any cell is run.** Nothing to peek at: the cells do not exist yet.

## Why this test

`notes/audit_dare_2026-08-04.md` established that DARE + task arithmetic cannot
beat task arithmetic in expectation. Rescaling makes DARE an unbiased estimator
of the merged delta, so Jensen bounds it from below by plain TA for a convex
loss, and the measured penalty follows the predicted `(1-density)/density`.

That argument has a stated limit, and this test is that limit. **It does not
apply to DARE composed with TIES.** TIES trims by magnitude and elects signs,
both biased operations, and the DARE mask changes *which* parameters survive the
trim. So `E[TIES(mask(delta))] != TIES(delta)` and nothing forbids an
improvement. DARE-TIES is also the composition the DARE paper leans on, so
testing only DARE + TA and concluding "DARE does not help" would be a real
overreach.

## What is at stake, fixed now

- If DARE **helps** TIES: our negative result is correctly scoped and must be
  written as "drop-and-rescale cannot help *composed with task arithmetic*". The
  paper must not say "DARE does not help". DARE works in its proper
  configuration and we say so.
- If DARE is **neutral or hurts**: the negative result generalises to both
  compositions. That is a stronger and more interesting claim, and note that
  **the unbiasedness mechanism does not explain it**, because it does not apply
  here. We would then have an unexplained empirical regularity. Report it as
  that. Do not invent a mechanism to cover it.

## Design

```
method     dare_ties = DARE drop-and-rescale per task, then TIES on the masked deltas
           (the official mask_merging wrapper with mask_apply_method="ties_merging")
bases      llama31_8b, mistral_7b, qwen25_7b, yi15_9b
cohorts    indep1, indep2, indep3   (the properly initialised regime, n=3)
densities  0.5, 0.2, 0.1            (DARE's keep fraction)
cells      4 x 3 x 3 = 36
metric     worst-task NLL excess, identical definition to every other cell
```

**TIES's own trim density is held at 0.2** inside `dare_ties`, matching the
standalone `ties` baseline exactly, so the only difference between the two arms
is the DARE mask. The reference arm is the **existing** `ties` cells on the same
4 bases and same 3 cohorts, already computed in `eval_a1_indep/`. They are not
re-run and their values are already known, which is fine: the new arm is what is
blind, and the comparison is between arms.

**Structural interaction, recorded before seeing anything, because it changes
how each density must be read.** DARE keeps a fraction `d`; TIES then keeps the
top 0.2 by magnitude.

- `d = 0.5`: both bind. TIES trims the 50 percent that DARE left.
- `d = 0.2`: they roughly coincide. TIES's trim keeps approximately the DARE
  survivors, since about 20 percent are nonzero and TIES wants the top 20
  percent. The rescale is a uniform scalar and cannot change which entries are
  kept or which signs are elected, only the merged magnitude.
- `d = 0.1`: **the TIES trim goes inert.** Only 10 percent of entries are
  nonzero, so keeping "the top 20 percent" keeps all of them. At this density
  `dare_ties` is DARE mask plus sign election plus disjoint merge, with no
  effective trim.

So the sweep also answers a second question for free: whether TIES's advantage
comes from the magnitude trim or from the sign election. If `d = 0.1` retains
most of TIES's advantage over TA, the sign election is doing the work.

## Threshold and gate

**0.005 nats**, the same value used in the W1s tie band, the A2 K1/K2 test, and
both A1 pre-registrations. Not tuned for this test.

**The same one-directional noise gate** as the A1 amendment: with `d_c` the
per-cohort difference and `SE = sd(d_1,d_2,d_3)/sqrt(3)`, a HELPS or HURTS call
is downgraded to NEUTRAL unless `|mean(d)| > 2 x SE`. It may only downgrade,
never promote. n = 3, so the sd carries 2 degrees of freedom and this is a
coarse screen, not an inference.

## Q1. PRIMARY, at density 0.2 only

Density 0.2 is the primary because it is the DARE density used everywhere else
in our matrix and it matches TIES's own trim density. **The other two densities
do not contribute to the primary verdict.** Naming one density in advance is
what stops this being a three-shot test reported as one.

Per base, `mean(d) = mean(ties) - mean(dare_ties)` over the 3 cohorts, positive
meaning DARE-TIES is better.

- **HELPS** if `mean(d) > 0.005` and the gate passes.
- **HURTS** if `mean(d) < -0.005` and the gate passes.
- **NEUTRAL** otherwise.

Verdict over the 4 bases:

- **DARE HELPS TIES** if helps on >= 2 and hurts on 0.
- **DARE HURTS TIES** if hurts on >= 2 and helps on 0.
- **DARE IS NEUTRAL** if neutral on >= 3.
- **MIXED** otherwise, and then neither direction is claimed.

## Q2. SECONDARY, the density sweep

Descriptive only. It cannot override Q1.

For each base, is the DARE-TIES penalty relative to TIES non-decreasing as
density falls across 0.5, 0.2, 0.1, the same monotone pattern DARE + TA showed
on 3 of 4 bases?

- **MONOTONE** on >= 3 of 4 bases: the mechanism behaves the same way in both
  compositions despite the unbiasedness argument not applying to this one. That
  is a fact worth reporting and explicitly not one we can currently explain.
- **NOT MONOTONE** otherwise. In particular, any base where DARE-TIES beats TIES
  at an interior density is evidence for DARE's stated mechanism, and must be
  reported as such even though Q1 is the primary.

## Binding constraints

1. No threshold in this document may be changed after the numbers are read.
2. The primary density is 0.2 and was named before the run. If a different
   density turns out to look better for DARE, that is a Q2 observation and is
   reported as secondary, not promoted to primary.
3. TIES's internal trim density stays 0.2 in both arms. No tuning of it in
   either direction after seeing results.
4. The `ties` reference cells are the existing `eval_a1_indep/` values. They are
   not re-run, re-seeded, or re-selected.
5. **Smoke test first.** This is a new merge path, so the project's smoke-first
   rule applies: a unit test asserting the composition is what it claims, plus a
   single real GPU cell inspected before the other 35 are dispatched.
6. All three densities get reported, including any that flatter DARE and any
   that do not.
7. If Q1 returns HELPS, the DARE audit note and the `decisions.md` entry from
   earlier today are both **amended in place** to scope the negative result to
   task arithmetic. That correction is not optional and is committed here.

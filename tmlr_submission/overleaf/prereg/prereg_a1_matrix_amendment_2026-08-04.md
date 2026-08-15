# Amendment 1 to the A1 merge matrix pre-registration

**Amends `prereg_a1_matrix_2026-08-03.md`. Written 2026-08-04, BEFORE any cell
of the `indep2` or `indep3` cohorts has been read.**

The parent document was written for a single cohort and says so in its own
"statistical ceiling" section: one seed per cell, no seed statistics possible,
every conclusion provisional pending replication. That replication has now
landed. 88 cells completed overnight (32 adapters, 56 matrix cells), all 56
parse with a finite `worst_task_excess`, balanced 28/28 across the two new
cohorts and 8 per method.

This amendment fixes how three cohorts are combined, before combining them.

## Blindness status, stated plainly

This document is **not fully blind**, and the difference matters:

- `indep1` **has been read.** Its full matrix and all three verdicts are in
  `notes/campaign_results_2026-08-03.md` and `decisions.md`.
- `indep2` and `indep3` **have not been read.** No value from either cohort has
  been opened, printed, or summarised. The only facts known about them are
  structural: file count, method balance, and that every value parses finite.

So the aggregation rule below is being chosen with one of three inputs visible.
The defence against that is not a claim of purity, it is the choice of rule: the
primary statistic is the obvious default (mean across cohorts, unchanged
threshold), not a rule selected from a menu. Where a genuine choice existed I
have taken the option that is harder on our own method, and said so at the point
of choice. Any reader may check that this file is committed before the
`indep2`/`indep3` numbers appear in any commit.

## Unit of analysis

The replicate is the **cohort**: `indep1`, `indep2`, `indep3`. They differ only
in the LoRA `A` initialisation draw (task seeds 101-104, 201-204, 301-304). The
data seed is pinned at `20260518` in all three, so the init draw is the only
variable. n = 3 per (base, method) cell.

This is the right replicate for the question. The gaps at issue arise from
subspace geometry, and geometry is set by the init draw, so varying the init is
varying exactly the thing that is suspected of driving the result.

## Threshold

**0.005 nats, unchanged.** Parent binding constraint 1 forbids changing it and
it is not changed here.

The parent's second threshold, the **0.013 nat Llama provisional band**, was
2 x a per-seed sd *borrowed from the shared-init cohort*, because no
independent-cohort sd existed. One now does. The borrowed constant is retired
and replaced by the directly measured cross-cohort spread, per base, as
specified below. This is a substitution of a measured quantity for a proxy of
the same quantity, not a relaxation: it can only be applied in the conservative
direction (see the gate).

## The noise gate

For each base, let `d_c` be the per-cohort difference (defined per question
below) and `SE = sd(d_1, d_2, d_3) / sqrt(3)`.

A **WINS** or **LOSES** call is downgraded to **TIES, inside noise** unless
`|mean(d)| > 2 x SE` as well as `> 0.005`.

The gate is **one-directional by construction**: it can only turn a
WIN or a LOSS into a TIE. It can never promote a TIE. With n = 3 the sd carries
2 degrees of freedom and is itself badly estimated, so this is a coarse screen
and not an inference. Used one-directionally it can only make our claims weaker,
which is the only safe way to use a statistic this thin.

---

## Q1'. Does rd-ridge's advantage survive, across three cohorts?

Per base:

1. For each of the 7 methods, take the mean of its 3 per-cohort values.
2. The **champion baseline** is the non-rd method with the lowest 3-cohort mean.
3. `mean(d) = mean(champion) - mean(rd_ridge)`, positive meaning rd-ridge is
   better. `d_c` for the gate uses that same champion in every cohort.
4. WINS if `mean(d) > 0.005`, LOSES if `< -0.005`, TIES otherwise. Then apply
   the noise gate.

The per-base verdict aggregation is **unchanged from the parent**: SURVIVES if
wins + ties >= 3 of 4 with at least 2 outright wins; WEAKENED if wins + ties
>= 3 of 4 with fewer than 2 outright wins; DOES NOT SURVIVE if losses >= 2 of 4.
Consequences for each are unchanged and carry over verbatim.

**Choice made against our own interest, recorded here:** the champion baseline
is selected on the 3-cohort mean, not per cohort. Selecting the best of five
baselines separately within each cohort would take a minimum over noise three
times and bias the comparison against rd-ridge. The mean-selected champion is
the less biased estimator. It is also the one more favourable to us, so the
per-cohort-champion variant is computed and reported alongside as a robustness
line, and if the two disagree the disagreement is reported, not resolved.

**Robustness line, pre-specified:** apply the parent's original single-cohort
rule independently to `indep1`, `indep2`, `indep3`, and report all three
verdicts. If the majority of the three disagrees with the primary verdict, the
result is reported as **UNSTABLE** and neither verdict is claimed in the paper.

## Q2'. Do rankings change between the shared-init and independent regimes?

The parent rule is unchanged: top-1 change and top-3 set change per base,
RANKINGS CHANGE MATERIALLY if top-1 differs on >= 2 of 4 bases OR the top-3 set
differs on >= 3 of 4. The independent side is now represented by the 3-cohort
mean ranking. Per-cohort rankings are reported alongside.

The design is **unbalanced and this is a limitation, not a finding**: three
cohorts on the independent side against a single `seed1` matrix on the shared
side. A ranking difference could reflect the extra averaging on one side alone.
Q4 below is what makes the comparison interpretable.

Rank correlation remains **forbidden**, for the reason the parent gives.

## Q3'. Is the salvage arc confounded by rank?

`mean(rd_rank16) - mean(rd_ridge)` per base, threshold 0.005 unchanged, noise
gate applied. Verdict rule unchanged: RANK IS IMMATERIAL if within threshold on
>= 3 of 4; RANK IS PART OF THE EFFECT if rank16 is worse by more than 0.005 on
>= 2 of 4.

`indep1` already returned RANK IS IMMATERIAL, which took audit finding A3 off
the fix list. If three cohorts reverse that, A3 goes back on the list and every
rd-ridge number in the paper must be reported at matched rank. Recorded now so
that reversal cannot be quietly absorbed.

## Q4. NEW: is ranking instability specific to the initialisation regime?

This is the control the parent could not run, and it gates how Q2' may be read.

For each base, compare top-1 and the top-3 set **among `indep1`, `indep2`,
`indep3`**, three cohorts that differ only in the init draw and are all in the
properly initialised regime. Count the bases where top-1 is **not unanimous**
across the three; call that k.

- **k >= 2 of 4: rankings are unstable within the regime.** Q2's cross-regime
  change is then **not attributable to initialisation**, and the claim that
  published merging benchmarks may be confounded by shared-init geometry is
  **withdrawn**. What the campaign found is that single-seed method rankings on
  this metric are noise, which is a real and publishable methodological point,
  but it is a different and smaller claim and must be written as that one.
- **k <= 1 of 4: rankings are stable within the regime.** Q2's cross-regime
  change then does bear on initialisation, and the parent's consequence text for
  RANKINGS CHANGE MATERIALLY applies unchanged, **including its precondition**:
  before any claim about the field, PEFT's default init and at least two
  published merging benchmarks must be checked for the shared-seed pattern. Our
  own repo showing the pattern is not evidence about anyone else's.

k is exhaustive over {0,1,2,3,4}, so this rule always returns.

---

## Binding constraints

Parent constraints 1 through 3 carry over unchanged. Constraint 4 ("one seed per
cell, every conclusion provisional pending replication") is what this amendment
discharges, and is replaced by:

4'. **n = 3 cohorts. This is replication of the initialisation draw only.** It
    is not replication of the eval data shuffle (pinned at `20260518` in all
    three), the base models, the tasks, or the hyperparameters, and it says
    nothing about whether `lambda*` transfers. Three is a small n and the sd
    behind every gate below carries 2 degrees of freedom.

5. **No threshold in this amendment may be changed after the numbers are read**,
   on the same terms as the parent.

6. **The noise gate may only downgrade.** If a future edit uses it to promote a
   TIE to a WIN, that is a violation of this document.

7. **All four questions get reported**, including Q4 when it withdraws a claim
   this project would rather keep. The parent says the credibility of the W1s,
   A2 and W3 verdicts rests on having reported them the same way. That is still
   the only reason any of this is worth anything.

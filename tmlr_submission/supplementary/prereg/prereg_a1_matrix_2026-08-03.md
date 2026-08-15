# Pre-registration: the A1 independent-init merge matrix

**Written 2026-08-03, BEFORE any cell in `results/phase3/eval_a1_indep/` has
been read.** The 28 cells completed at 14:28 [tz] and have been sitting unread.
The subspace geometry of this cohort has been measured
(`subspace_geometry_indep1.json`, reported in
`campaign_results_2026-08-03.md`), but no merge result from it has been opened.

Every other stage in this campaign had its decision rule fixed before its cells
ran. This one did not, because the matrix was scaffolded as a by-product of the
A1 geometry check. This document restores that discipline the only way still
available: by fixing the rules before looking. Anything decided after the
numbers are on screen is post hoc and must be labelled as such in the paper.

## What is in the matrix

28 cells, 4 bases x 7 methods, cohort `indep1`, **one seed per cell**:

```
bases    llama31_8b, mistral_7b, qwen25_7b, yi15_9b
methods  task_arithmetic, ties, dare, knots, tvq_b2, rd_ridge, rd_rank16
metric   worst-task NLL excess (nats), same definition as the published matrix
```

`rd_rank16` is the rank control for audit finding A3 (headline rd-ridge cells
used `realize="rank_deff"` = rank 64 while every baseline used rank 16).

## Statistical ceiling, acknowledged up front

**One seed per cell.** No seed statistics are possible from this matrix. The
per-seed standard deviations measured in W1s on the shared-init cohort are the
only calibration available:

```
llama31_8b  sd 0.0064      mistral_7b  sd 0.0029
qwen25_7b   sd 0.0005      yi15_9b     sd 0.0009
```

Llama is the noisy base and its sd exceeds several of the gaps this matrix will
be asked about. Any Llama difference below **0.013 nats** (2 sd) is provisional
regardless of what the rules below return, and must be reported as such. If the
matrix drives a paper claim, seed replication is required before submission.

## Threshold

**0.005 nats** for "a real difference", uniformly. This is the same value used
in the W1s tie band and the A2 K1/K2 test, chosen for consistency with rules
already executed in this campaign rather than tuned for this one.

---

## Q1. Does rd-ridge's advantage survive on a properly initialised cohort?

For each base, compare `rd_ridge` against the **best** of the five non-rd
baselines (task_arithmetic, ties, dare, knots, tvq_b2).

- **WINS** on a base if `rd_ridge` is lower by more than 0.005 nats.
- **TIES** if within 0.005 nats either way.
- **LOSES** if higher by more than 0.005 nats.

Verdict:
- **SURVIVES** if wins + ties >= 3 of 4, with at least 2 outright wins.
- **WEAKENED** if wins + ties >= 3 of 4 but fewer than 2 outright wins.
- **DOES NOT SURVIVE** if losses >= 2 of 4.

Consequences, fixed now:
- SURVIVES: the method result is real and general. The paper keeps rd-encoder
  ridge as a contribution and reports both cohorts.
- WEAKENED: the method is reported as competitive rather than winning, and the
  cohort dependence becomes part of the claim.
- DOES NOT SURVIVE: the advantage was an artifact of near-coincident subspaces.
  The method cannot headline. Say so plainly; do not retreat to "still wins on
  the degenerate cohort" as if that were the interesting case.

## Q2. Do method rankings change between cohorts?

Rank all 7 methods per base on the `indep1` cohort, and compare against the
matched **seed1** cells of the published shared-init matrix (same 7 methods,
same metric; the exact source directory is to be confirmed before extraction,
which does not require reading any value).

Two comparisons, both pre-specified:
- **top-1 change**: does the best method differ between cohorts?
- **top-3 set change**: does the set of the three best methods differ (as a set,
  ignoring order within it)?

Verdict:
- **RANKINGS CHANGE MATERIALLY** if top-1 differs on >= 2 of 4 bases, OR the
  top-3 set differs on >= 3 of 4 bases.
- **RANKINGS ARE STABLE** otherwise.

Consequences, fixed now:
- CHANGE: this is the strongest result available from the campaign. Merging
  method comparisons are confounded by an initialisation artifact. PEFT-style
  LoRA training seeds `A` from a global seed and starts `B` at zero, so `A`
  barely moves and every adapter in a cohort shares a subspace; if that is
  common practice, published merging evaluations may be ranking methods on
  cohorts whose geometry cannot exhibit the interference those methods target.
  This becomes the paper's lead, with the tuned-baseline result (W1) second.
  **Before it can be claimed, PEFT's default init and at least two published
  merging benchmarks must be checked for the shared-seed pattern.** Our own
  repo showing the pattern is not evidence about anyone else's.
- STABLE: the degeneracy is a defect in our evaluation only. Disclose it,
  report both cohorts, and do not build a claim on it.

**Do not compute a rank correlation for this.** With 7 methods,
`SD(rho) = 1/sqrt(6) = 0.41` under the null, which is the same n-too-small trap
that made the W5 downstream correlations unreadable. Count top-1 and top-3
changes instead.

## Q3. Is the salvage arc confounded by rank?

Compare `rd_ridge` (rank 64) against `rd_rank16` (rank 16) per base.

- **RANK IS IMMATERIAL** if `|rd_ridge - rd_rank16| <= 0.005` on >= 3 of 4.
- **RANK IS PART OF THE EFFECT** if `rd_rank16` is worse by more than 0.005 on
  >= 2 of 4.

Consequence if rank is part of the effect: audit finding A3 is upheld, the
published salvage arc is confounded, and every rd-ridge number in the paper must
be reported at matched rank against the baselines.

---

## Binding constraints

1. No threshold in this document may be changed after the numbers are read. If
   a rule turns out to be badly specified, record that as a limitation and
   report the result under the rule as written.
2. No re-tuning of `lambda` or `alpha` on this cohort after seeing these
   results. The values are inherited from the shared-init cohort, which is
   itself a limitation worth stating: `lambda*` was selected on a cohort whose
   geometry differs, so rd-ridge is arguably handicapped here. This is an
   argument for a follow-up sweep, not for adjusting anything now.
3. All three questions get reported, including the ones that come back
   unfavourable. The credibility of the W1s, A2 and W3 verdicts rests on having
   reported them the same way.
4. One seed per cell. Every conclusion is provisional pending replication.

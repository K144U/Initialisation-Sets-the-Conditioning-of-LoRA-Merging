# Pre-registration: the ridge sweep's shared arm at n = 3

Written 2026-08-15, **before the generator for these cells exists** and before
any cell is dispatched. The commit ordering is the evidence; if any result
commit of this test is an ancestor of this file's commit, the registration is
void and must be reported as void.

This completes R8 of the 2026-08-14 registration (`0c8b9e8`), which extended the
ridge sweep's *independent* arm to three cohorts and said, in its own words,
"The shared arm remains n = 1 (seed1). We state that, rather than implying the
gate is two-sided." Limitation 6 of the current draft says the same thing: the
gate exists, all four bases clear it by two to four orders of magnitude, and the
standard error behind it comes entirely from the independent side. A referee
reading the central claim as a shared-versus-independent contrast is being shown
three cohorts against one draw. This registration is what closes that, and it is
the first test in this project that can go badly for us by *replicating* rather
than by failing to.

## Conventions carried over unchanged

Nothing below is a fresh choice made with data in view.

- **Tie threshold** 0.005 nats. Fixed in the 2026-08-03 parent document
  (`0d9924b`) and unchanged in every registration since.
- **Noise gate** one-directional. A win or a loss is downgraded to a tie unless
  the mean difference clears the gate. The gate may only downgrade a verdict,
  never promote one.
- **Rank** pinned to `realize: rank_r` (rank 16) in every cell, so audit finding
  A3 cannot reappear.
- **Verdicts are reported in the branch the rule lands in**, including
  `PARTIAL` and `REFUTED`. We do not round to the nearer branch.

## Design

The E2 sweep (`eval_ridge_cond`, `fa5f3c3`) run on the two shared cohorts that
have never been swept:

- 4 bases x 7 lambdas x 2 cohorts (`seed2`, `seed3`) = **56 cells**.
- Lambdas: 0, 0.01, 0.03, 0.05, 0.13, 0.30, 1.00. The E2 grid, unchanged.
- `realize: rank_r`, `loader: plain`, `seed: 20260518`, `bits: 32`,
  `max_seq_length: 1024`. Verified equal, field by field, to both the existing
  `seed1` cells in `eval_ridge_cond` and the existing `indep2`/`indep3` cells in
  `eval_ridge_cohorts`, before this document was written. That check is the
  reason no cell here needs a matched re-run of anything else: the loader
  confound that corrupted an earlier Table 3 is excluded by construction rather
  than by belief.
- **No adapter training.** The `seed2` and `seed3` cohorts already exist, 16
  adapters each, all four bases and all four tasks. This test is 56 evaluation
  cells and zero training.
- The existing `seed1` cells are **reused and not re-run, re-seeded or
  re-selected**, on the same terms every previous registration applied to its
  reference arm.

## Blindness statement

Stated plainly, because this test is less blind than most in this project and
the difference matters.

**Not blind.** The `seed1` shared arm is published at n = 1 and we know what it
shows. All three independent cohorts are published. We know the direction the
paper currently claims, we know limitation 6 is the objection this is meant to
answer, and we would prefer seed2 and seed3 to look like seed1.

**Blind.** Every one of the 56 cells. No ridge sweep has been run on `seed2` or
`seed3` at any lambda, on any base. The `nll_tau` cache has no entry for either
cohort, which is independently checkable: `results/phase3/nll_tau_cache/`
contains `{base}__seed1.json` and no `seed2` or `seed3` file.

Because the prediction is a replication of a result we have already published,
the risk is real and one-directional: a confirmation changes nothing about what
the paper claims and only removes an objection, while a refutation withdraws the
operational half of the paper's central claim. That asymmetry is why the
consequence text below is fixed now.

## Statistics

Both arms are n = 3 after this test. Per base and per quantity Q:

- Shared arm: mean and sd over `seed1`, `seed2`, `seed3`; SE_s = sd_s / sqrt(3).
- Independent arm: mean and sd over `indep1`, `indep2`, `indep3`;
  SE_i = sd_i / sqrt(3).
- The difference between arms carries **SE = sqrt(SE_s^2 + SE_i^2)**.

This is the whole point of the exercise. R8's gate used SE_i alone because sd_s
did not exist. It exists after this test, and the gate is applied with both
terms from here on.

**The gate.** A directional call is downgraded to a tie unless
`|mean difference| > max(0.005, 2 x SE)` with SE as above. One-directional: it
may only downgrade. With n = 3 per arm each sd carries 2 degrees of freedom, so
this is a coarse screen and not an inference, and it is used only in the
direction that can weaken our own claims.

`lambda*` is selected **per cohort** as the arg-min over the grid, and the ridge
gain is averaged across the three cohorts within an arm. This is the rule R8
already applied to the independent arm and it is not changed here.

## Predictions, fixed now

Both mirror E2's originals in form. They are about the shared-versus-independent
contrast, now with both arms replicated.

- **P1 (ridge gain).** `G = L(lambda=0) - L(lambda*)` is larger on the shared
  arm than on the independent arm, on at least 3 of 4 bases, **clearing the
  two-sided gate**.
- **P2 (unregularised penalty).** The lambda = 0 excess is worse on the shared
  arm than on the independent arm by a factor of at least 2, on at least 3 of 4
  bases, **clearing the two-sided gate**.

## Decision rule

- **CONFIRMED** if P1 and P2 both hold on at least 3 of 4 bases.
- **REFUTED** if neither holds on at least 3 of 4 bases.
- **PARTIAL** otherwise.

## What we write in each branch, fixed now

This is the part that is worth nothing if written afterwards.

- **CONFIRMED.** Limitation 6 is **discharged and removed**, replaced by a
  sentence recording that the gate's provenance is now symmetric and the
  contrast is three cohorts against three. Section 7.2 and the abstract state
  the shared arm as n = 3. No number in the paper's headline changes, and we say
  that explicitly rather than presenting a replication as a new result.
- **PARTIAL.** The half that replicated is claimed and the half that did not is
  reported as having failed to replicate, in the abstract, Section 1, Section
  7.2 and Section 8. Limitation 6 is **narrowed, not removed**: it is restated as
  applying to the specific quantity that did not replicate.
- **REFUTED.** The shared-arm behaviour observed at n = 1 did not replicate. The
  operational half of the conditioning claim is withdrawn to "observed on a
  single shared cohort and not reproduced on two further draws" **everywhere**:
  abstract, Section 1, Section 6.4, Section 7.2 and Section 8. We do not retreat
  to "it still holds on seed1" as though that were the interesting case, and we
  do not demote the outcome to a limitation. The paper then reports that its own
  central contrast rests on a cohort that three draws do not support.

We record now, before seeing any cell, that REFUTED requires the largest rewrite
and is the outcome a reader most needs to know about.

## Reported regardless of outcome

1. **The shared arm's cross-cohort spread**, per base and per lambda, as a table.
   A reader should be able to see how variable the shared arm is without taking
   our word for it.
2. **Where `seed1` sits** relative to the three-cohort shared mean, in sd units,
   at lambda = 0 and at lambda*. If `|seed1 - mean| > 2 sd` on at least 2 of 4
   bases at either point, we state in the paper that the published
   single-cohort value was not representative of its own arm, in those words.
3. **The minimum detectable effect** under the two-sided gate, per base, defined
   as the smallest mean difference that would have cleared it.
4. All seven lambdas, including any that flatter the method and any that do not.

## Binding constraints

1. No threshold, gate or decision rule in this document may be changed after any
   number is read. If a rule turns out to be badly specified, that is recorded as
   a limitation and the result is reported under the rule as written.
2. No re-tuning of `lambda` on these cohorts. The grid is E2's and `lambda*` is
   the arg-min over it.
3. The `seed1` and `indep1`/`indep2`/`indep3` cells are the pre-existing ones.
   They are not re-run, re-seeded, or re-selected.
4. The generator asserts that the arms differ only in the cohort, and refuses to
   write cells otherwise, as in every generator since `2a49b3e`.
5. **Smoke test first.** One real cell is inspected before the remaining 55 are
   dispatched. The inspection checks configuration equality against the seed1
   arm and that the merge is not a no-op; it does **not** look at the direction
   of any effect. The smoke cell is one of the 56 verdict cells, so blindness for
   this test is 55 of 56, and the paper says so.
6. The analyzer is committed **before** the cells it reads have landed, as in the
   2026-08-03, 2026-08-07 and 2026-08-14 campaigns.
7. If this verdict contradicts a published verdict of ours, both are reported,
   with the superseded one named and dated, in the paper and not only in this
   repository.

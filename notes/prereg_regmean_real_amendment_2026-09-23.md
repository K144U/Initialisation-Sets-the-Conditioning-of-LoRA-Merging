# Amendment to prereg_regmean_real_2026-09-23: which Gram the adapter-only RegMean uses

Written 2026-09-23, after the parent registration and before any cell of
methods M2 or M3 has run (only M1 `regmean_act` cells, which do not use the
affected code, had started). It changes an implementation, not a threshold, a
prediction or a branch, and it adds cells; it makes none of our predictions
easier.

## What was found

`merging/regmean.py` builds each task's Gram from `A_t^T A_t` when the model
object provides a `get_layer` hook, and otherwise falls back silently to
`Delta_t^T Delta_t`. `merging/peft_model_view.py`, the object every
production eval cell passes in, never defined `get_layer` (git history has no
version of it that did). Therefore:

- every adapter-only RegMean cell the paper reports (Table 4, the lambda sweep
  on seed1/indep1, and the n = 3 shared-arm completion) ran with
  `sum_t Delta_t^T Delta_t`;
- the paper describes the method as using `sum_t A_t^T A_t`, and two analyses
  used that form: `scripts/merged_solution_norms.py` (Table 3, whose comment
  wrongly says PeftModelView provides the hook) and
  `scripts/regmean_published_default.py` (Section 5.5's alpha analysis).

Both Grams are built from the adapters' own factors and share the row space
range(A_t^T), so both are in the exposed class; but the paper attributed
numbers from one to the other, and that is an execution error to correct, not
a wording issue.

## What changes

1. `PeftModelView.get_layer` now exists and returns the adapter's factors, so
   adapter-only RegMean uses `A_t^T A_t` as described. This applies to every
   M2 and M3 cell of the parent registration (none had run) and to the
   public-cohort merges of `prereg_public_merge_2026-09-23.md` (none had run).
2. The parent's "reproducibility check against Table 4" for M3 is now a
   comparison between the two Grams and is reported as such.
3. **Added cells: Table 4 re-run with the described Gram.** Adapter-only
   RegMean at lambda in {1e-6, 0.01, 0.03, 0.05, 0.13, 0.30, 1.00} on seed1 and
   indep1, four bases: 56 NLL cells, configs identical to the published ones
   except for the output directory. They are judged by the rules the published
   table was registered under (`prereg_tmlr_2026-08-14.md` and its amendment):
   the ridge gain L(1e-6) - L(lambda*) larger on the shared arm on >= 3 of 4
   bases, and the minimally regularised excess worse there by >= 2x on >= 3 of
   4. Both Grams' results are reported side by side; the paper's Table 4 is
   replaced by the re-run, and the fallback numbers move to the audit-trail
   appendix with this explanation.

## Branches for the added cells

- Both published predictions hold with the described Gram: Table 4 is replaced
  and the claim stands unchanged.
- Either fails: the RegMean replication is stated for the Gram under which it
  holds, with both reported, and contribution 3 says which adapter-built Gram
  the solver result was shown for.

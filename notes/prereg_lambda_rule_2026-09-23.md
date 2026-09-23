# Pre-registration: a data-free ridge rule set by the audit (review item 9)

Written 2026-09-23, before the rule has been evaluated on any cohort.
Governs the `ridge_mode: kappa_target` option of `merging/regmean.py` and
`code/phase3/scripts/analyze_lambda_rule.py`, committed after this file.

## The rule

For adapter-only RegMean, per layer, with G = sum_t w_t A_t^T A_t and its
nonzero eigenvalues lambda_max >= ... >= lambda_min^+ (the range has
dimension at most Tr):

    lambda_rule = max(0, (lambda_max - k * lambda_min^+) / (k - 1)),   k = 100

which is the smallest isotropic ridge that brings the condition number of G
restricted to its range down to k. It needs only the adapter files, like the
audit. On a separated cohort (kappa ~ 1.5) it returns 0; on a collapsed one it
grows with the conditioning. **k = 100 is fixed now as a round number, not
fitted.** The paper's lambda* = 0.01 was selected on the original cohorts,
which is exactly why the rule is judged only on cohorts neither was chosen on.

## Where it is evaluated (all unseen by any lambda selection so far)

1. The four fixed-translation cohorts of Part T (`prereg_new_cohorts`).
2. The shared and independent cohorts of Part R at T in {2, 8} and
   r in {8, 64} (8 cohorts).
3. The public cohorts of `prereg_public_merge` (behaviour excess), added
   there as a fifth merge `rm_rule`.

Comparators on the same cohorts: lambda = 0.01 (the audit's fixed starting
point), RegMean's shipped alpha = 0.9, and lambda = 1e-6.

## Predictions

- **V1 (as good as the fixed start where the fixed start is good).** On the
  collapsed (shared) cohorts of 1 and 2, the rule's worst-task NLL excess is
  within 0.005 nats of, or below, lambda = 0.01's on >= 80% of them.
- **V2 (better than the shipped default on collapsed cohorts).** On the same
  cohorts, the rule's excess is below alpha = 0.9's by more than 0.005 on
  >= 60% of them.
- **V3 (harmless on separated cohorts).** On the independent cohorts of 1 and
  2, the rule's excess is within 0.005 of lambda = 1e-6's on >= 80% of them.
- **V4 (public cohorts).** On collapsed public cohorts, the rule's behaviour
  excess ratio to task arithmetic is <= 1.5 on >= 80%.

## Branches

1. **V1-V3 hold.** The paper's recommendation becomes the rule rather than
   "start at 0.01 and sweep": a one-line, data-free ridge computed from the
   same adapter files as the audit, shown on cohorts it was not tuned on.
2. **V1 fails.** The rule under-regularises or over-regularises relative to
   the fixed start; the paper keeps "start at 0.01 and sweep" and reports the
   rule as tested and not better.
3. **V3 fails.** The rule harms well-conditioned cohorts; it is not
   recommended.
4. **V2 fails.** Reported as measured; the comparison with the shipped default
   is left to the results of `prereg_regmean_real`.

No partial reads per group. k, the comparators and thresholds are fixed.

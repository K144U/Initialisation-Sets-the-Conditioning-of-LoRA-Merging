# Pre-registration: the heuristics null at three cohorts per arm, as an equivalence claim

Written 2026-09-23, before any cell below exists. Governs
`code/phase3/scripts/gen_heuristics_tost.py` and
`code/phase3/scripts/analyze_heuristics_tost.py`, committed after this file.

## Why

`prereg_heuristics_downstream` (91d4a4b) measured five heuristics in accuracy
with ONE cohort per arm and found no cell past a binomial gate of about 6
points (GSM8K) and 11 (HumanEval). Two limits follow and the paper states both:
"not detected" is weaker than "equivalent", and a sign lean of 26 to 12 towards
the shared arm (exact p = 0.034, not registered) was left dangling. This
registration adds the two further cohorts per arm that both need.

## Cells

Methods and configs exactly as in `eval_heur_downstream` (task arithmetic,
TIES, DARE, TVQ b=2, KnOTS as run there), same scorer, same templates, same
loader. New cohorts: seed2, seed3 (shared), indep2, indep3 (independent), on
all four bases, GSM8K (n = 500) and HumanEval (n = 164):
5 x 4 x 4 x 2 = **160 cells**. The existing seed1/indep1 cells of
`eval_heur_downstream` are reused as the third cohort of each arm; their
configs differ from the new ones only in the adapter directory, which the
generator checks.

## Statistics, fixed now

Per (method, base, benchmark): arm means over 3 cohorts,
d_b = mean_indep - mean_shared, SE_b = sqrt(sd_s^2/3 + sd_i^2/3).

**Equivalence (primary), per (method, benchmark), pooled over the four
bases:** dbar = mean_b d_b, SE = sqrt(sum_b SE_b^2) / 4. Two one-sided tests at
alpha = 0.05 with the normal approximation: equivalent at margin M iff
dbar - 1.645 SE > -M and dbar + 1.645 SE < M.

Margins: **M = 0.03 on GSM8K, M = 0.05 on HumanEval.** HumanEval has 164 items
and is half as precise; one margin for both would make HumanEval fail for
reasons of sample size alone. Also reported for every (method, benchmark),
as a descriptive number that governs nothing: the smallest M at which
equivalence holds, i.e. |dbar| + 1.645 SE.

**Per-base effect test (secondary), as in 91d4a4b:** effect iff
|d_b| > max(0.05, 2 SE_b).

**Sign test (now registered):** over the 40 (method, base, benchmark) values
of d_b at n = 3, exact two-sided binomial test on the non-zero signs.

## Predictions

- **E1.** Equivalent at M = 0.03 on GSM8K for >= 4 of 5 methods.
- **E2.** Equivalent at M = 0.05 on HumanEval for >= 4 of 5 methods.
- **E3.** No per-base effect cell on >= 38 of 40.
- **E4.** The sign test gives p >= 0.05, i.e. the n = 1 lean does not survive.

## Branches

1. **E1 and E2 hold.** The paper's null is restated as equivalence: "the five
   heuristics are equivalent across initialisation regimes within 3 accuracy
   points on GSM8K and 5 on HumanEval". The one-cohort table moves to the
   appendix.
2. **E1 or E2 fails for a method.** That method is named; the equivalence
   statement is made only for the methods that pass, and the smallest margin
   at which the failing one would pass is reported.
3. **E3 fails (3 or more effect cells).** The heuristics null is withdrawn
   in accuracy and the paper says the heuristics are affected in accuracy at
   n = 3; contribution 3's "invisible to heuristics" is removed.
4. **E4 fails.** The lean is reported as a registered finding in its
   measured direction, and the paper says which arm the heuristics favour.

## Binding constraints

No partial reads: the analyzer refuses to report until all 160 new cells exist.
No method, margin or test is changed after any cell is read.

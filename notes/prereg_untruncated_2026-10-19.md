# Pre-registration: is the heuristics' lean an artefact of rank truncation?

Committed 2026-10-19, before the code (`untruncated` option in the merge path,
`code/phase3/scripts/truncation_energy.py`, `gen_untruncated.py`,
`analyze_untruncated.py`).

## Why

Red-team item 6, professor A3. At three cohorts per arm the five heuristics
are equivalent across regimes but lean toward the shared arm (29 of 38 cells,
p = 0.0017; prereg_heuristics_tost branch 4), with no explanation. Our pipeline
truncates every merged update back to rank r = 16. On the independent arm a
merged update has rank up to Tr (more for the sparsifying methods) and loses
more to truncation: task arithmetic keeps 0.86 to 0.92 of the mean update's
norm there against 0.99 on the shared arm (Table "mechanism"). If that lost
energy is what costs the independent arm, merging without truncation should
remove most of the lean.

## Untruncated merge (fixed here)

The merged update is realised exactly as an adapter of rank equal to its
numerical rank (singular values above 1e-6 of the largest, capped at
min(d_out, d_in)), so nothing is discarded. Same methods and settings as the
TOST cells: task arithmetic, TIES (0.2), DARE (0.2, seed 20260518), TVQ (2
bits), repaired KnOTS (TIES inner).

## Parts and predictions

- **H1 (CPU, the premise).** For each method, base and cohort, the fraction of
  ||merged update||^2 discarded by rank-16 truncation, averaged over the 8
  sampled layers. Prediction: larger on the independent arm than the shared arm
  for every method on >= 3 of 4 bases.
- **H2 (accuracy, the test).** The three method x benchmark pairs with the
  largest registered lean: task arithmetic / HumanEval (d = -0.019), DARE /
  HumanEval (-0.016), TIES / GSM8K (-0.013). Untruncated, all four bases, three
  cohorts per arm: 72 cells. Prediction: the mean lean shrinks by at least half,
  d_untrunc >= d_trunc / 2 (d = indep - shared), on >= 2 of the 3 pairs.
- **H3 (NLL, descriptive).** All five methods untruncated at seed1 and indep1 on
  four bases (40 cells), reported beside the truncated values with no verdict.

## Branches, written now

1. **H1 and H2 hold.** The lean is a property of rank truncation in this
   pipeline; the paper says so where it reports the lean.
2. **H1 holds, H2 fails.** Truncation loses more on the independent arm but
   does not explain the lean; the paper says truncation was tested and
   excluded.
3. **H1 fails.** The premise is false; the lean stays unexplained.

## Binding constraints

Pairs, thresholds and the rank tolerance fixed here. No partial reads (72
accuracy cells for H2). Compute on STUDENT-ACCOUNT only.

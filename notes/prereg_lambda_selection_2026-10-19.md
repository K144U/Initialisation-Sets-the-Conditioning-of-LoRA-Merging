# Pre-registration: classical data-free ridge selection (GCV and L-curve)

Committed 2026-10-19, before the code. Implementation (`ridge_mode="gcv"` and
`"lcurve"` in `code/phase3/merging/regmean.py`), generator
`gen_lambda_selection.py` and analyzer `analyze_lambda_selection.py` follow.

## Why

Red-team item 2. "Add a ridge" is textbook and the paper cannot say how much:
the selected ridge does not track the conditioning, and our condition-number
rule failed (prereg_lambda_rule, branch 2). Numerical analysis has standard
parameter-choice rules that need no data. Adapter-only RegMean is a ridge
regression whose "samples" are the Tr columns of X = [A_1^T ... A_T^T] and whose
targets are Y = [D_1 A_1^T ... D_T A_T^T], both read off the adapter files, so
those rules apply as they stand. If one works, the paper gains a method: audit,
then set the ridge without data.

## Rules (per layer, fixed here)

Weights w_t = 1/T, X and Y scaled by sqrt(w_t) per task block, n = T r. With
the thin SVD X = U S V^T, for lambda on a log grid of 200 points from 1e-8 to
1e2: H = V diag(s^2/(s^2+lambda)) V^T, residual R = Y (I - H),
W_lambda = Y X^T (X X^T + lambda I)^{-1}.

- **GCV.** lambda_GCV = argmin n ||R||_F^2 / (n - tr H)^2.
- **L-curve.** lambda_L = the point of maximum curvature of
  (log ||R||_F, log ||W_lambda||_F) along the grid (Hansen's corner), with the
  curvature from second-order finite differences in log lambda.

Each layer gets its own lambda; the merge is adapter-only RegMean
(Gram sum_t A_t^T A_t) at that lambda, then the usual rank-r truncation. The
discrepancy principle is not registered: it needs a noise level, and these
targets are consistent (an exact interpolant exists whenever the subspaces are
independent), so there is none to supply.

## Cells

2 rules x 4 bases x 6 cohorts (seed1-3, indep1-3) = 48 NLL cells, with the
same loader and nll_tau as Table 4.

## Comparators (existing cells, not re-run)

- lambda = 0.01 and RegMean's shipped alpha = 0.9: three cohorts per arm
  (eval_regmean_real). Gate max(0.005, 2 SE of the paired by-cohort
  difference).
- lambda = 1e-6 and the oracle lambda* (best of the 7-point sweep): seed1 and
  indep1 only (eval_regmean_agram). Gate 0.005 nats.

## Predictions

- **L1 (repairs the collapse).** Shared arm: rule excess < lambda=1e-6 excess
  minus 0.005 on >= 3 of 4 bases (seed1).
- **L2 (no worse than the starting ridge).** Shared arm: rule excess <=
  lambda=0.01 excess + gate on >= 3 of 4 bases (n = 3).
- **L3 (near the oracle).** Shared arm: rule excess <= oracle excess +
  max(0.005, 0.1 x (L(1e-6) - L(lambda*))) on >= 3 of 4 bases (seed1).
- **L4 (no harm on separated cohorts).** Independent arm: rule excess <=
  lambda=1e-6 excess + 0.005 on >= 3 of 4 bases (indep1), and <= lambda=0.01
  excess + gate on >= 3 of 4 (n = 3).

Each rule is judged separately.

## Branches, written now

1. **A rule passes L1 to L4.** It becomes a method in the paper ("audit, then
   GCV/L-curve sets the ridge without data"), and stage 2 runs for that rule
   only: accuracy on all 6 cohorts x 4 bases x {GSM8K, HumanEval} (48 cells per
   rule), with **L5**: shared-arm accuracy >= shipped alpha = 0.9 + gate on >= 3
   of 4 bases on at least one benchmark (gate max(0.05, 2 SE)).
2. **L1 and L4 pass, L3 fails.** A safe data-free default that does not reach
   the oracle: reported as such, recommendation "start at the rule's lambda and
   sweep if data exist". No stage 2.
3. **L1 fails for both.** Classical data-free parameter choice does not set the
   ridge either. Reported, with the per-layer selected lambdas, as evidence that
   the right amount of regularisation is not readable from the adapters.

## Binding constraints

The grid, the curvature estimator and the decision thresholds are fixed here.
No partial reads (48 cells). Stage 2 runs only on branch 1. Compute on
STUDENT-ACCOUNT only.

# Solve precision check (professor B1): decision rule, written before computing

Committed 2026-10-24, before `code/phase3/scripts/check_solve_precision.py`
has been run on any adapter.

## Why

Every adapter-only RegMean and encoder merge solves in float32
(`merging/regmean.py`, `merging/rd_encoder.py`). At the minimal ridge the
system is badly conditioned: RegMean at lambda = 1e-6 inverts a 4096 x 4096
Gram whose eigenvalues off the A row span equal the ridge alone (condition
number ~1e6 and up), and the encoder at lambda = 0 keeps eigenvalues of the
(Tr x Tr) projector Gram down to 1e-6 of the largest, which is where float32
rounding sits. A reviewer can ask whether the collapse is rounding error.

## What is computed

For each base (Llama, Mistral, Qwen, Yi) and arm (seed1 shared, indep1
independent), on every LoRA module present in the adapter files (q, k, v, o
of every layer):

- **float32**: the production functions `merge_regmean` (ridge_lambda = 1e-6,
  weights 1/4 each, as the cells ran) and `merge_rd_encoder` (bits 32,
  ridge_lambda 0, realize rank_r), called unchanged on CPU with the stored
  float32 factors; the pre-truncation update W is captured by wrapping
  `svd_truncate_to_rank`, which the methods call on it.
- **float64**: the same formulas from the same stored factors in float64, with
  exact SVDs.

Reported per (base, arm, method): the per-module relative difference
||W32 - W64||_F / ||W64||_F before truncation and after exact rank-16
truncation of each; the amplification ||W|| / ||task arithmetic|| in both
precisions; and for the encoder, the kept dimension d_eff in both.

## Decision rule

- **Precision is not the cause** if, for every (base, arm, method), the median
  per-module relative difference after truncation is below 1e-2 and the
  shared/independent amplification ratio has the same order of magnitude in
  float64 as in float32 (ratio of the two ratios within [0.5, 2]). The paper
  then states the solve dtype and that float64 reproduces the merges.
- **Otherwise** the affected merges are rebuilt in float64 and re-evaluated in
  NLL on the cluster (same cells, same nll_tau), and the paper reports the
  float64 values beside the float32 ones, whatever they show.

Neither branch moves a threshold of any registered experiment.

# Pre-registration: RegMean with real activation Grams, and RegMean's shipped default end to end

Written 2026-09-23, before any cell below exists and before the generator and
the `regmean_act` merge have been run on any registered cohort. Governs
`code/phase3/scripts/gen_regmean_real.py` and
`code/phase3/scripts/analyze_regmean_real.py`, both committed after this file.

## Why

Two referee-level objections, both about §5.4 and §5.5 of the manuscript.

1. The "second solver" is RegMean in a **data-free adapter-only form** whose
   Gram is sum_t A_t^T A_t. Published RegMean (Jin et al., 2023) uses
   input-activation Grams G_t = X_t^T X_t. The collapse lives in the A factors,
   so the adapter-only Gram inherits it by construction; an activation Gram
   does not depend on the adapters' initialisation at all. A reviewer can fairly
   say our second solver was built to fail.
2. §5.5 says RegMean's shipped regulariser (alpha = 0.9 off-diagonal shrinkage)
   is about 20x too weak on a collapsed cohort, but that is a conditioning
   computation. No merge using it was ever evaluated.

**The mechanism makes a prediction about objection 1 that cuts against the
paper's current wording.** Real RegMean's normal equations are
sum_t Delta_t G~_t = Delta (sum_t G~_t): the adapters enter only on the right,
through Delta_t, and the operator being inverted is built from activations. So
the mechanism predicts real RegMean behaves like an averager on both arms. If
that holds, the claim narrows from "solve-based methods" to "solvers whose
normal equations are built from the adapter factors". We register the
narrowing now so it cannot be avoided later.

## Cells

Bases: llama31_8b, mistral_7b, qwen25_7b, yi15_9b.
Cohorts: shared {seed1, seed2, seed3}; independent {indep1, indep2, indep3}.
T = 4 (the translation adapter included, as everywhere in the paper).

Methods:

- **M1 `regmean_act`**: RegMean with activation Grams, published defaults:
  alpha = 0.9, no ridge. Grams from 256 training examples per task, forwarded
  through that task's own fine-tuned model (base + adapter t), all tokens,
  max length 1024, normalised by token count. q/k/v share one Gram (same input).
- **M2 `regmean` alpha = 0.9, ridge_lambda = 0**: the adapter-only form with
  RegMean's published regulariser and nothing else. This is "the shipped
  default" of §5.5.
- **M3 `regmean` ridge_lambda = 0.01, alpha = None**: the audit-recommended
  starting ridge of §4.4. Already measured on seed1 and indep1; re-run on all
  six cohorts so every comparison is within one batch. The seed1/indep1
  re-runs are also a reproducibility check against Table 4 (reported,
  governs nothing).

All merges SVD-truncated to rank 16 like every other method in the paper.
Configs inherit base_model, max_seq_length, seed, loader and task configs from
the existing `eval_regmean_cond` configs; only method, kwargs, cohort and output
path change. Every cell requests `min_free_gb: 30`.

- NLL: 3 methods x 4 bases x 6 cohorts = **72 cells**.
- Downstream: GSM8K exact match (n = 500, 256 new tokens) and HumanEval pass@1
  (n = 164, 512 new tokens), same scorers and configs as §5.9:
  3 methods x 4 bases x 6 cohorts x 2 = **144 cells**.

## Statistics

Per (method, base), arm means over 3 cohorts. For a shared-vs-independent
contrast, d = mean_shared - mean_indep and SE = sqrt(SE_s^2 + SE_i^2) with
SE = sd/sqrt(3) per arm. For a within-arm contrast between two methods, the
difference is paired by cohort and SE = sd(d_c)/sqrt(3).

Gates, one-directional (may only downgrade to "no effect"):
- NLL: |d| > max(0.005, 2 SE)  (the paper's tie threshold and gate).
- Accuracy: |d| > max(0.05, 2 SE).

## Predictions, fixed now

- **Q1 (real RegMean is immune, from the mechanism).** For M1 in NLL, no
  shared-vs-independent effect past the gate on >= 3 of 4 bases, AND
  mean_shared / mean_indep < 2 on all 4 bases.
- **Q1m (mechanism).** Median over layers of M1's pre-truncation
  amplification |Delta*| / |mean_t Delta_t| on the shared arm is <= 3 on all 4
  bases. (Data-free RegMean at minimal ridge reaches 35 to 51.)
- **Q2 (the shipped adapter-only default fails on a collapsed cohort).** For
  M2 in NLL, mean_shared >= 2 x mean_indep AND the difference clears the gate,
  on >= 3 of 4 bases.
- **Q3 (the recommended ridge repairs it, NLL).** On the shared arm,
  M2 - M3 > gate (paired by cohort) on >= 3 of 4 bases.
- **Q4 (the same in accuracy).** On the shared arm, acc(M3) - acc(M2) > gate
  on >= 3 of 4 bases, on at least one benchmark.
- **Q5 (real RegMean immune in accuracy too).** For M1, no shared-vs-independent
  effect past the accuracy gate on >= 3 of 4 bases, on both benchmarks.

## Branches, written now

1. **Q1, Q1m, Q5 hold; Q2, Q3 hold.** The solver claim is restated
   everywhere as: the collapse breaks solvers whose normal equations are built
   from the adapter factors (the paper's encoder, and RegMean in its data-free
   form, including at its shipped default); RegMean with activation Grams is
   immune, as the mechanism predicts. §5.4's title and contribution 3 change
   accordingly. §5.5 gains the end-to-end numbers. The abstract may not say
   "solve-based methods" without that qualifier.
2. **Q1 fails (real RegMean hurt by the collapse on >= 2 bases past the
   gate).** The mechanism's account of which methods are exposed is wrong at
   this step. The claim extends to activation-Gram RegMean, and §4 must say the
   mechanism under-predicted the exposure. Report both.
3. **Q2 fails.** The shipped default is adequate in NLL on collapsed cohorts;
   §5.5's "20 times too small" is kept only as a conditioning statement and
   the practical sentence built on it ("the default does not know which regime
   it is in, and the audit is what tells you") is removed from the intro and
   the abstract.
4. **Q4 fails.** The default-versus-ridge difference is reported in NLL only
   and the accuracy table is reported as a null at its gate. No sentence may
   describe the shipped default as scoring near zero unless the table shows it.

Q1m and the seed1/indep1 reproducibility check are reported whatever they
show and change no branch on their own.

## Binding constraints

- No partial reads: the analyzer refuses to report until all 72 NLL cells (for
  Q1-Q3) or all 144 downstream cells (for Q4, Q5) exist.
- The smoke test runs on `llama32_1b_smoke`, which is not in the registered
  set, and checks only that the code path completes and that the merged model
  generates. No registered cell is inspected before the full batch lands.
- Nothing here is re-selected on the results: alpha, the ridge, the number of
  Gram examples and the truncation are fixed above.

# Pre-registration: a published data-free method in the exposed class (WUDI-Merging)

Committed 2026-10-19, before any WUDI code exists in this repository. The
implementation (`code/phase3/merging/wudi.py`), generator
(`code/phase3/scripts/gen_wudi.py`) and analyzer
(`code/phase3/scripts/analyze_wudi.py`) are committed after this file.

## Why

Red-team item 1 and the professor's A4. Every published method we have run is
immune to the collapse; the two exposed merges are ours (the encoder and the
adapter-only RegMean variant). The exposure rule says a merge is exposed when it
solves against a matrix built from the adapters' own factors without data. We
searched for a published method of that kind and found one:

WUDI-Merging (Cheng et al., "Whoever Started the Interference Should End It:
Guiding Data-Free Model Merging via Task Vectors", arXiv 2503.08099; official
code github.com/nathanielyvo/WUDI-Merging, `vit/wudi_main.py`). Per linear layer
it minimises

    L(tau) = sum_t ||(tau - D_t) D_t^T||_F^2 / ||D_t||_F^2

where D_t is task t's update (out x in). The matrix is sum_t D_t^T D_t /
||D_t||^2, whose range for a LoRA update D_t = s B_t A_t is the row space of
A_t: exactly where the collapse lives. The paper gives the closed form (its
Eq. 20, with a ridge omega toward the per-task updates)

    tau = [sum_t w_t D_t (D_t^T D_t + omega I)] [sum_t w_t (D_t^T D_t + omega I)]^{-1},
    w_t = 1 / ||D_t||_F^2,

and says that choosing omega is hard, so as published it is solved by gradient
descent instead: Adam, lr 1e-5, weight decay 0, 300 iterations, initialised at
the SUM of the task updates, no rescaling coefficient. The paper also reports
LoRA results.

The mechanism makes a split prediction. The closed form inverts the collapsed
matrix and should be exposed. The published optimiser starts at the sum and
takes 300 small steps; directions with tiny curvature barely move, so early
stopping acts as an implicit ridge and the published version should not be.

## Design

- Methods, all per LoRA-targeted layer (q, k, v, o: every layer our adapters
  touch, which is what "every linear layer" means for a LoRA cohort), task
  updates D_t = scaling_t B_t A_t in float32, no task weights (as published):
  - `wudi_pub`: the official procedure exactly (Adam lr 1e-5, wd 0, 300
    iterations, init sum_t D_t, loss as above).
  - `wudi_cf`: the closed form Eq. 20 with omega = 1e-6, solved in float64.
    1e-6 mirrors the minimal-ridge reference of adapter-only RegMean.
  - `wudi_3k`: as `wudi_pub` with 3000 iterations (a dose arm, not a published
    setting).
  The merged update is then SVD-truncated to rank r = 16, as every merge in
  this pipeline is, so the comparison with Table 4 is like for like.
- Diagnostics recorded per merge, before truncation: amplification
  ||tau|| / ||mean_t D_t|| averaged over the 8 sampled layers of the drift
  analysis, and the tail fraction (share of ||tau||^2 in the bottom quartile of
  the eigen-directions of Hbar on its range, Hbar = mean row-space projector,
  the definition of Table "mechanism").
- Cohorts: llama31_8b, mistral_7b, qwen25_7b, yi15_9b x seed1 seed2 seed3
  (shared) | indep1 indep2 indep3 (independent).
- NLL: 3 methods x 4 bases x 6 cohorts = 72 cells, same loader, eval sets and
  nll_tau as the published RegMean cells.
- Accuracy: `wudi_pub` and `wudi_cf` x 4 bases x 6 cohorts x {GSM8K n=500,
  HumanEval n=164} = 96 cells, same harness and prompts as Table 5.

## Statistic and gate

Per base, d = mean_shared - mean_indep of worst-task NLL excess (3 cohorts each),
gate = max(0.005, 2 sqrt(SE_s^2 + SE_i^2)), as in every n = 3 test of this paper.
Accuracy: d_acc = mean_indep - mean_shared, gate = max(0.05, 2 SE), SE from the
three cohorts per arm (the regmean_real rule).

## Predictions

- **W1 (closed form exposed).** For `wudi_cf`, mean_shared >= 2 x mean_indep
  AND d > gate, on >= 3 of 4 bases.
- **W2 (published version protected by its optimiser).** For `wudi_pub`,
  d <= gate (no harmful effect past the gate) on >= 3 of 4 bases.
- **W3 (the mediator).** On the shared arm, tail fraction >= 0.5 for `wudi_cf`
  and <= 0.1 for `wudi_pub`, each on >= 3 of 4 bases.
- **W4 (dose).** The shared/indep ratio of `wudi_3k` lies strictly between
  those of `wudi_pub` and `wudi_cf` on >= 3 of 4 bases.
- **W5 (closed form in accuracy).** For `wudi_cf`, d_acc > gate on >= 3 of 4
  bases on at least one benchmark.
- **W6 (published version in accuracy).** For `wudi_pub`, d_acc <= gate on >= 3
  of 4 bases on both benchmarks.

## Branches, written now

1. **W1 and W2 hold.** The exposed class contains a published method's own
   closed form: WUDI as run is protected by early-stopped Adam, not by its
   design, and the protection is the implicit ridge the mechanism predicts.
   The paper's headline changes from "two merges of ours" to "the closed form
   of a published data-free method, and the two merges of ours; the published
   optimiser hides it".
2. **W2 fails (the published procedure is exposed).** The collapse breaks a
   published data-free method as its authors run it. This becomes the paper's
   lead result in the abstract and introduction.
3. **W1 fails.** The task-update Gram is not ill-conditioned enough to expose a
   solver at omega = 1e-6; the exposure rule is restricted to A-built Grams and
   the paper says WUDI was tested and is not exposed in either form.
4. **W3 fails while W1/W2 hold.** The outcome is right for a reason we did not
   measure; reported as a failed mechanism link.

W4 to W6 are reported whatever branch is reached.

## Binding constraints

- omega, the iteration counts, the learning rate and the initialisation are
  fixed here and not re-selected.
- The implementation is checked on a toy problem before any registered cell:
  Adam from the sum at a large iteration count must approach the omega -> 0
  closed form, and both must reduce the loss below its value at the sum.
- No partial reads: the analyzer refuses to report until all 72 NLL cells (for
  W1 to W4) or all 96 accuracy cells (for W5, W6) exist.
- Compute runs on STUDENT-ACCOUNT only; base models and adapters re-staged into
  that account on 2026-10-19 (the sanjay.g tree no longer holds them), with the
  restage verified against published cells before any registered cell runs.

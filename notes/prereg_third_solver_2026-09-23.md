# Pre-registration: which methods the collapse reaches, tested on three we did not build

Written 2026-09-23, before any cell below exists and before TSV-M or MaTS
has been run on any registered cohort. Governs
`code/phase3/scripts/gen_third_solver.py` and
`code/phase3/scripts/analyze_third_solver.py`, committed after this file.

## The prediction being tested

The mechanism (§4, and `prereg_drift_theory_2026-09-23.md`) says the collapse
matters to a merge exactly when the merge **inverts or whitens a matrix built
from the adapters' own factors**, because that is where near-collinear task
subspaces become small eigenvalues. It does not matter when the matrix comes
from data (activations, gradients) or is diagonal (a diagonal cannot see
overlap between subspaces). This sorts three published methods, none of
ours, before any of them is run:

| method | what is inverted or whitened | predicted |
|---|---|---|
| **TSV-M** (Gargiulo et al., CVPR 2025), official code | the stacked singular vectors of the task updates, whitened by SVD | **exposed** |
| **MaTS** (Tam et al., 2023), KFAC Fisher, CG, official defaults | activation and output-gradient Grams from data | **immune** |
| **Diagonal Fisher merging** (Matena & Raffel, 2022), data-free F = Delta^2 | a diagonal | **immune** |

Implementations: `merging/tsv_merge.py` (verbatim port of
`compute_and_sum_svd_mem_reduction`, sv_reduction 1/T, alpha 1.0),
`merging/mats_kfac.py` (100 CG iterations from the average, model_lambda 1.0,
256 examples per task, Gram length 512 tokens), `merging/fisher_avg.py`
(unchanged). All truncated to rank 16 like every method in the paper.

## Cells

Bases: the four. Cohorts: seed1-3 (shared), indep1-3 (independent). T = 4.

- NLL: 3 methods x 4 bases x 6 cohorts = **72 cells**, configs inherited from
  `eval_regmean_cond` (plain loader).
- Downstream, GSM8K (n = 500) and HumanEval (n = 164), configs inherited from
  `eval_heur_downstream`: TSV-M on all six cohorts (48 cells), MaTS on seed1
  and indep1 only (16 cells, plain loader because its Fisher needs a backward
  pass). **64 cells.**

## Statistics

As in `prereg_regmean_real_2026-09-23.md`: arm means over 3 cohorts,
d = mean_shared - mean_indep, SE = sqrt(SE_s^2 + SE_i^2), NLL gate
max(0.005, 2 SE), accuracy gate max(0.05, 2 SE). For MaTS downstream (one
cohort per arm) the binomial gate of `prereg_heuristics_downstream` applies.

## Predictions

- **S1 (TSV-M exposed).** Worst-task NLL excess worse on the shared arm, past
  the gate, on >= 3 of 4 bases.
- **S2 (MaTS immune).** No shared-vs-independent effect past the gate on
  >= 3 of 4 bases, and mean_shared / mean_indep < 2 on all 4.
- **S3 (diagonal Fisher immune).** Same criteria as S2.
- **S4 (TSV-M reaches accuracy).** Accuracy lower on the shared arm, past the
  accuracy gate, on >= 2 of 4 bases on at least one benchmark.

## Branches

1. **S1-S3 hold.** The paper states the exposure rule as tested on methods
   it did not build: merges that invert or whiten a matrix built from the
   adapters' factors are exposed (the paper's encoder, adapter-only RegMean,
   TSV-M); merges whose matrix comes from data or is diagonal are not
   (RegMean with activations, MaTS, diagonal Fisher). S4 decides whether the
   TSV-M row appears in the accuracy table as an effect or as a null.
2. **S1 fails.** Whitening is not exposed in practice. The rule narrows to
   methods that *solve* against an adapter-built matrix, and the paper says the
   whitening case was predicted and did not hold.
3. **S2 or S3 fails.** The mechanism under-predicts: a data-derived or
   diagonal merge is hurt by the collapse. That method is named and §4 must say
   which step of the argument failed.

## Binding constraints

No partial reads (72 NLL cells for S1-S3; 64 downstream cells for S4). The
smoke test runs on `llama32_1b_smoke`, outside the registered set, and checks
only that each method completes. No setting above is changed after a cell is
read.

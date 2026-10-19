# Pre-registration: accuracy on all four tasks, and reference rows

Committed 2026-10-19, before the new metrics in
`code/phase3/eval/downstream_metrics.py` (chrF, IFEval), the generator
`gen_all_tasks.py` and the analyzer `analyze_all_tasks.py`.

## Why

Red-team item 7, professor S2 and S3. Each cohort has four task adapters but
only math (GSM8K) and code (HumanEval) are scored in accuracy, and no table
shows the base model or the individual adapters on those benchmarks.

## Benchmarks (fixed here)

- **Translation**: WMT19 en->de validation (the translation task's eval split),
  first 500 pairs after a fixed shuffle (seed 0), greedy decoding, at most 256
  new tokens, corpus chrF (sacrebleu defaults).
- **Instruction following**: IFEval, a fixed random subset of 300 of its 541
  prompts (seed 0, chosen for compute), greedy, at most 512 new tokens,
  prompt-level strict accuracy with the official checker.

## Cells (per base; four bases)

Merges on GSM8K-style chat prompting as in Table 5. For the two new benchmarks:
base model (no adapter), the four seed1 adapters alone, task arithmetic and
TIES on seed1 and indep1, and the encoder at lambda = 0 and lambda* on seed1 and
indep1 (the merges of Table "downstream-cond"): 13 evaluations x 2 = 26.
Reference rows on GSM8K and HumanEval: base model and the four seed1 adapters,
5 x 2 = 10. 36 per base, 144 cells.

## Predictions

Gates: chrF max(2 points, 2 bootstrap SE); IFEval max(0.05, 2 binomial SE).

- **T1 (the solver collapse reaches all tasks).** On seed1, encoder lambda*
  beats lambda = 0 by more than the gate on >= 3 of 4 bases, on each new
  benchmark.
- **T2 (heuristics equivalent on all tasks).** |shared - indep| <= gate for
  task arithmetic and for TIES on >= 3 of 4 bases on each new benchmark (one
  cohort per arm, so this is a null at the gate's resolution, and is reported
  as such).
- Reference rows are descriptive: no verdict.

## Branches, written now

1. **T1 and T2 hold.** The paper's accuracy claims extend from math and code to
   all four tasks.
2. **T2 fails on a benchmark.** "Averaging is unaffected" is narrowed to the
   benchmarks where it holds, as the professor's S2 asks.
3. **T1 fails on a benchmark.** The collapse is reported as invisible on that
   task, with the adapter's own reference row beside it (the translation
   adapter is known to be weak).

## Binding constraints

Subsets, decoding and thresholds fixed here. No partial reads. Compute on
STUDENT-ACCOUNT only. This experiment runs last: it adds coverage, not a test of
the mechanism.

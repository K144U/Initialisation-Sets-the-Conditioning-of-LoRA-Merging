# Pre-registration: three new cohort families (translation fix, T and r sweep, PiSSA)

Written 2026-09-23, before any adapter below is trained. Governs
`code/phase3/scripts/gen_new_cohorts.py` (training configs and eval cells) and
`code/phase3/scripts/analyze_new_cohorts.py`, both committed after this file.
The first-order formulas referred to below are those of
`prereg_drift_theory_2026-09-23.md`; the audit is Algorithm 1 of the paper.

All training uses the paper's configuration (r = 16 unless stated, alpha = 2r,
q/k/v/o, 2 epochs, lr 5e-5, 7500 examples) and writes to the student-account
work tree; nothing in the original artifact tree is modified.

---------------------------------------------------------------------------
## Part T: the translation adapter, fixed (review item 4, scoped)

Retrain the translation adapter with `stream_shuffle_buffer: 50000` (the
buffer the paper's diagnosis used for its shuffled control) and nothing else
changed, on **Llama-3.1-8B and Mistral-7B** (the first two bases in the paper's
order), in **seed1** (shared; seeds 1/1) and **indep1** (independent; seeds
104/104, data 20260518), exactly the seeds those adapters had. 4 adapters. The
fixed cohort is the original three adapters plus the retrained one. Evaluation
task configs are unchanged (so held-out sets match every existing table).

Cells on the fixed cohorts (2 bases x 2 arms): the paper's encoder at
lambda = 0 and at the lambda* its committed sweep selected for that
base/cohort; adapter-only RegMean at lambda = 1e-6 and 0.01; the five
heuristics (as in `eval_heur_downstream`). NLL: 2 x 2 x 9 = 36 cells.
Downstream (GSM8K, HumanEval): encoder at lambda = 0 and lambda*, and the five
heuristics: 2 x 2 x 7 x 2 = 56 cells.

Predictions:
- **T1 (the fix works).** On all 4, the retrained adapter improves on the
  base model's translation NLL by >= 5% relative and is the best of the four
  adapters on the translation task.
- **T2 (geometry reproduces).** Median principal cosine > 0.9 on both shared
  fixed cohorts and < 0.1 on both independent ones.
- **T3 (solver consequence reproduces).** Adapter-only RegMean at 1e-6:
  shared / independent worst-task excess >= 2 on both bases; and the ridge gain
  L(1e-6) - L(0.01) larger on the shared arm on both.
- **T4 (heuristics null reproduces).** |shared - independent| < 0.02 nats in
  >= 8 of the 10 (method, base) NLL cells.
- **T5 (accuracy reproduces).** On the shared arm the encoder's lambda* beats
  lambda = 0 by more than max(0.05, 2 SE_binom) on both bases on at least one
  benchmark.

Branches: T1 fails -> report that the loader fix did not make the adapter
learn, and the scope caveat stays as written. T1 holds and T2-T5 hold -> the
paper reports the fixed-cohort replication as removing the "T = 4 nominal, 3
effective" qualification for these two bases. T1 holds and any of T2-T5 fails
-> the untrained member was carrying that part of the result; that claim is
narrowed to the cohorts where it holds and the paper says so.

---------------------------------------------------------------------------
## Part R: task count and rank (review item 10), Llama-3.1-8B only

Arms: shared (every adapter seed 1/1) and independent (distinct seeds, data
seed 20260518). Translation is the fixed one of Part T throughout.

- **T sweep at r = 16:** T = 2 {gsm8k, alpaca}; T = 4 {gsm8k, alpaca,
  magicoder, translation}; T = 8 = T = 4 plus {dolly (databricks-dolly-15k),
  xsum, codealpaca (CodeAlpaca-20k), translation_fr (wmt14, en->fr,
  shuffled stream)}. New adapters: 4 tasks x 2 arms = 8.
  Independent seeds for the new tasks: 105, 106, 107, 108.
- **r sweep at T = 4:** r in {8, 16, 64}; r = 16 is Part T's Llama cohort.
  New adapters: 4 tasks x {8, 64} x 2 arms = 16. Independent seeds 101-104
  as in indep1.

24 new adapters. Per configuration (T in {2,4,8} at r16; r in {8,64} at T4,
both arms): audit geometry, the exact kappa and amplification, the first-order
predictions (reference-free), and NLL for adapter-only RegMean at 1e-6 and 0.01
and task arithmetic: 4 new configs x 2 arms x 3 = 24 NLL cells (the T4/r16
cells come from Part T).

Predictions:
- **R1 (regime is set by initialisation at every T and r).** Median cosine
  > 0.9 on every shared configuration and < 0.3 on every independent one.
- **R2 (conditioning grows with T).** On the shared arm,
  kappa(T=8) > kappa(T=4) > kappa(T=2).
- **R3 (the first-order formula holds across T and r).** kappa_hat / kappa and
  alpha_hat / alpha in [0.75, 1.33] on every shared configuration.
- **R4 (the independent identity).** Independent-arm amplification within 5%
  of T at every T.
- **R5 (solver consequence across T and r).** Adapter-only RegMean at 1e-6:
  shared / independent >= 2 on >= 4 of the 5 configurations; task arithmetic:
  |shared - independent| < 0.02 on >= 4 of 5.

Branches: R1 fails at some (T, r) -> the regime claim is scoped to the
configurations where it holds. R2 fails -> reported as measured; no claim that
more tasks worsen conditioning. R3 fails at some configuration -> the
first-order proposition is stated with the configurations where it was
checked. R5 fails -> the solver consequence is scoped to T = 4, r = 16.

---------------------------------------------------------------------------
## Part P: PiSSA initialisation (review item 7)

PiSSA sets A_0 from the top-r singular vectors of the base weight, so it does
not depend on the seed. Train the four tasks of the paper on Llama-3.1-8B with
`init_lora_weights="pissa"` (PEFT) and the **independent** seeds 101-104.
4 adapters, plain PEFT (Unsloth has no PiSSA path).

- **P1.** Median principal cosine > 0.9: collapsed despite independent seeds.
- **P2.** The trained A factors' row spaces stay within principal cosine 0.9
  of the base weight's top-r right singular vectors (median over layers).

Branch: P1 fails -> PiSSA training moves A far enough to separate the tasks,
and the paper says PiSSA cohorts start collapsed but do not stay so.

Not registered, because it is a reading of source code rather than a
measurement: whether LLaMA-Factory, axolotl, TRL and Unsloth, at their
defaults, meet the Part A condition of `prereg_prevalence` (a fixed global seed
set before the adapter is created). That is reported as a documented finding
with source locations and pinned versions.

## Binding constraints

No partial reads per part. Seeds, tasks, buffer size and thresholds above are
not changed after any adapter is trained. If a training run fails and is
re-run, the re-run uses the identical config.

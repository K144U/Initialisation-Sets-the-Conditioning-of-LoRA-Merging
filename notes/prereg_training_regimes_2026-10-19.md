# Pre-registration: what sets the regime, and whether the theory transfers

Committed 2026-10-19, before the training-code changes (`seeds.init`,
`lora.init_scale`, `train.lr_A_mult`, `lora.init_from`, `lora.use_dora` in
`code/phase3/training/train_lora.py`), the generator `gen_training_regimes.py`,
the measurement script `measure_regimes.py` and the analyzer
`analyze_training_regimes.py`.

Common to every new run: base Llama-3.1-8B-Instruct, the four tasks (gsm8k,
alpaca, magicoder, flores), r = 16, lora_alpha = 32, dropout 0.05, targets
q/k/v/o, 2 epochs, lr 5e-5, batch 8 x 4, data-split seed 20260518 (the
independent cohorts' value), A_0 saved before the first step. "Measured" means
Algorithm 1's median principal cosine, kappa(Hbar) and the amplification on the
8 sampled layers, plus the tilt norm t of the drift theory (consensus W0).

## Part S. The 2x2 seed ablation (professor A2, red-team "nothing else differs")

Today the shared and independent cohorts differ in three things: the LoRA init
seed, the trainer seed (data order, dropout) and the data-split seed. Holding the
split fixed at 20260518, cross the first two:

| cohort | init seed | trainer seed |
|---|---|---|
| SA-ST | 1 for all tasks | 1 for all tasks |
| SA-IT | 1 for all tasks | 101-104 per task |
| IA-ST | 101-104 per task | 1 for all tasks |
| IA-IT | indep1 (existing: 101-104 / 101-104, split 20260518) | |

12 new adapters. Each cohort gets adapter-only RegMean at lambda = 1e-6 and task
arithmetic in NLL (8 cells, 2 of them existing).

- **S1.** Median cosine >= 0.9 for SA-ST and SA-IT; <= 0.5 for IA-ST.
- **S2.** kappa(Hbar) of each shared-init cohort >= 10 x that of each
  independent-init cohort.
- **S3.** RegMean(1e-6) excess of each shared-init cohort >= 2 x, and at least
  0.01 nats above, each independent-init cohort's.
- **S4.** Task arithmetic's excess varies by < 0.01 nats across the four.

Branch: if S1 or S3 fails for SA-IT or IA-ST, the regime is not set by the init
seed alone; the paper then names what does ("a shared training seed"), and the
mechanism text changes. If all hold, "nothing else differs" is replaced by this
ablation.

## Part K. Training regimes, and the theory out of sample (red-team item 3, S4)

Shared init (init = trainer seed = 1, split 20260518). Seven regimes beyond the
baseline SA-ST: lr x 1/3; lr x 3; 1 epoch; 4 epochs; A's learning rate x 4
(B at the base lr); LoRA+ at its recommended ratio 16 (B's lr 16 x A's, A at the
base lr); A_0 scaled by 0.25 after creation. Independent counterparts (init and
trainer seeds 101-104) for lr x 3 and A-lr x 4, the two regimes expected to move
A furthest. 36 new adapters; RegMean(1e-6) NLL for all 10 cohorts.

Calibration, fixed now and computed before any Part K adapter exists:
C_k = median of kappa t^2 and C_a = median of amp t over the existing Llama
shared cohorts (seed1-3, final) and the four drift checkpoints (seven points),
with t from the drift-theory code.

- **K1 (the regime persists).** Median cosine >= 0.9 on all 8 shared-init
  cohorts; <= 0.5 on both independent ones.
- **K2 (the scalar law transfers).** Predicted kappa = C_k / t^2 is within a
  factor 2 of the exact kappa on >= 6 of 8 shared cohorts, and predicted
  amp = C_a / t within a factor 1.25 on >= 7 of 8.
- **K3 (monotone).** Spearman rho(t, kappa) <= -0.8 across the 8 shared
  cohorts.
- **K4 (no escape by tuning).** The tilt needed to reach the independent arm's
  median kappa, t* = sqrt(C_k / kappa_indep), exceeds the largest observed t by
  >= 3 x.
- **K5 (the consequence persists).** RegMean(1e-6) excess of the lr x 3 and
  A-lr x 4 shared cohorts >= 2 x their independent counterparts'.

Branches: K2 fails -> the theory is descriptive at the cohort level and the
paper says "explains", never "predicts" (professor C2). K1 fails for a regime ->
that regime escapes the collapse and becomes a practical recommendation. K4
fails -> some tuning could plausibly escape; say which direction.

## Part C. Adapters continued from a shared adapter (red-team item 8)

Parent: the existing Llama dolly adapter `artifacts_new/lora/llama31_8b/dolly/indep1fx`.
Children: the four tasks, each initialised from the parent's A and B and trained
as usual with independent seeds 201-204. 4 adapters; RegMean(1e-6) and TA NLL.

- **C1.** Median cosine >= 0.9 although no seed is shared.
- **C2.** RegMean(1e-6) excess >= 2 x indep1's.

Branch: if C1 holds, the audit's prevalence reading covers derived adapters
(professor B3), and the paper names continued training as a second route into
the regime.

## Part D. DoRA (red-team item 8)

Shared (seed 1) and independent (101-104) DoRA cohorts through plain PEFT
(`use_dora=True`), 8 adapters, geometry only: merging DoRA needs magnitude
handling outside this pipeline.

- **D1.** Median cosine of the A row spaces >= 0.9 shared, <= 0.5 independent.

rsLoRA is not trained: it draws A from the same distribution and changes only
the scale alpha/sqrt(r), so its collapse at creation is identical by
construction; the paper states that rather than measuring it.

## Part L. Library defaults, measured (professor A1)

Unsloth (the version in our environment) and LLaMA-Factory (current pip
release, separate environment) at default settings, base Llama-3.2-1B-Instruct,
tasks gsm8k and alpaca, 50 steps. Median principal cosine between the two
tasks' A at creation and after 50 steps.

- **L1.** >= 0.99 at creation for both libraries.
- **L2.** >= 0.9 after 50 steps for both.

Branch: if LLaMA-Factory cannot be installed on the cluster, Part L reports
Unsloth measured and LLaMA-Factory from its source (the seed call before model
creation), and says which is which.

## Binding constraints

All thresholds and the calibration set are fixed here; C_k and C_a are computed
and committed before any Part K adapter is trained. Analyzer reports a part only
when all its cells exist. Compute on STUDENT-ACCOUNT only.

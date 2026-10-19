# Pre-registration: the collapse in the vision benchmark (CLIP ViT-B/32, eight tasks)

Committed 2026-10-19, before the vision code (`code/phase3/vision/`), generator
and analyzer.

## Why

Red-team item 4. Everything so far is language, four tasks, 7B to 9B models.
The merging community's standard benchmark is CLIP ViT-B/32 on eight image
classification tasks, and the vision cohorts released with KnOTS are separated
(median cosine 0.110 and 0.095), so the vision literature has never been in the
regime. Training LoRA cohorts there in both regimes tests whether the
mechanism is a property of LoRA or of language models, and places WUDI on the
benchmark its authors used.

## Design (fixed here)

- Model: `openai/clip-vit-base-patch32` (Hugging Face). Classification by a
  frozen zero-shot head per task built from the CLIP text encoder with the
  standard task-vector prompt templates (the copy in the WUDI repository's
  `vit/datasets/templates.py`).
- Tasks: SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD, from
  Hugging Face mirrors (FusionBench's `tanganke/*` where available). Test sets
  capped at 2000 examples per task by a fixed random subset (seed 0).
- LoRA: r = 16, alpha = 32, no dropout, on q, k, v and out_proj of every vision
  attention block. AdamW lr 2e-4, weight decay 0.1, batch 128, 1000 steps,
  cosine schedule with 100 warmup steps, CLIP preprocessing, no augmentation.
- Cohorts: three shared (cohort k: init and trainer seed k for all eight
  tasks) and three independent (init and trainer seed 100k + task index).
  48 adapters.
- Merges, applied as dense weight updates (no rank truncation, unlike the
  language pipeline): task arithmetic (lambda 0.3, the benchmark's usual
  value), TIES (density 0.2, lambda 1.0), TSV-M (official settings),
  adapter-only RegMean at lambda in {1e-6, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1} and
  at RegMean's alpha = 0.9, RegMean with activation Grams (alpha 0.9, 256
  training images per task), WUDI as published and its closed form at
  omega = 1e-6 (prereg_wudi). Reference rows: zero-shot and each individual
  adapter.
- Metric: mean top-1 accuracy over the eight tasks (worst task reported too).
  d = mean_indep - mean_shared over three cohorts per arm;
  gate = max(0.01, 2 sqrt(SE_s^2 + SE_i^2)).

## Predictions

- **V1.** Median principal cosine >= 0.9 for the shared cohorts, <= 0.5 for the
  independent ones.
- **V2 (solver exposed).** Adapter-only RegMean at lambda = 1e-6: d > gate.
- **V3 (the ridge repairs it).** The best lambda on the grid improves the shared
  arm by more than it improves the independent arm, by more than the gate.
- **V4 (heuristics equivalent).** Task arithmetic and TIES: the 90% interval of
  d lies inside +-0.02 (two accuracy points) for both.
- **V5 (data-built and whitening immune).** RegMean with activation Grams and
  TSV-M: d <= gate.
- **V6 (WUDI).** Closed form: d > gate; as published: d <= gate.

## Branches, written now

1. **V1 to V3 hold.** The mechanism is a property of LoRA, not of language
   models; the vision result goes into the main text beside the language one.
2. **V1 holds, V2 fails.** The collapse is there but this solver survives it in
   vision; the exposure claim is scoped to language models and the paper says
   so.
3. **V1 fails.** Shared init does not collapse ViT LoRA cohorts under this
   training; reported as a scope limit.

V4 to V6 are reported whatever branch is reached.

## Binding constraints

Hyperparameters, grids and thresholds fixed here; no re-tuning on test
accuracy. No partial reads: the analyzer needs all 48 adapters and every merge
on all six cohorts. Compute on STUDENT-ACCOUNT only.

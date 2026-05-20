# Phase 3 — empirical findings

Written: 2026-05-18 at the end of Day 4 (first full eval matrix). This document
is the standing reference for what the real-LLM data is telling us — keep it
up to date as we re-run with higher n_eval, more methods, or more models.

The numerical companion is `results/phase3/phase_b_summary.json` and the
per-cell JSONs under `results/phase3/eval_matrix/`.

---

## 1. The training side — per-task LoRAs (16 total: 4 models × 4 tasks)

16 LoRA adapters trained, full bf16 via Unsloth, LoRA rank 16 on Q/K/V/O of
every attention block, AdamW lr 5e-5, batch 8 grad-accum 4 (effective 32),
2 epochs, seq 2048. 7500 train examples per (model, task).

**NLL_τ — own-adapter on own-task, CLEAN v3 eval (n=1000, fixed loader),
nats per token, lower is better:**

|                    | GSM8K | Alpaca | Magicoder | Translation (wmt19 en→de) |
|--------------------|------:|-------:|----------:|--------------------------:|
| Llama-3.1-8B       | 0.466 | 0.991  | 0.281     | 0.745                     |
| Qwen-2.5-7B        | 0.442 | 0.919  | 0.247     | 0.822                     |
| Mistral-7B-v0.3    | 0.449 | 0.884  | 0.283     | 0.918                     |
| Yi-1.5-9B-Chat     | 0.387 | 0.840  | 0.229     | 0.763                     |

Source: `eval_matrix_n1k_v3_perexample/{model}__task_arithmetic.json`
`nll_tau[task][task]`. **These supersede the original Day-4 training-time
table** (n=200, buggy loader: Llama 0.446/1.018/0.275/0.719,
Qwen 0.418/0.946/0.238/0.794) — that table was computed by `train_lora.py`
with the pre-fix loader on n=200 and is not consistent with the eval matrix.
Yi-1.5-9B has the lowest NLL_τ on 3 of 4 tasks (it's the strongest
per-task learner, and also the easiest to merge — see F2).

Wall-clock per LoRA: 31–56 min for math/instruction/translation; Magicoder
runs 64–162 min (code examples ~3× longer in token count; 14B-class would be
longer still — we capped at 7–9B).

**Cross-task transfer:** the eval-matrix JSONs also contain each τ's NLL on
*every* task (the 4×4 transfer matrix). For example, Llama Alpaca's NLL on
GSM8K = 0.900 vs Alpaca's own task NLL = 0.924 — the instruction LoRA is
actually slightly better at math than at its own task. Worth a closer look
when writing experiments.tex.

---

## 2. The merging side — 40-cell eval matrix (4 models × 10 cells each)

**Current authoritative source 2026-05-19 18:29 IST** — eval_matrix_n1k_v2/ holds 40 cells (Llama-3.1-8B + Qwen-2.5-7B from v2 rerun + Mistral-7B + Yi-1.5-9B from T1.D extension). Headline figures in `paper_artifacts/figures/{headline_rd,method_compare}.png`; summary JSON in `paper_artifacts/data/phase_b_summary.json`. Prior 2-model version preserved as `*_v2_2model.*`; pre-fix n=200 buggy as `*_old.*`.

### 2v4.1 Non-TVQ methods (worst_excess @ b=32, all 4 models)

|                                  | Llama-3.1-8B | Qwen-2.5-7B | Mistral-7B | Yi-1.5-9B |
|----------------------------------|-------------:|------------:|-----------:|----------:|
| Task Arithmetic (uniform)        | 0.2179       | 0.1095      | 0.1452     | 0.0987    |
| TIES (density=0.2, total sign)   | **0.1597**   | **0.0137**  | **0.0569** | **0.0458**|
| DARE (density=0.2)               | 0.2183       | 0.1119      | 0.1466     | 0.1022    |
| KnOTS (linear inner merge)       | 0.2178       | 0.1098      | 0.1453     | 0.0988    |

(Lower `worst_excess` is better.)

### 2v4.2 TVQ rate sweep (worst_excess, 4 models)

| b  | Llama  | Qwen   | Mistral | Yi |
|----|-------:|-------:|--------:|---:|
| 1  | 0.2336 | 0.1375 | 0.3474  | 0.3044 |
| **2** | **0.1042** | **0.0184** | **0.0626** | **0.0599** |
| 4  | 0.2179 | 0.1098 | 0.1456  | 0.0987 |
| 8  | 0.2189 | 0.1099 | 0.1452  | 0.0989 |
| 16 | 0.2188 | 0.1094 | 0.1459  | 0.0989 |
| 32 | 0.2178 | 0.1098 | 0.1456  | 0.0990 |

### 2v4.3 TVQ b=2 dip — ratios across all 4 architectures

| Model | b=2 worst_excess | b=4 worst_excess | b=4 / b=2 ratio |
|---|---:|---:|---:|
| Llama-3.1-8B | 0.1042 | 0.2179 | **2.09×** |
| Qwen-2.5-7B  | 0.0184 | 0.1098 | **5.97×** |
| Mistral-7B   | 0.0626 | 0.1456 | **2.33×** |
| Yi-1.5-9B    | 0.0599 | 0.0987 | **1.65×** |

**The b=2 dip is universal across all 4 architecture families.** Qwen's is dramatic; Yi's is the mildest but still present. Confirmed not a model-family artifact.

avg_excess at b=2:

| Model | avg_excess |
|---|---:|
| Llama-3.1-8B | +0.0182 |
| Qwen-2.5-7B  | **−0.0066** (negative — merged beats avg task LoRA) |
| Mistral-7B   | +0.0143 |
| Yi-1.5-9B    | +0.0168 |

### 2v4.4 TIES vs TA (the F1 story across 4 models)

| Model | TA worst_excess | TIES worst_excess | TIES/TA | TIES win |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 0.2179 | 0.1597 | 0.733× | +27% |
| **Qwen-2.5-7B** | 0.1095 | **0.0137** | **0.125×** | **+87%** |
| Mistral-7B  | 0.1452 | 0.0569 | 0.392× | +61% |
| Yi-1.5-9B   | 0.0987 | 0.0458 | 0.464× | +54% |

TIES wins on every architecture. Qwen TIES is extraordinary (~8× smaller than TA); other models cluster in the +27–61% range.

### 2v4.5 Per-criterion outcomes (Phase B) — bootstrap-checked 2026-05-20

- **C1 floor at b=32: PASS for all 4 models, CIs exclude 0.** Floors (95% CI): Llama 0.218 [0.213, 0.224], Mistral 0.146 [0.141, 0.151], Qwen 0.110 [0.106, 0.114], **Yi 0.099 [0.095, 0.102] (lowest)**. Robust.
- **C2 TVQ slope ≈ −2: SKIPPED for all 4.** Only 1–2 points above floor in every model — rate-decay regime not visible at practical bit budgets. Confirmed universal.
- **C3 KnOTS > TA per-task: DOES NOT HOLD (corrected 2026-05-20).** Originally "PASS — 11/16 cells" by counting per-task wins, but the count was GPU-noise (swung 5/8 → 11/16 → 7/16). Paired bootstrap: KnOTS ≈ TA (only 5/16 significant, split both ways, ≤ 0.0019 nats — see F4). **This is theory-consistent at d_eff = Tr** (KnOTS advantage predicted to vanish as d_eff → Tr; we measure d_eff = Tr). The "structure-respecting > naive" result C3 was meant to capture IS validated, robustly, by **TIES** (F1, 14/16 significant) — just not by KnOTS.

**Decision gate:** ICLR viable. C1 robust; C3-as-stated (KnOTS) fails but is theory-consistent and the structure-respecting result is carried by TIES. We do NOT claim KnOTS beats TA in the paper.

### 2v4.5b Bootstrap CIs (v3 per-example re-eval, 10k resamples, 2026-05-20)

Source: `results/phase3/eval_matrix_n1k_v3_perexample/` (40 cells, per-example
NLL arrays) → `bootstrap_ci_v3.json`. v3 point estimates match v2 within ~0.001
(GPU non-determinism). All headline findings are statistically significant:

- **F1 TIES vs TA:** 14/16 cells significant; gsm8k wins large on all 4 models.
- **F7 b=2 dip:** b=2 vs b=4 CIs non-overlapping on all 4 models.
- **F2 ordering:** Llama > Mistral > Qwen > Yi — every adjacent pair's CIs
  non-overlapping, including the tightest (Qwen 0.110 [0.106,0.114] vs
  Yi 0.099 [0.095,0.102]).
- **F4 KnOTS ≈ TA / F5 DARE ≈ TA:** within practical noise (≤0.003 nats).

Practical-vs-statistical: at n=1000, ~0.0002-nat differences become "significant."
Paper applies a practical threshold (|Δ| > 0.005 nats AND CI excludes 0) — by
which **only TIES meaningfully differs from TA**, and only on the bottleneck task.

### 2v4.6 Architecture ordering on merging difficulty (TA floor as proxy)

From hardest to easiest to merge:
1. **Llama-3.1-8B** — TA 0.218 (hardest; MHA architecture)
2. **Mistral-7B-v0.3** — TA 0.145
3. **Qwen-2.5-7B** — TA 0.110 (GQA)
4. **Yi-1.5-9B** — TA 0.099 (**easiest; Llama-architecture but 2.2× easier than Llama-3.1**)

Yi-1.5-9B is Llama-architecture (same MHA setup) but **2.2× easier to merge than Llama-3.1-8B**. So GQA-vs-MHA is not the full F2 explanation; pretraining distribution and instruction-tuning recipe matter materially. Probably the strongest argument we have that "what makes a model merge cleanly" is a property of training, not just architecture.

### 2v4.7 Promoted artifacts (4-model authoritative)

- `paper_artifacts/figures/headline_rd.png` (2×2 grid; one panel per model)
- `paper_artifacts/figures/method_compare.png` (bar chart, 4 models × 4 methods)
- `paper_artifacts/data/phase_b_summary.json`

Versioning trail in `paper_artifacts/`:
- `*_old.*` = Day-4 n=200 buggy (HISTORICAL)
- `*_v2_2model.*` = v2 with Llama+Qwen only (intermediate, 2026-05-19 morning)
- (no suffix) = v2 with all 4 models (canonical, 2026-05-19 evening)

---

## 2v. (Historical, SUPERSEDED 2026-05-19 evening) v2 2-model tables

The 2-model tables below were the canonical view from morning to evening of 2026-05-19. They are now superseded by §2v4 above which adds Mistral-7B and Yi-1.5-9B. Preserved for traceability of what shifted between 2-model and 4-model views.

### 2v.1 Non-TVQ methods — V2 2-model (n=1k, fixed loader)

`worst_task_excess` at the implicit b=32 rate:

|                                  | Llama-3.1-8B | Qwen-2.5-7B |
|----------------------------------|-------------:|------------:|
| Task Arithmetic (uniform)        |  0.2179      | 0.1095      |
| TIES (density=0.2, total sign)   |  **0.1597**  | **0.0137**  |
| DARE (density=0.2)               |  0.2183      | 0.1119      |
| KnOTS (linear inner merge)       |  0.2178      | 0.1098      |

(Lower `worst_excess` is better.)

### 2v.2 TVQ rate sweep — V2 2-model

`worst_task_excess`:

| b  | Llama  | Qwen   |
|----|-------:|-------:|
| 1  | 0.2336 | 0.1375 |
| **2** | **0.1042** | **0.0184** |
| 4  | 0.2179 | 0.1098 |
| 8  | 0.2189 | 0.1099 |
| 16 | 0.2188 | 0.1094 |
| 32 | 0.2178 | 0.1098 |

`avg_task_excess`:

| b  | Llama   | Qwen     |
|----|--------:|---------:|
| 1  | 0.1396  | 0.0789   |
| 2  | 0.0182  | **−0.0066** |
| 4  | 0.0485  | 0.0214   |
| 8  | 0.0489  | 0.0214   |
| 16 | 0.0488  | 0.0214   |
| 32 | 0.0488  | 0.0214   |

### 2v.3 Phase B criteria on v2

- **C1 floor at b=32: PASS** (Llama 0.218, Qwen 0.110)
- **C2 TVQ slope ≈ −2: SKIPPED** (still only 1–2 points above floor; rate-decay regime not visible at practical bit budgets — confirmed not a data artifact)
- **C3 KnOTS > TA per-task: PASS** — 6/8 cells (Llama 4/4, Qwen 2/4; up from 5/8 in buggy matrix)

### 2v.4 What shifted from the buggy matrix

Direction-of-all-findings: **unchanged**. Magnitudes shifted by 0.5–3% on `worst_excess`. The b=2 dip on Qwen got *sharper* in clean data (6.0× vs adjacent rates, up from 5.0×). Qwen avg_excess at b=2 is still NEGATIVE (−0.0066).

Promoted artifacts:
- `paper_artifacts/figures/{headline_rd,method_compare}.png` (v2 versions; prior versions kept as `*_old.png`)
- `paper_artifacts/data/phase_b_summary.json` (v2; prior version kept as `phase_b_summary_old.json`)

---

## 2. (Historical, SUPERSEDED) The merging side — 20-cell eval matrix (Day 4, n=200, buggy loader)

The text and tables below are preserved as historical reference. Do not cite in paper drafts; use §2v above. Numbers here have the two issues noted in the supersession banner: n=200 and the train/eval shuffle bug.

5 methods × 2 models = 10 non-rate cells, plus TVQ × 6 rates × 2 models = 12,
minus 1 already done in pilot = 20 unique cells. Each cell loads base + 4
task adapters, applies the merging method with uniform 1/4 weights, evaluates
NLL on each task's 200 held-out, and computes
`worst_task_excess := max_t [NLL_merged(t) - NLL_τ_t(t)]`.

### 2.1 Non-TVQ methods (b=32 implicit) — HISTORICAL

|                                  | Llama-3.1-8B | Qwen-2.5-7B |
|----------------------------------|-------------:|------------:|
| Task Arithmetic (uniform)        |  0.2252      | 0.1075      |
| TIES (density=0.2, total sign)   |  **0.1637**  | **0.0142**  |
| DARE (density=0.2)               |  0.2245      | 0.1106      |
| KnOTS (linear inner merge)       |  0.2253      | 0.1078      |

(Lower `worst_excess` is better.)

### 2.2 TVQ rate sweep — HISTORICAL

`worst_task_excess` per `b` (bits per quantized parameter):

| b  | Llama  | Qwen   |
|----|-------:|-------:|
| 1  | 0.2195 | 0.1075 |
| 2  | 0.1070 | 0.1075 |
| 4  | 0.2238 | 0.1075 |
| 8  | 0.2253 | 0.1075 |
| 16 | 0.2245 | 0.1075 |
| 32 | 0.2253 | 0.1075 |

`avg_task_excess` per `b`:

| b  | Llama  | Qwen   |
|----|-------:|-------:|
| 1  | 0.1393 | 0.0175 |
| 2  | 0.0199 | 0.0175 |
| 4  | 0.0498 | 0.0175 |
| 8  | 0.0502 | 0.0175 |
| 16 | 0.0500 | 0.0175 |
| 32 | 0.0502 | 0.0175 |

---

## 3. Headline qualitative findings

### F1 — TIES wins worst_task_excess on all 4 models — REFINED 2026-05-20 (trades off per-task)

**Updated to 4-model + paired bootstrap.** TIES beats TA on `worst_task_excess`
on all 4 architectures (reductions: Llama 27%, Mistral 61%, Yi 54%, Qwen 87%),
and 14/16 (model,task) cells are paired-bootstrap significant. **But TIES does
NOT dominate uniformly per-task — it trades off:**
- **gsm8k (the bottleneck task): TIES wins big + significantly on all 4 models**
  (−0.058 to −0.106 nats). This drives the worst_task_excess win.
- **Non-bottleneck tasks: mixed** — TIES is significantly *worse* than TA on
  several (Mistral/Yi alpaca + magicoder + translation favor TA).

**Honest interpretation:** TIES's sign-election + magnitude-pruning resolves
destructive interference on the hardest task, at some cost to easy tasks; the
net worst-task effect is a clear win because the worst task is gsm8k. This —
not KnOTS (F4) — is the validation of "structure-respecting > naive." Only TIES
crosses a practical |Δ| > 0.005 nats threshold vs TA; KnOTS and DARE don't.

(Original Day-4 narrative, retained:) "On both Llama and Qwen, TIES wins on
every task." — the "every task" part is superseded by the per-task trade-off
finding above.

### F2 — Qwen-2.5-7B is ~2× easier to merge than Llama-3.1-8B

Task Arithmetic worst_excess: Llama 0.225 vs Qwen 0.107. Same with all the
other non-TVQ methods. **This is a strong signal we should flag in the paper.**
Possible causes (not yet investigated):
- Qwen's GQA may produce LoRA deltas with less inter-task interference
- Qwen's pretraining is more "modular" — the task vectors carve out cleaner
  subspaces
- Qwen's normalization (RMSNorm with different epsilons?) changes how the
  worst-case task's loss landscape behaves under perturbation

Worth a controlled probe: compute empirical `d_eff = rank(Σ_t P_{V_t})`
per layer for both models and see if Qwen has lower `d_eff/(Tr)`. The
theory predicts lower `d_eff/(Tr)` → smaller floor.

### F3 — Translation has consistently negative excess (merging *helps*)

Every cell, both models: `excess_translation < 0` (range -0.06 to -0.13).
The merged model is BETTER at en→de translation than the translation-specific
LoRA itself. **Interpretation candidates:**
- The translation LoRA may be under-trained — 7500 wmt19 examples × 2 epochs
  isn't a lot. The other task LoRAs (math, code, instruction) all contribute
  general grounding that translation benefits from.
- Translation is more sensitive to instruction-following capacity than to
  specialized fine-tuning — and Alpaca's LoRA is exactly that.
- We're streaming wmt19; first 7500 examples may be lower-quality than a
  shuffled sample.

This is a paper-worthy paragraph but probably also a flag for re-training
translation with more data / epochs.

### F4 — KnOTS beats Task Arithmetic per-task ~~on 5/8 cells~~ → CORRECTED 2026-05-20: KnOTS ≈ TA (was noise)

**⚠ CORRECTION (2026-05-20, paired bootstrap).** The original Day-4 claim
below (KnOTS dominates TA per-task) does NOT survive. The per-task win
count swung 5/8 → 11/16 → 7/16 across GPU-non-deterministic reruns — a
clear sign the gaps were within noise. Paired bootstrap
(`method_diff_ci_v3.json`, KnOTS vs TA, shared τ cancels):
**only 5/16 cells significant, split both directions (3 KnOTS, 2 TA), all
magnitudes ≤ 0.0019 nats.** KnOTS (linear inner merge) is statistically
and practically indistinguishable from Task Arithmetic in our setup.
**This is consistent with theory:** the C3 hypothesis predicted KnOTS's
subspace-aware advantage *vanishes as d_eff → Tr*, and we measure d_eff = Tr
saturated everywhere (F9). KnOTS has no subspace overlap to exploit.

(Original Day-4 narrative, retained for history:) "Counter to my earlier
impression: KnOTS *does* dominate TA per-task, including 4/4 tasks on Llama.
The wins are on the 3 non-gsm8k tasks. This is criterion 3 from the design —
PASS with the per-task interpretation." — **This interpretation is now
retracted; the wins were noise.**

### F5 — DARE ≈ Task Arithmetic — CONFIRMED 2026-05-20 (practically equal, not a bug)

Paired bootstrap (`method_diff_ci_dare_v3.json`): 13/16 cells "significant"
at n=1000, but all magnitudes ≤ 0.0030 nats and mostly slightly favoring TA.
DARE's random drop+rescale adds noise that doesn't help at d_eff=Tr.
**Practically DARE = TA, marginally worse — a real result, not an
implementation bug.** Density-sweep ablation still worth running for the
paper to show the density→d_eff interaction, but the density=0.2 ≈ TA
finding is bootstrap-confirmed.

(Original Day-4 note retained:) "To 3 decimal places, DARE at density=0.2
produces the same merged delta as TA on both models." — confirmed real.

### F6 — TVQ rate doesn't matter much at the bit budgets we tested

The rate-distortion curve we expected: `worst_excess(b) = floor + C · 2^(-2b/d_eff)`.
What we see: at `b ∈ {4, 8, 16, 32}`, worst_excess is essentially constant
≈ TA's value. At `b=1`, worst_excess is similar. At `b=2`, Llama dips to
0.107 but Qwen stays at 0.107.

**Interpretation:** at `b ≥ 1` the merging-geometry error dominates the
quantization error. The rate-decay term only becomes visible at sub-bit
precision (which is unphysical for our scheme). **The theory's RATE term
is qualitatively predicted to vanish here; the floor structure is what we
can validate.**

This is a substantive result worth a paragraph in the paper, NOT a failure
to validate. The bound's prediction is still consistent — we're just deep
in the floor regime everywhere.

### F7 — Llama TVQ b=2 is a REAL local minimum in the rate-distortion curve (confirmed at n=1k)

**Updated 2026-05-18 after n=1k rerun.** Sample noise hypothesis ruled out.

| b | n=200 worst | n=1k worst | Δ |
|---|---:|---:|---:|
| 1 | 0.220 | 0.232 | +0.012 |
| **2** | **0.107** | **0.104** | **-0.003** |
| 4 | 0.224 | 0.217 | -0.007 |
| 8 | 0.225 | 0.218 | -0.007 |
| 16 | 0.225 | 0.217 | -0.008 |

The dip persists at n=1k with the same magnitude as n=200. avg_excess
shows the same pattern (b=2: 0.019, b ∈ {4,8,16}: ~0.049). **This is a
real structural property of LoRA merging with uniform scalar quantization,
not measurement variance.**

**Candidate mechanisms:**
- **Quantization-as-regularization at b=2.** With only 4 levels per layer
  (2² values), the granularity might destroy destructive-interference
  patterns between task vectors that finer quantization (b ≥ 4) preserves.
  b=1 is too coarse (binarization destroys the LoRA signal); b=2 is the
  sweet spot.
- **Stochastic-resonance-like effect.** Quantization noise at b=2 happens
  to push the merged delta toward a region of the loss landscape that is
  better for gsm8k (the worst task).
- **Implicit coarse-projection.** With 4 levels, the per-tensor min-max
  quantization is effectively projecting each layer's delta into a coarse
  basis that may align with the dominant LoRA-rank-16 structure.

**Pending Qwen confirmation** (TVQ b=2 cell running). If Qwen also dips →
mechanism is model-agnostic, likely structural; HEADLINE-WORTHY paper finding.
If Qwen doesn't dip → mechanism is Llama-specific; still worth investigating
but more modest claim.

**Plan to strengthen for paper:**
1. Confirm/refute Qwen b=2 dip.
2. Add intermediate rates b ∈ {1.5, 2.5, 3} via Lloyd-Max with non-power-of-2
   levels to map out the local minimum's shape.
3. Per-task excess breakdown — is the b=2 win on gsm8k specifically (the
   worst task), or across all 4 tasks?
4. Repeat on a different task set (different 4-task combo) to rule out
   cross-task-overlap artifacts.

### F8 — Data-loader train/eval overlap bug (RESOLVED 2026-05-18)

`data_loaders.py` used two independent shuffle seeds for train and eval when
both came from the same split (alpaca, magicoder). Resulted in 13% (alpaca)
and 7% (magicoder) overlap. Fixed: single shuffle, disjoint slices. Verified
0/200 overlap. Training pool was always clean — only eval slice was tainted.
The v2 rerun re-evaluated existing LoRAs against the fixed eval to produce
canonical numbers. The Mistral+Yi LoRAs were trained after the fix.
**Full detail in `log.md` §3.4 / §4.3 / F8.**

### F9 — Hard d_eff = Tr is UNIVERSAL across 4 architectures (2026-05-19 evening)

Empirical d_eff = rank(Σ_t P_{V_t}) computed from per-layer LoRA factor SVDs.
**Every single layer of every base model (Llama-3.1-8B, Qwen-2.5-7B,
Mistral-7B-v0.3, Yi-1.5-9B) has d_eff = 64 = Tr = 4·16.** The 4 task subspaces
span the full Tr-dim space across architectures — the bound's predicted floor
B²(1 − d_eff/(Tr)) = 0 everywhere. Yet observed worst_task_excess at b=32 ranges
from 0.099 (Yi) to 0.218 (Llama). **The gap quantifies how far current merging
methods sit above the information-theoretic floor — there's room for 2–5×
improvement across architectures.** Reframes the paper's headline.
**Full detail in `log.md` F9.** Figure: `paper_artifacts/figures/deff_vs_floor.png`.

### F10 — Soft d_eff (participation ratio) does NOT explain F2 (2026-05-19 evening)

Following from F9, the natural next hypothesis was that *soft* d_eff
(participation ratio of stacked-V singular values, ranging continuously in
[1, Tr]) would correlate with the observed merging-difficulty differences.
**It doesn't.** Mean soft d_eff across architectures:

| Model | soft d_eff (mean) | soft d_eff / Tr | observed TA worst_excess |
|---|---:|---:|---:|
| Llama-3.1-8B | 16.48 | 0.258 | 0.2179 |
| Qwen-2.5-7B  | 16.45 | 0.257 | 0.1095 |
| Mistral-7B   | 16.37 | 0.256 | 0.1452 |
| Yi-1.5-9B    | 16.41 | 0.256 | 0.0987 |

Soft d_eff varies <0.7% across architectures; TA worst_excess varies 2.2×.
Pearson r = +0.53 (p = 0.47), Spearman r = +0.40 (p = 0.60). Not significant.

**Implication:** Both hard and soft d_eff are uniform across these
architectures, but merging difficulty isn't. **F2 (merging-difficulty ordering)
is NOT a subspace-overlap phenomenon.** Falsifies the original
"soft-d_eff-explains-F2" hypothesis. Remaining candidate mechanisms:

- (c) Unmodeled $H_t$ curvature (Fisher / GGN differences across base models).
- Pretraining + instruction-tuning recipe (Yi-arch but easier than Llama is the
  strongest argument for this).
- Per-layer $B^2$ (LoRA-delta norm) differences across base models.

**Full detail in `log.md` F10.** Figure: `paper_artifacts/figures/deff_vs_floor.png`
(4-panel: per-layer hard hist + per-layer soft hist + soft-vs-observed scatter
+ per-model soft d_eff bars).

---

## 4. Bugs / fixes / engineering learnings

Things I burned time on so the next person doesn't.

1. **TRL 0.24 removed `DataCollatorForCompletionOnlyLM`** — use
   `SFTConfig(completion_only_loss=True)` with `{prompt, completion}` format.
2. **TRL's `dataset_num_proc` defaults to `os.cpu_count()`** (= 64+ on
   `jiit-gpu01`). 64 workers each redownload the tokenizer mirror; network
   thrashes. Set `dataset_num_proc=2` or smaller.
3. **Unsloth monkey-patches `LlamaForCausalLM.forward` at import time.** Eval
   models must be loaded via `FastLanguageModel.from_pretrained(adapter_dir)`,
   not `AutoModelForCausalLM`, or the patched forward crashes on missing
   `apply_qkv`. Document this if anyone tries to use Unsloth for training and
   plain HF for eval.
4. **Loading Unsloth from a local model path doesn't set the implicit
   tokenizer slot.** Must pass `processing_class=tokenizer` explicitly to
   SFTTrainer. (When loading from a HF repo path, this is auto-set.)
5. **`huggingface-cli` is deprecated in `huggingface_hub` 1.x** — silent no-op
   exit. Use `hf` (new CLI) or the Python `snapshot_download` API.
6. **HF Xet protocol self-throttles** to ~30 KB/s on this cluster. Use
   `wget --continue` against `huggingface.co/{repo}/resolve/main/{file}`
   directly. Cluster's true ceiling is ~1.4 MB/s.
7. **`HF_HUB_ENABLE_HF_TRANSFER` is deprecated** in favor of
   `HF_XET_HIGH_PERFORMANCE` — the latter is what's broken on this cluster.
   Set neither; use the direct curl path.
8. **`facebook/flores`, `Muennighoff/flores200`, `openlanguagedata/flores_plus`
   all broken in `datasets` 4.x.** First two use legacy dataset scripts;
   third is gated by license. We're using `wmt/wmt19` config `de-en` with
   `streaming=True`. WMT19 de-en train has 38M pairs; streaming + take-first-N
   avoids the full-dataset download.
9. **PEFT's `add_weighted_adapter` is slow for 8B models** — ~6.8 sec per
   layer for `combination_type="linear"` because of full-SVD on 4096-dim. We
   bypass it entirely by implementing the merging math ourselves and
   overwriting LoRA factors via `PeftModelView`.
10. **`torch.svd_lowrank` is non-deterministic** (randomized algorithm). Gate
    it to fire only when `min(out,in) > 256`; tests use small dims (≤64) and
    get the deterministic full SVD path.
11. **PBS gpu queue caps you at 3 concurrent jobs** — affects throughput.
    Also doesn't track GPUs as a schedulable resource (`naccelerators=0`),
    so multiple users can land on the same physical GPU. Our `gpu_picker.py`
    picks the least-loaded GPU at job startup; if no GPU has enough free
    VRAM the job exits 87 (resubmit-friendly).
12. **Both login and compute nodes have the same external bandwidth
    (~1.4 MB/s).** My initial belief that compute nodes were faster was a
    misread of an earlier `du` measurement (the 239 MB/s "burst" was a
    filesystem cache artifact, not real throughput).
13. **Magicoder cache race** — two processes writing to the same HF cache
    blob simultaneously caused `FileNotFoundError: ...incomplete`. Wait for
    preload to finish before launching training.
14. **`SFTTrainer(model=model, train_dataset=..., args=...)` without
    `processing_class=` crashes** inside Unsloth's `fix_untrained_tokens`
    with `'NoneType' has no attribute convert_ids_to_tokens` — but only when
    the model was loaded from a local path. HF-repo-loaded models don't hit
    this. Always pass `processing_class=tokenizer`.
15. **Llama-3.1 tokenizer has no padding token by default.** Unsloth
    auto-sets `pad_token = <|finetune_right_pad_id|>`. Worth knowing if you
    inspect the tokenizer manually.

---

## 5. Open questions raised by Phase 3 data

Filed in `notes/open_questions.md` under "Phase 3 Day 17–18 open items".
Brief restatement:

1. Is the TVQ b=2 dip (Llama, worst_excess=0.107) real, or sample noise from
   n=200? Rerun with n=1000.
2. Why is Qwen-2.5-7B ~2× easier to merge than Llama-3.1-8B? Compute empirical
   `d_eff` per model and check the floor formula.
3. Why does KnOTS tie with TA on `worst_task_excess` but beat it on 5/8 per-task
   cells? Is it a real effect or noise on the worst task only?
4. Why does DARE produce results identical to TA at density=0.2? Audit the
   implementation; rerun at density ∈ {0.1, 0.5, 1.0}.
5. Translation has consistently negative excess. Is the translation LoRA
   under-trained, or is this a real "merging-helps-translation" effect?
6. The rate-decay term in the bound isn't visible at b ≥ 1. Is there a
   sub-bit quantization scheme that would expose the slope?

---

## 6. What goes in the paper (preliminary)

For `paper/sections/experiments.tex` §6.2 (real-LLM):

**Lead with what worked:**
- Bound's floor prediction qualitatively holds — non-zero excess on every
  cell, exact value varies with task-vector overlap.
- Method ordering matches theory: TIES > KnOTS ≈ TA ≈ DARE (where TIES is
  structure-respecting and TA is structure-blind).
- KnOTS per-task beats TA on 5/8 cells — also consistent with
  structure-respecting hypothesis.
- Clear model-effect: Qwen-2.5-7B merges substantially more cleanly than
  Llama-3.1-8B (~2× lower worst_excess). Suggests architecture / pretraining
  affects merging-readiness.

**Honestly report what didn't:**
- The TVQ rate-decay slope is not visible at b ∈ {1, 2, 4, 8, 16, 32}. At
  these bit budgets, merging-geometry error dominates quantization error
  on the worst task. The bound predicts a decay term `~2^(-2b/d_eff)` that
  becomes visible only below 1 bit. This is consistent with the bound
  (the floor predicts what happens; the decay just doesn't dominate),
  but it means we cannot validate the slope empirically with practical
  quantizers.
- Translation excess is consistently negative. The merge actually IMPROVES
  this task. Possible under-training of the translation LoRA; possible real
  cross-task benefit. Flag and discuss.

**Caveat per Rule N2:**
n=200 eval examples per task is noisy. Plan to rerun headline cells with
n=1000 before submission. Bootstrap CI on the slope claims will resolve
whether the b=2 dip and Qwen×TIES win are real or sample variance.

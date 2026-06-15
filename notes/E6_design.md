# E6 — T sweep on real adapters (design)

**Status:** DRAFT (2026-06-15 ~11:15 IST) — scaffolding only. Do not
launch on cluster without explicit sign-off.
**Master plan reference:** §E6.
**Purpose:** real-data complement to E4's synthetic T sweep. Test
whether the achievability ratio of E1's encoder vs the lower bound
remains stable as $T$ grows, and whether the algorithmic ordering
(TIES dominates baselines, b=2 dip persists) survives at $T$ beyond 4.

---

## 1. Why E6 matters for the paper

- E1 / §6.2 established the regularized achievability salvage at $T=4$.
- E4 measured the achievability *constant*'s growth in $T$ on
  synthetic data: log-T with $1/\sqrt{r}$ slope correction.
- E6 closes the loop: same growth pattern, but on real adapters
  across $T \in \{2, 4, 8, 12\}$. Confirms or contradicts the
  log-$T$ behavior on the kind of LoRA $\Delta$ tensors the
  ridge-regularized encoder actually has to merge.

If E6 confirms log-$T$ on real adapters, the paper has the full
chain: lower bound (theorem) → matching achievability (E1 + ridge)
→ T-scaling consistent with theory (E4 + E6).

---

## 2. Task pool — 12 tasks

Per master plan:

| Task | Dataset | Type | Notes |
|---|---|---|---|
| GSM8K | openai/gsm8k | math | already have v1 adapters |
| MATH | hendrycks/competition_math | math | new |
| AQuA-RAT | aqua_rat | math (multiple choice) | new |
| Magicoder | ise-uiuc/Magicoder-OSS-Instruct-75K | code | already have v1 |
| CodeAlpaca | sahil2801/CodeAlpaca-20k | code | new |
| Alpaca-cleaned | yahma/alpaca-cleaned | instruct | already have v1 |
| Dolly | databricks/databricks-dolly-15k | instruct | new |
| WMT19 en-de | wmt/wmt19 (de-en) | translation | already have v1 (flores key) |
| WMT19 en-fr | wmt/wmt19 (fr-en) | translation | new |
| SQuAD QA | rajpurkar/squad_v2 | extractive QA | new |
| XSum summarization | EdinburghNLP/xsum | summarization | new |
| MNLI | nyu-mll/multi_nli | NLI | new |

**Currently on disk:** 4 of 12 tasks have trained v1 adapters
(gsm8k, magicoder, alpaca, flores). **8 new tasks** need v1 adapters
trained.

---

## 3. Models

Per master plan: **two** base models —

- **Llama-3.1-8B-Instruct** — hardest case per the algorithmic gap
  ordering, AND the E3 §6.5 outlier whose behavior we want to
  characterize at higher T.
- **Yi-1.5-9B-Chat** — easiest, anchor for the algorithmic floor.

Rank 16, target modules q/k/v/o, single seed (use 20260615 for
clean separation from prior seeds).

---

## 4. Cost estimate

### Training the 8 new adapters per model

8 tasks × 2 models = **16 new trainings**.

Per task at ~$7500$ examples × 2 epochs:
- 7B model (yi is 9B but close): ~$60$ min/training on a single A100
- 8B model (llama): ~$70$ min/training

So 16 × ~$65$ min average = ~$17$ GPU-h. On 3-wide GPUs 2/4/6:
~6 wall-clock hours.

### Merge + evaluation matrix

For each $T \in \{2, 4, 8, 12\}$:

- $T = 2$: $\binom{12}{2} = 66$ task pairs is too many; use one
  fixed nested chain plus 3 random draws.
- $T = 4$: $\binom{12}{4} = 495$ subsets; use one fixed nested
  chain plus 3 random draws.
- $T = 8$: similar; one nested chain + 3 random draws.
- $T = 12$: only $\binom{12}{12} = 1$ choice.

So per model: $(1 + 3) + (1 + 3) + (1 + 3) + 1 = 13$ subset draws.

For each subset:
- 6 merge methods (TA, TIES, DARE, KnOTS, TVQ b=2, rd_encoder ridge
  λ=0.05)
- Evaluation on all $T$ tasks held-out at $n=500$

Per (model, subset, method) cell: ~$T \cdot n$ generations $\approx$
2-12 × 500 = 1000-6000 NLL evals. NLL is fast (~$0.05$ s per example
on bf16). Plus merge: ~$10$ s. Total cell time:
- $T = 2$: ~$2$ min
- $T = 4$: ~$3$ min
- $T = 8$: ~$5$ min
- $T = 12$: ~$8$ min

Per model: 13 subsets × 6 methods × average ~$4$ min = ~$5$ GPU-h.

For 2 models: ~$10$ GPU-h merge-eval.

### d_eff analysis

For each subset draw, compute hard + soft $d_{\mathrm{eff}}/(Tr)$ per
layer. CPU only, ~$1$ min per subset. 26 subsets total. **Trivial.**

### Total budget

$17 + 10 = $ **~27 GPU-hours**. Comfortably fits in a 24h walltime
on 3-wide; really ~$9$ wall-clock hours.

---

## 5. Pilot (recommended before full launch)

Before committing to the full 16-training + 26-subset matrix:

- **Pilot:** Yi-1.5-9B only, train the 8 new tasks, run the merge
  matrix at $T \in \{4, 8, 12\}$ only (skip $T=2$ for the pilot
  since $T=2$ is similar to the existing matrix).
- Cost: 8 trainings ($\approx 8$ GPU-h) + 7 subsets × 6 methods ×
  ~$5$ min = ~$3.5$ GPU-h. Total ~$12$ GPU-h, ~$4$ wall-clock hours
  on 3-wide.
- Gate: does the achievability ratio grow log-$T$ on Yi's real
  adapters, matching E4's synthetic prediction?
- If GO → run the full sweep including Llama.

---

## 6. Open design choices needing sign-off

1. **8 new task adapter trainings** — same hyperparameters as v1
   (rank 16, alpha 32, AdamW lr 5e-5, 2 epochs, bs 8 × ga 4).
   Confirm? Yes/no.

2. **Random subset seed** — use 20260615 for the 3-random-draws-per-T
   selections. Yes/no, or pick another seed.

3. **Eval data per task** — use the same 1000 held-out per task that
   §6.1 used, plus first 1000 of the new tasks' standard test splits.
   Confirm? Yes/no.

4. **rd_encoder ridge_lambda value** — use $\lambda = 0.05$ from
   §6.2 (llama) or $\lambda = 0.13$ (mistral/qwen/yi)? On Yi, the
   cross-model sweep said $\lambda = 0.13$. **Recommend: use the
   model-specific best λ.** Yes/no.

5. **Pilot first vs full sweep** — start with Yi-only pilot to
   validate the pipeline before committing 27 GPU-h. **Recommend:
   pilot.** Yes/no.

6. **What to do if log-$T$ does NOT replicate on real data** — fall
   back to reporting the empirical curve as a finding ("synthetic
   $\log T$ does not transfer at $T \leq 12$"). Pre-registered
   honesty branch.

---

## 7. Files this protocol will produce (when sign-off received)

Training:
- `code/phase3/configs/lora_e6/<model>__<task>__seed20260615.yaml`
  (16 configs for the new tasks)
- `code/phase3/scripts/pbs_orchestrator_e6_train.sh`
- `artifacts/lora/<model>/<task>/seed20260615/` (16 adapter dirs)

Eval matrix:
- `code/phase3/configs/eval_e6/<model>__T<T>_<subset>_<method>.yaml`
  (~$13 \cdot 6 \cdot 2 = 156$ configs per model, ~$312$ total)
- `code/phase3/configs/e6_manifest.json`
- `code/phase3/scripts/pbs_orchestrator_e6_eval.sh`
- `code/phase3/scripts/analyze_e6_T_sweep.py` (extract excess vs T)
- `code/phase3/figures/e6_T_scaling.png` (the headline figure for
  §6.6 paper section)

---

## 8. Paper integration

E6 produces the §6.6 paragraph (after §6.5 / E3):
"On real adapters across $T \in \{2, 4, 8, 12\}$, the achievability
ratio of E1's regularized encoder grows as $\rho \cdot \log T$ with
$\rho \approx [\text{measured}]$, consistent with E4's synthetic
prediction within $[\pm \text{spread}]$. The cross-task algorithmic
ordering (TIES dominates baselines, $b=2$ dip persists) holds at
all $T$ tested, validating the §6.1 NLL methodology beyond the
$T=4$ benchmark."

Combined with E1/E2/E3 this gives the paper a complete
"theory $\to$ synthetic $\to$ real, at multiple $T$" chain.

%%% END DESIGN — awaiting sign-off %%%

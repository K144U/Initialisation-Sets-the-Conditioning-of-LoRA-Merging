# Phase 3 design — real-LLM empirical validation

**Status:** draft, 2026-04-26. To be edited by Sankalp; the goal is
to reach alignment on the metric definition *before* writing any
code. The single most important question this doc answers is:

> What exactly is "worst-task distortion" on a real LLM, what's the
> unit, what data do we use, and what's the baseline?

Everything else (which models, which tasks, which merging methods)
is downstream of that. If the metric is wrong, the experiment is
wrong.

---

## 0. Why this phase exists

The paper currently has:

- A clean theoretical RD lower bound on
  $\E_{P^\star}[\max_t \|w - \tau_t\|_{H_t}^2]$ matched by an
  explicit Gaussian-QR + uniform-scalar-quantization achievability
  scheme (Theorems 3 and 4).
- Synthetic validation: slopes match $-2.00 \pm 0.01$, achievability
  constant in $[11, 13]$ vs theoretical $\sim 17$.
- Discussion §6 candidly admits no real-LLM experiments. Section
  \ref{subsec:exp-lora} of `experiments.tex` defers them to "a
  future version of this paper."

This is the **"so what" gap.** The theory says merging *as a
quantization-of-task-vectors problem* has a fundamental floor and
a $2^{-2R/d_{\mathrm{eff}}}$ rate-distortion law. To make this a
paper that ICLR cares about (vs ISIT, where pure theory is fine),
we need to demonstrate that:

1. The bound's *floor term* $B^2(1 - d_{\mathrm{eff}}/(Tr))$
   predicts a non-trivial floor on real merged-LoRA loss.
2. The bound's *rate term* predicts the empirical
   compression-vs-quality tradeoff for quantization-aware merging
   methods (TVQ, 1bit-Merging).
3. Methods that respect the bound's structure (operate on the
   $\bar H$-metric, exploit null-space) outperform methods that
   don't, holding rate fixed.

Phase 3 is the experimental program that tests these three claims.
This doc plans only the metric and baselines; the full experimental
matrix is downstream.

---

## 1. The metric question

### 1.1 What the theory bounds

The theoretical object is

$$
\mathcal{D}^\star(R)
= \inf_{\text{rate-}R\ \text{codes}}
  \E_{P^\star}\Bigl[\max_t \|w^\star - \tau_t\|_{H_t}^2\Bigr],
$$

where $\tau_t \in \R^d$ is the task-$t$ LoRA delta, $w^\star$ is
the merged delta the decoder reconstructs from the rate-$R$ code,
and $H_t \succeq 0$ is the task-$t$ "loss curvature" matrix. The
$H_t$-norm captures how a perturbation $w - \tau_t$ translates to
extra loss on task $t$ via the local-quadratic approximation
$\mathcal{L}_t(\theta_0 + w) - \mathcal{L}_t(\theta_0 + \tau_t)
\approx \frac{1}{2}(w - \tau_t)^\top H_t (w - \tau_t)$.

### 1.2 What we can measure on a real LLM

The operational quantity we *actually want* is **excess loss
under merging**:

$$
\Delta\mathcal{L}_t(w)
:= \mathcal{L}_t(\theta_0 + w) - \mathcal{L}_t(\theta_0 + \tau_t),
$$

evaluated on a held-out eval set for task $t$, where $\theta_0$ is
the base model and $\tau_t$ is the rank-$r$ LoRA fine-tune for
task $t$. The natural Phase-3 analog of $\mathcal{D}^\star$ is

$$
\widehat{D}(w) := \max_t \Delta\mathcal{L}_t(w),
$$

evaluated over the $T$ tasks in the merge.

**Question that needs Sankalp's input:** is $\Delta\mathcal{L}_t$
measured as

  (a) NLL (cross-entropy) on a held-out split, in nats?
  (b) downstream accuracy / F1 / exact-match drop, in %-points?
  (c) instruction-following winrate vs the unmerged $\tau_t$, in %?

**Strong recommendation: (a), with (b) reported as a secondary
metric.** Reasons:

- (a) is what the local-quadratic bridge in §1.1 actually controls.
  The bound is on $\|w - \tau_t\|_{H_t}^2$ where $H_t$ is the
  Fisher information matrix; $H_t$-norm equals NLL Hessian, not
  accuracy Hessian. Reporting accuracy and claiming it validates
  an NLL-controlling bound is a category error — easy for a
  reviewer to flag.
- (b) is what practitioners care about, so we report it for
  legitimacy. But it's noisier and discontinuous (small NLL
  changes ↛ small accuracy changes), so slope fits will be
  worse-behaved.
- (c) is too expensive on single-GPU and requires a strong judge
  model. Skip.

**Implication for the metric:** define excess as nats per token,
averaged over a fixed eval split (≥ 2k tokens per task to keep CI
tight), and report a max over $t$.

### 1.3 Operational metric (proposed)

For each task $t \in [T]$ in the merge:

1. Hold out 2k–5k tokens from task $t$ as the eval split. Same
   split used to evaluate every checkpoint (base, $\tau_t$,
   merged $w$).
2. Compute $L_t(\theta) := -\frac{1}{N_t}\sum_{i \in \text{eval}_t}
   \log p_\theta(\text{token}_i)$ (mean NLL, nats per token).
3. Define $\Delta L_t(w) := L_t(\theta_0 + w) - L_t(\theta_0 + \tau_t)$.
   This is non-negative if $\tau_t$ is well-trained and $w$ is a
   compromise; could go negative if the merged delta accidentally
   helps task $t$ (rare but possible — flag and investigate).
4. **Phase-3 worst-task distortion:**
   $$
   \widehat{D}(w) := \max_t \Delta L_t(w).
   $$

This is one scalar per merging method per rate budget $R$. The
plot we're chasing is $R \mapsto \widehat{D}(w)$ for each
baseline, on the same $T$ tasks.

**Note.** The theoretical bound is over the *random-task-vector*
distribution $P^\star$. Real LoRA tasks are not Stiefel-random.
The right framing is: the bound is a **lower envelope on what is
achievable for the worst-case task-vector configuration**, and we
expect real merges to sit somewhere above this envelope. We are
NOT claiming real merges saturate the bound. We ARE claiming the
bound's *qualitative predictions* (floor exists, rate-2 decay,
better methods get closer) hold up.

### 1.4 What "rate" means on a real LLM

Total bits stored per merged delta. For an LLM with $L$ adapted
linear layers (typically $\sim 100$ for a 1-2B model with LoRA
on Q/K/V/O of every transformer block), each layer contributing
a rank-$r$ delta of dimension $r(d_{\text{in}} + d_{\text{out}})$
parameters, the rate is

$$
R_{\text{total}} = \sum_{\ell=1}^{L} R_\ell,
\qquad
R_\ell = b_\ell \cdot r(d_{\text{in},\ell} + d_{\text{out},\ell}),
$$

where $b_\ell$ is the bits-per-parameter for layer $\ell$. For a
fair compression-vs-quality plot, we'll report $b := R_{\text{total}}
/ \sum_\ell r(d_{\text{in},\ell} + d_{\text{out},\ell})$ — average
bits per LoRA parameter — on the x-axis. Standard merging methods
(Task Arithmetic, TIES, DARE) implicitly use $b = 32$ (full FP32
deltas). TVQ and 1bit-Merging hit $b \in \{1, 2, 4\}$.

---

## 2. Models

**Recommendation:** **Qwen2.5-1.5B** as primary, **Gemma-2-2B** as
secondary, both chat / instruct variants.

Why:

- Both fit in 24 GB VRAM (RTX 3090/4090) with LoRA fine-tuning
  at rank $r = 16$ on Q/K/V/O at full sequence length 2048.
- Both have a published instruct version (so existing instruction
  data formats), good HuggingFace ergonomics, permissive licenses.
- Different architectures (Qwen: GQA, Gemma: MHA-with-RoPE
  variants) — using both shows the bound is not architecture-
  specific.

Skip:

- Phi-3 — Microsoft licensing is fine but the architecture is
  doing weird things with sliding window that complicates Fisher
  estimation.
- Llama-3.2-1B — fine but Qwen / Gemma are stronger.
- Anything > 3B — out of single-GPU LoRA budget at reasonable
  sequence lengths.

---

## 3. Tasks (T = 4 proposed)

**Recommendation:** four tasks chosen to give *varying subspace
overlap* — this is the knob the theory cares about.

1. **Math (GSM8K-train, 7.5k examples).** Cleanly defined task,
   well-studied for LoRA fine-tuning. Eval: GSM8K-test (1.3k).
2. **Code (Magicoder-110k, sub-sampled to 7.5k).** Eval:
   HumanEval (164) + MBPP-test (500). Different vocabulary
   distribution from math.
3. **Instruction-following (Alpaca-cleaned, 7.5k).** Eval:
   AlpacaEval split or held-out 500. General-purpose.
4. **Translation (FLORES en→de + en→fr, 7.5k).** Eval: FLORES
   test set, BLEU + NLL. Designed to overlap *minimally* with
   the other three.

**Why these four:** math+code likely share substantial subspace
(both heavy-symbol-manipulation), math+translation likely don't.
This gives the bound's $d_{\mathrm{eff}}$ knob something to bite
on. We can compute $d_{\mathrm{eff}} = \rank(\sum_t P_{V_t})$
empirically by SVD-ing each $\tau_t$ to extract $V_t$ and looking
at the rank of the sum of projectors.

**Per-task LoRA training:** rank $r = 16$, target Q/K/V/O of every
attention block, AdamW lr 5e-5, 2 epochs, batch size 8 with
grad-accum to effective 32. Same hyperparameters across all four
tasks for cleanliness.

---

## 4. Baselines (5 methods)

Want to span the design space.

| Method | Year | Bits/param | Captures |
|---|---|---|---|
| **Task Arithmetic** (Ilharco et al. 2023) | 2023 | 32 | naive linear baseline |
| **TIES-Merging** (Yadav et al. 2023) | 2023 | 32 | sign-resolution / sparsification |
| **DARE** (Yu et al. 2024) | 2024 | 32 | drop-and-rescale / pruning |
| **KnOTS** (Stoica et al. 2025) | 2025 | 32 | SVD-based subspace alignment |
| **TVQ** (Tang et al. 2024) | 2024 | 1–4 | quantization-aware merge |

**Why exactly five:** we need the first three (Task Arithmetic,
TIES, DARE) as community-standard baselines; KnOTS is the closest
related-work-in-spirit since it explicitly thinks about subspaces;
TVQ is the only published method that operates at low bits/param,
so it's the only one that exercises the rate-axis of our theory.

**Skip for v1:**
- 1bit-Merging — paper is concurrent / unverified.
- DO-Merging, Core Space — too new, may not have public code.
- Fisher merging (Matena & Raffel 2022) — interesting but adds a
  Hessian-estimation confound we don't want to debug.

We can revisit any of these for v2.

---

## 5. Compute and budget estimate

Single GPU (RTX 3090/4090 or Colab Pro A100-40GB).

**Per-task LoRA training:** ~30 min on 4090 for 7.5k examples,
2 epochs, seq 2048, rank-16. Total: 4 tasks × 2 base models = 8
training runs, ~4 hours.

**Eval:** ~10 min per (model, task, merging-method, rate)
combination. 5 methods × 4 tasks × ~6 rate points (where
applicable) × 2 base models = 240 evals × 10 min = 40 hours.

**Total:** ~50 GPU-hours of active compute. Fits in 1-2 weeks
calendar time at 4 hours/day.

This is the *minimum viable* Phase 3 — single seed, single base
model size. For a paper-quality experiment we'd want 3 seeds and
maybe a third model size. Bumps total to ~150 GPU-hours.

---

## 6. What "validation" looks like

For Phase 3 to *succeed* and unlock ICLR-worthiness, we need:

1. **Floor reproduces qualitatively.** Across all five methods at
   $b = 32$ (no quantization), there is a non-zero $\widehat{D}$
   that does not vanish with more parameters or finer-grained
   merging. The theory predicts this floor is
   $\propto B^2(1 - d_{\mathrm{eff}}/(Tr))$; we'd expect to see
   the floor *shrink* on tasks with higher empirical
   $d_{\mathrm{eff}}$ (math+translation should have lower
   $d_{\mathrm{eff}}/(Tr)$ ratio → bigger floor, vs math+code
   which should overlap → smaller floor).
2. **Rate exponent reproduces.** For TVQ at $b \in \{1, 2, 4, 8\}$,
   excess over the $b = 32$ floor decays at slope $-2$ in $b$ (or
   close to it), validating $2^{-2R/d_{\mathrm{eff}}}$.
3. **Method ordering matches subspace-respecting hypothesis.**
   KnOTS (subspace-aware) should beat Task Arithmetic (subspace-
   blind) on tasks with low $d_{\mathrm{eff}}$, and the gap should
   narrow as $d_{\mathrm{eff}} \to Tr$.

If any of these fail, we honestly report that, frame it as a
limitation, and the paper is still a contribution as a clean
theoretical result. But (1) is the bare-minimum: if no floor
shows up at all, the theory is decoupled from practice in a way
reviewers will punish.

---

## 7. What this doc does NOT cover

- Concrete code (training scripts, eval scripts, merging
  implementations). Punt to a follow-up `phase3_implementation.md`
  once metric / models / tasks are locked.
- Hyperparameter sensitivity for LoRA training. Will check
  in-flight; the merging-comparison robustness should be
  insensitive to LoRA hyperparameters within reason.
- The $H_t$-norm vs Euclidean-norm question on real LLMs. The
  theory uses $H_t$ (Fisher); the standard merging literature
  uses Euclidean. We'll report both and discuss the gap.
- Failure modes (catastrophic forgetting on the base, distribution
  shift in eval). Real concerns; flag them in v2.

---

## 8. Open design questions for Sankalp

These are the points where the doc has me guessing — your call.

1. **Metric: NLL only, or NLL + accuracy?** Recommendation: both,
   NLL primary. (See §1.2.)
2. **Tasks: 4 vs 3 vs 5?** Four feels right. Three is too few
   to see overlap variance; five is too many to evaluate at
   single-GPU.
3. **Models: Qwen + Gemma, or just Qwen?** Just Qwen halves the
   compute budget but weakens the "not architecture-specific"
   claim. Recommendation: just Qwen for v1, add Gemma if there's
   slack before submission.
4. **Rate axis: report bits/param averaged, or bits per layer?**
   Averaged is cleaner. Per-layer is more honest about
   non-uniform allocations. Recommendation: averaged for the
   headline plot, per-layer in supplementary.
5. **Random task vectors as a control?** We could include a
   "synthetic Stiefel-random $\tau_t$" baseline at the same
   rank, evaluated by simulating the merge in float — to show
   the bound is tight on the synthetic distribution and loose
   (in a known direction) on the real distribution. Probably
   worth it. Recommendation: include as a single panel.
6. **Should we run Phase 3 on a *different* paper?** I.e., this
   experimental program could be its own paper ("RD theory
   meets practice for LoRA merging", JMLR or NeurIPS empirical
   track) and v1 of the theory paper goes to ISIT *without*
   real-LLM. Pro: faster ICLR submission, cleaner scope per
   paper. Con: theory paper looks toy without empirics. Decide
   after talking to Garg.

---

## 9. Next concrete step

Sankalp edits this doc, locks the metric definition, and decides
on the §8 questions. Once that's done, the next file is
`notes/phase3_implementation.md` covering the actual scripts and
a 2-week sprint plan.

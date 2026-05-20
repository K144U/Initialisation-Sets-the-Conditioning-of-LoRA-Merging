# rdmerge — paper-writing reference log

A single consolidated brain-dump: every finding, decision, speculation, fix,
and idea worth knowing when we sit down to write. Updated as we go.

**Last updated:** 2026-05-19 ~18:45 IST (Day 19 evening close: **T1.D
eval matrix COMPLETE** at 18:29:11 — 4-model × 10-cell = 40 cells total
in `eval_matrix_n1k_v2/`; **Phase B regenerated on 4-model data** via
the refactored `phase_b_analysis.py` (auto-detect models + dynamic
subplot grid); paper_artifacts/ now points at 4-model figures + summary;
**F1, F4, F7 confirmed universal across 4 architecture families;**
**Yi-1.5-9B is the easiest model to merge** (TA floor 0.099, beating
Qwen's 0.110 despite being Llama-architecture — strong "training matters
more than architecture" signal); ICLR 2027 target unchanged).

This file overlaps deliberately with `notes/daily_log.md` (chronological),
`notes/phase3_findings.md` (Phase 3 numerics), and `notes/open_questions.md`
(unresolved items). When writing the paper, start here.

> ### What's SOLID vs TENTATIVE as of 2026-05-19 evening (post-T1.D)
>
> **Solid (paper-quotable):** all theory (Thm 7, 8, 9, Lemma 6); Phase 2
> synthetic slope $-1.60 \pm 0.10$; Phase 2.5 general-T Chebyshev solver;
> LB-sharpening ruled out for $T \geq 3$; the 16 trained LoRAs (4 models
> × 4 tasks); the **d_eff = Tr structural finding (F9)** — computed from
> B@A factors, not eval data, on the original 8 LoRAs; and **all 40
> v2+T1.D eval-matrix numbers** (`results/phase3/eval_matrix_n1k_v2/`)
> — clean loader, n=1k, 4 architectures. Promoted to `paper_artifacts/`.
>
> **Tentative:** T1.B redesign decision (parked; soft d_eff T1.A2 is the
> recommended replacement; if green-lit, run on the 16 LoRAs to get
> per-layer participation-ratio d_eff across 4 architectures); the
> b=2-dip mechanism question (now confirmed universal across 4
> architectures); a fresh d_eff analysis on the 8 new Mistral+Yi LoRAs
> (not yet done — would complete the 4-model d_eff picture).
>
> **What just became solid (T1.D drop):** F1, F4, F7 generalize to 4
> architectures. F2 reframed: **Yi-1.5-9B (Llama-arch) is the easiest
> to merge, beating both Qwen-2.5-7B (GQA) and Llama-3.1-8B (MHA)** —
> so the "Qwen ≪ Llama" effect isn't really about GQA; it's about
> pretraining and instruction-tuning recipe. The b=2 universal dip
> survives clean data across all 4 families. C2 (slope ≈ −2) skipped
> on all 4 — confirmed not a 2-model artifact, universal regime.
>
> **2026-05-19 night additions:** **F9 hard d_eff = Tr universal across 4
> architectures** (every layer of every model = 64). **F10 soft d_eff
> ≈ 16.4 across all 4 (within 0.7%) but does NOT explain F2** —
> Pearson r=+0.53 (p=0.47) vs observed TA worst_excess. So F2 mechanism
> isn't subspace geometry; remaining candidates are H_t curvature,
> pretraining recipe, and B² scale. v3 per-example re-eval launched
> (~16 hr) to enable bootstrap CI on every cell — user decision: no
> compute shyness, no hidden uncertainty.
>
> **2026-05-20 — v3 lands + bootstrap CIs (canonical eval is now v3):**
> 40 cells re-evaluated with per-example logging (`eval_matrix_n1k_v3_perexample/`);
> point estimates within ~0.001 of v2 (GPU noise). 10k-resample bootstrap
> on every cell (`bootstrap_ci_v3.json`). **Headlines proven significant:**
> F1 TIES (14/16 cells, large), F7 b=2 dip (4/4 non-overlapping vs b=4),
> F2 full ordering Llama>Mistral>Qwen>Yi (all CIs non-overlapping incl.
> Qwen-vs-Yi). **Correction: F4 (KnOTS>TA) was NOISE** — paired bootstrap
> 5/16 sig, split both ways, ≤0.0019 nats; KnOTS ≈ TA, theory-consistent
> at d_eff=Tr. F5 DARE ≈ TA confirmed (≤0.0030 nats). **Only TIES
> meaningfully differs from TA** (practical threshold |Δ|>0.005). C3
> reframed: see §3.5.

---

## §1 Project at a glance

- **Working title:** *A Rate-Distortion Lower Bound for Model Merging, with Matching Achievability via Hadamard Incoherence.*
- **Thesis:** Given T task-specific LoRA adapters over a shared base, derive a Shannon-style lower bound on bits-per-parameter required to store a merged model that preserves ε-distortion on each task. Match it constructively via Hadamard rotation + uniform scalar quantization.
- **Audience the bound is for:** ML practitioners who merge LoRA adapters (TIES/DARE/KnOTS/TVQ communities), info-theory researchers (TurboQuant family).
- **Primary venue:** ICLR 2027 (deadline ~2026-09-24).
- **Backups:** AISTATS 2027 (~2026-10-08, 2 weeks later), TMLR (rolling), ISIT 2027 (~2027-01).
- **Authors:** Sankalp Pathak (lead), Prof. Sanjay Garg (joined 2026-04-25).
- **Paper contact email:** `pathaksankalp04@gmail.com`. (NEVER use the `kittuwastaken@gmail.com` Claude/system email on the paper.)
- **Repo:** `~/projects/rdmerge/` on `jiit-master` (HPC cluster).
- **Branch:** `phase3-bootstrap`.

---

## §2 What the theory says (Phases 0-2.5, all closed by 2026-04-24)

### §2.1 The objects

- T task-specific LoRA deltas: $\tau_t \in \mathbb{R}^d$ each, rank-$r$.
- A merged delta: $w \in \mathbb{R}^d$ produced by a rate-$R$ merging code.
- Task-$t$ loss curvature: $H_t \succeq 0$ (Fisher at base $\theta_0$).
- Worst-task distortion (the operational metric): $\widehat{D}(w) = \max_t \|\tau_t - w\|_{H_t}^2$.
- Effective dimension: $d_{\mathrm{eff}} = \mathrm{rank}(\sum_t P_{V_t})$ where $V_t$ is the top-$r$ right-singular subspace of $\tau_t$.

### §2.2 What we proved

- **Theorem 7** (Phase 1, in `theory/theorem_v1.tex`): if $H_t = P_{V_t}$ (idealized projection curvature), then $\mathcal{D}^\star(R) \geq B^2(1 - d_{\mathrm{eff}}/(Tr)) + C \cdot 2^{-2R/d_{\mathrm{eff}}}$ with explicit $C$.
- **Theorem 8** (Phase 1): generalizes to arbitrary $H_t \succeq 0$ and task-dependent $D_t$.
- **Lemma 6** (Phase 1): closed-form for the floor term $B^2(1 - d_{\mathrm{eff}}/(Tr))$.
- **Theorem 9** (Phase 2, achievability): Hadamard-rotation + uniform scalar quantization scheme achieves distortion within constant $C = Tc^2/3$ of the lower bound.
- Phase 2 numerics (T=2 shared-V): slope $-1.60 \pm 0.10$ across 5 anisotropy regimes (via `code/synthetic/day14g_final_lock.py`). At $c=11.5\sigma_{pc}$, 1000 trials, bootstrap CI $\pm 0.010$.
- Phase 2.5 numerics (general-T, T=3,4): slopes match the linear prediction $-2r/(r+|A|-1)$ after switching to **excess-over-cheb²-per-trial** metric (NOT excess-over-avg-floor, which flattens at high R for T≥3 — don't use it).

### §2.3 What's RULED OUT (don't re-attempt)

- LB-sharpening at exponent $-2r/(r+|A|-1)$ for $T \geq 3$: empirically refuted by `day16_sharpened_lb_check.py`. A valid rate-R random-codebook encoder (no data-dependent shift) achieves slope $-1.45$ at T=3, r=3 vs the linear UB at $-1.20$ — would violate any LB at that exponent. True RD lies strictly between $-2r/r$ (loose Thm 8 LB) and $-2r/(r+|A|-1)$ (loose linear UB). Exact RD is open but out of scope.
- "Excess-over-avg-floor" metric for $T \geq 3$ anisotropy shared-V: flattens at high R. Use excess-over-cheb²-per-trial.

### §2.4 What's still open theoretically (low priority for this paper)

- Tightening the matching-UB constant $C = Tc^2/3$ to $O(\log T)$ or $O(1)$.
- Promoting the Fisher/cross-entropy extension from a remark to a theorem (preempts the "your bound is MSE, LLMs are CE" objection).
- Exact RD for general T (likely a future paper; not this one).

---

## §3 What Phase 3 (real-LLM) actually showed (Days 0–4, 2026-05-17 → 2026-05-18)

### §3.1 The setup

- **Hardware:** JIIT HPC cluster, 8× A100-SXM4-80GB on `jiit-gpu01` (PBS gpu queue, 3-job cap per user, no GPU resource tracking). External bandwidth to HuggingFace ≈ 1.4 MB/s (cluster-wide).
- **Software:** Python 3.11, torch 2.10.0+cu128, transformers 5.5.0, peft 0.19.1, trl 0.24.0, accelerate 1.13.0, datasets 4.3.0, unsloth 2026.5.2. (Most pinned in `plan.md` were overridden by Unsloth's transitive deps.)
- **Models:** Llama-3.1-8B-Instruct + Qwen-2.5-7B-Instruct, full bf16 (NO nf4). Both downloaded into `models/` via plain `wget --continue` (after Xet self-throttled to 30 KB/s).
- **Tasks (T=4):**
  - GSM8K (math, openai/gsm8k main)
  - Alpaca-cleaned (instruction-following)
  - Magicoder-OSS-Instruct-75K (code)
  - wmt19 de-en (translation; replaced facebook/flores after `datasets` 4.x dropped legacy scripts)
- **LoRA config (uniform across all 8):** rank 16, target Q/K/V/O of every attention block, AdamW lr 5e-5, 7500 train examples × 2 epochs, batch 8 grad-accum 4 (effective 32), seq 2048.

### §3.2 The 8 LoRA adapters (NLL_τ on n=200 held-out, nats/token)

|                | GSM8K | Alpaca | Magicoder | Translation |
|----------------|------:|-------:|----------:|------------:|
| Llama-3.1-8B   | 0.446 | 1.018  | 0.275     | 0.719       |
| Qwen-2.5-7B    | 0.418 | 0.946  | 0.238     | 0.794       |

Per-LoRA wall-clock: 32–42 min for math/instruction/translation; 65–90 min for Magicoder (code examples are ~3× longer).

### §3.3 The 40-cell eval matrix — 4 models (n=1k, fixed loader, 2026-05-19 evening)

**Authoritative**, supersedes the v2 2-model tables (preserved in `notes/phase3_findings.md` §2v historical block) and the Day-4 n=200 buggy tables (`§2` historical). Source dir: `results/phase3/eval_matrix_n1k_v2/` (40 cells).

Non-TVQ (worst_excess at implicit b=32):

|                                  | Llama-3.1-8B | Qwen-2.5-7B | Mistral-7B | Yi-1.5-9B |
|----------------------------------|-------------:|------------:|-----------:|----------:|
| Task Arithmetic (uniform)        | 0.2179       | 0.1095      | 0.1452     | 0.0987    |
| TIES (density=0.2)               | **0.1597**   | **0.0137**  | **0.0569** | **0.0458**|
| DARE (density=0.2)               | 0.2183       | 0.1119      | 0.1466     | 0.1022    |
| KnOTS (linear inner merge)       | 0.2178       | 0.1098      | 0.1453     | 0.0988    |

TVQ rate sweep (worst_excess):

| b  | Llama  | Qwen   | Mistral | Yi |
|----|-------:|-------:|--------:|---:|
| 1  | 0.2336 | 0.1375 | 0.3474  | 0.3044 |
| **2** | **0.1042** | **0.0184** | **0.0626** | **0.0599** |
| 4  | 0.2179 | 0.1098 | 0.1456  | 0.0987 |
| 8  | 0.2189 | 0.1099 | 0.1452  | 0.0989 |
| 16 | 0.2188 | 0.1094 | 0.1459  | 0.0989 |
| 32 | 0.2178 | 0.1098 | 0.1456  | 0.0990 |

b=2 dip ratios (b=4 / b=2): Llama **2.09×**, Mistral **2.33×**, Qwen **5.97×**, Yi **1.65×** — **universal across all 4 architecture families.** avg_excess at b=2 is negative only for Qwen (−0.0066); positive but small for the other three (+0.014 to +0.018).

**Merge-difficulty ordering** (TA floor as proxy, hardest → easiest): **Llama-3.1-8B (0.218) > Mistral-7B (0.145) > Qwen-2.5-7B (0.110) > Yi-1.5-9B (0.099)**. Yi-1.5-9B is Llama-architecture (same MHA) yet is **2.2× easier to merge than Llama-3.1-8B** — so GQA-vs-MHA is not the full F2 explanation; pretraining + instruction-tuning recipe materially affects merging-readiness.

Promoted headline figures: `paper_artifacts/figures/{headline_rd,method_compare}.png` (4-model; 2-model version preserved as `*_v2_2model.png`; Day-4 buggy as `*_old.png`). Source: `code/phase3/figures/v2_4model/`.

### §3.4 The 7 qualitative findings

- **F1 — TIES wins worst_task_excess on every architecture (4 of 4), paired-bootstrap significant.** TIES vs TA on worst_task_excess: Llama-3.1-8B **27%**, Mistral-7B **61%**, Yi-1.5-9B **54%**, Qwen-2.5-7B **87%** smaller. **Paired bootstrap (2026-05-20, `method_diff_ci_ties_v3.json`): 14/16 (model,task) cells significant.** BUT the structure matters and we report it honestly: TIES does NOT dominate uniformly per-task — it **trades off**. On **gsm8k (the bottleneck task) TIES wins big and significantly on all 4 models** (−0.058 to −0.106 nats), and that's what drives the worst_task_excess win. On non-bottleneck tasks TIES is mixed and sometimes significantly *worse* than TA (e.g. Mistral/Yi alpaca + magicoder + translation favor TA). Honest framing: **TIES's sign-election + magnitude-pruning resolves destructive interference on the hardest task, at some cost to easy tasks; net worst-task effect is a clear win because the worst task is gsm8k.** This — not KnOTS (see F4) — is the validation of "structure-respecting > naive."
- **F2 (REFRAMED on 4-model data) — Merging difficulty is ordered Llama > Mistral > Qwen > Yi (TA floors 0.22 > 0.15 > 0.11 > 0.10).** Yi-1.5-9B is Llama-architecture (same MHA) but is **2.2× easier to merge** than Llama-3.1-8B. So GQA-vs-MHA is NOT the full F2 explanation as originally conjectured — pretraining distribution and instruction-tuning recipe materially affect merging-readiness. **The strongest empirical argument we have that "what makes a model merge cleanly" is a property of training, not just architecture.**
- **F3 — Translation has consistently negative excess.** Every cell, both models: merged model is BETTER at translation than the translation-only LoRA. Possibly: translation LoRA under-trained (7.5k wmt19 examples + 2 epochs is light); possibly real cross-task benefit (math/code/instruction LoRAs add general grounding). Action: retrain translation at 15k × 3ep and re-eval.
- **F4 — CORRECTED 2026-05-20: KnOTS ≈ TA. The earlier "KnOTS beats TA per-task" claim was NOISE.** The per-task win count swung 5/8 → 11/16 → 7/16 across GPU-non-deterministic reruns — a red flag the per-task gaps were within noise. **Paired bootstrap (`method_diff_ci_v3.json`; KnOTS vs TA — the shared τ baseline cancels, so the test is on the merged-NLL difference): only 5/16 cells significant, split BOTH directions (3 favor KnOTS, 2 favor TA), all magnitudes ≤ 0.0019 nats — practically zero.** 11/16 within noise. **KnOTS (linear inner merge) is statistically and practically indistinguishable from Task Arithmetic in our setup.**

  **This is consistent with theory, not a contradiction.** The original C3 hypothesis (handoff §12) was precise: KnOTS's subspace-aware advantage holds *on low-d_eff task sets* and *narrows as d_eff → Tr*. We measure **d_eff = Tr saturated everywhere (F9)** — the regime where the theory predicts the advantage *vanishes*. KnOTS ≈ TA is exactly what the theory predicts at d_eff = Tr: the subspace-aware method has no overlap to exploit when task subspaces are already linearly independent. So F4 "failing" closes a loop with F9 rather than weakening the paper.

- **F5 — DARE ≈ TA (practically), bootstrap-confirmed 2026-05-20.** Paired bootstrap (`method_diff_ci_dare_v3.json`): 13/16 cells "significant" at n=1000, but all magnitudes ≤ 0.0030 nats and mostly slightly *favoring TA* (DARE's random drop+rescale adds noise that doesn't help at d_eff=Tr). Practically DARE = TA, marginally worse. Same story as KnOTS: nothing to exploit when subspaces are independent. (Density-sweep ablation still worth doing for the paper to show the d_eff→density interaction, but the density=0.2 ≈ TA result is now bootstrap-confirmed real, not a bug.)

  **Practical-vs-statistical note:** at n=1000, differences as small as ~0.0002 nats become "statistically significant." For the paper we apply a practical threshold (|Δ| > ~0.005 nats AND CI excludes 0). By that standard, **only TIES meaningfully differs from TA** — and only on the bottleneck task. KnOTS and DARE never cross the practical threshold.
- **F6 — TVQ rate doesn't matter at b ≥ 1.** worst_excess is essentially constant ≈ TA from b=4 upward. The merging-geometry error dominates the quantization error. Theory's rate-decay term only visible at sub-bit precision (unphysical for our scheme).
- **F7 — TVQ b=2 dip is UNIVERSAL across 4 architecture families** (Llama-3.1 MHA, Qwen-2.5 GQA, Mistral-v0.3, Yi-1.5 Llama-arch). Confirmed 2026-05-19 evening on the full 4-model panel:
  - Llama: b=2 worst_excess = 0.1042 vs b=4 = 0.2179 → **2.09× smaller at b=2**
  - **Qwen: b=2 = 0.0184 vs b=4 = 0.1098 → 5.97× smaller** (the dramatic one; avg_excess at b=2 is **−0.0066** — only model where merged beats avg task LoRA)
  - Mistral: b=2 = 0.0626 vs b=4 = 0.1456 → **2.33× smaller**
  - Yi: b=2 = 0.0599 vs b=4 = 0.0987 → **1.65× smaller** (mildest but unmistakable)
  - Both b=1 (too coarse) and b ≥ 4 (too fine) give ~TA-level excess across all 4 models. The b=2 dip is a clean local minimum.
  - **Universality across 4 distinct architecture families is the strongest possible empirical claim for a structural finding.** Not a model artifact; not a GQA-vs-MHA artifact; not training-recipe-specific. The bound we proved doesn't preclude this (it's an upper bound on quantization-induced distortion; nothing says quantization can't HELP). Possible framing for the paper: "the bound characterizes worst-case distortion; we observe an empirical regime where quantization improves the merge — consistent with the bound but not predicted by it."
  - Mechanism candidates in §5.5 (regularization-via-coarsening, stochastic resonance, implicit coarse-projection). **Headline alongside TIES > TA and the d_eff-saturation gap (F9).**

- **F9 — Hard d_eff is SATURATED (= Tr = 64) UNIVERSALLY across 4 architectures (2026-05-19 evening, extended from the 2-model 2026-05-18 result).** For Llama-3.1-8B (128 layers, MHA), Qwen-2.5-7B (112 layers, GQA), Mistral-7B-v0.3 (128 layers), and Yi-1.5-9B (192 layers, Llama-arch): **every single layer of every model has d_eff = 64 = Tr = 4·16**. The 4 task subspaces V_t span the full Tr-dimensional space across architectures — they are linearly independent across tasks regardless of architecture family. **The bound's predicted hard floor B²(1 − d_eff/(Tr)) = 0 in this regime, for every model.** But observed worst_task_excess varies from 0.10 (Yi) to 0.22 (Llama) at b=32. **The gap quantifies how far current merging methods are from the information-theoretic optimal — there's room for 2–5× improvement.** Reframes the paper from "we proved a bound, here are some numbers" to "we proved a bound that real LoRAs lie WELL above the predicted floor across 4 architectures; the gap is a call to action for better mergers."

- **F10 — Soft d_eff (T1.A2, participation ratio of stacked-V singular values) is UNIFORM across 4 architectures and does NOT explain F2 (2026-05-19 evening).** Computed as $\mathrm{soft\_}d_{\mathrm{eff}} = (\sum_i \sigma_i^2)^2 / \sum_i \sigma_i^4$ where σ are singular values of $M = [V_1 | \ldots | V_T] \in \mathbb{R}^{\text{in\_dim} \times Tr}$. Mean values per architecture: **Llama 16.48, Qwen 16.45, Mistral 16.37, Yi 16.41** — range <0.7%. Soft $d_{\mathrm{eff}}/(Tr) \approx 0.256$ for every architecture. But observed TA worst_excess varies by 2.2× across the same models. **Correlation: Pearson r = +0.53 (p = 0.47), Spearman r = +0.40 (p = 0.60) — not significant.** So both hard AND soft d_eff are uniform across these 4 architectures, but merging difficulty isn't. **F2 (merging-difficulty ordering Llama > Mistral > Qwen > Yi) is therefore NOT a subspace-overlap phenomenon.** This falsifies log.md §5.6 candidate (b) ("hard d_eff is too coarse; soft d_eff should correlate with observed excess"). The remaining candidate mechanisms for F2 are:
  - (c) **Unmodeled $H_t$ curvature.** The theorem uses $H_t$-norm distortion; the $H_t$ (Fisher / GGN) eigenstructure differs across base models. Yi's local loss curvature near $\theta_0$ might be flatter → same task-vector perturbation hurts less. Would require explicit per-model Fisher estimation to test.
  - **Pretraining + instruction-tuning recipe.** Same architecture (Yi is Llama-arch) but very different merge behavior → recipe matters. This isn't directly modelable inside the theorem but is the cleanest empirical story.
  - **Per-layer $B^2$ (LoRA-delta norm) differences.** Empirical norms ‖B@A‖_F may scale differently across base models; the floor formula's $B^2$ pre-factor would amplify or dampen accordingly. Worth tabulating per-model per-layer $B^2$ stats.
  Implication for the paper: lead F9 + F10 together. F9 is the "real LoRAs sit above the bound's floor universally" headline; F10 honestly closes the door on the "soft d_eff explains it" hypothesis and points to curvature/recipe as the open mechanism. Cleaner story than waving at d_eff alone. See `paper_artifacts/figures/deff_vs_floor.png` for the 4-panel figure (per-layer hard hist + soft hist + soft-vs-observed scatter + per-model soft d_eff bars).

- **F8 — Data audit (2026-05-18): train/eval split bug found and fixed.** All 4 LoRA training datasets are real, public, and authoritative (openai/gsm8k, yahma/alpaca-cleaned, ise-uiuc/Magicoder-OSS-Instruct-75K, wmt/wmt19 de-en). However, `data_loaders.py` used two *independent* shuffle seeds (seed vs seed+1) for train and eval when both came from the same split — silently allowed ~13% overlap on alpaca and ~7% on magicoder. **Fixed 2026-05-18:** now uses ONE shuffle, then disjoint slices `[0:n_train]` and `[n_train:n_train+n_eval]`. Verified post-fix: 0/200 overlap. Impact: NLL_τ values for alpaca/magicoder cells in the existing matrix are biased ~5–10% downward. Worst-task excess values largely unaffected (gsm8k is always the bottleneck; gsm8k uses separate train/test splits). TIES > TA, Qwen-easier-than-Llama, b=2 dip — all robust to the bias. Action: rerun the 20-cell n=1k matrix after the current one finishes; new output dir `eval_matrix_n1k_v2/`.

### §3.5 Phase B validation criteria — 4-model outcomes, bootstrap-checked (per `notes/phase3_design.md` §6)

- **(1) Floor exists at b=32: PASS on all 4 models, CIs exclude 0.** Llama 0.218 [0.213, 0.224], Mistral 0.146 [0.141, 0.151], Qwen 0.110 [0.106, 0.114], **Yi-1.5-9B 0.099 [0.095, 0.102] (lowest floor)**. Robust.
- **(2) TVQ slope ≈ -2: SKIPPED on all 4.** Only 1–2 points sit above the floor in every model; rate-decay regime not visible at practical bit budgets. Confirmed universal — not a 2-model artifact.
- **(3) KnOTS > TA per-task: DOES NOT HOLD (corrected 2026-05-20).** Originally scored "PASS" by counting per-task wins, but paired bootstrap shows KnOTS ≈ TA (only 5/16 cells significant, split both directions, magnitudes ≤ 0.0019 nats — see F4). **However, this is the theory-predicted outcome at d_eff = Tr** (the KnOTS advantage is predicted to vanish as d_eff → Tr; we measure d_eff = Tr everywhere). And the *spirit* of C3 ("structure-respecting > naive") IS validated, robustly, by **TIES** (F1: 14/16 cells significant, large gsm8k wins) — just not by KnOTS's subspace-projection variant.

**Decision-gate outcome:** ICLR 2027 still viable. C1 holds robustly (floors significant on all 4 models). C3-as-literally-stated (KnOTS) fails, but (a) that failure is theory-consistent at d_eff=Tr, and (b) the structure-respecting result it was meant to capture is carried by TIES with large, bootstrap-significant margins. **Honest reporting:** we do NOT claim KnOTS beats TA; we claim TIES beats TA on the bottleneck task, and we explain the KnOTS null via the d_eff=Tr regime. The 4-architecture comparison + bootstrap CIs make this much harder to dismiss as model-specific or noise.

---

## §4 Engineering / cluster lessons

Every gotcha I burned time on so the next iteration doesn't.

### §4.1 Library / API drift (Unsloth's transitive deps moved everything forward)

1. **TRL 0.24 removed `DataCollatorForCompletionOnlyLM`.** Use `SFTConfig(completion_only_loss=True)` + dataset format `{"prompt": str, "completion": str}`.
2. **TRL's `dataset_num_proc` defaults to `os.cpu_count()`** (= 64+ on jiit-gpu01). 64 workers re-downloading the tokenizer mirror over a 1.4 MB/s pipe is a network storm. Set `dataset_num_proc=2`.
3. **Unsloth monkey-patches `LlamaForCausalLM.forward` at import time.** Patched forward calls `apply_qkv` (set by `FastLanguageModel.get_peft_model`). After training, reloading via `AutoModelForCausalLM.from_pretrained` crashes with `AttributeError: 'LlamaAttention' object has no attribute 'apply_qkv'`. Fix: reload via `FastLanguageModel.from_pretrained(adapter_dir)` so `apply_qkv` is re-set up.
4. **Unsloth loaded from a local path doesn't set the implicit tokenizer.** Must pass `processing_class=tokenizer` to SFTTrainer explicitly. (HF-repo loads auto-set it.)
5. **`huggingface-cli` is deprecated in `huggingface_hub` 1.x.** It silent-exits with no-op. Use `hf` (new CLI) or Python `snapshot_download`.
6. **`HF_HUB_ENABLE_HF_TRANSFER` deprecated.** Replaced by `HF_XET_HIGH_PERFORMANCE`. The latter is what's broken on this cluster — set neither.

### §4.2 Cluster networking

1. **HF Xet protocol self-throttles to ~30 KB/s on this cluster** (visible in `cache/hf_home/xet/logs/*.log`: "Concurrency control for download: Decreased concurrency from 4 to 4; reason: success ratio below threshold"). Bypass with plain `wget --continue` against `huggingface.co/{repo}/resolve/main/{file}`. Gets the full ~1.4 MB/s.
2. **Login node and compute nodes have the SAME external bandwidth** (~1.4 MB/s). My initial "compute node is 100× faster" reading was a `du` race artifact, not real throughput.
3. **PBS gpu queue caps users at 3 concurrent jobs.** Affects throughput of any matrix-style work. Plan around it.
4. **PBS doesn't track GPUs as a schedulable resource** (`naccelerators=0` in pbsnodes output). Multiple users can land on the same physical GPU. We pin via `gpu_picker.py` at job startup; if no GPU has `--min-free-gb` available, exit 87 (cleanly handled by wrappers).
5. **All large artifacts must live under `/home/sanjay.g/projects/rdmerge/`** (BeeGFS, 366 TB free) NOT under `~/.cache/` (the user `/home/sanjay.g/` is on a different mount that's 86% full).

### §4.3 Datasets

1. **`facebook/flores`, `Muennighoff/flores200`, `openlanguagedata/flores_plus` all broken in `datasets` 4.x.** First two use legacy dataset scripts (dropped support); third is license-gated. We use `wmt/wmt19` config `de-en` with `streaming=True` — WMT19 de-en train has 38M pairs; streaming avoids the full download.
2. **Magicoder cache race:** when a preload script and a training process both write to `cache/hf_home/.../incomplete` blobs simultaneously, you get `FileNotFoundError` on rename. Wait for preload before launching training.
3. **Two-shuffle-seeds train/eval overlap bug (found+fixed 2026-05-18).** When `split_eval == split_train` (alpaca, magicoder), the original `data_loaders.py` called `shuffle(seed=s).select(0..n_train)` for train and `shuffle(seed=s+1).select(n_train..)` for eval — two *independent* permutations meant ~13% (alpaca) / ~7% (magicoder) of "held-out" examples ended up in training. Fixed by using ONE shuffle and slicing disjointly. NEVER use independent shuffles when slicing a single split — always one shuffle, then disjoint indices.

**Dataset authority audit (2026-05-18 verified):**

| Dataset | Source | Size | Role | Status |
|---|---|---|---|---|
| openai/gsm8k (main) | OpenAI, Cobbe+'21 (arXiv:2110.14168) | 7473 train + 1319 test | math LoRA | ✓ canonical |
| yahma/alpaca-cleaned | yahma cleanup of Stanford Alpaca | 51760 (train only) | instruction LoRA | ✓ widely used; ChatGPT-generated, quality-filtered |
| ise-uiuc/Magicoder-OSS-Instruct-75K | Wei+'23 (ISE-UIUC) | 75197 (train only) | code LoRA | ✓ standard code-tune dataset |
| wmt/wmt19 (de-en) | WMT 2019 official | ~38M train + 2998 val | translation LoRA | ✓ gold standard MT |

**Synthetic data clearly labeled:**

- `code/synthetic/day1..day16` — Phase 0/1/2.5 *theory validation*. Synthetic τ_t with controlled spectra. NOT used in Phase 3 LoRA training.
- `code/phase3/eval/stiefel_control.py` (T1.B, written 2026-05-18) — *adversarial worst-case* synthetic LoRAs for the bound-tightness control panel. Pure synthetic by design.
- `code/phase3/merging/tests/_fake_model.py` — synthetic FakePeftModel mocks used only for the 17 CPU unit tests. Not on the science path.

**No synthetic data was used to train any of the 8 (model, task) LoRA adapters.** All real, public, third-party data.

### §4.4 PEFT / merging code

1. **PEFT's `add_weighted_adapter` is slow on big LoRAs** — ~6.8 sec per layer for `combination_type="linear"` because of full-SVD on 4096-dim. We bypass it by computing merges ourselves and overwriting LoRA factors via `PeftModelView`.
2. **`torch.svd_lowrank` is randomized** (non-deterministic across calls). Gate to fire only when `min(out_dim, in_dim) > 256`; tests use small dims and get the deterministic full SVD.
3. **Llama-3.1 tokenizer has no padding token by default.** Unsloth auto-sets `pad_token = <|finetune_right_pad_id|>`.
4. **PEFT's `density` parameter** is fraction RETAINED (1.0 = no pruning, 0.0 = all pruned). Default we used: 0.2 for TIES/DARE (matches TIES paper).

### §4.5 Architecture decisions

- **Why one merging code path (FakePeftModel) for both tests and production:** keeps test/prod identical and avoids PEFT's slow `add_weighted_adapter`. Implementation lives in `code/phase3/merging/{task_arithmetic,ties,dare,knots,tvq}.py`; the real PEFT model is adapted via `PeftModelView` in `peft_model_view.py`.
- **Why KnOTS from paper not from repo:** the algorithm in arXiv:2410.19735 §3 is ~80 lines and explicit. Skipping the port avoided ViT-vs-LM layer naming issues.
- **Why no nf4:** we have A100-80GB. Full bf16 keeps the §5 quantized-base caveat from `handoff.md` from applying. The rate-budget in the bound is over the LoRA deltas, not the base; keeping the base un-quantized makes the metric clean.

---

## §5 Speculations + ideas worth exploring

### §5.1 Why is Qwen ~2× easier to merge than Llama?

Three speculative mechanisms:
- (a) **GQA reduces inter-task interference.** Qwen uses grouped-query attention with 8 KV heads on a 4096-dim model (so KV proj is rank-reduced to 1024). LoRA on Q/K/V/O of GQA naturally lives in a smaller-rank space — the effective `d_eff/(Tr)` may be inherently lower.
- (b) **Pretraining objective / scale differences.** Qwen-2.5 was trained on substantially more bilingual + math data than Llama-3.1. Maybe the "task vector" for math is closer to the pretrained subspace, so merging doesn't perturb gsm8k as much.
- (c) **Normalization choices.** Both use RMSNorm but with different epsilons / per-layer details. The loss landscape's local curvature might be flatter for Qwen, so the same task-vector perturbation hurts less.

To test: compute empirical `d_eff = rank(Σ_t P_{V_t})` per layer per model. If the bound's floor formula $(1 - d_{\mathrm{eff}}/(Tr))$ explains the 2× gap, that's a strong paper figure.

### §5.2 Why does translation have negative excess?

Three speculative mechanisms:
- (a) **Translation LoRA under-trained.** 7500 wmt19 examples × 2 epochs ≈ 15k example-runs ≈ 469 steps. That's light for translation; the LoRA likely hasn't fully captured en→de morphology. Merging with the other LoRAs adds capacity (more parameters trained) that helps.
- (b) **Cross-task transfer is real.** Math (numeric formatting), code (precise vocabulary), and instruction (response structure) all contribute to translation quality. The merged model has 4 LoRAs' worth of grounded supervision; the translation-only LoRA has just 1.
- (c) **Streaming first-7500 bias.** WMT19 train isn't shuffled; the first 7500 pairs may be lower-quality / specific-domain. A randomly-sampled 7500 might give a better baseline τ.

To test: retrain translation at 15k × 3 epochs with shuffled sampling. If excess goes positive → it was under-training (and we should retrain for paper numbers). If excess stays negative → real cross-task benefit (paper finding worth ~½ page).

### §5.3 Why does DARE ≈ TA exactly?

Three possibilities:
- (a) **Density=0.2 is too aggressive** — most LoRA parameters get dropped, the rescale doesn't recover them, the residual ≈ noise around TA's average.
- (b) **Implementation bug** — maybe the rescale-by-1/density isn't applied correctly.
- (c) **DARE works on full task vectors, not LoRA factors.** The LoRA factors are already low-rank; random pruning at the factor level might not match what DARE was designed for. The paper merges full deltas; we merge B@A — minor but might matter.

To test: density-sweep at {0.1, 0.5, 0.7, 0.9, 1.0}. At 1.0 must equal TA exactly (sanity check on impl). At 0.5/0.7/0.9 should differ visibly from TA.

### §5.4 Why doesn't the rate-decay slope show up?

The theory predicts $\mathcal{D}(R) = \text{floor} + C \cdot 2^{-2R/d_{\mathrm{eff}}}$. With $d_{\mathrm{eff}}$ approximately $Tr = 4 \times 16 = 64$ per layer, the decay constant is $2^{-2b/64}$ — for b=1 that's $2^{-1/32} \approx 0.978$, b=8 is $2^{-1/4} \approx 0.84$, b=32 is essentially 1. So even at b=1 the rate-decay term is "0.978 × something" — meaning we'd see a 2% reduction in the leading factor between b=1 and b=∞.

That 2% × C (the leading constant) is what we'd need to detect against a measurement floor of ~0.01 nats/token (n=200 sample noise). Plausibly within noise, especially because $C$ for our quantizer/distribution might be small.

**Implication for the paper:** the slope claim is essentially un-validatable empirically at practical bit budgets for LoRA-rank-16 setups. We should reframe: the bound predicts the *floor shape* (which holds — see F2) and provides a *constant-time bound for any quantizer* (also holds at b=32). The slope is theoretically real but operationally invisible.

### §5.6 The d_eff = Tr saturation (CONFIRMED 2026-05-18 night)

**Setup:** for every (model, layer) we compute V_t = top-r right-singular subspace of
`scaling · B_t @ A_t` per task t, then d_eff = rank of M = [V_1 | ... | V_T] (in_dim × T·r).

**Finding:** d_eff = Tr = 64 in 128/128 Llama layers and 112/112 Qwen layers.
The 4 task subspaces span Tr-dim space; bound's floor formula gives 0.

**Implication for the paper.** Three candidate interpretations, ordered by how
much they would change the narrative:

(a) **The bound is correct AND tight for its setting (worst-case Stiefel-random).
Real LoRAs aren't worst-case.** Bound is loose for real-LoRA-like distributions
because real V_t are not Stiefel-random — they have structure (correlations, hub
directions, etc.) that we're not modeling. The empirical excess of ~0.2 represents
the gap between "what current merging algorithms achieve" and "what the bound
says is possible (which is 0 here)." **Recommended paper framing.**

(b) **d_eff is too coarse a summary of "how much the subspaces overlap."** The
hard-rank measure can't distinguish "subspaces span Tr dims but with high principal
angles" from "subspaces span Tr dims orthogonally." A *soft* d_eff (effective rank
via Roy-Vetterli participation ratio of M's singular values) would give a
non-trivial intermediate value, and the formula might match the observed excess
much better. **Worth implementing as a side analysis.**

(c) **There's an unmodeled term in the bound.** Maybe the H_t curvature (Fisher)
in real LLMs introduces additional unrecovered distortion that the bound doesn't
capture. Less likely given the bound was derived for general H_t ≽ 0, but worth
considering if (a) and (b) don't fully explain.

**Paper-narrative recommendation.** Lead with (a): the bound predicts the *worst-
case floor*; real LoRAs sit ABOVE the bound by a gap of ~0.10–0.22 nats/token; this
gap is *what better merging algorithms can close*. The KnOTS/TIES per-task wins
support this — these methods chip away at the gap, and the bound says there's
more room.

Soft-d_eff variant: implement (b) as a "Figure 5" diagnostic — show the
singular-value spectrum of M and compute participation-ratio d_eff. If the soft
metric correlates with observed excess across layers, that's a tighter empirical
prediction.

### §5.5 The TVQ b=2 local minimum (RESOLVED — real, not noise)

**Updated 2026-05-18 after n=1k rerun.** The Llama TVQ b=2 dip survives at n=1000:
worst_excess = 0.104 (n=1k) vs 0.107 (n=200), against b=4 worst_excess = 0.217.
The dip is not measurement noise — it's a **real local minimum in the rate-distortion
curve, ~2× better than coarser or finer quantization**.

Llama TVQ rate sweep (n=1k confirmed):

| b  | worst_excess | avg_excess |
|----|-------------:|-----------:|
| 1  | 0.232 | 0.140 |
| **2** | **0.104** ← real local minimum | **0.019** |
| 4  | 0.217 | 0.049 |
| 8  | 0.218 | 0.049 |
| 16 | 0.217 | 0.049 |

Hypothesis (a) "sample noise" is **ruled out**. The remaining candidate mechanisms:

- (a) **Quantization-as-regularization at b=2.** With only 4 distinct values per
  layer (b=2 → 2² = 4 quantization levels), the injected coarsening might
  destroy the destructive-interference patterns between task vectors that
  finer quantization (b ≥ 4) preserves with too much fidelity. b=1 is too
  coarse (binarization kills the LoRA signal entirely); b ≥ 4 are too fine
  to break the interference structure. The "sweet spot" at b=2 happens to
  align with the granularity of the merge problem.

- (b) **Stochastic-resonance-like effect.** The quantization noise at b=2
  happens to push the merged delta into a region of the loss landscape that
  is genuinely better for the worst task (gsm8k). Pure coincidence at the
  weight scale, but structural at the geometry scale.

- (c) **Implicit Hadamard alignment.** Our TVQ implementation uses per-tensor
  min-max scaling; at b=2 with 4 levels, the granularity is so coarse that
  the LoRA delta gets projected into one of 4 "principal directions" per
  layer. This might be playing a role similar to the achievability scheme's
  Hadamard rotation — coarsifying enough to wash out incoherent interference
  while still preserving the dominant rank-r direction.

**This is paper gold.** A "less is more" rate-distortion finding runs strongly
against the naive "more bits = better merge" intuition. To strengthen the
claim, before submission:
1. Confirm Qwen TVQ b=2 also dips (pending; if yes, model-agnostic; if no, model-specific). [Currently in flight as of 2026-05-18 ~12:30.]
2. Run TVQ at intermediate rates b ∈ {1.5, 2.5, 3} (via Lloyd-Max with non-power-of-2 levels) to map out the dip's shape.
3. Per-task excess breakdown — is the b=2 win specifically on gsm8k (the bottleneck task), or across all 4?
4. Repeat on a different task set to rule out cross-task-overlap artifacts.

If the dip generalizes across models AND task sets, **this is a candidate
headline contribution** alongside the TIES > TA result and the Qwen-easier-than-Llama
finding. The bound we proved doesn't EXPLAIN this dip — but it's consistent
with it. We could frame this as: "the bound says quantization-induced distortion
is at most 2^(-2R/d_eff); it doesn't preclude quantization being *beneficial*
when it destroys destructive interference between task vectors. We observe one
such regime empirically at b=2."

### §5.6 What about Fisher merging?

Skipped per `notes/phase3_design.md` §4 ("adds a Hessian-estimation confound we don't want to debug"). But for paper credibility we should at least mention it. The bound holds whether or not Fisher is used in the merge — Fisher is just one way to weight task vectors. We could add Fisher as a v2 baseline if reviewers push.

### §5.7 What about 1bit-Merging?

Skipped per the design (concurrent / unverified). If 1bit-Merging proves viable in published form before submission, we should include it — it directly exercises the sub-bit regime where our slope SHOULD become visible (assuming a sub-bit scheme exists).

### §5.8 Diversity of merging methods we DIDN'T cover

DO-Merging, Core Space, TVQ-residual (paper variant), TSPA, ARM. These are mostly subspace-aware variants. If the reviewer asks "why these 5 and not others," answer: standard baselines (TA/TIES/DARE) + closest related-work-in-spirit (KnOTS) + only published low-bit method (TVQ). Others are post-publication or concurrent.

---

## §6 Ideas for the paper narrative

### §6.1 What to lead with

Updated 2026-05-18 after the d_eff finding (F9):

> **We prove a Shannon-style lower bound on bits-per-parameter for LoRA merging. The bound's floor formula is sharp for the worst-case Stiefel-random distribution of task vectors. Empirically, real LoRA task vectors lie ABOVE the bound by 0.10–0.22 nats/token of worst-task NLL — quantifying how far state-of-the-art merging methods are from the information-theoretic optimum, and motivating a class of better algorithms. We further find a "less-is-more" rate-distortion regime: 2-bit task-vector quantization (TVQ) outperforms higher-bit quantization by 2–5× on worst-task excess, consistent across architectures.**

This is the headline. We then show:
- TIES beats TA cleanly on both models.
- Qwen merges more cleanly than Llama (~2×) — possibly because of GQA / training-data overlap.
- KnOTS gives small per-task wins where its SVD alignment helps.
- The bound's *floor structure* matches.
- **The TVQ rate-distortion curve has a non-monotonic minimum at b=2 (~2× better than b=8 on Llama, ~5× better on Qwen) — a "less is more" finding consistent with quantization-as-regularization.** Confirmed model-agnostic 2026-05-18: both Llama (½ of adjacent) and Qwen (⅕ of adjacent, avg_excess NEGATIVE!) show the dip. Headline contribution on its own, alongside TIES > TA.

### §6.2 What to honestly report

Per Rule N2 (re-derive before externalizing):
- **TVQ rate-decay slope isn't empirically visible** at practical bit budgets (b ≥ 1). The merging-geometry error dominates. Don't claim slope = -2; instead: "the bound predicts the floor; the rate-decay constant becomes operationally relevant only below sub-bit precision."
- **Translation has negative excess** — merging actually helps. This is either under-training of the translation LoRA or real cross-task benefit. Plan: rerun translation at 15k × 3ep before submission; report whichever explanation survives.
- **DARE ≈ TA exactly** at density=0.2. Need ablation before submission. If real, report as a finding ("DARE's noise averages out at this density/rank"). If a bug, fix.
- **n=200 eval is noisy.** Rerunning at n=1000 (in flight as of 2026-05-18 night) will give defensible CIs.

### §6.3 The headline figure

Two-panel rate-distortion plot (one per model):
- x-axis: bits-per-parameter (log scale, 1 to 32)
- y-axis: worst-task excess NLL (linear)
- TVQ as a curve through the b values
- Non-TVQ methods as horizontal lines at "b=32 equivalent"
- Shaded band: bootstrap-CI from 1000-example rerun

Already drafted at `code/phase3/figures/headline_rd.png` (n=200, no CI). Will regenerate with n=1k.

### §6.4 Suggested section structure (for `paper/sections/experiments.tex` §6.2)

1. **Setup** (~½ page): 2 base models, 4 tasks, LoRA hyperparams, eval split sizes.
2. **The 8 LoRAs trained cleanly** (Table 1: NLL_τ matrix). One line stating the baseline performance.
3. **The 20-cell merge matrix** (Table 2: worst_task_excess per method × model + TVQ rate sweep). Two lines describing the headline finding (TIES wins, Qwen easier).
4. **Floor validation** (Figure: floor vs $d_{\mathrm{eff}}/(Tr)$ if we compute it; or just per-task excess heatmap). One paragraph: "the bound's floor structure matches the observed worst-task error."
5. **Rate axis discussion** (Figure: TVQ rate sweep). One paragraph honestly: "at b ≥ 1 the merging error dominates the quantization error; the bound's rate-decay term is operationally invisible in this regime."
6. **Method ordering discussion**: one paragraph each on TIES, KnOTS, DARE.
7. **Limitations** (½ page): n=200 noise (or n=1k if we've done the rerun); single-seed; specific tasks; non-finetuned-with-RLHF.

### §6.5 Things to NOT claim

- Don't claim "slope ≈ -2" — we can't show it.
- Don't claim "real merges saturate the bound" — they sit above the bound (this is correct! The bound is a *lower bound on the worst case under P*; real LoRAs are above the worst-case envelope).
- Don't claim "TVQ is the best low-bit method" — we only tested uniform scalar quantization, not the residual variant from the paper.

### §6.6 Things to consider for v2

- Bigger eval (n=2k or full test set).
- 3-seed reruns of the headline cells (deferred — currently single-seed).
- LoRA rank ablation (r ∈ {4, 16, 64}) to test floor scaling with r.
- T ∈ {2, 4, 8} sweep to test floor scaling with T.

### §6.7 Tier 1 additions in the pipeline (committed 2026-05-18 afternoon)

Three additions to strengthen the paper before submission. All committed; one
already runs in background.

**T1.A — empirical d_eff + floor-formula validation** (script: `code/phase3/eval/deff_analysis.py`).
For each (model, task) LoRA, extract V_t = top-r right-singular subspace of B@A
per layer, compute d_eff = rank(Σ_t P_{V_t}) per layer, predict floor =
B²(1 - d_eff/(Tr)), compare to observed worst_excess. **Status:** script
written + syntax-clean, running on login-node CPU as of 2026-05-18 afternoon.
Reuses existing 8 LoRA adapters; no new compute.

**T1.B — synthetic Stiefel-random control panel** (script: `code/phase3/eval/stiefel_control.py`).
Generate fake LoRAs with controlled subspace overlap (overlap parameter α ∈
{0, 0.1, ..., 1.0}), run all 5 merging methods, plot predicted vs observed
floor and bound-tightness ratio. Standard "RD paper control panel."
**Status:** script written + syntax-clean, ready to run (CPU-only, ~5 min
wall-clock).

**T1.C/T1.D — 3 more model families (Mistral-7B, Yi-9B, gemma-3-12b)**.
Train 4 task LoRAs per new model (12 LoRAs total) + extend the eval matrix
by 33 cells (15 non-TVQ + 18 TVQ). **Status:** downloads in flight on workq
(job 39627, doesn't compete with gpu queue). After downloads complete,
12 new LoRAs to train then 33 new eval cells to run. Estimated total
compute: ~25 hr wall-clock at 3-concurrent gpu cap. Done overnight.

**T1.E — regenerate Phase B with full Tier 1 data**. Update headline figures
once T1.C/D land: 5 model families on the rate-distortion plot, error bars
where available, predicted-vs-observed-floor scatter from T1.A,
bound-tightness panel from T1.B.

Why these three: T1.A directly tests the theorem's quantitative prediction;
T1.B is standard for RD bound papers (shows bound is tight on adversarial
data, loose on real); T1.C/D converts "Qwen vs Llama" (n=2) into a
5-architecture comparison that's much harder for reviewers to dismiss as
model-specific.

---

## §7 Open questions (synced from `notes/open_questions.md` Phase 3 section)

1. **O17.1** — Is the TVQ b=2 Llama anomaly real? Rerun at n=1000 (in flight).
2. **O17.2** — Why is Qwen ~2× easier to merge than Llama? Compute empirical $d_{\mathrm{eff}}$ per model.
3. **O17.3** — KnOTS ties TA on worst_excess but beats it 5/8 per-task. Per-task error bars needed.
4. **O17.4** — DARE = TA exactly at density=0.2. Density-sweep ablation needed.
5. **O17.5** — Translation has negative excess. Retrain translation at 15k × 3ep to disambiguate under-training vs cross-task benefit.
6. **O17.6** — Rate-decay slope not testable at b ≥ 1. Paper-narrative question; surface to Garg.

---

## §8 Future work (beyond this paper)

- **Sub-bit quantization schemes** to make the rate-decay term visible — would substantiate the slope claim empirically.
- **Theoretical exact RD for general T** (the open problem from Phase 2.5).
- **Per-layer $d_{\mathrm{eff}}$ analysis** — can we predict at PER-LAYER granularity which layers will saturate the bound? Could inform mixed-precision quantization schemes.
- **Bound for LoRA over base + base-quantization noise** — i.e., dropping the "base is fixed and bf16" assumption. Practically relevant for deployment.
- **Bound extension to RLHF-tuned LoRAs** — the Fisher curvature changes under preference tuning.

---

## §9 Key references (in `paper/references.bib`)

- **Zandieh et al. 2025 TurboQuant** (arXiv:2504.19874) — Thm 3 is our template. Cold-emailed authors 2026-04-26; no reply as of 2026-05-02.
- **Stoica et al. 2025 KnOTS** (ICLR, arXiv:2410.19735) — closest related work in spirit; explicitly thinks about subspaces.
- **Yadav et al. 2023 TIES** (NeurIPS) — sign-resolution + sparsification.
- **Yu et al. 2024 DARE** (ICML) — drop-and-rescale.
- **Ilharco et al. 2023 Task Arithmetic** (ICLR) — the naive baseline.
- **Kim et al. 2025 TVQ** (ICCV, arXiv:2503.06921) — task vector quantization (we implement uniform-scalar version; residual variant is theirs).
- **Ortiz-Jimenez et al. 2023** (NeurIPS Oral) — weight disentanglement; H_t = Fisher structure.
- **Matena & Raffel 2022 Fisher merging** (NeurIPS) — interesting baseline, skipped per design.

---

## §10 Submission mechanics

**Decision (2026-05-18):** **NO arXiv preprint.** Submit directly to ICLR 2027.
The work stays confidential through review. Skipping the preprint also removes
the endorsement-wait blocker and the arXiv BibTeX/abstract polish from the
critical path. Files `arxiv_checklist.md`, `ARXIV_TODO.md`, and the
`preprint_repo/` staging folder are obsolete for this submission; leave them
on disk as historical reference (do not delete in case a v2 preprint is ever
needed) but do not work on them.

- **Target venue:** ICLR 2027 (deadline ~2026-09-24).
- **Backups (in order):** AISTATS 2027 (~2026-10-08), ICML 2027 (~2027-01-30), TMLR (rolling).
- **Paper contact email:** `pathaksankalp04@gmail.com`. (Never the `kittuwastaken@gmail.com` Claude/system email.)
- **ICLR-specific:** 9 pages main + appendix (no length limit on appendix), double-blind review, conference template (auto-format via the `iclr2027_conference.sty` when released).

**Three cardinal rules (still apply, just without the arXiv flow):**
- **Re-derive (not re-read) before externalizing** (Rule N2): any send to Garg, OpenReview submission, or email to researchers requires a full re-derivation pass.
- **Pair proofs with same-day numerics** (Rule N1): no theorem in isolation.
- **PDF-only externalization** (Rule N3): drafts shared with collaborators are PDF only; `.tex` source stays internal.

**Decision gate (end of July 2026, per `target.md`):** if Phase 3 is "informative" (criterion 1 + at least one of 2 or 3 hold — currently satisfied per §3.5 above), commit to ICLR. If it later becomes uninformative, pivot to TMLR (rolling). Decision-gate outcome as of 2026-05-18: **ICLR target viable.**

---

## §11 Where to look for what

- **Latest data**: `results/phase3/eval_matrix/` (n=200, done) and `results/phase3/eval_matrix_n1k/` (n=1000, in flight as of 2026-05-18 night).
- **Phase B summary**: `results/phase3/phase_b_summary.json`.
- **Adapters**: `artifacts/lora/{model}/{task}/v1/`.
- **Models**: `models/{Llama-3.1-8B-Instruct,Qwen2.5-7B-Instruct,Llama-3.2-1B-Instruct}/`.
- **All code**: `code/phase3/`.
- **Plan**: `~/.claude/plans/ok-now-you-know-abundant-tulip.md`.
- **Daily log**: `notes/daily_log.md`.
- **Open questions**: `notes/open_questions.md`.
- **Per-phase findings**: `notes/phase3_findings.md`.
- **THIS file**: paper-writing reference. Updated as new findings land.

---

## §12 Things I'd want to remember if I were the future me writing the paper

- The story is "the bound predicts the *shape*, not the exact numbers." Don't oversell.
- The Qwen vs Llama gap is the most interesting empirical finding. Lean into it.
- TIES dominates. Period. Don't bury it.
- Be honest about the slope. The bound is still correct; the slope just isn't operationally relevant at practical bit budgets.
- Translation's negative excess is a real puzzle. Either resolve it or report it honestly with both candidate explanations.
- The Phase B criteria check matters as a research-honesty signal. C2 SKIPPED is fine to report; we tried, the data couldn't bear it, we said so.
- The smoke→pilot→matrix→figure progression is well-defined; reviewers love it as a reproducibility story.
- If reviewers push back: every numerical result has a JSON with `meta.git_sha`, `meta.config_hash`, `meta.pbs_jobid`. Everything is reproducible.

---

## §13 2026-05-19 operational status (Day 19)

### v2 eval matrix (rerun with fixed `data_loaders.py`)

- Output dir: `results/phase3/eval_matrix_n1k_v2/`
- Progress at last check: **16 / 20 cells done.** Currently running: Qwen
  TVQ b4, b8, b16/b32 (in tail of submission queue).
- Walltime per cell: ~1h45m for non-TVQ (KnOTS is the slow one but
  cleared the 3 hr cap; the n=1k matrix had killed it at 2h01m). TVQ
  cells faster.
- ETA to `ALL_EVAL_CELLS_DONE`: ~1–2 hr from now.
- v2 launcher: `code/phase3/scripts/launch_eval_matrix.py` (PID
  2979771); blocks at the 3-concurrent cap, exits on full drain.

### T1.B Stiefel control panel — RAN AND PARKED

`code/phase3/eval/stiefel_control.py` executed 2026-05-19 ~03:17 IST.
The mixing construction
`V_t = sqrt(1-α) V_indep + sqrt(α) V_shared` keeps the stacked V's
generically linearly independent for α < 1, so `d_eff = Tr` across
the sweep except at α = 1.0. Predicted floor is 0 throughout;
"observed / predicted" ratio is ~10¹². Outputs at
`results/phase3/stiefel_control.json` +
`code/phase3/figures/stiefel_control.png` (**NOT** promoted to
`paper_artifacts/`).

The negative result is itself informative: hard `d_eff` saturates
generically. Same phenomenon F9 already showed on real LoRAs. Two
recovery paths:

- **(i) Redesign T1.B** with explicit partial-shared-basis construction
  (share r' < r exact basis directions per task → `d_eff = r' + T(r − r')`),
  sweepable.
- **(ii) Soft d_eff (T1.A2)** via participation ratio of stacked-V
  singular values — continuous metric; subsumes (i); aligns with §5.6
  candidate (b).

Recommend **(ii)** after v2 lands, contingent on Sankalp sign-off.

### T1.C robustness training — IN QUEUE (8 LoRAs)

Original handoff §4 plan was 12 LoRAs (Mistral + Yi + gemma-3) and the
session-time consideration added Qwen-14B for 16. 2026-05-19 user
decision dropped 12B and 14B:

- **gemma-3-12b-it** dropped — `Gemma3ForConditionalGeneration` is the
  multimodal head, not pure causal LM; Unsloth's `FastLanguageModel`
  path is risky.
- **Qwen-2.5-14B-Instruct** dropped — realistic peak VRAM ~41 GB; on a
  shared-GPU cluster typically showing 14–46 GB free per device,
  slot-finding is unreliable. Revisit in a quieter window.

**Final T1.C scope:** 2 models × 4 tasks = **8 LoRAs.**

| Model | min_free_gb |
|---|---:|
| mistral_7b (Mistral-7B-Instruct-v0.3) | 30 |
| yi15_9b (Yi-1.5-9B-Chat) | 35 |

Configs at `code/phase3/configs/lora/{mistral_7b,yi15_9b}_*.yaml`. Launcher
`code/phase3/scripts/submit_t1c_lora.sh` is polling on a combined
`rdm_(eval|train)` 3-cap counter — yields cleanly to v2. First T1.C
submission fires when v2 frees a slot. `mistral_7b_gsm8k` goes first
(by-task batches across both models so model-specific failures surface
early).

**Reviewer story preserved:** 4 architecture families (Llama-3.1 +
Qwen-2.5 + Mistral-v0.3 + Yi-1.5) in the ~7–9B band — same headline
diversity, minus the within-Qwen scaling sub-point.

### Honest VRAM accounting (LoRA training, full bf16, Unsloth + grad-ckpt)

Recorded here so we don't over-ask again. Peak ≈ `base_size + 9–13 GB`
(LoRA + Adam fp32 for LoRA + activations at bs=8/seq=2048 + working
buffers). The original `llama31_8b_*.yaml` `min_free_gb: 40` was already
conservative for an actual ~25–30 GB peak.

| Model class | Realistic peak | Sane min_free_gb |
|---|---:|---:|
| 7B (Llama/Qwen/Mistral)  | ~23 GB | 30 |
| 9B (Yi)                  | ~27 GB | 35 |
| 12B (gemma-3)            | ~35 GB | 45 |
| 14B (Qwen-14B)           | ~41 GB | 50 |

`pbs_train.sh` hardcodes `gpu_picker --min-free-gb 40`. For 7B/9B the
picker is the binding check; for 12B/14B (if ever revived) the YAML
threshold becomes binding and the picker needs to be made
config-aware, otherwise `assert_env` fails inside `train_lora.py` after
a wasted 5–10 min of model load.

### GPU situation reminder

- Login node `jiit-master`: no GPU.
- Compute node `jiit-gpu01`: 8× A100-SXM4-80GB, shared informally
  (`naccelerators=0` in PBS). Picker via `utils/gpu_picker.py` polls
  nvidia-smi at job start, picks least-loaded.
- Per-user cap on the gpu queue: 3 concurrent jobs. v2 saturates this
  until drain; T1.C yields by counting both job-names.

### Next-up checklist (when v2 lands)

1. Re-run `code/phase3/eval/phase_b_analysis.py` on v2 data; regenerate
   headline figures (`headline_rd.png`, `method_compare.png`).
2. Supersede the n=200 / old-n=1k tables in §3 / `notes/phase3_findings.md`.
3. Confirm directions of F1, F2, F7 hold; quote v2 magnitudes from here on.
4. T1.C training continues in background (~3–4 hr after slot opens).
5. Surface T1.A2 (soft d_eff) decision; if green-lit, that's the next CPU run.

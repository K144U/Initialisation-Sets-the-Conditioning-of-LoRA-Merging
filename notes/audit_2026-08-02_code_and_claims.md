# Independent audit of the ICLR build against code + released results
**Date:** 2026-08-02. **Auditor:** Claude (Opus 5), reading `rdmerge/` end to end.
**Scope:** `ICLR/` (the current submission build), `code/phase3/`, `code/synthetic/`,
`results/phase3/`, `artifacts/lora/`, `decisions.md`, `CLAUDE.md`.

Everything below was verified against the shipped data or code, not inferred from prose.
Reproduction snippets are inline. Nothing here has been changed in the paper yet.

Severity key: **A** = invalidates a headline claim as written. **B** = a reviewer
who reads the code/data will find it and it costs the paper. **C** = fix before
submission, low risk.

---

## A1. The measured LoRA task subspaces are near-identical, not independent

The paper's central empirical claim (§6.2 finding 1, Intro "the lens", App B, R5) is:

> "Computing $d_{\mathrm{eff}} = \mathrm{rank}(\sum_t P_{V_t})$ directly from the trained
> adapters gives $d_{\mathrm{eff}} = Tr = 64$ in *every* layer of *every* model: the four
> task subspaces are linearly independent, the predicted floor is exactly zero, and the
> $0.10$–$0.22$ nats gap is method suboptimality."

Measured on `artifacts/lora/llama31_8b/*/seed1`, layer 9 `v_proj` (representative;
same on all four bases and all layers sampled):

```
pairwise mean principal cosine between the top-16 right-singular subspaces
           gsm8k  alpaca  magico  flores  codeal   dolly    xsum
gsm8k      1.000   0.983   0.980   0.986   0.058   0.058   0.056
alpaca     0.983   1.000   0.979   0.984   0.061   0.061   0.058
magicod    0.980   0.979   1.000   0.981   0.059   0.056   0.054
flores     0.986   0.984   0.981   1.000   0.055   0.055   0.054
codealp    0.058   0.061   0.059   0.055   1.000   0.989   0.979
dolly      0.058   0.061   0.056   0.055   0.989   1.000   0.984
xsum       0.056   0.058   0.054   0.054   0.979   0.984   1.000
```

All 16 principal angles between any two of the four $T=4$ tasks are near zero
(median cosine 0.996). The stacked matrix $[V_1|\dots|V_4]$ has
$\sigma_{\max} = 1.9995 \approx \sqrt{T}$, the exact signature of **coincident**
subspaces, and participation ratio $16.4/64 \approx r$, the exact
signature of *one* shared 16-dimensional subspace. Across all four bases
(seed1, same layer): soft $d_{\mathrm{eff}}$ = 16.41 / 16.32 / 16.45 / 16.26 out of 64,
$\sigma_{\max}$ = 1.9995 / 1.9993 / 1.9994 / 1.9996.

`hard = 64/64` only because `deff_analysis.py:157` sets the rank tolerance at
`S[0] * 64 * eps * 100` ~ $10^{-3}\sigma_1$, and the smallest stacked singular
value is $8\times10^{-3}$ to $2.4\times10^{-2}$. The hard test is a
machine-precision rank test; it cannot distinguish "independent" from
"identical to within 1%".

**Root cause: all $T$ adapters in a cohort share one LoRA `A` initialization.**
`train_lora.py:59` calls `seed_everything(cfg["seeds"]["global"])`, and every task
config for a given seed uses the same value (`configs/lora_seeds/*_seed1.yaml` all have
`seeds: {global: 1, train: 1}`). LoRA initializes `B = 0`, so `A` receives almost no
gradient signal and stays near its init. Measured:

```
relative ||A_x - A_y|| / ||A_x||   (two unrelated Gaussians -> 1.414)
  ACROSS TASKS, same seed1: gsm8k vs alpaca     0.176   <- same init
  ACROSS SEEDS, same task : gsm8k s1 vs s2      1.410   <- independent init
  row-space principal cosines, tasks  : max 0.9995 median 0.9957
  row-space principal cosines, seeds  : max 0.1567 median 0.0538
```

**Consequences, in order of damage:**

1. The regime diagnosis is inverted. The cohorts sit at (or adjacent to) the
   *maximum-overlap* limit $V_t = V$, $d_{\mathrm{eff}} = r$, which Lemma 2 assigns
   floor $B^2(1 - 1/T) = 0.75B^2 > 0$ — the floor-**positive** regime the paper
   devotes Appendix B to arguing "does not arise naturally". `deff_analysis_4model.json`
   already records this: `floor_predicted_soft_per_layer_mean = 0.0655` (Llama), sum
   8.39 across 128 layers. The paper reports only the hard floor (0.0) and never
   surfaces the soft one.
2. App B's stated mechanism is wrong. "Real fine-tuning resists subspace overlap: the
   top-$r$ right-singular basis of $\Delta_t$ is set by the per-task
   answer-distribution loss" — no. It is set by the shared `A` init. The
   $\alpha$-mixture arm could not move `hard` $d_{\mathrm{eff}}$ because the subspaces
   were already maximally shared and the statistic is blind to it.
3. It explains, and de-mystifies, the §6.3 spectral failure.
   $\bar H = \frac1T\sum V_tV_t^\top$ with $V_t \approx V$ has 16 eigenvalues $\approx 1$
   and 48 near-zero slivers; $\bar H^{+}$ divides by the slivers, hence the
   $25$–$94\times$ norm blow-up and the "median eigenvalue $2\times10^{-3}$, min $10^{-5}$"
   observation. The ridge $\lambda$ is doing the job a rank truncation at $d=r$ would do.
   That is a fine story, but it is a story about degenerate LoRA geometry, not about a
   rate-distortion construction.
4. The $T=7$ pool is geometrically two tasks, not seven. The three pilot adapters
   (codealpaca/dolly/xsum) come from a *different* training run with a different init
   (relative `A` distance 1.42 from the v1 four). Pool structure: two mutually orthogonal
   16-dim subspaces, each carrying 4 and 3 near-identical adapters. Soft
   $d_{\mathrm{eff}} = 31.98/112 \approx 2\times 16$. §6.4's semantic reading of the
   win-share pattern ("six adapters sharing instruction-following structure vote in a
   common subspace, the code adapter is outvoted") is describing this 4-vs-3 init split.

**The experiment that settles it:** retrain one cohort with per-task independent `A`
init (distinct `seeds.global` per task, or reseed before `get_peft_model`), then re-measure
soft/hard $d_{\mathrm{eff}}$ and re-run the $T=4$ matrix. Two possible outcomes, both
publishable, both better than the current text:
- soft $d_{\mathrm{eff}} \to \approx Tr$: the floor-zero claim becomes true *and*
  attributable, but the $\bar H$ degeneracy (and probably the ridge salvage with it)
  should largely vanish — which would move rd-encoder-ridge's headline.
- geometry unchanged: then the overlap is a property of fine-tuning after all, and the
  paper gains a genuinely strong, currently-missing control.

Either way this must be run before submission; it is ~16 adapter trainings + 20 eval
cells and it is the load-bearing fact of the paper.

---

## A2. KnOTS as implemented is algebraically identical to Task Arithmetic

`merging/knots.py` with the registry default `inner_combination="linear"`
(`registry.py:52`) computes, per layer,

```
V      = right singular vectors of [Δ_1; …; Δ_T]        (in_dim × k)
C_t    = Δ_t V ;  C_merged = Σ w_t C_t ;  Δ_merged = C_merged Vᵀ
       = (Σ w_t Δ_t) V Vᵀ  =  Σ w_t Δ_t                 (V Vᵀ = I)
```

`full_matrices=False` makes `V` square-orthogonal (or a projector onto a row space that
already contains every $\Delta_t$), so the round trip is the identity. Verified
numerically: `max|KnOTS − TA| = 2.9e-06` at scale 4.18. It is why the shipped cells match
TA to 3–4 decimals in all 100 matrix cells (e.g. Llama 3-seed: TA 0.2196, KnOTS 0.2194).

The paper uses this no-op as **positive evidence for its theory**, in four places:

- Intro: "Zero overlap also predicts which methods can work: subspace-alignment merging
  (KnOTS) has nothing to exploit and is statistically indistinguishable from naive
  averaging."
- §6.2 (3): "KnOTS and DARE are statistically indistinguishable from TA … *exactly as the
  theory predicts at zero subspace overlap, where alignment has nothing to exploit*."
- Related work: "our measurements reconcile the two … we measure KnOTS indistinguishable
  from Task Arithmetic."
- App J: "KnOTS … *tracks TA exactly* at $T = 7$ on Llama."

Combined with A1 the claim is wrong twice over: the implementation cannot differ from TA,
and the overlap is maximal rather than zero.

DARE $\approx$ TA is *not* a bug (unbiased drop-and-rescale has expectation TA); that row
is fine and should stay.

**Fix:** re-run KnOTS with `inner_combination="ties"` (the code already supports it; it is
the actual KnOTS-TIES of Stoica et al.), and align the projection with the paper's
construction (concatenate the LoRA `A` factors, not the materialized deltas). ~20 cells,
~5 GPU-h. Until then, every KnOTS-based inference must come out of the paper.

---

## A3. rd-encoder ridge is deployed at rank 64; every baseline at rank 16

`rd_encoder.py` supports two realizations. The headline $T=4$ cells use
`realize="rank_deff"`, which injects an adapter of rank $d_{\mathrm{eff}} = Tr = 64$
carrying $W^\star$ exactly ("no SVD, no truncation", docstring line 137). Every baseline —
and `rd_encoder`'s own default `rank_r` path — SVD-truncates to $r = 16$.

Verified from the shipped `method_kwargs`:

| cells | realize | rank |
|---|---|---|
| `eval_ridge_seed/`, `eval_ridge_xmodel/`, `eval_seed_rdridge_regmean/` (Table 1, Fig 2, App G) | `rank_deff` | **64** |
| `eval_e1_seed/` — the $\lambda = 0$ "exact centroid fails at 0.340" cells | *(default)* `rank_r` | **16** |
| T-scaling, Mistral $T=7$, GSM8K, HumanEval cells | *(default)* `rank_r` | 16 |

Two problems:

1. **The head-to-head is not storage-matched.** Table 1 ("lowest worst-task excess of the
   ten methods we benchmark") compares a rank-64 adapter against nine rank-16 adapters, in
   a paper whose premise is "at a given storage budget". The paper never states the rank.
   (`CLAUDE.md` §10 asserts "All methods are SVD-truncated back to rank r after the
   merge" — true of the registry, false of the cells that produced Table 1.)
2. **The salvage arc is confounded.** Figure 2 reads $\lambda = 0 \Rightarrow 0.340$
   versus $\lambda^\star \Rightarrow 0.094$ as the effect of the ridge, but the two cells
   also differ in deployed rank (16 vs 64, i.e. ~30% of solution mass discarded in one arm
   and none in the other).

**Good news: the fix is cheap and the result survives.** Rank-16 rd-ridge cells already
exist in `eval_e11_quadbridge/*__a100.json` ($\alpha = 1.0$ = the unscaled $T=4$ merge,
seed1 adapters):

| base | rd-ridge rank-16 | rd-ridge rank-64 (seed1) | TIES (seed1) | TA (seed1) |
|---|---|---|---|---|
| Llama-3.1 | **0.0849** | 0.0822 | 0.1471 | 0.2132 |
| Yi-1.5 | **0.0360** | 0.0356 | 0.0449 | 0.0989 |

So the rank cap costs ~0.003 nats and rd-ridge still wins comfortably. Run the two missing
rank-16 cells (Mistral, Qwen) × 3 seeds = 6 cells, report rank-matched numbers everywhere,
and keep rank-64 as an ablation. This turns an unfair-comparison objection into a
robustness result.

---

## A4. The HumanEval harness silently discards up to 79% of generations, method-dependently

`downstream_metrics.py:_strip_humaneval_completion` breaks at the first top-level
`def`/`class` *after* any output line. Combined with the markdown strip above it
(`gen.split("```python", 1)[1]` leaves a leading newline), any generation that wraps its
answer in a fenced code block yields `lines = ["", "def …"]` → `out = [""]` → break →
**empty completion**. The candidate file is then the bare prompt (signature + docstring),
which executes and fails every assertion.

Empty-completion counts out of 164, seed1:

| base | TA | DARE | KnOTS | TIES | TVQ₂ | rd-ridge |
|---|---|---|---|---|---|---|
| Llama-3.1 | 122 | 123 | 122 | 62 | 2 | 0 |
| Mistral-7B | 129 | 130 | 128 | 0 | 0 | 63 |
| Qwen-2.5 | 126 | 126 | 125 | 11 | 0 | 2 |
| Yi-1.5 | 4 | 5 | 3 | 1 | 1 | 2 |

pass@1 conditioned on a non-empty completion, pooled over 3 seeds:

| method | Llama-3.1 | Mistral-7B | Qwen-2.5 | Yi-1.5 |
|---|---|---|---|---|
| TA | 0.200 (n=135) | 0.211 (n=123) | 0.073 (n=109) | 0.062 (n=481) |
| DARE | 0.213 | 0.235 | 0.064 | 0.071 |
| KnOTS | 0.206 | 0.182 | 0.073 | 0.071 |
| TIES | 0.481 (n=362) | 0.251 (n=490) | 0.629 (n=428) | 0.082 |
| TVQ₂ | 0.448 | 0.244 | 0.647 | 0.082 |
| rd-ridge | 0.480 | 0.252 | 0.623 | 0.110 |

The ordering survives, but the paper's headline magnitude does not: "TA/DARE/KnOTS cluster
$2$–$40\times$ below them on pass@1" becomes roughly $2$–$9\times$ once the harness stops
throwing away three quarters of one group's generations. Yi is the control that proves the
mechanism: it is the one base where almost nothing is discarded, and it is the one base
where the method spread collapses.

It also dissolves a result the paper spent a retraction on: rd-ridge's "seed-instability"
on Mistral HumanEval (SD 0.084, attributed to "degenerate greedy generations") tracks the
empty counts exactly — 63 / 36 / 20 empties across seeds 1/2/3 → 0.079 / 0.213 / 0.281.

**Fix:** strip leading blank lines before the `def` scan, or (better) use the standard
HumanEval convention: take the fenced block if present, else the raw text, and never
return an empty body. Re-run 24 + 12 cells.

---

## A5. The GSM8K extractor fails on 60–81% of generations, method-dependently

`_GSM8K_FINAL_PATTERNS` requires the final number to be **at the end of the string**
(patterns 3 and 4 are `…$`-anchored). Any answer that finishes with words
("Thus, 12 students are good at math.") extracts `None`, which `gsm8k_score` counts as
wrong. Standard GSM8K harnesses take the last number *anywhere*.

Extraction-failure rate (`pred is None`), 3 seeds pooled:

| method | Llama-3.1 | Mistral-7B | Qwen-2.5 | Yi-1.5 |
|---|---|---|---|---|
| TA | 0.611 | 0.809 | 0.676 | 0.062 |
| DARE | 0.611 | 0.805 | 0.691 | 0.073 |
| KnOTS | 0.603 | 0.813 | 0.681 | 0.063 |
| TIES | 0.676 | 0.221 | 0.101 | 0.019 |
| TVQ₂ | 0.620 | 0.297 | 0.144 | 0.017 |
| rd-ridge | **0.693** | 0.107 | 0.025 | 0.016 |

The paper states the regex "succeeded in $95\%$ of generations in an $n = 100$ pilot"
(App E). The shipped runs contradict that on three of four bases.

This is very likely the explanation for the paper's most prominently flagged mystery —
the (Llama-3.1, GSM8K EM) inversion, $\rho = -0.60$, "empirically unexplained", listed as
limitation (5) and the subject of two falsified hypotheses (H1 chat-template, H2 delta
magnitude). On Llama, rd-ridge has the **highest** extraction-failure rate of the six
methods (0.693) and consequently the lowest EM. H3 was never the right hypothesis; the
right one is answer-format compliance, and it was never tested.

Diagnostic (weak, because `gen_text` is stored truncated to 200 chars and 75–88% of
previews hit that cap — this is a strict undercount): re-scoring the stored previews with
last-number-anywhere flips the Llama ordering, rd-ridge $0.201 \to 0.323$ against TA
$0.315 \to 0.225$. Not conclusive, but enough that the cell cannot stand as reported.

**Fix:** re-score with a standard extractor. Requires re-running generation (24 + 12
cells) because full generations were not stored — and store them this time.

---

## B1. Train/eval overlap on alpaca and magicoder in the multi-seed matrix

The Reproducibility Statement says "Training and evaluation splits are disjoint (verified
by a separate zero-overlap audit)."

`data_loaders.py:105` guarantees disjointness only when `split_eval == split_train` **and
both processes shuffle with the same seed**:

```python
shuffled  = ds[train_split].shuffle(seed=seed)
train_raw = shuffled.select(range(n_train))
eval_raw  = shuffled.select(range(n_train, n_train + n_eval))
```

They do not use the same seed:
- training: `train_lora.py:94` passes `seed=cfg["seeds"]["global"]` = **1 / 2 / 3**
  (`configs/lora_seeds/*`), or **20260517** for the v1 cohort.
- evaluation: `run_eval_cell.py:211` passes `seed=cfg.get("seed")` = **20260518**
  (20260520 for the seed3 cells).

alpaca and magicoder both use `split_eval: train`, so the "held-out" 1000 examples are a
different permutation's slice. Expected contamination
$= n_{\text{eval}}\, n_{\text{train}} / N$:
alpaca-cleaned ($N = 51{,}760$) $\approx 145/1000 = 14.5\%$;
Magicoder-OSS-Instruct-75K ($N = 75{,}197$) $\approx 100/1000 = 10\%$.
That is the same bug `log.md` F8 recorded on 2026-05-18 and believed fixed; the fix closed
the within-process case only. GSM8K (`test` split) and WMT19 (`validation`) are unaffected.

Impact on the headline is probably small — worst-task excess is GSM8K in 80/100 matrix
cells — but the Reproducibility Statement is currently false as written, per-task
alpaca/magicoder numbers are biased, and Yi's "headroom" figures (which drive R2 and the
whole saturation narrative) are computed from contaminated $\mathrm{NLL}_\tau$.
**Fix:** set the eval cell `seed` to the adapter's training seed, re-run, or state the
measured overlap honestly.

## B2. Three different numbers are reported for the same quantity

| quantity | §6.2 main text | Table 1 | Fig 2 caption / App G |
|---|---|---|---|
| Llama TA worst-excess | 0.218 | 0.213 | 0.220 |
| Llama TIES worst-excess | — | 0.147 | 0.154 |

Sources: §6.2's "(TA: Llama 0.218, Mistral 0.146, Qwen 0.110, Yi 0.099)" is the **v1**
adapter cohort (`eval_matrix_n1k_v3_perexample`: 0.2191/0.1454/0.1097/0.0988); Table 1 is
**seed-1 only** (verified against `e10_baselines_summary.json` and
`eval_e12_regmean_adamerging/`); App G and Fig 2 are **3-seed means** (recomputed from
`eval_matrix_seeds/`: TA 0.2196, TIES 0.1539). Table 1's caption discloses seed-1 for the
four *added* baselines but is silent about the five matrix baselines, while rd-encoder
ridge in the same table is a 3-seed mean. The 2026-07-01 "matched seed1/2/3 re-run" fixed
rd-ridge's numbers but not the baseline numbers scattered through the main text.

The direction is conservative (seed-1 baselines are the *best* of the three seeds, so
rd-ridge's margin is understated) — but the inconsistency is visible on inspection.

## B3. The held-out-$\lambda$ check is circular on 3 of 4 bases

App G presents $\lambda = 0.13$ as a "leave-one-base-out modal choice … with no per-base
selection". But `eval_ridge_xmodel` swept only $\lambda \in \{0.05, 0.07, 0.10, 0.13\}$ on
Mistral/Qwen/Yi, and the response is **monotone decreasing to the grid edge** on all three
(Mistral 0.127→0.075→0.046→0.038; Qwen 0.019→0.014→0.011→0.010; Yi 0.120→0.055→0.041→0.034).
So (a) $\lambda^\star = 0.13$ is "the largest value we tried", not a measured optimum, and
App G's "the sweep is monotone-then-flat on the other three bases" has no data past 0.13;
(b) $\lambda = 0.13$ *is* those three bases' selected value, which is why
`tab:rd-ridge-heldout` prints identical numbers in the $\lambda^\star$ and $\lambda = 0.13$
columns for three of four rows. The genuine held-out test has $n = 1$ (Llama, 0.125).
**Fix:** extend the sweep to $\lambda \in \{0.2, 0.3, 0.5, 1.0\}$ on the three bases
(12 cells) and restate the check honestly.

## B4. Every $T = 7$ point is a single merge cell

`e6_T_scaling_summary*.json` / `mistral_t7_summary.json`: `T7__*` has `n_subsets = 1`
(subset `"all"`), one training seed. So:
- The Yi TIES inversion — TIES 0.1476 vs TA 0.1279, gap **0.0197** — is one cell with no
  error bar. The subset-to-subset range at $T=4$ on the same base is 0.017–0.019, i.e. the
  same size as the whole effect.
- The pre-registered Mistral verdict, Figure 1's right-hand points, and the "salvage grows
  with $T$" ratios all rest on $n = 1$ per (base, method).
- The $\log T$ fits are 2-parameter fits to **3 points**; $R^2 \ge 0.93$ on 16/18 cannot
  distinguish $\log T$ from linear or $\sqrt{T}$.

Figure 1's caption ("bands are per-subset min/max") is silent that the band collapses to a
point at $T = 7$. **Fix:** run 3 more 7-task subsets per base (drop-one cohorts), 3 bases ×
3 subsets × 6 methods = 54 cells; or state the $n = 1$ limitation prominently.

## B5. The Td2 synthetic's Pearson 0.998 is near-tautological

`td2_overlap_sweep.py` drives a single knob $\rho$ and reports
$\mathrm{Pearson}(R,\ \mathrm{TIES}-\mathrm{TA}) = 0.998$, concluding "the win-share range
is the **causal** control variable". Both $R$ and the gap are monotone functions of the
same knob; correlating two monotone transforms of one dial over a monotone sweep returns
$\approx 1$ by construction. $\rho$ is the causal variable; $R$ is a second readout of it.
The generator is also 6 identical majority tasks + 1 designated minority, so the max-over-$t$
excess is the minority's by design.
**Fix:** either soften to "monotone coupling" (the appendix's own "robust claims" sentence
already does this — the "causal control variable" phrasing contradicts it), or vary a
second factor (magnitude jitter, minority count) and show $R$ still predicts the gap.

## B6. The floor formula is dimensionally incomparable to the reported excess

$\Phi^\star_{\mathrm{lower}} = B^2(1 - d_{\mathrm{eff}}/(Tr))$ is in weight-space
$H$-metric units (`deff_analysis.py` uses $B^2 = \max_t\|\Delta_t^\ell\|_F^2$, per layer);
the measured excess is nats per response token. R5 and App B step 4 tell practitioners to
compare them ("only $\Phi^\star_{\mathrm{meas}} - \Phi^\star_{\mathrm{lower}}$ is
recoverable"). That is only vacuously fine while the floor is 0. Given A1, the recipe needs
an explicit conversion or an explicit statement that it returns a zero/non-zero verdict, not
a nats-comparable quantity.

## B7. The RD theorem is not exercised by any headline experiment

Every headline rd-encoder cell runs at `bits = 32`, i.e. **no quantization**
(`rd_encoder.py:20`: "bits >= 32: no quantization"). The rate term
$c_{\mathrm{TQ}}(B^2 d_{\mathrm{eff}}/Tr)2^{-2R/d_{\mathrm{eff}}}$ and Theorem 2's
achievability are therefore never tested on real adapters; what is tested is a
ridge-regularized $\bar H$-weighted centroid, and the centroid comes from Lemma 1 (an
algebraic identity), not from the rate-distortion bound. The finite-$b$ cells that *do*
exercise the rate (`eval_e1/llama31_8b__rd_b{1,2,3,4,8,16}`) all lose badly (0.36–11.4).
Combined with the floor being zero, the theorem's content at every measured operating point
is $\mathcal{D}^\star \ge 0$.

This is a framing problem more than an error, and the Scope paragraph gestures at it, but a
reviewer will ask "what does the rate-distortion theorem buy the practitioner here?" and the
paper needs a direct answer. The honest and still-strong one: the *identity* (Lemma 1) plus
the measured geometry motivates the estimator; the rate machinery is validated
synthetically.

---

## C. Smaller items

- **C1** ~~`references.bib` has `concurrent2026merging` (`journal = {Anonymous preprint}`),
  never cited — delete.~~ **CORRECTED 2026-08-02**: this is not a leftover
  placeholder. The cluster checkout carries an uncommitted diff renaming
  `pathak2026merging` to `concurrent2026merging` and stripping the authors
  (Pathak, Garg) and DOI (`10.21203/rs.3.rs-9189872/v1`) from the authors' own
  Research Square preprint, deliberately, for double-blind review. Do not
  delete it. Two real issues remain: it is currently **uncited**, so it is
  anonymizing nothing; and an `Anonymous / Anonymous preprint` entry is *more*
  conspicuous to a reviewer than an ordinary third-person self-citation, which
  is the accepted way to cite your own prior work under double-blind. Either
  cite it in third person with full metadata, or drop it.
  `regmeanpp2025` and `lorm2024` and `tspa2025` have no eprint/venue.
  `tara2025` has key-year 2025 but `year = {2026}`. `NOTES.md` already flags three 2026
  concurrent works to add (2603.09463 merging-collapse w/ rate-distortion, 2601.22285,
  2606.19549); 2603.09463 in particular needs differentiating — it reportedly uses the same
  lens.
- **C2** "Worst-task" is GSM8K in **80 of 100** matrix cells (alpaca 19, magicoder 1,
  translation 0). The max is not doing much work; report the argmax distribution.
- **C3** The translation adapter has **negative** headroom on Llama ($-0.011$) and Mistral
  ($-0.025$): fine-tuning made translation NLL worse than the frozen base, so merged models
  routinely beat $\tau_{\text{translation}}$ (excess $-0.12$ on Llama TA). One of four tasks
  has no usable reference. `log.md:284` already suspected the cause (WMT19 streaming takes
  the unshuffled first 7500 pairs). Either retrain it or drop it and say why.
- **C4** Yi's magicoder headroom (0.045) is *lower* than its alpaca headroom (0.096), but
  only Alpaca is flagged as near-saturated in App D. The saturation narrative should name
  both.
- **C5** HumanEval "sandboxed exec" is `subprocess.run(["python", f])` with a timeout — no
  isolation. Reword or sandbox properly.
- **C6** `ties._trim_topk` uses `>= threshold`, keeping more than $k$ entries under ties;
  negligible at fp32 but it makes the effective density mildly method-favourable at low
  density. Worth a line in the appendix or a `topk`-index mask.
- **C7** TVQ's per-tensor min-max quantizer at $b = 2$ is dominated by outliers, so most
  coefficients collapse toward the near-zero bucket — i.e. $b{=}2$ TVQ ≈ shrunken TA. App C
  lists "uniform shrinkage of large coefficients" as candidate (i) but never tests it. The
  control is one line: sweep TA with a scalar $\alpha$ and see whether $\alpha$-tuned TA
  reproduces the $b{=}2$ dip. Cheap, and it also bears on rd-ridge (larger $\lambda$ rescales
  the merged delta).
- **C8** Data-free RegMean's Gram $\sum_t A_t^\top A_t$ has rank $\le Tr = 64$ out of
  in_dim 4096, so the $\lambda I$ term dominates the solve on the 4032-dim null space. The
  baseline is heavily handicapped relative to real (data-dependent) RegMean, and — per A1 —
  its Gram is also the shared-init subspace. The paper says this variant is "what most public
  re-implementations actually ship"; that claim needs a citation or should be softened, since
  the tuned-RegMean-beats-us-on-Yi result is the paper's most-cited honesty point.
- **C9** `arxiv/v2/main.tex` still has `[EMAIL PLACEHOLDER]` ×2 and
  `reproducibility.tex` still says "accompanies the camera-ready version" in a venue-neutral
  preprint build.
- **C10** The anonymized audit-trail bundle the paper promises (Intro footnote,
  Reproducibility Statement, App I) does not exist yet. `ICLR/NOTES.md` has the recipe.

---

## Suggested order of work

1. **A1 init experiment** — retrain one cohort with per-task independent LoRA `A` init and
   re-measure. Everything geometric in the paper depends on the answer, including whether
   the ridge salvage survives. Do this first; it has the longest lead time.
2. **A4 + A5 metric re-scoring** — fix both harnesses, re-run 36 downstream cells, store
   full generations. Likely resolves the paper's one "unexplained" outlier.
3. **A2 KnOTS re-run** (~5 GPU-h) and **A3 rank-matched rd-ridge** (6 cells). Both are small
   and both remove a clean reviewer kill-shot.
4. **B1 seed alignment + re-run** of alpaca/magicoder-affected cells, or an honest measured
   overlap statement.
5. **B2/B3/B4** number harmonization, extended $\lambda$ grid, extra $T=7$ subsets.
6. **B5–B7 + C** rewrites: framing, caveats, bibliography, placeholders.

## What holds up

Worth stating plainly, because most of this audit is negative:

- The 3-seed NLL result itself is solid and reproduces exactly from the raw cells
  (rd-ridge 0.0945 / 0.0441 / 0.0097 / 0.0366 vs TIES and TA; every number in
  `tab:rd-ridge-sweep` and `tab:rd-ridge-heldout` recomputed to the printed digit).
- The multi-seed bootstrap, the $\lambda$ U-curve, the RegMean-$\lambda$ and density sweeps,
  and the tuned-RegMean-wins-on-Yi retraction are all honestly reported and reproduce.
- The pre-registration audit trail is genuine: commit `3582799` precedes every Mistral
  $T=7$ result commit.
- The 2026-07-01 retraction of the "best on all 8 downstream cells" claim was the right call
  and was made against the author's own interest.
- The engineering (orchestrator idempotency, per-example NLL arrays, diagnostics written to
  disk) is what made this audit possible at all. Very little of the above could have been
  found in a paper that shipped only summary tables.

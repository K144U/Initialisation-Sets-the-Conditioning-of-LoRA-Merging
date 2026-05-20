# Open questions

Pre-seeded from `plan.md` §3.2 (Phase 1 technical questions). Add as they
surface, close with a date and a one-line answer when resolved.

## Phase 1 technical questions (plan.md §3.2)

1. ~~**Rank-r effect on the bound.**~~ **Resolved 2026-04-28 — Theorem 7
   of `theory/theorem_v1.tex`** (proof in new §5) gives, for
   $H_t = P_{V_t}$ with $\tau_t \in V_t$ Stiefel-random,
   $\mathcal{D}^\star \geq B^2(1 - \deff/(Tr))
   + c_\mathrm{TQ}(B^2 \deff/(Tr)) \cdot 2^{-2R/\deff}$
   with $\deff := \rank(\sum_t P_{V_t})$. Closed-form floor in
   Lemma 6. Three sanity-check limits (Props 3–5) fall out as
   one-line-reduction corollaries. Numerical Check (b) passes
   cleanly on the Hbar-weighted quantization-error metric
   (slope $-2.00$ across all shared-$V$ cells).

2. **T-scaling.** Does the bound depend on $T$ (number of tasks)
   linearly, logarithmically, or at some intermediate rate?

3. **Adversarial vs random tightness.** Is the bound tight under
   adversarial task vectors, or only under random / Gaussian task
   vectors?

4. **MSE → CE extension.** Can the bound be extended from MSE
   distortion on quadratic loss to cross-entropy distortion on real
   LLM loss surfaces, at least via local-quadratic-approximation
   arguments?

## Phase 1 open items (from `theory/theorem_v1.tex`)

- ~~**Prove Conjecture 2 for $H_t = P_{V_t}$.**~~ **Resolved
  2026-04-28 — Theorem 7 of `theorem_v1.tex`** (new §5).
- ~~**Closed form for $f_\mathrm{floor}$.**~~ **Resolved
  2026-04-28 — Lemma 6 of `theorem_v1.tex`** (new §4):
  $f_\mathrm{floor} = 1 - \E_P[\deff]/(Tr)$. Orthogonal-limit
  worry resolved in the opposite direction: floor is $0$, not
  $B^2$ (task vectors become mutually invisible, $\tauH$
  reconstructs each $\tau_t$ exactly).
- ~~**Scale-adaptive quantization for numerical verification.**~~
  **Resolved 2026-04-28:** `day8_rank_r_sanity.py` now has
  `quantize_tauH_adaptive` with per-coord range $\pm c\sigma_{\mathrm{pc}}$
  and reports a second `excess_hbar = $\|w^\star-\tauH\|^2_{\bar H}$`
  metric. Shared-$V$ slope on `excess_hbar` is $-2.00$ across all
  cells. Finding: the Day-8 slope deficit was a cross-term in the
  `max $-$ floor` metric, not a quantizer artifact; the adaptive
  scaling plus the right metric choice both contribute.
- **Robustness of $\tauH$ for small-overlap $V_t$.** When
  $\sum_t P_{V_t}$ is close to rank-$Tr$ block-diagonal, $\bar H$ is
  well-conditioned. For irregular overlaps it may not be. Does the
  RD step need a conditioning assumption? The current Theorem 7
  proof step 5 uses an $O(\deff)$ stabilizer argument that requires
  $\deff$ a.s.\ constant — holds under Stiefel-random $V_t$, may
  fail more broadly. Day 10+.
- ~~**General $H_t \succeq 0$ beyond projectors** (Alt.~C of Phase 0
  §6.2).~~ **Resolved 2026-04-29 — Theorem 8 of `theorem_v1.tex`
  §5.5** (`thm:general`). Generalized hard distribution $P^\star$
  with $\tau_t$ uniform on the $H_t$-ellipsoid in $V_t$ makes the
  $H_t$-weighted energy isotropic; Step 5's $O(W)$-equivariance
  transfers because $H_t = U_t D_0 U_t^\top$ transforms as
  $OH_t O^\top$ under $O\in O(d)$. Bound is identical in form to
  Theorem 7 — same floor, same compression term, no $D_0$ dependence
  after the $D_0^{-1/2}$ stretch. Assumption: $D_0$ (eigenvalue
  spectrum inside $V_t$) is common across tasks. Task-dependent
  deterministic $D_t$ is a new Day-11 open item.
- ~~**Random-$V$ $\deff$ fits at $r \geq 16$.**~~ **Resolved
  2026-04-29** — the Day-9 `quantize_tauH_adaptive` quantizes
  $Q^\top\tauH$ in a fixed per-coord range, but per-coord std of
  $Q^\top\tauH$ scales as $B/\sqrt{T\lambda_i r}$ (non-uniform for
  non-identity $\bar H$). `day10_general_ht.py:quantize_eta_adaptive`
  quantizes $\eta = \bar H^{1/2}\tauH$ directly; per-coord std is
  uniform $B/\sqrt{Tr}$ by the trace identity, slope is
  $-2.00\pm 0.01$ across all cells including random-$V$ and
  non-identity $D_0$ with $c=5$ clipping range.

## Phase 1 Day-11 open items (new, 2026-04-29)

- ~~**Task-dependent deterministic spectra $D_t$.**~~ **Resolved
  2026-04-30 — Day-10 hedge retired.** Close re-read of the
  Theorem 8 proof showed that Lemma 6' and Step 5 are *both* $D_t$-
  free: the floor identity $H_t H_t^+ H_t = H_t$ is per-task, and
  $v_t = B D_t^{-1/2} u_t$ is $O(d)$-free (lives in $\R^r$)
  regardless of which $D_t$ is baked in. Numerical verification in
  `day11_task_dep_Dt.py` across 8 task-mismatched configurations:
  floor/trace to 4–5 digits, MP-UB respected, slope $-2.00\pm 0.01$.
  Theorem 8 now states $(D_t)_{t=1}^T$ arbitrary; Remark "why the
  bound doesn't depend on $(D_t)_t$" updated; CE-Fisher remark
  simplified (no longer needs common-$D_0$ modeling assumption).
- **Achievability upper bound.** *(In progress 2026-05-01, partial
  closure.)* Day 12: randomized Walsh-Hadamard quantization of
  $\eta = \bar H^{1/2}\tauH$ matches Theorem 8 LB to a constant
  factor $\approx 11$ in floor-zero cells but has padding overhead
  for non-pow-2 $d_{\text{eff}}$. Day 13 Fix-1: Gaussian-QR mixer
  (no padding) → constant factor $\leq 13$, slope $-2$, across all
  Stiefel-random cells. Theorem 9 to be written in §9 of
  `theorem_v1.tex` (Day 14). **Shared-V / floor > 0 gap:** the
  $H_t$-Chebyshev-center fix doesn't close the $2^{-R/d_{\text{eff}}}$
  scaling because at a cheb center with $\geq 2$ active tasks, the
  KKT condition makes perturbation-induced max-distortion *linear*
  in $\|\delta w\|$, not quadratic. This is fundamental for
  linear/deterministic encoders. Either the LB is loose here (an
  artifact of $\max \geq \text{avg}$ reducing to an avg-distortion
  RD bound), or a smarter encoder (null-space dithering) is needed.
  Phase 2.5 item; not a paper-blocker since the Stiefel-random
  generic case (floor-zero) is the headline regime.
  **Day 14 update (2026-05-02):** null-space-aware bit allocation
  partially closes the gap. Optimal balanced allocation between
  parallel ($|A|-1$ dims) and perpendicular ($r-|A|+1$ dims) gives
  excess $\asymp 2^{-2R/(r+|A|-1)}$, strictly better than naive
  $2^{-R/r}$. Verified for iso+iso shared-V $T=2, r=4$: slope
  improved from $-1.12$ (naive) to $-1.66$ (null-split), matching
  theoretical prediction $-8/5 = -1.60$. Non-iso cells fail due to
  Chebyshev-solver convergence issue (fixable for $T=2$ via
  closed-form KKT root-find; solver bug, not theory bug).
  **Day 14 closeout update (2026-04-24):** closed for $T=2$ across
  all anisotropy regimes. `day14b` closed-form Chebyshev solver
  (brentq on KKT $\alpha$) + `day14c` fractional per-coord bits
  + `day14g` locked clip $c=11.5\sigma_{pc}$ yields slope
  $-1.60 \pm 0.10$ across iso+iso, iso+geom, geom+twos, lin+twos,
  geom+iso at $n_{trials}=1000$ (bootstrap 95% CI $\pm 0.010$ on
  each slope). Theoretical prediction $-2r/(r+|A|-1) = -1.60$
  empirically confirmed as the central tendency; anisotropy
  adds $\pm 0.10$ structural spread.
  **Day 15--16 update (2026-04-24):** Phase 2.5 items partially
  resolved. (a) *General-$T$ null-space split*: SOCP Chebyshev
  solver (cvxpy CLARABEL) + Gauss--Newton KKT refinement
  (`day15_cheb_general_T.py`) extends the construction to
  $T\geq 3$. Slopes match or beat the linear prediction
  $-2r/(r+|A|-1)$ for $T=3, 4$ after switching the metric from
  "excess-over-avg-floor" to "excess-over-cheb-squared"
  (the latter is what the theory bounds; the former saturates
  when cheb$^2 \neq$ avg floor, which happens for $T\geq 3$
  under anisotropy).
  (b) *LB sharpening ruled out*: a valid rate-$R$ random-codebook
  encoder (`day16_sharpened_lb_check.py`) achieves slopes
  strictly more negative than the linear UB $-2r/(r+|A|-1)$ for
  $T=3$ ($-1.45$ vs linear $-1.20$), which would violate any
  purported LB at that exponent. Hence the truth in the shared-$V$
  regime lies strictly between $-2r/r$ (current Thm 8, loose) and
  $-2r/(r+|A|-1)$ (linear UB, also loose). Exact RD is open and
  beyond Phase 2 scope.

## Day 3 follow-ups (from Ortiz-Jimenez)

- **Overlap coefficient $\gamma$.** *(In progress 2026-04-27 —
  `theorem_v1.tex` §3 adopts $\gamma := \|T^{-1}\sum_t P_{V_t}
  \|_\mathrm{op}$ as the candidate knob.)* Day-8 numerics suggest
  the stronger invariant for $\deff$ is $\rank(\sum_t P_{V_t})$, not
  $\gamma$ itself. $\gamma$ may still parametrize $f_\mathrm{floor}$;
  Day-9 integral should reveal.

## Phase 0 open items (from toy_theorem_v0.tex)

- Sphere packing vs orthogonal tuples for the worst-case construction.
- $T=1$ reduction to known scalar-quantization RD bounds — does it
  land on Berger / Lloyd-Max? Sanity check.
- Is the adversarial packing realizable, or do we need to weaken to
  average-case over a Gaussian $\tau$ prior?
- **EPI rewrite of Lemma 2** (added 2026-04-21 after bug-fix addendum):
  replace the $\rho$-conditioning plus Chebyshev-on-$\|\bar\tau\|^2$
  concentration argument with an entropy-power-inequality derivation
  that gives a fully non-asymptotic absolute-scale bound. Needed to
  cleanly drop the $(1 + o_d(1))$ prefactor currently absorbed into
  $c_{\mathrm{TQ}}$ by convention. Captured in
  `theory/toy_theorem_v0.tex` §\ref{sec:open} item 1.

## Day 1 items (from reading TurboQuant Thm 3)

- ~~**Distortion measure:** raw vs excess-over-Chebyshev vs sum.~~
  **Resolved 2026-04-20 by numerical sanity check:** use **max** (or
  equivalently excess-over-Chebyshev). Sum collapses to single-vector
  quantization of $\bar\tau$ and reduces to TurboQuant Thm 3 — not
  novel. See `notes/turboquant_thm3.md` §9.
- **Expected bound shape:** $D^\star(R) \geq B^2 \cdot h(T) \cdot
  2^{-2R/d}$ for some $h(T)$. What is $h$? Linear in $T$, log in $T$,
  or $\sqrt{T}$? Answer governs whether the bound is tight enough to
  survive review.
- **Multi-source RD literature:** does the multi-source max-distortion
  bound we need already exist? Candidates to check:
  - El Gamal & Cover 1982 *Achievable Rates for Multiple Descriptions*.
  - Berger & Yeung *Multiple-Description Coding*.
  - More recent: common-reconstruction coding, Gray-Wyner.
  If yes, we cite and focus effort on the LoRA-specific achievability.
  If no, we prove our own Shannon-LB analog — bigger contribution but
  more work.
- **OpenReview camera-ready check:** did the TurboQuant camera-ready
  version change Thm 3's constants or structure relative to arXiv v1?
  Deferred from Day 1; fold into Day 2 reading.

## Day 1 follow-ups (from numerical experiment)

- **Two-regime RD curve under max-distortion.** Empirically $4^{-b}$ at
  low rate, $2^{-b}$ at high rate (linear cross-term
  $2\|w^\star-\bar\tau\|\sqrt{\text{Cheb}^2}$). The $2^{-b}$ regime is
  the novel technical hook. Does the lower bound actually bite here,
  or can an adversarial packing make $\text{Cheb}^2$ vanish?
- **Chebyshev radius of structured task vectors.** For rank-r LoRA
  vectors (not iid sphere), what is the typical $\text{Cheb}^2$? Phase
  1 question.
- **Is the linear regime present for the optimal encoder-decoder?**
  The quantized-mean merge is linear in $\tau$ and gives $2^{-b}$ at
  high rate. Could a non-linear encoder (encode the Chebyshev center
  directly) achieve $4^{-b}$ throughout? If yes, the $2^{-b}$ regime
  is an artifact of linear merges, not a fundamental RD phenomenon.
  Decide by reading Day 4/5 proof attempts.

## Day 4 items (from theorem drafting)

- ~~**Is the $2^{-R/d}$ regime fundamental?**~~ **Resolved 2026-04-24
  by Day 5 numerics (`code/synthetic/day5_lower_bound_sanity.py`):**
  NO, the apparent $2^{-R/d}$ regime in the Day 1 plot was an artifact
  of using the *mean* $\bar\tau$ as the quantization center. The
  quantized-Chebyshev-center merge exhibits pure $2^{-2R/d}$ decay of
  the excess over $\cheb^2$ across the full rate range tested — at
  $T=4,d=128$, excess drops by factor $\sim 25\times$ per 2 bits.
  Consequence: the Step-A-only lower bound $\cheb^2 + c(B^2/T)
  \cdot 2^{-2R/d}$ is **tight up to constants** — the RD function of
  max-distortion merging under iid uniform-sphere tuples is
  $\Theta\bigl(\cheb^2 + (B^2/T)\cdot 2^{-2R/d}\bigr)$. Case (b) of
  the original open-item dichotomy was correct: the mean-merge upper
  bound was loose, and the Chebyshev-center merge closes the gap.
- **Constant in the Gaussian-centroid Shannon LB.** TurboQuant Lemma 3
  gives the sphere-uniform Shannon LB directly. For our $\bar\tau$
  (mean of $T$ iid sphere, approximately $B/\sqrt{T} \cdot S^{d-1}$
  with fluctuations), do we apply Thm 3 directly on the
  $B/\sqrt{T}$-sphere and absorb fluctuations into the constant, or
  do we use the Gaussian LB (Cover \& Thomas Thm 10.3.2) with
  variance $B^2/(Td)$? Decide on Day 5 — whichever gives the cleaner
  constant.

## Day 3 follow-ups (from Ortiz-Jimenez)

- ~~**Overlap coefficient $\gamma$.**~~ Moved to the Phase 1
  section above; adopted in `theorem_v1.tex` §3 as
  $\gamma := \|T^{-1}\sum_t P_{V_t}\|_\mathrm{op} \in [1/T, 1]$.
  Day-8 numerics suggest the stronger invariant is
  $\rank(\sum_t P_{V_t})$, not $\gamma$, for $d_\mathrm{eff}$;
  $\gamma$ may still parametrize $f_\mathrm{floor}$.
- **Partial resolution of Phase 1 OQ #1 (rank-$r$ effect).** Weight
  disentanglement formalizes why rank matters: the Hessian $H_t$ of
  task-$t$ loss near $\tau_t$ is supported on the rank-$r$ subspace
  that task-$t$ fine-tuning actually moved. Packing dimension for the
  max-over-$t$ bound is $\mathrm{rank}(\sum_t P_{V_t}) \leq \min(Tr, d)$,
  not $d$. *Day-8 update:* numerical fits strongly confirm this is
  the right $d_\mathrm{eff}$; full resolution is Day 9–10 proof of
  Conjecture 2 in `theorem_v1.tex`.
- **Partial resolution of Phase 1 OQ #4 (MSE → CE extension).** The
  NTK / local-quadratic bridge is the intended extension path. CE
  loss near $\tau_t$ is locally $H_t$-quadratic with $H_t$ the
  Fisher-information matrix — exact same RD machinery applies on the
  support of $H_t$. *Day-8 update:* the $H_t$-weighted max $\geq$ avg
  identity (Lemma 1 of `theorem_v1.tex`) is exactly the tool needed;
  extends cleanly once the $H_t = P_{V_t}$ case closes.

---

## Phase 3 Day 17–18 open items (2026-05-18, after first full eval matrix)

Following the 20-cell eval matrix and Phase B criteria check (see
`notes/phase3_findings.md` for full results), these are the next-action
questions in roughly the order I'd hit them.

### O17.1 — TVQ b=2 dip — RESOLVED (real, not noise). New question: WHY?

**Resolved 2026-05-18 via n=1k rerun:** Llama TVQ b=2 dip is REAL — worst_excess
0.104 at n=1k (vs 0.107 at n=200) against b=4 worst_excess = 0.217. 5× more
eval data; dip magnitude unchanged. Not sample variance; structural.

Now the open question is the MECHANISM. Three candidates (none confirmed):
- (a) **Quantization-as-regularization at b=2** — destroys destructive
      interference patterns between task vectors that finer rates preserve.
- (b) **Stochastic-resonance-like effect** — b=2 noise pushes merged delta
      toward a region of the loss landscape that is better for gsm8k.
- (c) **Implicit coarse-projection** — 4 levels per layer aligns with the
      dominant LoRA-rank-16 structure.

**Pending Qwen confirmation** (TVQ b=2 cell still running as of 2026-05-18
afternoon). Whether Qwen shows the same dip will inform whether the effect
is structural (both models → mechanism (a) or (c)) or model-specific (Llama
only → mechanism (b) or Llama-specific weight distribution).

**Action items for the paper:**
1. Wait for Qwen TVQ b=2 cell to finish.
2. Map out the dip's shape: TVQ at b ∈ {1.5, 2.5, 3} via Lloyd-Max with
   non-power-of-2 levels.
3. Per-task excess breakdown — is the b=2 win on gsm8k (the bottleneck), or
   across all 4 tasks?
4. Repeat on a different 4-task subset to rule out cross-task-overlap artifacts.

**Potentially a paper-headline contribution.** The bound we proved doesn't
preclude this (it's an upper bound on quantization-induced distortion;
nothing says quantization can't HELP). Possible framing: "the bound
characterizes worst-case distortion; we observe an empirical regime where
quantization improves the merge — consistent with the bound but not predicted
by it."

### O17.2 — Why is Qwen-2.5-7B ~2× easier to merge than Llama-3.1-8B?

Task Arithmetic worst_excess: Llama 0.225 vs Qwen 0.107. Same gap across
all non-TVQ methods. Both models trained on the same 4 tasks with identical
LoRA configs.

The theory says floor scales like $B^2 (1 - d_{\mathrm{eff}}/(Tr))$. If Qwen's
task vectors carve cleaner orthogonal subspaces, $d_{\mathrm{eff}}/(Tr)$ is
closer to 1 (smaller floor). Action: compute empirical
$d_{\mathrm{eff}} = \mathrm{rank}(\sum_t P_{V_t})$ per layer for both models;
plot floor vs $d_{\mathrm{eff}}/(Tr)$ across the 2-model × 4-method × 4-task
grid. If the bound's floor formula explains the gap, that's a strong
paper-worthy finding.

### O17.3 — KnOTS ties TA on worst_excess but beats it 5/8 per-task. Real or noise?

Possible the "tie on worst" is because gsm8k always dominates worst_excess
and the noise floor on a single task with n=200 is ~0.01 — within which TA
and KnOTS are indistinguishable. The per-task wins on the other 3 tasks
might be real signal.

Action: include per-task error bars (bootstrap CI on the NLL_merged - NLL_τ
difference) in the headline figure. If KnOTS's wins are outside the CI
on multiple tasks, it's a real effect.

### O17.4 — DARE = TA exactly. Implementation bug or hyperparameter?

To 3 decimal places, DARE at density=0.2 produces identical worst_excess
to TA on both models. Either:
- The expected drop-rescale recovers TA in expectation, and our seed gave
  near-zero variance from the mean → DARE ≈ TA empirically.
- Our DARE implementation is buggy (e.g., applying the rescale wrong).
- Density=0.2 is the wrong knob — DARE papers usually report density in
  {0.5, 0.7, 0.9}.

Action: ablation. Run DARE at density ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}.
At density=1.0 it MUST equal TA exactly (no drop). At density=0.5, should
differ from TA. If it doesn't, suspect the implementation.

### O17.5 — Negative translation excess: under-trained LoRA or real cross-task benefit?

Every cell, both models: NLL_merged(translation) < NLL_τ_translation(translation).
The merge IMPROVES translation. Magnitude: -0.06 to -0.13 nats.

This is theoretically possible — task vectors might share useful structure
(grammar, vocabulary) that translation alone didn't learn fully from 7500
wmt19 examples. But it's also a red flag that the translation LoRA is
under-trained.

Action: retrain translation LoRA on 15000 examples × 3 epochs and re-eval.
If excess still negative → real cross-task benefit (paper finding). If
excess goes positive → original LoRA was just weak; report retrained
numbers.

### O17.6 — Rate-decay slope not testable at practical bit budgets

The bound predicts $\mathcal{D}(R) \geq \text{floor} + C \cdot 2^{-2R/d_{\mathrm{eff}}}$.
At b ∈ {1, 2, 4, 8, 16, 32} bits per parameter, the decay term is too small
to dominate worst_excess. The empirical slope is not -2; it's essentially 0.

This is OK for the paper (the floor part of the bound is what we validate),
but it would be nice to have something to say about the slope. Options:
- Find a sub-bit quantization scheme. 1bit-Merging exists but is concurrent.
- Look at the linear regime: maybe at b=0.5 (e.g., 1-bit per 2 parameters
  via product quantization) the slope becomes visible.
- Argue from the data that the floor-only regime is the physically relevant
  one for LoRA merging.

This is a paper-narrative question more than an experimental one. Surface
to Garg for discussion.

---

## Phase 3 Day 18 evening — Tier 1 strengthening committed

After the "what would make the paper stronger" discussion, three items were
committed to the pipeline. Numbering continues from T1.A onward.

### T1.A — Validate floor formula B²(1−d_eff/(Tr)) empirically

Script `code/phase3/eval/deff_analysis.py` (~330 lines) loads each of the 8
LoRA adapters, extracts V_t = top-r right-singular subspace per layer,
computes d_eff = rank(Σ_t P_{V_t}) per layer, predicts floor, compares to
observed worst_excess from the eval matrix. **Running as of 2026-05-18
evening on login-node CPU.** Reuses existing artifacts; no new compute.

Expected outcomes:
- **Perfect match** → headline figure: predicted-vs-observed-floor scatter
  validates the theorem quantitatively.
- **Match up to a multiplicative constant** → theorem structurally right;
  constant needs tightening (already a v2 item).
- **No match** → opens a substantial theoretical question (the qualitative
  shape still holds, but the formula doesn't).

### T1.B — Stiefel-random synthetic control panel

Script `code/phase3/eval/stiefel_control.py` (~260 lines) generates fake
LoRAs with controlled subspace overlap α ∈ {0, 0.1, ..., 1.0}, runs all
5 merging methods, plots:
- observed worst-excess vs predicted floor (line + y=x reference)
- bound-tightness ratio (observed / predicted) vs d_eff/(Tr)

Demonstrates the bound is TIGHT on adversarial Stiefel-random data and
LOOSE on real LoRAs (with the gap being the "real LoRAs are not worst-case"
slack). Standard for RD-bound papers. **Ready to run; CPU-only, ~5 min.**

### T1.C / T1.D — 3 more model families (Mistral-7B, Yi-9B, gemma-3-12b)

Downloads in flight on workq (job 39627, 24-hr walltime, doesn't compete
with gpu queue). After downloads complete:
- T1.C: train 4 LoRAs × 3 new models = 12 LoRAs (~3 hr at 3-concurrent gpu)
- T1.D: extend eval matrix by 15 non-TVQ + 18 TVQ rate cells = 33 cells
  (~9 hr at 3-concurrent gpu)

Converts the "Qwen vs Llama" (n=2 architecture) finding into a comparison
across 5 architecture families. Strengthens any model-specific claim.

### T1.E — regen Phase B with full data

After T1.A/B/D complete, update `code/phase3/eval/phase_b_analysis.py` to
include the new model families + d_eff predicted-floor scatter + Stiefel
tightness panel. New headline figure set goes into the paper.

---

These do NOT supersede the existing O17.1–O17.6 questions (those are still
open and tracked above). Tier 1 work targets paper strengthening; the
mechanism questions (e.g., b=2 dip) remain.

### T1.audit — Dataset reliability check + train/eval overlap bug — RESOLVED 2026-05-18

User asked to confirm Phase 3 datasets are reliable. **All 4 LoRA training
datasets verified real and authoritative**: openai/gsm8k (Cobbe+'21),
yahma/alpaca-cleaned, ise-uiuc/Magicoder-OSS-Instruct-75K, wmt/wmt19.
No synthetic data used in Phase 3 LoRA training. (Synthetic data is reserved
for theory validation in `code/synthetic/` and the Stiefel control panel in
`code/phase3/eval/stiefel_control.py` — both clearly labeled as synthetic.)

**Bug found during audit and fixed.** `data_loaders.py` used two independent
shuffle seeds for train vs eval when both came from the same split (alpaca,
magicoder). Resulted in 13% (alpaca) / 7% (magicoder) overlap between train
and eval. Fix: one shuffle, disjoint slices. Verified 0% overlap post-fix.

**Action item (for the v2 rerun):** rerun the 20-cell n=1k matrix with
fixed loader to `eval_matrix_n1k_v2/`. ~5 hr compute. Schedule after the
current n=1k matrix finishes.

---

## Phase 3 Day 19 — 2026-05-19 status & scope changes

### T1.B — Stiefel control panel — PARKED (uninformative as designed)

Ran 2026-05-19 (`results/phase3/stiefel_control.json` +
`code/phase3/figures/stiefel_control.png`). The mixing construction
`V_t = sqrt(1-α) V_indep + sqrt(α) V_shared` leaves the stacked subspaces
`[V_1 | ... | V_T]` generically linearly independent for **every α < 1**,
so `d_eff = Tr = 16` everywhere except α = 1.0 (where it collapses to
`d_eff = r = 4`). Predicted floor is 0 across the sweep; "observed /
predicted" ratio explodes to ~10¹². Outputs NOT promoted to
`paper_artifacts/`. Two paths forward (decision deferred to after v2):

(i) Redesign T1.B with explicit partial-shared-basis construction (share
    r' < r exact basis directions, keep r − r' independent per task,
    sweep r' → d_eff = r' + T(r − r')). Produces a real
    `predicted ∈ (0, Tr)` axis.
(ii) Pivot to T1.A2 soft d_eff (participation ratio of stacked-V
     singular values). Subsumes (i)'s purpose with a continuous metric
     that should correlate with the real-LoRA observed gap (F9).

**Recommendation:** (ii). Aligns with `log.md` §5.6 candidate (b) and
gives us a usable Stiefel panel via the soft metric in one move.

### T1.C — Trimmed from 16 LoRAs to 8 (2 models × 4 tasks)

**Dropped 2026-05-19:**

- **gemma-3-12b-it** — architecture is `Gemma3ForConditionalGeneration`
  (multimodal head). Unsloth's `FastLanguageModel.from_pretrained`
  is built for causal-LM heads; risk of load failure or partially-broken
  adapter not worth taking with shared-GPU contention.
- **Qwen-2.5-14B-Instruct** — realistic peak VRAM ~41 GB. With shared
  GPUs typically showing 14–46 GB free per device (and ~18 other STDIN
  sessions competing), 14B slot-finding is unreliable. Worth revisiting
  in a quieter window or with reservation rights.

**Retained:**

| Model | min_free_gb (revised) |
|---|---:|
| mistral_7b (Mistral-7B-Instruct-v0.3) | 30 |
| yi15_9b (Yi-1.5-9B-Chat) | 35 |

Final scope: **8 LoRAs** (4 tasks × 2 models). Reviewer story still
shows n=4 architecture families across the original 2 + new 2.

### T1.A2 (soft d_eff) — DEFERRED until v2 lands and Sankalp signs off

Per `log.md` §5.6 candidate (b). Now has independent motivation from
the T1.B finding above. Pure CPU, ~5–10 min when we run it.

### Standing rule: quote only v2 numbers going forward

For F1, F2, F7 magnitudes and any "0.10–0.22 nats/token" style number,
wait for `results/phase3/eval_matrix_n1k_v2/` to complete and re-run
`code/phase3/eval/phase_b_analysis.py` against it. *Directions* of
F1/F2/F7 are likely robust because `worst_task_excess` is
gsm8k-bottlenecked (gsm8k uses separate train/test splits, no bug
exposure). The 8 trained LoRAs and the d_eff = Tr structural finding
are clean of the data-loader bug.

---

## Model-class scope — out-of-scope families for the current paper

These were raised on 2026-05-19; logged here so reviewer-anticipated
"why didn't you test X" questions have a canonical answer.

### Q-SCOPE-1 — Reasoning-tuned models (R1-Distill, QwQ, Qwen2.5-Math, etc.)

**Status:** EXCLUDED from the 4-task comparison. Per `handoff.md` §6
don't-list and `notes/phase3_design.md`.

**Why exclude:**

1. **Output-length / CoT inflation.** Reasoning models emit
   `<think>...</think>` blocks or long CoT prefixes before the answer.
   Our `_compute_nll` in `code/phase3/training/train_lora.py` masks
   prompt tokens with `-100` and scores NLL over the answer region;
   with CoT, the "answer" region balloons with reasoning chains.
   Nats-per-token is no longer directly comparable across model
   families or even within-family pre/post LoRA.
2. **Pretraining-distribution overlap.** R1-Distill, QwQ, Qwen2.5-Math
   have been distilled on math benchmarks that include GSM8K-like
   problems. A 7500-example LoRA's marginal effect is drowned by the
   base's existing competence; the task-vector becomes tiny or noisy.
3. **Preference-tuning side effects.** RLHF / DPO patterns produce
   refusals or over-elaboration depending on prompt template; eval
   becomes template-sensitive in ways our 4 tasks aren't designed for.
4. **Scope drift.** The paper's thesis is about the rate-distortion
   structure of LoRA merging across architecture families with a
   shared frozen base. Adding reasoning models shifts the research
   question to "does the bound apply to reasoning models," which is a
   bigger methodology surface (separate NLL-stripping path, separate
   prompt templates, sample-efficiency re-check).

**Status decision for current paper:** keep excluded; add the
reviewer-anticipated paragraph to `paper/sections/experiments.tex` §
limitations:

> "Empirical validation focuses on instruction-tuned base models with
> standard causal-LM heads; reasoning-tuned variants (R1-Distill,
> QwQ family) are deferred because chain-of-thought outputs inflate
> the answer-token NLL metric and pretraining distributions overlap
> heavily with GSM8K. The theoretical bound is metric-agnostic over
> local-quadratic loss surfaces; future work could extend the
> empirical validation with a CoT-stripped per-token NLL or with
> answer-conditional cross-entropy."

**Future-work hook (out of scope for ICLR 2027 v1):** a follow-up
paper or v2 appendix could include 1–2 reasoning models with a
modified eval pipeline (CoT prefix detection + masking, longer
context window, separate template per family).

### Q-SCOPE-2 — Multimodal models (vision-language, audio-language)

**Status:** EXCLUDED from the current paper. Came up indirectly on
2026-05-19 when we discovered `gemma-3-12b-it` is
`Gemma3ForConditionalGeneration` (multimodal head) and dropped it
from T1.C for architecture-class risk.

**Why exclude:**

1. **Our 4 tasks are text-only** (GSM8K, Alpaca, Magicoder, wmt19
   en→de). No vision-language task in the matrix; adding multimodal
   models without multimodal tasks wastes their capability.
2. **Multiple LoRA-target conventions.** Multimodal architectures
   (LLaVA-style adapters, Qwen-VL, gemma-3-vision) have a vision
   encoder + language decoder + sometimes a projector. There's no
   canonical "where do you put the LoRA" — each paper picks
   differently (LM only, projector only, both). Our theory applies
   to any choice, but the experimental comparison needs a fixed
   convention which doesn't exist yet in the merging literature.
3. **NLL metric ambiguity.** Per-token NLL is well-defined for text
   tokens but ambiguous for vision-token embeddings (which aren't
   sampled from a discrete vocab in the same way). The metric our
   theorems use ($\widehat D(w) = \max_t \Delta L_t(w)$) needs
   re-derivation for the mixed-modality case.
4. **Different rate-distortion geometry.** The $d_{\mathrm{eff}}$
   structure for vision-attention layers may differ qualitatively
   from text-attention; an early scoping experiment could show this
   either way but requires multimodal tasks to test.

**Status decision for current paper:** keep excluded; add a
reviewer-anticipated paragraph similar to Q-SCOPE-1, framing
multimodal as a natural extension that requires its own
methodology pass.

**Future-work hook:** a dedicated multimodal-merging paper using the
same RD framework, with tasks like VQA + image-captioning +
visual-instruction + image-classification, and the choice of which
component(s) to LoRA-merge spelled out explicitly. The theory carries
over; the experimental design needs fresh thought.

### Standing reviewer-response template

For any "why didn't you test {family X}" question, the structure is:

1. State which family is excluded.
2. Cite the methodological reason (eval metric breaks, task overlap,
   scope drift, etc.) — not "we didn't have time."
3. Affirm the bound is family-agnostic in theory (the math doesn't
   care about reasoning vs not, or modality, beyond local-quadratic
   loss assumption).
4. Hook to future work.

This pattern goes in `paper/sections/experiments.tex` § limitations
and in the response-to-reviewers if invited.

---

## STANDING DECISION (2026-05-20) — Public transparency / decision log: build now, publish ONLY after double-blind review

**Decision:** We will maintain a public-facing "Reproducibility & Decision Log"
documenting all events and decisions (the data-loader bug + fix, model-track
choices and the gemma-3 / Qwen-14B drops with reasons, the v2 → v3 re-eval
lineage, honest negative findings like F10 soft-d_eff, the C2-SKIPPED admission,
etc.) for full transparency and research ethics. **BUT it must NOT be published
publicly until after the ICLR 2027 double-blind review concludes.**

**Why the timing guardrail:**
- The 2026-05-18 venue decision (`log.md` §10) is "NO arXiv preprint; work stays
  confidential through review." A public transparency doc describing this exact
  paper, attributable to Sankalp, would functionally act like a preprint.
- **ICLR 2027 is double-blind.** A public, attributable document during the
  review window risks de-anonymization (reviewers can search) and contradicts
  the confidentiality decision.

**The contemporaneous record already exists** and is the backing material:
`notes/daily_log.md` (timestamped, append-only), `notes/open_questions.md`,
`log.md`. No reconstruction needed — distill these into the public version
at the right time. Contemporaneous > reconstructed for credibility.

**Before ANY public release (camera-ready companion / post-decision):**
- Scrub `kittuwastaken@gmail.com` (HF/system email — never public). Paper
  contact is `pathaksankalp04@gmail.com`.
- Scrub HPC absolute paths (`/home/sanjay.g/...`) and cluster node names.
- Scrub author identity for the double-blind window.
- PDF only, never `.tex` / `.md` source (Rule N3).

**Form:** plan it as a "Reproducibility & Decision Log" appendix to the paper,
or a companion PDF released at camera-ready. Add a short "Reproducibility
statement" section during Tier 3 drafting that points at this log as the
backing record.

**Action gate:** do NOT publish, push to a public repo, post to AF/Reddit, or
attach to any external-facing channel until Sankalp confirms the double-blind
review window has closed. If in doubt, ask first.

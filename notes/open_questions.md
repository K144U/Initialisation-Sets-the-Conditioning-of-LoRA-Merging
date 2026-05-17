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

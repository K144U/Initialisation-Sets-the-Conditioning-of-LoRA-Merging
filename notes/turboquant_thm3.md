# TurboQuant Theorem 3 — annotated notes for merging transfer

**Source:** Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874 v1, §3.3 "Lower Bounds".

Read 2026-04-20 (Day 1 of Phase 0).

---

## 1. The theorem verbatim

> **Theorem 3 (lower bound on best achievable compression distortion).** For any randomized quantization algorithm $Q : S^{d-1} \to \{0,1\}^{bd}$ with bit-width $b$ and any reconstruction map $Q^{-1} : \{0,1\}^{bd} \to \mathbb{R}^d$, there exists a hard input instance $x \in S^{d-1}$ such that
>
> $$D_{\mathrm{mse}}(Q) := \mathbb{E}\bigl[\|x - Q^{-1}(Q(x))\|_2^2\bigr] \;\geq\; \frac{1}{4^b}.$$
>
> Furthermore, there exists a $y \in S^{d-1}$ such that
>
> $$D_{\mathrm{prod}}(Q) = \mathbb{E}\bigl[\bigl|\langle y, x\rangle - \langle y, Q^{-1}(Q(x))\rangle\bigr|^2\bigr] \;\geq\; \frac{1}{d} \cdot \frac{1}{4^b}.$$

## 2. The supporting lemma

> **Lemma 3 (Shannon LB on the sphere).** Let $x \in S^{d-1}$ be uniform on the unit hypersphere. Then for any bit budget $B \geq 0$,
>
> $$D(B) \;\geq\; 2^{-2B/d}.$$

Proof of Lemma 3 (reconstructed):

1. Shannon's classical lower bound: for a continuous source with differential entropy $h(x)$ and bit budget $B$, squared-error RD satisfies $D(B) \geq \tfrac{1}{2\pi e} \cdot 2^{2h(x)/d} \cdot 2^{-2B/d}$ (per-coordinate).
2. For uniform-on-sphere: $h(x) = \log_2 A_d$ where $A_d = 2\pi^{d/2}/\Gamma(d/2)$ is the surface area of $S^{d-1}$.
3. Stirling: $A_d \geq (2\pi e/d)^{d/2} \sqrt{2d/\pi} \cdot (1 - O(1/d))$.
4. Substitute: the $(2\pi e/d)^{d/2}$ in $A_d^{2/d}$ cancels the $1/(2\pi e)$ prefactor, leaving $D(B) \geq 2^{-2B/d}$ up to a $1-O(1/d)$ factor. ∎

Takeaway: **the hard source is uniform-on-sphere**, and its differential entropy is exactly large enough to wipe out the constants in Shannon's LB.

## 3. The proof of Theorem 3 (three-step distillation)

**Step 1 — Yao's minimax.**
> "The expected MSE of the optimal randomized compression algorithm for worst-case inputs equals the expected MSE of the optimal deterministic compression algorithm when applied to inputs drawn from a maximally difficult randomized distribution."

Formally: $\min_{Q_{\text{rand}}} \max_x \mathbb{E}\|x - \hat x\|^2 \;=\; \max_{\mu} \min_{Q_{\text{det}}} \mathbb{E}_{x \sim \mu}\|x - \hat x\|^2$. Standard game-theoretic equivalence.

**Step 2 — Pick the bad distribution: uniform on $S^{d-1}$.**
Rather than solving the $\max_\mu$, they just plug in a specific $\mu$ — uniform on sphere — and get a valid lower bound (since any $\mu$ gives a lower bound, and this one is tight up to constants).

**Step 3 — Apply Lemma 3.**
Bit budget is $bd$, so by Lemma 3, $D_{\text{mse}} \geq 2^{-2(bd)/d} = 2^{-2b} = 4^{-b}$. Done for MSE.

**Step 4 — Inner-product bound via pigeonhole.**
Decompose MSE as a sum of d per-coordinate errors: $D_{\text{mse}} = \sum_{j=1}^d \mathbb{E}|\langle e_j, x - \hat x\rangle|^2$. Since the sum is $\geq 4^{-b}$, at least one term is $\geq (1/d) \cdot 4^{-b}$. Take $y$ to be the standard basis vector along that coordinate.

## 4. Structural anatomy (for transfer purposes)

| Ingredient | Role | Abstraction level |
|---|---|---|
| Yao's minimax | Swap "worst-case input, randomized alg" ↔ "worst input distribution, deterministic alg". | Generic. Works for any loss, any space. |
| Choice of $\mu$ = uniform on sphere | Picks a source with high differential entropy relative to its diameter. | Setting-specific. Analogue is whatever "hardest natural distribution" is. |
| Shannon LB (Lemma 3) | $D(B) \geq \tfrac{1}{2\pi e} 2^{2h/d} 2^{-2B/d}$, then Stirling kills constants for sphere. | Generic source-coding lemma. Applies to any continuous source. |
| Pigeonhole for inner-product | Sum-of-coords ≥ $X$ ⟹ some coord ≥ $X/d$. | Trivial combinatorics. |

## 5. Constants and dependencies

- The bound $4^{-b}$ is for a **unit-norm** vector in $d$ dims with $bd$ bits. Equivalently: $D_{\text{mse}} \geq 2^{-2R/d}$ at total budget $R$.
- For a vector with $\|x\| \leq B$: scale to get $D \geq B^2 \cdot 2^{-2R/d}$.
- The achievability side (Theorems 1, 2) gives $D_{\text{mse}} \leq (\sqrt 3 \pi / 2) \cdot 4^{-b}$. Ratio $\sqrt 3 \pi / 2 \approx 2.72$. This is the celebrated "within 2.7× of Shannon" claim.
- There is **no $T$** (no multi-task concept) in the TurboQuant setting.

---

## 6. Transfer to LoRA merging — what carries, what doesn't

### 6.1 Direct mapping to the Day 4 setting

| TurboQuant | Merging (plan.md §2.1 Day 4) |
|---|---|
| Input $x \in S^{d-1}$ (single unit vector) | Input tuple $(\tau_1, \ldots, \tau_T) \in (B \cdot S^{d-1})^T$ (T bounded vectors) |
| Encoder-decoder $(Q, Q^{-1})$, output $\hat x \in \mathbb{R}^d$ | Encoder-decoder $(E, D)$, output single merged $w^\star \in \mathbb{R}^d$ |
| Bit budget $bd$ = $R$ bits total | Bit budget $R$ bits total |
| Distortion $\mathbb{E}\|x - \hat x\|^2$ | Distortion $\max_t \|w^\star - (w_0 + \tau_t)\|^2$ |

Two structural asymmetries jump out:

1. **$T$d-dim input → $d$-dim output.** We're collapsing a tuple into a single vector. This is fundamentally lossier than vector quantization — there is an **irreducible distortion** even at infinite rate, namely the Chebyshev radius of the set $\{w_0 + \tau_t\}_{t=1}^T$.
2. **Max, not sum.** TurboQuant's distortion is a sum/expectation over coordinates. Ours is a max over tasks. Different beast.

### 6.2 What transfers cleanly

1. **Yao's minimax (Step 1).** Works verbatim. For any game-tree-structured input/output/loss triple. → **transfers.**
2. **Pigeonhole as a glue step.** If we can lower-bound a sum of task-specific distortions, we get a max lower bound via pigeonhole. (This is the reverse of TurboQuant's use, but same mechanic.) → **transfers.**
3. **High-level strategy: pick a hard input distribution, invoke Shannon LB.** → **transfers in principle**, but the "right" distribution needs to be identified.

### 6.3 What needs new work

1. **Hard input distribution $\mu$ on tuples.** Natural candidate: $\tau_t \overset{\text{iid}}{\sim}$ uniform on $B \cdot S^{d-1}$. This is the product-of-spheres analog of TurboQuant's uniform-on-sphere.

2. **The Shannon LB analog for max-distortion, multi-source.**
   The classic Shannon LB bounds the RD function of a source under average squared-error distortion. We need:

   $$R(D) = \inf_{p(w^\star \mid \tau_{1:T})} I(\tau_{1:T}; w^\star) \quad \text{s.t.} \quad \max_t \mathbb{E}\|w^\star - (w_0+\tau_t)\|^2 \leq D.$$

   This is a **vector-sum source coding problem with a max-distortion constraint** — closer to multiple-description coding or common-reconstruction coding than classical RD. There may not be a clean single-source-LB analog; we may need to bound each task's RD separately and pigeonhole, rather than bound them jointly.

3. **Irreducible floor.** At $R = \infty$, max-distortion is lower-bounded by the expected Chebyshev radius of T iid samples from $B \cdot S^{d-1}$. For $T = 2$: $\mathbb{E}\|\tau_1 - \tau_2\|^2/4 = B^2/2$ at infinite rate. The RD bound should have the form

   $$D^\star(R) \;\geq\; \underbrace{\rho(T, d, B)}_{\text{Chebyshev floor}} + \underbrace{C \cdot B^2 \cdot 2^{-2R/d}}_{\text{Shannon-like term}}$$

   (rough guess — actual form depends on whether the two terms are additive or multiplicative). This shape is not in TurboQuant because their floor is zero.

4. **Distortion measure choice.** The plan.md §2.1 formulation $\Delta_t = f_t(w^\star) - \min_w f_t(w)$ reduces here to $\tfrac{1}{2}\|w^\star - (w_0+\tau_t)\|^2$ because $\min_w f_t = 0$. That is: the "excess" is vacuous in this quadratic setting. **Consider switching to "excess over Chebyshev optimum"**, i.e.,

   $$\widetilde\Delta(w^\star) \;=\; \max_t \|w^\star - (w_0+\tau_t)\|^2 \;-\; \min_{w'} \max_t \|w' - (w_0+\tau_t)\|^2.$$

   This excess → 0 at infinite rate, gives a clean RD statement, and factors the irreducible floor out of the bound. **Flag this for Day 4 decision.**

### 6.4 Packing construction sketch (Day 5 target)

Informal plan:

1. Build a packing $\{(\tau_1^{(i)}, \ldots, \tau_T^{(i)})\}_{i=1}^N$ of tuples such that for $i \neq j$, the **Chebyshev centers** of $\{\tau_t^{(i)}\}_t$ and $\{\tau_t^{(j)}\}_t$ are separated by at least $\delta$ in $\mathbb{R}^d$.
2. If $R < \log_2 N$ bits, by pigeonhole two tuples $(i,j)$ must map under $E$ to the same codeword, hence to the same $w^\star$.
3. Show that no single $w^\star$ can serve both Chebyshev-separated tuples with max-distortion $< \delta^2 / 4$.
4. Optimize $N$ vs. $\delta$ — larger packing ⟹ smaller $\delta$. Balance à la classical Fano.

Candidate concrete constructions to try:
- **Orthogonal-tuple packing:** pick tuples where each $\tau_t^{(i)}$ lies along a distinct axis, vary the chosen axes across $i$. Easy to analyze.
- **Random Gaussian packing:** sample $N$ tuples iid, show w.h.p. Chebyshev-centers are separated by $\Omega(B\sqrt{\log N / d})$. Converts to a volumetric argument.
- **Hadamard/BCH-coded packing:** for T = 2, vary the sign pattern of $\tau_1$ and $\tau_2$ along $\log_2 N$ independent axes.

### 6.5 Rank-r LoRA extension (Phase 1 concern, not Week 1)

Replace $\tau_t \in \mathbb{R}^d$ with $\tau_t = A_t B_t^\top$, rank $\leq r$, where $A_t \in \mathbb{R}^{d_1 \times r}$, $B_t \in \mathbb{R}^{d_2 \times r}$. Effective degrees of freedom: $r(d_1 + d_2) - r^2$ rather than $d_1 d_2$. Packing count drops accordingly. The Shannon LB needs to be redone for the rank-r manifold — likely gives a $2^{-2R/(r(d_1+d_2))}$ scaling instead of $2^{-2R/d}$.

---

## 7. Day 1 bottom line

- Theorem 3's proof is **three ingredients**: Yao minimax, a well-chosen hard distribution, and the Shannon LB on that distribution. Plus pigeonhole for the inner-product corollary.
- Steps 1 and "pigeonhole-as-glue" transfer to merging immediately.
- The **real work** is Step 2 (pick the right hard tuple distribution and the right distortion measure) and then re-proving a Shannon-LB analog for multi-source max-distortion.
- The toy theorem should probably use **excess-over-Chebyshev** distortion, not raw squared distance, to get a clean 0-at-infinite-rate statement.
- Constants: TurboQuant bound is $D \geq B^2 \cdot 2^{-2R/d}$ at total rate $R$ with $\|x\|\leq B$. Expect merging bound to look like $B^2 \cdot h(T) \cdot 2^{-2R/d}$ for some function $h(T)$ that grows in $T$ (more tasks → harder).

## 8. Open items raised today (added to open_questions.md)

- Should the toy theorem use raw max-distortion, or excess-over-Chebyshev?
- What is $h(T)$? Linear, log, or something else?
- For the packing, is the orthogonal-axes construction sharp, or does Gaussian random give a better constant?
- Does the multi-source Shannon LB we need already exist in the multiple-description coding literature (El Gamal & Cover 1982, Berger-Yeung)?

---

## 9. Day 1 numerical sanity check — findings (update)

Ran `code/synthetic/day1_distortion_measures.py`: quantized-mean merge,
T iid uniform-sphere task vectors, B=1, sweeps over T ∈ {2,4,8},
d ∈ {128, 512, 2048}, b ∈ {1, 2, 3, 4, 6, 8, 12}. 30 trials each,
median reported. Plots in `code/synthetic/figures/`.

### 9.1 Empirical plateaus match theory

For iid unit-sphere task vectors in high d, the mean is approximately
the Chebyshev center, and distances concentrate tightly:

| metric            | high-b plateau (observed)      | theory prediction   |
|-------------------|--------------------------------|---------------------|
| raw max           | 0.50, 0.76, 0.89 at T=2,4,8    | $1 - 1/T$           |
| sum               | 1.00, 3.00, 7.00 at T=2,4,8    | $T - 1$             |
| excess (vs floor) | $\to 0$                        | 0 (by construction) |

### 9.2 **Critical identity: sum reduces to single-vector quantization**

For any $w^\star$,
$$
  \sum_t \|w^\star - \tau_t\|^2 \;-\; \min_{w'} \sum_{t} \|w' - \tau_t\|^2
  \;=\; T \cdot \|w^\star - \bar\tau\|^2,
$$
where $\bar\tau = \tfrac{1}{T}\sum_t \tau_t$. This is an exact algebraic
identity (expand the squared norms). **Consequence:** under sum
distortion, the rate-distortion problem for merging is identical to
single-vector quantization of $\bar\tau$. TurboQuant Thm 3 applied to
$\bar\tau$ (norm bounded by some $B_\text{mean}$, possibly $\leq B$)
immediately gives
$$
  \sum_t \|w^\star - \tau_t\|^2 - \min_{w'}(\cdot) \;\geq\; T \cdot B_\text{mean}^2 \cdot 2^{-2R/d}.
$$
**This makes the sum-distortion theorem a corollary, not a novel result.**
The multi-task structure is absorbed into $\bar\tau$ and disappears.

### 9.3 **Max distortion has genuinely novel rate-distortion structure**

Empirically, the excess-max metric decays at ~$4^{-b}$ at low b but at
~$2^{-b}$ at high b. Explanation via triangle inequality:
$$
  \max_t \|w^\star - \tau_t\|^2 - \text{Cheb}^2
  \;\lesssim\; \|w^\star - \bar\tau\|^2
      \;+\; 2\,\|w^\star - \bar\tau\|\,\sqrt{\text{Cheb}^2}.
$$
- Low-b regime (quantization noise $\gg$ Chebyshev radius): quadratic
  term dominates; decay is $4^{-b}$.
- High-b regime (quantization noise $\ll$ Chebyshev radius): **linear
  cross term dominates; decay is $2^{-b}$.**

This two-regime structure does NOT reduce to TurboQuant Thm 3 because
$\sqrt{\text{Cheb}^2}$ is a data-dependent quantity that does not
appear in single-vector quantization. **The max-distortion theorem is
genuinely novel territory.**

Numerical check, T=2, d=2048:
- b=8:  excess = 0.014
- b=12: excess = 0.0009
- Ratio over 4 bits: 15.5.  $4^4 = 256$ (rejected), $2^4 = 16$ (consistent).

### 9.4 Recommended Day 4 theorem formulation

**Use max distortion**, likely in its excess-over-Chebyshev form for
clean "$D \to 0$ at $R = \infty$" aesthetics. Mention sum as an
appendix corollary.

Theorem target shape (conjectured):
$$
  \max_t \|w^\star - \tau_t\|^2 - \text{Cheb}^2
  \;\geq\; c_1 \cdot B^2 \cdot 2^{-2R/d}
  \;+\; c_2 \cdot B \cdot \sqrt{\text{Cheb}^2} \cdot 2^{-R/d}
$$
for some universal constants $c_1, c_2 > 0$. The **$2^{-R/d}$ term is
the interesting part** — it reflects the irreducible "half-rate"
penalty from the mismatch between scalar ($L_2$) merge errors and
$L_\infty$-over-tasks distortion.

### 9.5 Open items resolved by Day 1 experiment

- **Distortion measure**: max (or excess-over-Chebyshev). Sum is out —
  it's a TurboQuant corollary. — Close open_questions item.
- **What's novel**: the $2^{-R/d}$ linear regime has no TurboQuant
  analog. This is the paper's technical hook.

### 9.6 Items surfaced by the experiment (add to open_questions)

- How does $\text{Cheb}^2$ behave for rank-r LoRA task vectors (not
  iid sphere)? Phase 1 question.
- Does $\text{Cheb}^2$ shrink adversarially in the packing
  construction, making the linear-regime term zero at worst case?
  If so, the lower bound degenerates back to $4^{-R/d}$ and matches
  achievability.
- Can we construct a packing where Cheb² is large, forcing the
  linear-regime lower bound to bite?

# Ortiz-Jimenez et al. 2023 — Task Arithmetic in the Tangent Space

**Paper:** *Task Arithmetic in the Tangent Space: Improved Editing of
Pre-trained Models.* Ortiz-Jimenez, Favero, Frossard. NeurIPS 2023.
arXiv:2305.12827.

**Why we're reading it (Day 3 of Phase 0).** The Day 1 / Day 2
toy-theorem setting uses the geometric distortion
$\max_t \|w^\star - (w_0 + \tau_t)\|^2$. This paper is the canonical
reference that says: weight-space distance is the wrong quantity —
the right quantity is a **function-space** distance, mediated by a
**task-specific subspace** of weights. If true, it reshapes what a
distortion measure even *means* for our lower bound. Question we
want answered by end of day: does disentanglement give us a tighter
domain-specific distortion measure, or does it just reduce the
*effective dimension* inside the classical $\|\cdot\|^2$ bound?

## 1. Setup and notation

- Pretrained weights $\theta_0 \in \mathbb{R}^d$.
- Fine-tuned weights on task $t$: $\theta_t^\star$.
- Task vector: $\tau_t := \theta_t^\star - \theta_0$.
- Task arithmetic (Ilharco et al. 2023):
  $$\theta_{\text{merged}}(\alpha) = \theta_0 + \sum_{t=1}^T \alpha_t \tau_t.$$
- Each task $t$ has a data distribution $\mu_t$ on input space
  $\mathcal{X}$; its **support** is (roughly) a subset $\mathcal{D}_t
  \subseteq \mathcal{X}$.

## 2. Weight disentanglement (Definition 3.1, paraphrased)

A model $f:\mathcal{X}\times\Theta \to \mathcal{Y}$ is *weight
disentangled* with respect to task vectors $\{\tau_t\}$ and initial
weights $\theta_0$ if there exist functions $g_t$ and disjoint supports
$\mathcal{D}_t \subset \mathcal{X}$ such that
$$
f(x; \theta_0 + \sum_t \alpha_t \tau_t)
\;=\; \sum_{t=1}^T g_t(x; \alpha_t \tau_t)\,\mathbf{1}[x\in\mathcal{D}_t]
\;+\; g_0(x)\,\mathbf{1}[x\notin \bigcup_t \mathcal{D}_t].
$$
In words: when you merge, **each task's input region sees only its
own task vector's contribution**. The merge is "free" on task-$t$
inputs — no cross-talk.

### Disentanglement error

Quantitatively, the failure of this idealization is measured by
$$
\xi(\alpha_1,\dots,\alpha_T)
\;=\; \sum_{t=1}^T \mathbb{E}_{x\sim\mu_t}\!\left[
\mathrm{dist}\bigl(f(x;\theta_0+\alpha_t\tau_t),\,
f(x;\theta_0+\textstyle\sum_k \alpha_k\tau_k)\bigr)\right]. \tag{Eq. 2}
$$
$\xi = 0 \Leftrightarrow$ perfect disentanglement.

## 3. Tangent-space linearization

Define
$$
f_{\text{lin}}(x;\theta) := f(x;\theta_0) + (\theta-\theta_0)^\top
\nabla_\theta f(x;\theta_0).
$$
Fine-tuning $f_{\text{lin}}$ on each task is **kernel regression with
the empirical NTK** $K(x,x') = \nabla f(x;\theta_0)^\top \nabla
f(x';\theta_0)$.

**Main empirical claim.** Fine-tuning in the tangent space (rather
than the nonlinear model) **amplifies weight disentanglement** — the
$\xi$ of Eq. 2 shrinks. Task arithmetic (accuracy when evaluating
$\theta_0 + \sum_t \alpha_t \tau_t$) improves substantially.

**Mechanism claim.** NTK eigenfunctions have *spatial localization* —
each eigenfunction concentrates its support. Task vectors in the
NTK basis therefore already correspond to disjoint spatial supports,
which is exactly the disentanglement condition.

## 4. What this means for our distortion measure

Here's the translation to our RD setup.

**Our setup (Day 1–2 toy theorem).** Distortion
$D(w^\star; \tau_1,\dots,\tau_T) = \max_t \|w^\star - \tau_t\|^2$
(shifting $w_0 = 0$).

**Ortiz-Jimenez setup.** What actually matters for user-visible task
performance is the **function-space** loss
$$
\ell_t(\theta) = \mathbb{E}_{x\sim\mu_t}
\bigl[L\bigl(f(x;\theta),\,y_t(x)\bigr)\bigr].
$$
Under weight disentanglement, $\ell_t(\theta_0 + \sum_k \alpha_k
\tau_k) \approx \ell_t(\theta_0 + \alpha_t \tau_t)$ — task $t$'s loss
only sees its own component.

### The right distortion, under disentanglement

If disentanglement holds exactly, for any quantized $w^\star$ the
excess task-$t$ loss is
$$
\ell_t(w^\star) - \min_\theta \ell_t(\theta)
\;\approx\; \text{(function-space error on task-$t$ inputs only)}.
$$
Under local quadratic approximation of $\ell_t$ near $\tau_t$ (Hessian
$H_t$), this becomes
$$
\ell_t(w^\star) - \ell_t(\tau_t) \;\approx\; \tfrac{1}{2}
(w^\star - \tau_t)^\top H_t (w^\star - \tau_t).
$$
Crucially, **$H_t$ is low-rank under disentanglement**: $H_t$ is
supported on the subspace $V_t \subset \mathbb{R}^d$ of weight
directions that affect task-$t$ outputs. If task $t$ is a LoRA
fine-tune with rank $r$, then $\mathrm{rank}(H_t) = O(r)$ (not $d$).

So the "right" distortion is
$$
D_{\text{fn}}(w^\star) \;=\; \max_t\; (w^\star - \tau_t)^\top H_t
(w^\star - \tau_t),
$$
**not** $\max_t \|w^\star - \tau_t\|^2$.

### How this changes the RD bound

Three regimes to compare:

1. **Geometric (our Day 1–2 setup):** distortion $\max_t \|w^\star -
   \tau_t\|^2$, effective dim $d$. RD: $D(R) \gtrsim B^2 \cdot 2^{-2R/d}$
   at high rate.

2. **Disentangled, shared subspace:** all $H_t$ share a single
   subspace $V$ of dim $d' \leq d$ (common feature directions). RD:
   $D(R) \gtrsim B^2 \cdot 2^{-2R/d'}$. **Tighter** when $d' < d$,
   i.e., per-bit progress is faster on the relevant directions.

3. **Fully disentangled, disjoint subspaces:** each $H_t$ lives on its
   own $V_t$ of dim $r$ with $V_t \cap V_{t'} = \{0\}$. Merging is
   *free*: put $\tau_t$ directly into $V_t$, $w^\star = \sum_t \tau_t$
   achieves $D = 0$ with only $O(Tr \log(1/\epsilon))$ bits total.
   **No RD trade-off** — the theorem is trivial in this regime.

The interesting regime is **(2) + overlap**: $V_t$'s overlap with
overlap coefficient $\gamma \in [0,1]$. The Day-2 two-step proof
sketch (packing on mean + cross-term) should carry over with $d$
replaced by the effective rank of the union $V_1 + \dots + V_T$.

## 5. Implications for Phase 0 toy theorem (Day 4)

**Decision: keep the Day-2 geometric formulation for the toy
theorem.** Reasoning:

- The geometric $\max_t \|w^\star - \tau_t\|^2$ is the correct
  distortion **in the worst case over $H_t$** (take $H_t = I_d$). Any
  lower bound in the geometric setting is a lower bound in the
  disentangled setting with $V_t = \mathbb{R}^d$.
- The two-regime shape ($4^{-b}$ then $2^{-b}$, the novel piece)
  emerges from the geometry of the $\max_t$ over a point cloud —
  it is purely **distributional / combinatorial**, independent of the
  $H_t$ structure. Proving it in the clean geometric setting isolates
  the structural claim.
- Phase 1 (Week 2–3) can lift this to the $H_t$-weighted version as
  a "refined theorem" once the rank-$r$ LoRA effective dimension is
  plugged in (open question #1).

**Decision: note the $H_t$ reduction explicitly in the theorem
statement as a remark.** This pre-empts the obvious reviewer
objection ("why are you measuring weight-space distance when it's
function-space distance that matters?") without forcing us to prove
a function-space bound on Day 4.

## 6. Open questions added / refined

- **New OQ (to `open_questions.md`, §Day 2/3 follow-ups):** Is the
  overlap coefficient $\gamma$ (how much $V_t$'s share directions)
  the right knob? Can we formalize it as
  $\gamma = \|\tfrac{1}{T}\sum_t P_{V_t}\|_{\text{op}}$ or similar?
  This determines whether Phase 1 can produce a clean
  $d_{\text{eff}}(\gamma, T, r)$ statement.
- **Partial answer to Phase 1 OQ #1 (rank-$r$ effect on bound):**
  Weight disentanglement formalizes the intuition. Under
  disentanglement, the packing dimension is $\mathrm{rank}(\sum_t P_{V_t})
  \leq \min(Tr, d)$, not $d$. So rank-$r$ LoRA updates **do** change
  the bound — stronger per-bit decay in the "interesting dimensions",
  but potentially weaker overall if $Tr < d$ and the merge concentrates
  there.
- **Phase 1 OQ #4 (MSE → CE extension):** the paper's
  linearized/NTK framework is the natural bridge. Local quadratic
  approximation of CE near $\tau_t$ gives exactly the $H_t$-weighted
  distortion above. This is not a problem — it's the intended
  extension.

## 7. One-line takeaway

**Weight disentanglement is a statement about the Hessian structure
of the task-loss landscape, not the geometry of task vectors.** For
the Phase 0 toy theorem we keep the geometric setting (worst-case
$H_t = I$). For Phase 1 we refine to rank-$r$-supported $H_t$ via
the same packing machinery on the effective subspace.

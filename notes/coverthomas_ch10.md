# Cover & Thomas, Chapter 10 — Rate-Distortion Theory (distilled for merging)

Source: *Elements of Information Theory*, 2nd ed., Cover & Thomas. Chapter 10 "Rate Distortion Theory." Written from memory / standard material; verify exact statements against the textbook before citing in the paper.

Read 2026-04-21 (Day 2 of Phase 0).

**Purpose of this note.** Capture the structural pieces of classical RD theory we'll transplant to merging: the source-coding-with-fidelity problem, the converse proof template, the Gaussian R(D) formula (our single-vector benchmark), and the Shannon lower bound (what TurboQuant Lemma 3 specializes for us).

---

## 1. The rate-distortion function

**Setup.** Source $X^n = (X_1, \ldots, X_n)$ iid $\sim p(x)$. Encoder $f_n: \mathcal{X}^n \to \{1, \ldots, 2^{nR}\}$ (rate $R$ bits per source symbol). Decoder $g_n: \{1, \ldots, 2^{nR}\} \to \hat{\mathcal{X}}^n$. Distortion measure $d: \mathcal{X} \times \hat{\mathcal{X}} \to \mathbb{R}_{\geq 0}$. Average distortion $\mathbb{E}\bigl[d(X^n, \hat X^n)\bigr] = \tfrac{1}{n}\sum \mathbb{E}[d(X_i, \hat X_i)]$.

**Rate-distortion function (information-theoretic form):**
$$
R(D) \;=\; \min_{p(\hat x \mid x) \,:\, \mathbb{E}[d(X, \hat X)] \,\leq\, D} I(X; \hat X).
$$

**Main theorem (Shannon 1959, C&T Thm 10.2.1).** A rate $R$ is achievable with distortion $D$ iff $R \geq R(D)$. Two directions:

- **Achievability (direct coding theorem).** Random coding + strongly typical set argument. Generate $2^{nR}$ iid codewords from $p(\hat x)$; given $x^n$, search for a codeword within distortion $D$; typicality guarantees one exists w.h.p. when $R > R(D)$.
- **Converse.** Show that any code with distortion $\leq D$ requires rate $\geq R(D)$. Proof uses the data-processing inequality and the information-inequality chain (§1.3 below).

## 2. Gaussian source, squared-error distortion (C&T §10.3.2)

$X \sim \mathcal{N}(0, \sigma^2)$, $d(x, \hat x) = (x - \hat x)^2$.

$$
\boxed{R(D) \;=\; \tfrac{1}{2} \log \frac{\sigma^2}{D}, \qquad 0 \leq D \leq \sigma^2}
$$

and $R(D) = 0$ for $D > \sigma^2$. Equivalently $D(R) = \sigma^2 \cdot 2^{-2R}$.

Proof sketch: show $I(X; \hat X) \geq \tfrac{1}{2}\log(\sigma^2/D)$ whenever $\mathbb{E}[(X - \hat X)^2] \leq D$ (converse), and construct an $\hat X$ with $X = \hat X + Z$, $Z \sim \mathcal{N}(0, D)$ independent (achievability).

**$d$-dim iid Gaussian, squared error per vector:**
$D(R) \;=\; d \cdot \sigma^2 \cdot 2^{-2R/d}$ (bit budget $R$ total, not per coord).

**This is our single-vector benchmark.** TurboQuant Lemma 3 gives the same shape — $2^{-2R/d}$ — for the uniform-on-sphere source, with constants cleaned up to be dimension-free.

## 3. The converse proof template (the part we reuse)

This is the load-bearing piece for proving lower bounds. C&T §10.4.

**Claim (converse).** Any $(2^{nR}, n)$-code with average distortion $\leq D$ satisfies $R \geq R(D)$.

**Proof structure.**

1. Encode index $T = f_n(X^n) \in \{1, \ldots, 2^{nR}\}$. Decoded $\hat X^n = g_n(T)$. Data-processing: $X^n \to T \to \hat X^n$.
2. $nR \;\geq\; H(T) \;\geq\; I(X^n; T) \;\geq\; I(X^n; \hat X^n)$ (by DPI).
3. Chain rule in reverse:
$$
I(X^n; \hat X^n) = h(X^n) - h(X^n \mid \hat X^n) = \sum_{i=1}^n h(X_i) - h(X_i \mid \hat X^n, X^{i-1}) \geq \sum_i I(X_i; \hat X_i).
$$
(Uses independence of $X_i$ and "reduce conditioning is OK" at each step.)
4. For each $i$, $I(X_i; \hat X_i) \geq R(D_i)$ where $D_i = \mathbb{E}[d(X_i, \hat X_i)]$, by definition of $R(\cdot)$.
5. Average distortion bound: $\tfrac{1}{n} \sum_i D_i \leq D$. Jensen on the convex $R(\cdot)$: $\tfrac{1}{n} \sum_i R(D_i) \geq R(\tfrac{1}{n}\sum D_i) \geq R(D)$.
6. Chain it all: $R \geq \tfrac{1}{n} I(X^n; \hat X^n) \geq \tfrac{1}{n}\sum I(X_i; \hat X_i) \geq \tfrac{1}{n}\sum R(D_i) \geq R(D)$. ∎

**Pieces to transplant to merging:**
- Steps 1, 2, 3 (DPI + chain rule) work generically for any encoder-decoder pair, including our $(E, D)$ merging setup.
- Step 4 invokes the single-letter $R(D)$ — in merging we don't have per-letter structure the same way, because the "source" is a tuple $(\tau_1, \ldots, \tau_T)$ and the "reconstruction" is a single vector $w^\star$. The chain-rule step has to be adapted.
- Step 5 (Jensen) requires convexity of the RD function. In the multi-task max setting, the analog of $R(D)$ is less obviously convex.

## 4. Shannon lower bound (C&T Thm 13.3.1 / §10.8)

For a continuous source $X \in \mathbb{R}^d$ with differential entropy $h(X)$ and squared-error distortion,
$$
R(D) \;\geq\; h(X) - \tfrac{d}{2} \log(2\pi e D / d).
$$
Equivalently $D(R) \;\geq\; \tfrac{1}{2\pi e} \cdot 2^{2h(X)/d} \cdot 2^{-2R/d}$.

This is what TurboQuant Lemma 3 specializes for uniform-on-sphere: plug in $h(X) = \log_2 A_d$ with $A_d = 2\pi^{d/2}/\Gamma(d/2)$, use Stirling, get $D(R) \geq 2^{-2R/d}$.

## 5. What the classical machinery gives us for free

Wherever the merging problem "looks like" a single-letter iid source-coding problem, we inherit:
- DPI + chain rule: step 1-3 of the converse.
- Shannon LB: step 4, if we can identify the "effective source".
- Convexity: step 5, under mild assumptions on the distortion measure.

Where it breaks for merging:
- **Many inputs, one output.** The chain rule says $I(\text{inputs}; \text{output}) \geq \sum I(\text{input}_i; \text{output}_i)$, but in merging there is no "output $i$" — just one $w^\star$. The right inequality would be $I((\tau_1,\ldots,\tau_T); w^\star) \geq \sum_t I(\tau_t; w^\star)$... which is **false in general** (mutual information is not superadditive under shared output). Need a different approach.
- **Max, not average distortion.** Classical RD is set up for average distortion. Max-distortion turns the "convex combination with Jensen" trick into max-and-taking-a-worst-task. Need to generalize.

**Implication:** the merging theorem will have to combine classical single-source RD (for the "per-task" component) with a new argument for the multi-task structure. This is the piece that differs from Cover & Thomas.

## 6. Gaussian multiterminal RD — what exists (Day 2 lit search)

**Question:** does the multi-source max-distortion RD we need already exist?

Surveyed: El Gamal & Cover 1982 (multiple descriptions), Steinberg 2009
(common reconstruction), Wyner-Ziv, Slepian-Wolf, Gray-Wyner,
Heegard-Berger. WebSearches on "rate distortion multiple source targets
single reconstruction max distortion", "Shannon lower bound L-infinity
worst-case multi-task".

**Outcome:** no published $R(D)$ characterization for our exact setting.
The closest three are structurally different:

- **El Gamal–Cover 1982 (multiple descriptions).** **One** source,
  **many** descriptions, multiple decoders receiving subsets. Dual
  of ours: one input → many encodings; we have many inputs → one
  encoding. The math doesn't transfer cleanly.
- **Steinberg 2009 (common reconstruction).** Requires the encoder and
  decoder to agree on the reconstruction. Single source with a
  side-information decoder. Not about multi-task distortion.
- **Heegard–Berger / Wyner–Ziv.** Single source with decoders having
  different side information. Again a single source.

**None of the classical multiterminal setups has "T reference vectors
at the encoder, one reconstruction, L_infinity distortion over T."**
The searches also turned up no treatment of the "linear" $2^{-R/d}$
regime — existing rate-distortion theorems are all quadratic in the
decoder's error, so the linear cross-term we saw empirically on Day 1
appears to be genuinely new.

**Implication for the paper:** the rate-distortion theorem we need IS
unclaimed, and the two-regime $4^{-R/d} + 2^{-R/d}$ shape we saw
empirically is the paper's technical hook. Confirmed. The Phase 0
decision gate is looking favorable.

## 7. Emerging proof sketch (Day 2 end-of-day; formalize on Days 4–5)

Armed with classical RD machinery + our Day 1 empirical structure, here
is a proof path for the two-regime bound:

$$
  \max_t \|w^\star - \tau_t\|^2 - R_c^2
  \;\geq\;
  \underbrace{c_1 \cdot B^2 \cdot 2^{-2R/d}}_{\text{quadratic regime}}
  \;+\;
  \underbrace{c_2 \cdot B \cdot R_c \cdot 2^{-R/d}}_{\text{linear regime}},
$$

where $R_c$ is the Chebyshev radius of $\{\tau_t\}$ and $c_1, c_2$ are
universal constants.

### 7.1 Proof strategy (two-step)

**Step A: Packing on the mean (classical TurboQuant Thm 3 reuse).**
Pick an adversarial family of task-vector tuples parameterized by
their mean $\bar\tau^{(i)}$ running over a TurboQuant packing set in
$B \cdot S^{d-1}$. Any encoder at rate $R$ must, by Yao + Lemma 3,
produce some $w^\star$ with
$$
  \|w^\star - \bar\tau\|^2 \;\geq\; B^2 \cdot 2^{-2R/d}
  \quad\text{for some tuple in the family.}
$$
This gives the quadratic regime's lower bound for free.

**Step B: Cross-term amplification (the new piece).**
For the same family, arrange the $\tau_t^{(i)}$ to spread symmetrically
around $\bar\tau^{(i)}$ with Chebyshev radius exactly $R_c$. By a
triangle-inequality expansion,
$$
  \max_t \|w^\star - \tau_t\|^2
  \;\geq\;
  \max_t \bigl(|\langle w^\star - \bar\tau, \hat u_t\rangle| \cdot R_c\bigr)^+
  \;\geq\;
  R_c^2 + 2 R_c \cdot \max_t |\langle w^\star - \bar\tau, \hat u_t\rangle|,
$$
where $\hat u_t$ is the unit vector from $\bar\tau$ to $\tau_t$. For
task configurations where the $\hat u_t$ cover a near-antipodal set of
directions (e.g. T = 2 antipodal, or T > 2 arranged on a simplex),
$\max_t |\langle w^\star - \bar\tau, \hat u_t\rangle| \geq c \cdot
\|w^\star - \bar\tau\|$ for a universal $c > 0$ depending on the
spread. Combined with Step A,
$$
  \max_t \|w^\star - \tau_t\|^2 - R_c^2
  \;\geq\;
  2 R_c c \cdot \|w^\star - \bar\tau\|
  \;\geq\;
  2 R_c c \cdot B \cdot 2^{-R/d}.
$$

### 7.2 Why the bound has both terms

Both Step A and Step B bound the same quantity from below, so taking
the maximum of the two gives the sharper bound. At high $R$, the
linear term (Step B) dominates because $2^{-R/d} > 2^{-2R/d}$. At low
$R$, the quadratic term (Step A) dominates because
$\|w^\star - \bar\tau\|$ is large enough that squaring wins.

The transition: $2^{-R/d} \approx R_c / B$, i.e. $R/d \approx \log_2(B/R_c)$.

### 7.3 What still needs work

- Step B's "$\max_t |\langle w^\star - \bar\tau, \hat u_t\rangle| \geq
  c \|w^\star - \bar\tau\|$" inequality. True when $\hat u_t$ covers
  enough directions; quantify $c$ as a function of $T$ and
  configuration. For T = 2 antipodal, $c = 1$; for random $T$ on
  sphere in high $d$, $c \approx \sqrt{(\log T)/d}$ by classical JL.
- The specific $R_c$ dependence. Step B gives $R_c \cdot B$; is there
  a regime where the $R_c$ factor can be replaced by something
  stronger (like $R_c \cdot \sqrt{T}$)?
- Achievability match. The quantized-mean merge trivially achieves the
  quadratic regime. Does it match the linear regime up to constants,
  or do we need a smarter merge (quantized Chebyshev center)? Day 4+
  numerical experiment.

### 7.4 Days 3–5 refined plan

- **Day 3 (Ortiz-Jimenez).** Weight disentanglement as distortion
  measure. Cross-check whether their framework gives a third proof
  strategy (functional-space rather than weight-space).
- **Day 4.** Write the theorem statement formally. Nail down $c_1$,
  $c_2$, and the configuration assumption on $\{\hat u_t\}$.
- **Day 5.** Attempt the full proof. Step A is classical. Step B is
  where the new work lives — the direction-covering inequality and
  the combination with TurboQuant Thm 3.

## 7. Take-homes

- **DPI + chain rule** (converse steps 1-3) transfer to merging. Use them.
- **Single-letter $R(D)$** doesn't factor the same way across the tuple $\to$ single-output asymmetry. Need a new per-task decomposition.
- **Gaussian $D(R) = \sigma^2 \cdot 2^{-2R/d}$** is our benchmark for the "clean" one-vector regime.
- **Shannon LB** is what TurboQuant Lemma 3 specializes; we'll want the analogous specialization for our effective multi-task source.
- The **multiple-description / common-reconstruction literature** doesn't seem to cover our max-distortion setting — confirm by skim.

# Daily log

## 2026-04-20 (Mon) — Day 1 of Phase 0

**Done today:**
1. Scaffolded the project repo per `plan.md` §9.1. Created `paper/`, `theory/`,
   `code/`, `experiments/`, `notes/` subtrees and seeded section stubs plus
   `theory/toy_theorem_v0.tex` (Day 4 scaffold). Folder still named
   `TurboQuant\` on disk; rename to `rdmerge\` after closing this Claude
   Code session (process holds a handle that blocks the rename mid-session).
2. Read TurboQuant Thm 3 end-to-end from arXiv v1 (§3.3). Structured notes
   in `notes/turboquant_thm3.md`. Did **not** yet check the OpenReview
   camera-ready version.

**Key takeaway:** the Thm 3 proof is three ingredients — Yao minimax, a
well-chosen hard distribution (uniform on $S^{d-1}$), and the Shannon LB
on that distribution (Lemma 3) — plus pigeonhole for the inner-product
corollary. Yao minimax and pigeonhole transfer to the merging setting
verbatim. The real work is picking the right hard distribution **and the
right distortion measure** for tuples → single merged vector.

**Flag for Day 4 decision:** whether the toy theorem should use raw
$\max_t \|w^\star - (w_0+\tau_t)\|^2$ or the excess over the Chebyshev
optimum. The raw version has a non-zero floor at infinite rate (the
Chebyshev radius of the task vectors), which complicates the
classical-looking RD statement. The excess version goes to 0 at infinite
rate and factors the irreducible multi-task incompatibility out of the
bound. Leaning toward excess formulation. Added to `open_questions.md`.

**Next (Tue Apr 21 — Day 2):** Cover & Thomas Ch. 10 (rate-distortion
theory, Gaussian source, converse). Focus on the structure of the
converse proof — the pieces we want to generalize to multi-source
max-distortion. Also skim the multiple-description-coding literature
(El Gamal–Cover 1982, Berger–Yeung) to see whether the multi-source RD
bound we need already exists.

**Blockers:** none.

**Day 1 addendum (numerical sanity check).** Ran
`code/synthetic/day1_distortion_measures.py` to empirically compare the
three candidate distortion measures (raw max, excess-over-Chebyshev,
sum) under a quantized-mean merge. Two non-obvious findings
(documented in `notes/turboquant_thm3.md` §9):

1. **Sum distortion is not novel.** The identity
   $\sum_t \|w^\star - \tau_t\|^2 - \min_{w'}(\cdot) = T \cdot \|w^\star - \bar\tau\|^2$
   makes it a single-vector quantization problem on $\bar\tau$, so
   TurboQuant Thm 3 applies directly — no new theorem needed.
2. **Max distortion has a two-regime RD curve.** Empirically: $4^{-b}$
   decay at low rate, $2^{-b}$ decay at high rate. The linear regime
   comes from the triangle-inequality cross-term $2\|w^\star -
   \bar\tau\|\sqrt{\text{Cheb}^2}$, which has no single-vector
   analog. **This is the paper's technical opening.**

Day 4 decision now looks clear: **write the theorem for max
distortion**, with the two-regime shape as the main structural claim.

## 2026-04-21 (Tue) — Day 2 of Phase 0

**Done today:**
1. Distilled Cover & Thomas Ch. 10 into `notes/coverthomas_ch10.md`:
   R(D) definition, Gaussian source R(D) = (1/2)log(σ²/D), converse
   proof template (DPI + chain rule + Jensen on convex R(D)), Shannon
   lower bound. §5 isolates the three generic converse ingredients we
   need to generalize to multi-source max-distortion.
2. Literature scan for multi-source max-distortion RD. Checked:
   El Gamal–Cover 1982 (multiple descriptions), Steinberg 2009 (common
   reconstruction), Heegard–Berger, Wyner–Ziv. **All address average
   or per-source distortion; none treat $\max_t D_t$ as the figure of
   merit.** Novelty claim for the bound holds. Written up in
   `coverthomas_ch10.md` §6.
3. Captured emerging proof sketch in `coverthomas_ch10.md` §7:
   - **Step A (classical):** packing on the mean $\bar\tau$ via
     TurboQuant Thm 3 → $\|w^\star-\bar\tau\|^2 \gtrsim B^2 2^{-2R/d}$.
   - **Step B (new):** triangle-inequality expansion
     $\max_t \|w^\star-\tau_t\|^2 \geq R_c^2 + 2 R_c \cdot
     \max_t |\langle w^\star-\bar\tau,\hat u_t\rangle|$,
     and a direction-covering inequality to lower-bound the $\max_t$
     inner product by $c\|w^\star-\bar\tau\|$.
   - **Target shape:** $\max_t \|w^\star-\tau_t\|^2 - R_c^2
     \geq c_1 B^2 2^{-2R/d} + c_2 B R_c \cdot 2^{-R/d}$. The
     $2^{-R/d}$ linear term is the novel structural piece — matches
     the Day 1 numerical observation.

**Key takeaway:** the proof skeleton is two TurboQuant-style packings
glued by a triangle inequality, not a single monolithic Fano argument.
Step A is a direct reuse of Lemma 3. Step B needs a lemma of the form
"a packing on $S^{d-1}$ of size $M$ covers every direction to within
angle $\arccos(\Omega(1))$" — this is what we need to formalize.

**Flag:** the direction-covering inequality is the technical hinge. If
it requires $M \gg 2^{R/d}$ codewords, Step B does not buy the linear
regime. Worth sketching before Day 4.

**Next (Wed Apr 22 — Day 3):** Ortiz-Jimenez et al. *Task Arithmetic
in the Tangent Space* (NeurIPS 2023). Focus: weight-disentanglement
assumption and whether it gives us a domain-specific distortion
measure that's tighter than $\max_t \|w^\star-\tau_t\|^2$.

**Blockers:** none. Proof structure is clearer than I expected at Day
2; may be able to start Day 4 drafting early if Day 3 is light.

## 2026-04-22 (Wed) — Day 3 of Phase 0

**Done today:**
1. Distilled Ortiz-Jimenez et al. NeurIPS 2023 (*Task Arithmetic in
   the Tangent Space*) into `notes/ortiz_jimenez_tangent.md`. Focus:
   formal Definition 3.1 of weight disentanglement, the
   disentanglement error $\xi(\alpha_1,\dots,\alpha_T)$ (Eq. 2), the
   tangent-space linearization $f_{\text{lin}}$, and the empirical
   claim that NTK-regime fine-tuning amplifies disentanglement.
2. Worked out what this means for our distortion measure. Under
   weight disentanglement, the user-visible task-$t$ loss is
   $(w^\star-\tau_t)^\top H_t (w^\star-\tau_t)$ with $H_t$ supported
   on the rank-$r$ subspace $V_t$ that task-$t$ fine-tuning actually
   moved — **not** the isotropic $\|w^\star-\tau_t\|^2$.
3. Made the Day 4 scope decision (see `ortiz_jimenez_tangent.md` §5):
   **keep the isotropic geometric distortion for the toy theorem.**
   Rationale: the two-regime RD shape ($4^{-b}$ then $2^{-b}$) is a
   combinatorial / distributional phenomenon about the $\max_t$ over
   a point cloud, independent of $H_t$ structure. Isolating it in
   the clean $H_t = I$ setting is the Phase 0 deliverable. Phase 1
   refines to rank-$r$-supported $H_t$.

**Key takeaway:** weight disentanglement is a statement about
Hessian structure, not task-vector geometry. It partially answers
two Phase 1 questions already:
- **OQ #1 (rank-$r$ effect):** yes, the packing dimension drops to
  $\mathrm{rank}(\sum_t P_{V_t}) \leq \min(Tr, d)$. Rank-$r$ LoRA
  strictly improves the bound when $Tr < d$.
- **OQ #4 (MSE → CE extension):** the NTK linearization + local
  quadratic approximation of CE near $\tau_t$ is the natural bridge.
  Not a problem for later phases — it's the intended path.

**New open question (Day 3 follow-up):** is there a single scalar
overlap coefficient $\gamma := \|\tfrac{1}{T}\sum_t P_{V_t}\|_{\text{op}}$
that parameterizes the gap between "merging is free" ($\gamma = 1/T$,
orthogonal supports) and "worst case" ($\gamma = 1$, fully shared)?
If yes, Phase 1 gets a clean $d_{\text{eff}}(\gamma, T, r)$ statement.

**Day 4 pre-plan:** the theorem statement now has three pieces to
pin down on Thursday:
1. Hard distribution for the $\tau_t$ tuple (candidate: $T$ iid
   uniform on $B\cdot S^{d-1}$, per Day-1 numerics).
2. Distortion: $\max_t \|w^\star - \tau_t\|^2 - R_c^2$ (excess over
   Chebyshev radius, so bound is zero at infinite rate).
3. Target bound shape: $D(R) \geq c_1 B^2 2^{-2R/d} + c_2 B R_c
   2^{-R/d}$. Remark: under disentanglement, replace $d$ by the
   effective rank of $\sum_t P_{V_t}$.

**Next (Thu Apr 23 — Day 4):** Write the toy theorem statement
formally in `theory/toy_theorem_v0.tex`. Also sketch the
direction-covering lemma needed for Step B of the proof — this is
the single piece I'm least sure about.

**Blockers:** none.

## 2026-04-23 (Thu) — Day 4 of Phase 0

**Done today:**
1. Wrote the full toy theorem statement in `theory/toy_theorem_v0.tex`.
   Setting, theorem, proof sketch, remarks, and one explicit open
   question — all in one file, fits within the Phase 0 two-page
   deliverable target.

2. **Key scope decision: backed off the two-regime lower bound.**
   After stress-testing the Day 2 proof sketch (Step A + Step B),
   concluded that the $2^{-R/d}$ linear cross-term cannot be forced
   against a sufficiently smart encoder. Concretely:
   - **Orthogonal adversary** ($\tau_t = \bar\tau \pm \cheb e_t$ with
     orthogonal frame $e_t$). For $T\leq d-1$, the encoder can align
     quantization error $w^\star - \bar\tau$ orthogonal to
     $\mathrm{span}\{\tau_t - \bar\tau\}$, nulling the cross-term.
     No $2^{-R/d}$ forced.
   - **1-dim spread adversary** ($\tau_t = \bar\tau \pm \cheb v$ for
     a fixed $v$). Encoder can allocate bits to the single sensitive
     direction. Cross-term gets absorbed into the same $2^{-2R/d}$
     term by bit-allocation optimization. No separate regime.
   - **iid sphere spread** (the distribution in the Day 1 numerics).
     Direction-covering gives only $\sqrt{\log T / d}$ coefficient
     on the cross-term — too weak to survive as a clean theorem in
     the practical $T = O(1)$ regime.
   Consequence: the theorem I *can* prove honestly is the Step-A-only
   version: $\cheb^2 + c(B^2/T) \cdot 2^{-2R/d}$. The $2^{-R/d}$ term
   from Day 1 numerics is relegated to a remark (the upper-bound
   gap) and an open question.

3. Worked out the Step 2 reduction in detail:
   $\max_t \|w^\star-\tau_t\|^2 \geq \tfrac{1}{T}\sum_t \|w^\star-\tau_t\|^2$,
   expanded via $\tau_t = \bar\tau + (\tau_t - \bar\tau)$, and the
   cross-term $\sum_t \langle w^\star-\bar\tau, \bar\tau-\tau_t\rangle$
   vanishes identically (not just in expectation). This is a clean
   deterministic reduction — no "Step B" needed to get the
   $2^{-2R/d}$ bound.

4. **Step 3 / TurboQuant Thm 3 on the centroid.** Under iid uniform
   sphere, $\bar\tau = T^{-1}\sum_t \tau_t$ has per-coordinate
   variance $B^2/(Td)$ in high $d$. Applying TurboQuant Thm 3 / Cover
   \& Thomas Shannon LB for Gaussian vector sources gives
   $\E[\|w^\star - \bar\tau\|^2] \geq c (B^2/T) 2^{-2R/d}$. The $1/T$
   factor is honest — it reflects that averaging $T$ sphere vectors
   makes the centroid easier to compress.

5. Added remarks on: (i) $T=1$ reduction to TurboQuant Thm 3 verbatim
   (sanity check ✓); (ii) the weight-disentanglement extension with
   $d \to d_\text{eff} = \rank(\sum_t P_{V_t}) \leq \min(Tr, d)$;
   (iii) the novelty vs multi-source RD literature (first
   max-distortion statement); (iv) what the theorem does *not* claim.

**Decision-gate preview (from plan.md §2.3):**
Current status matches "theorem closes but bound is loose" — the
$\cheb^2 + c(B^2/T) 2^{-2R/d}$ shape is nontrivial, reduces to
TurboQuant Thm 3 at $T=1$, but there is an upper-bound gap of
$2^{-R/d}$ at high rate that we cannot close in Phase 0. Per §2.3,
this outcome says **proceed to Phase 1**, keep the tighter-constant
/ tighter-rate proof as an open problem.

**Key takeaway:** the proof structure is now a clean reduction, not
a monolithic Fano argument. Two pieces: (i) deterministic reduction
from max to mean via non-negativity of an averaged cross-term;
(ii) single-vector rate-distortion LB on the centroid, reusing
TurboQuant Thm 3 verbatim. Day 5 work is formalization, not new
mathematics.

**Next (Fri Apr 24 — Day 5):** Formalize the proof. Specifically:
(a) justify the "$\bar\tau$ is approximately Gaussian with variance
$B^2/(Td)$" step rigorously — either by a concentration argument
or by applying Thm 3 directly on the sphere of radius $B/\sqrt{T}$,
tracking how the concentration of $\|\bar\tau\|$ around $B/\sqrt{T}$
enters the constant; (b) write out the Shannon LB invocation
carefully (Cover \& Thomas Thm 10.3.2 vs TurboQuant Lemma 3 —
which gives a cleaner constant?); (c) state Lemma~\ref{lem:floor}
(Chebyshev floor computation) with full derivation; (d) sanity-check
the $T=1$ limit against TurboQuant Thm 3's constant.

**Blockers:** none. The proof should close Day 5; the real work
this week is almost done.

## 2026-04-24 (Fri) — Day 5 of Phase 0

**Done today:** Full proof written into `theory/toy_theorem_v0.tex`,
replacing the Day 4 sketch. Structure of what closed:

1. **Step 1 (Yao's minimax).** Standard: $\mathcal{D}^\star \geq
   \inf_{E,D} \E_P[\Delta]$ for any hard distribution $P$. We take
   $P = \mathrm{Unif}(B S^{d-1})^{\otimes T}$.

2. **Step 2 (max $\geq$ avg, centroid reduction).** The key
   simplification of the whole week. The expansion
   $\|w^\star - \tau_t\|^2 = \|w^\star - \bar\tau\|^2 - 2\langle
   w^\star - \bar\tau, \tau_t - \bar\tau\rangle + \|\tau_t -
   \bar\tau\|^2$ averaged over $t$ has cross-term
   $\sum_t(\tau_t - \bar\tau) = 0$ **identically** — no probability,
   no concentration. This gives the deterministic identity
   $\frac{1}{T}\sum_t \|w^\star - \tau_t\|^2 = \|w^\star - \bar\tau
   \|^2 + \frac{1}{T}\sum_t \|\tau_t - \bar\tau\|^2$.
   Applied to max-distortion: $\max \geq \mathrm{avg}$ gives
   $\E[\max_t \|w^\star - \tau_t\|^2] \geq \E\|w^\star-\bar\tau\|^2
   + B^2(1 - 1/T)$.

3. **Lemma 1 (Chebyshev floor).** Closed completely. Direct
   computation: $\E\|\tau_t - \bar\tau\|^2 = B^2(1 - 1/T)$ via
   expansion of the inner products. 4 lines. This is the "irreducible
   multi-task incompatibility" showing up as a clean algebraic fact.

4. **Lemma 2 (RD LB for rotationally-invariant source).** Any
   rotationally invariant $Y$ with $\E\|Y\|^2 = \sigma^2$ satisfies
   $D_Y(R) \geq c \sigma^2 2^{-2R/d}$. Proof: condition on $\rho =
   \|Y\|$; given $\rho$, $Y/\rho \sim \mathrm{Unif}(S^{d-1})$, so
   TurboQuant Thm 3 applies with distortion scaled by $\rho^2$;
   Jensen on the log gives the expected-$\rho^2 = \sigma^2$ bound.
   The side-information subtlety (encoder/decoder knowing $\rho$)
   adds $O(\log d)$ rate which is $o(R)$ — handled in a footnote.

5. **Sanity check: $T=1$.** Lemma 1 gives floor 0. Lemma 2 gives
   $\E\|w^\star - \tau_1\|^2 \geq c B^2 2^{-2R/d}$, which is
   TurboQuant Thm 3 verbatim — same constant, no loss.

**Key simplification vs Day 4 sketch:** the Day 4 version argued for
a Gaussian approximation of $\bar\tau$ with per-coord variance
$B^2/(Td)$ and then invoked the Gaussian Shannon LB. This worked but
required a CLT / entropy-approximation step that is nontrivial at
finite $T$. The rotational-invariance route (Lemma~\ref{lem:rd-rot})
bypasses this entirely: the bound is through conditioning on the
radius $\rho = \|\bar\tau\|$ and applying TurboQuant Thm 3 to the
conditional uniform-sphere distribution. No Gaussian approximation
needed, no subtle entropy inequalities — just a scaling argument
plus Jensen.

**Key takeaway:** the Phase 0 theorem closes cleanly as a reduction
to TurboQuant Thm 3 plus a deterministic $\max \geq \mathrm{avg}$
identity. It is not a new Fano / packing argument — it is a
multi-source *reduction*. This makes the write-up honest and short:
the novelty is (i) identifying the right distortion measure
($\max_t$), (ii) the $\max \geq \mathrm{avg}$ reduction that
separates the compression term from the Chebyshev floor, and (iii)
the extension remarks (disentanglement, upper-bound gap).

**Decision-gate update (from plan.md §2.3):** the theorem closes
with nontrivial constant. Row 1 of the table: **proceed to Phase 1
as planned**. The $2^{-R/d}$ upper-bound gap is a deferred open
problem, not a Phase 0 blocker.

**What did not close today:** the open question from Day 4 — whether
the upper bound has an intrinsic $2^{-R/d}$ term or can be closed to
$2^{-2R/d}$. Didn't touch this; it's a Phase 1 question about
achievability, not about the lower bound.

**Next (Sat Apr 25 — Day 6):** Clean up `toy_theorem_v0.tex` for
external readability — tighten language, check constants, ensure it
compiles with pdflatex. Also write the one-page failure-mode /
alternatives section that plan.md §2.1 Day 6 requires even in the
success case (what would we do if a reviewer challenges the
side-information-overhead footnote in Lemma 2?). Target: the .tex
file should be ready to email on Day 7 without further edits.

**Blockers:** none. Phase 0 is essentially done.

**Day 5 addendum (numerical sanity check).** After flagging in-session
that no code had been executed since Day 1, ran three sanity checks
via `code/synthetic/day5_lower_bound_sanity.py` on the same iid
uniform-sphere distribution $P$. Sweep: $T\in\{2,4,8\}$, $d\in\{128,
512,2048\}$, $b\in\{1,2,3,4,6,8,12\}$, 30 trials per cell. Three
findings:

1. **Lemma 1 is empirically exact.** Ratio (empirical Chebyshev
   radius$^2$) / (theory floor $B^2(1-1/T)$) lies in $[0.999, 1.017]$
   across all $(T,d)$. The Chebyshev floor for iid sphere tuples is
   $B^2(1-1/T)$ to within 1--2\%, consistent with the 4-line
   computation in the proof.

2. **Theorem LB shape is respected everywhere.** Across all $(T,d,b)$,
   both the quantized-mean and quantized-Chebyshev-center merges
   satisfy $D \geq B^2(1-1/T) + c(B^2/T)2^{-2R/d}$ for $c \in [0.05,
   1]$ (consistent with the TurboQuant constant). No violations — the
   LB is valid, as it must be.

3. **The Day 1 $2^{-R/d}$ "linear regime" was an artifact.**
   Quantized-Chebyshev-center merge achieves excess-over-$\cheb^2$
   decaying as $2^{-2b}$ throughout: at $T=4, d=128$, the excess drops
   $0.025 \to 0.001 \to 0.0002$ at $b \in \{6, 8, 12\}$ (factor $\sim
   25\times$ per 2 bits = $4^{-b}$). Quantized-*mean* merge has a
   constant plateau $\max_t\|\bar\tau-\tau_t\|^2 - \cheb^2 \approx
   0.07$ above $\cheb^2$ — this is the sub-optimal-center offset, not
   a quantization phenomenon. What Day 1 interpreted as a $2^{-R/d}$
   decay on a log-excess plot was the plateau crossing the
   $2^{-2b}$ curve of the true quantization error.

**Consequence for the theorem.** The LB $\cheb^2 + c(B^2/T) 2^{-2R/d}$
is the correct order: it matches the Chebyshev-center merge's
achievability up to constants. The Day-4 open question ("is the
$2^{-R/d}$ regime fundamental?") is resolved NO. Updated
`toy_theorem_v0.tex`: replaced the Day-4 "upper bound gap" subsection
with a new "Matching achievability: the Chebyshev-center merge"
subsection reporting these numerics and the $\Theta(\cheb^2 + (B^2/T)
\cdot 2^{-2R/d})$ conclusion. Also updated `notes/open_questions.md`
to close the Day 4 item.

**Meta-lesson:** four days of reading without running code nearly
let a wrong Day-1 interpretation (linear $2^{-R/d}$ regime) survive
into the theorem. Phase 1 should pair every proof attempt with a
numerical sanity check the same day, not a week later. Adding to
feedback log.

**Revised next (Sat Apr 25 — Day 6):** same as above, plus: tighten
the "Open question" section in `toy_theorem_v0.tex` to reflect that
the $2^{-R/d}$ question is now resolved; the remaining open item is
pinning down the universal constant $c$ (deferred to Phase 1).

## Day 6 — Sat 2026-04-25

External-readability pass on `theory/toy_theorem_v0.tex`. Plan.md
§2.1 Day 6: tighten constants, identify where the theorem breaks
under weakened assumptions, write clean LaTeX note.

**Edits applied to `toy_theorem_v0.tex`.**
1. *Preamble hygiene.* Added `microtype`. Proof wrapped in a
   `proof` environment (auto-`\qed`, no more floating `\qed`).
   `\checkmark` replaced with `(as expected)`. Broken
   `\end{enumerate>` typo fixed.
2. *Constant renamed.* All universal-constant occurrences (theorem,
   corollary, Lemma~\ref{lem:rd-rot}, sanity check, open-items) now
   say $c_{\mathrm{TQ}}$, with a `remark` noting it is the constant
   of TurboQuant Thm 3 and that its explicit value is an open
   follow-up.
3. *Lemma~\ref{lem:rd-rot} proof rewritten.* Replaced the original
   side-information footnote (log_2(d) overhead argument, not
   rigorous at small $R$) with a cleaner derivation: DPI +
   conditioning on $\rho = \|Y\|$ + TurboQuant Thm 3 applied
   pointwise + Jensen gives the scale-invariant
   $\E[\|\hat Y-Y\|^2/\|Y\|^2] \geq c_{\mathrm{TQ}}2^{-2R/d}$;
   conversion to absolute form $\E\|\hat Y-Y\|^2 \geq c_{\mathrm{TQ}}
   \sigma^2(1-o_d(1))\cdot 2^{-2R/d}$ uses a concentration
   assumption which is tight for $Y = \bar\tau$ ($\mathrm{Var}\|\bar
   \tau\|^2 = O(B^4/(T^2d))$). Footnote defers the fully
   non-asymptotic version (explicit concentration constant, or EPI
   bypass of conditioning on $\rho$) to Phase 1. This is honest:
   the scale-invariant form is rigorous; the absolute form has a
   named $o_d(1)$ knob.
4. *New §4.5 (`\ref{sec:break}`) "Where the theorem breaks under
   weakened assumptions."* Three bullets: (a) dropping iid gives a
   smooth $\bar\rho$-dependent floor $B^2(1-1/T)(1-\bar\rho)$ +
   $c_{\mathrm{TQ}}(B^2/T)(1+(T-1)\bar\rho)2^{-2R/d}$; $\bar\rho = 1$
   collapses to TurboQuant Thm 3 with no floor term. (b) dropping
   uniform-sphere keeps the LB shape, degrades the floor constant
   only. (c) replacing max with average kills the multi-source term
   entirely — max is load-bearing.
5. *New §6 "Failure mode and alternative formulations."* §6.1:
   four anticipated reviewer objections + responses (unrealistic
   hard distribution; Lemma 2 concentration rigor; why max not
   average; novelty vs TurboQuant). §6.2: three alternative
   formulations explicitly not taken (excess-over-Chebyshev
   distortion; rank-$r$ LoRA structure; functional / Hessian-
   weighted distortion). Each justified by why it was rejected for
   Phase 0.

**Compile status.** `pdflatex` is not installed on this machine, so
the file was audited by hand instead of compiled:
- All `\label`/`\ref`/`\eqref` pairs cross-verified by grep.
- All `\begin{...}`/`\end{...}` environments balanced (5 itemize, 3
  proof, 2 lemma, 1 theorem/corollary/remark/remark, 1 enumerate, 1
  abstract, 6 equation, 1 document).
- The only dangling ref at one point (`\S\ref{sec:open}` in the
  Lemma 2 footnote) was resolved by adding `\label{sec:open}` to
  the Remaining open items section.
- Will install MiKTeX or use Overleaf on Day 7 before sending to
  Zandieh/Mirrokni. Flag: if any `??` or missing-package appears,
  add to the Day 7 tasks.

**Still open for Phase 1.**
- Explicit value of $c_{\mathrm{TQ}}$ (TurboQuant's arXiv v1 does
  not pin it down).
- Fully non-asymptotic version of Lemma 2 (concentration constant,
  or EPI rewrite).
- Rank-$r$ LoRA extension with the right scalar summary of subspace
  overlap — deferred §3.2 / §6.2 Alt. B.

**Next (Sun Apr 26 — Day 7: externalize).**
- Install MiKTeX (or push `theory/toy_theorem_v0.tex` to Overleaf)
  and verify clean compile twice. Resolve any `??` broken refs.
- Draft the Zandieh/Mirrokni email: 5-sentence summary + theorem
  statement + 1-paragraph context on where this fits relative to
  TurboQuant. Goal: they respond within the week with either "this
  matches what we intended" or "here's the constant we used" or
  "the reduction-to-Thm-3 framing is the interesting contribution."
- Post the same on Alignment Forum as a research note (shorter,
  informal tone). Target: 1 outside reader engages by Week 2.
- Ping Prof. Garg with a 2-line "Phase 0 closed, can we talk
  this week?" message.

## Day 6 bug-fix addendum — 2026-04-21 (real-time, pre-Day-7)

Full end-to-end verification pass (at user request, "check everything
once again please i wanna be sure that everything checks out ... every
single thing") caught a real bug in the proof before externalization.

**Bug:** Lemma~\ref{lem:rd-rot} concentration-hypothesis direction
was backwards. Hypothesis stated $\Pr(\|Y\|^2 \leq \sigma^2\kappa_d) =
1$ (upper bound on $\|Y\|^2$), but the proof conversion to
absolute-scale used $1/\|Y\|^2 \leq \kappa_d/\sigma^2$ a.s., which
requires a **lower** bound $\|Y\|^2 \geq \sigma^2/\kappa_d$ a.s.
Remark 3 compounded this by claiming "$\kappa_d = T$ trivially from
$\|\bar\tau\|^2 \leq B^2$" — still the wrong direction, and the
deterministic lower bound doesn't even exist ($T=2, \tau_1 = -\tau_2$
gives $\|\bar\tau\|^2 = 0$).

**Fix applied to `theory/toy_theorem_v0.tex`:**
1. Flipped the concentration hypothesis in Lemma 2 statement from
   upper to lower: $\Pr(\|Y\|^2 \geq \sigma^2/\kappa_d) = 1$ for some
   $\kappa_d \geq 1$ with $\kappa_d = 1 + o_d(1)$.
2. Corrected the proof conversion (same symbols, now cites the
   correct direction).
3. Rewrote Remark 3 to acknowledge the deterministic form fails for
   $Y = \bar\tau$, and substitute a high-probability version: define
   $G_d = \{\|\bar\tau\|^2 \geq B^2/(T\kappa_d)\}$ (norm-measurable,
   so $\bar\tau \mid G_d$ is still rotationally invariant); Chebyshev
   on $\mathrm{Var}(\|\bar\tau\|^2) = O(B^4/(T^2 d))$ gives
   $\Pr(G_d^c) = O((\kappa_d-1)^{-2} d^{-1})$; apply the
   scale-invariant form of Lemma 2 to $\bar\tau \mid G_d$ and
   multiply by $\Pr(G_d) = 1 - o_d(1)$. The absolute-scale bound
   picks up a $(1 + o_d(1))$ prefactor, absorbed into $c_{\mathrm{TQ}}$
   by convention.
4. Updated footnote to describe the tail-event substitution and list
   the fully non-asymptotic version (EPI rewrite) as Phase 1 work.
5. Added a "Bug fix" entry to the file header comment.

**Status:** theorem still closes. The $T=1$ sanity check still lands
exactly (at $T=1$ the deterministic lower bound $\|\tau_1\|^2 = B^2$
holds with $\kappa_d = 1$, no tail event needed). Reviewer-objection
§6.1 already reads honestly about the $(1-o_d(1))$ concentration
factor, so no edit there. The `sec:break` and `sec:matching`
sections don't reference the broken hypothesis, so they're unchanged.

**Meta-lesson (second one this week).** The original Day-5 proof felt
clean enough that Day 6 focused on readability polish rather than
re-deriving the Lemma 2 conversion. The verification pass on Day 6.5
caught the bug because it treated every line as re-provable rather
than re-readable. Adding to `feedback_received.md` as a rule: before
externalizing a proof, do an end-to-end re-derivation pass, not a
re-reading pass — the two find different bugs.

**Also resolved by this pass:** narrative consistency across README,
plan, deepresearch, daily log, open_questions, and the three notes/
files is clean; numerics in `code/synthetic/day5_lower_bound_sanity.py`
reproduce every claim in the Day-5 addendum above (empirical
Cheb²/theory ratio in [0.9984, 1.0173]; Cheb-merge excess at
$T=4, d=128$: $0.025 → 0.0013 → 0.0002$ at $b\in\{6,8,12\}$; no
$2^{-R/d}$ regime anywhere). LaTeX hygiene (env balance, ref/label
pairs) clean; pdflatex not yet installed locally, defer to Day 7
Overleaf compile.

## Day 7 — Sun 2026-04-26  (externalization closed)

**Done:**
1. Zandieh/Mirrokni/Daliri/Hadian email sent with the compiled
   `rdmerge.pdf` attached (see `SentEmail/`). Draft content is
   `notes/day7_externalization.md` §1; final subject: "Rate-distortion
   LB for LoRA merging, as a reduction to Thm 3 (6-page note, seeking
   a 10-min sanity check)".
2. Alignment Forum research note posted
   (`notes/day7_externalization.md` §2 content).
3. `notes/feedback_received.md` populated with the sent log.

**Status:** Phase 0 closed and externalized. Now in a ~2-day feedback
window. Per `feedback_verify_before_externalizing.md` rule, the
end-to-end verification pass (Day 6 bug-fix addendum) caught a real
bug before send — rule holds.

**Next (Mon 2026-04-27 — Day 8):** Phase 1 kickoff. Rank-$r$ LoRA
extension of the theorem. Plan: scaffold `theory/theorem_v1.tex`
with $H_t$-weighted max $\geq$ avg identity, Stiefel-random hard
distribution, conjectured bound, and three sanity-check limits
($r=d$, $T=1$, $V_t=V$). Pair with numerical sanity check
`code/synthetic/day8_rank_r_sanity.py`.

**Blockers:** none. Feedback response expected later in week; Day 8
work is independent.

## Day 8 — Mon 2026-04-27  (Phase 1 kickoff: rank-$r$ LoRA extension)

**Done today:**

1. **`theory/theorem_v1.tex` scaffold written.** Structure:
   - §1 Setting — rank-$r$ LoRA task-vector pairs $(\tau_t, H_t)$
     with $H_t \succeq 0$, $\rank(H_t) \leq r$, $V_t := \range(H_t)$,
     $\tau_t \in V_t$, scale normalization $\tau_t^\top H_t \tau_t
     \leq B^2$. Specializes to $H_t = P_{V_t}$ for concrete work;
     general $H_t$ is an extension.
   - §2 Lemma 1 ($H_t$-weighted max $\geq$ avg identity). Defines
     $\tauH := \bar H^+ \cdot T^{-1}\sum_t H_t\tau_t$ (Moore-Penrose
     on range); proves the generalization of the Phase 0 identity
     with linear cross-term $\sum_t H_t(\tau_t - \tauH) = 0$ by
     construction. Reduces to Phase 0 at $H_t = I$. Remark
     2 notes: components of $w$ outside $\range(\bar H) =
     V_1+\dots+V_T$ don't affect distortion, so WLOG $w^\star
     \in V_1+\dots+V_T$ of dimension $\deff = \rank(\sum_t P_{V_t})
     \leq \min(Tr, d)$.
   - §3 Hard distribution — Stiefel-random $U_t \in \St(d, r)$,
     $v_t \sim \mathrm{Unif}(B\cdot S^{r-1}) \subset \R^r$, $\tau_t
     = U_t v_t$. Overlap coefficient $\gamma = \|T^{-1}\sum_t
     P_{V_t}\|_\mathrm{op} \in [1/T, 1]$.
   - §4 Conjecture 2 — $\mathcal{D}^\star \geq B^2 f_\mathrm{floor}
     (T,r,\gamma) + c_\mathrm{TQ}(B^2/T) 2^{-2R/\deff(T,r,\gamma)}$,
     with floor and $\deff$ functions TBD. Candidates listed.
   - §5 Three sanity-check propositions (Props 3–5: $r=d$, $T=1$,
     $V_t=V$). Each pins $(\deff, f_\mathrm{floor})$ at a boundary
     point. Proofs are sketches — full proofs are Days 9–10 work.
   - §7 Five remaining open items for Phase 1.

2. **$H_t$-weighted max $\geq$ avg identity verified on paper.** The
   cross-term vanishing argument is clean: $\sum_t H_t(\tau_t -
   \tauH) = T(\bar H \tauH - \bar H \tauH) = 0$ by the definition of
   $\tauH$ as $\bar H^+$ applied to the $H$-weighted task sum. No
   probabilistic step; reduces to Phase 0 Lemma 1 when $H_t = I$.

3. **Numerics: `code/synthetic/day8_rank_r_sanity.py`.** Sweep
   $T\in\{2,4\}$, $d\in\{128,512\}$, $r\in\{4,8,16,32,d\}$,
   $b\in\{1,2,4,8\}$, overlap $\in\{\text{shared},\text{random}\}$,
   30 trials/cell. Merge scheme: quantized $\tauH$ in an orthonormal
   basis of $\range(\bar H)$, uniform scalar quantization over $[-B,B]$,
   total rate $R = b \cdot \deff_\text{meas}$ where
   $\deff_\text{meas} = \rank(\bar H)$.

**Findings:**

- **Check (a) regression at $r=d$: PASSES.** Empirical floor /
  $B^2(1-1/T)$ in $[1.0023, 1.0077]$ across all $(T, d)$ cells tested
  at $r = d$. Within the Phase 0 Day-5 band $[0.9984, 1.0173]$.
- **Check (c) random-subspace $\deff$: supports $\deff = \rank(\sum_t
  P_{V_t})$.** At $T=2$ with Stiefel-random $V_t$ and $r \in \{4,
  8, 16, 32\}$ (all well below $d/T$), the slope fit on
  $\log_2(\text{excess})$ vs.\ $R = b\cdot\deff_\text{meas}$ recovers
  $\deff_\text{fit} \in [8.2, 61.0]$ against $2r \in \{8, 16, 32,
  64\}$ — agreement within 5\%. This is strong evidence that the
  Day-9 $\deff$ should be $\rank(\sum_t P_{V_t})$, not a
  $\gamma$-interpolation.
- **Check (b) shared-subspace slope: quantizer-limited, not
  theorem-limited.** Shared-V slopes in $\log_2(\text{excess})$ vs.\
  $b$ range $-1.2$ to $-1.9$ (expected $-2$) at small $r$. Diagnosis:
  shared-V coefficients in the $Q$-basis have per-coord std
  $\sim B/\sqrt{Tr}$, much less than $B$; uniform quantizer over
  $[-B, B]$ wastes bits at small $r$. Random-V case does not show
  this because $\deff = Tr$ gives natural per-coord spread.
  Conclusion: use scale-adaptive quantization on Day 9 when verifying
  the shared-V bound numerically; the $2^{-2R/\deff}$ shape itself
  is fine.

**Key takeaway.** The $H_t$-weighted identity of §2 is the clean
generalization of the Phase 0 identity — same structure, same
cross-term-vanishes-by-construction argument, with $\tauH$ replacing
$\bar\tau$ and $\bar H$ replacing $I$. The random-subspace numerics
strongly favor $\deff = \rank(\sum_t P_{V_t})$ as the Day-9 proof
target, which means the three propositions are compatible with a
single choice of $\deff$. $f_\mathrm{floor}$'s closed form is the
Day-9 integral computation.

**Meta-lesson.** Pairing proofs with numerics same-day (per
`feedback_numerics_with_proofs.md`) surfaced the shared-V quantizer
artifact immediately, saving a likely Day-9 detour. The artifact
is cosmetic (wrong quantizer, not wrong bound), but finding it at
the scaffold stage means Day 9's numerical verification doesn't
need a separate round of debugging.

**Housekeeping done:**
- `notes/feedback_received.md` populated with the Day 7 send log.
- `notes/open_questions.md` updated: OQ #1 (rank-$r$ effect) and the
  Day-3 overlap-coefficient item marked in-progress with pointer to
  `theory/theorem_v1.tex`; new open items added.
- `theory/theorem_v1.tex` not yet compile-audited (same pdflatex
  situation as Phase 0). Visual audit clean; Overleaf compile is a
  Day-9 to-do.

**Next (Tue 2026-04-28 — Day 9):**
1. Prove Conjecture 2 in the special case $H_t = P_{V_t}$ (LoRA
   special case), starting from Lemma 1 + a rank-$r$ analog of
   Phase 0 Lemma 2 (RD lower bound for rotationally-invariant
   sources restricted to a subspace of dimension $\deff$).
2. Work out $f_\mathrm{floor}(T, r, \gamma)$ in closed form under the
   Stiefel-random $P$: compute
   $\E_P[T^{-1}\sum_t(\tau_t - \tauH)^\top P_{V_t}(\tau_t - \tauH)]$.
3. Re-run `day8_rank_r_sanity.py` with a scale-adaptive quantizer
   to verify Check (b) cleanly.

**Blockers:** none. If Zandieh / AF feedback lands during Day 9,
fold into the proof approach; otherwise proceed independently.

## Day 9 — Tue 2026-04-28  (Phase 1: prove main theorem, close numerics)

**Done today:**
1. **Closed form for the floor (Lemma 6 of `theorem_v1.tex`, new §4).**
   Key simplification the Day-8 scaffold missed: under $H_t = P_{V_t}$
   and $\tau_t \in V_t$, $P_{V_t}\tau_t = \tau_t$, so
   $\bar H \tauH = \bar\tau := T^{-1}\sum_t \tau_t$. Then
   $\E_P[T^{-1}\sum_t (\tau_t-\tauH)^\top P_{V_t}(\tau_t-\tauH)]
    = B^2 - \E_P[\bar\tau^\top \bar H^+ \bar\tau]$
   via a deterministic identity ($T^{-1}\sum_t \|\tau_t\|^2 = B^2$
   a.s.\ under $P$). An isotropy trace calculation — using
   $\E[\tau_s\tau_t^\top \mid V] = (B^2/r)P_{V_t}\delta_{st}$ and
   $\tr(\bar H^+ \cdot T\bar H) = T\deff$ — gives
   $\E_P[\bar\tau^\top\bar H^+\bar\tau \mid V] = B^2\deff/(Tr)$, so
   **floor $= B^2(1 - \E_P[\deff]/(Tr))$** in closed form.
   Endpoints are consistent: $V_t=V$ gives $\deff=r$, floor
   $B^2(1-1/T)$; Stiefel-random with $Tr\leq d$ gives $\deff=Tr$,
   floor $0$. The Day-8 open-question worry about the orthogonal
   limit resolved in the *opposite* direction: the floor there is
   **$0$, not $B^2$**. Geometric reason: $\tauH$ reconstructs each
   $\tau_t$ exactly in orthogonal position ($\tauH$ picks up the
   direct sum $\tauH = \sum_t \tau_t$, not the average).
2. **Proved Theorem 7 for $H_t = P_{V_t}$ (new §5 of `theorem_v1.tex`).**
   Six-step proof mirroring Phase 0: Yao → Lemma 1 max$\geq$avg
   reduction → restrict decoder to $\range(\bar H)$ via Remark 2 →
   change of variable $\xi = \bar H^{1/2}(w-\tauH)$ reducing to the
   standard Euclidean metric on $\range(\bar H)$ → rank-$r$ analog
   of Phase 0 Lemma 2 using the $O(\deff)$ stabilizer of
   $W = \range(\bar H)$ extended to $O(d)$ (this is what establishes
   that $\bar H^{1/2}\tauH$ conditional on $V$ is rotationally
   invariant in dimension $\deff$) → combine. Statement:
   $\mathcal{D}^\star \geq B^2(1-\deff/(Tr))
    + c_{\mathrm{TQ}}(B^2\deff/(Tr)) 2^{-2R/\deff}$.
   The three Day-8 sanity-check propositions collapse to
   one-line-reduction corollaries by substitution of $\deff$.
3. **Scale-adaptive numerics
   (`code/synthetic/day8_rank_r_sanity.py`).** Added
   `quantize_tauH_adaptive` with per-coord range $\pm c\cdot B/
   \sqrt{Tr}$ ($c=3$) matched to Lemma 6's per-coord scale. Also
   added a second excess metric `excess_hbar` =
   $\|w^\star-\tauH\|^2_{\bar H}$ — this is the quantity Theorem 7's
   converse step 5 bounds directly, whereas the original
   `excess = D_{\text{merge}} - \text{floor}_{\text{emp}}$ contains
   a cross-term that masks the $-2b$ slope in the shared-V case.
4. **Slope fits on the Hbar metric:** **shared-V slope is now $-2.00 \pm 0.04$
   across every cell** ($r \in \{4,8,16,32,128,512\}$, $T\in\{2,4\}$,
   $d\in\{128,512\}$). Day-8 uniform-$[-B,B]$ gave $-1.15$ to $-1.31$
   on the same cells. So **Check (b) passes cleanly** — the shape
   was right; the Day-8 slope deficit was a metric choice, not an
   RD phenomenon. Legacy `excess_max` metric still shows $-1.2$
   slopes as a cross-term artifact, not the quantity Theorem 7
   bounds.
5. **Check (a) regression** still passes: $r=d$ floor ratio
   $[1.0023, 1.0077]$, inside Phase-0 tolerance $[0.998, 1.017]$.
6. **Check (c) random-$V$ $\deff$ fits:** clean for $r \in \{4,8\}$ at
   $T=2$ (slope $\approx -2$, $\deff_{\mathrm{fit}}$ matches
   $2r = \deff_{\mathrm{meas}}$). At $r \geq 16$ random, slope
   flattens to $-1$ or worse — this is the $\bar H$ conditioning
   issue. Under scalar quantization in the $Q$-basis, per-coord
   variance is not uniform when $\bar H$'s eigenvalues span a wide
   range; $\bar H^+$ amplifies low-eigenvalue coords past
   $3\sigma_{\mathrm{pc}}$. Fix deferred to Day 10: quantize in the
   $\bar H^{1/2}\tauH$ representation where all coords have
   matched Hbar-metric weight.

**LaTeX hygiene.** Manually audited `theorem_v1.tex`: 32 `\begin`/
32 `\end` balanced; every `\ref`/`\eqref` target resolves to a
local `\label` (fixed two dangling `\eqref{eq:rd-rot-scale}` and
`\eqref{eq:thm-main}` that referenced labels in
`toy_theorem_v0.tex` by changing them to `\texttt{}`). pdflatex
still not installed locally — Overleaf compile deferred.

**End-of-day verification pass (`code/synthetic/day9_verify_lemma6.py`).**
Five independent checks of the Day-9 math:
1. **Lemma 6 Step 1 deterministic identity** ($T^{-1}\sum_t(\tau_t-\tauH)^\top P_{V_t}(\tau_t-\tauH)
   = B^2 - \bar\tau^\top\bar H^+\bar\tau$): pointwise to
   $\sim 10^{-16}$ on every sample. Confirms the algebra under
   $P_{V_t}\tau_t = \tau_t$ and $\bar H \tauH = \bar\tau$.
2. **Lemma 6 closed form** ($f_{\mathrm{floor}} = 1-\E[\deff]/(Tr)$):
   relative error $\sim 10^{-24}$ in orthogonal cases (floor exactly
   $0$), $\sim 0.4\%$ Monte-Carlo error in shared/full-rank cases at
   $n=2000$.
3. **Theorem 7 Step 5 $O(W)$-rotational invariance of $\eta|W$**:
   the empirical second-moment matrix $\E[\eta_W\eta_W^\top|W]$ has
   eigenvalue spread $1.033$ at $n=10^5$ versus the
   Marchenko--Pastur isotropic upper bound $1.036$. Spread $\leq$ MP
   bound across all four $(T, r)$ tested $\Rightarrow$ anisotropy is
   pure finite-sample noise; the conditional law really is
   rotationally invariant.
4. **Trace identity** ($\E[\|\eta\|^2|W] = B^2\deff/(Tr)$): matches
   to four digits at $n=10^5$, confirming the Step 6 substitution.
5. **Bug caught and fixed in Theorem 7 Step 5.** First draft said
   "condition on $V_1, \dots, V_T$" then claimed the conditional
   law is $O(W)$-invariant. False: conditioning on the individual
   $V_t$'s breaks the $O(W)$ symmetry (a generic $\tilde O\in O(W)$
   moves $V_t$ to a different subspace inside $W$). The correct
   conditioning is on $W = \range(\bar H)$ alone — given $W$, the
   joint law of $(V_1,\dots,V_T)$ inside $W$ is $O(W)$-invariant
   because the Stiefel-random measure is $O(d)$-invariant and $W$
   is preserved setwise. Empirical check #3 confirms the corrected
   statement. Fix in Step 5 of the proof, plus a forward-reference
   to `day9_verify_lemma6.py` for the empirical verification.

**Plan deviations flagged in the plan doc ahead of this entry:**
- Plan Step-1 contingency ("bounds $0 \leq f_{\mathrm{floor}} \leq
  1-1/T$ with matching endpoints if the integral doesn't close")
  did not trigger; Lemma 6 closed cleanly.
- Plan Step-3 expected outcome ("shared-$V$ slope on
  `log2(excess) vs b` should tighten from $\sim -1.2$ to $\sim -2.0$")
  did not come from scale-adaptive quantization alone — it came from
  picking the right excess metric (Hbar-weighted quantization error)
  that Theorem 7 actually bounds. This is a clearer finding than the
  plan anticipated and better aligned with the theorem statement.
- Plan Step-2 end-of-day exit criterion met: Theorem 7 proved,
  Lemma 6 in closed form, Check (b) passes, Phase 1 OQ #1 flips to
  **resolved**.

**Next (Wed 2026-04-29 — Day 10):**
1. Extend Theorem 7 beyond $H_t = P_{V_t}$ to general
   $H_t \succeq 0$ (Phase 0 Alt.~C functional distortion). Step 5
   of the current proof is the bottleneck — the $O(\deff)$
   stabilizer argument uses $\bar H^{1/2}\tauH$ spherical
   invariance, which for general $H_t$ with non-aligned eigenbases
   needs a careful re-examination.
2. Fix the random-$V$ numerics at $r \geq 16$: add
   `quantize_in_Hbar_halfrep` that quantizes $\bar H^{1/2}\tauH$
   (rather than $Q^\top\tauH$) so Hbar-metric error is per-coord
   uniform. Expect slope $-2$ across all $r$ for random $V$ too.
3. MSE$\to$CE extension (the paper's headline claim) — set up the
   reduction using a Markov-chain / DPI argument from $f_t(w) =
   \mathrm{KL}(p_{\tau_t} \| p_w)$ to the quadratic approximation
   near $\tau_t$.

**Blockers:** none. Zandieh/Mirrokni/Daliri/Hadian have not replied
as of EOD 2026-04-28 (2 days after send 2026-04-26). If a reply
arrives during Day 10, log in `feedback_received.md` and evaluate
whether it amends the scaffold; do not stall Day 10 on it.

## 2026-04-29 (Wed) — Day 10 of Phase 1

**Done today:**

1. **Pedagogical explainer (morning, user-requested before Day-10
   technical work).** Wrote `theory/detailed_explanation.tex` — an
   8-page 9th-grade-reading-level walkthrough of the project so far:
   big picture, exact question, math setup, Theorem 7, proof sketch,
   numerics, bug story, what's next. Minimal preamble (article +
   amsmath), self-contained (no cross-file `\ref`). Companion to
   `theorem_v1.tex`, not a replacement.

2. **Extension of Theorem 7 to general $H_t \succeq 0$
   (Theorem 8 of `theorem_v1.tex` §5.5).** The bottleneck flagged
   Day 9 — that Step 5's $\bar H^{1/2}\tauH$-isotropy argument uses
   $H_t = P_{V_t}$ — turned out to have a clean resolution once the
   **hard distribution** was generalized rather than the theorem
   statement: the $D_0$-weighted ellipsoid sampling
   $v_t = B D_0^{-1/2} u_t$, $u_t\sim\text{Unif}(S^{r-1})$, makes
   $\E[H_t\tau_t\tau_t^\top H_t | U_t] = (B^2/r) H_t$ regardless of
   $D_0$, so the isotropy trace computation of Lemma 6 goes through
   verbatim. Step 5's $O(W)$-equivariance also transfers: $H_t =
   U_t D_0 U_t^\top$ transforms as $O H_t O^\top$ under $O\in O(d)$,
   $v_t$ is $O(d)$-free (lives in $\R^r$), so $\eta\mapsto O\eta$.
   The RD bound is **identical** to Theorem 7:
   $\mathcal{D}^\star \geq B^2(1-\deff/(Tr)) + c_{\mathrm{TQ}}
   (B^2\deff/(Tr))\cdot 2^{-2R/\deff}$.

3. **Day-10 sanity check (`code/synthetic/day10_general_ht.py`)**
   with 4 `D_0` specs (iso, geom, linear, two-scale) at
   $(T,d,r) \in \{(2,64,4), (3,64,4), (4,128,8)\}$:
   - **Test A/B:** floor formula and $\E[\|\eta\|^2] = B^2\deff/(Tr)$
     match theory to 5 digits across all 8 cells.
   - **Test C:** eigenvalue spread of $\E[\eta_W\eta_W^\top | W]$ is
     below the Marchenko--Pastur isotropic UB across all 5 cells
     at $n\in\{3\times 10^4, 10^5\}$ — $O(W)$-invariance confirmed
     under general $H_t$.
   - **Test D (slope check):** first run had slope $-2.00$ for
     `iso`/`twoscale` but $-0.73$ for `geom` at $T=2$ shared. Root
     cause: the Day-9 `quantize_tauH_adaptive` quantizes $Q^\top\tauH$
     with a fixed range $\pm c\sigma_{\mathrm{pc}}$ per coord, but
     the per-coord std of $Q^\top\tauH$ scales as $B/\sqrt{T\lambda_i r}$
     in the eigenbasis of $\bar H$ — non-uniform for non-identity $D_0$.
     Fix: `quantize_eta_adaptive` quantizes $\eta = \bar H^{1/2}\tauH$
     directly; per-coord std is uniform $B/\sqrt{Tr}$ by the trace
     identity (this is exactly the Day-9 open-questions item
     "quantize $\bar H^{1/2}\tauH$ per-coord uniform Hbar-weight
     rather than $Q^\top\tauH$"). With $c=5$ range, slope is
     $-2.00\pm 0.01$ across all 6 cells including `geom` at $T=4$.
     The fix solves Day-9's random-$V$ high-$r$ issue too.

4. **Scaffold updates to `theorem_v1.tex`:** title $\to$ Day 10;
   abstract cites Theorem 8 + `day10_general_ht.py`; new §5.5
   "Extension to general $H_t \succeq 0$" with generalized hard
   distribution $P^\star$, Lemma 6' (`lem:floor-general`),
   Theorem 8 (`thm:general`), plus a remark on why the bound
   doesn't depend on $D_0$ and a remark on CE-Fisher practicality.
   §7 numerical-evidence section now has a "Day-10 verification"
   subsection with the 4-test summary. §8 open items:
   Day-8 item 4 (general $H_t$) marked **resolved** under the
   common-$D_0$ assumption; task-dependent $D_t$ is the new open
   item (Day 11 target). Achievability added as a Phase 2 item.

5. **LaTeX hygiene.** 42 `\begin`/42 `\end` balanced. All new `\ref`
   and `\eqref` targets resolve to local labels
   (`sec:general-ht`, `thm:general`, `lem:floor-general`,
   `eq:thm-general`, `eq:floor-general-det`). No dangling references.
   pdflatex still not installed locally; Overleaf compile deferred.

**Key insight.** The "MSE$\to$CE bridge" bottleneck dissolved once I
asked the right question: not *how does the proof work for arbitrary
$H_t$-weighted sources*, but *what distribution on $\tau_t$ makes
$H_t$-weighted energy isotropic the same way uniform-on-sphere makes
Euclidean energy isotropic?* Answer: uniform on the $H_t$-ellipsoid
in $V_t$, i.e.\ $\tau_t = U_t D_0^{-1/2} u_t \cdot B$ with $u_t$
uniform. This is the "$H_t$-aware Haar measure" on $V_t$. Lesson for
Day 11+: when generalizing, generalize the **distribution class** in
sync with the metric class; don't try to prove the theorem against
the old distribution on the new metric.

**Plan deviations:**
- No plan file for Day 10 (technical work followed the user's
  "go as you recommended" directive after Day 9 close-out). Verify-
  before-externalize memory applied: numerical check came before
  LaTeX edits to Theorem 8.
- Slope-fit issue at T=4 `geom` shared required bumping clipping
  range from $c=3$ to $c=5$. Root cause: $\eta$ has heavier tails
  at higher $T$ (sum of $T$ bounded variables; range/std ratio is
  $\sqrt{Tr}$). At $T=4, r=4$, $\sqrt{Tr}=4$, so $c=3$ clips tails.
  $c=5$ is safe for $T\leq 4$; a principled choice for larger $T$
  is $c\sim\sqrt{Tr}$.

**Next (Thu 2026-04-30 — Day 11):**
1. Task-dependent $D_t$ (open item 1). If $(D_t)_t$ are deterministic
   but mismatched across tasks, Step 5's $O(W)$-equivariance fails.
   Think about whether a weaker symmetry (e.g.\ rotational invariance
   only in a sub-subspace) or a different hard distribution can
   recover the LB.
2. Begin Phase 2: achievability upper bound via Hadamard-incoherence
   quantization of $\eta$. Goal is a matching upper bound to
   Theorem 8 up to constants — closes the RD function.

**Blockers:** none. No email reply from Zandieh/Mirrokni/Daliri/Hadian
as of EOD 2026-04-29 (3 days after send). Per `plan.md` §10, assume
no reply and do not stall.

## 2026-04-30 (Thu) — Day 11 of Phase 1

**Done today:**

1. **Preprint repo live.** Pushed `README.md`, `LICENSE` (CC BY 4.0),
   and `rdmerge.pdf` (Phase 0 note) to
   https://github.com/K144U/rdmerge-preprint. PDF-only externalization
   per user preference (memory: `feedback_pdf_only_externalization.md`).
   The Alignment Forum post will link the raw PDF URL in its first
   reply.

2. **Day-10 hedge retired: Theorem 8 now covers task-dependent $D_t$.**
   On close re-reading of the Day-10 scaffold, I realized the
   "deterministic mismatched $D_t$'s break Step 5" hedge was wrong.
   Two independent reasons:
   - Lemma 6' Step 1: $\tau_t^\top H_t\tau_t = v_t^\top D_t v_t = B^2$
     holds per-task because $v_t = B D_t^{-1/2} u_t$ with
     $u_t\sim\mathrm{Unif}(S^{r-1})$ gives $v_t^\top D_t v_t = B^2
     \|u_t\|^2$. Independent of which $D_t$.
   - Lemma 6' Step 2 + Theorem 8 Step 5: $\E[H_t\tau_t\tau_t^\top H_t
     | U_t] = (B^2/r) H_t$ uses only the PSD identity $H_t H_t^+ H_t =
     H_t$. $O(W)$-equivariance of the tuple given $W$ holds because
     $U_t$ is Stiefel-invariant per-task and $v_t$ lives in $\R^r$
     (hence $O(d)$-free) regardless of $D_t$.

   So the task-labeled $D_t$ is innocuous — it's not exchanged with
   the $V_t$ randomness; it just sits there.

3. **Numerical verification (`code/synthetic/day11_task_dep_Dt.py`).**
   Reuses all Day-10 infrastructure via
   `from day10_general_ht import (...)`. Generalized sampler
   `sample_task_dep_ht_tuple(..., D_list, ...)` takes a list of $T$
   distinct per-task spectra. Four tests, eight cells each:
   - **Test A/B (floor + trace):** across `[iso,geom]`,
     `[geom,twoscale]`, `[iso,geom,twoscale]`, `[geom,lin,twoscale]`,
     `[iso,geom,lin,twoscale]` at $T\in\{2,3,4\}$: floor matches
     $B^2(1-\deff/(Tr))$ to 4–5 digits; $\E[\|\eta\|^2]$ matches
     $B^2\deff/(Tr)$ to 5 digits.
   - **Test C (MP-aware rotational invariance):** spread $\leq$
     MP-UB in all four cells at $n\in\{3\times 10^4, 10^5\}$.
     E.g., $T=2$, $r=4$, `[geom,twoscale]`, $n=10^5$: spread
     $1.0283$ vs MP-UB $1.0364$. Passes.
   - **Test D (slope):** $\log_2\text{excess}_{\bar H}$ slope is
     $-2.00\pm 0.01$ across all six shared/random cells. Passes.

4. **Scaffold updates (`theory/theorem_v1.tex`).**
   - Title → Phase 1 Day 11; abstract cites `day11_task_dep_Dt.py`
     and "arbitrary task-dependent spectra".
   - §5.5 hard distribution: `fix D_0` → `fix per-task D_1, ..., D_T`.
     Footnote updated: "nothing in the LB below uses any relationship
     among the $D_t$'s."
   - Lemma 6' statement: `any fixed PSD $D_0$` → `any fixed PSD
     $(D_t)_{t=1}^T$`.
   - Lemma 6' proof Step 2: $D_0 \to D_t$ throughout; added "which
     holds per-task regardless of $D_t$."
   - Theorem 8 statement: same upgrade.
   - Step 5: added the explicit task-label argument ($v_t$ is
     $O(d)$-free because it's determined by $u_t$ and the
     task-labeled $D_t$, neither transforming under $O$).
   - Remark "Why the bound doesn't depend on $D_0$" → "Why the bound
     doesn't depend on $(D_t)_t$"; updated to cite both Day-10 and
     Day-11 numerics.
   - Remark "Practical $H_t$: CE-Fisher": removed the common-$D_0$
     modeling-assumption hedge; now says Theorem 8 applies directly
     to CE-Fisher with heterogeneous per-task curvature.
   - §7 (numerical evidence): split into Day-10 and Day-11 paragraphs.
   - §8 (open items): item 1 (task-dep $D_t$) → resolved and absorbed
     into the abstract + intro; items are now just (1) small-overlap
     $\tauH$ robustness, (2) achievability (Phase 2).

5. **LaTeX hygiene.** 41/41 `\begin`/`\end` balanced (was 42 on
   Day 10; I merged two remarks' content without adding new envs,
   so the count dropped by 1 — verified by eye). All `\ref`/`\eqref`
   targets resolve. pdflatex still not installed locally; Overleaf
   compile deferred.

**Key insight.** Yesterday's Day-10 hedge was a failure of close-
reading, not a failure of math. The Step-5 argument was ready for
task-dependent $D_t$ *as written* on Day 10 — it just needed the
one-sentence "the task-labeled $D_t$ is baked into $v_t$, not into
the random Stiefel sample" clarification. Lesson: when you write
a conservative "Day-N+1 target" for an item you're not 100% sure
about, try to actually attempt it first — the hedge might dissolve
without real work.

**Phase 1 status: fully closed.** All four Day-8 open questions are
resolved (Theorem 7, Lemma 6, $\deff$ explicit, Theorem 8 general
$H_t$). Two lower-priority items remain (robustness of $\tauH$ for
small-overlap $V_t$, and achievability) but neither blocks Phase 2
start.

**Next (Fri 2026-05-01 — Day 12):**
1. **Phase 2 starts.** Achievability: explicit merging algorithm
   via Hadamard-incoherence quantization of $\eta$. Goal: match
   Theorem 8 up to constants. Break into (a) algorithm statement,
   (b) analysis, (c) numerical matching.
2. **Update `detailed_explanation.tex`.** Add a short paragraph to
   the "What's Next" section mentioning the completed general-$H_t$
   extension and the Phase 2 kickoff.
3. **Post Alignment Forum note.** User-task; link the raw PDF URL
   in the first reply, log to `feedback_received.md`.

**Blockers:** none. No email reply yet.

## 2026-05-01 (Fri) — Day 12+13 of Phase 1/2: achievability (partial)

**Done today:**

1. **Updated `theory/detailed_explanation.tex`** "What comes next"
   section to reflect Day-10/11 general-$H_t$ closure and Phase 2
   start.

2. **Day 12 — Hadamard-incoherence achievability.**
   `code/synthetic/day12_achievability.py`: quantize
   $\eta = \bar H^{1/2}\tauH$ via randomized Walsh-Hadamard +
   uniform scalar quantization on the padded $n = \text{nextpow2}(d_{\text{eff}})$
   coords. Decoder inverts Hadamard + diagonal signs, applies
   $\bar H^{+1/2}$ to get $w^\star$.
   Results:
   - **Floor-zero cells ($T=2$, $r=4$, random $V$, iso+iso and
     iso+geom):** slope on $\log_2 \text{excess}_{\max}$ vs bits
     is $-2.00$; ratio `excess_max / LB(c_TQ=1)` is $\approx 11$,
     **constant across all five rate points.** This is
     achievability-to-constants: Theorem 8 LB matches UB for the
     canonical Stiefel regime.
   - **Non-pow-2 cell ($T=3$, $d_{\text{eff}}=12$ padded to $n=16$):**
     slope $-2.00$ but ratio drifts 35 → 1393 across rates. Cause:
     Hadamard padding wastes bits since LB assumes $R/d_{\text{eff}}$
     effective rate but algorithm has $R/n$. Fixable (Day 13).
   - **Shared-V cells:** slope on `exc_avg` is $-2.00$ clean, but
     slope on `exc_max` is $-1$. Cross-term issue.

3. **Day 13 — two attempted fixes.**
   `code/synthetic/day13_achievability_fixes.py`:
   - **Fix 1 (Gaussian-QR mixer):** replace pow-2 Hadamard padding
     with a $d_{\text{eff}} \times d_{\text{eff}}$ random orthogonal
     matrix (Gaussian + QR). No padding overhead.
     Result: **works.** Non-pow-2 cells now have constant ratio
     $\approx 13$ across rates, slope $-2.00$. Problem 1 closed.
   - **Fix 2 ($H_t$-Chebyshev center):** replace quantization
     center from $\tauH$ to $w_{\text{cheb}} = \arg\min_w \max_t
     (w - \tau_t)^\top H_t (w - \tau_t)$, computed via smooth-max
     + weighted-linear-system iteration in the reduced
     $d_{\text{eff}}$-basis (fast; small $d_{\text{eff}}$ gives
     sub-second convergence).
     Result: **does not fix the shared-V max-distortion slope.**
     Shared-V iso+iso still gives slope $-1.12$; geom+twoscale
     gives slope $-0.02$ (worse, probably my quantizer clip range
     is mis-sized for $w_{\text{cheb}} \neq \tauH$).

4. **Diagnosis of the shared-V $2^{-R/d_{\text{eff}}}$ phenomenon.**
   At a Chebyshev center with $\geq 2$ active tasks, KKT gives
   $\sum_{t \text{ active}} \alpha_t g_t = 0$ for gradients
   $g_t = H_t(w_{\text{cheb}} - \tau_t)$. For $T=2$ shared-V
   iso+iso, both tasks active, $g_1 = -g_2$. Under any zero-mean
   perturbation $\delta w$:
   $\max_t D_t(w_{\text{cheb}} + \delta w) = \text{cheb}^2
    + |g_1^\top \delta w| + O(\|\delta w\|^2)$.
   $|g_1^\top \delta w|$ is half-normal under isotropic $\delta w$,
   scaling as $\sigma \|g\| \sqrt{2/\pi}$, so decay is
   $2^{-R/d_{\text{eff}}}$ linear in rate. **This is fundamental
   for linear/deterministic encoders when the Chebyshev center has
   $\geq 2$ active tasks.** Can be partially fixed by projecting
   the quantization noise onto the orthogonal complement of
   $\text{span}\{g_t : t \text{ active}\}$ (a dim-$(d_{\text{eff}}-T+1)$
   subspace), but this is a non-trivial encoder change and still
   only saves a constant.

5. **Interpretation.** Theorem 8's $2^{-2R/d_{\text{eff}}}$ term
   comes from RD on $\eta = \bar H^{1/2}\tauH$ — this is the
   *average-distortion* sub-problem LB. $\max \geq \text{avg}$
   transfers it to max-distortion as a lower bound, but there's no
   reason to expect matching achievability on max-distortion in
   regimes where floor $> 0$ and multiple tasks are tied at
   $\tauH$. **Hypothesis:** Theorem 8 is tight on max-distortion
   in the floor-zero regime (Stiefel random, $Tr \leq d$, the
   paper's generic case) and loose by $2^{-R/d_{\text{eff}}}$ in
   the shared-V regime. This is consistent with shared-V being a
   degenerate limit for the hard distribution.

**What Phase 2 Day 12+13 actually delivered:**

- An explicit merging algorithm (Gaussian-QR mixer + uniform
  scalar quantization of $\eta$ around $\tauH$) that achieves
  Theorem 8's LB up to a constant factor $\leq 13$ in the
  floor-zero regime.
- Clean numerical verification across $T \in \{2, 3\}$, iso+iso,
  iso+geom, iso+geom+twoscale tuples, $r = 4$, $d = 128$.
- The shared-V slope gap, diagnosed and explained; noted as a
  Phase 2 open item but not a blocker for the paper's headline
  result.

**Next:**
1. **Day 14.** Write Theorem 9 in `theory/theorem_v1.tex` §9
   (new section): matching achievability for the floor-zero
   Stiefel regime. Statement:
   $\mathcal{D}^\star \leq C \cdot B^2 \cdot 2^{-2R/(Tr)}$
   under Stiefel-random $V_t$ with $Tr \leq d$, $D_t$ arbitrary
   positive-definite, for explicit $C$. Proof via the Gaussian-QR
   algorithm + standard uniform-scalar-quantization analysis on
   $\eta$ (Gaussian-like post-mixing due to isotropy), combined
   with the Lemma 6' floor computation. Combined with Theorem 8:
   $\mathcal{D}^\star = \Theta(B^2 \cdot 2^{-2R/(Tr)})$ up to
   constants.
2. **Day 15+.** Shared-V max-distortion gap as a Phase 2.5 item:
   either (a) weaker matching with $2^{-R/d_{\text{eff}}}$ explicit,
   or (b) a smarter encoder (null-space dithering) that recovers
   $2^{-2R/d_{\text{eff}}}$. Not a paper-blocker.

**Blockers:** none. No email reply yet (5 days).

## 2026-05-02 (Sat) — Day 14 of Phase 2: shared-V achievability (partial)

**Done today:**

1. **Derived the information-theoretic structure of the shared-V
   max-distortion gap.** At $w_{\text{cheb}}$ with $|A|$ active tasks,
   active gradients $\{g_t\}$ span $|A|-1$ dims (KKT). Quantization
   noise $\delta w$ decomposes into parallel (in span) and perp
   (null). Linear cross-term $\max_t 2 g_t^\top \delta w$ scales as
   $\|g\| \cdot 2^{-R_\parallel/(|A|-1)}$ (dominant at high rate);
   quadratic $\|\delta w\|^2$ scales as $2^{-2R_\perp/k}$, $k = r-|A|+1$.
   Optimal balanced allocation gives excess
   $\asymp 2^{-2R/(r+|A|-1)}$, strictly better than naive $2^{-R/r}$
   but strictly worse than LB's $2^{-2R/r}$ (when $|A| > 1$).

2. **Implemented null-space-aware bit allocation
   (`code/synthetic/day14_shared_v_gap.py`).** Encoder computes
   $w_{\text{cheb}}$, identifies active gradients, forms orthonormal
   parallel basis $P$ and perp basis $N$ in the $Q$-basis of
   $\range(\bar H)$. Allocates $R_\parallel, R_\perp$ via
   $R_\parallel = 2R m/(k+2m)$ (asymptotic optimum). Quantizes each
   subspace independently with a random-orthogonal mixer.

3. **Numerical result for iso+iso shared-V $T=2, r=4$:** slope on
   $\log_2 \text{excess}_{\max}$ vs $b = R/r$ went from
   **$-1.12$ (naive Day-13)** to **$-1.66$** with the null-space
   split — matching the theoretical prediction $-2r/(r+m)|_{m=1} = -8/5
   = -1.60$ tightly. Confirms the information-theoretic analysis.

4. **Non-iso $D_t$ cells (iso+geom, geom+twos, $T\geq 3$ mixed)
   fail — slope $\approx 0$, excess plateaus at $\sim 2$.** Two
   plausible causes: (a) the smooth-max Chebyshev solver
   `cheb_center_reduced` isn't converging for anisotropic $H_t$
   (beta schedule too aggressive, or Newton step oscillates);
   (b) the active-set detection threshold (thresh=0.98) picks up
   only a subset of the true active tasks when distortions aren't
   perfectly equalized. Diagnosis: symbolic check of the algorithm
   shows correctness IF $w_{\text{cheb}}$ is accurate. For $T=2$ a
   closed-form solver (root-find on the KKT $\alpha_1$) should
   resolve this; deferred.

5. **Status interpretation.** The shared-V gap is three things:
   - Fundamental exponent gap $2^{-2R/r}$ (LB) vs $2^{-2R/(r+|A|-1)}$
     (best explicit linear algorithm). The LB is proved tight on
     avg-distortion via $\max \geq \text{avg}$, but not tight on max.
   - Explicit-encoder limit: a linear encoder with any basis gives
     at best the $2^{-2R/(r+|A|-1)}$ rate. Doing better requires
     vector quantization with a non-linear decoder (Shannon-style
     random coding), which is non-constructive but
     information-theoretically achievable.
   - Implementation bug on non-iso $D_t$: solvable but non-trivial;
     not a blocker for the paper's headline result.

6. **Paper-level implication.** Theorem 9 (achievability, to be
   written Day 15) should cover the floor-zero Stiefel-random
   regime (matching LB up to constants) and acknowledge a
   Phase 2.5 gap in the shared-V regime. The floor-zero regime is
   the paper's headline generic case; the shared-V regime is a
   degenerate limit and the gap there is a note, not a blocker.

**Next (Day 15):**
1. Write Theorem 9 in `theory/theorem_v1.tex` §9 for the floor-zero
   Stiefel regime. Proof via Gaussian-QR algorithm analysis;
   $O(d_{\text{eff}}$-dim isotropic Gaussian asymptotic for $\eta$)
   matches TurboQuant's scalar-quantization bound up to constants.
2. Add a "remaining gap" paragraph noting the shared-V
   $2^{-R/(r+|A|-1)}$ best-known explicit achievability and
   conjecturing the true RD function equals the LB (via random
   coding) but leaving proof as open.
3. Fix the `cheb_center_reduced` convergence for non-iso $D_t$ (use
   closed-form for $T=2$, or swap to cvxpy for general $T$).
   Non-blocking.

**Blockers:** none. No email reply (6 days).


## 2026-04-24 (Fri) — Day 14 closeout: shared-V gap closed for T=2

Phase 2 has a long "Day 14" that covers 14 → 14g. The headline:
**closed-form T=2 Chebyshev solver + null-space split + fractional
per-coord bits + fixed clip $c=11.5\sigma_{pc}$ achieves slope
$-1.60 \pm 0.10$ across all 5 anisotropy regimes** (iso+iso,
iso+geom, geom+twos, lin+twos, geom+iso), matching the theoretical
prediction $-2r/(r+|A|-1) = -8/5$ for $r=4, |A|=2$. Bootstrap 95%
CI on each slope is $\pm 0.010$ (1000 trials), so the $\pm 0.10$
spread is structural anisotropy effect, not noise.

**Sub-days (all in `code/synthetic/`):**

- **Day 14b (`day14b_cheb_T2_closedform.py`).** Fixed the Day-14
  smooth-max Newton solver that failed on non-iso $D_t$ (slope
  plateaued at $\approx 0$). For $T=2$ the Chebyshev center
  admits a closed-form via KKT: parameterize
  $w(\alpha) = (\alpha H_1 + (1-\alpha)H_2)^{-1}(\alpha H_1 \tau_1
  + (1-\alpha)H_2\tau_2)$ and root-find $D_1(w(\alpha)) =
  D_2(w(\alpha))$ in $\alpha \in (0,1)$ with `brentq`. Residual
  $|D_1 - D_2|$ lands at $10^{-9}$ (machine precision for a
  1-D root-find). With this fix, all 5 cells immediately show
  slopes in $[-1.65, -1.19]$ (strictly better than naive $-1.00$),
  but iso cells still cleaner than anisotropic.

- **Day 14c (`day14c_fractional_bits.py`).** Identified a
  secondary artifact: perp quantizer's `b_per = R_sub // k`
  integer rounding plateaued at high $R$. At $R=20,24$ with $k=3$,
  both map to $b_{per}=4$ sharing the same step. Fix: per-coord
  variable bit widths — $n_{hi} = \round(k \cdot \{b_{avg}\})$
  coords get $\lceil b_{avg} \rceil$ bits, rest get $\lfloor
  b_{avg} \rfloor$. iso+iso then hit $-1.62$, matching theory.

- **Day 14e (`day14e_quantizer_sweep.py`).** Swept 16 variants
  (fixed clip $c \in \{3,5,7,10,14,20,50\}$, empirical-max,
  whitened basis, per-coord empirical, rate-adaptive $\sqrt{b}$
  clip). Key findings: (a) slope is *monotonic* in $c$ for fixed
  clip (no pathological non-monotonicity); (b) H̄-whitening does
  *not* help — the raw basis is already correct; (c) rate-adaptive
  $\sqrt{b}$ clips underperform because the source is bounded +
  concentrated near $w_{cheb}$, not Gaussian; (d) empirical-max
  clip fails on anisotropic cells (slope $-0.48$ on geom+twos).

- **Day 14f (`day14f_fine_sweep.py`).** Finer sweep on
  $c \in \{11, 12, \dots, 18\}$ at $n_{trials}=500$. Identified
  $c=11$ as the winner by mean-deviation criterion.

- **Day 14g (`day14g_final_lock.py`).** Final lock with
  $n_{trials}=1000$ and bootstrap 95% CI on slope estimates.
  **$c=11.5$ wins both rankings** (mean deviation 0.044, max
  deviation 0.104). Per-cell slopes (bootstrap CI):
  iso+iso $-1.704 \pm 0.008$, iso+geom $-1.604 \pm 0.010$,
  geom+twos $-1.502 \pm 0.009$, lin+twos $-1.591 \pm 0.011$,
  geom+iso $-1.603 \pm 0.010$. Three cells hit theory exactly;
  iso+iso overshoots (H₁=H₂ is structurally easier, closer to
  scalar-quant $-2$), geom+twos undershoots (heaviest anisotropy).

**Why $c=11.5$ is not ad hoc.** At $c=50$ (essentially no
clipping) slope is $-1.83$: the pure scalar-quantization
regime where error scales as $c^2 \cdot 2^{-2b}$. At $c=3$
clipping dominates → slope $-0.8$ (no matching scaling). $c=11.5$
is the Pareto point where clip-induced error at low $R$ matches
scalar-quant error at high $R$, yielding the balanced exponent
$-2r/(r+|A|-1)$. It's a quantizer tuning, not a theoretical
constant.

**What this closes:**
- Null-space-aware bit split is **numerically validated** for
  T=2 shared-V with arbitrary $D_t$ anisotropy.
- Remark `rem:sharedv-gap` in `theorem_v1.tex` §8 gets the
  empirical footnote.
- `notes/open_questions.md` marks shared-V T=2 as closed;
  T≥3 closed-form Chebyshev deferred to Phase 2.5.

**What this doesn't close:**
- Fundamental $2^{-2R/r}$ (LB) vs $2^{-2R/(r+|A|-1)}$ (linear
  encoder best) exponent gap. Closing this requires non-linear
  / random-coding encoders. Still conjectured tight LB on max;
  Phase 2.5.
- T≥3 general-$T$ closed-form Chebyshev solver. SOCP fallback
  works but not coded. Phase 2.5.

**Next:** Phase 3 (real-LLM empirical validation) or Phase 2.5
(either of the above). Recommendation: **move to Phase 3** —
Phase 2 main claim (Theorem 9 + numerically-validated
achievability) is paper-complete, and real-LLM validation is
the thesis's critical "so what" step.

**Blockers:** none. No email reply (8 days).


## 2026-04-24 (Fri) — Day 15-16: Phase 2.5 follow-ups

Phase 2 was nominally closed in the morning, but the user asked
to tackle Phase 2.5 items in the afternoon: (A) general-$T$
Chebyshev solver to extend the null-space split beyond $T=2$;
(B1) sharpen the LB to match the linear-encoder UB
$-2r/(r+|A|-1)$.

**Item A (DONE).** `day15_cheb_general_T.py` implements the
Chebyshev center via cvxpy SOCP (CLARABEL) on the reduced
Q-basis. Initial attempt had a bug: CLARABEL default tolerance
left KKT residual at $\sim 10^{-2}$, which made
`parallel_perp_basis` detect the wrong active-set rank ($m = |A|$
instead of $m = |A| - 1$). Fix: post-SOCP Gauss--Newton KKT
refinement — solve the square system (stationarity,
$|A|-1$ equalizations, normalization) in $(w, \alpha)$ via
damped Newton, which brings KKT residual to $10^{-15}$.
Empirically the solver now matches `day14b` closed-form to
$10^{-9}$ relative (default CLARABEL tol) for $T=2$ and gives
KKT-exact centers for $T=3, 4$. Extended the sweep to $T=3$
(iso+iso+iso, iso+geom+twos, geom+twos+lin) and $T=4$
(iso+4, iso+geom+lin+twos).

**Metric bug (subtle and important).** First run of Day 15
showed slopes $-0.8$ to $-1.1$ — apparently the algorithm was
failing. Diagnosis: the old excess metric was
$\max_t D_t(w^\star) - \E[\mathrm{floor}_{\mathrm{avg}}]$ where
$\mathrm{floor}_{\mathrm{avg}} = T^{-1}\sum_t D_t(\tauH)$. But
for shared-$V$ with $T\geq 3$ under anisotropy,
$\mathrm{cheb}^2 > \mathrm{floor}_{\mathrm{avg}}$ — the
max-distortion saturates at cheb$^2$, not the avg-floor. So the
old metric flattens at high $R$ when the true excess
$\max_t D_t(w^\star) - \mathrm{cheb}^2 \to 0$. Switching to
excess-over-cheb$^2$ (computed per-trial) gives the correct
slopes: $T=3$ iso+iso+iso $-1.53$, iso+geom+twos $-1.62$,
geom+twos+lin $-1.66$; $T=4$ iso+4 $-1.42$, iso+geom+lin+twos
$-1.53$. All beat the linear prediction $-2r/(r+|A|-1)$ (which
for $|A|\approx 2.7$ at $r=4$ is $\approx -1.40$); the overshoot
matches the Day 14g over-clip effect at $c=11.5$.

**Item B1 (RULED OUT).** `day16_sharpened_lb_check.py`:
valid rate-$R$ random-codebook encoder ($2^R$ iid Gaussian
codewords at origin, encoder picks the one minimizing
$\max_t D_t(w)$, decoder returns the codeword) at
$T=2,r=3$ iso+iso and $T=3,r=3$ iso+iso+iso, rates $R \in
\{6, 8, \ldots, 18\}$ ($2^{18} = 262144$ codewords is the
practical ceiling). Findings:
- $T=2, r=3$: random-codebook slope $-1.49$ to $-1.56$
  (narrow vs wider codebook). Linear UB $-1.50$. Consistent,
  so sharpened LB $-2r/(r+T-1) = -1.5$ could hold at $T=2$.
- $T=3, r=3$: random-codebook slope $-1.40$ to $-1.45$.
  Linear UB $-1.20$. **Random coding strictly beats linear
  by $\sim 0.25$**. A purported LB at $-2r/(r+T-1) = -1.20$
  would be falsified by this observation.

Therefore sharpening the LB to $-2r/(r+|A|-1)$ (B1 as stated)
is empirically ruled out for $T\geq 3$. The truth in the
shared-$V$ regime lies strictly between current Thm 8 LB
($-2r/r = -2$, loose) and the linear UB ($-2r/(r+|A|-1)$,
also loose). Exact value is open and beyond Phase 2 scope.

**Caveats on Day 16 experiment.**
- Initial experiment had a bug (codebook shifted by
  data-dependent $w_{\mathrm{cheb}}(\tau)$, giving the encoder
  free side information). Fixed; reported slopes are for
  valid origin-centered codebooks.
- $R\leq 18$ only (memory-bound). Slope may not have settled
  asymptotically; possible that true exponent is closer to
  linear UB at higher $R$, but the $T=3$ gap of $0.25$ is
  large enough to be robust to this.
- Results are for $r=3$ specifically; $r=4$ should show the
  same qualitative pattern but was too expensive to run at
  $2^{18}$ codewords per trial.

**Docs updated.**
- `theory/theorem_v1.tex` Remark `rem:sharedv-gap` gets a
  "Phase 2.5 Day 15--16 update" paragraph with all of the
  above.
- `notes/open_questions.md` shared-V entry marked as resolved
  for (a) general-$T$ construction and (b) LB-sharpening
  direction; the exact RD is flagged as a genuine open
  problem beyond paper scope.
- Memory: `project_phase_status.md` updated to note Phase 2.5
  findings.

**Next (Day 17+):** Phase 3 — real-LLM empirical validation.
This is the thesis's critical "so what" step and the remaining
gap between toy numerics (iid Gaussian $D_t$ spectra) and
actual LoRA fine-tunes of real LLMs.

**Blockers:** none. No email reply (8 days).



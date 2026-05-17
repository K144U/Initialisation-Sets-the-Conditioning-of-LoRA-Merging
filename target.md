# `target.md` — publication strategy + reality check

*Written 2026-04-24 by Claude at Sankalp's request. This is a
working document meant to be honest about where the paper
actually stands. Revise as things change.*

---

## 1. Executive summary

The **theory is submission-quality** (Theorems 7, 8, 9 in
`theory/theorem_v1.tex`, all proven). The **empirical story is
missing entirely** — zero real-LLM experiments, only synthetic
random-matrix tests. The **paper prose does not yet exist** —
the 8 section files in `paper/sections/` are 1–4-line TODO stubs.

**Blunt read.** Submitted *as of today* the paper is a TMLR or
ISIT paper, not an ICLR paper. ICLR reviewers will ask "does
the bound explain real LoRA-merging performance?" and there is
no answer yet. ICLR 2027 becomes realistic *only* if Phase 3
(real-LLM experiments, June–July) lands cleanly and the LB
turns out to predict something reviewers care about.

**Single highest-leverage action**: post `rdmerge.pdf` to arXiv
within 2 weeks. Priority matters — LoRA-merging theory is
warming up (arXiv 2508.16082, 2511.21437). Everything else
downstream depends on this anchor.

---

## 2. What's done vs what's missing

| Area | Done | Missing / Risky |
|---|---|---|
| **Theory** | Lemma 1 (max ≥ avg), Lemmas 6/6' (floor closed form), Theorem 7 (projector LB), Theorem 8 (general-$H_t$ LB), Theorem 9 (matching UB, floor-zero Stiefel, $C = Tc^2/3$). | Constant $C = O(T)$ in Thm 9 is loose; Fisher/CE-loss extension currently a *remark* rather than a theorem; T-scaling question open; adversarial-vs-random tightness open. |
| **Synthetic empirics** | 20 `day*.py` scripts, T ∈ {2,3,4,8}, d ∈ {128, 512, 2048}, r ∈ {4, 8, 16}, ~1000 trials/config, bootstrap CI ±0.01 on slopes. 46+ PNG figures. | All synthetic. No real-LLM runs yet. |
| **Real-LLM empirics** | **Zero.** | **This is the #1 gap for any ML venue.** Phase 3 (Qwen 0.5–3B, Gemma-2B, TinyLlama, Phi) not started. |
| **Paper prose** | `paper/main.tex` skeleton + section files exist. | Abstract, intro, related work, setup, LB exposition, achievability exposition, experiments, discussion — *all* TODO stubs (1–4 lines each). Zero written sentences. |
| **Related work** | Landscape mapped in `deepresearch.md` (KnOTS, TSPA, DO-Merging, ARM, Core Space, TIES, DARE, TVQ, 1-bit-Merging, Task Arithmetic, Ortiz-Jimenez, TurboQuant lineage). | No actual prose. Competing paper arXiv 2511.21437 (Nov 2025) claims subspace methods fail at LLM scale — *not* addressed. |
| **External** | `rdmerge.pdf` on `K144U/rdmerge-preprint` GitHub; Zandieh/Mirrokni/Daliri/Hadian emailed 2026-04-26; Alignment Forum post same date. | **Not on arXiv.** No replies to emails in 8 days. No endorsement / feedback loop from an established theorist yet. |

---

## 3. Venue options — ranked honestly

Ordered by *fit today*. Each row: deadline, page limit, reviewer
pool, the single biggest concern.

### 3.1 Best fits *today* (theory-first, no real-LLM required)

1. **TMLR** — *rolling*, 12 pages + appendix, theory/ML
   reviewers, no mandatory experiments.
   **Concern**: low prestige relative to ICLR. But a fast
   acceptance with DOI would anchor priority and serve as the
   "camera-ready version" for workshop/venue resubmission.
   **Action**: realistic in 4–6 weeks once related-work prose
   is written.

2. **ISIT 2027** — deadline *~Jan 2027*, 5 pages, info-theory
   reviewers who will appreciate the RD framing directly.
   **Concern**: 5 pages is tight; full proofs go to a companion
   arXiv version. Reviewers will want tighter constants in Thm 9.
   **Action**: natural home for the theory paper. If Phase 3
   stalls, this is the de-facto primary.

3. **IEEE Trans. Info Theory / JSAIT** — rolling journal,
   long format, top-tier theory reviewers.
   **Concern**: review timeline is 6+ months; not compatible
   with a fast turnaround. Good companion to ISIT.

### 3.2 Fits *after Phase 3* lands cleanly

4. **ICLR 2027** (Sankalp's target) — deadline **2026-09-24**,
   9 pages + appendix, broad ML reviewer pool.
   **Concern**: reviewer lottery is brutal; pure theory papers
   without empirics get bounced. Need Phase 3 experiments that
   *predict* at least one practitioner-facing claim (e.g., "the
   LB explains why method X fails on task class Y").
   **Action**: keep as primary target but treat August as a
   decision gate.

5. **AISTATS 2027** — deadline **2026-10-08**, 9 pages,
   theory-friendlier than ICLR.
   **Concern**: only 2 weeks after ICLR deadline; if the
   paper's revised for ICLR-style presentation, minimal tweak
   to resubmit here. Natural backup.

6. **ICML 2027** — deadline **2027-01-30**, 9 pages, broad ML.
   **Concern**: similar bar to ICLR. Gives you 4 extra months
   if the Phase 3 story needs more development.

7. **NeurIPS 2027** — deadline **~May 2027**, 9 pages.
   **Concern**: latest deadline; only use if ICLR, AISTATS,
   and ICML all fall through and you want to iterate more on
   experiments. Slow timeline means a competing paper could
   land first.

### 3.3 Workshop / de-risking paths

- **NeurIPS M3L workshop** (Mathematics of Modern Machine
  Learning) — short paper, fast feedback, positions the work
  in the theory community without a main-conference commitment.
- **ICML HiLD workshop** (High-dimensional Learning Dynamics)
  — theory-friendly, good fit.
- **ICML / NeurIPS LoRA / merging workshops** — if one exists
  in 2026, ideal audience.
  **Action**: treat workshop submission in June/July as a
  feedback pass, not a final destination.

### 3.4 What I'd avoid

- **COLT 2027**. Strong learning-theory bent; RD bounds aren't
  learning-theoretic in the sample-complexity sense they
  prefer. Weak fit.
- **ALT 2027**. Same reason.
- **UAI 2027**. No clear probabilistic-graphical-model angle.
  Wasted shot.

---

## 4. Four critical gaps — highest-leverage work

### A. Real-LLM experiments (highest leverage)

The biggest gap by a wide margin. **Minimum viable Phase 3**:

- **Base models**: Qwen-1.5B or Qwen-0.5B (small, fast). Gemma-2B
  or TinyLlama as a second to show generality.
- **Tasks**: 3–5 LoRA fine-tunes on distinct tasks (GSM8K,
  MMLU-subset, HumanEval, a chat-style task, a classification
  task). Pick for task-vector diversity.
- **Baselines**: Task Arithmetic, TIES, DARE, KnOTS, TVQ. At
  least 3 of these; all 5 if time permits.
- **The money plot**: theoretical LB (from `theorem_v1.tex`
  Thm 8) vs. empirical merging-induced task loss for each
  method. Does the bound predict the *ranking* of merging
  methods? Does it predict where simple methods fail?

If the LB and empirics are uncorrelated → pivot to ISIT. If
there's even a partial signal → ICLR story is viable.

**Do NOT** design Phase 3 to "validate" the LB as tight.
Design it to *test* whether the LB is informative. Honest
calibration matters more than a clean graph.

### B. Related-work section (currently zero prose)

Need ~1 full page. Structure:

1. **Model-merging landscape** (1 paragraph): Task Arithmetic
   → TIES/DARE → subspace/rotation methods (KnOTS, TSPA,
   DO-Merging, ARM, Core Space) → quantization-aware (TVQ,
   1-bit-Merging). Position *this* paper as the first to
   give information-theoretic limits for the problem class.

2. **Rate-distortion / vector quantization ancestry**
   (1 paragraph): TurboQuant (Zandieh et al.) as direct
   ancestor; Cover & Thomas Ch. 10 for classical RD; El Gamal
   & Cover 1982 for multi-description (but none treat
   max-distortion as the figure of merit — our contribution).

3. **Weight disentanglement** (1 paragraph): Ortiz-Jimenez et
   al. (NeurIPS 2023) provides the $H_t$ = Fisher / rank-$r$
   structure we use. Task arithmetic's empirical success
   suggests the rank-$r$ RD framing is the right one.

4. **The Nov 2025 negative result** (1 paragraph, important):
   arXiv 2511.21437 claims subspace-based merging methods fail
   at LLM scale. **Their result is an *upper bound* on specific
   algorithms; ours is a *lower bound* on *any* algorithm.** A
   failed algorithm is one witness to a method class being
   suboptimal; an LB tells you what is unavoidable. These are
   compatible and our paper *explains* why their methods hit
   a wall when the rank-$r$ RD floor is non-zero.

This section should be written **before** Phase 3, because it
shapes the narrative the experiments will test.

### C. Framing / narrative

Current framing ("Shannon-style RD LB for merging") is
accurate but dry. ML reviewers will skim. Stronger framing:

> *What is the best possible LoRA-merging algorithm, and how
> far are existing methods from it?*

This converts an LB from a technical curiosity into a
practitioner-facing tool. Concrete framing moves:

- **Title**: "Rate-Distortion Limits of Model Merging" (ML
  venues) or "Multi-Source Rate Distortion for Max Distortion"
  (ISIT). Pick per venue.
- **Abstract** (150 words):
  *motivation* (LoRA merging is everywhere, quality varies
  across methods with no principled reason) → *question* (is
  there a fundamental limit?) → *result* (yes: $\Theta(B^2
  \cdot 2^{-2R/(Tr)})$ in the generic regime, matched by an
  explicit algorithm up to a universal constant) → *so what*
  (existing methods sit at $X \times$ the limit on task $Y$;
  we show the limit *can* be approached in practice).
- **Intro**: lead with a worked example (two LoRAs merged for
  multi-task serving; show the task loss as a function of
  merge quality). Earn the theory.
- **Discussion**: name the open questions honestly — shared-V
  exponent gap, T-scaling, Fisher extension. Honest future-work
  lists beat overclaiming.

### D. Matching-UB constant + Fisher extension

Two smaller gaps that are easy wins:

- **$C = Tc^2/3$ in Theorem 9**. Linear-in-$T$ is loose; a
  straightforward sharper analysis (e.g., using a better
  Hadamard-incoherence bound, or swapping in sub-Gaussian
  tail bounds on the rotation) should give $C = O(\log T)$ or
  $O(1)$. ML reviewers *will* ask about this. A 2-page
  appendix tightening the constant is low-risk, high-yield.
- **Fisher / CE-loss extension**. Sits in a remark today.
  Promote to a theorem (even a short, corollary-style one)
  for ML venues. This pre-empts the #1 reviewer question:
  "your bound is for MSE loss, why should I care for LLMs?"

---

## 5. Framing / narrative — concrete templates

**Abstract (ML-venue version, first draft):**

> LoRA fine-tunes are now routinely merged into a single
> deployable model, with dozens of heuristics (Task Arithmetic,
> TIES, DARE, KnOTS, TVQ) competing on benchmarks with no
> principled performance ceiling. We establish a fundamental
> information-theoretic lower bound: for $T$ rank-$r$ LoRA
> task vectors merged at rate $R$ bits, the expected
> worst-task distortion is at least $\Omega(B^2 \cdot
> 2^{-2R/(Tr)})$ in the generic (floor-zero) regime. We match
> this bound with an explicit Gaussian-QR + scalar-quantization
> algorithm up to a universal constant. On real LoRA fine-tunes
> of [base models], the bound explains why [X], predicts [Y],
> and suggests [Z]. Our results give practitioners a
> calibration curve for merging-algorithm quality and a target
> for future algorithms.

(Placeholders X, Y, Z filled after Phase 3.)

**Title options (per venue):**

- ICLR/NeurIPS: "Rate-Distortion Limits of Model Merging"
- ISIT: "Multi-Source Rate-Distortion Theory for Max Distortion"
- TMLR: "A Rate-Distortion Function for Model Merging"

---

## 6. Execution timeline — month by month

- **May 2026 (Month 1)**: related-work prose; intro draft; arXiv v0
  upload. *Exit criterion*: 1-page RW in `paper/sections/`, intro
  skeleton in place, arXiv DOI in hand.
- **June–July 2026 (Months 2–3)**: Phase 3 real-LLM experiments.
  Start with Qwen-0.5B + 3 tasks + 3 baselines. Expand only if
  signal is promising. *Exit criterion*: 3–5 money plots; honest
  assessment of whether LB predicts anything.
- **End of July — decision gate.** If Phase 3 shows LB is
  informative → commit to ICLR 2027. If not → pivot to ISIT
  (Jan 2027 deadline) or TMLR (rolling).
- **August 2026 (Month 4)**: paper v1 draft. Internal
  feedback pass (ideally with Prof. Garg and/or Zandieh if
  they've replied by then).
- **September 2026 (Month 5)**: revision, polish, submit by
  2026-09-24 (ICLR) or hold for AISTATS 2026-10-08.

---

## 7. Risks — honest list

1. **Phase 3 shows the LB is uninformative on real LoRAs.** Most
   likely failure mode. Synthetic τ, H tuples are drawn iid
   Stiefel-random; real fine-tunes have structure (correlated
   tasks, low-rank Fisher, heavy-tailed task gradients) that
   the current hard distribution $P^\star$ ignores. Mitigation:
   if it happens, pivot to ISIT / TMLR (the theory stands on
   its own); write a follow-up paper addressing the structural
   gap.

2. **Concurrent / scooping work.** LoRA-merging theory is warming
   up. arXiv 2508.16082 (Aug 2025), 2511.21437 (Nov 2025), and
   likely others between now and September. Mitigation: arXiv
   upload within 2 weeks to establish priority. This is the
   single cheapest insurance available.

3. **ICLR reviewer lottery.** Reviewer pool for theory-heavy
   ML papers is thin. A skeptical reviewer who doesn't know
   RD theory can tank a good paper. Mitigation: make the intro
   accessible to a non-theorist; move proofs to supplementary;
   lead with the practitioner-facing framing in §5.

4. **Constants / proof polish still loose.** Thm 9's $C = O(T)$;
   Lemma 2 EPI-rewrite sits in the open-questions list.
   Mitigation: submit with honest constants; tighten during
   revision if accepted with conditions.

5. **No reply from Zandieh / Mirrokni.** External validation
   from an established theorist would materially strengthen
   the submission. Mitigation: if no reply by end of April,
   ping once more with a crisp 3-bullet summary; beyond that,
   move on and seek feedback via workshop / arXiv comments /
   Alignment Forum discussion.

6. **Solo-researcher bandwidth.** Phase 3 + paper writing in
   5 months is aggressive for one person with no collaborators
   on experiments. Mitigation: minimum viable Phase 3 (1 base
   model, 3 tasks, 3 baselines); cut scope rather than quality.

---

## 8. Recommendation

**Keep ICLR 2027 as the primary target** because the deadline
forces a clean narrative, and because a LoRA-merging paper
belongs in an ML venue if the experiments support it. But
**de-risk on three axes simultaneously**:

1. **arXiv upload within 2 weeks** — this costs almost nothing
   and buys priority.
2. **Write related-work + intro in May** — before Phase 3, so
   the experiments are designed to test specific claims the
   narrative makes.
3. **Treat end-of-July as a decision gate** — if Phase 3 is
   uninformative, pivot immediately to ISIT / TMLR. Don't
   sunk-cost into ICLR with a weak empirical story.

The paper *will* get published somewhere. The open question is
whether it gets published at ICLR with a strong story or at
ISIT / TMLR with a pure-theory framing. Both are good outcomes;
optimize for the first without burning the bridge to the
second.

---

## 9. Honest sanity check (three questions)

1. **If a reviewer asked "why is this ICLR material?"** — Today
   the honest answer is "it isn't yet." After Phase 3, the
   answer should be "the bound predicts X about LoRA merging
   in practice." If you can't write that sentence by end of
   July, pivot.

2. **If Phase 3 fails, what's the fallback?** — ISIT 2027 (Jan
   2027 deadline) + TMLR rolling. The theory is strong enough
   that both are realistic. No wasted work.

3. **Is each venue recommendation grounded?** — Yes. TMLR is
   rolling + theory-friendly; ISIT is the natural info-theory
   home; ICLR needs empirics; AISTATS is a theory-friendlier
   ICLR; ICML is 4 months later; NeurIPS is 8 months later.
   Backups in chronological order: ICLR (Sep 2026) → AISTATS
   (Oct 2026) → ICML (Jan 2027) → ISIT (~Jan 2027) → NeurIPS
   (May 2027) → TMLR (rolling, anytime).

---

*Last updated 2026-04-24. Revise after Phase 3 decision gate
in late July 2026.*

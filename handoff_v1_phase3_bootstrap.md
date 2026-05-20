# Handoff to HPC Claude Code instance

**Written:** 2026-05-17 by the local-workstation Claude Code session
on Sankalp's Windows machine.
**For:** the new Claude Code instance that will run on the
supercomputer to continue Phase 3 (real-LLM empirical validation)
and parallel paper / arXiv prep work for the **rdmerge** paper.
**Scope:** full continuation. Phase 3 execution, arXiv v1 prep,
paper drafting, and any Phase 4–5 work from `plan.md`.

---

## §0. READ THIS FIRST

You are stepping into a solo theoretical-ML research project that
has finished its theory phase and is about to start its empirical
phase on real LLMs. The user is **Sankalp Pathak**. The paper is
*A Rate-Distortion Lower Bound for Model Merging, with Matching
Achievability via Hadamard Incoherence*, targeting ICLR 2027
(deadline ~2026-09-24, backups: TMLR rolling and AISTATS 2027).

**One-line state:** Theorems 7, 8, 9 are closed and synthetic
numerics validate them at slope $-2.00 \pm 0.01$; Phase 3 design
is locked but no implementation code exists; you are here to
build that.

**If you only read one other file, read** `notes/phase3_design.md`
(361 lines) — it is the operational spec for what you're about to
build.

**What to do today:** go to §10 of this file (first-day checklist).
Do *not* start writing training code before walking that checklist.

**Cardinal rules** (from Sankalp's feedback memory, §1 of this
file): (1) pair proof work with same-day numerics, (2) full
verification pass before any externalization, (3) externalize
PDFs only — never `.tex` source.

---

## §1. User identity & engagement norms

**Sankalp Pathak** — independent / solo theoretical ML researcher.
Working on a self-directed paper. Comfortable with Shannon
rate-distortion theory, Fano / packing arguments, Cover & Thomas
Ch. 10, LoRA / task arithmetic literature, convex optimization
(SOCP, KKT, Chebyshev problems). Engages at proof-level detail,
not surface summary. Says directly when something feels shaky
("before I go ahead and make a fool of myself I want you to
verify everything").

**Co-author (as of 2026-04-25):** Prof. **Sanjay Garg** —
advisor-like contact who joined after seeing the v1 draft.
References to "Garg" or "advisor" in the daily log mean him.

**Compute:** previously single-GPU consumer (RTX 3060 Ti 8 GB on
Windows 11). Moving to the supercomputer you are running on now.
This is a *material change in the compute envelope* — see §4.

**Paper contact email:** `pathaksankalp04@gmail.com` (use this
on title pages, arXiv submission, formal correspondence).
**Claude Code system email:** `kittuwastaken@gmail.com` (for
tool auth only — never put this on the paper).

**Other contacts referenced in the daily log:**
- Amir Zandieh & Vahab Mirrokni — TurboQuant authors at Google
  Research (cold-emailed 2026-04-26; their Thm 3 is the template
  this paper adapts)
- Sina Daliri, Hanieh Hadian — also on the 2026-04-26 email

### Three load-bearing feedback rules

These come from past correction events. Follow them.

**Rule N1 — Pair proofs with same-day numerics.**
When working on a proof or theorem, run a numerical sanity check
on the same day, not a week later. **Why:** Phase 0 Day 1 numerics
showed a spurious $2^{-R/d}$ linear regime. That interpretation
propagated through Days 2–4 of proof-drafting and nearly ended up
in the theorem statement as a "two-regime bound." Day 5 was the
first time code got re-run; it revealed the linear regime was an
artifact of using $\bar\tau$ as the quantization center —
Chebyshev-center merge gave clean $2^{-2R/d}$. **How to apply:**
when Sankalp says "prove X" or "derive Y," ask early whether
there's a cheap numerical check. Treat proof-heavy chains without
code as a risk signal.

**Rule N2 — Full verification pass before externalization.**
Before any email to a researcher, AF / Reddit post, send to Garg,
or arXiv submission, do an end-to-end re-derivation pass over
every theorem statement, proof, numeric, cross-reference, and
narrative claim. **Why:** the Day-6 polish of
`theory/toy_theorem_v0.tex` Lemma 2 missed a hypothesis-direction
bug ($\Pr[\|Y\|^2 \le \cdot]$ where the proof needed
$\Pr[\|Y\|^2 \ge \cdot]$) because polish is *re-reading*. The
2026-04-21 verification pass caught it because it was
*re-deriving*. **How to apply:** before externalizing a proof,
**re-derive** each step. Re-reading catches typos; re-deriving
catches sign / direction errors.

**Rule N3 — PDF-only externalization.**
When sharing the paper publicly (GitHub preprint repo, Alignment
Forum, email attachments), compile to PDF and share the PDF only.
Do **not** push `.tex` source publicly. **Why:** source files
expose work-in-progress notation, TODOs, private comments. **How
to apply:** filter any external push for `*.pdf`. The
`preprint_repo/` staging folder is PDF-only by convention.
Numerics code (`.py`) is a separate call — decide per-case.

---

## §2. Project one-pager

**Working title:** *A Rate-Distortion Lower Bound for Model Merging,
with Matching Achievability via Hadamard Incoherence.*

**Thesis:** Given $T$ task-specific LoRA adapters over a shared
base, derive a Shannon-style lower bound on bits-per-parameter
required to store a merged model that preserves $\varepsilon$-distortion
on each task, via a Fano / packing argument analogous to TurboQuant
Theorem 3 (arXiv 2504.19874). Achievability via Hadamard rotation
plus uniform scalar quantization.

**Why this paper:** no prior work frames LoRA merging as
rate-distortion. The rotation-based merging space is crowded
(KnOTS, TSPA, DO-Merging, ARM, Core Space, TVQ, 1bit-Merging) but
none prove a bits-per-parameter lower bound.

**Folder history:** named `TurboQuant/` Apr 20–21 2026 (after the
source paper whose Thm 3 template this paper adapts), renamed to
`rdmerge/` on 2026-04-21. Old transcripts live under
`.claude/projects/D--my-research-papers-i-see-god-TurboQuant/`
and contain Phase 0 Days 1–6.

**Timeline:**
- Started: 2026-04-20
- ICLR 2027 deadline: ~2026-09-24
- Backups: TMLR (rolling), AISTATS 2027 (early October 2026)
- 22-week solo program, seven phases per `plan.md`

When Sankalp says "the theorem," "the bound," "Phase N," or "the
paper," he means this project.

---

## §3. Current status as of 2026-05-17

**Phase 0 — CLOSED (2026-04-21).** Toy theorem
$D(R) \geq R_c^2 + c_{\mathrm{TQ}}(B^2/T) \cdot 2^{-2R/d}$.
Decision-gate row 1 passed.

**Phase 1 — CLOSED (2026-04-30).** Theorem 7 ($H_t = P_{V_t}$),
Theorem 8 (general $H_t$, task-dependent $D_t$), Lemma 6
(floor closed-form) all in `theory/theorem_v1.tex`.

**Phase 2 — CLOSED (2026-04-24 AM).** Theorem 9 (matching
achievability, floor-zero Stiefel) in `theorem_v1.tex` §8 with
constant $C = T c^2 / 3$. Shared-V null-space split for $T=2$
validated: slope $-1.60 \pm 0.10$ across 5 anisotropy regimes via
`day14b_cheb_T2_closedform.py` + `day14c_fractional_bits.py` +
`day14g_final_lock.py` (the validated quantizer config:
$c = 11.5\sigma_{pc}$, 1000 trials, bootstrap CI $\pm 0.010$).

**Phase 2.5 — PARTIALLY CLOSED (2026-04-24 PM).**
- **Item A (DONE).** General-T Chebyshev solver: `cvxpy` SOCP +
  Gauss-Newton KKT refinement in `day15_cheb_general_T.py`. KKT
  residuals at $10^{-15}$. Extended null-space split to $T=3, 4$;
  slopes match the linear prediction $-2r/(r + |A| - 1)$ after
  switching to **excess-over-cheb² per-trial** metric. (The
  natural "excess-over-avg-floor" metric flattens at high $R$
  when cheb² $\neq$ avg_floor; do not use it for $T \geq 3$
  anisotropy shared-V.)
- **Item B1 (RULED OUT).** Sharpening the LB to match the linear
  UB exponent $-2r/(r + |A| - 1)$ is empirically ruled out for
  $T \geq 3$. A valid rate-$R$ random-codebook encoder
  (`day16_sharpened_lb_check.py`, no data-dependent shift) achieves
  slope $-1.45$ at $T=3, r=3$ vs the linear UB at $-1.20$, which
  would violate any LB at that exponent. True RD lies strictly
  between $-2r/r$ (loose Thm 8 LB) and $-2r/(r + |A| - 1)$ (loose
  linear UB). **Don't re-attempt this direction.** Exact RD is
  open but out of Phase 2/3 scope.

**Phase 3 — NEXT (your job).** Real-LLM empirical validation.
Design locked in `notes/phase3_design.md`. No implementation code
exists yet.

**arXiv v1 — pre-launch.** ~4–5 hrs active work per
`arxiv_checklist.md` (abstract polish, experiments prose, BibTeX
verification, retitle, endorsement). Endorsement is the blocker.
BibTeX verification details in `ARXIV_TODO.md` (10 high/medium
priority entries needing manual lookups).

**External feedback as of 2026-05-02 (end of daily log):**
- Email to Zandieh / Mirrokni / Daliri / Hadian sent 2026-04-26.
- Alignment Forum research note posted same date.
- No replies recorded as of last log entry. Check
  `notes/feedback_received.md` for anything that arrived between
  2026-05-02 and 2026-05-17.

---

## §4. The Phase 3 model-track decision (DO FIRST)

**This is the most important section.** This session's local-machine
constraint (8 GB VRAM on a 3060 Ti) forced a specific choice that
**may or may not apply on your hardware**. Walk the decision tree
yourself with actual `nvidia-smi` output in hand. Do **not** commit
to a model track without explicit confirmation from Sankalp.

### §4.1. Original plan (per `notes/phase3_design.md` §2)

- **Models:** Qwen2.5-1.5B-Instruct (primary), Gemma-2-2B-Instruct
  (secondary)
- **Training:** full bf16 LoRA, rank 16, target Q/K/V/O of every
  attention block, AdamW lr 5e-5, 2 epochs, batch 8 with grad-accum
  to effective 32, seq 2048
- **VRAM required:** ≥24 GB (RTX 3090/4090 / A100-40GB)
- **Why these models:** §2 of phase3_design rationale — fit in 24 GB
  at full bf16, both have instruct variants, different architectures
  (Qwen GQA vs Gemma-2 hybrid sliding-window/global attention)
- **Phase3 §8 Q3 fallback:** "Just Qwen for v1, add Gemma if there's
  slack before submission"

### §4.2. This session's pivot (2026-05-17, hardware-driven)

When Sankalp asked about running the experiment locally on his
3060 Ti 8 GB, we walked through three points:

1. The 24 GB target of the original plan does **not** fit at
   batch 8. Even at micro-batch 1 + grad-accum 32 + FA2 +
   gradient checkpointing, the 8 GB card is tight.
2. Unsloth's nf4 (bitsandbytes 4-bit) pipeline halves the base
   memory footprint and adds ~2× speedup. With Unsloth,
   Llama-3.1-8B-Instruct fits in ~6.5–7 GB during LoRA training
   at seq 2048 micro-batch 1.
3. An 8B-class single-model story is **genuinely stronger for
   ICLR reviewers** than 1.5B + 2B. The merging-methods
   literature (TIES, DARE, KnOTS, TVQ) almost all benchmarks on
   7B-class.

The session's recommended track if forced onto consumer hardware:
**Llama-3.1-8B-Instruct via Unsloth nf4 as v1 single model;
Qwen2.5-7B-Instruct as v2 robustness add.** This is option 3 of
phase3_design §8 Q3 — bigger single model rather than smaller.

### §4.3. Decision tree (run this on day one)

```
Run nvidia-smi, sinfo, squeue --me to identify GPU type / count / VRAM.

A) Single GPU, A100/H100 40–80 GB
   → Original Qwen2.5-1.5B + Gemma-2-2B plan works trivially at full
     bf16. BUT the 8B-class single-model story is stronger for ICLR.
     Surface both to Sankalp:
       (a) Original Qwen+Gemma plan, full bf16 — most faithful to
           phase3_design.md
       (b) Llama-3.1-8B-Instruct at full bf16 (no nf4 needed because
           VRAM is plentiful), Qwen2.5-7B as v2 — stronger reviewer
           story; this session's preferred option translated to
           full precision
     Do not pick silently.

B) Multi-GPU node (4–8× A100/H100)
   → Best of both. Headline plot from Qwen+Gemma at full bf16
     (faithful to design); robustness panel from Llama-3.1-8B + Qwen2.5-7B
     for the stronger story. Parallelize the eval matrix (240 combos
     in phase3_design §5) across GPUs. Surface options to Sankalp.

C) Smaller single GPU (V100, T4, 3090, 4090, single A10)
   → Fall back to this session's nf4 plan: Llama-3.1-8B-Instruct via
     Unsloth nf4 as v1; Qwen2.5-7B as v2.
   → The quantized-base caveat in §5 below now applies — add the
     justifying paragraph to paper/sections/experiments.tex.
```

### §4.4. Unsloth model shortlist (only relevant under track C)

VRAM estimates assume nf4 base + bf16 LoRA + Adam state + FA2 +
gradient checkpointing, micro-batch 1, seq 2048.

**Tier 1 — primary**

| Model | Unsloth Hub ID | Train VRAM |
|---|---|---|
| Llama-3.1-8B-Instruct | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | ~6.5–7 GB |
| Qwen2.5-7B-Instruct | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | ~5.5–6.5 GB |

**Tier 2 — smoke test only (do not put in headline numbers)**

| Model | Unsloth Hub ID | Train VRAM |
|---|---|---|
| Llama-3.2-3B-Instruct | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | ~3 GB |
| Llama-3.2-1B-Instruct | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | ~1.5 GB |

**Tier 3 — skip**

- **Gemma-2-9B-Instruct** — at the 8 GB edge; recurring FA2
  bnb-4bit kernel bugs with hybrid sliding-window attention. Not
  worth debugging on a consumer card.
- **Mistral-7B-Instruct-v0.3** — fits, but architecturally close
  to Llama. Marginal diversity gain over Llama + Qwen.
- **Qwen2.5-Math / Qwen2.5-Coder** — specialized variants bias
  per-task NLL and break the 4-task comparison.
- **Phi-3 / Phi-4** — phase3_design §2 explicitly skipped Phi for
  sliding-window Fisher-estimation complications.
- **DeepSeek-R1-Distill-\*** — reasoning-tuned, NLL evaluation
  becomes noisy and incomparable.
- **Anything ≥ 12B** (Mistral-Nemo-12B, Phi-4-14B, Qwen2.5-14B,
  Gemma-2-27B) — won't fit in 8 GB even in nf4. On HPC, revisit.

---

## §5. Quantized-base caveat (applies if you end up on track C)

If you end up on the Unsloth nf4 track, you owe the paper one
extra paragraph in `paper/sections/experiments.tex`. The story to
tell:

1. **What's compressed.** The rate budget is over LoRA *deltas*
   $\tau_t$, not the base $\theta_0$. The base is shared and held
   fixed; its quantization choice is part of the eval setup, like
   the tokenizer.
2. **Why the metric is still valid.** The excess-NLL metric
   $\Delta L_t(w) = L_t(\theta_0^{(Q)} + w) - L_t(\theta_0^{(Q)} + \tau_t)$
   cancels base-quantization noise because the *same* quantized
   base appears in both terms.
3. **Reviewer-anticipated objection.** "Is your $H_t$-norm
   well-defined when $\theta_0$ has a quantization grid?" Answer
   template: "$H_t$ is used only as the local quadratic of the
   smooth-in-LoRA-params loss; the base quantization is held
   fixed and is a constant of the eval setup. The bound only
   predicts the *shape* (floor + $2^{-2R/d_{\mathrm{eff}}}$ decay)
   — that shape comes from the rate-distortion structure of the
   deltas, not the base."
4. **Reframe as feature, not bug.** The empirical $d_{\mathrm{eff}}$
   you measure with a quantized base is the $d_{\mathrm{eff}}$ for
   *quantized-base LoRA*, which is the relevant quantity for the
   practical deployment scenario most LoRA merging is actually
   deployed in.

---

## §6. Files to read, in priority order

All paths are relative to the project root,
`D:\my research papers\i see god\rdmerge\` on Sankalp's local box.
On the HPC, replace with whatever path the project lives at after
`git clone` / `rsync`.

### P0 — read on day one before doing anything

| File | Lines | Purpose |
|---|---:|---|
| `handoff.md` | (this file) | orientation |
| `target.md` | 364 | 2026-04-24 strategic reality check; venue ranking; the four gaps that block ICLR-readiness |
| `notes/phase3_design.md` | 361 | Phase 3 operational spec; metric / models / tasks / baselines; §8 open questions |

### P1 — read when working on Phase 3 implementation

| File | Lines | Purpose |
|---|---:|---|
| `theory/theorem_v1.tex` | 975 | full theorem statements + proofs; Phase 3 numerics are validating *these* objects |
| `paper/sections/setup.tex` | 134 | admissible task vectors, worst-task distortion, rate-R merging codes (formal objects) |
| `paper/sections/experiments.tex` | 81 | synthetic results already written; real-LLM section is the deferred-to-v2 placeholder you fill in |

### P1 — read when working on arXiv prep

| File | Lines | Purpose |
|---|---:|---|
| `arxiv_checklist.md` | 151 | content blockers + submission mechanics; ~4–5 hrs active work |
| `ARXIV_TODO.md` | 125 | 10 high/medium priority BibTeX entries needing manual verification (~1–2 hrs) |

### P2 — reference / context

| File | Lines | Purpose |
|---|---:|---|
| `plan.md` | 417 | master 22-week roadmap, Phases 0–5, risk table, venue calendar |
| `notes/daily_log.md` | 1361 | Phase 0–2.5 day-by-day; ends 2026-05-02 (15-day gap to handoff date) |
| `notes/open_questions.md` | 255 | tracker; final entry is Day 15–16 shared-V sharpening update |
| `notes/feedback_received.md` | 24 | external feedback log (likely sparse) |
| `paper/main.tex` | 75 | paper skeleton; section files are mostly stubs |
| `paper/references.bib` | — | bibliography (subject of ARXIV_TODO) |

### P3 — code style reference for Phase 3 implementation

Phase 3 code should follow these patterns: modular quantizers,
per-config trial loops with seed control, bootstrap CI reporting,
PNG figures alongside the script.

| File | Pattern it sets |
|---|---|
| `code/synthetic/day14g_final_lock.py` | validated T=2 Chebyshev with $c = 11.5\sigma_{pc}$, 1000 trials, bootstrap CI $\pm 0.010$ — the canonical sanity-check template |
| `code/synthetic/day15_cheb_general_T.py` | cvxpy SOCP + Newton KKT refinement for general $T$ — pattern for future synthetic work |
| `code/synthetic/day16_sharpened_lb_check.py` | random-codebook pattern — also documents *why* the $-2r/(r+|A|-1)$ sharpening direction is ruled out |

---

## §7. Auto-memory system & how to recover it

**Local memory dir:**
`C:\Users\patha\.claude\projects\D--my-research-papers-i-see-god-rdmerge\memory\`

**Contents (6 individual memory files + MEMORY.md index):**

| File | Type | What's in it |
|---|---|---|
| `user_sankalp.md` | user | identity, technical level, contacts, paper vs Claude system emails, Garg co-author status |
| `project_rdmerge_paper.md` | project | thesis, venue, timeline, folder history |
| `project_phase_status.md` | project | Phase 0–2.5 status snapshot — slightly stale (20 days old at handoff date) |
| `feedback_numerics_with_proofs.md` | feedback | Rule N1 above |
| `feedback_verify_before_externalizing.md` | feedback | Rule N2 above |
| `feedback_pdf_only_externalization.md` | feedback | Rule N3 above |

### Recovery on the HPC

The new instance creates a NEW per-project memory dir (different
project ID because the filesystem path differs). It starts empty.
Two options:

1. **(Recommended) Memory copy.** Sankalp identifies the HPC's
   per-project memory dir on first run (typically
   `~/.claude/projects/<project-id>/memory/`) and `rsync`s the
   six local memory files + MEMORY.md into it. The new instance
   then reads memory normally and any future memory writes are
   linked into the existing graph.
2. **(Fallback) Inline-only.** All critical memory content is
   inlined in §1 and §2 of this handoff. New instance can operate
   without memory copy, but won't auto-save new memories that link
   back to the existing ones via `[[name]]` references. Future
   memories will form a new disconnected component.

If unsure, ask Sankalp before assuming.

---

## §8. What NOT to do

- **Don't re-attempt LB sharpening at exponent $-2r/(r + |A| - 1)$**
  for $T \geq 3$. Ruled out empirically by
  `day16_sharpened_lb_check.py`. The true RD lies strictly between
  $-2r/r$ and $-2r/(r + |A| - 1)$ but characterizing it exactly is
  out of Phase 2 / 3 scope.
- **Don't use the "excess-over-avg-floor" metric** for $T \geq 3$
  anisotropy shared-V — flattens at high $R$. Use
  excess-over-cheb² per-trial.
- **Don't pick specialized model variants** (Qwen2.5-Math,
  Qwen2.5-Coder, DeepSeek-R1-Distill) for the 4-task comparison —
  they bias per-task NLL and break the comparison.
- **Don't share `.tex` files externally** — PDFs only (Rule N3).
- **Don't externalize anything without a full verification pass**
  (Rule N2). Re-derive, don't re-read.
- **Don't claim real merges saturate the bound.** The bound is a
  worst-case lower envelope over the random-task-vector
  distribution $P^\star$; real LoRA tasks are not Stiefel-random
  and sit *above* the envelope. The claim is that the bound's
  *qualitative predictions* (floor exists, $-2$ rate decay, better
  methods get closer) hold up.
- **Don't commit to a Phase 3 model track without explicit
  confirmation** from Sankalp after walking §4.3.
- **Don't run unsupervised long training jobs without a smoke
  test first.** §10 step 6.
- **Don't put `kittuwastaken@gmail.com` on the paper or arXiv
  submission** — that's the Claude system email. The paper
  contact is `pathaksankalp04@gmail.com`.

---

## §9. Open questions to surface to Sankalp

From `notes/phase3_design.md` §8, still open as of session end:

| # | Question | Session recommendation |
|---:|---|---|
| Q1 | Metric: NLL only, or NLL + accuracy? | Both, NLL primary (§1.2 of phase3_design) |
| Q2 | Tasks: 4 vs 3 vs 5? | 4 |
| Q3 | Models: Qwen + Gemma, or just one? | Hardware-conditional — see §4.3 |
| Q4 | Rate axis: bits/param averaged, or per-layer? | Averaged headline, per-layer supplementary |
| Q5 | Synthetic Stiefel-random control panel? | Yes, one panel |
| Q6 | Split Phase 3 into its own paper? | Now that Garg is co-author (2026-04-25), this can be a Garg conversation. Surface before doing major Phase 3 work. |

Also surface to Sankalp early:

- **Is the local memory dir being copied to the HPC?** (§7
  recovery options)
- **HPC hardware specs after `nvidia-smi`** — determines §4.3
  branch.
- **Is there reply traffic from the 2026-04-26 email or AF post?**
  If yes, log to `notes/feedback_received.md` and verify any new
  claims that need responses.

---

## §10. First-day checklist for the new instance

Execute in order. Do not skip.

1. **HPC inventory.** Run `nvidia-smi`, `sinfo`, `squeue --me`.
   Report GPU type / count / VRAM / queue policy to Sankalp.
2. **Project files.** Confirm `git status` clean and `ls` shows
   the expected top-level layout (`code/`, `paper/`, `theory/`,
   `notes/`, `plan.md`, `target.md`, `notes/phase3_design.md`,
   this `handoff.md`).
3. **Memory recovery.** Ask Sankalp whether he wants to copy the
   local memory dir (§7). If yes, locate the HPC per-project
   memory path and `rsync`. If no, proceed using §1 + §2 inlines.
4. **Read the P0 set** (this file + `target.md` + `notes/phase3_design.md`).
   ~30 minutes.
5. **Read §4–§5 of this handoff carefully.** Internalize the
   model-track decision tree.
6. **Walk §4.3 with actual HPC specs in hand.** Surface the model
   choice to Sankalp. Do **not** commit silently. Confirmed
   answers feed §11 logging and the next step.
7. **Smallest possible smoke test FIRST.** Once a model is chosen,
   train one LoRA on Llama-3.2-1B (or the smallest model that
   runs in the chosen stack) on a 100-example subset of GSM8K,
   run a 2-task merge with Task Arithmetic, eval NLL on 100
   held-out examples. <1 hr. This catches every integration bug:
   tokenizer, PEFT format, FA2 fallback, eval data loading,
   merging implementation API.
8. **Only after smoke test passes, scale up** to the chosen model
   and the full 4-task / 5-method matrix.
9. **In parallel, if you have idle cycles:** arXiv v1 prep is
   ~4–5 hrs of mostly-unblocked work — see §6 P1 list. Surface
   to Sankalp whether he wants you to interleave this with Phase
   3 setup.

---

## §11. Communication & artifact rules

- **Daily progress** appends to `notes/daily_log.md`. Don't
  rewrite past entries. Date-stamp each new one
  (absolute dates: `2026-05-21`, never relative "Thursday").
- **Open theoretical / empirical questions** go in
  `notes/open_questions.md`, with date stamps and resolution
  status.
- **External feedback** (replies to the 2026-04-26 email, AF
  replies, Garg sessions, any future cold-emails) goes in
  `notes/feedback_received.md`. Quote what they said; note what
  changed in response.
- **Phase 3 code** should live under `code/phase3/` (does not
  exist yet — create on first commit). Subdirs:
  - `code/phase3/training/` — per-task LoRA training
  - `code/phase3/merging/` — implementations / wrappers for the
    5 merging baselines
  - `code/phase3/eval/` — NLL + secondary metric eval loops
  - `code/phase3/figures/` — plots
  - `code/phase3/configs/` — per-model / per-task config YAMLs
  Follow the `day14g` / `day15` style (modular, seed-controlled,
  bootstrap CI, results saved to JSON alongside the script).
- **Compiled paper** lives at `paper/main.tex` → PDF (via your
  preferred local LaTeX build, e.g. `latexmk -pdf`). The PDF
  artifact that ships externally lives at
  `preprint_repo/` per Rule N3.
- **Anything you'd consider an external send** (PDF, email draft,
  AF post draft) → run Rule N2 first.

---

## §12. Glossary / mental model

Use these definitions consistently. They appear in the theorem
statements and the paper prose.

| Symbol | Meaning |
|---|---|
| $\tau_t \in \mathbb{R}^d$ | task-$t$ LoRA delta (the rank-$r$ update for task $t$) |
| $w \in \mathbb{R}^d$ | merged delta produced by a rate-$R$ merging code |
| $\theta_0$ | shared base model parameters |
| $H_t \succeq 0$ | task-$t$ loss curvature (Fisher information matrix at $\theta_0$); $H_t$-norm measures distortion's impact on task-$t$ loss via local-quadratic bridge |
| $\Delta L_t(w)$ | excess NLL on task $t$: $L_t(\theta_0 + w) - L_t(\theta_0 + \tau_t)$ |
| $\widehat D(w) = \max_t \Delta L_t(w)$ | worst-task distortion (the Phase-3 operational metric) |
| $V_t$ | top-$r$ right-singular subspace of $\tau_t$; $P_{V_t}$ its orthogonal projector |
| $d_{\mathrm{eff}} = \mathrm{rank}(\sum_t P_{V_t})$ | effective support dimension (the knob the theory bound depends on) |
| $\bar H = \frac{1}{T}\sum_t H_t$ | arithmetic-mean curvature |
| $P^\star$ | worst-case "hard" distribution over task vectors used in the LB (Stiefel-random subspaces $V_t$) |
| $R$ | total bits in the merge code |
| $b = R / \sum_\ell r(d_{\mathrm{in},\ell} + d_{\mathrm{out},\ell})$ | bits per LoRA parameter (Phase-3 x-axis) |
| floor term | $B^2(1 - d_{\mathrm{eff}}/(Tr))$ — non-vanishing distortion floor when subspaces overlap |
| rate term | $2^{-2R/d_{\mathrm{eff}}}$ — exponential decay in $R$ once above the floor |

**Three Phase-3 validation criteria** (from `phase3_design.md` §6):

1. **Floor reproduces qualitatively.** At $b = 32$ (no quantization),
   there is a non-zero $\widehat D$ across all 5 merging methods,
   and the floor shrinks on tasks with higher empirical
   $d_{\mathrm{eff}}$.
2. **Rate exponent reproduces.** For TVQ at $b \in \{1, 2, 4, 8\}$,
   excess over the $b = 32$ floor decays at slope $\approx -2$ in $b$.
3. **Method ordering matches subspace-respecting hypothesis.**
   KnOTS (subspace-aware) beats Task Arithmetic (subspace-blind)
   on low-$d_{\mathrm{eff}}$ task sets; the gap narrows as
   $d_{\mathrm{eff}} \to Tr$.

If any criterion fails, **report it honestly** as a limitation and
frame the paper as a clean theoretical result (Rule N2 applies —
don't paper over failures). Criterion (1) is the bare minimum: if
no floor shows up at all, the theory is decoupled from practice in
a way reviewers will punish.

---

## §13. Provenance

This handoff was assembled in a Claude Code session on Sankalp's
local Windows workstation on **2026-05-17**. The session began with
the question "if I want to run the complete experiment from my
system itself, how long do you think it's gonna take?" and
concluded with the request to write this file for handoff to an
HPC instance.

Key session-derived content:
- §4.2, §4.3, §4.4 — model-track recommendation conditional on
  hardware, derived in this session from VRAM math for nf4 LoRA
  training on 8 GB consumer cards.
- §5 — quantized-base caveat for the nf4 track, derived in this
  session from the theory's rate budget being over deltas (not
  the base).

All other content (project state, memory rules, file inventory,
glossary) is sourced from the files referenced above. Where this
file disagrees with a referenced file, **trust the referenced
file** — it is authoritative; this handoff is a pointer.

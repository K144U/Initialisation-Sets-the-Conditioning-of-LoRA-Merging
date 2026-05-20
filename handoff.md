# Handoff to the next Claude Code instance

**Written:** 2026-05-18 evening by the Claude instance that bootstrapped
Phase 3 on the JIIT HPC cluster.
**For:** the new Claude Code instance that will continue from where Phase 3
empirical work paused (n=1k v2 matrix running, gemma-3 download running,
T1.C/D/E pending) and eventually move into paper drafting.
**Scope:** full continuation. Phase 3 v2 results, T1.A2/B/C/D/E execution,
paper drafting per `log.md` §6 narrative, any Phase 4–5 work from `plan.md`.

The previous bootstrap handoff is preserved at
`handoff_v1_phase3_bootstrap.md` as historical reference. It captures the
state on 2026-05-17 before any Phase 3 code was written. Read it ONLY if
you need to understand the original architectural decisions; this file is
the live one.

---

## §0. READ THIS FIRST (5-minute orientation)

You are stepping into a solo theoretical-ML research project ~24 hours into
its empirical phase. The user is **Sankalp Pathak** with co-author
**Prof. Sanjay Garg**. The paper is *A Rate-Distortion Lower Bound for
Model Merging, with Matching Achievability via Hadamard Incoherence*,
targeting **ICLR 2027** (deadline ~2026-09-24). **No arXiv preprint** —
direct conference submission decided 2026-05-18.

**One-line state (2026-05-18 evening):** theory closed (Phases 0–2.5),
Phase 3 first 20-cell eval matrix run twice (n=200, n=1k buggy, n=1k-v2
in flight), 5 merging methods implemented + 17/17 CPU tests green, d_eff
analysis shows **the bound is loose vs reality by 0.10–0.22 nats/token —
this is the paper's strongest finding** (see §4 of this file or `log.md`
F9), 3 of 4 robustness-panel models downloaded, gemma-3 download running,
v2 matrix running.

**What to do today:** §10 first-day checklist of this file. Don't write
training code without walking the checklist first.

**Single most important file to read for paper writing:**
**`/home/sanjay.g/projects/rdmerge/log.md`** — 12-section consolidated
reference. Has every finding, fix, speculation, and paper-narrative idea.

**Cardinal rules** (from Sankalp's feedback memory, §1 of this file):
(1) pair proof work with same-day numerics, (2) full re-derivation pass
before any externalization, (3) PDF-only externalization. arXiv was
dropped this session (skip arXiv prep in Day 5–6).

---

## §1. User identity & engagement norms

**Sankalp Pathak** — independent / solo theoretical ML researcher.
Comfortable with Shannon rate-distortion theory, Fano/packing arguments,
Cover & Thomas Ch. 10, LoRA/task arithmetic literature, convex optimization
(SOCP, KKT). Engages at proof-level detail. Says directly when something
feels off ("bro you are stuck", "wait?? what??"). Wants honest assessments
of paper strength, not boosterism.

**Co-author (since 2026-04-25):** Prof. **Sanjay Garg** — advisor-like
contact. "Garg" or "advisor" in any log entry means him.

**Compute:** **JIIT HPC cluster**. 8× NVIDIA A100-SXM4-80GB on
`jiit-gpu01` via PBS `gpu` queue (3-concurrent-job cap per user). Login
node `jiit-master` has no GPU. CPU `workq` queue is separate from `gpu`
queue and does not compete with GPU jobs. External HTTPS bandwidth to
HuggingFace caps at ~1.4 MB/s for most repos (Google CDN for gemma-3 hits
~17 MB/s — outlier). PBS does NOT track GPUs as a schedulable resource;
multiple users share each physical GPU. Pin via `CUDA_VISIBLE_DEVICES`
through `utils/gpu_picker.py` at job startup.

**Paper contact email:** `pathaksankalp04@gmail.com`. Put this on the
paper, OpenReview submission, Garg correspondence. **NEVER** put
`kittuwastaken@gmail.com` (the HuggingFace account email for cluster
auth) on the paper.

**Other contacts referenced in the daily log:**
- Amir Zandieh & Vahab Mirrokni — TurboQuant authors at Google Research
  (cold-emailed 2026-04-26, no reply as of 2026-05-02). Our paper extends
  their Theorem 3. See `log.md` §9 for the lineage.
- Sina Daliri, Hanieh Hadian — also on the 2026-04-26 email.

### Three load-bearing feedback rules

These come from past correction events. Follow them.

**Rule N1 — Pair proofs with same-day numerics.**
When working on a proof or theorem, run a numerical sanity check on the
same day. Phase 0 lost 4 days defending a spurious linear regime that a
same-day numerics check would have caught.

**Rule N2 — Re-derive (not re-read) before externalizing.**
Before any email to Garg, OpenReview submission, AF post, or arXiv (if
ever revisited), do an end-to-end re-derivation pass over every theorem
statement, proof, numeric, cross-reference, and narrative claim. The
Day-6 `theory/toy_theorem_v0.tex` Lemma 2 had a hypothesis-direction bug
that a re-read missed and a re-derivation caught.

**Rule N3 — PDF-only externalization.**
Compile to PDF, share the PDF. Do not push `.tex` source publicly.
For ICLR submission this is moot (anonymous OpenReview upload is PDF),
but the rule applies to any drafts shared with Garg or external
researchers.

### Venue decision (2026-05-18): direct to ICLR, NO arXiv preprint

User chose to skip the arXiv preprint stage. Direct submission to ICLR
2027. Files `arxiv_checklist.md`, `ARXIV_TODO.md`, and `preprint_repo/`
are now historical-reference only. **Do not work on them.** Backups in
case ICLR rejects: AISTATS 2027 (~2026-10-08), TMLR rolling.

---

## §2. Project one-pager

**Working title:** *A Rate-Distortion Lower Bound for Model Merging,
with Matching Achievability via Hadamard Incoherence.*

**Thesis:** Given T task-specific LoRA adapters over a shared base, derive
a Shannon-style lower bound on bits-per-parameter required to store a
merged model that preserves ε-distortion on each task, via a Fano/packing
argument analogous to TurboQuant Theorem 3 (arXiv 2504.19874).
Achievability via Hadamard rotation + uniform scalar quantization.

**TurboQuant lineage (cite prominently):**
- TurboQuant gives the single-source Shannon RD bound template (Fano +
  packing + Hadamard-incoherence achievability with constant `c_TQ`).
- Our paper generalizes to the T-source worst-task setting with
  `H_t`-curvature-weighted distortion, derives the new floor term
  `B²(1 - d_eff/(Tr))` capturing subspace interference, proves matching
  achievability, and validates empirically on real LoRA merging.
- Per `theory/theorem_v1.tex` line 54: "T=1 recovers TurboQuant Theorem 3
  on a rank-r source." Our Thm 7 IS their Thm 3 in the special case.

**Timeline:**
- Started: 2026-04-20.
- ICLR 2027 deadline: ~2026-09-24.
- 22-week solo program; about week 5 of empirical work as of this handoff.

**Folder history:** named `TurboQuant/` Apr 20–21 2026, renamed to
`rdmerge/` on 2026-04-21. Old `TurboQuant/` Claude transcripts live under
`.claude/projects/D--my-research-papers-i-see-god-TurboQuant/` if you
need ancient context.

When Sankalp says "the theorem," "the bound," "Phase N," or "the paper,"
he means this project.

---

## §3. Current status as of 2026-05-18 evening

**Phase 0 — CLOSED (2026-04-21).** Toy theorem.

**Phase 1 — CLOSED (2026-04-30).** Theorem 7 (`H_t = P_{V_t}`),
Theorem 8 (general `H_t`, task-dependent `D_t`), Lemma 6 (closed-form
floor) all in `theory/theorem_v1.tex`.

**Phase 2 — CLOSED (2026-04-24).** Theorem 9 (matching achievability,
floor-zero Stiefel) with constant `C = Tc²/3`. `T=2` shared-V slope
$-1.60 \pm 0.10$ validated.

**Phase 2.5 — PARTIALLY CLOSED (2026-04-24).** General-T Chebyshev solver
done. LB-sharpening at exponent $-2r/(r+|A|-1)$ ruled out for $T \geq 3$
via `code/synthetic/day16_sharpened_lb_check.py`. **Don't re-attempt
this direction.** Exact RD is open but out of paper scope.

**Phase 3 — IN PROGRESS (Days 0–4 done, 2026-05-17 → 2026-05-18).**

Done:
- HPC bootstrap; project conda env at `.conda/envs/rdmerge` (torch
  2.10.0+cu128, transformers 5.5.0, peft 0.19.1, trl 0.24.0, accelerate
  1.13.0, datasets 4.3.0, unsloth 2026.5.2 — versions overrode plan's
  pins because Unsloth's transitive deps demand newer).
- **8 LoRA adapters trained** (2 models × 4 tasks): Llama-3.1-8B-Instruct
  and Qwen-2.5-7B-Instruct × GSM8K, Alpaca, Magicoder, wmt19-de-en. Full
  bf16 via Unsloth. Adapters at `artifacts/lora/{model}/{task}/v1/`.
- **5 merging methods** in `code/phase3/merging/`: task_arithmetic, ties,
  dare, knots, tvq. KnOTS from arXiv:2410.19735 §3 directly (no GitHub
  port). TVQ uniform scalar at `b ∈ {1,2,4,8,16,32}`.
- **17/17 CPU unit tests green** (`code/phase3/merging/tests/test_synthetic.py`).
- **First 20-cell eval matrix at n=200** (`results/phase3/eval_matrix/`).
- **Rerun at n=1k (19/20 cells, llama_knots walltime-killed)** —
  confirmed: TIES > TA robust, b=2 dip is REAL, Qwen ≪ Llama is REAL.
- **CRITICAL BUG found and fixed (2026-05-18):** `data_loaders.py` used
  two independent shuffle seeds for train vs eval when both came from the
  same split. Caused 13% (alpaca) / 7% (magicoder) train/eval overlap.
  **Fixed; verified 0/200 overlap post-fix.** The 8 LoRAs are FINE
  (training data was clean), but the n=200 and n=1k eval matrices have
  biased NLL_τ for alpaca/magicoder. The v2 rerun (in flight) uses the
  fixed loader.
- **d_eff analysis (T1.A) complete:** `d_eff = Tr = 64` in every single
  layer of both models. Bound's floor formula predicts 0; observed 0.10
  (Qwen) – 0.22 (Llama). **The 0.10–0.22 gap is the paper's strongest
  finding.** Reframes contribution as "we prove a bound; real LoRA
  merging is well above it; gap is a quantified call to action for
  better mergers."
- **Synthetic Stiefel-random control panel (T1.B) script written**, not
  yet run. Ready when needed.
- **Robustness-panel models downloaded:** Mistral-7B-Instruct-v0.3 (14
  GB), Yi-1.5-9B-Chat (17 GB), Qwen-2.5-14B-Instruct (28 GB). gemma-3-12b
  download in flight on workq (60% done, ~10 min remaining).
- **Documentation complete:** `log.md` is the consolidated reference,
  `notes/phase3_findings.md` for empirical state, `notes/daily_log.md`
  chronology, `notes/open_questions.md` action items.
- **Paper artifacts dir** `paper_artifacts/{figures,data}/` (NOT
  gitignored) holds the durable copies of figures and JSON data.

**Phase B validation criteria (per `notes/phase3_design.md` §6):**
- C1 floor exists at b=32: **PASS** (Llama 0.225, Qwen 0.108)
- C2 TVQ slope ≈ -2: **SKIPPED** (data doesn't support; quantization
  isn't the dominant excess source at b ≥ 1; reframe in paper)
- C3 KnOTS > TA per-task: **PASS** (5/8 cells)
- **Decision-gate outcome: ICLR 2027 viable.** Per the plan.

### Three headline findings for the paper

1. **The bound is loose vs reality by 0.10–0.22 nats/token** (F9 in
   `log.md`). d_eff = Tr in every layer; bound predicts floor = 0;
   observed positive. **Lead with this.**
2. **TVQ at b=2 is a real local minimum, model-agnostic** (F7). Llama:
   ½ of adjacent rates. Qwen: ⅕ of adjacent rates, avg_excess NEGATIVE.
   "Less is more" rate-distortion finding.
3. **TIES > TA on every task, both models. Qwen-2.5-7B is ~2× easier to
   merge than Llama-3.1-8B.** Suggests architecture/pretraining affects
   merging-readiness.

---

## §4. What's running right now (2026-05-18 evening)

When you start, run `qstat -u $USER` first.

| Job | What | ETA |
|---|---|---|
| Various PBS gpu jobs | v2 eval matrix (`launch_eval_matrix.py`, output dir `results/phase3/eval_matrix_n1k_v2/`) — uses fixed data_loaders | ~5 hr to finish all 20 cells |
| `39645` workq | gemma-3-12b download (~24 GB; was at 14 GB / 10 min in) | <30 min |
| Launcher python proc | `launch_eval_matrix.py` in background (look for `pgrep -af launch_eval_matrix`) | exits when v2 finishes |
| Watcher bash | Polls `logs/launch_eval_matrix_n1k_v2.log` for `ALL_EVAL_CELLS_DONE` | exits with launcher |

**Background analyses NOT in PBS:**
- d_eff analysis: done. Output at `results/phase3/deff_analysis.json` and
  `code/phase3/figures/deff_vs_floor.png` (plus durable copy in
  `paper_artifacts/`).

**Pending pipeline (next 24 hr after v2 + gemma-3 land):**
- **T1.A2** — soft-d_eff via participation ratio of stacked-V singular values. Refines F9 with a continuous metric. Pure CPU.
- **T1.B** — run the Stiefel control panel script. ~5 min CPU.
- **T1.C** — train 12 new LoRAs (3 models × 4 tasks): Mistral-7B + Yi-9B + gemma-3-12b. ~3 hr at 3-concurrent gpu. Requires gpu queue.
- **T1.D** — extend eval matrix by 33 new cells. ~9 hr at 3-concurrent gpu.
- **T1.E** — regenerate Phase B with 5-model data + d_eff panel + Stiefel panel. ~30 min.

---

## §5. Files to read, in priority order

### P0 — read on day one before doing anything

| File | Lines | Purpose |
|---|---:|---|
| `handoff.md` (this file) | — | orientation |
| `log.md` | ~530 | **single consolidated paper-writing reference. Sections §1–§12. All findings, bugs, speculations, paper narrative.** |
| `notes/phase3_findings.md` | ~280 | Phase 3 numerical results + 9 qualitative findings |
| `notes/daily_log.md` | last 250 | chronology of Phase 3 sessions; latest entry is the d_eff finding |

### P1 — read when picking up specific work

| File | Lines | Purpose |
|---|---:|---|
| `notes/open_questions.md` | last 200 | actionable open items (Phase 3 Day 17–18 sections + T1 items) |
| `theory/theorem_v1.tex` | 975 | full theorem statements + proofs |
| `notes/phase3_design.md` | 361 | original Phase 3 spec (metric, baselines, validation criteria) |
| `target.md` | 364 | 2026-04-24 strategic context; venue ranking |

### P2 — reference

| File | Lines | Purpose |
|---|---:|---|
| `plan.md` | 417 | master 22-week roadmap, Phase 0–5 |
| `paper/main.tex` | 75 | paper skeleton; section stubs |
| `paper/references.bib` | — | bibliography (TurboQuant, KnOTS, TIES, DARE, TVQ entries verified) |
| `handoff_v1_phase3_bootstrap.md` | 600+ | original handoff before Phase 3 started. Useful for "why was X decided" |

### P3 — code reference for Phase 3 implementation

| Path | What it does |
|---|---|
| `code/phase3/merging/{task_arithmetic,ties,dare,knots,tvq}.py` | 5 method implementations |
| `code/phase3/merging/registry.py` | name → callable mapping + default kwargs |
| `code/phase3/merging/peft_model_view.py` | bridges real PeftModel ↔ FakePeftModel interface |
| `code/phase3/training/train_lora.py` | single-LoRA training driver |
| `code/phase3/training/data_loaders.py` | task-specific dataset loading **(contains the train/eval-disjointness fix from 2026-05-18)** |
| `code/phase3/eval/run_eval_cell.py` | one cell of the eval matrix |
| `code/phase3/eval/deff_analysis.py` | T1.A — d_eff analysis (optimized with svd_lowrank + svdvals) |
| `code/phase3/eval/stiefel_control.py` | T1.B — synthetic Stiefel control panel (script written, not yet run) |
| `code/phase3/eval/phase_b_analysis.py` | regenerate headline figures + criterion checks |
| `code/phase3/scripts/launch_eval_matrix.py` | submitter for the full 20-cell matrix |
| `code/phase3/scripts/pbs_*.sh` | PBS wrappers (train, eval, downloads) |
| `code/synthetic/day14g_final_lock.py` | bootstrap-CI pattern (port reference) |

---

## §6. What NOT to do

- **Don't work on arXiv prep.** Venue decision (2026-05-18) is direct ICLR submission, no preprint. `arxiv_checklist.md`, `ARXIV_TODO.md`, `preprint_repo/` are historical-reference only.
- **Don't re-attempt LB sharpening** at exponent $-2r/(r + |A| - 1)$ for $T \geq 3$. Ruled out empirically.
- **Don't use the "excess-over-avg-floor" metric** for $T \geq 3$ anisotropy shared-V. Use excess-over-cheb² per-trial.
- **Don't use sub-6B models for the headline experiments.** Sankalp explicitly said 6B+ only. Llama-3.2-1B is fine for SMOKE TESTS only (code-correctness, no science).
- **Don't add specialized model variants** (Qwen2.5-Math, Qwen2.5-Coder, DeepSeek-R1-Distill) for the 4-task comparison — they bias per-task NLL.
- **Don't reintroduce the train/eval shuffle bug.** `data_loaders.py` now uses ONE shuffle and disjoint slices when `eval_split == train_split`. Verify with the audit script if you change it.
- **Don't use synthetic data in Phase 3 LoRA training.** All real public datasets: openai/gsm8k, yahma/alpaca-cleaned, ise-uiuc/Magicoder-OSS-Instruct-75K, wmt/wmt19. Synthetic data is for theory validation (`code/synthetic/`) and the Stiefel control panel (`code/phase3/eval/stiefel_control.py`) only.
- **Don't claim "slope ≈ -2" in the paper.** TVQ rate-decay term is not empirically visible at b ≥ 1; merging-geometry error dominates. Frame as "bound's floor structure is what we validate; rate-decay term is below the detection threshold for practical bit budgets."
- **Don't claim "real LoRA merging saturates the bound."** Real LoRA merging sits ABOVE the bound by 0.10–0.22 nats/token (F9). The gap is the paper's main contribution.
- **Don't put `kittuwastaken@gmail.com` on the paper.** That's the HF auth email.
- **Don't use `huggingface-cli`** — deprecated in `huggingface_hub` 1.x (silent no-op). Use `hf` CLI or Python `snapshot_download`. For models, use the `wget --continue` flow in `code/phase3/scripts/dl_model_curl.sh` — bypasses HF Xet protocol which self-throttles to 30 KB/s on this cluster.
- **Don't try to do interactive `qsub -I` sessions** — no TTY available. Use batch jobs with PBS scripts.
- **Don't put downloads on the gpu queue.** Use workq. The gpu queue's 3-concurrent cap is precious; downloads don't need GPU.
- **Don't run d_eff analysis on CPU at full SVD.** It will take ~8 hours. The optimized version uses `svd_lowrank` and `svdvals` on stacked V (in_dim × T·r ≪ in_dim × in_dim). Finishes in ~200 sec.
- **Don't change anything that's currently in flight without checking jobs first.** Run `qstat -u $USER` and `pgrep -af launch_eval_matrix` first.

---

## §7. Open questions to surface to Sankalp (when he checks back in)

From `notes/open_questions.md` Phase 3 Day 17–18 + Tier 1 sections:

- **The 0.10–0.22 nats/token gap** between predicted floor (0) and observed worst_excess. Three candidate interpretations in `log.md` §5.6. **Recommended framing: "bound is for worst-case Stiefel-random; real LoRAs aren't worst-case; the gap is the suboptimality of current merging methods, motivating better algorithms."** Worth pinging Garg for narrative alignment.
- **Should we implement soft-d_eff (T1.A2)?** Hard d_eff hit ceiling = 64 = Tr everywhere. A soft metric (participation ratio) might give continuous per-layer values that correlate with observed excess. Could be a Figure 5.
- **Translation negative excess** — under-trained vs real cross-task benefit. Need to retrain translation LoRA at 15k×3ep before submission.
- **DARE ≈ TA exactly** at density=0.2 — density-sweep ablation needed.
- **Mechanism of the b=2 dip** — three candidates (regularization, stochastic resonance, implicit coarse-projection). Worth a Figure or paragraph; intermediate-rate sweep (b ∈ {1.5, 2.5, 3}) might map out the dip's shape.
- **3-seed reruns?** Currently single-seed. Reviewers may push for error bars. ~3× the compute.

Also:
- Has Garg seen the d_eff finding yet? It's the strongest empirical result and significantly reframes the paper. He should see it before paper draft starts.

---

## §8. First-day checklist for the new instance

Execute in order.

1. **Read `log.md` cover-to-cover.** It's the consolidated paper-writing reference and the single most important file. ~30 min.
2. **Read this file's §1–§4 again.** Internalize current state.
3. **Check what's running:**
   ```
   qstat -u $USER
   pgrep -af launch_eval_matrix
   ls results/phase3/eval_matrix_n1k_v2/ | wc -l
   du -sh models/gemma-3-12b-it
   ```
4. **Verify nothing crashed overnight:**
   - Check `qstat -x JOBID` for any jobs that finished with non-zero exit
   - Check `logs/pbs/*.OU` for the latest few jobs
   - Verify `paper_artifacts/{figures,data}/` is intact
5. **Read `notes/daily_log.md` latest entries** (the Day 18 ones) for chronological context.
6. **Decide next action based on what completed overnight:**
   - If v2 matrix done → run `phase_b_analysis.py` against new data; regenerate figures
   - If gemma-3 done → kick off T1.C (train 12 new LoRAs)
   - If both done → T1.C + T1.B (Stiefel) in parallel
   - If neither done → wait, or work on T1.A2 (soft d_eff, pure code)
7. **Don't add new analyses without checking with Sankalp** unless they're listed in `open_questions.md` or this file's §7.

---

## §9. Communication & artifact rules

- **Daily progress** appends to `notes/daily_log.md`. Date-stamp absolute dates (e.g., `2026-05-19`, never "tomorrow").
- **All findings worth keeping** go in `log.md` (the consolidated paper reference) — update §3, §5, §6 sections as new data lands.
- **Open theoretical/empirical questions** go in `notes/open_questions.md` with date stamps and resolution status.
- **Paper-quality figures** go in TWO places:
  - `code/phase3/figures/` (regenerable, gitignored)
  - `paper_artifacts/figures/` (durable, NOT gitignored)
- **Phase 3 code** lives under `code/phase3/`. Follow the existing module structure.
- **Compiled paper** lives at `paper/main.tex` → PDF (via `latexmk -pdf`).
- **Anything you'd consider an external send** (PDF to Garg, OpenReview submission, etc.) → run Rule N2 first.

---

## §10. Glossary / mental model

Use these definitions consistently. They appear in the theorem statements and the paper prose.

| Symbol | Meaning |
|---|---|
| $\tau_t \in \mathbb{R}^d$ | task-$t$ LoRA delta (the rank-$r$ update for task $t$) |
| $w \in \mathbb{R}^d$ | merged delta produced by a rate-$R$ merging code |
| $\theta_0$ | shared base model parameters |
| $H_t \succeq 0$ | task-$t$ loss curvature (Fisher information matrix at $\theta_0$) |
| $\Delta L_t(w)$ | excess NLL on task $t$: $L_t(\theta_0 + w) - L_t(\theta_0 + \tau_t)$ |
| $\widehat{D}(w) = \max_t \Delta L_t(w)$ | **worst-task distortion** (the Phase-3 operational metric) |
| $V_t$ | top-$r$ right-singular subspace of $\tau_t$; $P_{V_t}$ its orthogonal projector |
| $d_{\mathrm{eff}} = \mathrm{rank}(\sum_t P_{V_t})$ | effective support dimension |
| $\bar H = \frac{1}{T}\sum_t H_t$ | arithmetic-mean curvature |
| $P^\star$ | worst-case "hard" distribution over task vectors used in the LB (Stiefel-random subspaces $V_t$) |
| $R$ | total bits in the merge code |
| $b = R / \sum_\ell r(d_{\mathrm{in},\ell} + d_{\mathrm{out},\ell})$ | bits per LoRA parameter (Phase-3 rate axis) |
| **floor term** | $B^2(1 - d_{\mathrm{eff}}/(Tr))$ — non-vanishing distortion floor when subspaces overlap |
| **rate term** | $2^{-2R/d_{\mathrm{eff}}}$ — exponential decay in $R$ once above the floor |

---

## §11. Provenance

This handoff was written 2026-05-18 evening by the Claude instance that
worked Phase 3 Days 0–4 on the JIIT HPC cluster. The session began with
the previous instance's handoff at 2026-05-17, executed the smoke test
and first real training, then the full 20-cell eval matrix, identified
the TVQ b=2 dip + the d_eff=Tr finding (which reframes the paper), built
out the Tier-1 work (5 architecture families, soft d_eff, Stiefel control
panel) and queued the next pipeline steps before signing off.

Key session-derived content:
- **§3 status** — Phase 3 work through Day 4.
- **§6 don't-do list** — augmented with venue decision, dataset bug, TVQ
  rate claims, d_eff/saturation framing.
- **`log.md`** (full paper-writing reference) — created this session.
- **`paper_artifacts/`** (durable figure+data store) — created this session.

All other content (project state, memory rules, glossary) is sourced from
`handoff_v1_phase3_bootstrap.md` and the `notes/*` files.

Where this file disagrees with a referenced file, **trust the referenced
file** — it is authoritative; this handoff is a pointer.

Welcome aboard. Read `log.md` first.

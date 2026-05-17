# Research Plan: Rate-Distortion Theory of LoRA Merging

**Working title (placeholder):** *A Rate-Distortion Lower Bound for Model Merging, with Matching Achievability via Hadamard Incoherence*

**Author:** Sankalp Pathak (solo, independent)
**Start date:** April 20, 2026
**Target submission:** ICLR 2027 (deadline late September 2026)
**Backup venue:** TMLR (rolling), AISTATS 2027 (early October 2026)
**Compute:** single consumer GPU (RTX 3090/4090 or Colab Pro)

---

## 1. Project overview

### 1.1 Research question
Given T task-specific LoRA adapters over a shared base model, what is the minimum number of bits required to store a single merged model that achieves per-task distortion at most epsilon on each of the T tasks, and does any existing merge method approach this bound?

### 1.2 Core contributions (target)
1. A Shannon-style lower bound on the bits-per-parameter required to merge T task vectors while preserving epsilon-distortion on each task, proved via a Fano or packing argument analogous to TurboQuant Theorem 3.
2. An achievability construction via Hadamard-rotation plus per-coordinate scalar quantization, shown to match the lower bound up to a small constant.
3. An empirical characterization of where existing merge methods (Task Arithmetic, TIES, DARE, KnOTS, TVQ, 1bit-Merging) sit relative to the bound, turning the EMSE empirical work into evidence for a theoretical claim.

### 1.3 Why this paper can exist
The literature search turned up zero papers framing LoRA merging as rate-distortion. Rotation-based merging is crowded (KnOTS, TSPA, DO-Merging, ARM, Core Space all published since October 2024), but none of them prove a bits-per-parameter lower bound. The tools to prove one exist and are sharp: TurboQuant's Yao-minimax argument for vector quantization transfers cleanly to task-vector merging once the distortion measure is pinned down.

### 1.4 The single biggest risk
The lower bound is vacuous or has no matching achievability. This kills the paper. Phase 0 is designed to find this out in week one rather than month three.

---

## 2. Phase 0: Theorem stress test (Week 1: April 20 to April 26)

**Purpose:** decide in seven days whether the theorem actually closes. Everything downstream is contingent on this.

### 2.1 Daily breakdown

**Day 1 (Mon Apr 20):** Read TurboQuant Theorem 3 and its proof end to end. Take notes on the Yao-minimax structure, the packing construction, and the constants that appear. Identify exactly which steps transfer to the merging setting and which do not. The proof is in arXiv 2504.19874, check both v1 and the OpenReview camera-ready version.

**Day 2 (Tue Apr 21):** Read Cover and Thomas Chapter 10 (rate-distortion theory) with focus on the Gaussian source and the converse theorem. This is the classical template for the kind of bound you want to prove. Also skim Berger's *Rate Distortion Theory* (1971), Chapter 2, if you can find a PDF.

**Day 3 (Wed Apr 22):** Read Ortiz-Jimenez et al. "Task Arithmetic in the Tangent Space" (NeurIPS 2023). Their weight disentanglement error is essentially a distortion measure. Note how they define it and whether your bound reduces to a disentanglement statement in any limit.

**Day 4 (Thu Apr 23):** Write the theorem statement. Target the simplest possible setting that is still interesting:
- T tasks
- Task vectors tau_1,...,tau_T in R^d, each with norm at most B
- Per-task loss f_t(w) = (1/2) * ||w - (w_0 + tau_t)||^2 (pure quadratic)
- Merged vector w* must be encodable in R bits total
- Distortion: max_t [f_t(w*) - min_w f_t(w)]
- Goal: show distortion is at least C * B^2 * g(T, d, R) for explicit g and constant C

Write out the theorem in LaTeX. Do not try to prove it yet, just get the statement crisp.

**Day 5 (Fri Apr 24):** Attempt the proof. Strategy:
1. Construct a packing of 2^k well-separated task-vector tuples (tau_1^(i),...,tau_T^(i)) for i = 1,...,2^k. "Well-separated" means any two tuples differ by at least delta in some norm.
2. Show that any merging algorithm M with R < k bits cannot distinguish packing points by pigeonhole.
3. Conclude the worst-case distortion is at least a function of delta.
4. Optimize over the packing to get the tightest bound.

This is a standard Fano / Assouad argument. The key technical step is constructing the packing so that the induced merged-model distortion is provably large.

**Day 6 (Sat Apr 25):** If the proof closed: tighten constants, identify where it breaks under weakened assumptions, write up a clean two-page LaTeX note. If the proof did not close: write up what you tried, where it broke, and at least three alternative formulations (different distortion measure, stronger task-vector assumption, functional rather than weight-space distortion).

**Day 7 (Sun Apr 26):** Externalize. Post the two-page note somewhere you can get eyes on it. Options:
- Email directly to Amir Zandieh (TurboQuant first author) and Vahab Mirrokni at Google Research, with a short note: "I'm a solo researcher trying to apply TurboQuant-style rate-distortion analysis to LoRA merging. Here's a two-page proof sketch. Would appreciate 10 minutes of your time to tell me if I'm missing something obvious."
- Post on LessWrong / Alignment Forum under "research" tag with a "looking for feedback" framing.
- Post on r/MachineLearning with a clear title like "Seeking feedback: rate-distortion lower bound for LoRA merging."
- Send to Prof. Sanjay Garg and any other contact who has a theory background.

### 2.2 Phase 0 deliverable
A two-page LaTeX note (`toy_theorem_v0.tex`) containing:
- Setting and assumptions
- Theorem statement
- Proof (if closed) or proof sketch plus explicit obstacles (if not)
- One open question

### 2.3 Phase 0 decision gate (End of Week 1)

| Outcome | Next action |
|---------|-------------|
| Theorem closes with non-trivial constant | Proceed to Phase 1 as planned |
| Theorem closes but bound is loose / constant is huge | Proceed to Phase 1, but note tighter-constant proof as an open problem |
| Proof does not close, but has a clear path forward | Extend Phase 0 by one more week, try alternate formulations |
| Proof does not close, no path forward | STOP. Reassess. Options: (a) pivot to a purely empirical paper on rotation-based merging (riskier given crowded landscape), (b) pivot to the incoherence-theorem version as a smaller contribution, (c) pick a different problem entirely |

**Do not proceed to Phase 1 without passing this gate.** The biggest failure mode for solo theory work is spending two months on a plausible-sounding theorem with a subtle fatal flaw.

---

## 3. Phase 1: Formalization (Weeks 2 to 3, Apr 27 to May 10)

**Purpose:** move from toy setting to a theorem that actually applies to LoRA.

### 3.1 Goals
1. Extend the toy theorem to rank-r task vectors (actual LoRA structure, not arbitrary d-dimensional vectors).
2. Derive tight constants where possible, document conjecture constants where not.
3. Identify which assumptions are essential and which can be weakened.
4. Start drafting the paper introduction and related work sections.

### 3.2 Key technical questions to answer
- Does the rank-r structure of LoRA updates change the bound? Intuition: yes, because the "effective dimension" is 2rd instead of d^2, which changes the packing count.
- Does the bound depend on T (number of tasks) linearly, logarithmically, or at some intermediate rate?
- Is the bound tight under adversarial task vectors, or only under random / Gaussian task vectors?
- Can the bound be extended from MSE distortion on quadratic loss to cross-entropy distortion on real LLM loss surfaces, at least via local-quadratic-approximation arguments?

### 3.3 Deliverables
- `theorem_v1.tex`: full theorem statement with proof, rank-r extended, constants tracked.
- `intro_draft.tex`: 2-3 pages of introduction and positioning against prior work (KnOTS, TSPA, DO-Merging, ARM, Core Space, TVQ, 1bit-Merging).
- `related_work.tex`: ~100 citations organized by theme.

### 3.4 External feedback checkpoint (end of Week 3)
Share `theorem_v1.tex` with any feedback channel that responded in Phase 0. If no one responded, try a second round including an arxiv preprint or a "work in progress" post on a blog. Low-stakes ways to get a sanity check.

---

## 4. Phase 2: Achievability construction (Weeks 4 to 6, May 11 to May 31)

**Purpose:** show that a concrete algorithm achieves the lower bound up to a constant.

### 4.1 The construction (borrowed and adapted from TurboQuant)
1. Apply a Hadamard rotation H (or Walsh-Hadamard transform) to each task vector: tilde_tau_i = H * tau_i.
2. Quantize each coordinate of tilde_tau_i using a per-coordinate Lloyd-Max scalar quantizer designed for the post-rotation Beta-distributed coordinates.
3. Average or otherwise merge the quantized representations in the rotated space.
4. Invert: merged model = H^T * merged_quantized + w_0.

### 4.2 Goals
- Prove that this construction achieves total distortion matching the lower bound up to a small constant factor (ideally <= 4 * lower_bound, ambitious goal <= 2 * lower_bound).
- Identify what changes if Hadamard is replaced by Haar-random orthogonal (likely tighter constant but slower).
- Identify what changes if the averaging step is replaced by more sophisticated merge operators (Task Arithmetic, TIES, DARE applied in rotated space).

### 4.3 Small-scale validation
Synthetic experiments only at this stage, no real LLMs yet:
- Generate T random task vectors in R^d for d in {128, 512, 2048, 8192}, T in {2, 4, 8, 16}.
- Run the construction, measure empirical distortion.
- Plot empirical vs theoretical distortion across the sweep.
- Verify the scaling matches the theorem's predictions.

Code target: ~500 lines Python + NumPy, no LLM libraries needed yet. Should run on CPU.

### 4.4 Deliverables
- `achievability.tex`: proof of the upper bound.
- `synthetic_experiments/`: Jupyter notebook with empirical validation, plots saved as PDF for paper.
- `theorem_v2.tex`: merged document with both lower bound and achievability.

---

## 5. Phase 3: Experimental validation on real LoRA merging (Weeks 7 to 10, Jun 1 to Jun 28)

**Purpose:** show that the theoretical predictions hold on actual LLM LoRA merging, and that the rotation-plus-quantization construction is competitive with state-of-the-art merge methods.

### 5.1 Models (feasible on single RTX 3090/4090)
Primary: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B (smallest for iteration, 3B for final results).
Secondary if time: TinyLlama-1.1B, Phi-2, Gemma-2B.

### 5.2 Tasks
Mix of existing fine-tuned LoRA adapters from HuggingFace Hub plus self-trained ones:
- Code: HumanEval, MBPP (your existing EMSE pipeline)
- Math: GSM8K
- Reasoning: ARC, HellaSwag
- NLP: a GLUE subset (SST-2, MNLI)

Train one LoRA adapter per task per base model. Rank r in {8, 16, 32, 64}. Keep training minimal (few epochs, small data) since the merging is the contribution.

### 5.3 Baselines (exhaustive list, this is the make-or-break section)
- Task Arithmetic (Ilharco 2023)
- TIES (Yadav 2023)
- DARE (Yu 2024)
- Model Soup / uniform average
- KnOTS (Stoica 2025)
- TSPA (2025)
- DO-Merging (2025)
- Core Space (Panariello 2025)
- TVQ (Kim ICCV 2025)
- 1bit-Merging (2025)
- Your Hadamard-rotation + quantization construction at bit rates {1, 2, 3, 4, 8, 16}

### 5.4 Evaluation
- Per-task accuracy / pass@1 / exact-match for each merging method.
- For your construction, also report the effective bit rate.
- Compute the empirical distortion-rate curve and compare to the theoretical bound.
- Statistical rigor: 3 seeds minimum, bootstrap CIs, significance testing (same rigor you used in the QLoRA sentiment paper).

### 5.5 Deliverables
- `experiments/` directory with reproducible scripts for every baseline.
- Results tables for the paper.
- Figures: distortion-rate curve (theoretical vs empirical), accuracy vs bit-rate plot, ablations.

### 5.6 Phase 3 decision gate (end of Week 10)
If your construction matches the theoretical predictions AND is competitive with or beats the best baselines at matched bit rates, the paper is in good shape. If not, figure out why:
- Is the theorem too weak (underestimates distortion for real LLMs)?
- Is the construction suboptimal (Hadamard not the right rotation)?
- Is the baseline implementation wrong?
Fix or descope before moving to writing.

---

## 6. Phase 4: Paper writing (Weeks 11 to 18, Jun 29 to Aug 23)

**Purpose:** write a NeurIPS/ICLR-quality paper.

### 6.1 Structure
1. Abstract (1 paragraph)
2. Introduction (1.5 pages)
3. Related work (1 page, compressed)
4. Problem setup and preliminaries (1 page)
5. Lower bound (2 pages, main theorem + proof sketch)
6. Achievability construction (1.5 pages)
7. Experiments (2 pages)
8. Discussion and limitations (0.5 pages)
9. Appendices: full proofs, additional experiments, reproducibility

### 6.2 Milestones
- Week 11: Full draft v1 (rough, gaps marked as TODO)
- Week 13: Full draft v2 (all sections complete, figures finalized)
- Week 15: Full draft v3 (polished, sent to external readers if possible)
- Week 17: Final revisions based on feedback
- Week 18: Submission-ready version, arxiv preprint posted

### 6.3 External feedback checkpoint
Week 15 is the hard deadline for getting at least two external reads on the full draft. Options:
- Prof. Sanjay Garg (has been your co-author)
- TurboQuant authors (Zandieh, Mirrokni) if they responded to the initial ping
- Any research contacts from LinkedIn, Twitter, or academic networks
- Worst case: pay an AI safety researcher on Upwork or similar for a four-hour review ($200-400)

---

## 7. Phase 5: Revision and submission (Weeks 19 to 22, Aug 24 to Sep 20)

**Purpose:** buffer for ICLR 2027 deadline (typically late September).

### 7.1 Activities
- Address external reader feedback
- Run any final ablations requested
- Polish writing
- Prepare supplementary material
- Submit to ICLR 2027 OpenReview
- Post arxiv preprint simultaneously

### 7.2 Backup plan if not ready
- Submit to TMLR instead (rolling deadline, no rush)
- Submit to AISTATS 2027 (early October)
- Continue iterating for ICML 2027 (late January 2027)

---

## 8. Reading list (prioritized)

### 8.1 Week 1 essential reading
1. Zandieh et al. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (arXiv 2504.19874). Read Theorem 3 and its proof in detail.
2. Cover and Thomas. *Elements of Information Theory*, 2nd ed., Chapter 10 (rate-distortion theory).
3. Ortiz-Jimenez et al. *Task Arithmetic in the Tangent Space* (NeurIPS 2023). Weight disentanglement as distortion measure.

### 8.2 Weeks 2-3 important
4. Ilharco et al. *Editing Models with Task Arithmetic* (ICLR 2023).
5. Yadav et al. *TIES-Merging* (NeurIPS 2023).
6. Stoica et al. *KnOTS: Model Merging with SVD to Tie the Knots* (ICLR 2025).
7. Chee et al. *QuIP: 2-bit Quantization with Guarantees* (NeurIPS 2023).
8. Tseng et al. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* (ICML 2024).
9. Jin et al. *On Task Vectors and Gradients* (arXiv 2508.16082).

### 8.3 Weeks 4-6 construction-relevant
10. Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs* (NeurIPS 2024).
11. Liu et al. *SpinQuant: LLM Quantization with Learned Rotations* (ICLR 2025).
12. Zandieh et al. *QJL: 1-Bit Quantized JL Transform* (2024).
13. RoLoRA: Xi et al. *Fine-tuning Rotated Outlier-free LLMs* (2024).

### 8.4 Phase 3 baselines
14. Yu et al. *DARE* (ICML 2024).
15. Panariello et al. *Core Space* (2025).
16. Kim et al. *Task Vector Quantization* (ICCV 2025).
17. *1bit-Merging* (2025).
18. *TSPA: Leveraging Rotation Symmetry for LoRA Merging* (OpenReview 2025).
19. *DO-Merging* (NeurIPS 2025).
20. *ARM / Streaming Merging* (arXiv 2602.03237).

### 8.5 Negative result to confront
21. *A Systematic Study of Model Merging Techniques in LLMs* (arXiv 2511.21437, November 2025). This paper argues subspace methods fail at LLM scale; your rate-distortion framing must address why.

### 8.6 Classical background (read as needed)
22. Shannon. *Coding Theorems for a Discrete Source with a Fidelity Criterion* (1959).
23. Berger. *Rate Distortion Theory* (1971), Chapter 2.
24. Conway and Sloane. *Sphere Packings, Lattices and Groups*, relevant chapters.

---

## 9. Infrastructure checklist (do once, early)

### 9.1 Repository setup
```
rdmerge/
  paper/
    main.tex
    sections/
    figures/
    references.bib
  theory/
    toy_theorem_v0.tex
    theorem_v1.tex
    theorem_v2.tex
  code/
    synthetic/
    lora_merging/
    evaluation/
  experiments/
    configs/
    results/
    logs/
  notes/
    daily_log.md
    feedback_received.md
    open_questions.md
```

### 9.2 Tooling
- LaTeX: overleaf or local TeXLive
- Python environment: uv or conda, Python 3.11+
- Core libs: torch, transformers, peft, datasets, numpy, scipy, matplotlib
- Merging tooling: mergekit (for baselines)
- Experiment tracking: wandb or just CSV logs (keep simple)
- Version control: git, push to GitHub under K144U (public repo at submission time)

### 9.3 Compute
- Daily driver: RTX 3090/4090 or Colab Pro
- Burst compute: rent one A100 for ~48 hours during Phase 3 (~$50) if needed for Qwen-3B experiments
- Backup: your existing Hostinger VPS for non-GPU tasks (data preprocessing, eval scripts)

---

## 10. External feedback strategy

Solo theory work without an advisor fails primarily from undetected errors. Build redundant feedback loops:

1. **Week 1:** email TurboQuant authors with two-page note.
2. **Week 3:** post theorem_v1 on Alignment Forum or r/ML for informal review.
3. **Week 6:** share achievability proof with anyone who responded in Weeks 1 or 3.
4. **Week 10:** share experimental results informally.
5. **Week 15:** hard deadline for two external reads of the full draft.

Keep a `notes/feedback_received.md` log with every piece of feedback, who it came from, and what action you took. This becomes useful when writing the acknowledgements section.

---

## 11. Risk management

### 11.1 Top risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Lower bound does not close | Medium | Fatal | Phase 0 gate; pivot in week 1 not month 3 |
| Bound closes but is vacuous / huge constant | Medium | High | Fall back to "tightest known bound" framing, still a contribution |
| Someone at Google scoops with same idea | Low-Medium | Fatal | Move fast, post on arxiv as soon as theorem is clean |
| Experiments show bound is way off reality | Medium | High | Narrow the setting (Gaussian task vectors, quadratic loss) where bound is provably right |
| Qwen-3B merging is too slow on consumer GPU | Medium | Medium | Descope to Qwen-1.5B or rent A100 for 48 hours |
| Solo writing quality not top-tier | High | Medium | External readers in Week 15, budget for paid review if needed |
| Time slips, miss ICLR deadline | Medium | Low (fallback venues exist) | TMLR accepts rolling, AISTATS and ICML 2027 as backups |

### 11.2 Stop-loss rule
If by end of Week 3 the theorem is still not closed and there is no clear path, STOP and pivot. Do not grind for another month hoping it works. Acceptable pivots:
- Purely empirical paper on "quantization-aware LoRA merging" (smaller contribution, workshop-tier)
- Pivot to characterizing *when* rotation helps merging empirically (different paper, still publishable)
- Drop this line of work, return to a tighter follow-up of the EMSE paper

---

## 12. Venue calendar

| Venue | Deadline | Notes |
|-------|----------|-------|
| NeurIPS 2026 | ~May 15, 2026 | Too tight, skip |
| ICLR 2027 | ~Sep 24, 2026 | Primary target |
| AISTATS 2027 | ~Oct 8, 2026 | Good backup |
| ICML 2027 | ~Jan 30, 2027 | If ICLR misses |
| TMLR | Rolling | Always available |
| COLT 2027 | ~Feb 2027 | If theory ends up being the dominant contribution |

Check exact deadlines two months before each.

---

## 13. Daily operating rhythm

- Morning (2 hours): deep work on current phase deliverable (theorem proof, code, writing)
- Afternoon (1 hour): reading, literature tracking, note-taking
- Evening (30 min): update `daily_log.md` with progress, blockers, open questions

Weekly: Sunday evening, update this plan.md with actual progress, slipped deadlines, new risks surfaced.

Monthly: review open_questions.md and decide which to tackle, which to drop, which to defer to follow-up paper.

---

## 14. Follow-up paper ideas (post-submission)

Not for this project, but worth noting to keep the research agenda coherent:
1. Empirical study of rate-distortion curves for cross-modality LoRA merging (MedMNIST + your existing Anthropic AI for Science project).
2. Connecting the bound to federated averaging: FedAvg as rate-distortion optimal under bandwidth constraints.
3. Code-specific merging under the bound (pairs with your EMSE work).
4. Extending to weight-quantization-aware merging (combining with QuIP#, QuaRot, SpinQuant in a single theoretical framework).

Keep these in `notes/future_work.md` as you go, they will form your PhD research statement if you apply.

---

## 15. Immediate next actions (today, April 20, 2026)

1. Create the repository structure in section 9.1 (30 min)
2. Download TurboQuant paper PDF and start reading Theorem 3 (2 hours)
3. Create `notes/daily_log.md` and make first entry (5 min)
4. Draft email to Zandieh and Mirrokni for end-of-week send (20 min, save as draft)
5. Block calendar for Day 1-7 deep work sessions (15 min)

Do not start coding anything today. Do not start writing the paper today. The only thing that matters this week is whether the theorem closes.

---

*Last updated: April 20, 2026. This is a living document. Update weekly.*

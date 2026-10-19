# Before submitting to Neurocomputing: the plan

Source: the professor's review of the short paper (2026-10-06), Tier 1 to 3.
Facts checked while writing this list:

- Solves run in **float32** (`code/phase3/merging/regmean.py:104-129`,
  `rd_encoder.py`), not bf16/fp16. At the shipped alpha=0.9 the adapter Gram's
  condition number is ~8e3, where float32 is accurate, so precision is unlikely
  to explain Mistral's 0.000. At lambda=1e-6 it is ~2e6, where float32 loses
  digits: that case needs the float64 check below.
- The theory's dial (`drift_theory.json`) stores **exact** values only; the
  O(s^2) error test needs predicted values recomputed per s, which needs the
  drift cohort's saved A0 (on the cluster, read-only copy, ~0.4 GB).
- The local GPU is an 8 GB RTX 3060 Ti in use by another project, so every
  model experiment below runs on the cluster.

## MUST before submitting

### A. New experiments (cluster GPU)
- [ ] **A1. Library defaults, measured** (review item 4). Unsloth and
  LLaMA-Factory at defaults, two runs on two tasks each; report the principal
  cosine at creation and after a short run. A small base is enough: the claim
  is about the seed, not the model. About 1 GPU-hour.
- [ ] **A2. 2x2 seed ablation** (item 3). A-init seed shared/independent x
  training seed (data order, dropout) shared/independent, or inject A0
  directly. One base; geometry, adapter-only RegMean and the five heuristics
  in NLL. Replaces "nothing else differs". ~8 adapters + ~30 NLL cells.
- [ ] **A3. Untruncated heuristics** (item 3). Five heuristics, four bases,
  both arms, merged into full weights. Tests whether rank truncation explains
  the 29/38 lean. Largest item; start at one cohort per arm in NLL, then
  accuracy.
- [ ] **A4. One published adapter-built method** (item 1). Core Space or LoRM
  on both arms, four bases, NLL then accuracy. **Register the prediction before
  running it.** Its outcome sets the paper's framing either way.

### B. Checks on existing data (CPU, no GPU)
- [ ] **B1. Solve precision.** Recompute the minimal-ridge adapter-only
  RegMean and encoder merges in float64 and compare with float32; state the
  solve dtype in Methods.
- [ ] **B2. Theory error vs s.** Predicted vs exact kappa and amplification at
  each s; show the kappa error falls as O(s^2). Needs the A0 copy above.
- [ ] **B3. Classify the 17 collapsed public cohorts** from their configs and
  READMEs: shared seed, same-run checkpoints, derived/continued adapters,
  frozen A. Note that "same publisher + base" is not "tasks one would merge".
- [ ] **B4. Statistics.** Replace the 29/38 sign test with one that respects
  shared bases/cohorts/eval sets (e.g. per-base means or a cohort permutation
  test); report that TIES on Llama exceeds the HumanEval margin.
- [ ] **B5. Verify every reference**, [20], [21], [25], [29] first: titles,
  venues, arXiv ids.

### C. Text (no compute)
- [ ] **C1. Headline.** "The adapter-only RegMean variant and our encoder"
  everywhere: abstract, highlights, intro, cover letter. Say alpha=0.9 is
  applied out of its intended (data-Gram) use. "The operator an adapter-built
  solver inverts", not "a merge must invert". State that both failing methods
  are ours.
- [ ] **C2. Theory wording.** "Explains", not "predicts". Delete the
  "eps^-2 gives 59, a coincidence" paragraph. Say the amplification match is
  algebraic under the consensus W0. Fix "five predictions" vs four bullets.
  Make "within 0.11%" agree with the printed ratio range.
- [ ] **C3. Pre-registration wording.** "Every confirmatory test"; disclose
  that the collapse was found post hoc, the unplanned check, the amendment.
- [ ] **C4. Abstract precision.** The ridge recovers most of the loss on three
  of four bases, not on Mistral.
- [ ] **C5. Define "our encoder"** in the paper (or drop it).
- [ ] **C6. Ill-posedness.** In the collapsed regime exact interpolation is
  ill-posed, not only ill-conditioned, which is why averaging is the right
  method there.
- [ ] **C7. Related work.** Linear mode connectivity, Git Re-Basin, model
  soups; LoRI, ZipLoRA. Position as the LoRA A-factor case.
- [ ] **C8. Why run the audit** if "add a ridge and sweep" works: a sweep
  needs data and one merge per lambda, default users hit the zeros, and the
  audit decides heuristic vs solver before any of that.
- [ ] **C9. Public-cohort merges** show prevalence, not a consequence; say so.
- [ ] **C10. Missing details**, from the configs: LR, steps, batch size,
  lora_alpha, dropout, target modules, training precision, the 8 sampled
  layers, TIES k, DARE p, merge coefficients and what they were tuned on,
  decoding settings, prompts, how the 500 GSM8K problems were chosen, the
  public-cohort prompt set, the independent seeds, the solve dtype.

### D. Data and admin
- [ ] **D1. Deposit the adapters** (Zenodo or Hugging Face) and replace
  "available on request".
- [ ] **D2. External timestamps**: Software Heritage save of the public repo
  and a Zenodo snapshot of the registration documents. These prove existence
  from now on; they cannot backdate the past, so C3's wording matters more.
- [ ] **D3. Public repo push** (pending the user's OK), bios and photos.
- [ ] **D4. Polish**: "Analyzers" vs British spelling; the GitHub URL is
  identical everywhere.

## SHOULD (reviewers will comment)
- [ ] **S1. Translation-fixed cohorts on all four bases** (now two).
- [ ] **S2. Accuracy on instruction following and translation**, or narrow
  "averaging is unaffected" to math and code.
- [ ] **S3. Reference rows**: base model and individual adapters in Table 2 or
  Figure 5 (multitask optional).
- [ ] **S4. Training-regime generality**: collapse at two learning rates and
  two step budgets (geometry only; cheap).

## The three the professor would do first
A1; A2 + A3; C1 + A4.

# Master Plan: Rate-Distortion Limits of Model Merging, ICLR 2027

Prepared June 11, 2026. Assumed deadline: late September 2026 (ICLR pattern: Sep 24-Oct 1; verify when CFP opens and shift everything accordingly). Working window: roughly 14 weeks. Full freeze target: 2 weeks before deadline.

Goal state at submission: a paper where (1) the theory has been tested in both regimes, not just calibrated at floor zero, (2) the paper's own optimal encoder has been run on real adapters, (3) every result has seed-level error bars and at least one downstream metric, (4) every open question the paper raises about itself has either been answered or moved to honestly labeled future work, and (5) the prose passes a 60-second cold read.

---

## Part I: Workstreams overview

Four parallel workstreams. W = writing, E = experiments, T = theory, I = infrastructure.

- **W:** the 8-item rewrite (already specified in iclr_revision_plan.md), then continuous integration of new results, then the 9-page cut.
- **E:** eleven experiments in three tiers, dependencies mapped below.
- **T:** the T-dependence gap (Remark 5) and floor-estimation methodology; one focused effort with Prof. Garg.
- **I:** training/eval pipeline hardening. Past PBS/Torque jobs on the JIIT cluster all failed; nothing in Tier 2 starts until the pipeline survives a kill-and-resume test.

Dependency spine:

```
I0 (pipeline hardening)
  -> E1 (encoder on real adapters)      [needs existing adapters only]
  -> E2 (multi-seed)                    [needs I0]
  -> E5 (floor-positive)                [needs I0 + E5 design week]
  -> E6 (T sweep)                       [needs I0; shares task pool with E5]
E4 (synthetic T sweep)                  [no dependencies, CPU, do immediately]
E3 (downstream metrics)                 [needs eval harness extension, no training]
E7 (b=2 mechanism)                      [needs existing adapters only]
T1 (T-dependence theory)                [triggered by E4 outcome]
W  (rewrite)                            [no dependencies, starts day 1]
```

---

## Part II: Experiments, full specifications

### E1. Optimal encoder on real adapters (Tier 1, highest priority)

**Question:** does the paper's own near-optimal encoder (Section 5.1) close the 0.10 to 0.22 nats/token gap on real adapters, or is the gap an artifact of the quadratic-loss approximation?

**Design.**
- Inputs: the existing 16 adapters (4 models x 4 tasks), frozen.
- Per layer: stack the 4 task vectors; choose H_t. Run two H_t variants and report both: (a) projector surrogate H_t = P_{V_t} (cleanest, matches the deff measurements), (b) diagonal empirical Fisher at tau_t estimated on 512 training examples per task (closer to the theory's intended metric). The (a)/(b) comparison is itself informative about the MSE-to-CE bridge.
- Compute H_bar, tau_bar_H, eta = H_bar^{1/2} tau_bar_H per layer. Apply Gaussian-QR rotation (shared seed) + uniform scalar quantization at b in {1, 2, 3, 4, 8, 16} bits per coordinate of eta. Decode to w*, patch into the base, evaluate worst-task excess NLL on the same n = 1000 held-out splits.
- Also evaluate the b = infinity point (no quantization, w* = tau_bar_H exactly): this isolates "centroid reconstruction error" from "quantization error" and is the single most diagnostic number in the experiment. Under the theory, in the floor-zero regime, tau_bar_H should achieve near-zero worst-task distortion in the quadratic surrogate; whatever excess NLL remains at b = infinity is the quadratic-approximation residual.

**Decision rules (write these into the paper whichever branch occurs):**
- If b = infinity excess is near zero (< 0.02 nats) and finite-b tracks the predicted decay: the gap was algorithmic slack and the optimal encoder closes it. Headline result; the paper's central claim is confirmed end to end.
- If b = infinity excess is substantial (comparable to TA's 0.10 to 0.22): the gap is dominated by the quadratic-loss approximation, not encoder suboptimality. Report honestly; this redirects Section 7 and actually motivates the explicit-Mt theorem as the key open problem. Still a strong, clean finding.
- Intermediate: decompose the gap into approximation component (b = infinity) and algorithmic component (TA minus b = infinity), report both. This decomposition is publishable on its own.

**Cost:** no training. Implementation ~3 to 4 days; evaluation 6 b-values x 4 models x 4 tasks x 1000 examples, ~40 to 60 GPU-hours. **Duration: 1.5 weeks.**

### E2. Multi-seed training (Tier 1)

- Retrain all 16 adapters with seeds {1, 2} added to the existing seed {0}: 32 trainings.
- Re-run the full merge matrix (5 methods + TVQ sweep) per seed. Report mean +/- seed-level range alongside per-seed bootstrap CIs. Re-verify deff = Tr per seed (expected, but now it is a measured claim across 3 seeds x 4 models).
- Check specifically whether the cross-model ordering (Llama > Mistral > Qwen > Yi) and the b=2 dip survive across seeds; these are the two findings whose seed-robustness the paper currently argues only by margin.
- **Cost:** 32 trainings x 2-4 GPU-h + merge-matrix evals, ~180 to 250 GPU-hours. **Duration: 2 weeks wall-clock, fully parallelizable.**

### E3. Downstream metrics (Tier 1)

- Add to the eval harness: GSM8K exact-match accuracy (8-shot or 0-shot CoT, pick one and freeze), HumanEval + MBPP pass@1 for code, COMET-22 for en-de translation, IFEval strict for instruction following.
- Evaluate: per-task adapters (baselines), TA, TIES, DARE, KnOTS, TVQ at b in {1, 2, 4}, and E1's encoder at its best b. Worst-task metric = max per-task accuracy drop relative to that task's own adapter.
- **Claim to test:** the NLL-based conclusions (only TIES separates; KnOTS = TA; b=2 dip) survive on task metrics. Report rank correlation between excess NLL and accuracy drop across the merge matrix.
- **Risk to pre-empt:** generation-based metrics are noisier than NLL; n = 1000 per task may give wide CIs on accuracy. Use paired comparisons (same prompts across methods) and report bootstrap over prompts.
- **Cost:** generation-heavy, ~80 to 120 GPU-hours. **Duration: 1.5 weeks, parallel with E2.**

### E4. Synthetic T sweep at T in {4, 8} for the achievability constant (Tier 1, do first, CPU)

- The paper says this measurement "would decide" whether the linear-T factor in C = Tc^2/3 is real (Remark 5). Run the existing Monte Carlo at T in {2, 3, 4, 8, 16}, r in {4, 8}, d sized so Tr <= d, 1000 trials, measure the achievability ratio vs the lower bound.
- **Decision rule:** ratio flat in T -> the linear-T factor is an analysis artifact; this triggers theory workstream T1 (prove a T-free or log-T bound). Ratio grows linearly -> the gap is real; soften Remark 5 accordingly and state it as a sharp open problem.
- **Cost:** CPU only, a day of compute. **Duration: 2 to 3 days including plots.**

### E5. Inducing the floor-positive regime on real adapters (Tier 2, the headline experiment)

**Question:** does the floor formula B^2(1 - deff/(Tr)) predict measured irreducible worst-task loss when task subspaces actually overlap? This is the experiment that upgrades the paper from "bound calibrated at zero" to "validated theory."

**Design week first (do not skip):** one week of pure design + pilot before any large training. The hard problems are (i) inducing controllable overlap, (ii) measuring "irreducible" loss operationally, (iii) mapping the floor from surrogate units to NLL.

**Overlap induction, three arms (run pilot on one model, Qwen-2.5-7B, cheapest of the strong ones):**
- Arm 1, semantic proximity: adapter sets of graded relatedness. Far: {GSM8K, Alpaca, Magicoder, WMT}. Mid: {GSM8K, MATH, Magicoder, Alpaca}. Near: {GSM8K, MATH, AQuA-RAT, SVAMP} (all math). Hypothesis: deff/(Tr) decreases far -> near.
- Arm 2, data-mixture control: T = 4 adapters trained on mixtures alpha * D_shared + (1 - alpha) * D_unique for alpha in {0, 0.25, 0.5, 0.75, 0.9}. This gives a continuous overlap knob; expect deff/(Tr) monotone decreasing in alpha. This is the cleanest arm for a prediction curve.
- Arm 3, geometric forcing: rank r in {64, 128} on narrow projections so Tr approaches the layer dimension, forcing deff < Tr by dimension counting. Mechanically guaranteed to produce deff < Tr; useful as a positive control even though it is the least "natural."

**Pilot gate (end of design week + 1 pilot week):** Arm 2 at alpha in {0, 0.5, 0.9}, one model, measure deff per layer. Go if deff/(Tr) at alpha = 0.9 drops below 0.8 in a majority of layers. If even alpha = 0.9 keeps deff = Tr (possible: shared data need not imply shared subspaces), fall back to Arm 3 as the primary arm and report Arm 2's null as a finding in itself (real fine-tuning resists subspace overlap, which strengthens the practical message that the algorithmic regime is the universal one).

**Measurement of the irreducible component:**
- For each overlap level: merge with all methods plus E1's encoder at b = infinity and large b. The floor prediction concerns what no encoder can achieve, so the operational proxy for "irreducible loss" is the minimum over all available methods (including the near-optimal encoder at effectively infinite rate) of worst-task excess.
- Primary comparison in surrogate units: compute the predicted floor B^2(1 - deff/(Tr)) per layer with B^2 estimated as the mean tau_t^T H_t tau_t over tasks (state this estimator explicitly), and compare against the measured quadratic distortion (w - tau_t)^T H_t (w - tau_t) of the best merge. This is apples-to-apples and avoids the MSE-to-CE mapping problem.
- Secondary, in NLL units: report measured worst-task excess NLL vs deff/(Tr) as a monotonicity claim only (higher overlap -> higher irreducible excess), not a quantitative fit. Do not overclaim a unit mapping the theory does not yet provide; reviewers will respect the discipline.

**Deliverable:** one figure, predicted floor vs measured minimum distortion across overlap levels, with the zero-overlap points from the original 16 adapters anchoring the left edge. If the prediction tracks, this is the paper's new centerpiece. New subsection 6.3.

**Cost:** pilot ~10 trainings (~30 GPU-h) + main run: Arm 2 full (5 alphas x 4 adapters x 3 seeds = 60 trainings) + Arm 1 (3 sets x 4 adapters, 1 to 2 seeds = 12 to 24) + Arm 3 (8 trainings) + merge matrices: ~350 to 500 GPU-hours. **Duration: 1 week design, 1 week pilot, 3 weeks main run.**

### E6. T sweep on real adapters (Tier 2)

- Task pool of 12: GSM8K, MATH, AQuA, Magicoder, CodeAlpaca, Alpaca, Dolly, WMT en-de, WMT en-fr, SQuAD-style QA, summarization (XSum or CNN/DM), NLI (MNLI). Two base models (Llama-3.1-8B hardest, Yi-1.5-9B easiest), rank 16, single seed (E2 covers seed variance at T = 4).
- Merge nested subsets T in {2, 4, 8, 12} (fixed nesting order, plus 3 random subset draws at T = 4, 8 to control for subset choice). Measure: deff/(Tr) vs T (does independence persist as Tr grows toward d?), worst-task excess vs T per method, and the achievability ratio of E1's encoder vs T (the real-data complement to E4).
- **Cost:** 2 models x 12 tasks = 24 trainings (~80 GPU-h) + a large merge-eval matrix (~150 to 250 GPU-h). **Duration: 2.5 weeks, can overlap E5 main run.**

### E7. b = 2 mechanism tests (Tier 2, cheap, high payoff per hour)

The paper states two falsifiable predictions for the "implicit TIES" hypothesis. Test both on the existing adapters:
- Prediction 1: the dip tracks sparsity. Measure the fraction of coordinates zeroed by 2-bit quantization per model; correlate dip depth (b=4 excess / b=2 excess) with it across the 4 models and, after E2, across 12 model-seed pairs. Add a direct manipulation: explicit magnitude pruning at the same sparsity level as b=2 TVQ, no quantization. If pruning alone reproduces the dip, the mechanism is confirmed.
- Prediction 2: per-task win pattern of b=2 TVQ correlates with TIES's per-task win pattern (paired per-task differences, rank correlation).
- **Decision rule:** both confirmed -> the mechanism moves from Section 7 hypothesis into Section 6 as a measured result, and Finding 4 becomes a real contribution. Either fails -> hypothesis stays in 7.1, now with evidence against it, which is also honest and fine.
- **Cost:** ~30 to 50 GPU-hours. **Duration: 1 week, parallel anytime after E1's code exists.**

### E8 to E11 (Tier 3, only if Tiers 1 and 2 are done by Week 10)

- **E8 rank sweep:** r in {4, 8, 32, 64} on one model, original 4 tasks; when does deff < Tr emerge naturally? (~60 GPU-h)
- **E9 scale probe:** repeat the core merge matrix on Llama-3.2-3B and (QLoRA) Llama-3.1-70B; does the algorithmic gap shrink or grow with scale? (~150 to 300 GPU-h; 70B eval is the expensive part)
- **E10 added baselines:** Fisher-weighted averaging, DELLA, and one 2026 method, on the original matrix + best E5 setting, so the method set cannot be called thin. (~60 GPU-h)
- **E11 quadratic-bridge check:** for small perturbations around each tau_t, compare measured Delta L_t against the Fisher-quadratic prediction; empirical support for Appendix B. (~30 GPU-h)

---

## Part III: Theory workstream

**T1. The T-dependence of the achievability constant.** Triggered by E4. If the empirical ratio is flat in T, spend one focused week (you + Prof. Garg) on a high-probability version of Theorem 4's Step 3: for Stiefel-random V_t and isotropic quantization error, the T per-task projections concentrate, so max_t <= (1 + o(1)) avg + deviation term, plausibly giving C = O(c^2 (1 + sqrt(log T / r))) or similar. Even a partial result (e.g., expected-max bound via sub-Gaussian projections) converts the paper's most visible loose end into a closed one. Timebox: 1 week + 1 week writeup. If it does not fall, state the sharpened conjecture with the E4/E6 evidence and move on.

**T2. Floor estimation methodology (needed by E5).** Write a half-page formal recipe: how B^2, H_t, and deff are estimated from trained adapters, with the estimator's assumptions stated. This becomes part of Section 6.3 and makes the diagnostic actually usable by practitioners, which is the paper's practical pitch.

**T3 (stretch, only if time):** the explicit-M_t cross-entropy theorem sketched in Appendix B. Realistically too big for this cycle; keep as stated future work unless E1's b = infinity result forces it forward.

---

## Part IV: Infrastructure (I0), week 1, blocking

Past PBS/Torque jobs on the JIIT cluster all failed. Tier 2 does not start until all of the following pass:

1. Pinned environment: containerized (Docker/Apptainer) or locked conda env with exact versions; one env for training, one for eval; hashes recorded.
2. Checkpoint/resume: every training job checkpoints every N steps and survives a manual kill-and-resume test. Every eval job writes per-example results incrementally and is idempotent.
3. Experiment registry: one CSV/sqlite of every run (config hash, seed, status, output path). No orphan outputs.
4. Decide the compute substrate now: if AWS (AEGIS credits), get Dr. Saini's explicit OK for non-AEGIS use first; otherwise budget spot-instance A100/H100 hours independently and use the JIIT cluster only for CPU-side work (E4, deff computations, analysis).
5. Eval determinism: fixed seeds, fixed prompt formats, greedy decoding for generation metrics, version-locked metric implementations (COMET, IFEval).

---

## Part V: Writing workstream

- **Week 1:** apply rewrite items 1 to 6 from iclr_revision_plan.md (abstract, intro, contributions, measured/interpretive split, worked-example move, reproducibility fix). Send abstract + intro to Prof. Garg with the 60-second cold-read test.
- **Continuous:** every experiment lands as a self-contained subsection draft within 3 days of its analysis finishing, written claim-evidence-claim-evidence. Maintain a single results.md with every number and its provenance (run IDs from the registry).
- **Week 10 to 11:** restructure for the new content. Expected final main-text shape: Sections 1 to 5 as in the revision plan; Section 6 becomes 6.1 synthetic (incl. E4), 6.2 floor-zero real-LLM calibration (original matrix + E1 + E2 + E3 + E7), 6.3 floor-positive validation (E5) and T scaling (E6); Section 7 interpretation/limitations/implications. The 9-page cut will be brutal; everything in Part II above produces appendix material by default and earns main-text space only by being load-bearing for a contribution.
- **Weeks 12 to 13:** freeze, two full human read-throughs (you, Garg, ideally one outsider), ICLR style compliance, anonymization audit (Zenodo DOI not linked, self-citation in third person, no acknowledgments), LLM-disclosure compliance per the CFP, reproducibility statement final.

---

## Part VI: Timeline (weeks of)

| Week | Dates (2026) | E | W / T / I |
|---|---|---|---|
| 1 | Jun 15 | E4 (CPU); E1 implementation | I0 hardening; W rewrite items 1-6; Garg sync on this whole plan |
| 2 | Jun 22 | E1 eval runs; E7 starts | I0 sign-off gate; Garg cold-read feedback |
| 3 | Jun 29 | E1 analysis + writeup; E2 launches; E5 design week | T1 starts if E4 says flat |
| 4 | Jul 6 | E2 training; E3 harness build; E5 pilot launches | T1 continues |
| 5 | Jul 13 | E2 merge matrices; E3 evals; **E5 pilot gate decision** | T1 writeup or conjecture statement |
| 6 | Jul 20 | E5 main run launches; E3 analysis | W: integrate E1/E4/E7 sections |
| 7 | Jul 27 | E5 training continues; E6 training launches | W: integrate E2/E3 |
| 8 | Aug 3 | E5 merge matrices; E6 continues | T2 floor-estimation writeup |
| 9 | Aug 10 | E5 analysis + the centerpiece figure; E6 evals | W: draft 6.3 |
| 10 | Aug 17 | E6 analysis; **Tier 3 go/no-go**; start E10 if go | W: full restructure begins |
| 11 | Aug 24 | Tier 3 mop-up or buffer | W: 9-page cut complete; full draft to Garg |
| 12 | Aug 31 | Buffer for reruns | Read-through 1; fix pass |
| 13 | Sep 7 | Nothing new launches | Read-through 2; anonymization + compliance audit |
| 14 | Sep 14 | **FREEZE** | Final polish only; submit when portal opens |

Buffer policy: weeks 12 to 13 absorb slippage. If E5's pilot gate fails on Arm 2, the fallback (Arm 3 primary) costs ~1 week, which the buffer covers. If two or more major slips occur, cut E6 before cutting anything in Tier 1 or E5.

---

## Part VII: Compute budget summary

| Item | GPU-hours (est.) |
|---|---|
| E1 encoder on real adapters | 50 |
| E2 multi-seed | 180 to 250 |
| E3 downstream metrics | 80 to 120 |
| E4 synthetic T sweep | CPU |
| E5 floor-positive (pilot + main) | 380 to 530 |
| E6 real T sweep | 230 to 330 |
| E7 b=2 mechanism | 30 to 50 |
| Tier 3 (all of E8-E11) | 300 to 450 |
| Reruns/failures (25% overhead) | ~300 |
| **Total, Tiers 1+2** | **~1,000 to 1,400** |
| **Total with Tier 3** | **~1,400 to 1,900** |

On A100-80GB spot pricing this is very manageable; wall-clock parallelism, not money, sets the schedule.

---

## Part VIII: Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ICLR 2027 deadline earlier than predicted | Low-med | Everything Tier 1 + E5 done by Week 9; only Tier 3 and polish are after |
| E5 Arm 2 fails to induce overlap | Medium | Pilot gate Week 5; Arm 3 fallback is mechanically guaranteed; a null on Arm 2 is itself reportable |
| E1 shows the gap is approximation, not algorithmic | Medium | Pre-written decision branch; reframes, does not sink, the paper |
| Cluster/infra failures recur | Medium-high given history | I0 is a hard gate; cloud fallback decided Week 1 |
| Your time (4 concurrent projects + internship) | High | This paper takes priority slot 1 through Week 9; KVCompressor Week-0 and sign-conflicts venue decision are the only other active threads; AEGIS items that can slip to October, slip |
| Reviewer says "incremental over TurboQuant" | Low-med | Intro paragraph 2 + Lemma 1 make the multi-source/max-distortion gap explicit; E5 gives a result TurboQuant cannot state |
| Generation metrics too noisy (E3) | Medium | Paired prompts, bootstrap over prompts; if still noisy, report NLL primary + accuracy as supporting, stated as such |
| Scooped on the RD-merging framing over the summer | Low | Zenodo DOI timestamps priority; monitor arXiv weekly for "rate-distortion merging" terms |

---

## Part IX: Decision log to maintain

Keep a dated log (decisions.md) recording: pilot gate outcomes, decision-rule branches taken (E1, E4, E7), anything cut and why. This feeds the reproducibility statement honestly and protects you if a reviewer asks why a design choice was made.

---

## Part X: Immediate next actions (this week)

1. Send this plan + the rewritten abstract/intro to Prof. Garg; get his agreement on E5 as the headline addition and book the T1 theory week with him.
2. Confirm compute substrate (Dr. Saini re AWS credits, or budget cloud hours) and start I0.
3. Run E4 today or tomorrow; it is CPU-only and its outcome schedules T1.
4. Start implementing E1 (the encoder is ~200 lines on top of your existing merge code).
5. Apply writing items 1 to 6 to the LaTeX.
6. Decide and freeze the held-out splits, prompts, and metric versions for E3 before any evals run, so nothing is chosen after seeing results.

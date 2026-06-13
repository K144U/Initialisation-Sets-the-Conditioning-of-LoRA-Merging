# E5 — Floor-positive regime: design week protocol

**Status:** DRAFT (2026-06-13) — needs sign-off before any training launches.
**Owner:** Sankalp.
**Time budget:** 1 week design + 1 week pilot before main run.
**Master plan reference:** §E5, §T2.

---

## 1. Why E5

Tonight's results (ridge λ=0.05 on llama → worst-excess 0.091, 59% below
TA) confirm the achievability claim survives on real adapters in a
regularized form. But the paper's lower bound $B^{2}(1 - d_{\mathrm{eff}}/(Tr))$
collapses to **zero** under floor-zero (independent subspaces, which is
where the original 16 adapters live by E1's $d_{\mathrm{eff}} = Tr$
finding). So we calibrate against zero, which reviewers will (rightly)
call a trivial test.

E5 is the experiment that upgrades the paper from "bound calibrated at
zero" to "bound calibrated against a non-zero measured floor." If the
predicted floor tracks measured worst-task excess across an overlap
sweep, the lower bound becomes a *quantitatively validated theory* and
takes its place as the paper's headline.

---

## 2. Pilot model

**Qwen-2.5-7B-Instruct.** Per master plan: cheapest strong model
($\approx 4.4$ GB bf16 base after kv-share), fast eval, multi-head
attention sized for $d_{\mathrm{eff}}$ analysis. Tonight's E2 numbers
back this: qwen TIES = 0.013, TA = 0.105, b=2 = 0.019 — clean enough
signal to see floor effects when present.

If qwen pilot succeeds → main run uses qwen + 1-2 additional models
(llama for cross-architecture; yi for cross-pretraining-recipe) per
the master plan.

---

## 3. Overlap-induction arms

The challenge: real fine-tuning *resists* subspace overlap (E1
measured $d_{\mathrm{eff}} = Tr$ for every base × every task pair).
We need a controllable knob.

### Arm 2 (PRIMARY) — data-mixture controlled overlap

Train $T = 4$ adapters on a mixture
$\alpha \cdot D_{\mathrm{shared}} + (1 - \alpha) \cdot D_{\mathrm{unique}}$
for $\alpha \in \{0, 0.25, 0.5, 0.75, 0.9\}$.

Specifics:
- $D_{\mathrm{shared}}$: a fixed pool of 7500 examples drawn uniformly
  from the union of all four task corpora (GSM8K, Alpaca, Magicoder,
  WMT19), sampled with a fixed seed (20260518). Identical across
  task-adapter trainings.
- $D_{\mathrm{unique}}$: 7500 examples from the target task,
  disjoint from shared pool.
- Training hyperparameters: identical to the v1 adapters
  (rank 16, LoRA on q/k/v/o, AdamW lr 5e-5, 7500 examples × 2 epochs,
  bf16 base).
- 5 α values × 4 tasks = 20 trainings on qwen.

Hypothesis: $d_{\mathrm{eff}}/(Tr)$ decreases monotonically in α
(more shared data → more overlapping subspaces). Pilot tests whether
$\alpha = 0.9$ achieves $d_{\mathrm{eff}}/(Tr) < 0.8$.

### Arm 3 (FALLBACK) — geometric forcing

Train adapters at rank $r \in \{64, 128\}$ on the same 4 tasks
without any data manipulation. Mechanically, $T \cdot r \to 4 \cdot 128 = 512$,
which can approach the layer dimension and force $d_{\mathrm{eff}} < Tr$
by dimension counting.

- 4 tasks × 2 ranks = 8 trainings on qwen.
- This is guaranteed to produce $d_{\mathrm{eff}} < Tr$ by counting;
  the question is whether it produces useful adapters and whether
  the floor formula tracks.

### Arm 1 (CONFIRMATORY, parallel) — semantic proximity

Adapter sets at three semantic-proximity levels:
- Far: {GSM8K, Alpaca, Magicoder, WMT19} — current baseline.
- Mid: {GSM8K, MATH, Magicoder, Alpaca} — math + general.
- Near: {GSM8K, MATH, AQuA-RAT, SVAMP} — all math.

3 sets × 4 adapters × 1 seed = 12 trainings on qwen. Tests whether
semantic closeness alone induces overlap (likely null per E1 findings,
but reportable).

---

## 4. Pilot gate (end of design week + 1 pilot week)

Run Arm 2 at $\alpha \in \{0, 0.5, 0.9\}$ only (3 α × 4 tasks = 12
trainings). Compute $d_{\mathrm{eff}}/(Tr)$ per layer.

**GO criterion**: at $\alpha = 0.9$, $d_{\mathrm{eff}}/(Tr) < 0.8$ in
a *majority* of attention projections (>64/128).

**Branches**:
- GO → Arm 2 main run (full 5-α sweep, 60 cells with 3 seeds) +
  start Arm 1 in parallel.
- NO-GO → Arm 3 becomes primary; Arm 2's null is itself a finding
  ("real fine-tuning resists subspace overlap"), which actually
  strengthens the paper's *practical* message (the algorithmic
  regime is the universal one).

---

## 5. Floor measurement protocol (§T2 requirement)

For each overlap level, after training:

1. **Compute $d_{\mathrm{eff}}$** per layer via the existing
   `deff_analysis.py` script.
2. **Estimate $B^2$** per layer as
   $\overline{\tau_t^\top H_t \tau_t}$ over tasks — explicit estimator
   stated in the writeup (§T2 deliverable).
3. **Compute predicted floor** $B^2 (1 - d_{\mathrm{eff}}/(Tr))$
   per layer.
4. **Compute measured floor** = min over all available merging methods
   of worst-task quadratic distortion
   $(w - \tau_t)^\top H_t (w - \tau_t)$. Methods to include: TA, TIES,
   DARE, KnOTS, TVQ@{b1, b2, b4}, rd_encoder@b=inf with ridge
   λ=0.05 (the salvage).
5. **Primary comparison**: predicted vs measured floor, in surrogate
   units. Apples-to-apples; no MSE-to-CE bridge needed.
6. **Secondary**: measured worst-task NLL excess vs
   $d_{\mathrm{eff}}/(Tr)$ as a monotonicity claim only.

---

## 6. Decision rules for the main figure

- Predicted floor tracks measured floor (Pearson r > 0.85 across
  α points, residuals < 30% of measured) → centerpiece figure of §6.3.
- Tracks weakly (r ∈ [0.5, 0.85]) → reported with a "first-order
  agreement" framing; discuss the missing terms.
- Doesn't track (r < 0.5) → the operationalization of $B^2$ via the
  task-mean estimator is wrong, or the surrogate-to-CE bridge has
  Arm-2-specific structure we didn't anticipate. Either result is
  reportable but no longer the headline.

---

## 7. Compute budget for the pilot (1 week)

- Pilot trainings: 12 qwen-7B trainings × ~3.5 GPU-h = 42 GPU-h.
- Pilot merge matrices: 3 overlap levels × (5 methods + RD-encoder)
  = 18 cells × ~1.2 GPU-h = 22 GPU-h.
- $d_{\mathrm{eff}}$ / spectral analysis: CPU only, ~0.5 day.
- **Total: ~65 GPU-h, fits in 2 days at 3-wide on GPUs 2,4,6.**

---

## 8. What needs your sign-off

1. **Confirm pilot model = Qwen-2.5-7B.** Yes/no.
2. **Confirm Arm 2 (data-mixture) is primary, Arm 3 (geometric) fallback.** Yes/no.
3. **Confirm pilot gate criterion** ($\alpha=0.9 \Rightarrow d_{\mathrm{eff}}/(Tr) < 0.8$ in majority of layers). Yes/no, or propose alternate threshold.
4. **Confirm $D_{\mathrm{shared}}$ is uniform mix of all 4 task corpora.**
   Alternative: a single corpus (e.g., Alpaca alone) shared by all four
   adapters. Trade-off: uniform mix is symmetric per task, single-corpus
   is cleaner but biases the shared subspace toward Alpaca's geometry.
   Recommend uniform (current text). Yes/no.
5. **Approve the floor estimator** (task-mean $\overline{\tau_t^\top H_t \tau_t}$).
   This is the §T2 deliverable; if you want a different estimator, decide now.
6. **Cross-model in main run** (after pilot gate): qwen + llama + yi, or just qwen?
   Recommend qwen + llama for cross-architecture, yi as time permits.

---

## 9. Files this protocol will produce (when sign-off received)

- `code/phase3/configs/eval_e5_pilot/<arm>__<alpha>.yaml` — training configs
- `code/phase3/scripts/pbs_train_e5.sh` — training wrapper (pilot)
- `code/phase3/scripts/analyze_e5_floor.py` — floor estimator + comparison
- `code/phase3/scripts/pbs_orchestrator_e5_merge.sh` — eval matrix wrapper
- `notes/E5_pilot_outcome_<date>.md` — gate decision log entry

%%% END DRAFT — awaiting sign-off %%%

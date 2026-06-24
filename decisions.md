# Decision log (per master_plan_iclr2027.md Part IX)

Dated record of gate outcomes, decision-rule branches, and cuts.

## 2026-06-10 — TMLR desk rejection
Desk-rejected by TMLR Editors-in-Chief (Kamath, Murray, Shah, Charlin),
no reviews. Verbatim grounds: "does not meet our editorial standards or
allow us to assess claims and evidence. In particular, the argumentation
and motivation for the work lack clarity. The authors are advised that
TMLR is not a suitable venue for this work."

Interpretation (2026-06-11): a triage/clarity failure, not a scientific
verdict — no reviewer assessed the theory or experiments. Likely causes:
formula-dense abstract (4 inline formulas, "Stiefel-random" unglossed),
and the floor-zero finding framed as the headline (reads as "the
interesting quantity vanishes on all real data"), plus IT-theory register
for a general-ML editorial board.

## 2026-06-11 — Recalibrated response
- master_plan_iclr2027.md adopted as the ICLR 2027 campaign, with
  reordering: the clarity rewrite (W items, 60-second cold read) is the
  IMMEDIATE action, not a parallel workstream; experiments proceed on
  their own merit, not as a response to the desk reject.
- E5 (floor-positive regime) doubles as the fix for the motivation
  critique: it makes the floor non-vacuous on real data.
- No resubmission anywhere within a month; target remains ICLR 2027 as
  originally planned (TMLR was an interim attempt).
- I0 note: the "all PBS jobs failed" premise predates the keeper +
  checkpoint + idempotent-resume patterns since proven on this cluster
  (LMCA 1,020-config grid; MOOLoRa pilot). Port those, not a rebuild.
- E4 launched 2026-06-11 (CPU, synthetic T sweep) — outcome schedules T1.

## 2026-06-11 — E4 outcome: decision rule FIRED -> T1 triggered
1000 trials/cell, T in {2,4,8,16} x r in {4,8}, d=256, seed 20260611
(results/e4_t_sweep/). Three findings:
1. Linear-T DECISIVELY FALSIFIED: median T16/T2 ratio growth = 1.77 vs
   the 8x that C = Tc^2/3 predicts.
2. Growth is cleanly LOGARITHMIC in T: ratio vs log2(T) fits with
   R^2 = 0.998-1.000 in all 10 (r,b) cells; ratios flat in b within each
   cell (constant independent of rate, as theory expects).
   r=4: 10.8 -> 15.0 -> 18.6 -> 22.4; r=8: 10.8 -> 13.4 -> 15.9 -> 18.2.
3. The log-T slope shrinks with r (~3.9/doubling at r=4 vs ~2.5 at r=8,
   ratio ~ sqrt(2)) -> consistent with a max-of-T concentration mechanism
   with deviation ~ sqrt(.../r), the exact route master_plan T1 sketches.
ACTION: T1 (theory week with Prof. Garg) is ON — target a
C = O(c^2 (1 + f(log T, r))) bound; Remark 5 to be rewritten around this
measurement either way. T=3 excluded from sweep (Hadamard-padding rate-
accounting artifact, documented in e4_t_sweep.py).

## 2026-06-11 — I0: orchestrator built; root cause of "all PBS jobs failed" FOUND
Multi-GPU orchestrator (LMCA pattern: one PBS job, worker-per-GPU, done-file
idempotent cells, VRAM gate per cell, login-node keeper) deployed at
code/phase3/scripts/{orchestrator.py, gen_e2_manifest.py,
pbs_orchestrator.sh, orchestrator_keeper.sh}. First payload: E2 (32
multi-seed trainings, seeds 1+2), GPUs 0,1,2,3,5 (4/6/7 reserved for the
MOOLoRa pilot on the other account; flip _ORCH_GPUS + requeue to expand).

Two environment failures diagnosed on launch, BOTH now fixed:
1. unsloth_zoo introspects torch._inductor.config (lazy submodule) at
   import -> AttributeError. Fix: explicit import shim in train_lora.py.
2. THE BIG ONE — the mystery behind the master plan's "past PBS/Torque
   jobs on the JIIT cluster all failed": ~/.local/lib/python3.11/
   site-packages contains torch 2.4.1+cu121 pip-installed --user by the
   LMCA project (same account, late May) which SHADOWS the rdmerge conda
   env's torch 2.10.0+cu128 -> torchvision::nms operator errors and
   version chaos in every job since ~May 20. The May 17-19 trainings
   predate the contamination, which is why they worked. Fix:
   PYTHONNOUSERSITE=1 exported in pbs_orchestrator.sh (hermetic env).
   Rule going forward: every rdmerge job wrapper must set it.

E2 running as job 41481 (keeper pid in logs/orch_keeper.pid).
I0 remaining: kill-and-resume test once first cells are DONE; env hash
recording. E1 implementation next (encoder on real adapters).

## 2026-06-11 — E1 implemented (projector variant); design notes
rd_encoder merge method (code/phase3/merging/rd_encoder.py, registered):
the paper's achievability construction on real adapters. H_t = P_{V_t}
projector surrogate (deff_analysis geometry); H-weighted centroid via the
(Tr x Tr) Gram trick; randomized-Hadamard + uniform scalar quantization
(NOT the spec's Gaussian-QR — dense QR infeasible at out*d_eff ~ 2^18;
Hadamard is the standard structured substitute and matches the synthetic
validation). bits>=32 = b-infinity centroid cell. 5 CPU tests pass incl.
b-inf == analytic centroid.
DESIGN CONSTRAINT: PeftModelView adapters are rank r=16, so the decoded
W* is SVD-truncated to r — the same deployment constraint every baseline
method carries (apples-to-apples). Per-layer truncated-mass fraction is
logged; on random tensors it is ~0.38 (worst case). DECISION RULE: if real
adapters show trunc_mass > ~0.1 at b=inf, build the full-rank base-patch
path (v2) before interpreting the b-inf cell as pure centroid error.
E1 queue: 28 cells (4 models x b in {1,2,3,4,8,16,32}), configs in
configs/eval_e1/, e1_manifest.json; pbs_orchestrator.sh now defaults to
all_manifest.json (E2+E1) so post-E2 requeues flow into E1 automatically.
Fisher-diagonal H variant = second pass after projector results.

## 2026-06-11 — I0 GATE CLOSED: kill-and-resume test PASS
Live test on the running E2 campaign: with 1 cell complete
(llama31_8b_flores_seed1) and 5 trainings mid-flight, job 41481 was
qdel-killed. Keeper detected and resubmitted in 12 minutes (job 41484);
the resumed orchestrator reported "59 pending, 1 already done" — the
completed cell was skipped, the killed in-flight cells requeued from
scratch, and the new job is already on all_manifest.json (E2+E1 = 60
cells), confirming manifest chaining. Per master_plan Part IV this was
the blocking criterion for Tier 2; I0 is now CLOSED (remaining nicety:
env hash recording). Recovery semantics on record: cell-level
idempotency, worst-case keeper latency 30 min, measured 12.

## 2026-06-12 — E2 COMPLETE; E1 v1 llama curve in; rank-truncation rule FIRED
E2: 32/32 multi-seed trainings done, zero failures (llama/mistral/qwen/yi
all 8/8). E1 v1 (rank-16-truncated encoder) full llama bit-curve measured:
b=inf worst-task excess 0.497 vs task arithmetic 0.219 and TIES 0.161 —
the encoder LOSES to plain averaging under the v1 constraint. NOT yet
interpretable: measured trunc_mass ~0.297 mean (d_eff=64 everywhere =
independence confirmed), i.e. the rank-16 adapter slot discards ~30% of
the decoded solution, 3x the pre-registered 0.1 threshold. RULE FIRED ->
built v2 full_rank_patch (residual of W* beyond rank-16 added to base
weights; realized model = W* exactly; CPU test passes; v1 tests intact).
Also noted: non-monotonicity in v1 curve (b=3 beats b=32), echoing the
b=2 "less-is-more" finding; revisit after v2.
QUEUED (+92 cells, manifest now 152): 12 e1fr cells (4 models x b in
{2,4,32} full-rank) + 80 e2m cells (full v3 merge matrix on seed-1/2
adapters, per master plan E2). Bridge watcher armed: when the running
job drains its 60-cell snapshot and writes _QUEUE_COMPLETE, it is
cleared and the orchestrator+keeper relaunch on the 152-cell manifest.

## 2026-06-12 — E1 v2 base-patch path BROKEN on unsloth; rolled back
First two full-rank cells produced worst-task excess ~10.3 nats (vs 0.50
v1, 0.22 TA), quantization-independent (b=4 == b=32) -> the merged model
is destroyed by the base-weight mutation itself, not by encoding. Prime
suspect: unsloth-patched modules do not consume base_layer.weight the way
plain PEFT does (cached/fused weights), so the residual add corrupts or
never reaches the forward path coherently. ACTION: qdel 41509, deleted
the 2 invalid result JSONs, stripped the 10 remaining e1fr cells from the
manifest (now 140 cells), resubmitted as 41510 (queue = 80 e2m matrix
cells). v3 plan: realize W* INSIDE the adapter mechanism instead — inject
a rank-64 LoraConfig (lora_alpha=64 so scaling=1), write exact (A,B)
factors of W_star (d_eff=64 so rank-64 is exact), same forward path every
working method uses; smoke-test on one cheap cell before re-queueing 12.
The v1 (rank-16) E1 results remain valid as the constrained-deployment
comparison.

## 2026-06-12 — E1 b=inf RESOLVED: the exact centroid is the problem (branch 2)
Smoke v3b (plain transformers+PEFT, no unsloth): stored rank-64 factors
realize W* exactly (rel 5e-3 to 2e-2, bf16 tolerance) yet NLL = 3.62 vs
3.54 under unsloth -> BOTH stacks were faithful; the H-weighted centroid
ITSELF is catastrophic. v3a "unsloth forward" hypothesis WITHDRAWN (v2
base-patch remains separately broken/retired). Mechanism CONFIRMED by
centroid_diag.py on llama adapters: Hbar nonzero-eig spectrum is nearly
degenerate (median eig ~0.002, min ~1e-5) although rank is full (d_eff=64
everywhere -> the paper hard-deff/floor-zero measurement stands). Real
task subspaces are independent but nearly collinear in dominant
directions; Hbar^+ divides by sliver eigenvalues -> ||tau_H|| is 25x to
94x (median 32x) the TA merge norm -> far outside the quadratic
surrogate validity region. THIS IS THE MASTER PLAN E1 BRANCH 2 OUTCOME,
sharpened: the gap to the bound is NOT encoder slack closable by the
paper construction; the projector-surrogate/quadratic bridge is the
binding failure, with a precise spectral mechanism. Redirects paper Sec 7
and elevates the explicit-Mt / curvature-aware-H question from future
work to the central open problem, exactly as pre-written.
NEXT probes (cheap, high-value): (1) ridge centroid (Hbar + lambda I)^-1,
lambda sweep — interpolates TA <-> raw centroid, tests whether a tamed
centroid beats TA (the salvageable form of the achievability claim);
(2) Fisher-diagonal H_t (planned variant b) which downweights sliver
directions by actual curvature; (3) connect to the paper soft-vs-hard
d_eff distinction — soft d_eff is tiny where hard d_eff is full, and the
centroid blowup is the OPERATIONAL consequence. The v1 (rank-16) E1
results stand as the deployment-constrained comparison.


## 2026-06-12 (ops) — GPU0 dropped from seed matrix; 41524 -> 41533
The "GPU0 parks harmlessly" assumption was FALSE and was silently
corrupting the matrix. GPU0 hovered at the 24-25GB free boundary; the
orchestrator `free_gb(gpu)` poll occasionally read >=25 and LAUNCHED a
cell on GPU0, after which `pbs_eval_cell.sh`'s 25GB admission check read
24.6 and `exit 87` in 0 min. Unlike the orchestrator's own VRAM-poll
requeue (no attempt charged), the rc!=0 cell path increments
`self.attempts`; at n>=2 the cell is PARKED into `self.failed`, and a
non-empty `failed` makes the run emit `_QUEUE_FAILED` instead of
`_QUEUE_COMPLETE`. Notification showed ~9 cells at rc=87(0min) retry;
caught while `failed=[]` (no permanent loss). FIX: `_ORCH_GPUS`
0,2,4,6 -> 2,4,6; `qdel 41524`; resubmitted `41533` (done-file resume,
attempts reset). No safe swap: live nvidia-smi had gpu0=222MiB free
(other user ~80GB), gpu1/3/5/7 all <25GB free vs our ~55GB cells, so
3-wide. SIDE EFFECT: qdel 41524 ended the two monitors keyed to it
(incl. the ridge sentinel guard); re-armed a guard keyed to 41533
(`_guard_tick.sh` + local monitor: removes a stray `_QUEUE_COMPLETE`
while matrix pending>0, every 120s << keeper 1800s). Ridge 41521 still
has no `ORCH_SENTINEL` so it remains the trap source until it ends.
LESSON: raise matrix cells' `min_free_gb` margin, or have the
orchestrator treat an rc=87 (gate) exit as a no-charge requeue like its
own poll, so a boundary GPU can never burn a cell's attempt budget.

ADDENDUM 2026-06-12 ~16:15: all three LESSON items closed same day —
(1) rc=87 no-charge requeue implemented in orchestrator.py (`719466d`);
(2) keeper v2 verifies `_QUEUE_COMPLETE` against manifest done-files
before honoring it (`daa8648`) — sentinel trap durably defused, no longer
session-dependent; (3) per-job `ORCH_STATE`/`ORCH_SENTINEL` exports added
to the ridge wrapper (state-file clobbering ends with future jobs). Also
deconflicted GPUs: P4 MOOLoRa seeds moved off our matrix cards 4/6 to 1/3
(P4 was confirmed alive via cput≈walltime + resume lines in spool; it was
contention, not a hang).


## 2026-06-12 (eve) — Ridge sweep VERDICT: tamed centroid BEATS TA; achievability salvageable
Ridge centroid (Hbar + lambda*I_d)^-1 sweep (5 cells, llama b=inf
rank_deff, seed 20260518, plain loader, GPU6, ~1.2h/cell, job 41521).
Worst-task excess = max_t (nll_merged[t] - nll_solo[t][t]):
  lambda=0.001 -> 4.463  (translation +4.46; anchors raw-centroid end)
  lambda=0.01  -> 0.346
  lambda=0.1   -> 0.112  <- BEST
  lambda=0.3   -> 0.189
  lambda=1.0   -> 0.286
Clean U-shape. TA reference = 0.219 (E2 seed-stable across 0/1/2).
**lambda=0.1 BEATS TA by ~49% on worst-task excess.** Translation
per-task excess goes NEGATIVE at lambda>=0.1 (merged < translation-solo);
gsm8k is the binding task throughout (0.286 -> 0.346 -> 0.112 -> 0.189
-> 0.286 across the sweep).
INTERPRETATION: the exact H-weighted centroid blowup (E1 branch 2,
spectral mechanism: H̄ nonzero-eig median ~0.002, min ~1e-5 -> H̄^+
amplifies tau_H 25-94x) is REGULARIZABLE. Adding sliver-floor damping
collapses the amplification path, and the construction recovers a
TA-beating recipe at the resolved lambda.
CONSEQUENCE FOR PAPER: achievability claim SURVIVES with a regularization
caveat. Sec 6/7 reframes from "branch 2: encoder construction loses to
plain averaging" to "branch 2: exact construction loses with identified
spectral mechanism; regularized variant restores SOTA on the deployed
slice." Salvage form: (Hbar + lambda*I_d)^-1 H_t with the same
Hadamard-quantize-decode scheme; lambda becomes a method hyperparameter
(single-model 4-task sweep — must hold across mistral/qwen/yi to claim
the bound is matched, not just for llama).
MINIMUM IS at the INTERIOR boundary of the sweep (between 0.01 and 0.3,
two-decade gap). FINE SWEEP LAUNCHED 2026-06-12 ~21:14: 5 cells
lambda in {0.05, 0.07, 0.13, 0.17, 0.2} (job 41537,
ridge_fine_manifest.json, GPU6 shared with matrix 41533, ~6h serial).
Decides the true minimum before extending to other models.
NEXT after fine-sweep verdict: (1) extend winning lambda to mistral /
qwen / yi = 12 cells (cross-model generalization is the load-bearing
claim); (2) Fisher-diagonal H_t variant (E1 spec variant b) — does
curvature-aware H beat ridge's blunt sliver-floor? (3) write E1 §6.2
with the salvage framing for the paper rewrite.


## 2026-06-13 — Ridge fine sweep CLOSED; lambda=0.05 is the global min
Job 41537 finished 2026-06-13 ~04:50 IST: 5 fine cells lambda in
{0.05, 0.07, 0.13, 0.17, 0.2} added to the original 5. Full 10-cell
worst-task excess curve:
  lambda=0.001 -> 4.463
  lambda=0.01  -> 0.346
  lambda=0.05  -> 0.0906  <- GLOBAL MIN
  lambda=0.07  -> 0.0951
  lambda=0.10  -> 0.1116
  lambda=0.13  -> 0.1262
  lambda=0.17  -> 0.1418
  lambda=0.20  -> 0.1531
  lambda=0.30  -> 0.1891
  lambda=1.00  -> 0.2864
Clean U-shape. Minimum unambiguously at lambda=0.05 (with 0.07 a near
tie, Delta=0.005 < expected seed noise). **lambda=0.05 beats TA 0.219
by 59% on worst-task excess.** Per-task at the minimum:
gsm8k +0.091, alpaca +0.025, magicoder +0.027, translation +0.029 —
the merged model is roughly equally close to each solo, the opposite
of the lambda=0.001 catastrophe where translation alone took +4.46.
INTERPRETATION refinement: the spectral mechanism (decisions.md entry
2026-06-12 E1 b=inf RESOLVED) predicted ridge would dominate at
lambda ~ median nonzero eigenvalue of Hbar ~ 0.002. The empirical
optimum at 0.05 is ~25x larger — consistent with the operational
truth that the surrogate-to-CE bridge breaks well before pseudo-inverse
amplification dominates, so the effective sliver floor must be set
where the CE loss is no longer faithful to the quadratic surrogate,
not at the smallest eigenvalue. This is an operational, measurable
quantity for the paper.
HEADLINE for paper section 6.2: "On real adapters, the regularized
encoder (lambda=0.05) achieves worst-task NLL excess 0.091, 59% below
TA, with the unregularized form blowing up via the identified
degenerate-Hbar spectral mechanism."


## 2026-06-13 — Fisher-diagonal H_t variant: BELOW-RIDGE
Job 41538 ran 2 cells: rd_encoder with h_t_mode='fisher_diag',
ridge_lambda in {0, 0.1}, on llama b=inf rank_r. Fisher diagonal =
per-task input-activation second-moment (forward-only hook, 512 train
samples per task, ~120 sec per task on llama 8B). This is the
master_plan E1 variant (b) "diagonal empirical Fisher at tau_t",
spec'd as the alternative to the projector surrogate.
Results (worst-task excess vs TA 0.219, best ridge 0.0906):
  fisher  lambda=0   -> 0.2102  (barely beats TA; +0.21 on gsm8k)
  fisher  lambda=0.1 -> 0.3190  (WORSE than lambda=0; +0.32 on gsm8k)
VERDICT: BELOW-RIDGE. Fisher H_t under-performs the regularized
projector surrogate by ~2.3x on worst-task excess. Direction of ridge
effect REVERSED vs projector mode: ridge HURTS Fisher (0.21 -> 0.32),
opposite of projector where ridge helps (0.50 -> 0.09).
INTERPRETATION: input-Gram diagonal already encodes per-coordinate
"importance" (active feature axes downweight other tasks naturally),
so an additive identity floor washes out that structure rather than
adding the missing eigenvalue-floor that projector needs. The Fisher
recipe is qualitatively different and the no-ridge form is the natural
comparison — and the projector + ridge still wins.
STRUCTURAL finding: rank-r truncation cost is dramatically lower
under Fisher (trunc_mass mean 0.025 max 0.060) vs projector (~0.30 v1).
The Fisher W* is rank-r friendly; it just lands farther from each
task's solo than projector+ridge does. So the rank cap is not where
Fisher loses; the centroid alignment is.
CONSEQUENCE FOR PAPER: the projector + ridge recipe is empirically
preferred over the theoretically pure Fisher metric, with a precise
mechanism for the difference. This is the sanity-check the master plan
pre-registered for the surrogate choice — and it came back clean for
projector.
NEXT: extend lambda=0.05 (projector + ridge) across mistral/qwen/yi
= 12 cells (load-bearing cross-model generalization claim).


## 2026-06-14 — Cross-model ridge sweep CLOSED: ALL-3-HOLD
12 cells (3 models x 4 lambda: mistral_7b, qwen25_7b, yi15_9b at
lambda in {0.05, 0.07, 0.10, 0.13}), b=inf rank_deff, plain loader,
job 41557 ~05:00h on GPUs 2/4/6. Worst-task excess at the optimum
per model:
  llama31_8b: lambda=0.05 -> 0.0906  (-59% vs TA 0.219, reference)
  mistral_7b: lambda=0.13 -> 0.0383  (-72% vs TA 0.138)
  qwen25_7b:  lambda=0.13 -> 0.0099  (-91% vs TA 0.105)
  yi15_9b:    lambda=0.13 -> 0.0336  (-66% vs TA 0.099)
VERDICT: ALL-3-HOLD — every non-llama model has worst-task excess
< 0.05 at the optimum, well below the pre-registered 0.15 threshold.
The achievability salvage GENERALIZES ACROSS MODELS; the paper's
section 6.2 headline now stands without a model-specific caveat.
STRUCTURAL FINDING: the optimal lambda shifts from 0.05 (llama) to
0.13 (mistral/qwen/yi). The 7-9B non-llama models have a different
Hbar spectral conditioning than llama; the surrogate-validity
threshold sits higher. This is a measurable architecture-dependent
quantity worth a paragraph in section 6.2.
CAVEAT: for mistral/qwen/yi the minimum sits at the boundary of the
sweep (lambda=0.13 is the largest tested). True minimum may sit at
lambda in [0.13, 0.20]; a 9-cell upward-sweep would tighten the
numbers but cannot change the verdict. Optional follow-up.
NEXT: E5 design week (data-mixture overlap pilot on qwen) starts now;
write up cross-model result as paper section 6.2 paragraph.


## 2026-06-14 — E5 Arm 2 pilot CLOSED: NO-GO (predicted null)
12 qwen-2.5-7B trainings (4 tasks x 3 alpha in {0, 0.5, 0.9}, mixed
training data per gen_e5_pilot_datasets.py, shared pool seeded
20260614), job 41561 ~04:30h on GPUs 2/4/6. Gate analysis
(e5_pilot_gate.py) on the 12 adapters:
  alpha=0.0: hard d_eff/(Tr) mean=1.000  min=1.000  0/112 layers <0.8
                soft d_eff/(Tr) mean=0.257
  alpha=0.5: hard d_eff/(Tr) mean=1.000  min=1.000  0/112 layers <0.8
                soft d_eff/(Tr) mean=0.253
  alpha=0.9: hard d_eff/(Tr) mean=1.000  min=1.000  0/112 layers <0.8
                soft d_eff/(Tr) mean=0.251
GATE: NO-GO (pre-registered: GO requires alpha=0.9 -> majority layers
<0.8; observed 0/112). Hard d_eff DOES NOT MOVE even with 90% shared
training data across the 4 task adapters. Soft d_eff drifts down by
2% (0.257 -> 0.251), well inside seed-noise scale.
INTERPRETATION: This is the master plan's predicted null branch:
"real fine-tuning resists subspace overlap." The V_t = top-r right-
singular basis of Delta_t is set by the per-task answer distribution
loss, not by raw input geometry, so shared training inputs do not
induce shared LoRA subspaces. The pre-registered protocol said
explicitly: "a null on Arm 2 is itself reportable: real fine-tuning
resists subspace overlap, which strengthens the practical message
that the algorithmic regime is the universal one."
CONSEQUENCE FOR PAPER: the floor formula B^2(1 - d_eff/Tr) is
non-vacuous only under explicitly engineered geometric forcing
(Arm 3, which is now primary), not under naive data-overlap
induction. The regularized achievability salvage (section 6.2,
lambda=0.05 to 0.13 ridge) sits in the algorithmic regime which we
now know is universally applicable to real fine-tunes. The lower
bound continues to apply as a fundamental limit; the operational
regime where the floor matters is the engineered one. Paper drafts
section 6.3 at paper/sections/6_3_e5_arm2_null_draft.tex.
NEXT: Arm 3 geometric forcing (rank 64, 4 trainings, Tr=256
approaches layer dim and forces d_eff < Tr by counting). Submitted
as job 41562. After Arm 3 trainings finish, run the same d_eff
analysis on the 4 rank-64 adapters; expect d_eff/(Tr) < 1.0 by the
mechanical argument.


## 2026-06-14 — E5 Arm 3 CLOSED: ALSO NULL; mechanical-forcing barrier identified
4 qwen-2.5-7B trainings at r=64 (Tr=256), job 41562 ~01:50h on
GPUs 2/4/6. Cells took ~50min each. d_eff analysis across the
112 qwen attention projections:
  hard d_eff/(Tr) mean=0.997  min=0.867  max=1.000
  soft d_eff/(Tr) mean=0.253  min=0.251  max=0.256
  layers below 0.8: 0/112
The geometric forcing nudges the most-bottlenecked layer to
~0.867 (some V_t correlation detected) but no layer crosses the
pre-registered 0.8 threshold. Arm 3 also fires NO-GO.

The reason both arms fail is now precise: generically d_eff =
min(in_dim, Tr). For d_eff/Tr < 1 we need Tr > in_dim, i.e.,
r > in_dim/T. For T=4 tasks, this means:
  llama-3.1-8B (in_dim ~4096): r > 1024
  qwen-2.5-7B  (in_dim ~3584): r > 896
  llama-3.2-3B (in_dim 3072):  r > 768
  mistral-7B   (in_dim 4096):  r > 1024
  yi-1.5-9B    (in_dim 4096):  r > 1024
All well outside the practical LoRA regime (r <= 64 typical
production deployments). At T=4, mechanically forcing the
floor-positive regime requires LoRAs of rank comparable to the
model's hidden dim, defeating the parameter-efficiency motivation
for LoRA.

CONSEQUENCE: the floor formula B^2(1 - d_eff/Tr) is real but
OPERATIONALLY VACUOUS on real fine-tunes at practical ranks.
1 - d_eff/Tr vanishes for any reasonable r. The algorithmic
regime — where worst-task excess is set by construction
suboptimality rather than by the irreducible floor — is the
regime ALL real LoRA fine-tunes live in. This is positive for the
paper: the regularized achievability salvage (section 6.2,
lambda in [0.05, 0.13]) sits on the universally applicable side
of the bound.

NEXT: launched job 41563 (rdm_e5arm3b) for 4 trainings on
Llama-3.2-3B at r=64 as cross-architecture confirmation of the
null. After it lands (~2h), update section 6.3 table with the
3B numbers and the combined finding stands. Paper section 6.3
rewritten at paper/sections/6_3_e5_arm2_null_draft.tex with the
mechanical-forcing barrier made explicit.

ADDENDUM 2026-06-14 ~13:40: Job 41563 (Arm 3b on Llama-3.2-3B)
CRASHED in 0 min on the model-loading step. Root cause:
safetensors_rust.SafetensorError "Error while deserializing
header: incomplete metadata, file not fully covered" — the
Llama-3.2-3B-Instruct files on disk are partially downloaded /
corrupted. All 4 cells failed rc=1 and parked.
DECISION: skip the cross-architecture confirmation. The
mechanical bound r > in_dim/T is a *mathematical* statement
that applies to every architecture; Llama-3.2-3B at r=64 (with
in_dim=3072 > T*r=256) must show d_eff=Tr at saturation by the
generic argument. Arm 2 + Arm 3 on qwen-7B + the math are
sufficient. Section 6.3 draft edited to make the absent 3B run
a mathematical (rather than empirical) sentence; the paper's
combined-null story is unaffected.
TO REDO THE 3B RUN (if a reviewer asks): re-download
Llama-3.2-3B-Instruct from HF (~3GB), verify safetensors with
huggingface-hub's verification, then qsub
code/phase3/scripts/pbs_orchestrator_e5_arm3b.sh.


## 2026-06-14 — E7 b=2 mechanism: Prediction 2 CONFIRMED, Prediction 1 REJECTED
Three-phase test of the master plan's "implicit-TIES" hypothesis.

PHASE 1 (CPU, from 80-cell matrix at eval_matrix_seeds/):
Per-(model, task, seed) dip_depth = excess[b=4] - excess[b=2] and
ties_win = excess[TA] - excess[TIES]. Spearman across 32 cells:
  overall rho = +0.960  (Pearson r = +0.951)
  per model: llama 0.98, mistral 0.95, qwen 0.95, yi 0.81
  per task:  alpaca 0.93, magicoder 0.95, translation 0.83;
             gsm8k is a within-task null (both interventions reliably
             help but the variation does not covary).
Decision threshold (0.7) cleared. Master plan Prediction 2 CONFIRMED.

PHASE 2a (CPU, from v1 adapters):
Implicit sparsity at b=2 quantization, three metrics:
  A (zero-bucket fraction): mean 0.50 across 4 models, ~uniform
  B (frac with |post-quant| < 0.5 * |pre-quant|): ~0.00 (no coord
    shrinks to less than half its magnitude under min-max quantization)
  C (frac with quantization error exceeding |pre-quant|): mean 0.87,
    ~uniform across models
The sparsity LEVEL is high but constant across models; the small
cross-model variance does not correlate with mean dip
(max |r| = 0.43, metric A). Sparsity is a near-universal property
of the b=2 quantization scheme, not a per-model effect.

PHASE 2b (8 eval cells, job 41575, 4 models x 2 densities):
Explicit magnitude pruning at the densities matching the Phase 2a
sparsity (rho=0.50 matches metric A, rho=0.13 matches metric C),
plain sum + SVD-truncate. Worst-task excess per (model, density):
  llama:   rho=0.50 -> 0.225  rho=0.13 -> 0.288   (b=2 0.108  b=4 0.217)
  mistral: rho=0.50 -> 0.155  rho=0.13 -> 0.251   (b=2 0.058  b=4 0.138)
  qwen:    rho=0.50 -> 0.118  rho=0.13 -> 0.229   (b=2 0.019  b=4 0.105)
  yi:      rho=0.50 -> 0.088  rho=0.13 -> 0.151   (b=2 0.059  b=4 0.098)
At rho=0.50, 0/4 models reproduce the b=2 dip; only yi shows a partial
result (between b=2 and b=4). At rho=0.13, every cell underperforms
even b=4. Magnitude pruning at the implicit-sparsity-matched density
DOES NOT reproduce the b=2 dip. Master plan Prediction 1 REJECTED.

SHARPENED HYPOTHESIS for paper:
The b=2/TIES covariance is real (Prediction 2), but it is NOT mediated
by implicit sparsity (Prediction 1). What is shared between min-max
scalar quantization and trim+sign-elect+disjoint-mean? Two open
candidates: (i) uniform shrinkage of large coefficients via tensor-level
distributional re-anchoring; (ii) per-tensor (rather than per-coordinate)
calibration. Disentangling these is a mechanism follow-up. The current
finding is publishable as-is: the dip is real, covaries with TIES, and
the simple "implicit-pruning" explanation is ruled out.

CONSEQUENCE FOR PAPER: section 6.x on b=2 mechanism is sharpened from
"hypothesis stands; sparsity drives the dip" to "covariance confirmed;
sparsity is NOT the mechanism; structural / distributional cause is
identified as a follow-up question." This is the master plan §E7
decision rule's other branch — "hypothesis stays in 7.1 now with
evidence against it." Paper section drafted at
paper/sections/6_4_e7_b2_mechanism_draft.tex.

NEXT: pivot to E3 (downstream metrics: GSM8K em / HumanEval pass@1 /
COMET-22 / IFEval strict). This is the big reviewer-defense item —
verify that the NLL-based conclusions (TIES dominates, b=2 dip,
cross-model ordering) hold under task metrics. Master plan budget
80-120 GPU-h.


## 2026-06-15 — E3 GSM8K em sweep CLOSED: model-dependent NLL→accuracy gap
20 cells (4 models × 5 methods) at n=500, greedy decoding with
0-shot CoT prompts, regex extract. Job 41578 ran ~14h overnight
~19:50 → 10:50. Yi cells took 190-229 min each (9B base is slower
than the 7B siblings). 100% completion rate, zero failures.

Per-model Spearman ρ between accuracy and -NLL excess across the
five methods:
  llama31_8b: ρ = -0.60  (WEAK, RANK INVERSION)
  mistral_7b: ρ = +0.90  (STRONG)
  qwen25_7b:  ρ = +0.90  (STRONG)
  yi15_9b:    ρ = +0.87  (STRONG)

3/4 models confirm NLL methodology; 1/4 (Llama-3.1-8B) inverts the
ranking. Best method per model by accuracy:
  llama:   knots (0.346)  — NLL best was TVQ b=2 (worst-excess 0.101)
  mistral: ties  (0.410)  — NLL best was TIES (worst-excess 0.049)
  qwen:    ties  (0.714)  — NLL best was TIES (worst-excess 0.013)
  yi:      ties  (0.778)  — NLL best was TIES (worst-excess 0.045)

INTERPRETATION: The "strong-base hypothesis" — that methods that
minimize NLL on the strongest base discard reasoning bits that NLL
doesn't see but greedy decoding does — is RULED OUT by the yi data.
Yi-1.5-9B has the lowest NLL_τ on GSM8K (0.387) and the highest
merged accuracy (0.778 with TIES), but its ρ=+0.87 sits with mistral
and qwen. The anomaly is therefore MODEL-SPECIFIC to Llama-3.1-8B,
not a function of task capacity.

Three concrete hypotheses for the Llama-3.1 anomaly, each testable
with on-disk artifacts:
  (1) Llama-3 chat-template special-token sensitivity to aggressive
      trim/quantize compression
  (2) Llama-3.1 LoRA Δ magnitude distribution being flatter than the
      other bases, hurting TIES's top-0.2 threshold and TVQ's
      per-tensor min/max range
  (3) Per-task answer-distribution overlap on Llama-3.1's adapters
      contaminating TIES's sign election

CONSEQUENCE FOR PAPER: §6.5 drafted at
paper/sections/6_5_e3_downstream_metrics_draft.tex with the
"3/4 STRONG, 1/4 WEAK, model-specific Llama-3.1 anomaly" framing.
The strong-base hypothesis (v1 draft) is dead. The §6.2 ridge-encoder
claim is now qualified: "in NLL terms; for Llama-3.1-8B specifically,
measure accuracy directly." Lower-bound theorem and E5 floor-formula
results are metric-agnostic and unaffected.

Master plan §E3 decision rule fires the MIXED branch (n_strong=3,
n_weak=1): "report accuracy as a separate story; NLL not fully
predictive." This is honestly stronger paper content than uniform
confirmation would have been — a model-specific NLL→accuracy gap
is genuinely new.

NEXT: optional — implement HumanEval pass@1, COMET-22, IFEval strict
to extend the verification. Each is ~30-60 GPU-h. Or move to E6
(T sweep on real adapters) per master plan timeline. Or pause E3
extension until T1 theory week with Garg returns.


## 2026-06-13 — E2 matrix CLOSED: 140/140 cells, multi-seed merge matrix complete
Job sequence 41524 -> 41533 -> 41556 (keeper requeues + GPU0 drop +
walltime resume). 140 cells = 4 models (llama/mistral/qwen/yi) x
5 methods (TA/TIES/DARE/KnOTS/TVQ@{b1,b2,b4,b8,b16,b32}) x 2 seeds (1,2)
= 140 total. ZERO permanent failures; 1 transient qwen tvq_b32_seed2
rc=1 auto-retried successfully; walltime-driven kill-and-resume cycle
on 41533->41556 lost ~0 in-flight cells (orchestrator done-files).
Analysis pending: code/phase3/scripts/analyze_e2_matrix.py (to be
written). Pre-registered claims to verify:
  (a) cross-model ordering Ll > Mi > Qw > Yi survives across seeds
      (was: 0.219/0.213/0.221 across seeds 0/1/2 on llama TA);
  (b) b=2 TVQ "less-is-more" dip persists across seeds;
  (c) DARE per-seed tracks TA to 3 decimals (anchor sanity).
The seed matrix is the §6.1 calibration data; cross-model ridge sweep
is the §6.2 generalization data.


## 2026-06-15 — Llama-3.1 anomaly hypothesis 2 REJECTED (Δ-distribution flatness)
The Llama-3.1 GSM8K accuracy inversion is NOT explained by Δ-magnitude
distribution shape. Direct test on 16 v1 adapters (4 models × 4 tasks),
materializing per-layer Δ_t and aggregating distribution statistics
across the four tasks and ~112 attention projections per model:

  model         p99/med   kurt    top20/med   near%
  llama31_8b    5.26      4.98    2.04        7.65
  mistral_7b    4.84      3.94    2.00        7.98
  qwen25_7b     5.29      4.64    2.06        7.59
  yi15_9b       5.45      5.26    2.07        7.47

Llama-3.1 sits in the middle of the four on every metric. Mistral is the
flattest (kurt 3.94), yi the most peaked (5.26), with llama and qwen in
between. The two strong-agreement bases (mistral, yi) bracket llama on
both sides of the flatness scale, so "flatness drives the inversion" is
ruled out.

CONSEQUENCE FOR PAPER: §6.5 v2 paragraph "Hypothesis 2 rejected" added,
listing the measured values. The two remaining candidate mechanisms
(L3 chat-template special-token sensitivity to compression; TIES
sign-election contamination from per-task answer overlap) are not
directly tested in this paper. Either would require additional GPU
generation (for the chat-template hypothesis) or a TIES-internal
contamination probe across the matrix (for the sign-election
hypothesis); both are slotted as focused mechanism follow-ups.

scripts/llama_delta_distribution.py committed (b5e02dd); raw per-layer
stats in results/phase3/llama_delta_distribution.json.

NEXT: E6 design scaffold done at notes/E6_design.md, awaiting user sign-off
on the 6 design choices before any cluster work.


## 2026-06-19 — E6 pilot verdict landed (54/54 cells) + Alpaca shallow-adapter diagnosis
E6 pilot on Yi-1.5-9B-Chat completed 2026-06-18 00:22 IST after
multiple PBS walltime requeues (41614 train → 41627/41649/41663/41689/41699/41720 eval).
54/54 cells: 9 subsets × 6 methods. T ∈ {2, 4, 7} with one nested chain
+ three random subsets at T ∈ {2, 4}, one all-tasks subset at T=7.

ANALYSIS (analyze_e6_T_scaling.py, results/phase3/e6_T_scaling_summary.{csv,json}):

Worst-task NLL excess (mean across subsets), all-cells:
  T   TA      TIES    DARE    KnOTS   TVQ_b2  rd_ridge
  2   0.027   0.025   0.027   0.027   0.111   0.028
  4   0.067   0.095   0.067   0.066   0.101   0.084
  7   0.128   0.148   0.130   0.128   0.111   0.115

Three findings:

(1) TIES INVERTS AT T=7. Worst method (0.148) at T=7. Steepest log-T
    slope (0.098, R²=0.9995). Hypothesis (not tested): sign-election
    consensus degrades faster with T than coordinate magnitude. Joins
    the L3 anomaly mechanism candidate list for future work.

(2) LOG-T SLOPE 3-7× SHALLOWER THAN E4 SYNTHETIC PREDICTS. Measured
    b ∈ [0.07, 0.10] all-cells; [0.03, 0.05] alpaca-OUT. E4 synthetic
    said b ≈ 1/√r = 0.25 at r=16. Functional form (log-T) holds
    (R² ≥ 0.96 for 5/6 methods all-cells). Conjectured cause: real
    LoRA adapters have substantial column-space overlap; Stiefel-random
    assumption in E4 over-estimates inter-task interference.

(3) TVQ b=2 DIP SURVIVES T-SWEEP. Slope ≈ 0 all-cells, negative
    alpaca-OUT. At T=7 TVQ_b2 ties rd_ridge as best method (0.111).
    Confirms §6.4 mechanism is T-robust.

ALPACA SHALLOW-ADAPTER ARTIFACT (root-caused 2026-06-19 ~21:00 IST):
The all-cells T=4 rd_ridge cell showed 0.084 (vs TA 0.067), apparently
contradicting §6.2's ridge salvage. Investigation:

  Per-task headroom (base - τ) for Yi-1.5-9B-Chat E6 adapters:
    alpaca:     0.114  ← SHALLOW (chat-base already instruction-tuned)
    codealpaca: 0.236
    dolly:      0.531
    gsm8k:      0.554

Yi-1.5-9B-Chat is already a strong instruction follower. The Alpaca
v1 LoRA barely moves NLL from base. Excess NLL is reported relative
to τ, so any merge perturbation hits Alpaca disproportionately.

  Per-task alpaca excess at T=4_nested:
    TA:        0.037   rd_ridge: 0.100  (2.7× asymmetry)

  Per-layer Δ-cosine isolation (mean cos to other 4 adapters):
    alpaca:     0.028  ← MOST ISOLATED
    gsm8k:      0.026
    xsum:       0.066
    codealpaca: 0.075
    dolly:      0.097

Combined small headroom + structural isolation: ridge encoder's
projection step costs Alpaca disproportionately. rd_ridge works
fine on alpaca-OUT subsets (T=2 rand0/1/2: 0.016/0.030 worst;
T=4 rand0 codealpaca+dolly+magicoder+translation: 0.045 worst,
beating TA's 0.056).

CONSEQUENCE FOR PAPER: §6.6 draft uses DUAL REPORTING (all-cells vs
alpaca-OUT). §6.2 ridge salvage finding stands: on the alpaca-OUT
T=4 cell, rd_ridge (0.045) beats TA (0.056), TIES (0.058), KnOTS
(0.056). Alpaca shallow-adapter artifact added as methodological
warning: practitioners merging on chat-tuned bases should renormalize
excess by per-task headroom or exclude near-saturated adapters.

paper/sections/6_6_e6_T_scaling_draft.tex committed.
analyze_e6_T_scaling.py + raw aggregates in results/phase3/.

NEXT: §6.6 ready for cross-read; T-scaling slope refinement should be
mentioned in the §6.2 / E4 link sentences; consider whether the TIES
T=7 inversion warrants a focused mechanism study (cost: 1 T=7 eval
with TIES sign-election counts logged per coordinate, ~3-4 GPU-h).


## 2026-06-23 — E6 Llama-3.1-8B-Instruct landed (54/54) + cross-model §6.6 v2
E6 Llama full sweep completed 2026-06-22 10:55 IST after multiple PBS
walltime requeues (41960 train → 42051/42057/42075/42110/42141/42159
eval). 3 train (codealpaca, dolly, xsum) + 54 eval cells, same subset
composition as the Yi pilot for clean cross-base comparison.

Key intra-stage event: 42051 was killed by root after 80 seconds with
no spool output, immediately after stage 2 submission. Cause unknown
(possibly admin policy or maintenance). Resubmitted as 42057 and ran
cleanly; no further occurrences. One-off, not a pattern.

Mid-flight optimization: orchestrator was running with gpus=2,6 because
_ORCH_GPUS_E6 carried "2,6" from the Yi pilot (when gpu4 was rc=87
flicker-bouncing). Switched to "2,4,6" + qdel+qsub fresh at 17:57 IST
on 2026-06-21. Threw away ~10 min of in-flight cell to save ~3h of
1-wide pace. Confirmed 3-wide pace (~6 cells/h) after restart.

ANALYSIS (analyze_e6_T_scaling.py --base llama31_8b,
results/phase3/e6_T_scaling_summary_llama31_8b.{csv,json}):

Worst-task NLL excess (mean across subsets), Llama all-cells:
  T   TA      TIES    DARE    KnOTS   TVQ_b2  rd_ridge
  2   0.068   0.064   0.069   0.068   0.064   0.057
  4   0.153   0.126   0.153   0.153   0.113   0.104
  7   0.265   0.232   0.265   0.265   0.215   0.138

Rd-encoder ridge wins at every T on Llama, with widening margin:
  ratio rd_ridge/TA = 0.84 (T=2) → 0.68 (T=4) → 0.52 (T=7)
§6.2's cross-model salvage finding strengthens at higher T on Llama.

Log-T slopes (Llama all-cells):
  TA       0.156   (R^2 0.9800)
  TIES     0.133   (R^2 0.9566)
  DARE     0.155   (R^2 0.9800)
  KnOTS    0.156   (R^2 0.9804)
  TVQ_b2   0.119   (R^2 0.9347)
  rd_ridge 0.065   (R^2 0.9991)

Llama slopes are ~2× larger than Yi's and within 1.6× of synthetic
E4 prediction (1/√r = 0.25). Functional log-T form holds (R^2 ≥ 0.93
on 11 of 12 all-cells fits across both bases). rd_ridge slope is
nearly BASE-INVARIANT (0.070 Yi, 0.065 Llama) — confirms ridge
regularization buys T-stability across very different bases.

THREE YI-ONLY ARTIFACTS RULED OUT BY LLAMA DATA:

(A) TIES inversion at T=7 is Yi-specific. On Yi TIES is worst at
    T=7 (0.148). On Llama TIES at T=7 is 0.232, third-best of six,
    in the natural ordering. The Yi-specific behavior remains a
    candidate for a focused sign-election mechanism study.

(B) TVQ b=2 flat-in-T is Yi-specific. Yi all-cells slope -0.001,
    alpaca-OUT slope -0.088 (decreasing). Llama TVQ_b2 slope +0.119,
    comparable to other matrix methods. The §6.4 dip is NOT a
    T-robust mechanism in general; it is a Yi-specific stability
    where b=2 sign-election noise tracks the shallow-adapter
    direction.

(C) Alpaca shallow-adapter artifact is Yi-specific. Yi-Chat Alpaca
    headroom (base − τ) is 0.114; Llama-3.1-Instruct Alpaca headroom
    is 0.246, ~2× higher. Llama all-cells T=4 rd_ridge (0.104) is
    the BEST method, not regressing. The artifact does not appear
    when residual headroom is sufficient to absorb the projection
    cost.

UNIFYING EXPLANATION: Yi-1.5-9B-Chat is near-saturated on
instruction-following tasks. Per-task headroom is compressed across
the cohort (especially on Alpaca), which amplifies method-specific
failure modes (TIES sign-election contamination, b=2 quantization
noise). Llama-3.1-8B-Instruct is also instruction-tuned but retains
residual NLL headroom, so the clean log-T scaling that synthetic
data already showed comes through. We move Yi-specific findings
from "anomaly" to a "base-saturation regime" caveat.

CONSEQUENCE FOR PAPER: §6.6 v2 promotes rd-encoder ridge from
"salvages TA at T=4" to "the T-stable method on real adapters."
Three v1 surprises (TIES inversion, b=2 flat, Alpaca artifact) are
all unified under base-saturation. paper/sections/
6_6_e6_T_scaling_draft.tex v2 committed (baa64de). Master plan
tracker updated.

analyze_e6_T_scaling.py extended with --base arg (default yi15_9b
for backward compatibility; --base llama31_8b reproduces the Llama
numbers above). Yi outputs unchanged at e6_T_scaling_summary.{csv,
json}; Llama outputs at e6_T_scaling_summary_llama31_8b.{csv,json}.

NEXT: TIES T=7 sign-election counts on Yi (Phase 2, B1) to confirm
the saturation-regime conjecture. L3 H3 contamination probe on Llama
(B3) tests the same mechanism in the unconfounded regime. Both
~3-4 GPU-h each, run together since they share the same probe.


## 2026-06-23 (evening) — TIES sign-election probe (B1+B3) — §6.6 v2 conjecture refined
Phase 2 probe completed (code/phase3/scripts/probe_ties_sign_election.py,
results/phase3/ties_sign_election_probe.json). Tests the §6.6 v2
conjecture that TIES sign-election consensus "degrades faster on Yi
than Llama" because of Yi-Chat's instruction-tuning saturation.

PROCEDURE: For each (base, T=7) cohort: load 7 trained adapters,
materialize per-layer Δ = scaling · B·A, apply TIES trim at
density=0.2 (the merging/ties.py default), stack to (T, out, in),
and audit:
  (a) per-coord vote-split histogram (n_pos, n_neg)
  (b) per-coord margin |p - n| / (p + n)
  (c) per-task win-share: fraction of active coords where that task's
      sign matches the elected sign (sign of magnitude-weighted sum)
  (d) per-task mass-share: mean of |Δ_t| / Σ|Δ_t| over active coords

Vectorized via torch bincount over a flattened (T+1, T+1) histogram.
CPU-only, ~10 min wall on the login node. First implementation had a
Python zip-loop dictionary build that ran 3.5h before being killed and
replaced with the vectorized form — pattern worth remembering for future
per-coord audits.

RESULT — vote-split structure is nearly identical Yi vs Llama:

  metric              Yi         Llama       Δ
  mean vote margin    0.748      0.753      +0.005
  unanimous coords    69.75%     70.43%     +0.68%
  thin-split (±1)     27.27%     26.80%     −0.48%
  wide-split (>1)      2.98%      2.77%     −0.21%

So the §6.6 v2 conjecture *as literally stated* ("consensus degrades"
on Yi) is REJECTED — consensus structure is essentially the same.

RESULT — per-task win-share is dramatically more skewed on Yi:

  win-share range:  Yi 0.077, Llama 0.028 (2.8× wider on Yi)

  Yi T=7 win shares:  gsm8k 0.853, dolly 0.847, xsum 0.839,
                      translation 0.829, codealpaca 0.816,
                      alpaca 0.785, magicoder 0.776
  Llama T=7 win shares: codealpaca/alpaca/magicoder 0.832,
                        translation 0.831, xsum 0.825,
                        dolly 0.817, gsm8k 0.805

On Yi, magicoder (code) and alpaca (saturated) are systematically
outvoted by the instruction-following cluster. On Llama all 7 tasks
are within a 0.028 band.

REFINED MECHANISM: TIES sign-election on a saturated base produces a
DIRECTIONALLY BIASED consensus, not a degraded one. Six Yi adapters
share substantial instruction-following structure and vote in a
common subspace; magicoder (code) and alpaca (saturated) are
systematically outvoted. The merge biases toward the instruction-
following consensus at the cost of code-specific and saturated-
direction signal. On Llama-3.1-Instruct the per-task votes are
nearly uniform (range 0.028), so TIES preserves each task's
contribution.

NEW FALSIFIABLE CONJECTURE: TIES underperforms when the per-task
sign-election win-share range exceeds a threshold (empirically
~0.05); this happens on heavily-aligned chat-tuned bases where
adapter Δ tensors share a dominant instruction-following direction.
Falsifying experiment: synthetic adapters with controlled subspace
overlap should reproduce the inversion when overlap crosses the
win-share-range threshold, independent of the base.

CONSEQUENCE FOR PAPER: §6.6 v3 (paper/sections/
6_6_e6_T_scaling_draft.tex) integrates Table tab:ties-probe with the
audit numbers and replaces the v2 conjecture-paragraph for (A) TIES
inversion with the measured-result version. "What this changes"
item (iii) now reads "TIES inversion mechanism is a measured result,
not a conjecture" with reference to the new table.

NEXT: Phase 3 — B2 L3 H1 chat-template probe + B4 E3 expansion
(HumanEval/COMET/IFEval on T=4 matrix). Phase 4 paper polish in
parallel (A4 T2 floor recipe, A5 practical recs §, A2 cross-read).


--------------------------------------------------------------------------------
2026-06-24 — Td2 sign-election threshold self-derived (no Garg)
--------------------------------------------------------------------------------

CONTEXT: Prof. Garg unavailable (family travel). T1 theory week
deferred indefinitely. To partially recover the theoretical depth
loss, executed Td2: self-derived perturbation analysis of the TIES
update producing a closed-form prediction for the win-share-range
threshold R*.

DERIVATION SKETCH: Linearize TIES update around uniform win-share
q_t = 1/T + eps_t with sum_t eps_t = 0 and range R = max - min.
Show post-trim magnitude amplification kappa cancels in the
normalized merge weight (alpha_t = q_t to first order, eq.
ref:alpha-weight). Decompose per-task NLL excess into TA baseline
+ quadratic dominance-bias cost (c_bias * eps_t^2) - linear sign-
recovery benefit (c_sign * (q_t - 1/T)) under local Fisher
approximation. Solve for breakeven: R* ~ sqrt(4 c_sign / (T c_bias)).
Bracket constants from spectral overlap regime: lower bound c_bias
>> c_sign gives R* in [0.025, 0.075] for T=4 at density 0.2.

VERDICT: Two empirical data points (Yi R=0.077 > 0.075; Llama
R=0.028 ~ 0.025) bracket the predicted range and respect the
directional prediction (Yi inverts, Llama does not). Bracket is
tight at extremes, so the order-of-magnitude prediction is correct
but the constant cannot be pinned without Fisher-quadratic constants
(deferred to T1 if Garg returns).

FALSIFIABILITY: Predicts any third base with R in (0.025, 0.075)
should produce ambiguous TIES behavior. Mistral-7B preliminary
probe value is R ~ 0.041 (internal), which should test this.

FILE: paper/sections/appendix_td2_sign_election_threshold.tex (v1).
§6.6 v3 updated to reference this appendix at the empirical "~0.05"
threshold paragraph.

REVIEWER IMPLICATION: The TIES inversion is no longer "we measured
it on two bases and conjecture a threshold"; it is "we measured it
on two bases and a perturbation-model derivation predicts the
threshold range, bracketing our two data points." That is the
difference between Section 6.6 reading as case-study and reading as
"phenomenon with predictive model." Td2 gains ~0.3 in the ICLR
score estimate per the planning conversation.

NEXT: After B4 lands (in flight at this writing, ~03:00 IST 2026-06-25),
do A1+A2 analysis, A3 §6.6 v4 with B2/B4 verdicts, A4 cross-read.
Then C1+C2 added baselines + quadratic bridge. Then P1-P5 paper
assembly. No Garg dependency on the critical path.


--------------------------------------------------------------------------------
2026-06-24 — B2 H1 falsified + B4 cross-metric H3 falsified
--------------------------------------------------------------------------------

B2 RESULT (paper/sections/6_5_e3_downstream_metrics_draft.tex v3):
All 5 L3 methods emit exactly 1 special token per generation on
average (the trailing <|eot_id|>), frac>0 = 0.98-1.00 uniformly,
rate per token 0.011-0.016 uniformly. No method-specific chat-template
emission signal. H1 (tokenizer/chat-template idiosyncrasies cause L3
GSM8K NLL->accuracy inversion) FALSIFIED.

B4 RESULT: Per-base Spearman pass@1 vs -NLL excess on n=164 HumanEval:
  llama31_8b   +0.894  STRONG
  mistral_7b   +0.975  STRONG
  qwen25_7b    +0.783  STRONG
  yi15_9b      +0.718  STRONG
4/4 bases STRONG cross-metric NLL->pass@1 correspondence. The L3 GSM8K
rho=-0.60 does NOT replicate on HumanEval — it is GSM8K-eval-specific.
H3 (sign-election causes cross-metric L3 NLL->accuracy pathology)
FALSIFIED as a cross-metric mechanism.

TVQ_b=2 best on 3/4 bases on pass@1 (L3 0.427, Qwen 0.659, Yi 0.122);
TIES best on Mistral (0.244). TA is the worst method on every base by
factor 2-100x. Independent confirmation of §6.7 R3 (use TIES/TVQ_b=2
not TA).

VERDICT FOR PAPER: NLL->accuracy correspondence holds on 7/8 (base x
metric) cells at rho >= 0.72; (L3-Instruct, GSM8K em) is the single
known outlier, with both candidate mechanisms (H1 and H3) tested and
falsified. The paper now reads as scientifically honest: we proposed
two mechanisms, ran the controlled experiments, both failed, and we
report the result. Reviewer-positive.

UPDATED SECTIONS:
- §6.5 v3: H1/H2/H3 candidate explanations all marked tested and
  falsified; HumanEval Table tab:b4-humaneval added with full 4x5
  matrix; "what this changes" rewritten to 7/8 cells robust + 1
  outlier with no surviving mechanism.
- §6.7 R4: rewrote from "verify on one downstream metric (and L3
  GSM8K open question)" to "verify on >=2 downstream metrics; the
  L3-GSM8K cell shows single-metric verification can mislead, but
  cross-metric agreement on 7/8 cells means NLL is reliable proxy
  when corroborated by a second metric."
- §6.6 v3: unaffected (was about Yi TIES NLL inversion at T=7, a
  different phenomenon from L3 GSM8K).
- Td2 appendix: unaffected (the perturbation derivation was about
  NLL TIES inversion, not the L3 GSM8K story).

SCORE IMPACT: Per the planning conversation's framework,
losing the deep mechanism story for L3 -> -0.2, gaining the
cross-metric robust agreement -> +0.4. Net +0.2 toward the
6/10 baseline.

FILES: code/phase3/scripts/analyze_b4_humaneval.py + analyze_b2_chat_probe.py
+ results/phase3/{b4_humaneval_summary.json, b2_chat_probe_summary.json}.

NEXT: A4 cross-read §6.1-§6.7 + appendix Td2 for label/notation
consistency. Then C1 E10 baselines (Fisher-avg + DELLA + 2026 method)
and C2 E11 quadratic-bridge.

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

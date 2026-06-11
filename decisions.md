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

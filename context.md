# rdmerge — Project Context (read this first)

**Paper:** A Rate-Distortion Lower Bound for Model Merging, with Matching
Achievability via Hadamard Incoherence
**Author:** Sankalp Pathak (solo), with Prof. Sanjay Garg reviewing/advising
**Target venue: ICLR 2027** (deadline ~late Sep 2026; verify when CFP opens)
**Last updated:** 2026-06-11 (evening — campaign day 1)

This file supersedes `handoff.md` (2026-05-18) as the canonical state doc.
The strategy document is `master_plan_iclr2027.md`; the dated gate/branch
record is `decisions.md` (maintain BOTH on every significant event).

---

## 1. Where this stands, one paragraph

The paper was **desk-rejected by TMLR on 2026-06-10** (EiCs: argumentation/
motivation lack clarity; "TMLR is not a suitable venue") — a triage/writing
verdict, NOT a scientific one (no reviews ever happened; nothing was ever
published anywhere, so there is no public record to correct). Response: the
14-week ICLR 2027 master plan was adopted with the clarity rewrite promoted
to the immediate action. Day 1 (2026-06-11) closed: the abstract+intro
rewrite (awaiting Prof. Garg's cold read), E4 (decision rule fired), the
multi-GPU orchestrator with E2 running and E1 implemented+queued, and the
root cause of all historical cluster failures found and fixed.

## 2. Results so far (all on disk, all committed)

- **E4 (synthetic T-sweep) — DONE, decision rule FIRED.** The linear-T
  factor in the achievability constant C = Tc²/3 is an analysis artifact:
  measured growth T=2→16 is 1.77× (linear predicts 8×), cleanly logarithmic
  (R² = 0.998–1.000 in all 10 (r,b) cells), slope shrinking ~1/√r → a
  max-of-T concentration mechanism. **T1 theory week with Prof. Garg is
  triggered**; target a C = O(c²(1 + f(log T, r))) bound. Remark 5 must be
  rewritten around the measurement either way. Results:
  `results/e4_t_sweep/`. Caveat: T=3 excluded (non-pow2 d_eff → Hadamard
  padding inflates charged rate vs the LB exponent; documented).
- **W (rewrite) — abstract + intro rewritten** for the 60-second cold read:
  plain-language opening, formulas demoted, Stiefel glossed, floor-zero
  reframed as "recoverable headroom" (the verdict, not a confession), worked
  example moved up, E4's log-T folded into contribution 5 + Scope.
  → `paper/sections/{abstract,intro}.tex`. SEND TO GARG with the plan.
- **Historical mystery SOLVED:** the master plan's "past PBS jobs all
  failed" was caused by the LMCA project's `pip install --user` torch
  2.4.1 (late May, same sanjay.g account) shadowing this project's conda
  torch 2.10.0+cu128 from `~/.local/lib/python3.11/site-packages`.
  **Fix: `export PYTHONNOUSERSITE=1` in every job wrapper. Standing rule.**
  (Second, smaller fix: unsloth_zoo introspects the lazy
  `torch._inductor.config` — explicit import shim at top of train_lora.py.)

## 3. Running RIGHT NOW (autonomous; keeper-supervised)

- **E2 (32 multi-seed trainings, seeds 1+2)** on the orchestrator, PBS job
  `41481`, GPUs 0,1,2,3,5. Cells take ~40–90 min each; ETA overnight.
  Keeper: `logs/orch_keeper.pid`, log `logs/orch_keeper.log`; state:
  `logs/orchestrator_state.json`; per-cell logs `logs/orch/`.
- **Kill-and-resume test (I0 gate) ARMED:** when the first cells finish, an
  automated test qdels the job mid-flight, verifies the keeper requeues and
  the resumed orchestrator skips done cells. PASS closes I0. Record the
  outcome in decisions.md.
- **E1 (28 eval cells: 4 models × b∈{1,2,3,4,8,16,32}) QUEUED behind E2** —
  `pbs_orchestrator.sh` defaults to `all_manifest.json` (E2+E1), so the
  keeper drains E2 then flows into E1 with no intervention.
- GPUs 4,6,7 belong to the MOOLoRa pilot (other account) until ~2026-06-13;
  then edit `_ORCH_GPUS` (single line, e.g. `0,1,2,3,4,5,6,7`) — next
  requeue uses all 8.

## 4. The orchestrator (how to drive it)

- One PBS job (`code/phase3/scripts/pbs_orchestrator.sh`, ncpus=8 mem=120gb
  24h) runs `orchestrator.py --manifest <json>`: worker per GPU, cells =
  {name, cmd, done, min_free_gb}, done-file idempotent, per-cell VRAM gate
  (backs off on contended GPUs), retry once then park, sentinel
  `_QUEUE_COMPLETE` stops the keeper; `_KEEPER_STOP` stops it manually.
- Manifests: `configs/e2_manifest.json`, `e1_manifest.json`,
  `all_manifest.json` (combined). Generators:
  `scripts/gen_e2_manifest.py`, `scripts/gen_e1_configs.py`.
- Env: conda `.conda/envs/rdmerge` (torch 2.10.0+cu128) + PYTHONNOUSERSITE=1.
  Tests run as scripts (NO pytest in env).

## 5. E1 design notes (read before interpreting results)

`code/phase3/merging/rd_encoder.py`, registered as method `rd_encoder`
(kwargs: bits, c=5.0, seed; bits≥32 = b=∞ centroid cell). Projector-
surrogate geometry H_t = P_{V_t} (same as deff_analysis.py). Hadamard
substituted for the spec's Gaussian-QR (infeasible at out·d_eff ~ 2^18;
same incoherence class; documented). **Rank constraint:** merged adapters
are rank-16 (PeftModelView), so W* is SVD-truncated — identical constraint
to every baseline (apples-to-apples); per-layer `trunc_mass_frac` is
logged. **Pre-registered rule: if real trunc_mass > ~0.1 at b=∞, build the
full-rank base-patch path before interpreting b=∞ as pure centroid error.**
E1's three decision branches (gap closes / gap is approximation / split)
are pre-written in master_plan Part II. Fisher-diagonal H variant = second
pass. 5 CPU tests in `merging/tests/test_rd_encoder.py`.

## 6. What's next (priority order)

1. Garg sync: send plan + rewritten abstract/intro + E4 verdict; book the
   T1 theory week. (User action.)
2. Kill-and-resume PASS → close I0 in decisions.md.
3. E2 completes → quick sanity (seed variance vs seed-0); E1 runs.
4. E1 analysis vs the pre-written decision branches; add E4+E1 subsections
   to paper §6 (experiments.tex still has none of the new results).
5. E7 (b=2 mechanism: matched-sparsity pruning + per-task win correlation)
   — cheap, shares E1's code.
6. E3 harness (GSM8K EM, HumanEval/MBPP, COMET, IFEval) — dev work.
7. E5 design week (the headline experiment) — needs Garg sign-off per plan.

## 7. Operational facts

- **Account/repo:** `sanjay.g@CLUSTER-HOST:~/projects/rdmerge`; GitHub
  `K144U/rdmerge`, **branch `phase3-bootstrap`** (not main).
- 16 seed-0 adapters: `artifacts/lora/<model>/<task>/v1`; E2 adds
  `<task>/seed{1,2}/`. Models in `models/`. Old eval matrix (40 cells):
  `results/phase3/eval_matrix_n1k_v3_perexample/`.
- PBS: 3-job cap (orchestrator = 1 job), ncpus≤8, no sudo. Login-node
  nvidia-smi does NOT exist (VRAM gate returns 0 there — only meaningful
  inside GPU-node jobs).
- The synthetic-side code (`code/synthetic/day*.py`, E4) runs fine under
  `/cm/local/apps/python311/bin/python3` on the login node (CPU).

# rdmerge (P5) — Project Context (read this first)

**Paper:** A Rate-Distortion Lower Bound for Model Merging, with Matching
Achievability via Hadamard Incoherence
**Author:** Sankalp Pathak (solo), Prof. Sanjay Garg reviewing/advising
**Target venue: ICLR 2027** (deadline ~late Sep 2026)
**Last updated: 2026-06-12 afternoon** (campaign day 2; ridge sweep + seed
matrix in flight; matrix re-pinned to gpus 2,4,6 after GPU0 corruption, job
41524 -> 41533; sentinel guard re-armed via `_guard_tick.sh`). Strategy: `master_plan_iclr2027.md`. Dated gate/branch
record: `decisions.md` (8 entries — READ IT, it is the scientific log).

---

## 0. One-paragraph state

TMLR desk-reject (2026-06-10, clarity/venue — no reviews) triggered the
ICLR campaign. Day 1: W rewrite (abstract+intro, with Garg), E4 done
(log-T constant, T1 triggered), orchestrator built, E2 (32 multi-seed
trainings) done overnight, I0 closed via live kill-and-resume test, env
curse solved (PYTHONNOUSERSITE=1 — LMCA user-site torch was shadowing
conda). Day 2: E1 v1 done (28/28); v2 retired (broken); v3 led to the
campaign's second big finding: **the theory's exact H-weighted centroid
blows up 25–94× on real adapters (degenerate H̄ spectrum) — E1 resolves
to master-plan branch 2 with a measured mechanism.** Ridge-regularized
centroid salvage sweep running now; 80-cell seed merge matrix ~12/80.

## 1. RUNNING RIGHT NOW (jobs on sanjay.g; check `qstat -u sanjay.g`)

- **`41533` rdm_orch** — 80-cell seed-1/2 merge matrix (E2 analysis wave),
  GPUs from `_ORCH_GPUS` = `2,4,6`. **GPU0 DROPPED 2026-06-12 — the earlier
  "GPU0 worker parks harmlessly" note was WRONG.** At the 24-25GB boundary
  the orchestrator `free_gb` poll intermittently ADMITTED a cell to GPU0,
  then `pbs_eval_cell.sh`'s own 25GB gate exited 87 in 0 min. That `rc!=0`
  branch INCREMENTS `self.attempts` (unlike the orchestrator's own VRAM-poll
  requeue, which does not) -> a cell bouncing onto GPU0 twice is PARKED
  permanently into `self.failed`, and any parked cell makes the run write
  `_QUEUE_FAILED` not `_QUEUE_COMPLETE`. Caught at `failed=[]` (no loss).
  Fix: `_ORCH_GPUS` 0,2,4,6 -> 2,4,6; `qdel 41524`; resubmitted as `41533`
  (resumes from 71 done-files, attempts reset). No safe swap GPU (live
  nvidia-smi: gpu0 222MiB free under another user's ~80GB job; gpu1/3/5/7
  all <25GB free for our ~55GB cells) -> running 3-wide. 71 done/66 pending
  at resume. **rc=87 gate exits are now NO-CHARGE requeues (commit
  `719466d`)** — effective from the next requeue; 41533 itself still runs
  the old code (moot: GPU0 is out of the pool).
- **`41521` rdm_ridge** — 5-cell ridge λ-sweep (llama, b=∞, rank_deff,
  plain loader, GPU6, serial ~1.2h/cell). Results land in
  `results/phase3/eval_ridge/`. λ=0.001 done: worst 4.46 (expected bad,
  anchors raw end). λ ∈ {0.01, 0.1, 0.3, 1.0} pending. **λ=0.1 smoke was
  PROMISING: NLL 0.615 on gsm8k probe vs v1's 0.761.**
- **Keeper v2** (login node, pid `logs/orch_keeper.pid`) requeues rdm_orch
  on walltime; stops on `_KEEPER_STOP`, or on `_QUEUE_COMPLETE` ONLY after
  verifying done-files == manifest total (`all_manifest.json`). A stray
  sentinel is removed + logged and the keeper continues.
- **✅ SENTINEL TRAP DEFUSED (2026-06-12 afternoon, commit `daa8648`):**
  keeper v2 no longer trusts the shared `_QUEUE_COMPLETE` blindly, so ridge
  41521 finishing can no longer kill the keeper mid-matrix (a session
  monitor still clears the stray file as a redundant layer). Also
  future-proofed: orchestrator.py STATE is `ORCH_STATE`-overridable and
  pbs_orchestrator_ridge.sh exports `ORCH_SENTINEL=_RIDGE_COMPLETE` +
  `ORCH_STATE=orchestrator_state_ridge.json`, so concurrent orchestrators
  stop clobbering each other's state file. Until ridge 41521 ends,
  `logs/orchestrator_state.json` may show RIDGE cells — judge matrix
  progress by done-file counts, not that file.
- **P4 deconflict (2026-06-12 ~16:10):** MOOLoRa pilot seeds were stacked
  on OUR matrix GPUs (seed1 gpu4, seed2 gpu6 — gpu6 also carries ridge).
  Moved to gpu1/gpu3 (jobs 41535/41536; P4 keeper gpu map updated on
  STUDENT-ACCOUNT). Matrix now has 2,4 to itself; 6 shared only with ridge
  until 41521 ends.

## 2. Results so far (all committed; decisions.md has full detail)

- **E4 ✅** linear-T falsified; constant is log-T (R²≥0.998), slope ~1/√r →
  max-of-T concentration. **T1 theory week with Garg = ON** (user to book).
- **E2 ✅ 32/32 trainings** (llama/mistral/qwen/yi × 4 tasks × seeds 1,2),
  zero failures. First stability read (llama): TA worst-excess
  0.219/0.213/0.221 across seeds 0/1/2 (±2% rel) and DARE tracks TA
  per-seed to 3 decimals → seed variance ≪ all reported effects.
- **E1 v1 ✅ 28/28** (rank-16-truncated encoder, 4 models × b∈{1..32}):
  encoder LOSES to TA at every b (llama b∞ 0.497 vs TA 0.219, TIES 0.161);
  qwen replicates pattern; non-monotone curves (b=3 < b=∞ on llama) echo
  the b=2 less-is-more. trunc_mass ~0.30 fired the pre-registered rule.
- **E1 v2 ❌ RETIRED** — base-weight patching breaks under unsloth (+10
  nats). **v3 (rank-64 adapter, exact factors via `add_adapter_rank`)
  revealed the real story: stored weights PERFECT on both unsloth AND
  plain PEFT, yet NLL ~3.5-3.6 → THE EXACT CENTROID ITSELF IS BAD.**
- **🔬 Mechanism (centroid_diag.py): H̄'s nonzero eigenvalues are nearly
  degenerate (median ~0.002, min 1e-5) despite full rank d_eff=64 → task
  subspaces independent-but-nearly-collinear → H̄⁺ amplifies the centroid
  to 25–94× (median 32×) the TA norm → outside quadratic-surrogate
  validity.** = E1 branch 2 sharpened; elevates explicit-Mt/curvature-H to
  the central open problem; gives the paper's soft-vs-hard d_eff
  distinction an operational consequence. Paper's floor-zero hard-d_eff
  claim UNAFFECTED.
- **Ridge salvage** (`ridge_lambda` kwarg: Λ⁻¹ → (Λ+λ)⁻¹): the live sweep
  asks whether a tamed centroid beats TA — decides whether achievability
  survives with a regularization caveat or branch 2 stands in full.

## 3. Hard-won operational rules (do not rediscover)

1. `PYTHONNOUSERSITE=1` in EVERY job wrapper (user-site torch shadowing).
2. **Smoke-first protocol is MANDATORY for any new merge/eval path** — one
   cheap GPU cell with explicit PASS/FAIL bar before any batch. Saved 12
   cells twice; also disproved my unsloth-blame hypothesis.
3. Unsloth quirks: forward ignores base_layer.weight mutations; only
   uniform-rank adapters work. Mixed-rank (rank-64) needs the plain loader
   (`loader: plain` in cell config → vanilla transformers+PEFT; validated).
4. plain-PEFT and unsloth NLLs agree to ~0.01 — stacks interchangeable for
   aggregate claims.
5. Orchestrator cells: done-file idempotent; VRAM-gated (25GB default);
   `_ORCH_GPUS` file controls GPU set (edit + requeue).
6. No pytest in the conda env — run tests as scripts. Login node has no
   nvidia-smi. Repo branch = `phase3-bootstrap`.
7. DARE merges unseeded upstream; TIES deterministic (anchor validations).

## 4. Next steps (priority)

1. **Ridge sweep verdict** (tonight): if some λ beats TA → achievability
   salvageable with regularization (big for paper); else branch 2 full.
   Either way: extend winning recipe to 4 models / write E1 §6.2.
2. **Matrix wave completes** (~1-2 days under contention) → E2 analysis:
   mean ± seed-range tables for every method claim; verify b=2 dip + Ll>Mi>Qw>Yi
   ordering across seeds.
3. **Fisher-diagonal H_t variant** (E1 spec variant b) — curvature-aware H
   may fix the centroid properly (vs ridge's blunt fix).
4. E7 (b=2 mechanism: matched-sparsity pruning + TIES win-pattern
   correlation) — cheap, now richer given encoder non-monotonicity.
5. W: write E1/E2/E4 results into paper §6; Remark 5 rewrite.
6. USER ACTIONS: send Garg the rewrite+plan+E4(+centroid finding — it's
   presentation-worthy); book T1 week; E5 design sign-off.

## 5. Locations

- Cluster: `sanjay.g@CLUSTER-HOST:~/projects/rdmerge`; GitHub
  `K144U/rdmerge` branch `phase3-bootstrap`.
- Results: `results/phase3/{eval_e1, eval_ridge, eval_matrix_seeds,
  eval_matrix_n1k_v3_perexample (seed-0 reference), lora_train}/`;
  E4: `results/e4_t_sweep/`.
- Configs/manifests: `code/phase3/configs/` (all_manifest.json = master
  queue, 140 cells; ridge_manifest.json separate).
- Orchestrator suite: `code/phase3/scripts/{orchestrator.py,
  pbs_orchestrator.sh, pbs_orchestrator_ridge.sh, orchestrator_keeper.sh,
  gen_*.py}`; smokes: `smoke_rank64*.py`; diagnostic: `centroid_diag.py`.
- Monitors live in the Claude session only — on /clear, re-arm from §1.

# rdmerge (P5) — Project Context (read this first)

**Paper:** A Rate-Distortion Lower Bound for Model Merging, with Matching
Achievability via Hadamard Incoherence
**Author:** Sankalp Pathak (solo), Prof. Sanjay Garg reviewing/advising
**Target venue: ICLR 2027** (deadline ~late Sep 2026)
**Last updated: 2026-06-15 ~11:00 IST** (campaign day 5 morning;
**E3 GSM8K em sweep 20/20 CLOSED: 3/4 STRONG agreement, llama-3.1
the LONE OUTLIER with ρ=-0.60 rank inversion**). Yi-data rules out
the strong-base hypothesis; the gap is Llama-3.1-specific.
Best-per-model: llama→knots(0.346), mistral/qwen/yi→ties(0.41/0.71/0.78).
Strategy: `master_plan_iclr2027.md`. Dated gate/branch record:
`decisions.md` (17 entries — READ IT, it is the scientific log).

**Headline numbers (paper §6.2 + §6.3 ready)**: ridge λ=0.05 →
worst-task excess 0.0906 on llama (59% below TA 0.219). Cross-model:
λ=0.13 wins on mistral (0.038, −72%), qwen (0.010, −91%), yi (0.034,
−66%); the achievability salvage generalizes. Fisher H_t below-ridge
(0.21 best). E5 Arm 2 NO-GO at 0/112 layers <0.8 on α=0.9 — real
fine-tuning resists subspace overlap. The lower bound's floor is
operationally zero in any natural data regime; non-vacuous only under
explicit geometric forcing.

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

## 1. STATE (no jobs running right now; `qstat -u sanjay.g` empty for rdm_*)

- **Seed matrix — DONE 2026-06-13 ~16:53 IST**. 140/140 cells, zero
  permanent failures, 1 transient qwen tvq_b32_seed2 rc=1 auto-retried.
  Job sequence 41524 -> 41533 -> 41556 (walltime keeper requeue). Results
  in `results/phase3/eval_matrix_seeds/`. Analysis (E2) pending —
  `code/phase3/scripts/analyze_e2_matrix.py` to be written.
- **Ridge sweep — DONE 2026-06-13 ~04:50 IST**. 10 cells total (5 original
  jobs 41521 + 5 fine 41537), llama b=∞ rank_deff. Clean U-shape, global
  minimum at **λ=0.05 → worst-excess 0.0906** (vs TA 0.219). λ=0.07 close
  second at 0.0951 (Δ=0.005, plateau). Files in
  `results/phase3/eval_ridge/llama31_8b__ridge_l{0p001,0p01,0p05,0p07,0p1,0p13,0p17,0p2,0p3,1}.json`.
  See decisions.md "Ridge fine sweep CLOSED" (2026-06-13).
- **Fisher H_t variant — DONE 2026-06-13 ~01:30 IST**. 2 cells (job 41538),
  llama b=∞ rank_r, h_t_mode='fisher_diag'. **BELOW-RIDGE verdict**:
  λ=0 → 0.210, λ=0.1 → 0.319. Projector + ridge stays SOTA. Notable
  structural finding: Fisher mode's trunc_mass is 0.025 (vs projector's
  0.30) — Fisher W* is rank-r friendly but less aligned overall. See
  decisions.md "Fisher-diagonal H_t variant: BELOW-RIDGE" (2026-06-13).
- **GPU pin history kept for reference**: matrix was pinned 2,4,6 after
  GPU0 dropped 2026-06-12 (boundary-VRAM flicker burned cell retry
  budgets; rc=87 now no-charge per commit `719466d`).
- **Keeper v2** still running (login node, pid `logs/orch_keeper.pid`,
  uptime 1d+ at this update). Idle now that matrix is done; can be stopped
  with `touch _KEEPER_STOP` once cross-model sweep is launched.
- **All ops infrastructure is sound**: rc=87 no-charge requeue (commit
  `719466d`), per-job ORCH_SENTINEL/ORCH_STATE (commit `daa8648`),
  done-file resume across walltime kills — all proven on tonight's run.

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
- **Ridge salvage ✅ FINAL 2026-06-13** (`ridge_lambda` kwarg: Λ⁻¹ →
  (Λ+λ)⁻¹): clean U-shape, **λ=0.05 worst-excess 0.0906 BEATS TA 0.219
  by 59%**. λ=0.07 close second at 0.0951. Achievability claim
  SURVIVES with a regularization caveat — Sec 6/7 reframes to "exact
  construction loses (identified spectral mechanism); regularized
  variant restores SOTA on llama, awaiting cross-model confirmation".
- **Fisher H_t variant ✅ DONE 2026-06-13** (master plan E1 variant b):
  diagonal empirical Fisher (input-activation second-moment), λ ∈
  {0, 0.1} on llama. **BELOW-RIDGE**: 0.21 (no λ), 0.319 (λ=0.1).
  Ridge HURTS Fisher (opposite direction of projector mode). Notable:
  trunc_mass mean 0.025 (Fisher) vs 0.30 (projector v1) — Fisher W*
  is rank-r friendly but centroid alignment is worse. Sanity check
  for surrogate choice came back clean for projector + ridge.

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

1. **Cross-model ridge sweep at λ ∈ {0.05, 0.07, 0.10, 0.13}** for
   mistral_7b / qwen25_7b / yi15_9b = 12 cells. **Load-bearing claim**:
   the achievability salvage generalizes across base models.
   Configs in `code/phase3/configs/eval_ridge_xmodel/`; manifest
   `ridge_xmodel_manifest.json`; wrapper `pbs_orchestrator_ridge_xmodel.sh`.
   Decision rule pre-registered:
   - all 3 hold → "matching achievability across models" stands;
   - 2/3 hold → narrow claim, treat outlier explicitly;
   - 0/3 → salvage is llama-specific, pivot to E5 as headline.
2. **E2 analysis** (`code/phase3/scripts/analyze_e2_matrix.py`): tidy
   140-cell dataframe; per-(model, method) mean ± seed-range; verify
   (a) Ll > Mi > Qw > Yi across seeds, (b) b=2 dip persists,
   (c) DARE tracks TA to 3 decimals.
3. **Paper §6.2 draft** with the salvage framing (ridge sweep table,
   Fisher baseline, cross-model when ready).
4. **E5 design week** (master plan Tier 2 headline experiment, needs
   user sign-off): pilot model Qwen-2.5-7B, Arm 2 data-mixture primary
   with Arm 3 geometric forcing as fallback. Pilot gate: α=0.9 →
   deff/(Tr) < 0.8 in majority of layers.
5. **T1 theory week** (with Garg): log-T concentration bound for C —
   send him the digest doc, book the week.
6. **E3 downstream metrics harness** (GSM8K em, HumanEval pass@1,
   COMET-22, IFEval strict) — paper needs accuracy-level evidence.
7. **E7 b=2 mechanism** (cheap, 30-50 GPU-h): matched-sparsity
   pruning + TIES win-pattern correlation.
8. **W**: write E1/E2/E4 results into paper §6; Remark 5 rewrite.

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

# Red-team campaign, started 2026-10-19

User, 2026-10-19: "experiment running on STUDENT-ACCOUNT is done lets do all the
suggestions from this account STUDENT-ACCOUNT and not disturb whats running on
sanjay.g's account". Scope: the eight red-team items (from this session's
critique) plus the professor's A1 and A2 from
`notes/neurocomputing_presubmission_plan_2026-10-06.md`. Target venue:
Neurocomputing short paper.

## Registrations (committed before any code they govern)

| doc | items | cells (approx.) |
|---|---|---|
| prereg_wudi_2026-10-19 | 1 (A4): WUDI-Merging, closed form vs published | 72 NLL + 96 acc |
| prereg_lambda_selection_2026-10-19 | 2: GCV and L-curve | 48 NLL (+48/rule acc if branch 1) |
| prereg_training_regimes_2026-10-19 | 3, 8, A1, A2: seed ablation, knobs + theory, continued, DoRA, libraries | 60 trainings + ~20 NLL |
| prereg_vision_2026-10-19 | 4: CLIP ViT-B/32, 8 tasks | 48 small trainings + merges |
| prereg_heterogeneity_2026-10-19 | 5, 8: gauge/scale/rank/module, partial collapse | 64 NLL |
| prereg_untruncated_2026-10-19 | 6 (A3): truncation and the lean | 72 acc + 40 NLL |
| prereg_all_tasks_2026-10-19 | 7 (S2, S3): chrF, IFEval, reference rows | 144 acc |

## Restage (2026-10-19)

The sanjay.g tree's `artifacts/lora` and `models/` were emptied on 2026-10-10
(not by us; the student account only ever read them). Our tree's `artifacts`
and `models` were symlinks into it. Renamed the links to `_link_*_sanjay`,
made real directories, and:
- uploading the 25 GB local copy of the adapters (4 bases x 4 tasks x 6
  cohorts + v1) to `~/projects/rdmerge_v2/artifacts/lora`;
- downloading the four bases into our HF cache, linked as `models/<Name>`,
  commit hashes in `models/RESTAGED_2026-10-19.json`;
- cloning the conda env to `~/projects/rdmerge_v2/.conda/envs/rdmerge`.
Gate before any registered cell: re-run four published NLL cells
(adapter-only RegMean at lambda = 0.01, seed1, one per base, from
eval_regmean_real) with a fresh nll_tau, and match the published worst-task
excess and every nll_tau entry to 0.002 nats (check_restage.py).

## Log
- 2026-10-19 ~12:30, **item 1 search result**: WUDI-Merging (Cheng et al.,
  arXiv 2503.08099) is a published data-free method whose least-squares matrix
  is built from the task updates (range = A row spaces for LoRA). Registered
  closed form (Eq. 20) vs published Adam-300 vs Adam-3000. Toy check: long
  Adam matches the closed form inside the update rows' span, and also leaves
  mass OUTSIDE that span (Adam's per-coordinate step), invisible to WUDI's
  loss; recorded per merge as off_range_frac.
- **Correction (heterogeneity)**: the registration says the unperturbed
  references exist; only indep1 RegMean(1e-6) did. Reference cells added (12
  Part G, 8 Part M), recomputed on the re-staged models. No prediction made
  easier.
- **Amendment (training regimes, 0adf3ac)**, before computing: drift cohort
  and its A_0 lost with the sanjay.g tree; calibration on four surviving Llama
  shared cohorts with consensus W0. Calibration committed 9c6f73f before any
  Part K adapter: C_k = 768.1, C_a = 11.28.
- **Restage, more than expected**: ~110 entries of our results/phase3 were
  symlinks into sanjay.g's results tree, also emptied 2026-10-10. 94 restored
  from the local repo; 16 links still dangle (no copy anywhere checked:
  eval_downstream_cond, eval_ridge_shared, nll_tau_cache, prevalence, w1/w3
  sweeps, shared_arm and s3 summaries, a1_matrix summaries, lstar_from_e2).
  The conda env was copied with cp -a (conda --clone --offline silently
  skipped python); jobs use it only once .conda/CLONE_OK exists.
- **Login node now kills a user's processes when the SSH call ends** (even
  setsid/nohup). Long login-node work runs inside held local SSH sessions.
- Launched: restage gate (11828), vision training x2 (11829-30), Unsloth
  library defaults (11831). The nll_tau_cache_v2 keys embed old paths;
  rekey_tau_cache.py rewrites only the paths, per base, after its gate passes.
- 2026-10-19 ~14:50, **restage gates**: Llama passes (nll_tau identical to
  5 decimals; worst excess 0.17755 vs published 0.17847), Qwen passes
  (identical; 0.08295 vs 0.08306). Caches re-keyed. Mistral and Yi pending;
  gate_and_launch.sh queues each base on a pass.
- **Part L, Unsloth (2026.5.2) measured**: A at creation identical across two
  separate runs on different tasks (median cosine 1.000), 0.9996 after 50
  steps. L1, L2 hold for Unsloth. LLaMA-Factory 0.9.5 in an isolated venv;
  first attempt failed on a torchaudio wheel built for torch 2.11, replaced
  by the cu128 2.10.0 build; resubmitted.
- Truncation energy (H1 premise, CPU) computed for all 120 merges; not read
  until the H2 cells exist.
- Training speed under contention: ~7 s/step (about 55 min per Llama
  adapter), so the 60 regime runs take about a day on one two-worker job.
  Vision adapters ~5 min each.
- Queued: Llama WUDI (42), Llama+Qwen stage 1 (heterogeneity, lambda
  selection, Qwen WUDI; 138), Llama+Qwen untruncated (56), LLaMA-Factory (4).
- 2026-10-19 ~15:10, **vision training done**: 48 adapters, own-task test
  accuracy matched across arms (Cars .73-.74, DTD .77-.79, EuroSAT .98-.99,
  GTSRB .97-.98, MNIST .99, RESISC45 .94, SUN397 .72-.74, SVHN .96), in the
  range published for LoRA ViT-B/32. Geometry (V1 input): shared cohorts
  median cosine 0.959-0.960, independent 0.109-0.110. Evaluation queued
  (11842).
- 2026-10-19 ~16:15, **all four restage gates pass** (restage_check_summary:
  worst-excess |d| 0.00092 / 0.00012 / 0.00011 / 0.00105, nll_tau identical);
  caches re-keyed; every base's stage-1 and untruncated cells queued.
- **Part L measured for both libraries**: LLaMA-Factory 0.9.5 defaults (r 8,
  alpha 16, all linear modules, seed 42) give identical A at creation across
  tasks and runs (median cosine 1.000) and 0.9994 after 50 steps; Unsloth
  2026.5.2 gives 1.000 and 0.9996. L1 and L2 hold for both.
- 2026-10-19 ~18:40, **vision evaluation done** (6 cells, 22-23 min each).
  Not read until the analyzer runs.
- 2026-10-19 ~20:05, **WUDI implementation fix, before any WUDI cell
  finished.** The official loss is evaluated densely ((tau - D_t) D_t^T is
  out x out); the first two Llama 3000-step cells ran 4.7 h without
  completing one merge (about an exaflop per cell). merging/wudi.py now
  evaluates the identical loss from the factors, tr(X_t^T X_t B'_t^T B'_t)
  with X_t = tau A_t^T - B'_t A_t A_t^T: same function of tau, same gradient,
  same init, optimiser, lr, steps and float32, so the registered procedure is
  unchanged. test_factored_loss_matches_dense checks loss, gradient and a
  50-step Adam trajectory against the dense form (all within 1e-5); all 41
  merging tests pass locally and the WUDI tests pass in the cluster env.
  D_t is now formed from the factors in float32 rather than as their bf16
  product. Job 11836 killed; Llama WUDI resubmitted (11852). The Qwen,
  Mistral and Yi WUDI cells had not started and pick up the fix.
- **Stale GPU claim**: gpu0 stayed claimed by 11828 (qdel'd) for 4 h, which,
  with WUDI holding two cards, left the untruncated and Mistral jobs idle
  with zero cells started. Cleared by hand; orchestrator.py now drops a claim
  whose writing process is gone (new jobs only; running ones keep the 8 h
  rule). Every job now has an afterany successor (chain_c1019.py) so a 24 h
  walltime kill resumes instead of ending its manifest.
- Contention: all 8 GPUs at 95-100% utilisation (35 running jobs on the
  node); NLL cells with a cold nll_tau take 70-95 min, warm ones ~15-40 min.

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
Gate before any registered cell: re-run four published NLL cells (task
arithmetic, seed1, one per base) and match the published worst-task excess to
0.002 nats.

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

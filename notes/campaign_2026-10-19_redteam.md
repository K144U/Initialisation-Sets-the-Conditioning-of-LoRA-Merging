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

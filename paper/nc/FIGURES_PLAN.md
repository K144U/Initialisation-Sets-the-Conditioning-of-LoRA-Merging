# Figure rebuild plan (Neurocomputing short paper)

Status 2026-10-06: BUILT, all six figures, entirely on the local machine (no cluster compute; the indep1-3 adapters were copied read-only into artifacts/lora). Kept below as the record of what each panel reads. The current five
figures in `nc/figures/` work but are thin (mostly two clusters of dots). The
user asked for "much better and much more informative figures". Rebuild in
`nc/make_figures.py`; every number must come from a result file.

## Target set (6 figures, ~18 panels)

1. **The collapse** (2x2)
   - (a) drift ||A_t - A_0|| / ||A_0|| over training, shared vs indep arms
     (both reach 0.13). Data: `results/phase3/drift_summary.json`
     (`arms.drift_shared.drift_trace`, `drift_indep.drift_trace`; 5 points). LOCAL.
   - (b) distribution of ALL principal cosines (pairs x 16 angles x 8 layers),
     shared vs indep, 4 bases. NEEDS NEW STATS (see below).
   - (c) singular-value spectrum of stacked bases M=[V1..V4] (64 values):
     shared = 16 at ~2 (=sqrt T) + 48 near 0; indep = 64 near 1. NEW STATS.
   - (d) T/r sweep + PiSSA (current fig1b). `campaign_2026-09/new_cohorts_summary.json`. LOCAL.
2. **Theory** (2x2)
   - (a) exact eigenvalues of Hbar vs first-order prediction (r near 1, (T-1)r
     equal to eig of Sigma_X), shared Llama; indep eigenvalues (~1/T) for
     contrast. NEW STATS.
   - (b, c, d) keep current dial kappa, dial amplification (+ rival s^-2), and
     the 21-case ratio panel. `campaign_2026-09/drift_theory.json`. LOCAL.
3. **Prevalence**: scatter median cosine (x) vs rel_A_diff (y) for the 45
   public cohorts, marginal histogram on x, sqrt(2) line, thresholds; our own
   cohorts (A_rel_distance 0.17 shared / 1.41 indep, `subspace_geometry_*.json`)
   and KnOTS releases (`campaign_2026-09/prevalence_part_b_prevalence_summary.json`)
   as reference markers. `nc/data/prevalence_summary.json` (fields
   `median_principal_cosine`, `rel_A_diff`, `status`). LOCAL.
4. **Mechanism**: amplification (log x) vs tail fraction (y) for every merged
   solution, shared (filled) vs indep (open), 4 bases. Data:
   `results/phase3/merged_solution_norms.json` (keys `seed1|<base>`,
   `indep1|<base>`; methods task_arithmetic, ties, dare, tvq_b2, knots,
   regmean, regmean_l0p13, rd_encoder_l0, rd_encoder_l0p05, mean, tau_H; each
   {amp, tail}). LOCAL. Uses the A^T A Gram (see app_audit_trail).
5. **Ridge** (2x4): top row worst-task NLL excess vs lambda per base, shared vs
   indep, adapter-only RegMean with the A^T A Gram; bottom row Gram condition
   number vs lambda with RegMean's alpha=0.9 star. NLL cells are ON THE CLUSTER:
   `jiit-student:~/projects/rdmerge_v2/results/phase3/eval_regmean_agram/`
   (56 files `<base>__rm_l<tag>__<seed1|indep1>.json`, key `worst_task_excess`,
   tags 1em6 0p01 0p03 0p05 0p13 0p30 1p00). scp read-only. Conditioning:
   `results/phase3/regmean_published_default.json`. LOCAL.
6. **Accuracy**: current fig4 scatter plus error bars (sd over 3 cohorts; the
   per-cohort lists are in `heur_tost_summary.json` per_base and
   `regmean_real_summary.json` downstream.table) and arrows from alpha=0.9 to
   lambda=0.01 per base. LOCAL.

## NEW STATS needed for 1b, 1c, 2a

Independent-cohort adapters are NOT local (local `artifacts/lora/<base>/<task>/`
has only seed1-3 and v1). Run on the cluster as STUDENT-ACCOUNT (read sanjay.g's
`artifacts/lora`, which has indep1-3), PYTHONNOUSERSITE=1, light CPU only.
Write a script that, for each base and cohort in {seed1, indep1} (optionally
seed2-3, indep2-3), over the same 8 sampled layers as
`measure_subspace_geometry.py` (sorted lora_A keys, step len//8), saves:
- all principal cosines per pair (svdvals of V_i^T V_j)
- svdvals of M = cat(V_t) (64 values)
- exact eigenvalues of Hbar = (1/T) sum P_t in the range of M
- predicted eigenvalues: r values near 1 plus eig of Sigma_X = Y Y^T / T with
  W0 = consensus_W0 (reuse `exact`, `predicted`, `consensus_W0`, `orth_rows`
  from `code/phase3/scripts/drift_theory.py`)
Output one small JSON to `paper/nc/data/geometry_stats.json`, scp back.

## Then
Update captions and the figure references in s3/s4/s5/s6, rebuild
(`bash paper/nc/build.sh`), check every figure in greyscale
(`figures/preview/*_grey.png`), run `python paper/nc/check.py`.

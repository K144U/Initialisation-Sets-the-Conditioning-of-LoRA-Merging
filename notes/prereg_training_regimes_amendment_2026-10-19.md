# Amendment to prereg_training_regimes_2026-10-19, Part K calibration

Committed 2026-10-19, after the registration and before C_k and C_a are
computed or any Part K adapter is trained.

## What changes, and why

The registration calibrates C_k = median(kappa t^2) and C_a = median(amp t) on
seven points: Llama seed1-3 and the four drift-cohort checkpoints. Writing the
calibration code showed two problems:

1. The drift cohort's adapters and saved A_0 were on the sanjay.g tree, which
   was emptied on 2026-10-10; only their summary values survive in
   drift_theory.json.
2. Those stored tilt norms use the saved A_0 as the reference subspace W0,
   while seed1-3 have no saved A_0 and can only use the consensus W0 (the
   top-r left singular vectors of the stacked row-space bases). Mixing the two
   definitions inside one constant would be meaningless.

So:

- **t is defined with the consensus W0 for every cohort**, calibration and test
  alike (the Part K cohorts save A_0, but it is not used for t).
- **The calibration set is the four Llama T = 4, r = 16 shared cohorts that
  still exist**: seed1, seed2, seed3, and the translation-fixed cohort of
  prereg_new_cohorts Part T (seed1 with flores replaced by flores/seed1fx).
- Thresholds (factor 2 for kappa on >= 6 of 8, factor 1.25 for amp on >= 7 of
  8, rho <= -0.8, the 3x escape margin) are unchanged.

Nothing measured on a Part K cohort informs this change: none exists yet.

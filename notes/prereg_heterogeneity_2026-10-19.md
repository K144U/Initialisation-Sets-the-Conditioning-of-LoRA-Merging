# Pre-registration: what breaks a solver on a separated cohort, and partial collapse

Committed 2026-10-19, before the perturbation script
(`code/phase3/scripts/make_perturbed_cohorts.py`), the generator
`gen_heterogeneity.py` and the analyzer `analyze_heterogeneity.py`.

## Why

Red-team items 5 and 8. On public cohorts unregularised adapter-only RegMean
failed in both regimes (prereg_public_merge, branch 3), which undercuts the
prevalence claim: a regime that does not predict failure in the wild needs an
account of what does. Public cohorts differ from ours in ways our cohorts hold
fixed. One of them we can name in advance: adapter-only RegMean is not
invariant to the LoRA gauge. Rescaling A_t by c and B_t by 1/c leaves the update
D_t unchanged but multiplies that task's Gram A_t^T A_t by c^2, so publishers
with different learning rates, alphas or initialisations hand the solver very
different weights for identical updates. Separately, the audit summarises a
cohort by the MEDIAN pairwise cosine, which a single collapsed pair cannot move.

## Part G. Controlled heterogeneity on separated cohorts

Bases llama31_8b and qwen25_7b, cohorts indep1 indep2 indep3. Each perturbation
is applied to the cohort's adapter files (new copies; originals untouched):

- **P_gauge**: gsm8k adapter A <- 4A, B <- B/4 (update unchanged exactly).
- **P_scale**: gsm8k adapter B <- 4B (update x 4; a 4x larger lora_alpha).
- **P_canon**: every adapter re-factorised into balanced SVD form,
  A = S^{1/2} V^T, B = U S^{1/2} (updates unchanged exactly).
- **P_rank**: P_canon, then the gsm8k and magicoder adapters truncated to
  rank 4.
- **P_module**: the gsm8k adapter's k_proj and v_proj updates removed (a
  publisher who targeted only q and o).

Cells: adapter-only RegMean at lambda = 1e-6 for all five (30); task
arithmetic for P_scale, P_rank and P_module (18), since for P_gauge and P_canon
the updates are identical and task arithmetic is unchanged by construction.
48 NLL cells. Unperturbed references exist.

Per base, d = mean over cohorts of (perturbed - unperturbed) excess, gate
max(0.005, 2 SE).

- **G1 (the gauge breaks the solver).** P_gauge: RegMean(1e-6) excess >= 2 x
  unperturbed and d > gate on both bases.
- **G2 (canonical gauge is harmless).** P_canon: RegMean(1e-6) excess within a
  factor 1.5 of unperturbed on both bases.
- **G3 (the other heterogeneities are not solver-specific).** For P_scale,
  P_rank and P_module, the ratio perturbed/unperturbed for RegMean(1e-6) is at
  most 2 x the same ratio for task arithmetic on both bases.
- **G4 (the public link, CPU).** Over the public cohorts merged in
  prereg_public_merge, the within-cohort A-norm spread (max/min of ||A_t||_F,
  median over layers) is at least 2 x larger, in median, among cohorts where
  unregularised adapter-only RegMean failed (registered statistic F >= 2) than
  among those where it did not.

Branches: G1 and G4 hold -> gauge mismatch is the named cause of the public
failures that are not regime-specific; the audit adds an A-norm-spread check
and the paper recommends canonicalising the gauge before any adapter-built
solve (G2 says it is harmless). G1 holds, G4 fails -> a real sensitivity that
does not explain the public data. G1 fails -> the gauge is not the cause; the
public failures stay unexplained and the paper says so.

## Part M. Partial collapse and the audit's statistic

All four bases, n = 1, built from existing adapters:

- **M1**: gsm8k and magicoder from seed1, alpaca and flores from indep1 (one
  collapsed pair of six).
- **M2**: gsm8k, magicoder and alpaca from seed1, flores from indep1 (three
  collapsed pairs of six).

Cells: RegMean(1e-6) and task arithmetic for both (16 NLL cells); geometry on
CPU.

- **M-a (the median misses partial collapse).** For M1 the median cosine is
  < 0.5 (the audit says "separated"), yet RegMean(1e-6) excess >= 2 x indep1's
  on >= 3 of 4 bases.
- **M-b.** The maximum pairwise cosine is >= 0.9 for M1 and M2 on all bases.
- **M-c.** kappa(Hbar) of M1 >= 10 x indep1's on >= 3 of 4 bases.

Branch: M-a holds -> the audit's statistic changes from the median to the
maximum pairwise cosine (or the count of collapsed pairs), Algorithm 1 and its
cutoffs are restated, and the 45-cohort prevalence is re-read under the new
statistic. M-a fails -> one collapsed pair does not hurt the solver and the
median stands.

## Binding constraints

Perturbation factors (4, rank 4, the module pair) fixed here. Perturbed cohorts
whose updates are exactly unchanged reuse the unperturbed nll_tau; the others
recompute it. No partial reads. Compute on STUDENT-ACCOUNT only.

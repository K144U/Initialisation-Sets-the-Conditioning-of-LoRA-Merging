# Solve precision check (professor B1): result and the rebuild it triggers

Written 2026-10-24 after the check finished and before any rebuild cell was
generated. The registration, `notes/check_solve_precision_2026-10-24.md`
(b82b81e), is not edited. Run 1 was void (it compared two different truncation
algorithms) and was fixed in 456afe2; this is run 2, the first valid run.

## Outcome: the second branch

`code/phase3/scripts/apply_precision_rule.py` applied the committed rule to
the four per-base outputs; the summary is
`results/phase3/campaign_2026-10/solve_precision.json`, the logs are in
`results/phase3/campaign_2026-10/solve_precision_logs/`.

| base | arm | RegMean 1e-6 rel_trunc | encoder l0 rel_trunc | amp ratio 64/32 (enc) |
|---|---|---|---|---|
| Llama | shared | 6.55e-3 | 5.21e-3 | 1.0000 |
| Llama | indep | 4.46e-3 | 2.66e-6 | |
| Mistral | shared | 6.51e-3 | 3.26e-3 | 1.0000 |
| Mistral | indep | 4.09e-3 | 3.46e-6 | |
| Qwen | shared | 5.86e-3 | 8.79e-3 | 0.9996 |
| Qwen | indep | 5.60e-3 | 2.26e-6 | |
| Yi | shared | 6.52e-3 | **1.09e-2** | 1.0007 |
| Yi | indep | 5.21e-3 | 2.41e-6 | |

Every amplification ratio is within 0.1% of 1, far inside [0.5, 2]. Fifteen of
the sixteen (base, arm, method) medians are below 1e-2. **Yi, shared arm,
encoder at lambda = 0 is at 1.09e-2 and fails.** The rule therefore lands in
its second branch: the affected merges are rebuilt in float64 and re-evaluated
in NLL on the cluster, same cells and same nll_tau, and the paper reports the
float64 values beside the float32 ones whatever they show. The threshold is
not moved: 1.09e-2 fails a rule of "below 1e-2".

## What "the affected merges" are

The failing case is one merge: the encoder (bits 32, ridge 0, rank_r,
projector) on the four Yi seed1 adapters. A search of `results/phase3` for
every cell with that method, those settings and those adapters, truncated and
not renormalised, finds one NLL cell:
`eval_ridge_cond/yi15_9b__rd_l0__seed1.json` (worst-task excess 0.249). It is
the cell the check took its configuration from.

## The rebuild

- **Registered rebuild (`rd_l0_f64`)**: the same cell configuration with
  `solve_dtype: float64` and `truncation: exact` added to the method kwargs.
  `merge_rd_encoder` then builds each task's right-singular basis exactly from
  its factors and solves in float64, which is the check's float64 reference,
  and realises the merge at rank r by an exact SVD of its factored form, which
  is the truncation the rule's metric used. Checked locally against
  `check_solve_precision.encoder64` + full SVD on six Yi modules: agreement
  3.6e-8, the float32 rounding of the stored factors. The defaults reproduce
  the production merge; the encoder unit tests pass.
- nll_tau comes from `nll_tau_cache_v2/yi15_9b__seed1.json`, whose base values
  equal the published cell's to every digit.

Two cells are added that the registration did not ask for, labelled as such
wherever they are reported:

- **`rd_l0_rerun`**: the production configuration unchanged, on the restaged
  models, so the float64 value is compared with a float32 value from the same
  setup and not only with the August cell.
- **`rd_l0_f32exact`**: the float32 solve with exact truncation. Production
  truncates with a randomised SVD at q = r and four power iterations, which on
  the shared arm is a median 30 to 38% (relative Frobenius) from the exact
  rank-r truncation of the same W (`prod_gap`). This cell separates the effect
  of precision (f64 vs f32exact) from that of the truncation algorithm
  (f32exact vs rerun).

## A finding from the local verification

On the most ill-conditioned modules the float32 encoder is not reproducible
across random states, because its per-task bases come from a randomised SVD
whose float32 rounding the lambda = 0 solve amplifies. Yi seed1, layer 0:
two float32 merges that differ only in the torch seed differ by 14% (k_proj)
and 49% (v_proj) before truncation, about as much as each differs from float64
(12% and 55%). On layer 31 both differences are 0.3%. The median over modules
is the 1.09e-2 above. This is a property of the unregularised float32 solve on
the collapsed operator and is reported with the rebuild.

## Rebuild results (2026-10-24, PBS 12608-12610)

Worst-task excess (nats), same nll_tau in all four (checked equal):

| merge | gsm8k | alpaca | magicoder | translation | worst | avg |
|---|---|---|---|---|---|---|
| float32, published cell (Aug) | 0.0858 | 0.2490 | 0.1249 | 0.1973 | 0.2490 | 0.1643 |
| float32, rerun (control) | 0.0847 | 0.2489 | 0.1223 | 0.1782 | 0.2489 | 0.1585 |
| **float64, exact truncation (registered)** | 0.0805 | 0.2448 | 0.1148 | 0.1452 | **0.2448** | 0.1463 |
| float32, exact truncation (control) | 0.2850 | 0.7201 | 0.8365 | 0.7846 | 0.8365 | 0.6565 |

The independent cohort's float32 cell is 0.1209, so shared over independent
is 2.06 in float32 and 2.02 in float64. The registered rebuild reproduces the
float32 merge's damage to 0.004 nats: rounding is not what makes this merge
fail.

The f32exact control does not do what it was designed to do. It changes the
truncation, and with it the random stream: production's randomised truncation
draws from the global generator after each layer, so removing it shifts the
random state of every later layer's randomised basis computation, which on
this cohort moves the float32 W by up to tens of percent on the worst modules
(local finding above). The control therefore differs from the rerun in two
things at once and cannot attribute its 0.837 to truncation. What it does show
is that a float32 realisation of this merge can land far from 0.249.

## Follow-up (unregistered, descriptive), written before its cells

To separate the two, the encoder gets an `rng_seed` kwarg (default None, no
change) that seeds the global generator at the start of the merge. Four cells
on the same template: float32 with production truncation at rng_seed 1 and 2,
and float32 with exact truncation at rng_seed 1 and 2. If the float32 values
scatter across seeds under both truncations, the float32 encoder at lambda 0
on this cohort is unstable and the paper says so, with the float64 value as
the reference; if exact truncation is high under every seed and production
truncation low, the difference is the truncation algorithm. Either way it is
reported as a follow-up that was not registered.

## Follow-up results (PBS 12611-12614)

| float32 encoder, Yi seed1, lambda 0 | stream | worst | avg |
|---|---|---|---|
| production truncation | default (Aug and rerun) | 0.2490 / 0.2489 | 0.1643 / 0.1585 |
| production truncation | rng_seed 1 | 0.2164 | 0.1404 |
| production truncation | rng_seed 2 | 0.2794 | 0.1621 |
| exact truncation | default | 0.8365 | 0.6565 |
| exact truncation | rng_seed 1 | 0.3887 | 0.2830 |
| exact truncation | rng_seed 2 | 0.2604 | 0.1625 |
| float64, exact truncation (deterministic) | none | 0.2448 | 0.1463 |

Reading: the float32 values scatter with the random stream under both
truncations, so the f32exact control's 0.837 is the unstable float32 solve,
not the truncation algorithm as such; exact truncation widens the scatter
(0.26 to 0.84) relative to production truncation (0.22 to 0.28), which is
consistent with it keeping the largest components of the rounding noise
exactly. The float64 value lies inside the production-truncation spread. In
all seven float32 realisations the shared cohort is worse than the
independent one (0.121, float32-stable to 2e-6), so the direction the paper
relies on does not depend on the realisation; the size of a single float32
value does, by about +-0.03 nats here under the truncation every merge uses.
For the paper: report the registered rebuild (0.245 vs 0.249; 0.121
independent), the instability with this spread, and that it was measured only
on the cohort the rule flagged. RegMean at 1e-6 is float32-stable on every
case (rel_trunc max 9.7e-3).

## Seed sweep on the other three bases (unregistered, descriptive), written before its cells

Asked for by the user on 2026-10-25. Every float32 value the paper gives for
the encoder at lambda 0 on a shared cohort is one draw of the default random
stream. On Llama, Mistral and Qwen the rule passed on the median (5.2e-3,
3.3e-3, 8.8e-3), but single modules there differ between float32 and float64
by up to 8.1e-2, 6.29e-1 and 3.20e-1 (Yi: 4.84e-1), so those values may
scatter as Yi's do.

Cells (6): for each of llama31_8b, mistral_7b and qwen25_7b, the published
cell `eval_ridge_cond/<base>__rd_l0__seed1.json` with `rng_seed` 1 and 2
added, float32 and production truncation, everything else unchanged; named
`<base>__rd_l0_rerun_s<k>__seed1` in `eval_precision_f64/`. Same nll_tau as
the published cells: `nll_tau_cache_v2/<base>__seed1.json`, whose nll_tau
equals each published cell's to every digit (checked 2026-10-25). Not
included: float64 rebuilds of these bases (the rule did not call for them)
and exact-truncation draws.

What is reported, whatever it shows: per base, the three draws (published,
s1, s2) of worst and average excess, their range, and whether every draw of
the shared cohort is worse than that base's independent float32 cell (Llama
0.0713, Mistral 0.1504, Qwen 0.0256, all float32-stable to below 1e-4 on
every module). Reading fixed now: if a base's range reaches its
shared-minus-independent gap, the paper says that on that base the direction
does not survive the random stream; otherwise it gives the range as the
float32 scatter on that base. The paper's sentence that the spread was
measured on the Yi cohort only is replaced by the four-base result.

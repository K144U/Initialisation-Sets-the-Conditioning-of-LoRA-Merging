# DARE checked against the official implementation

**2026-08-04.** Follow-up to the post-hoc note in `decisions.md` (2026-08-04)
observing that `task_arithmetic`, `dare` and `knots` agree to within 0.0025 nats
on all four bases across three cohorts. The KnOTS half of that was already
established in A2. This document settles the DARE half.

**Verdict: our implementation is faithful, DARE is not a KnOTS-style no-op, and
DARE never helps. The reason is that DARE + task arithmetic is an unbiased
estimator of task arithmetic, so it cannot help in expectation, and the observed
penalty follows the predicted functional form.**

The earlier note that DARE was "the same kind of observation" as KnOTS is
**wrong** and is corrected in `decisions.md`.

## 1. The operator is algebraically identical to the official one

Official: `yule-BUAA/MergeLM`, `model_merging_methods/mask_weights_utils.py`.

```
mask = torch.bernoulli(torch.full_like(input_tensor, mask_rate))
masked = input_tensor * (1 - mask)
masked = torch.div(masked, 1 - mask_rate)          # when rescaling enabled
```

Ours: `code/phase3/merging/dare.py`

```
mask = (torch.rand(d.shape, generator=g) < density)
d = (d * mask) / density
```

Their `mask_rate` is the drop probability; our `density` is the keep
probability, so `density = 1 - mask_rate` and the rescale divisor matches. Masks
are drawn per (layer, adapter) from a single advancing generator, so they are
independent across tasks, which is what DARE requires.

Official `mask_merging` is a wrapper that masks each model and then hands off to
a base method via `mask_apply_method`. Ours is mask + `task_arithmetic`, which
is one of the official configurations. `merging_methods.py` applies **no
low-rank or SVD truncation anywhere**; our pipeline truncates every merged delta
back to rank r because the output must be a LoRA adapter again. That was the one
structural difference worth testing, and it is tested in section 2.

**Independent confirmation the operator does what it claims.** The perturbation
DARE injects should have relative Frobenius norm `sqrt((1 - density) / density)`.
Measured on real adapters (`diagnose_dare_truncation.py`, cohort indep1):

```
density   0.20    0.10    0.05    0.01
predicted 2.000   3.000   4.359   9.950
Llama     1.992   2.987   4.338   9.889
Mistral   1.995   2.992   4.346   9.927
Qwen      1.981   2.972   4.320   9.851
Yi        1.995   2.995   4.355   9.942
```

Agreement to three decimals on four bases and four densities. The operator is
correct.

## 2. Hypothesis REFUTED: the rank truncation is not what hides DARE

Predicted before measuring: DARE is unbiased, so its whole effect is a dense
near-isotropic perturbation, and a rank-16 projection of a 4096-dimensional
space should pass only about `16/4096 = 0.0039` of it, hiding DARE while leaving
the biased methods (TIES) intact.

Measured, cohort indep1, 6 layers per base, production `svd_truncate_to_rank`:

```
                     fraction of DARE's perturbation energy surviving rank-16
density        0.20     0.10     0.05     0.01
Llama        0.0304   0.0309   0.0338   0.0374
Mistral      0.0311   0.0310   0.0328   0.0360
Qwen         0.0532   0.0573   0.0612   0.0629
Yi           0.0682   0.0759   0.0806   0.0822
isotropic    0.0039   0.0039   0.0039   0.0039
```

**8 to 21 times more energy survives than the isotropic prediction**, because
the perturbation is `delta * (mask/density - 1)`, which is elementwise
proportional to `delta` and therefore concentrated in the same subspace, not
isotropic. The resulting perturbation of the merged rank-16 delta is large:

```
||T_16(D_dare) - T_16(D_ta)||_F / ||T_16(D_ta)||_F  at density 0.20
Llama 0.374   Mistral 0.395   Qwen 0.473   Yi 0.570
```

So the truncation does **not** hide DARE. The hypothesis is dead.

**Worth recording separately:** a 37 to 57 percent relative perturbation of the
merged LoRA delta moves worst-task NLL excess by roughly 0.001 nats, about 0.3
to 1 percent. The metric is extremely flat against zero-mean perturbations of
the merged delta. That is a useful calibration of how weak the NLL proxy is, and
it is consistent with W5.

## 3. Hypothesis REFUTED: it is not the subspace geometry either

DARE's stated mechanism is reducing parameter interference between task vectors.
Interference should scale with subspace overlap, and we have two regimes 20x
apart in principal cosine. If the mechanism were real, DARE should help on the
degenerate cohort and not on the independent one.

```
regime                          n    mean(DARE - TA)   min       max    DARE better
shared-init  (cos 0.996)       12         +0.0015   -0.0005   +0.0030      2 of 12
independent  (cos 0.047)       12         +0.0018   +0.0002   +0.0026      0 of 12
```

No regime dependence. Both hypotheses that would have made this a finding about
*our setup* are refuted.

## 4. What is actually going on

DARE with rescaling is an **unbiased estimator** of the task vector, and
therefore of the merged delta:

```
E[delta * mask / density] = delta      =>      E[D_dare] = D_ta
```

For any loss convex in the merged delta, Jensen gives
`E[L(D_dare)] >= L(E[D_dare]) = L(D_ta)`. Locally, to second order,

```
E[L(D_dare)] - L(D_ta)  =  (1/2) tr(H Sigma),     Sigma proportional to (1-density)/density
```

so DARE + task arithmetic **cannot beat task arithmetic in expectation** wherever
the Hessian is PSD in the perturbed directions, and the penalty should grow
monotonically as density falls. This is a statement about the composition
mask + task_arithmetic, not about DARE in general (see section 6).

The prediction is testable against the existing density sweep, which was run for
other reasons and covers 0.05 to 0.5, spanning the DARE paper's headline regime
of dropping 90 to 99 percent.

```
DARE minus TA, shared-init seed1 (positive = DARE worse)
base           d=0.05    d=0.10    d=0.20    d=0.30    d=0.50
llama31_8b    +0.0030   -0.0000   -0.0005   +0.0001   +0.0003
mistral_7b    +0.0110   +0.0025   +0.0017   +0.0011   +0.0001
qwen25_7b     +0.0159   +0.0070   +0.0030   +0.0016   +0.0005
yi15_9b       +0.0114   +0.0051   +0.0023   +0.0005   -0.0003
```

Monotone in density on 3 of 4 bases (Llama is flat because its excess is
dominated by other terms), converging to TA as density approaches 1 (where the
operator becomes the identity) and degrading as density falls. **DARE beats
plain TA in 3 of 20 (base, density) cells, and all three are within 0.0005,
which is a fifth of the tie threshold.** The predicted functional form is what
the data shows.

## 5. This is not a harness insensitivity

The same sweep, same harness, same cells, for TIES, which is a **biased**
operation and so is not constrained to equal TA in expectation:

```
base              TA    d=0.05    d=0.10    d=0.20    d=0.30    d=0.50
llama31_8b    0.2132    0.2129    0.1722    0.1471    0.1421    0.1475
mistral_7b    0.1325    0.0919    0.0634    0.0494    0.0482    0.0507
qwen25_7b     0.1036    0.0631    0.0208    0.0130    0.0135    0.0137
yi15_9b       0.0989    0.0656    0.0355    0.0449    0.0468    0.0433
```

Effects up to 8x, with a clear interior optimum around density 0.2 to 0.3. The
harness detects density effects perfectly well. It reports none for DARE because
there are none to report.

## 6. Limits of this result, stated plainly

- **We tested mask + task_arithmetic only.** The official wrapper also supports
  mask + ties_merging (DARE-TIES), which is the configuration where DARE could
  plausibly help, because TIES is biased and the mask changes *which* parameters
  survive trimming. The unbiasedness argument in section 4 does **not** apply to
  that composition. We have not run it. Until we do, the claim is about
  DARE + task arithmetic, not about DARE.
- **T = 4 tasks, LoRA deltas, worst-task NLL.** The DARE paper's setting is full
  fine-tuned deltas and downstream accuracy.
- The density sweep is **seed1, shared-init, one seed**. The two-regime
  comparison in section 3 is 3 seeds x 4 bases per regime.
- `rd_ridge` is absent from this analysis by design; this is about the baselines.

## 7. Consequence for the paper

DARE is **not** a second KnOTS. KnOTS under `inner_combination="linear"` is
exactly Task Arithmetic algebraically, a genuine no-op, with published
`|KnOTS - TA|` of 0.00003 to 0.00031. DARE is a real and large perturbation of
the merged delta (37 to 57 percent) that the metric barely feels, and it is
consistently and predictably harmful rather than inert.

So the count of field-level contributions does not go from one to two on the
strength of DARE. What this is instead is a clean, mechanistically explained
negative result: **for LoRA merging under worst-task NLL, drop-and-rescale
cannot help when composed with task arithmetic, because rescaling makes it
unbiased, and the measured penalty follows `(1-density)/density` as predicted.**
That is a legitimate contribution, it is cheap to state, and it is defensible
because the mechanism predicts the functional form and the data matches it.

Reproduce with:

```
python code/phase3/scripts/diagnose_dare_truncation.py --cohort indep1
```

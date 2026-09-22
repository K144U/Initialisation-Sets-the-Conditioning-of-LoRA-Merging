# Pre-registration: a first-order drift theory of the collapse

Written 2026-09-23, before any quantity below has been computed on any
checkpoint. Governs `code/phase3/scripts/drift_theory.py` (to be committed
after this file) and nothing else.

Nothing here has been computed yet. The only numbers already known are those
in the paper: final-checkpoint kappa, amplification and median cosine on the
shared cohorts, and the drift trace 0.092/0.111/0.126/0.131 on the drift
cohort. Intermediate-checkpoint kappa and amplification have never been
computed.

## The derivation being tested

Per layer, A_t (r x d), V_t = row space of A_t, P_t its projector,
Hbar = (1/T) sum_t P_t, D_t = B_t A_t, G = (1/T) sum_t D_t, and the exact
interpolator tauH = G Hbar^+ (the paper's definitions, `floor_conditioning.py`).

Shared initialisation: A_t = A_0 + E_t. Let W_0 (d x r) be an orthonormal basis
of the row space of A_0 and P_perp = I - W_0 W_0^T. Every V_t with
W_0^T A_t^T invertible is the graph of a tilt

    X_t = P_perp A_t^T (W_0^T A_t^T)^{-1}          (d x r, columns in V_0^perp)

i.e. V_t = col(W_0 + X_t), and the principal angles between V_t and V_0
satisfy tan(theta_i) = sigma_i(X_t) exactly. ||X_t|| <= ||E_t|| / sigma_min(A_0)
to first order, so the tilt is O(eps), eps = ||E|| / ||A_0||.

Expanding P_t to second order in X_t and taking the Schur complement of Hbar
on V_0^perp:

    small eigenvalues of Hbar  ~=  nonzero eigenvalues of
    Sigma_X := (1/T) sum_t (X_t - Xbar)(X_t - Xbar)^T          (rank (T-1) r)

while the top r eigenvalues are 1 - O(eps^2). Hence

  (K)  kappa(Hbar) ~= kappa_hat := 1 / lambda_min^+(Sigma_X).

For the interpolator, write D_t = M_t (W_0 + X_t)^T with
M_t = D_t W_0 (in the graph coordinates the exact interpolator w must satisfy
w (W_0 + X_t) = M_t for every t). Splitting w = a W_0^T + b with b supported on
V_0^perp and in the span of the centred tilts:

    b Y = [M_1 - Mbar, ..., M_T - Mbar],   Y := [X_1 - Xbar, ..., X_T - Xbar]
    b = dM Y^+,    a = Mbar - b Xbar

  (A)  amplification ~= alpha_hat := ||[a, b]|| / ||G||.

Consequences for scaling. Y is O(eps). So under a uniform rescaling of the
displacement (E_t -> s E_t), kappa scales as s^-2 and the amplification as
s^-1. **The amplification is predicted to be first order in 1/eps, not
second.** The external review's rough version gave amplification ~ 1/eps^2
(~59 at eps = 0.13, near the measured 43-64); this derivation predicts that
agreement is coincidental. The large kappa (1e4 to 8e4, against
1/eps^2 ~ 59) is predicted to come from the spectrum of the displacement
spread, lambda_min^+(Sigma_X) << eps^2, i.e. from the fact that training
displacements are far from isotropic, not from eps alone.

## Data

- Drift cohort (A_0 saved): `artifacts/lora/llama31_8b/*/drift_shared/`,
  `adapter_A0.safetensors` plus checkpoints 117/234/351/468 (25/50/75/100%).
  Four tasks, T = 4, r = 16. The final adapter equals checkpoint 468.
- Final shared cohorts `seed1/2/3` on all four bases (A_0 not saved; see PT4).
- Layer sample: the same 8 `lora_A` keys the paper's geometry scripts take
  (`keys[::len//8][:8]` over sorted keys). Every statistic is the median over
  those 8 layers unless stated.

## Predictions, fixed now

- **PT1, conditioning formula.** On the drift cohort's shared arm, at each of
  the four checkpoints, median over layers of kappa_hat / kappa lies in
  [0.75, 1.33]. Holds if it does at >= 3 of 4 checkpoints.
- **PT2, amplification formula.** Same, for alpha_hat / alpha, same band,
  >= 3 of 4 checkpoints.
- **PT3, the order in eps (the discriminating test).** Counterfactual dial on
  the final drift_shared adapters: A_t(s) = A_0 + s (A_t - A_0), B_t unchanged,
  s in {0.05, 0.1, 0.2, 0.5, 1.0}. Fit log-log slopes over these s of the
  EXACT kappa and EXACT amplification (not the predicted ones). Predicted:
  kappa slope in [-2.2, -1.8]; amplification slope in [-1.2, -0.8].
  The rival (review) form predicts amplification slope in [-2.2, -1.8].
- **PT4, reference-free version on every base.** For seed1/2/3 on all four
  bases (12 cohorts), with A_0 unknown, take W_0 := top-r eigenvectors of
  Hbar (the consensus subspace) and compute kappa_hat and alpha_hat the same
  way. To first order the centring removes the dependence on the reference.
  Predicted: kappa_hat/kappa and alpha_hat/alpha in [0.75, 1.33] on >= 10 of
  12 cohorts, each.
- **PT5, direction across real training.** On the drift cohort's shared arm,
  exact kappa and exact amplification are both non-increasing across
  25 -> 50 -> 75 -> 100% (tolerance 5% at each step), since drift grows over
  those checkpoints. No magnitude is registered: the displacement spectrum can
  change shape during training, which a uniform dial cannot capture.

## Branches, written now

1. **PT1-PT4 hold.** The paper gains a proposition (first-order form of kappa
   and amplification in the displacement spread) with the dial as its direct
   test, and states that amplification is first order in 1/eps. The measured
   43-64 is then explained by alpha_hat, not by 1/eps^2.
2. **PT3 amplification slope lands in [-2.2, -1.8].** The review's form is
   right and the derivation above is wrong at the interpolator step. Report
   that, and state whichever scaling was measured.
3. **PT1 or PT2 fails but PT3 holds.** The scaling is right and the
   first-order constant is not; the proposition is stated as a scaling law
   only, without the closed-form predictor.
4. **PT3 fails in both slopes.** The first-order picture does not describe
   these adapters at eps ~ 0.13; the theory item is dropped and the paper
   says it was tested and did not hold.

PT5 is reported whatever it shows and changes no branch.

## Explicitly NOT registered here

- Any explanation of why lambda* is 0.01 on seven of eight RegMean arms. The
  lambda* values are already known, so any account of them written now is
  post hoc. If the theory suggests a lambda rule, that rule is registered
  separately and tested only on cohorts it has not seen.
- Anything about the independent arm, where the tilt is not small and the
  derivation does not apply (its amplification is T by a separate identity).

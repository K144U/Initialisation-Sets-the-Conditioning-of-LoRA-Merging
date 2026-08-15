# Pre-registration: does conditioning have an operational consequence?

Written 2026-08-07, after `2fdb223` established that the exact instance floor is
zero in both regimes and that what actually differs between shared and
independent initialisation is the conditioning of `Hbar` (cond 17.6k-83k against
1.5-1.6; exact-interpolator amplification 43-64 against exactly 4.0 = T).

Committed BEFORE any of the cells below are dispatched. No rule here may be
changed after any number is read.

## Blindness statement

What is already known and therefore NOT blind:
- The geometry of both cohorts (committed, `cdbec03` and earlier).
- The head-to-head null: independent init better on 2 of 20 cells, tie on 17,
  worse on 1, mean +0.0014 nats (`2fdb223`).
- That rd-encoder ridge wins on shared-init cohorts and loses to TIES on
  independent ones (`89fde0e`).
- That the published lambda-star values were 0.05 on Llama and 0.13 elsewhere,
  fitted on the SHARED cohort only.

What is blind: every cell below. No lambda sweep has been run on an
independently initialised cohort, and no untruncated rate sweep has been run at
all.

The E2 prediction below is a genuine forecast, and it is the one the paper's
spine now rests on. It is stated before the compute precisely so that it can
fail.

---

## E2 (PRIMARY). Does the optimal ridge track the conditioning?

### Motivation

A Tikhonov ridge is a conditioning fix. If shared-init cohorts are
ill-conditioned and independent ones are not, the ridge should be doing real
work in the first case and nothing in the second. If true, this explains the
otherwise unexplained Q1' result: our encoder's advantage on shared cohorts was
the ridge repairing conditioning, and it evaporated on independent cohorts
because there was nothing left to repair.

### Design

`rd_encoder` with `ridge_lambda` swept over
`{0, 0.01, 0.03, 0.05, 0.13, 0.30, 1.00}` (7 values), on 4 bases, on two
conditions: `seed1` (shared) and `indep1` (independent). 56 cells. Rank is
pinned to 16 in BOTH arms, so this sweep is not confounded by the rank
mismatch recorded in audit finding A3.

Metric is worst-task NLL excess, as everywhere else. `lambda*` is the
arg-min over the grid.

### Pre-registered prediction

**P1.** `lambda*(shared) >= 10 x lambda*(independent)` on at least 3 of 4 bases,
treating a `lambda* = 0` on the independent arm as satisfying the inequality for
any non-zero shared `lambda*`.

**P2.** The ridge gain `G = L(lambda=0) - L(lambda*)` is larger on the shared
arm than the independent arm by more than 0.005 nats on at least 3 of 4 bases.

### Decision rules

- **CONFIRMED** if P1 and P2 both hold. Conditioning has an operational
  consequence, it explains Q1', and it becomes the paper's spine.
- **PARTIAL** if exactly one holds. Report both, claim only the one that held,
  and do not narrate the other as a trend.
- **REFUTED** if neither holds. Then conditioning differs by four orders of
  magnitude and has no operational consequence for any method we can test. That
  outcome is reported as the headline negative result, and the paper becomes
  the theory plus a documented failure of the geometry to matter. It does NOT
  get quietly downgraded to a limitation.

Ties: a 0.005 nat threshold and a one-directional 2 x SE noise gate, identical
to every previous pre-registration in this project. The gate may only downgrade.

---

## E1 (DECISIVE FOR AN EXISTING CLAIM). Is the flat rate curve a truncation
artifact?

The paper currently reports the finite-rate exponent as falsified, with slopes
-0.10 and -0.21 against a predicted `[-2.4, -1.6]`. An external review points
out that every method is SVD-truncated back to rank r after merging, that this
discards most of an independently initialised merge (rank up to Tr = 64), and
that a dominant truncation error would produce exactly the observed flatness.

### Design

Repeat the quantizer-width sweep `b in {1,2,3,4,8,16}` on 4 bases on `indep1`
with the post-merge rank truncation DISABLED. 24 cells. The existing truncated
sweep is the comparison arm and is not re-run.

### Decision rules, fixed now

- **FALSIFICATION VOID** if the untruncated slope enters `[-2.4, -1.6]` on at
  least 3 of 4 bases. Then the flatness was our setup, the §7.3 falsification is
  withdrawn, and we say so in those words.
- **FALSIFICATION STANDS** if the untruncated slope satisfies `|slope| < 0.5` on
  at least 3 of 4 bases. Truncation is then exonerated as the explanation.
- **UNRESOLVED** otherwise. Reported as unresolved, not resolved in whichever
  direction is more convenient.

Also recorded now, because it is the honest reading of a live anomaly: on two
bases the truncated sweep produced finite-rate excess BELOW the `b = infinity`
value, which is impossible under the model. If that persists without truncation,
it is a harness problem and will be investigated and reported as such rather
than described as the model not allowing it.

---

## E3 (SANITY, blocks the geometry claim). Are the adapters undertrained?

The shared-init mechanism requires that `A` moves little. Measured
`||dA||/||A||` is 0.16 to 0.19, which is also what undertraining looks like. We
have never reported per-adapter task performance, so a reader cannot currently
tell whether these adapters learned their tasks.

### Design

1. Per-adapter task loss against the base model, all 4 bases x 4 tasks x both
   conditions. This is largely recoverable from existing eval cells, since
   worst-task excess is already measured relative to the per-task adapter.
2. `||dA||/||A||` and median principal cosine at training checkpoints
   (25/50/75/100 percent of steps) for one base, both conditions.

### Decision rule

If, at the final checkpoint, `||dA||/||A||` is still rising by more than 20
percent between the last two checkpoints, undertraining is NOT excluded, and
every geometry claim in the paper must be scoped to "at this training budget".
No exceptions and no arguing from the other checkpoints.

---

## E4 (CORRECTNESS, no claim attached). Split overlap.

Training and evaluation currently overlap by about 14.5 percent (instruction
following) and 10 percent (code) because the two draws use different seeds over
one pool. Fix the split construction to be disjoint by construction and re-score
every cell used in the paper. This changes absolute numbers and is expected to
change no comparison; if it does change a comparison, that fact is reported.

---

## Binding constraints

1. No threshold or rule above may be edited after any number is read.
2. E2 is primary. If E1 or E3 produce something more flattering, they do not
   become the headline.
3. Smoke-first applies to every new path here: the untruncated merge in E1 and
   the rank-pinned ridge sweep in E2 are both new configurations, and one real
   cell of each is inspected before the rest are dispatched.
4. All lambda values are reported, including those that flatter the method and
   those that do not.
5. If E2 is REFUTED, the paper is written around that outcome. The result is not
   demoted to a limitation and the spine is not quietly replaced with whichever
   secondary result survived.

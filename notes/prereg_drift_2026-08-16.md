# Pre-registration: how far does A travel from its initialisation?

Committed BEFORE any adapter of this arm is trained and before the generator
exists. No rule here may be changed after a checkpoint lands.

## What this discharges

Registration E3 asked for `||dA||/||A||` at 25, 50, 75 and 100% of training
steps, and attached a scoping rule: if the sweep cannot be evaluated, every
geometry claim in the paper is scoped to "at this training budget". We reported
it as permanently unevaluable, because we did not retain initial weights or
intermediate checkpoints for the original cohorts and cannot recover them.

That was true of the *original* cohorts and is not a reason to leave the rule
undischarged forever. This registration trains a fresh cohort with the
checkpoints E3 needs.

## Why it matters, stated plainly

The mechanism in the paper's §4.1 has two halves:

1. `B` is initialised to zero, so the gradient reaching `A` is proportional to
   `B` and vanishes at initialisation. **This is a property of the
   parameterisation, not a measurement**, and the paper says so.
2. Therefore `A` moves comparatively little, and each task keeps the row space
   it started in.

We currently measure only the *consequence* of (2), the between-adapter
distance, and never (2) itself. A reader is entitled to ask whether `A` really
does stay near its initialisation or whether the row spaces coincide for some
other reason. This measures it.

**A null result here is publishable and we say in advance what it would mean.**
If `A` travels far from its initialisation and the row spaces are still
near-collinear, then the mechanism as stated in §4.1 is wrong, the collapse has
some other cause, and we would rewrite that subsection to say we do not know
the cause. The empirical results in §4 and §5 do not depend on the mechanism
being right, because they are measurements of the consequence.

## Design

One base model, **Llama-3.1-8B-Instruct**, chosen now and not after seeing
anything: it is the base with the most complete existing coverage in this
paper, so the new cohort is comparable to the most other cells.

- **Two arms**, matching the paper: one shared-initialisation cohort (all four
  tasks from one global seed) and one independent cohort (a separate seed per
  task). 8 training runs total.
- **Four tasks**, the same four the paper uses, at the same rank `r = 16`,
  the same hyperparameters and the same data seeds as the existing configs.
  Nothing about the training recipe changes except what is listed below.
- **Two changes, both necessary and both declared here.**
  1. `A_0` is written to disk before the first optimiser step. Without it the
     quantity is undefined.
  2. Checkpoints at 25, 50, 75 and 100% of total steps, which E3 requires.
- The **translation task keeps `stream_shuffle_buffer` unset**, i.e. the same
  unshuffled prefix the original cohorts saw. This cohort exists to measure
  drift, not to fix the translation adapter, and changing two things at once
  would make it comparable to nothing.

## Quantities, fixed now

At each checkpoint fraction `f` in {25, 50, 75, 100}, per layer and averaged
over the same eight sampled layers the geometry audit uses:

- `drift(f) = ||A_f - A_0||_F / ||A_0||_F`, per task, then the median over the
  four tasks.
- `cos(f)` = median principal cosine between the four task row spaces, so that
  drift and collapse are tracked together.

## Predictions, fixed now

- **P1 (primary).** `drift(100) < 0.5` on the shared arm. The mechanism claims
  `A` moves little; half its own norm is a generous ceiling for "little".
- **P2.** `drift` is monotone non-decreasing in `f`, within numerical noise of
  0.01. A non-monotone trace would indicate a checkpointing or ordering fault
  rather than a finding, and would be reported as such.
- **P3.** `cos(100) > 0.9` on the shared arm, i.e. the new cohort reproduces
  the collapse the paper reports. If it does not, this cohort is not a
  replication of the phenomenon and the drift numbers say nothing about it.

## Decision rule, all branches written now

1. **P1, P2 and P3 all hold.** E3 is discharged. Appendix C.3 changes from a
   withdrawal to a measurement, the "scoped to this training budget" caveat is
   removed from geometry claims, and §4.1's mechanism is stated as measured on
   this cohort rather than argued from the parameterisation.
2. **P3 fails.** The cohort did not reproduce the collapse. We report the drift
   numbers as descriptive, discharge nothing, and keep the scoping caveat.
3. **P1 fails with P3 holding.** `A` moves a long way and the subspaces still
   collapse. **This falsifies the mechanism as stated.** We rewrite §4.1 to
   report that the collapse is real and its cause is not what we said, and we
   say in the abstract that the mechanism is not established. The empirical
   results do not change.
4. **P2 fails.** Treated as a fault, not a result. We report it, do not use the
   trace, and do not discharge E3.

## Binding constraints

1. No checkpoint is inspected until all four fractions exist for all eight
   runs, or a run has failed terminally.
2. The 0.5 threshold in P1 and the 0.9 in P3 are fixed. If either moves after a
   checkpoint lands, this registration is void and the result is exploratory.
3. `A_0` is written before the first optimiser step by the training script
   itself, not reconstructed afterwards from a seed. A reconstruction would be
   a claim about RNG determinism, which is a different and weaker thing.
4. The scoping consequence of E3 is discharged **only for this base model and
   this training budget**. One cohort on one base does not license a claim
   about all training budgets, and the paper will say so.

## Reported regardless of outcome

The full drift and cosine traces at all four fractions for both arms, the
per-task spread, and the number of runs that completed.

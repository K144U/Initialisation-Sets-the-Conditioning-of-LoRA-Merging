# Pre-registration: is the heuristics null visible in downstream accuracy?

Committed BEFORE the generator for these cells exists and before any cell is
dispatched. No rule here may be changed after a cell lands. The commit that
introduces this file is the audit anchor; every result commit for these cells
must post-date it.

## Why this exists

The paper makes two claims about conditioning. The positive one, that
solve-based merges collapse on a shared-initialisation cohort and one ridge
term repairs them, is now measured in two units: worst-task NLL excess
(§7.3) and downstream accuracy on GSM8K and HumanEval (§7.8). The null, that
the collapse is invisible to task arithmetic, TIES, DARE, TVQ and the
repaired KnOTS, is measured in **one** unit only, and it is the unit §8
disclaims.

That asymmetry is the problem, and it is ours rather than a referee's to
find. §8 reports that worst-task NLL excess does not predict downstream
accuracy across merge methods at fixed cohort, and on HumanEval predicts it
with the wrong sign (rho = +0.45). Limitation 2 answers that the failed
correlation is across methods at fixed cohort while the null is across
cohorts at fixed method, which is true and which we keep. It does not answer
the magnitude objection: the null's effects are hundredths of a nat, which is
precisely the range in which we have told the reader not to trust this
metric. A null in a disputed unit, with the flattering half of the same
section validated in a second unit and the null not, is an asymmetry that
undercuts the paper's own standard.

So we run the null in the second unit.

## Blindness statement, stated plainly

- The **independent arm has never been evaluated downstream** for any of the
  five heuristics. No cell exists. This half is blind.
- The **shared arm has been evaluated before**, in `eval_e3_gsm8k` and
  `eval_b4_humaneval` (20 cells each), and those numbers **have been read**:
  they are the data behind §8's correlation. This half is not blind and we do
  not claim it is.
- Those existing cells were scored with the **pre-repair** GSM8K and
  HumanEval scorers, which §8 reports discarded most generations
  method-dependently. They are therefore not comparable to anything scored
  with the repaired scorers, and we will not compare against them.

The consequence, fixed here: **both arms are re-run from scratch with the
repaired scorers**, and the existing 40 cells are used for nothing in this
test. Reusing the shared arm would be cheaper by half and would import a
known scoring defect into the comparison, and it would mean the paired
difference was computed across two scorer versions.

We record that the shared arm's *pre-repair* accuracies have been seen. What
has not been seen is any post-repair accuracy for any heuristic on either
arm, which is what every rule below is stated over.

## Design

5 methods x 4 bases x 2 cohorts x 2 benchmarks = **80 cells**.

- **Methods**: `ta`, `ties`, `dare`, `tvq_b2`, `knots` (the repaired KnOTS of
  §8, not the no-op that reduced to task arithmetic). These are exactly the
  five of the §7.2 null. No method is added or dropped after the fact.
- **Bases**: `llama31_8b`, `mistral_7b`, `qwen25_7b`, `yi15_9b`.
- **Cohorts**: `seed1` (shared) and `indep1` (independent). One cohort per
  arm, matching §7.2's per-cell unit; see the ceiling below.
- **Benchmarks**: GSM8K at n = 500, HumanEval at n = 164 (the whole test
  split, per the 2026-08-15 amendment).
- Every cell inherits `base_model`, `max_seq_length`, `seed`, `min_free_gb`
  and `loader` from the corresponding NLL cell in `eval_a1_indep`, asserted
  by the generator before anything is written. The two cohorts of a pair must
  differ **only** in the adapter directory.
- Merge hyperparameters are read from the existing NLL configs and are never
  re-selected here. In particular TIES density stays 0.2 and TVQ stays b = 2.

## Statistical ceiling, acknowledged up front

This design has one cohort per arm, so the paired difference for a given
(method, base) is a single number and carries no across-cohort standard
error. The gate below is therefore a **binomial** gate on the two accuracies,
not the 2 x SE-over-cohorts gate used elsewhere in this project. It is a
weaker instrument and we say so rather than presenting it as equivalent.

The minimum detectable effect follows from n and is large:

- GSM8K, n = 500: 2 x SE of a difference of two independent proportions near
  0.5 is about **0.089**, i.e. ~9 accuracy points.
- HumanEval, n = 164: about **0.156**, i.e. ~16 points.

**Any null from this test is a statement of the form "no effect detectable at
about 9 points on GSM8K and 16 on HumanEval", and must be written that way.**
It is not evidence that the effect is zero, and we will not write it as if it
were. We register this sentence now so that it cannot be softened later.

We choose n = 1 cohort per arm over fewer methods at n = 3 deliberately, and
the reason is fixed here: the claim under test is about the five methods as a
class, so dropping methods to buy cohorts would change the claim rather than
strengthen the test of it.

## Threshold and gate

For each (method, base, benchmark) cell, let `d = acc_indep - acc_shared`.

- **Threshold.** |d| must exceed **0.05** (5 accuracy points) to count as an
  effect at all. Fixed now, chosen before any cell, and chosen to be
  comparable to the accuracy differences §7.8 already reports as meaningful.
- **Gate.** |d| must also exceed `2 * SE_binom`, with
  `SE_binom = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)` computed from the two measured
  accuracies. The gate is **one-directional**: it may downgrade an effect to
  "no detectable effect", never promote.
- A cell is an **effect** only if both are cleared. Otherwise it is **no
  detectable effect**.

## Predictions, fixed now

- **P1 (primary).** At most **4 of the 40** (method, base, benchmark) cells
  show a detectable effect, and the paper's null carries over to accuracy.
- **P2.** No method shows a detectable effect on **3 or more** of its 8 cells
  (4 bases x 2 benchmarks). A method that did would be behaving like a solver
  and would contradict the mechanism of §7.2.
- **P3 (directional, secondary).** Among any cells that do clear, the
  direction favours the independent arm, matching the five NLL cells that
  cleared in §7.2. A cell clearing in favour of the *shared* arm is the
  outcome we consider most informative against us and it will be reported
  first if it occurs.

## Decision rule and what we write in each branch

Fixed now, for all four branches, so that no outcome can be narrated
favourably after the fact.

1. **P1 and P2 both hold.** The null holds in a second unit. §7.2 gains a
   sentence saying so and the accuracy table goes in the main text. We
   report the MDE alongside, in the words fixed above.
2. **P1 holds, P2 fails.** One method is doing something the others are not.
   We name it, report its cells, and **remove it from the null**, restating
   contribution 3 over the remaining methods. The class claim narrows.
3. **P1 fails with fewer than 10 cells clearing.** The null does not carry
   over cleanly. We report the count, keep the NLL null as measured, and
   **state in §1 and in the abstract that the invisibility claim holds in NLL
   and is not confirmed in accuracy.** Contribution 3 is rewritten to claim
   only what survives.
4. **P1 fails with 10 or more cells clearing.** The invisibility claim is
   wrong as stated. We withdraw it, say so in the abstract, and the paper's
   practical message reduces to the solver half. We would rather find this
   than have a referee find it.

## Binding constraints

1. No cell is inspected until all 80 have landed or a cell has failed
   terminally. Partial reads are how thresholds move.
2. The threshold (0.05) and the gate (2 x SE_binom, one-directional) are
   fixed. If either is changed after any cell lands, this registration is
   void and the result is reported as exploratory.
3. Merge hyperparameters are read from the committed NLL configs. Nothing is
   re-tuned on accuracy under any outcome.
4. Cells that fail to produce a scorable generation are reported as such,
   with counts, and are not silently dropped. The repaired scorers are used
   as committed; no scorer change is permitted during or after this run.
5. The existing pre-repair cells are not used in any comparison here.

## Reported regardless of outcome

The full 40-row table of `d` values with their gates, the per-cell empty and
discard counts, the MDE sentence, and the number of cells clearing each
criterion. If the run is incomplete, the number completed and which are
missing.

# Pre-registration: is the conditioning collapse visible in downstream accuracy?

Written 2026-08-15, **before any cell of this test is generated or dispatched**.
The commit ordering is the evidence; if any result commit of this test is an
ancestor of this file's commit, the registration is void and must be reported
as void.

## The question, and why it is the one that matters

Limitation 2 of the current draft says worst-task NLL excess does not predict
downstream accuracy, and on HumanEval predicts it backwards
($\rho = +0.45$, CI $+0.10$ to $+0.73$). That limitation is stated about
*fine-grained comparisons between merge methods*, where the differences are
hundredths of a nat. A reader is entitled to extend it to the whole paper: if
the metric does not track accuracy, why should anyone care about any of our
numbers?

The conditioning effect is not a fine-grained difference. At $\lambda = 0$ the
shared-versus-independent gap runs from $0.13$ to $9.63$ nats, one to three
orders of magnitude above both the metric's resolution and the method-to-method
differences the correlation was computed over. An effect that large either
shows up in accuracy or the metric is measuring something other than model
quality. We do not know which, and this test is designed so that either answer
is reportable.

## Design

The merges are ones the paper already reports; only the evaluation is new.

- 4 bases $\times$ 2 cohorts $\times$ 2 ridge values $\times$ 2 benchmarks =
  **32 cells**.
- Cohorts: `seed1` (shared) and `indep1` (independent), the two arms of the E2
  sweep.
- Ridge values: $\lambda = 0$, and $\lambda^\star$ taken **per (base, cohort)**
  as the arg-min over the E2 grid from `eval_ridge_cond`, already on disk. No
  new tuning, and no value chosen after seeing an accuracy number.
- Benchmarks: GSM8K exact-match and HumanEval pass@1, both with the **repaired**
  scorers whose discard rate is now zero. HumanEval is included precisely
  because it is where the inverted correlation was measured.
- `realize: rank_r`, `loader: plain`, `n_eval_metric: 500`, matching the NLL
  cells this test is meant to be compared against. The generator asserts that
  equality rather than assuming it.

## Statistics

Accuracy on $n = 500$ items is a proportion, so a difference of two accuracies
carries $\mathrm{SE} = \sqrt{p_1(1-p_1)/n + p_2(1-p_2)/n}$, which is at most
about $0.032$ and typically smaller. That is enough to gate a comparison without
replicating cohorts, which is why this test does not need $n = 3$ arms.

**Threshold.** A difference in accuracy counts only if it exceeds
$\max(5\ \text{percentage points},\ 2\times\mathrm{SE})$. The 5-point floor is
fixed now, is not derived from any measurement, and exists so that a difference
which is statistically resolvable but practically irrelevant is not reported as
an effect.

**The gate is one-directional**, as everywhere else in this project: it may
only downgrade a directional call to a tie, never promote one.

## Predictions, fixed now

- **P1 (the collapse is visible).** On the **shared** arm, accuracy at
  $\lambda^\star$ exceeds accuracy at $\lambda = 0$ by more than the threshold,
  on at least 3 of 4 bases, on at least one benchmark.
- **P2 (the asymmetry matches the NLL result).** The accuracy gain from
  $\lambda = 0$ to $\lambda^\star$ is larger on the shared arm than on the
  independent arm, on at least 3 of 4 bases, on at least one benchmark.

P2 is the one that matters. P1 alone could be explained by the ridge helping
every merge everywhere; P2 is the claim that it helps *because* of conditioning,
which is what the paper argues.

## Decision rule

- **CONFIRMED** if P1 and P2 both hold.
- **PARTIAL** if exactly one holds.
- **REFUTED** if neither holds.

## What we write in each branch, fixed now

- **CONFIRMED.** Limitation 2 is **rewritten, not removed**. It is restated as
  applying to fine-grained comparisons between methods, where it was measured,
  and the paper states that the conditioning effect specifically does reach
  downstream accuracy, giving the numbers. Section 1 and the abstract say so.
  We do not use this to claim the metric predicts accuracy in general, because
  this test does not measure that.
- **PARTIAL.** Only the half that held is claimed, in the same words.
  Limitation 2 is narrowed to exclude what replicated and keeps everything else.
- **REFUTED.** A $0.13$ to $9.63$ nat effect does not move either benchmark.
  Limitation 2 is then **strengthened, and moved into Section 1**, and the paper
  states plainly that the quantity it is written in has no demonstrated
  connection to downstream behaviour at any effect size we can produce. The
  practice recommendations are cut rather than defended, per the standing
  instruction in this project that a section we cannot support is removed and
  not argued for.

We record now, before any cell, that REFUTED is the outcome that most damages
the paper and is also the one a reader most needs, and that a $0\%$ accuracy at
$\lambda = 0$ on the shared arm would be the single most useful number this
project could produce.

## Reported regardless of outcome

1. **All 32 cells**, as a table, including both benchmarks and both arms,
   whether or not they support P1 or P2.
2. The **discard rate** of each scorer on every cell. The earlier downstream
   numbers were void because scorers silently dropped $61$ to $81\%$ of
   generations; every cell here reports its own rate, and any cell with a
   non-zero rate is flagged in the table rather than averaged in silently.
3. **Absolute accuracies, not only differences**, so a reader can see whether a
   merge is near the floor of the benchmark.
4. The $\lambda^\star$ used per (base, cohort), and that it was read from the
   existing sweep rather than fitted here.

## Binding constraints

1. No threshold or rule above may be changed after any number is read.
2. $\lambda^\star$ comes from the committed E2 sweep. It is not re-selected on
   accuracy, in either arm, under any outcome.
3. The generator asserts that the two arms differ only in the adapter cohort,
   and refuses to write cells otherwise.
4. Smoke test first: one real cell inspected for configuration equality and a
   non-zero scorer yield before the remaining 31 are dispatched. The inspection
   does not look at the direction of any effect. The smoke cell is one of the
   32, so blindness is 31 of 32 and the paper says so.
5. The analyzer is committed **before** the cells it reads have landed.
6. If this contradicts a published verdict of ours, both are reported, with the
   superseded one named and dated, in the paper and not only here.

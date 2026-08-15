# Amendment 1 to the downstream accuracy pre-registration

Written 2026-08-15, after `df832fe` and **before any cell of the test is
generated or dispatched**. Amends the evaluation size only. Nothing else in the
parent document changes.

## What prompted it

The parent registration fixes `n_eval_metric: 500` for every cell, "matching
the NLL cells this test is meant to be compared against". Before generating
anything we checked the benchmark sizes and found that clause cannot be
satisfied for one of the two benchmarks.

**HumanEval has 164 problems in total.** `openai/openai_humaneval` ships a
single `test` split of 164 items and there is no larger split to draw from. A
request for 500 is not a stricter setting, it is an impossible one; the harness
would silently evaluate all 164 and report a number produced under a
configuration different from the one registered.

GSM8K is unaffected: its test split has 1319 items, so 500 is a genuine
subsample and the parent's value stands.

## The amendment

1. `n_eval_metric` is **500 for GSM8K** and **164 for HumanEval**, the latter
   being the whole benchmark. No other field changes.
2. The **gate adapts to the actual $n$**, which the parent already implies but
   which is worth stating because the two benchmarks now differ. With
   $\mathrm{SE} = \sqrt{p_1(1-p_1)/n + p_2(1-p_2)/n}$, the worst-case
   $2\times\mathrm{SE}$ is about $0.063$ on GSM8K at $n=500$ and about $0.110$
   on HumanEval at $n=164$. The threshold remains
   $\max(5\ \text{points},\ 2\times\mathrm{SE})$, so in practice HumanEval is
   gated near 11 points and GSM8K near 6.
3. Because HumanEval's gate is roughly twice GSM8K's, **the two benchmarks are
   not equally powered and we will not read a null on HumanEval as equivalent
   to a null on GSM8K.** Any HumanEval null is reported together with its
   minimum detectable effect, and the parent's requirement that a prediction
   hold "on at least one benchmark" is unchanged: it was written that way
   precisely because the benchmarks differ.

## What this amendment cannot do

It changes an evaluation size to a physically achievable value and it widens a
gate, which makes our own predictions harder to satisfy rather than easier. It
does not touch a threshold in the direction that would flatter us, does not
change either prediction, and does not change the decision rule. It was written
before any cell existed, which is checkable in the commit graph.

Had it moved the 5-point floor, or been written after a single cell had landed,
it would be worth nothing and should be read as worth nothing.

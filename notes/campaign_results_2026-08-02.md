# Campaign results log, 2026-08-02

Live record of the review-response campaign. Appended as cells land.
Plan and decision thresholds: `RUNBOOK_review_response.md`.
Weakness triage: `review_response_2026-08-02.md`.

---

## Smoke gate (job 45689, rerun 45690): 5/5 PASS

```
PASS  ta_alpha0p25 control   excess 0.2139 vs published 0.2132   (bar |d| < 0.005)
PASS  rd_renorm runs         excess 0.2240 vs unrenormed 0.0822  -> scale is load-bearing
PASS  knots_ties differs     |KnOTS-TIES - TA| = 0.02800         (knots-linear was 0.00014)
PASS  rd_ridge_b4 runs       excess 0.0845, q(4) = +0.0023
PASS  gsm8k extractor fixed  failure 0.000 vs published 0.611, 385/500 full gens stored
```

### The gate paid for itself immediately

`knots_ties` failed on the first attempt with

```
RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu!
  knots.py:60 in _inner_merge_ties
```

`_inner_merge_ties` built its weight tensor with a bare `torch.tensor()`, which
lands on CPU while `stacked` is CUDA. `merge_ties` in `ties.py` has always
passed `device=`; this path never did, because `inner_combination="ties"` had
never once been run on a GPU. Latent since the file was written. All 12 A2
cells would have crashed. Fixed at `6b6e6c4`, with a CPU-runnable device-safety
test (patch `torch.tensor`, assert every constructed tensor got an explicit
device) that was verified to catch the bug by reintroducing it.

Also fixed: my own `rd_ridge_b4` bar compared a seed-1 cell against the 3-seed
rd-ridge mean and would have reported a spurious FAIL.

---

## Result 1 (A5/W5): the GSM8K scorer was measuring formatting, not arithmetic

Llama-3.1, Task Arithmetic, seed1, same merge, same generations, new scorer:

| | EM | extraction failures |
|---|---|---|
| published | 0.326 | 300/500 (60.0%) |
| re-scored | **0.788** | **0/500 (0.0%)** |

**Plausibility check that should have caught this in the original run.**
Llama-3.1-8B-Instruct scores roughly 84% on GSM8K with CoT. The paper reports
0.315 to 0.323 for TA/DARE/KnOTS, i.e. *far below the base model's own ability*
on a task whose adapter has 0.68 nats of headroom. A merged model that has not
collapsed cannot be that bad at GSM8K. The published number was a property of
the regex, not of the merge.

385 of 500 generations exceed the old 200-char storage cap, which is exactly
why this could not be re-scored offline and needed GPU time.

**Correction to my own earlier estimate.** In `review_response_2026-08-02.md`
I estimated this cell would move 0.315 -> 0.225 by re-scoring the stored
previews. That was wrong in both magnitude and direction. The previews were
truncated mid-reasoning, so "last number in the preview" was usually an
intermediate quantity rather than the final answer. I flagged the estimate as
indicative-only at the time; treat it as void, not as a second data point.

Expect the whole downstream table and all eight Spearman correlations to move.
The (Llama-3.1, GSM8K EM) rho = -0.60 outlier, limitation (5), and the H1/H2/H3
falsification appendix all depend on numbers that are now known to be wrong.

## Result 2 (W1/V2): the norm is load-bearing, contradicting my prediction

Llama-3.1, seed1, rd-encoder ridge at lambda* = 0.05:

| variant | worst-task excess |
|---|---|
| rd-ridge as published (rank_deff) | 0.0822 |
| rd-ridge renormalized to \|\|TA\|\| | **0.2240** |
| Task Arithmetic, alpha = 0.25 | 0.2139 |

Forcing W\* to TA's Frobenius norm while keeping its direction costs **+0.142
nats** and lands it *worse than TA itself*. My pre-registered V2 threshold was
"renorm costs < 0.010 nats => the win is direction, not scale". That is
decisively refuted on this base.

**How this squares with the offline algebra.** Both facts hold at once:
- 84% of W\*'s Frobenius mass is orthogonal to every rescaling of TA
  (best alpha 1.31, residual 0.841), so it is *not* literally a scaled TA;
- but its advantage evaporates without its 2.44x norm.

So rd-ridge is a specific direction **at a specific large scale**, and the
scale is not incidental. This makes the TA alpha-sweep the decisive test rather
than a formality: if TA at alpha ~ 0.6 (matching the 2.44x norm) reaches 0.082,
the reviewer's W1 lands hard. If it does not, the paper has a much stronger
answer than the algebra alone provided, because it can then say the direction
*and* the scale are both necessary and neither alone suffices.

I was too confident in the earlier triage. The data corrected me.

## Result 3 (A2): KnOTS with a working inner merge

| | \|KnOTS - TA\|, Llama seed1 |
|---|---|
| published KnOTS (`inner_combination="linear"`) | 0.00014 |
| KnOTS-TIES (`inner_combination="ties"`) | **0.02800** |

A 200x difference. The published row could not have differed from TA; this one
does. Whichever way the remaining 11 cells fall, the four claims that cite
"KnOTS ~ TA" as evidence *for* the theory cannot stand as written, because
their evidence was a no-op.

## Result 4 (W3): the finite-rate path works

rd-ridge at b = 4, Llama, lambda* = 0.05: excess 0.0845 against 0.0822 at
b -> inf, so the quantization cost is q(4) = +0.0023 nats.

Small enough to be encouraging (4-bit quantization is nearly free here) and
small enough to be a warning for the slope fit: if the -2 exponent holds,
q(8) would be around 9e-6, well below cell-to-cell noise. The fit may end up
resting on b in {2, 3, 4}. Worth checking before reading the slope.

## Control: the harness has not drifted

TA at alpha = 0.25 *is* the published default, and it reproduces Table 1's TA
row to 0.0007 (0.2139 vs 0.2132). Every comparison above is therefore against
a live, verified baseline rather than a remembered number.

---

## In flight

| job | what | cells | status |
|---|---|---|---|
| 45691 `rdm_w5rs` | downstream re-score | 144 | running |
| 45692 `rdm_w1a` | TA alpha + renorm + rank16 | 52 | running |
| 45693 `rdm_a1tr` | independent-init training | 16 | queued (3-job cap) |

Then: A2 (12 cells), W3 (24), A1 stage 2 geometry (CPU), A1 stage 3 matrix (28).

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

## Result 5 (W1/V1): THE BIG ONE. A tuned scalar on TA matches the method on Llama

Seed-1 matched, identical adapters, 19/52 W1 cells in (Llama complete, Mistral
all but alpha=1.0, Qwen and Yi pending):

| base | TA @ alpha=0.25 (paper's default) | best TA | at alpha | rd-ridge | TIES | gap | reading |
|---|---|---|---|---|---|---|---|
| Llama-3.1 | 0.2139 | **0.0839** | 0.75 | 0.0823 | 0.1471 | +0.0016 | **TIE within seed noise** |
| Mistral-7B | 0.1328 | 0.0631 | 0.50 | 0.0473 | 0.0494 | +0.0157 | rd-ridge wins |

Llama sweep: `0.10=0.3181  0.15=0.2779  0.25=0.2139  0.35=0.1642  0.50=0.1184
0.75=0.0839  1.00=0.1025`
Mistral sweep: `0.10=0.3575  0.15=0.2455  0.25=0.1328  0.35=0.0875  0.50=0.0631
0.75=0.0965`

### What this means

**On Llama-3.1, the paper's flagship base, one tuned scalar on Task Arithmetic
reproduces the entire method.** 0.0839 vs 0.0823 is a gap of 0.0016 against a
seed-noise floor of about 0.004, i.e. a tie.

The paper's headline is "rd-encoder ridge attains 0.094, **57% below TA**
(0.220)". That 57% is measured against TA pinned at alpha = 1/T = 0.25. Against
TA at its own optimum the reduction is roughly **2%**. The margin was the merge
coefficient, not the rate-distortion construction.

Knock-on effects, all on Llama:
- §6.2 finding (3), "only TIES separates from the structure-blind cluster":
  tuned TA (0.0839) beats TIES (0.1471) outright. TA is not structure-blind
  once its one free parameter is set.
- Figure 2's salvage arc (0.340 -> 0.094) is measured against the same
  undertuned TA reference line.
- R1 "Default to rd-encoder ridge" has no margin left on this base.

**What survives.** rd-ridge still wins on Mistral by 0.0157, about 4x seed
noise, which is a real effect. Qwen and Yi are pending. So the honest current
claim is "ties a tuned TA on one base, beats it on another, two unknown".

**The fairness framing that is defensible.** Both methods get exactly one
tuned scalar: rd-ridge gets lambda, TA gets alpha. Under that symmetric rule
the comparison is legitimate, and the paper must report it. What is not
defensible is a swept lambda against a pinned alpha, which is what shipped.

**Both W1 sub-results now agree.** V2 said the norm is load-bearing (renorm to
TA's norm costs +0.14 nats). V1 now says TA at the matching norm gets there on
its own. The two are the same finding seen from opposite directions: on Llama,
scale is doing most of the work.

### FINAL, all four bases (2026-08-02 22:25)

| base | TA @ 0.25 | best TA | alpha\* | rd-ridge seed1 | rd-ridge 3-seed | TIES 3-seed | gap vs seed1 | gap vs 3-seed |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1 | 0.2139 | 0.0839 | 0.75 | 0.0823 | 0.0945 | 0.1539 | +0.0016 | **−0.0106** |
| Mistral-7B | 0.1328 | 0.0631 | 0.50 | 0.0473 | 0.0441 | 0.0527 | +0.0157 | +0.0190 |
| Qwen-2.5 | 0.1041 | 0.0159 | 0.50 | 0.0100 | 0.0097 | 0.0132 | +0.0058 | +0.0061 |
| Yi-1.5 | 0.0990 | 0.0530 | 0.50 | 0.0356 | 0.0366 | 0.0464 | +0.0175 | +0.0164 |

**Verdict: W1 upheld on 1 of 4 bases.** Below the pre-registered ">= 2 bases"
threshold, so the contribution does *not* need wholesale restating. rd-encoder
ridge survives as a method on Mistral, Qwen and Yi. But Llama is the flagship
base, and against the paper's own published 3-seed rd-ridge number a tuned TA
**wins there by 0.0106**, about 2.5x the seed-noise floor.

### The margin claims do not survive anywhere

| base | paper: rd-ridge below TA@0.25 | honest: rd-ridge below *tuned* TA |
|---|---|---|
| Llama-3.1 | 56% | **−13%** (worse) |
| Mistral-7B | 67% | 30% |
| Qwen-2.5 | 91% | 39% |
| Yi-1.5 | 63% | 31% |

TA's optimum is alpha = 0.50 on three bases and 0.75 on Llama; the paper's
fixed 1/T = 0.25 is undertuned on **all four**. Every "X% below TA" figure in
the paper is inflated by that choice. The abstract's "lowest worst-task loss of
the ten methods we benchmark, on all four base models" becomes "on three of
four, by 30 to 39% rather than 56 to 91%".

### "Only TIES separates from the structure-blind cluster" (§6.2 finding 3)

| base | tuned TA | TIES | holds? |
|---|---|---|---|
| Llama-3.1 | 0.0839 | 0.1539 | **NO, tuned TA beats TIES** |
| Mistral-7B | 0.0631 | 0.0527 | yes |
| Qwen-2.5 | 0.0159 | 0.0132 | yes |
| Yi-1.5 | 0.0530 | 0.0464 | yes |

TA is not "structure-blind"; it has one free parameter the paper never turned.
On Llama, turning it beats the method the paper says is the only separator.

### The defensible reframe

The paper's App. G argument is "a single globally-fixed lambda = 0.13 beats TA
and TIES on all four bases". The symmetric statement for TA is a single
globally-fixed alpha. At alpha = 0.50 (the modal optimum):

| base | TA @ 0.50 | rd-ridge @ lambda = 0.13 |
|---|---|---|
| Llama-3.1 | **0.1184** | 0.1250 |
| Mistral-7B | 0.0631 | 0.0441 |
| Qwen-2.5 | 0.0159 | 0.0097 |
| Yi-1.5 | 0.0530 | 0.0366 |

Even under the symmetric global-constant rule, rd-ridge loses Llama and wins
the other three. So Llama is where the method is weakest relative to TA under
*every* framing, not just the per-base-tuned one. Whatever the paper claims, it
cannot claim it on Llama, which is exactly the base Figure 2's salvage arc and
the 0.22 -> 0.094 headline are built on.

### Caveats before this is written into the paper

1. Seed-1 only. Needs the 3-seed TA-tuned comparison before any claim is
   restated in either direction.
2. best-alpha TA is selected on the same worst-task NLL it is scored on, which
   is exactly the objection the paper answers for lambda in App. G. The same
   held-out-alpha treatment is now owed to TA.
3. The alpha response is U-shaped with an interior optimum on both bases
   (0.75 on Llama, 0.50 on Mistral), so these are genuine optima, not grid
   edges. Unlike the lambda sweep on Mistral/Qwen/Yi, which sat at the edge.

## In flight

| job | what | cells | status |
|---|---|---|---|
| 45691 `rdm_w5rs` | downstream re-score | 144 | running |
| 45692 `rdm_w1a` | TA alpha + renorm + rank16 | 52 | running |
| 45693 `rdm_a1tr` | independent-init training | 16 | queued (3-job cap) |

Then: A2 (12 cells), W3 (24), A1 stage 2 geometry (CPU), A1 stage 3 matrix (28).

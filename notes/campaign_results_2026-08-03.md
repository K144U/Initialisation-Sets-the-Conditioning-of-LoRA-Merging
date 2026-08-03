# Campaign results, 2026-08-03

The 276-cell review-response campaign launched on 2026-08-02 completed at
**15:18 IST on 2026-08-03**. All stages hit target:

```
w1=52/52  w1s=8/8  w5=144/144  a1tr=16/16  a2=12/12  w3=24/24  a1ev=28/28
```

This note records every analyzer output verbatim, states what each result means,
and lists the claims it overturns. It supersedes the live numbers in
`campaign_results_2026-08-02.md`, which were mid-flight.

Every decision rule quoted below was fixed **before** its cells ran. None were
re-tuned after seeing results. The one exception is the 28-cell A1 merge matrix,
which had no pre-registered rule and is handled separately in
`prereg_a1_matrix_2026-08-03.md`.

Analyzers were run with `.conda/envs/rdmerge/bin/python` under
`PYTHONNOUSERSITE=1`. A non-interactive ssh login has no `module` command.

---

## Summary table

| stage | question | verdict |
|---|---|---|
| W1s | does a tuned TA beat the method on Llama at 3 seeds? | **tie**, method survives 3 wins / 1 tie / 0 losses |
| A2 | was "KnOTS ~ TA" a measurement? | **no**, it was an implementation identity |
| W3 | does the rate exponent hold on real adapters? | **no**, falsified, flat from b=2 |
| A1 | is the floor-zero regime real, or a shared-init artifact? | **real**, for independently initialised cohorts |
| W5 | do the downstream numbers survive fixed scorers? | **no**, all void; correlation not measurable at n=5-6 |

---

## W1s: tuned Task Arithmetic at three seeds

Pre-registered rule: `d = TA3(alpha*) - rd3` on Llama, with `d < -0.005` forcing
an abstract and Figure 2 rewrite, `|d| <= 0.005` a tie, `d > +0.005` a four-base
win.

```
base             a*  TA(a*) 3-seed  n  rd-ridge 3-seed  TIES 3-seed        d  verdict
llama31_8b     0.75         0.0910  3           0.0945       0.1539  -0.0035  tie
             TA per-seed: 0.0839  0.0964  0.0926   (sd 0.0064)
mistral_7b     0.50         0.0664  3           0.0441       0.0527  +0.0223  rd-ridge wins
             TA per-seed: 0.0631  0.0676  0.0685   (sd 0.0029)
qwen25_7b      0.50         0.0159  3           0.0097       0.0132  +0.0062  rd-ridge wins
             TA per-seed: 0.0159  0.0155  0.0165   (sd 0.0005)
yi15_9b        0.50         0.0541  3           0.0366       0.0464  +0.0175  rd-ridge wins
             TA per-seed: 0.0530  0.0546  0.0548   (sd 0.0009)

  tuned TA wins 0, ties 1, rd-ridge wins 3 of 4.
  Honest margin of rd-ridge below a TUNED TA:
    llama31_8b -4%,  mistral_7b 34%,  qwen25_7b 39%,  yi15_9b 32%
```

**The 2026-08-02 alarm was a seed artifact.** Llama's three TA seeds are 0.0839,
0.0964, 0.0926. The original comparison used 0.0839, the *best* of the three,
against a 3-seed rd-ridge mean. That mismatch produced the "tuned TA wins by
-0.0106" finding that drove the previous handoff's bottom line. On matched
3-seed means the gap is -0.0035, about one standard error (sd 0.0064, n=3,
SE ~ 0.0037). It is a genuine tie in both directions: not a loss, and not
hidden evidence for rd-ridge.

Supersedes: `HANDOFF_2026-08-03.md` section 3.1 table, and its section 7 claim
that the method "ties or loses on the fourth" base. Resolved to **ties**.

Unaffected: section 6.2 finding (3), "only TIES separates from the
structure-blind cluster", still **fails on Llama**, where tuned TA at 0.0910
beats TIES at 0.1539.

Still to fix in the paper: the reported **56-91%** margins were measured against
TA pinned at `1/T = 0.25`, which is undertuned on all four bases (optima are
0.50 to 0.75). The honest figures are **32-39% on three bases and a tie on the
fourth**.

---

## A2: KnOTS with a working inner merge

```
base           KnOTS(pub)        TA      TIES  KnOTS-TIES    vs TA   n  verdict
llama31_8b         0.2194    0.2196    0.1539      0.1935  -0.0261   3  differs from TA
mistral_7b         0.1396    0.1399    0.0527      0.0722  -0.0677   3  differs from TA
qwen25_7b          0.1075    0.1074    0.0132      0.0427  -0.0647   3  differs from TA
yi15_9b            0.0996    0.0996    0.0464      0.0476  -0.0520   3  differs from TA

  published |KnOTS - TA|:  llama 0.00014,  mistral 0.00031,
                           qwen  0.00009,  yi      0.00003  nats
  K1/K2: KnOTS-TIES differs from TA on 4/4 bases at the 0.005 nat threshold.
```

The shipped KnOTS is algebraically Task Arithmetic: `Delta_t V V^T = Delta_t`
cancels under `inner_combination="linear"`. With a real inner merge it separates
from TA on 4 of 4 bases, two orders of magnitude larger than the published
difference.

The four paper sites citing "KnOTS ~ TA" as evidence **for** the theory must be
rewritten. The evidence was a no-op. Worse for the theory than a null: the claim
being supported was that a subspace-alignment method lands in the structure-blind
cluster, and once KnOTS actually runs, it does not.

Silver lining: KnOTS-TIES is a genuine baseline, sitting between TA and TIES, and
rd-ridge beats it on all four bases (0.0945 vs 0.1935 Llama, 0.0441 vs 0.0722
Mistral, 0.0097 vs 0.0427 Qwen, 0.0366 vs 0.0476 Yi).

**OPEN AND LOAD-BEARING.** It is not established that this is a defect in the
*published* KnOTS method rather than a misconfiguration in our reimplementation.
In the published method the SVD alignment is meant to be followed by a real inner
merge such as TIES, which would make `inner_combination="linear"` our error. The
official implementation has **not** been checked. Until it is, A2 must be written
as an erratum about our own baseline, not as a finding about KnOTS. This
determines whether the campaign yields one field-level contribution or two.

---

## W3: rd-encoder ridge at finite rate

Pre-registered rule: slope in `[-2.4, -1.6]` on at least 3 of 4 bases means the
exponent holds on real adapters.

```
base               b=1      b=2      b=3      b=4      b=8     b=16    b=inf    slope     R2  verdict
llama31_8b       0.633    0.097    0.080    0.085    0.082    0.082    0.094       --     --  pending
mistral_7b       0.211    0.050    0.048    0.047    0.047    0.047    0.044    -0.10  0.433  OFF
qwen25_7b        0.075    0.011    0.010    0.010    0.010    0.010    0.010    -0.21  0.436  OFF
yi15_9b          0.210    0.041    0.036    0.036    0.036    0.036    0.037       --     --  pending

  fit over b in [2,3,4,8]; b=1 excluded (clipping-dominated).
  VERDICT: slope in [-2.4,-1.6] on 0/2 bases fitted.
```

**Falsified.** Zero of two fittable bases fall in band, at -0.10 and -0.21,
roughly an order of magnitude off. Llama and Yi could not be fitted at all
because their excess at *finite* rate falls **below** the `b=inf` value (Llama
0.082 at b=8 against 0.094 at b=inf; Yi 0.036 against 0.037), making the logged
quantity negative. On half the bases the curve does not decay toward the
asymptote at all.

What the data does say: with the ridge on, everything from b=2 up is flat. Two
bits buys the whole effect and 16 bits adds nothing. That is a real operational
finding and connects to section 6.4's TVQ b=2 dip, but it is not the theoretical
claim.

Contrast, the published lambda=0 sweep in `eval_e1/`, which never appeared in the
paper:

```
base               b=1      b=2      b=3      b=4      b=8     b=16     b=32  monotone?
llama31_8b      11.417    0.925    0.356    0.405    0.505    0.468    0.497  NO
mistral_7b      11.672    9.472    9.917    9.135    9.106    9.771   10.778  NO
qwen25_7b        9.261    1.728    0.349    0.232    0.210    0.168    0.191  NO
yi15_9b         13.537    5.420    0.524    5.035    0.613    0.251    0.376  NO
```

Non-monotone on 4 of 4, and pinned at 9.1 to 11.7 nats on Mistral.

**Caveat:** the whole W3 sweep is seed1 only. A single seed is thin ground for a
falsification, though the flatness is far too large and too consistent across
four bases to be seed noise.

Corrects the prior expectation, recorded in the previous handoff, that "b=8 may
fall below noise and the fit may rest on b in {2,3,4}". Reality is worse: the
whole curve is flat from b=2.

---

## A1: subspace geometry of the independent-init cohort

```
cohort = indep1
base             cos   |dA|   smax    soft  softFloor   hard d_eff by eps 1e-06 ... 1e-01
llama31_8b     0.047   1.42  1.107    63.2     0.0001                     64.0 ... 64.0
mistral_7b     0.047   1.41  1.106    63.2     0.0001                     64.0 ... 64.0
qwen25_7b      0.050   1.41  1.131    63.1     0.0003                     64.0 ... 64.0
yi15_9b        0.047   1.41  1.102    63.3     0.0000                     64.0 ... 64.0
```

Against the shared-init seed1 cohort measured on 2026-08-02:

| quantity | shared-init (published) | indep1 (new) |
|---|---|---|
| principal cosines, median | 0.996 | **0.047-0.050** |
| `\|dA\|/\|A\|` | 0.16-0.20 | **1.41 = sqrt(2)** |
| sigma_max | 1.999 ~ sqrt(T) | **1.10-1.13** |
| soft d_eff (of Tr = 64) | 16.3 | **63.1-63.3** |
| soft floor | 0.745 B^2 | **0.0000-0.0003 B^2** |
| hard d_eff = Tr | only at ~1e-3 sigma_1 | **stable at every eps, 1e-6 to 1e-1** |

**The floor-zero regime is real for properly initialised cohorts.** With
independent inits the task subspaces are near-orthogonal, the soft floor
collapses to essentially zero, and hard `d_eff = Tr` is stable across four
decades of threshold rather than surviving only at a loose tolerance.

That last row **dissolves reviewer W2's objection**, which was that rank is the
wrong stability class for this geometry. It was the wrong stability class *for
the degenerate cohort*. It is a perfectly stable one here.

The 2026-08-02 audit finding stands unchanged: the shipped cohort **was**
degenerate, because `train_lora.py` seeded `A` from `seeds.global` identically
across all four task configs and LoRA starts `B = 0`, so `A` barely moves. What
changes is the conclusion drawn from it. The property the paper claims is real;
the cohort used to demonstrate it could not show it.

**Consequence that is not yet resolved:** every headline experiment, W1, A2, W3,
W5 and the whole 4x7 matrix, was measured on the degenerate cohort. Whether the
method's advantage survives on properly initialised adapters is tested by the
28-cell `eval_a1_indep/` matrix, which is unread as of this writing.

---

## W5: downstream re-score with both scorers fixed

```
GSM8K em: published -> re-scored (3-seed means)
base                    ta         ties         dare        knots       tvq_b2     rd_ridge  rho pub  rho new
llama31_8b    0.315->0.778  0.256->0.775  0.317->0.783  0.323->0.771  0.299->0.745  0.201->0.663     -0.60    -0.50
mistral_7b    0.103->0.462  0.409->0.471  0.103->0.461  0.097->0.474  0.379->0.455  0.469->0.487     +0.60    +0.20
qwen25_7b     0.223->0.649  0.706->0.783  0.208->0.656  0.218->0.643  0.685->0.782  0.794->0.805     +1.00    +0.70
yi15_9b       0.763->0.783  0.783->0.782  0.751->0.783  0.759->0.785  0.769->0.774  0.791->0.789     +0.90    -0.60

  discard rate: 0.61-0.81 -> 0.00 (llama/mistral/qwen ta,dare,knots), 0.02-0.11 -> 0.00 elsewhere

HumanEval pass@1: published -> re-scored (3-seed means)
llama31_8b    0.055->0.628  0.354->0.626  0.059->0.626  0.057->0.630  0.441->0.561  0.457->0.571     +1.00    -0.90
mistral_7b    0.053->0.370  0.250->0.244  0.057->0.368  0.045->0.370  0.244->0.244  0.191->0.280     +0.60    -0.60
qwen25_7b     0.016->0.776  0.547->0.760  0.014->0.791  0.016->0.780  0.644->0.776  0.606->0.742     +0.80    -0.90
yi15_9b       0.061->0.083  0.081->0.085  0.069->0.091  0.069->0.071  0.081->0.087  0.108->0.120     +0.80    -0.20

  discard rate: 0.72-0.78 -> 0.00 (ta, dare, knots on llama/mistral/qwen)

D2  rd-ridge across-seed SD on its two published unstable cells
  llama31_8b   GSM8K em     published SD 0.074  re-scored SD 0.005  artifact
  mistral_7b   HumanEval    published SD 0.084  re-scored SD 0.049  real instability
```

**Every published downstream number is void.** Discard rates go to 0.00
everywhere. Llama's GSM8K TA cell moves 0.315 to 0.778, finally consistent with
Llama-3.1-8B-Instruct's known ability, which should have been caught when the
0.32 was first written down.

**The correlation is the serious problem, but not for the obvious reason.**
HumanEval rho flips sign on 4 of 4 bases and GSM8K becomes mixed. The tempting
reading is "the relationship inverted". That reading is wrong, and stating it
would be a mistake.

These Spearman values are computed over the **5 to 6 methods** in each row. Under
the null, `SD(rho) = 1/sqrt(n-1)`, which is **0.45 at n=6 and 0.50 at n=5**. So
`rho = -0.60` is 1.3 SD from zero, and the published `+0.60` and `+0.90` were
never evidence of anything either. W6 caught a symptom of this already: at n=5
Spearman can only take `1 - sum(d^2)/20`, so the published Mistral GSM8K `+0.67`
is not an attainable value.

The honest statement is that **this design cannot measure the correlation at
all**, in either direction. The 4-of-4 negative signs on HumanEval are
suggestive, not conclusive. The fix is more points per base, roughly 25 or more
by sweeping alpha, T, method and seed, not a different conclusion from n=6.

This removes the support for the 2026-06 reframe, which made "a 1-CPU-minute
audit that predicts when merging fails" the centerpiece on external advice.

Corrects the previous handoff, which attributed the "unexplained" Llama
`rho = -0.60` to the scorer bug. It survived re-scoring at -0.50, so that
explanation was wrong.

Also: `_strip_humaneval_completion` and the GSM8K extractor are **our** code in
`downstream_metrics.py`. These are local bugs. "We had two scorer bugs and fixed
them" is a disclosure, not a contribution, and must not be written up as a
field-level finding.

---

## Corrections to claims made earlier in this project

- Previous handoff, section 7: "the theory is untouched and still stands." **No
  longer accurate after W3.** The rate exponent does not appear on real adapters.
  The mathematics may be sound with the regime simply unreachable, since
  everything is flat from b=2, but a prediction that cannot be observed is a
  weakness in a paper claiming operational relevance.
- Previous handoff, section 3.1: Llama "tuned TA wins, -0.0106". **Superseded**,
  it is a tie at -0.0035 on matched 3-seed means.
- Previous handoff, section 3.1: the Llama `rho = -0.60` outlier is "likely an
  artifact" of the scorer. **Refuted**, it survives at -0.50.
- 2026-08-02 audit finding A1: the shared init "inverts the headline regime
  diagnosis." **Half right.** The cohort was degenerate, but the regime claim
  itself is correct and now properly supported.
- My own ETA estimates during this campaign were wrong in both directions, too
  pessimistic on W5 (predicted 4-5.3 cells/h from an overnight window, actual
  ~12/h) and too optimistic on W1s. Per-cell cost varies several-fold with base
  size and node contention; do not plan tightly on extrapolated rates.

## Operational notes recorded during the campaign

- The PBS `gpu` queue enforces **2 running jobs per user**, not the 3 documented
  in `CLAUDE.md` section 5. Evidence: `comment = Not Running: User has reached
  queue gpu running job limit.` on a queued job while two ran and the node had
  2 TB of memory and 26 of 96 CPUs free. The keeper's `MAXJOBS=3` counts queued
  plus running, so it stays within policy, but only two stages progress at once.
- Cells ran on GPUs 1, 2, 4 and 6, which confirms the GPU override is live and
  that `CLAUDE.md` section 5's "only GPUs 2, 4, 6, don't add GPU 0" is stale.
- Each eval cell holds 18 to 21 GiB, consistent with the 25 GiB free-VRAM gate.
- The campaign keeper **exits deliberately** when all stages reach target. A
  liveness check that does not account for this reports a false alarm.
- The VPN link to `CLUSTER-HOST` dropped three times during the day, for 100
  minutes on one occasion. The keeper and PBS jobs are unaffected, since both run
  on the cluster. Diagnostic: if `ping 8.8.8.8` succeeds and the routing table
  has no `172.16.x` route, the VPN is down, not the cluster.

---

# STEP 0: the A1 merge matrix

Run 2026-08-03 under the rules fixed in `prereg_a1_matrix_2026-08-03.md`, which
was committed at `8dad101` **before** any cell was read. Analyzer:
`code/phase3/scripts/analyze_a1_matrix.py`. Output:
`results/phase3/a1_matrix_summary.json`.

## The matrix, worst-task NLL excess

```
shared_seed1 (degenerate cohort, the one every published result used)
base          task_arithm        ties        dare       knots      tvq_b2    rd_ridge   rd_rank16
llama31_8b         0.2132      0.1471      0.2127      0.2130      0.1013      0.0823      0.0857
mistral_7b         0.1325      0.0494      0.1342      0.1320      0.0537      0.0473      0.0497
qwen25_7b          0.1036      0.0130      0.1066      0.1040      0.0192      0.0100      0.0104
yi15_9b            0.0989      0.0449      0.1012      0.0990      0.0574      0.0356      0.0361

indep1 (properly initialised cohort)
llama31_8b         0.2221      0.1351      0.2228      0.2214      0.1049      0.0714      0.0799
mistral_7b         0.1430      0.0543      0.1451      0.1423      0.0586      0.0575      0.0547
qwen25_7b          0.1040      0.0141      0.1067      0.1036      0.0165      0.0140      0.0143
yi15_9b            0.0960      0.0482      0.0979      0.0959      0.0524      0.0627      0.0638

delta = indep1 - shared (positive = worse on the proper cohort)
llama31_8b        +0.0089     -0.0120     +0.0100     +0.0085     +0.0036     -0.0109     -0.0058
mistral_7b        +0.0106     +0.0049     +0.0109     +0.0103     +0.0049     +0.0102     +0.0049
qwen25_7b         +0.0004     +0.0011     +0.0000     -0.0004     -0.0027     +0.0039     +0.0039
yi15_9b           -0.0029     +0.0033     -0.0033     -0.0031     -0.0050     +0.0272     +0.0278
```

## Q1: WEAKENED

rd-ridge against the best of the five baselines: **1 win, 2 ties, 1 loss.**

```
base           rd_ridge   best baseline    name       gap  result
llama31_8b       0.0714          0.1049  tvq_b2   +0.0334  WINS
mistral_7b       0.0575          0.0543    ties   -0.0032  TIES
qwen25_7b        0.0140          0.0141    ties   +0.0001  TIES
yi15_9b          0.0627          0.0482    ties   -0.0145  LOSES
```

Against TIES specifically, the rd family goes from **3 wins and 1 tie** on the
degenerate cohort to **1 win, 2 ties and 1 loss** on the proper one. The same
pattern holds for both rd_ridge and rd_rank16, so it is not a rank artifact.

Pre-registered consequence, applied: report rd-encoder ridge as **competitive
rather than winning**, with cohort dependence part of the claim. On properly
initialised adapters it is clearly best on Llama and indistinguishable from TIES
elsewhere. A reviewer will reasonably observe that TIES is simpler.

## Q2: RANKINGS CHANGE MATERIALLY (with a caveat about the rule)

top-1 changes 2 of 4 (Mistral, Yi), top-3 set changes 1 of 4 (Yi).

**The rule as written is weak and should be read with this in mind.** It counts a
top-1 change without requiring the swap to exceed the 0.005 threshold. Of the two
changes it counts:

- **Yi is decisive.** Shared: rd_ridge 0.0356 clearly best, ahead of ties 0.0449
  by 0.0093. indep1: ties 0.0482 clearly ahead of rd_ridge 0.0627 by 0.0145. Both
  gaps exceed threshold, and the top-3 set changes too (rd_rank16 out, tvq_b2 in).
  This is a genuine reversal.
- **Mistral is a noise-level swap.** rd_ridge leads ties by 0.0021 on shared and
  trails by 0.0032 on indep1. Both are inside the 0.005 threshold, so the
  ordering was never resolved in either cohort.

Per the pre-registration the rule is not being changed after the fact. But the
verdict rests on **one decisive case plus one coin flip**, and any paper claim
must say so.

**The cleaner statement, labelled post hoc:** the rd family is the most
cohort-sensitive method in the set. On Yi it degrades by +0.027 nats when the
initialisation is fixed while all five baselines move within ±0.005, a five-fold
larger shift concentrated entirely on the rd methods. Qwen shows the same sign at
+0.0039. This is descriptive, not a pre-registered finding.

## Q3: RANK IS IMMATERIAL

```
base           rd_ridge   rd_rank16      diff
llama31_8b       0.0714      0.0799   +0.0085   rank16 worse
mistral_7b       0.0575      0.0547   -0.0029   within threshold
qwen25_7b        0.0140      0.0143   +0.0003   within threshold
yi15_9b          0.0627      0.0638   +0.0011   within threshold
```

Within threshold on 3 of 4. **Audit finding A3's confound is not borne out** and
comes off the fix list. Llama's +0.0085 exceeds 0.005 but sits inside its own
2 sd provisional band of 0.013.

## Defect in the first run, disclosed

The first execution resolved the shared-cohort `rd_ridge` cell to
`eval_ridge_seed/`, which is **Llama-only**, so rd_ridge was silently absent from
Q2 on three bases (`n=6`). Fixed by adopting the same fallback chain
`w1_verdict_3seed.py` already uses (`eval_seed_rdridge_regmean/` first). This
**adds** an erroneously excluded method and changes no threshold.

The correction materially improved the result's integrity rather than its
favourability: under the buggy run the two top-1 changes were Qwen (a 0.0002 nat
coin flip) and Yi. With rd_ridge restored, Qwen shows **no** top-1 change and
Mistral takes its place. The verdict is unchanged.

## Standing limitations

One seed per indep1 cell, so no seed statistics are possible. Llama's measured
per-seed sd on the shared cohort is 0.0064, which exceeds the Mistral (-0.0032)
and Qwen (+0.0001) gaps outright. `lambda*` was tuned on the shared cohort, so
rd_ridge is arguably handicapped on indep1; that argues for a follow-up sweep,
not for adjusting anything now. Seed replication (indep2, indep3) is queued.

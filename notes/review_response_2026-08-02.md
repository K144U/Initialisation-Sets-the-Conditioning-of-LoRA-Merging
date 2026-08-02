# Response triage: professor's review, 2026-08-02

Verdict per weakness, with evidence run against the local adapters and results.
Companion to `audit_2026-08-02_code_and_claims.md`, which found five further
problems the review does not reach.

Legend: **UPHELD** = correct as stated. **UPHELD, WRONG MECHANISM** = the
requested fix is right, the stated cause is not. **ANSWERED** = settled by data
we already have. **CONFIRMED SERIOUS** = worse than the reviewer suspected.

---

## W1. Tuned merge coefficient. UPHELD (demand), WRONG MECHANISM (diagnosis)

The reviewer's algebra is right in the limit and wrong at the operating point.
Reconstructing $W^\star(\lambda)$ exactly as `rd_encoder.py` does (Llama seed1,
8 layers; $W^\star$ depends only on $\bar H$, so a QR basis of
$\mathrm{rowspace}(A_t)$ reproduces it):

| $\lambda$ | $\cos(W^\star,\mathrm{TA})$ | $\|W^\star\|/\|\mathrm{TA}\|$ | best $\alpha$ | residual after removing $\alpha\cdot$TA |
|---|---|---|---|---|
| 0.001 | 0.164 | 16.02 | 2.61 | 0.986 |
| 0.01 | 0.308 | 6.05 | 1.87 | 0.951 |
| **0.05 (λ\*)** | **0.540** | **2.44** | **1.31** | **0.841** |
| 0.10 | 0.705 | 1.60 | 1.12 | 0.708 |
| 0.20 | 0.858 | 1.11 | 0.95 | 0.510 |
| 0.30 | 0.919 | 0.92 | 0.85 | 0.390 |
| 1.00 | 0.990 | 0.52 | 0.52 | 0.140 |
| 10.0 | 1.000 | 0.09 | 0.09 | 0.015 |

Mistral at its $\lambda^\star = 0.13$: $\cos = 0.786$, $\alpha = 1.02$,
residual $0.618$.

So $W^\star \to \alpha\,\mathrm{TA}$ with $\alpha \to 0$ **is** the asymptotics,
exactly as the reviewer says. But at the selected $\lambda^\star$:

- $\alpha = 1.31$ (Llama), $1.02$ (Mistral): the method **amplifies**, it does
  not shrink. $\alpha$ only drops below 1 at $\lambda \approx 0.2$, well past
  the optimum.
- **62 to 84 percent of $W^\star$'s Frobenius mass is orthogonal to every
  rescaling of TA.** Whatever it is doing, it is not rediscovering a scalar.
- The excess is *worst* exactly where the method most resembles scaled TA:
  $\lambda = 0.3$ gives $\cos 0.92$ and excess $0.191$; $\lambda = 1.0$ gives
  $\cos 0.99$ and excess $0.289$; both far above $\lambda^\star$'s $0.094$.
  Shrinkage is the failure mode, not the mechanism.

What it actually does, and this follows from audit A1: because the task
subspaces are near-coincident, $\bar H$ has roughly 16 eigenvalues near 1 and
48 slivers, and those slivers are the *inter-task difference* directions.
$(\bar H + \lambda I)^{-1}$ leaves the 16 shared directions alone and amplifies
the difference directions by up to $1/\lambda = 20\times$. rd-ridge is
"TA plus amplified task-difference components", which is a real algorithmic
idea, but it is not the one the paper describes and not one the RD theory
predicts.

**Still run the control.** TA is pinned at $\alpha = 1/T$ with no sweep while
rd-ridge gets a tuned $\lambda$; that asymmetry is indefensible regardless of
mechanism. Add to Table 1: TA with $\alpha$ swept over
$\{0.1, 0.25, 0.5, 0.75, 1.0, 1.5\}$, and the reviewer's second ablation,
rd-ridge at $\lambda^\star$ renormalized to TA's Frobenius norm. The table above
predicts both will lose, which turns W1 from a threat into the paper's best
evidence. 24 + 12 cells.

## W2. Threshold-dependence of the floor. ANSWERED, but he is right for a deeper reason

He asked for the floor as a function of $\varepsilon$ over several decades, all
four bases. Run (seed1, 8 layers/base, $B^2$ units):

| $\varepsilon$ | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|---|---|
| Llama $d_{\mathrm{eff}}$ | 64.0 | 64.0 | 64.0 | 64.0 | 60.5 | 38.4 | 20.8 |
| Llama floor | 0 | 0 | 0 | 0 | 0.0003 | 0.0026 | 0.0044 |
| Qwen floor | 0 | 0 | 0 | 0 | 0.0024 | 0.0094 | 0.0133 |

Mistral and Yi track Llama. **The hard floor is robust across four decades**
and stays negligible even at an absurd $\varepsilon = 0.1$. His specific worry,
that the floor becomes non-negligible at $10^{-2}$ or $10^{-3}$, is answered no.
Report this table; it is a clean win and costs nothing.

But the tension he identifies is real and the $\varepsilon$-sweep does not
dissolve it. The threshold-free participation ratio is **16.3 of 64** on every
base, giving a soft floor of $0.745\,B^2$. Both statements are true at once:
the subspaces are linearly independent *and* they sit at near-zero principal
angles (median cosine 0.996, audit A1). Rank is simply the wrong stability
class for this geometry. The honest reframe is not "our threshold was fine", it
is "the rank-based $d_{\mathrm{eff}}$ of Lemma 2 is insensitive to the
conditioning that actually governs any encoder, and our cohorts are
near-degenerate because all $T$ adapters share one LoRA $A$ initialization."
That last clause is ours to disclose; he has not found it.

## W3. Rate axis inert. UPHELD, and worse than he knows

He cites TVQ's non-monotonicity. The paper's **own encoder** has a real-data
rate sweep in `results/phase3/eval_e1/` ($\lambda = 0$, rank-$r$), never shown
as a rate curve in the paper:

| base | b=1 | b=2 | b=3 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|---|---|
| Llama | 11.417 | 0.925 | **0.356** | 0.405 | 0.505 | 0.468 | 0.497 |
| Mistral | 11.672 | 9.472 | 9.917 | 9.135 | **9.106** | 9.771 | 10.778 |
| Qwen | 9.261 | 1.728 | 0.349 | 0.232 | 0.210 | **0.168** | 0.191 |
| Yi | 13.537 | 5.420 | **0.524** | 5.035 | 0.613 | 0.251 | 0.376 |

Non-monotone on 4 of 4, and on Mistral the encoder is simply broken at every
rate (9 to 11 nats). No $2^{-2R/(Tr)}$ slope is visible. Caveat in our favour:
this is $\lambda = 0$, so it is contaminated by the sliver blow-up rather than
being a clean test of the rate axis. His requested experiment, **ridge at finite
$b$**, has genuinely never been run: every rd-ridge cell in the paper is
`bits=32`. Run $b \in \{1,2,4,8,\infty\}$ at $\lambda^\star$, 4 bases, 20 cells.
If the slope appears, this is a much stronger paper; if not, the abstract must
stop implying the rate machinery is doing empirical work.

## W4. Audit fit on two points. UPHELD

No defence available. Add the confound he names: on the real bases $R$ and
saturation are perfectly collinear, and our own App. D blames three artifacts on
saturation. Compounding it (audit B4), **every $T=7$ point is a single merge
cell with one training seed**, and the Yi inversion gap is 0.0197 against a
$T=4$ subset-to-subset range of 0.017 to 0.019 on the same base. The inversion
may not survive replication. Soften the abstract to App. I.6's language, run
3 extra 7-task subsets per base, and treat a fourth base as required rather than
optional.

## W5. Selection metric fails on the flagship base. LIKELY AN ARTIFACT, verify before conceding

Do not concede this one yet. The GSM8K answer extractor requires the final
number at end-of-string, so any answer closing with words scores zero. Failure
rates, 3 seeds pooled:

| | Llama | Mistral | Qwen | Yi |
|---|---|---|---|---|
| TA | 0.611 | 0.809 | 0.676 | 0.062 |
| rd-ridge | **0.693** | 0.107 | 0.025 | 0.016 |

On Llama, rd-ridge has the *highest* extraction-failure rate of six methods and
therefore the lowest EM. The paper claims the regex "succeeded in 95% of
generations in an $n = 100$ pilot"; the shipped runs contradict that on three of
four bases. Re-scoring the stored 200-char previews with last-number-anywhere
flips the ordering (rd-ridge $0.201 \to 0.323$, TA $0.315 \to 0.225$), though
those previews are truncated so this is indicative, not conclusive. Re-run the
24 downstream cells with a standard extractor, storing full generations, before
answering W5. His fallback ask (a norm-constrained variant) is worth doing
anyway: audit A3 shows the deployed rd-ridge is a rank-64 adapter against
rank-16 baselines, and W1's table shows it amplifies norm by $2.4\times$.

## W6. Numerical inconsistencies. UPHELD, exactly right

Recomputed from `eval_matrix_seeds/`. Full provenance:

| quantity | §6.2 text | Table 1 | Fig 2 / App G |
|---|---|---|---|
| source cohort | **v1 adapters** | **seed-1 only** | **3-seed mean** |
| Llama TA | 0.218 | 0.213 | 0.220 |
| Llama TIES | | 0.147 | 0.154 |

Verified 3-seed means: TA 0.2196 / 0.1399 / 0.1074 / 0.0996, TIES 0.1539 /
0.0527 / 0.0132 / 0.0464. Table 1's five matrix baselines are all seed-1, which
the caption does not say. His scepticism about "conservative" is answerable:
seed-1 is the *lowest* of the three seeds for every baseline (TA Llama 0.2132 <
0.2211 < 0.2243), so seed-1 competitors flatter the competitors and understate
rd-ridge's margin. State that explicitly rather than asserting it.

## W7. Bibliography. CONFIRMED SERIOUS. Both entries were fabricated

Verified against arXiv. The works exist; every bibliographic field was invented.

| | claimed | actual |
|---|---|---|
| `lorm2024` title | LoRM: Low-Rank Merging of Multi-Task LoRA Adapters | Closed-Form Merging of Parameter-Efficient Modules for Federated Continual Learning |
| authors | Wen, Zheng, Zhao, Liu | Salami, Buzzega, Mosconi, Bonato, Sabetta, Calderara |
| venue | NeurIPS 2024 | arXiv:2410.17961 |
| setting | multi-task LoRA merging | federated continual learning |
| `regmeanpp2025` title | RegMean++: Better Regularization for Closed-Form Model Merging | RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging |
| authors | Park, Kim, Lee, Choi | Nguyen, Huu-Tien, Suzuki, Nguyen |
| venue | arXiv preprint 2025 (no number) | arXiv:2508.03121, TMLR |
| contribution | refines RegMean's Tikhonov term | adds cross-layer dependencies |

Both entries and the §2 sentence describing them are **fixed** as of commit
`410593f`. The `note = {Anticipated 2025 refinement...}` field was the tell.

Two of two spot-checks were fabricated, so treat the whole bibliography as
unverified until each entry is checked against a real record.

### Verification log (33 entries; update as they are checked)

| key | status | note |
|---|---|---|
| `lorm2024` | **was fabricated, FIXED** | real: Salami et al., arXiv:2410.17961. Wrong title, authors, venue and setting. |
| `regmeanpp2025` | **was fabricated, FIXED** | real: Nguyen, Huu-Tien, Suzuki, Nguyen, arXiv:2508.03121, TMLR. Contribution is cross-layer dependencies, not the Tikhonov term. |
| `zandieh2025turboquant` | **verified** | arXiv:2504.19874, Google + NYU. Title exact. Substantively right too: the method is random rotation then per-coordinate optimal scalar quantization, which is the construction §5 builds on. This is the load-bearing citation and it holds. |
| `tspa2025` | **real, metadata incomplete** | title exact, OpenReview `iE0dWuv6jU`. Currently `@misc` with no venue or eprint; fill them in. |
| `concurrent2026merging` | **real, self-citation** | anonymized `pathak2026merging` (Research Square, DOI `10.21203/rs.3.rs-9189872/v1`). Not a placeholder. See audit C1: it is uncited, so it anonymizes nothing, and `Anonymous / Anonymous preprint` is more conspicuous than an ordinary third-person self-citation. |
| `tara2025` | **real, key/year mismatch** | arXiv:2603.26299, "Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy". Title exact. Rename the key to `tara2026` or set `year = {2025}`; it currently claims CVPR 2026, which still needs confirming. |
| `domerging2025` | **verified** | arXiv:2505.15875, "Decouple and Orthogonalize: A Data-Free Framework for LoRA Merging". Title and eprint both exact. |
| remaining 26 | **unchecked** | priority: `arm2026streaming` (2602.03237), `panariello2025core` (2509.17786), `kim2025tvq`, `stoica2025knots`, `systematic2025merging`, `jang2025taskvectorbases`, `atm2024alternating`, `gradients2025taskvectorsgradients`. |

Pattern so far: the fabricated pair both had a hand-written `note` field
editorialising about a paper's contribution ("Anticipated 2025 refinement…",
"Successor of RegMean adapted to…"). Any remaining entry carrying such a note,
or lacking an eprint/DOI, should be checked first.

**Also surfaced while checking:** close concurrent work the paper does not cite.
arXiv:2606.03723 "Compress then Merge: From Multiple LoRAs into One Low-Rank
Adapter" merges $T$ LoRAs into one rank-$r$ LoRA using shared $r$-dimensional
subspaces computed from the LoRA weights alone. That is data-free, rank-bottlenecked,
subspace-based merging, i.e. our exact setup. Also arXiv:2607.20561 (CT-Merging,
SVD-based LoRA merging) and, per the earlier build notes, arXiv:2603.09463,
which reportedly uses a rate-distortion lens.

## W8. Anonymity. NOT CURRENTLY SCRUBBED

The bundle the paper promises does not exist yet, and the history it would be
built from carries:

```
K144U <pathaksankalp04@gmail.com>
K144U <pathaksankalp@gmail.com>
K144U <95154157+K144U@users.noreply.github.com>
sanjay.g <sanjay.g@jiit-master.cm.cluster>
origin  https://github.com/K144U/rdmerge.git
```

Two real personal emails, a GitHub handle that resolves to the author, and a
cluster hostname naming the institution. Build with `git filter-repo`
name/email callbacks, drop the remote, verify
`git log --format='%an %ae' | sort -u` shows only Anonymous, and confirm commit
`3582799` survives with its timestamp. Note config paths under `code/phase3/`
also contain `/home/sanjay.g/...`.

---

## What the review did not find

Ordered by how much they change the response. Details in the audit note.

1. **A1** all $T$ adapters per cohort share one LoRA $A$ init, so the subspaces
   are near-identical (principal cosines median 0.996). This is the cause of
   W2's tension and of the $\bar H$ degeneracy that W1 is really about.
2. **A2** `knots.py` with the default `inner_combination="linear"` is
   algebraically Task Arithmetic ($VV^\top = I$ cancels; verified
   $\max|{\cdot}| = 2.9\times10^{-6}$). The paper cites "KnOTS $\approx$ TA" as
   evidence *for* its theory in four places, including the intro.
3. **A3** headline rd-ridge is deployed at rank 64 against rank-16 baselines,
   and the $\lambda = 0$ comparison cell is rank 16, so the salvage arc is
   confounded. Rank-16 rd-ridge already exists and still wins (Llama 0.0849).
4. **A4** HumanEval harness returns an empty completion for markdown-fenced
   output: 122 to 130 of 164 empty for TA/DARE/KnOTS versus 0 to 11 for
   TIES/TVQ2/rd-ridge.
5. **B1** train/eval overlap about 14.5% (alpaca) and 10% (magicoder), because
   training shuffles with `seeds.global` and eval cells with `seed: 20260518`.
   The Reproducibility Statement's zero-overlap claim is false as written.

A2 in particular interacts with the review: it is a second instance of the
failure mode behind W7, a claim asserted without checking the artifact.

## Compute to answer the review

| experiment | cells | answers |
|---|---|---|
| TA $\alpha$-sweep + norm-matched rd-ridge | 36 | W1 |
| rd-ridge at $b \in \{1,2,4,8,\infty\}$, 4 bases | 20 | W3 |
| Downstream re-score, fixed extractors, full generations stored | 36 | W5, A4 |
| Independent-$A$-init cohort, retrain + rematrix | 16 train + 20 eval | A1, W2 |
| KnOTS with `inner_combination="ties"` | 20 | A2 |
| Rank-16 rd-ridge, Mistral and Qwen, 3 seeds | 6 | A3 |
| 3 extra 7-task subsets per base | 54 | W4 |
| Fourth base, high $R$ and low saturation | ~40 | W4 |

$\varepsilon$-sweep (W2), provenance labelling (W6), bibliography (W7) and the
anonymized bundle (W8) need no GPU.

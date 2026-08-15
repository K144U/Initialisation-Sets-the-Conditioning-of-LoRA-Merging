# Pre-registration: does shared-A initialisation occur in the wild?

Written 2026-08-15, **before any adapter has been downloaded** and before any
cohort has been measured. The commit ordering is the evidence; if any result
commit of this audit is an ancestor of this file's commit, the registration is
void and must be reported as void.

## Why this exists, and why it is not optional

The parent registration `prereg_a1_matrix_2026-08-03.md` (`0d9924b`) attached a
precondition to the benchmark-confounding claim, and the amendment
`prereg_a1_matrix_amendment_2026-08-04.md` (`d503347`) repeated it:

> before any claim about the field, PEFT's default init and at least two
> published merging benchmarks must be checked for the shared-seed pattern. Our
> own repo showing the pattern is not evidence about anyone else's.

The S3 margin-aware control (`analyze_s3_margin_control.py`, run 2026-08-15,
0 of 4 bases unstable) reinstated that claim as PROVISIONAL. The precondition
is therefore now live: it gates a claim we have a verdict on rather than a
hypothetical one. This document fixes the rules for discharging it.

Independently of the claim, the paper's motivation depends on this. Every
cohort measured so far is one we trained ourselves, in both conditions, on
purpose. That is a valid controlled experiment and it does not establish that
anyone else's adapters are affected.

## The three parts, as the precondition names them

### Part A. PEFT's default initialisation

Determine, from the PEFT source at a pinned version, the exact condition under
which two LoRA adapters trained by separate invocations receive **identical**
`A` factors. Report the version, the file, the initialisation function and the
condition, so a reader can check it against the source rather than take it from
us.

This part has no threshold and no verdict. It is a factual question about a
library and it is reported as one.

### Part B. Published merging evaluations

Identify **at least two** published merging papers or benchmarks that release
the adapter cohorts they evaluate on, and audit those cohorts specifically.
Selection is by the following rule, fixed now: work through the merging papers
cited in our related work section in citation order, and take the first two
that release adapters meeting the inclusion criteria below. If fewer than two
do, report exactly how many were checked and that the precondition cannot be
discharged on published benchmarks, rather than substituting a different corpus
and calling it the same thing.

### Part C. A sample of public LoRA cohorts

The broad prevalence estimate. Rules below.

## Sampling frame for Part C, fixed before any download

Cherry-picking is the failure mode this section exists to prevent, so the
selection rule is mechanical and is fixed here.

1. Query the HuggingFace API for models with `library_name=peft`, sorted by
   **downloads descending**, and walk that list in order.
2. Group candidates into **cohorts** by their `base_model` field: a cohort is
   two or more adapters declaring the same base model and published by the same
   namespace (user or organisation).
3. Take cohorts in the order the walk encounters them, and stop at the first
   **50** that satisfy the inclusion criteria, or when 2000 models have been
   examined, whichever comes first.
4. **No cohort is skipped after inspection.** A cohort that meets the criteria
   is measured and reported, whatever it shows. Cohorts may only be excluded by
   the criteria below, all of which are checkable before any geometry is
   computed.

Sorting by downloads rather than by recency or by search relevance is a choice
made now and stated: it biases the sample towards adapters people actually use,
which is the population the claim is about.

### Inclusion criteria, all checkable before measurement

- At least **two** adapters in the cohort.
- All adapters declare the same `base_model`.
- Each is a PEFT LoRA with a readable weight file and a parseable
  `adapter_config.json`.
- LoRA rank $r \geq 4$, so principal angles are not degenerate.
- The adapters share at least one target module in common; geometry is measured
  only on shared modules.

### Exclusion criteria, also fixed now

- Adapters that are quantised such that `A` cannot be recovered in floating
  point.
- Cohorts where all adapters are byte-identical, which indicates a re-upload
  rather than a cohort of distinct adapters. These are reported separately as
  a count and excluded from the prevalence denominator, because they are not
  evidence about training.

## What is measured

Per cohort, for each pair of adapters and each shared target module, on the
row spaces of the `A` factors, which is the same quantity as Table 2:

1. **Median principal cosine** between $\range(A_i^\top)$ and
   $\range(A_j^\top)$. Reported per cohort as the median over pairs and
   modules.
2. **Direct `A` similarity**, $\|A_i - A_j\|_F / \|A_i\|_F$. This is the more
   direct signature: if two adapters start from one `A` and `B` starts at zero,
   `A` moves little during training and the difference stays small. Our own
   shared cohorts measured $\|\Delta A\|/\|A\| = 0.16$ to $0.19$.

Both are computed by the existing one-CPU-minute audit, with no forward passes
and no evaluation data.

## Thresholds and predictions, fixed now

Anchored on our own two measured poles, which are median principal cosine
$0.995$ for shared initialisation and $0.047$ for independent:

- A cohort is **COLLAPSED** if its median principal cosine exceeds **0.9**.
- A cohort is **INTERMEDIATE** if it falls in $[0.5, 0.9]$.
- A cohort is **SEPARATED** below $0.5$.

$0.9$ is chosen because it sits far from both poles and closer to the shared
one, so a cohort clearing it is unambiguously in the regime the paper describes.
It is not tuned: no cohort has been measured at the time of writing.

**Prediction P1.** At least **20%** of included cohorts are COLLAPSED.

We state plainly that we do not know the answer and that the prediction is a
guess from one data point, namely that our own default-configured training
produced the collapsed regime without our intending it.

## What we write in each branch, fixed now

This is the part worth nothing if written afterwards.

- **P1 holds (>= 20% collapsed).** The precondition is discharged. The
  benchmark-confounding claim, reinstated as provisional by S3, is stated in
  the paper as provisional **and** supported by a prevalence figure, with the
  figure and the sampling frame given in the abstract and Section 1. The audit
  becomes a headline contribution rather than a diagnostic offered on faith.
- **Some but fewer than 20% collapsed.** The claim is stated as scoped: the
  regime occurs in public cohorts at the measured rate, which is reported
  exactly, and is not claimed to be typical. The abstract gives the rate.
- **No cohort collapsed.** The benchmark-confounding claim **stays withdrawn**,
  and the paper says in Section 1 and in the limitations that we looked for the
  regime in $N$ public cohorts and did not find it, giving $N$ and the frame.
  The paper is then explicitly about a failure mode we induced deliberately and
  can detect cheaply, with its prevalence unknown and reported as unknown. We do
  not retreat to "it could still be common in unreleased cohorts" as though that
  were evidence.
- **Fewer than 10 cohorts meet the inclusion criteria.** The audit is reported
  as inconclusive on sample size, with the number examined and the number
  excluded by each criterion, and the precondition is **not** treated as
  discharged. We do not relax a criterion to reach a sample size.

## Reported regardless of outcome

1. The **full distribution** of median principal cosine across included
   cohorts, as a figure, not only the fraction above threshold.
2. The **number examined, included and excluded**, with excluded counts broken
   down by criterion.
3. The **identity of every cohort measured**, so the audit is reproducible by a
   reader against public artifacts.
4. Cohorts whose adapters are byte-identical, counted separately as described.
5. Both measurements, principal cosine and direct `A` similarity, including any
   case where they disagree.

## Binding constraints

1. No threshold, criterion or sampling rule above may be changed after any
   cohort has been measured. If a rule turns out to be badly specified, that is
   recorded as a limitation and the result is reported under the rule as
   written.
2. The sampling frame is walked in order. Cohorts are not reordered, filtered by
   name, or selected for being interesting.
3. The analyzer is committed **before** any adapter is downloaded.
4. If the audit contradicts the paper's motivation, it is reported in the
   abstract and Section 1, not in an appendix.
5. This audit cannot on its own reinstate anything. S3 reinstated the claim as
   provisional under a rule registered at `0c8b9e8`; this audit discharges a
   precondition. Both facts are reported together, with S3 identified as the
   secondary control it is.

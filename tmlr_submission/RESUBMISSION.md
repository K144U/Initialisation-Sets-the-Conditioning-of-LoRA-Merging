# Resubmission notes for OpenReview

TMLR requires a revised version of a rejected submission to be entered as a
**new submission**, carrying a link to the previous one and a description of
what changed. Both go in the submission form, not in the paper.

---

## Before you submit: one thing to check

**Confirm submission 9529 was never de-anonymised.** Revealing author identity
on a rejected TMLR submission precludes submitting a revised version. That
means checking, on 9529's OpenReview page, that no author names were added, no
de-anonymising comment was posted by an author, and no camera-ready or
non-anonymous PDF was uploaded at any point. If any of that happened, stop and
write to the editors before submitting rather than after.

Nothing in this repository can tell you the answer; it is a property of the
OpenReview record.

---

## Link to the previous submission

Paste 9529's OpenReview forum URL into the "previous submission" field.

---

## Description of changes

Suggested text. It is written around the pivot, because that is the honest
summary and it is also the strongest thing we can say.

> This is a substantially rewritten version of submission 9529, which was
> rejected for dense prose that obscured the rationale. The central change is
> that the paper now leads with an empirical finding rather than with a
> theorem.
>
> **What the paper claims has changed.** The previous version's headline was a
> rate-distortion lower bound on model merging, whose floor term we reported as
> separating two cohorts by a factor of sixty. That was an error: we had
> substituted a soft participation-ratio surrogate into a formula derived for a
> rank. The floor is exactly zero on all four base models under every
> initialisation we tried, and the term is vacuous for LoRA merging as
> practised. The bound is now reported as provenance, in an appendix, and it is
> explicitly not a contribution.
>
> **What replaced it.** The quantity that does separate the cohorts is the
> conditioning of the operator built from the task subspaces, and what controls
> that is whether the adapters of a cohort shared one random initialisation of
> the LoRA A factor. This is measurable from the adapter files in about a CPU
> minute, with no forward passes and no evaluation data. Sharing the draw moves
> the median principal cosine between task subspaces from 0.05 to 0.99. The
> effect is invisible to merging methods that average or sparsify, and decisive
> for methods that solve a linear system, where one Tikhonov term is the
> difference between a working merge and a model that scores zero on GSM8K and
> HumanEval.
>
> **New experiments, all pre-registered before the compute ran.** A prevalence
> audit of public LoRA cohorts under a sampling frame fixed before any
> download, finding 17 of 45 collapsed. A replication of the method comparison
> on three independently initialised cohorts. The same conditioning test on
> RegMean, a solver we did not design, plus a measurement of what RegMean's own
> shipped default regularisation does. Downstream accuracy for the conditioning
> effect, and a further 80 cells replicating the null in accuracy rather than in
> loss. Eleven pre-registration documents govern these, and the anonymised git
> bundle in the supplementary material lets a reviewer verify by commit
> ordering that each was written before the compute it governs.
>
> **Claims withdrawn.** Our own encoder's advantage does not survive the move to
> properly initialised cohorts, and we say so. The claim that published merging
> benchmarks are confounded by initialisation geometry is withdrawn and stays
> withdrawn. A previously reported falsification of the rate exponent rested on
> a mismatched reference and is withdrawn; the corrected sweep turns out to lack
> the dynamic range to measure the exponent at all. Section 6 collects these.
>
> **Presentation.** The paper is reorganised so that the affirmative case runs
> first and the corrections are collected rather than scattered. The
> introduction and abstract carry the motivation in plain language, with the
> formal machinery moved to the appendices.

---

## What to upload

| field | file |
|---|---|
| paper PDF | `paper.pdf` (or compile `overleaf_tmlr.zip` on Overleaf) |
| supplementary | `supplementary.zip` |

`supplementary.zip` contains the anonymised git bundle and the eleven
pre-registration documents. It is 23 MB.

---

## Anonymity

The build is the anonymous form: `\usepackage{tmlr}` with **no option**. Do not
add `[accepted]` or `[preprint]` before acceptance. The source was scanned for
author names, emails, the repository URL, institutional strings, cluster
hostnames, absolute paths and timezone offsets; the bundle's history is
rewritten and its commit timestamps normalised to UTC.

# Pre-registration: merging the audited public cohorts

Written 2026-09-23, before any public cohort has been merged. Governs
`code/phase3/scripts/gen_public_merge.py`, `code/phase3/eval/run_behaviour_cell.py`
and `code/phase3/scripts/analyze_public_merge.py`, committed after this file.

## Why

The prevalence audit (Part C of `prereg_prevalence_2026-08-15.md`) shows that
17 of 45 public cohorts are collapsed. That is a prevalence with no measured
consequence. This test merges public cohorts, collapsed and separated, and
measures whether the failure the paper predicts actually happens in them.

## Selection, mechanical, fixed now

From the 45 scored cohorts of `results/phase3/prevalence/prevalence_summary.json`,
in the order listed there, include every cohort meeting all of:

1. the declared base is a decoder-only text model that `AutoModelForCausalLM`
   loads without `trust_remote_code` in the project environment
   (transformers 5.5.0) and is not a vision, speech, encoder or
   embedding model;
2. the base has at most 15B parameters (bf16 on one 80 GB card, with room for
   merging);
3. the base can be downloaded with the account's token;
4. every member adapter loads with PEFT on that base, and all members share the
   same rank and the same target modules (the merge operates per layer);
5. bases declared as an Unsloth mirror or a 4-bit Unsloth build are loaded from
   the full-precision upstream checkpoint they mirror.

Every cohort that passes is merged; none is dropped after inspection. A
cohort that passes 1-5 but fails at run time is reported with its error. The
regime label of each cohort is the one the audit already assigned
(collapsed > 0.9, separated < 0.5); the one intermediate cohort is reported
and excluded from both groups.

## Consequence measure (task-free, because public adapters rarely ship their data)

A fixed prompt set: the first 100 instructions of the alpaca-cleaned
evaluation slice the paper already uses (seed 20260518), the same for every
cohort, formatted with the base's chat template if it has one and as plain
text otherwise. For each member adapter t, its greedy completion y_t of at most
128 new tokens is generated from base + adapter t. For a merged model m,

    excess_t(m) = mean NLL_m(y_t | prompt) - mean NLL_t(y_t | prompt)

in nats per completion token, and the cohort's **worst-task behaviour excess**
is max_t excess_t(m). It asks how well the merge preserves each adapter's own
behaviour, which needs no task data.

Merges (all truncated to the adapters' rank): task arithmetic (mean);
adapter-only RegMean unregularised (lambda = 1e-6); adapter-only RegMean at
RegMean's shipped alpha = 0.9 (no ridge); adapter-only RegMean with the audit's
starting ridge lambda = 0.01.

## Predictions

Let F(m) = worst-task behaviour excess of merge m divided by that of task
arithmetic on the same cohort (task arithmetic is the reference because the
paper shows it is blind to the regime).

- **U1.** Unregularised RegMean has F >= 2 on at least 60% of collapsed
  cohorts and on at most 20% of separated ones.
- **U2.** At lambda = 0.01, F <= 1.5 on at least 80% of collapsed cohorts.
- **U3.** RegMean's shipped alpha = 0.9 has F >= 2 on at least 40% of collapsed
  cohorts.
- **U4.** The proportion of collapsed cohorts with F(unregularised) >= 2
  exceeds that of separated cohorts (Fisher's exact test, one-sided,
  p < 0.05), if both groups have at least 5 cohorts.

## Branches

1. **U1 and U4 hold.** The paper states that the failure is common, not only
   the regime: "on k of n collapsed public cohorts an unregularised
   adapter-built solve at least doubles the worst behaviour excess of task
   arithmetic, against j of m separated ones".
2. **U1 fails on the collapsed side.** Collapse in public adapters does not
   reliably break the solver; the paper keeps the prevalence claim and says
   the consequence was tested and appears in only k of n cohorts.
3. **U1 fails on the separated side.** The solver also fails on separated
   public cohorts, so something other than the collapse is at work there; the
   paper reports it and does not attribute those failures to the regime.
4. **U2 fails.** The fixed starting ridge is not enough on public cohorts; the
   recommendation "start at 0.01" is weakened to "sweep", and U2's counts are
   reported.
5. **U3** is reported as measured and decides whether the shipped-default
   sentence appears in the public-cohort section.

## Binding constraints

No partial reads: the analyzer reports only when every selected cohort has a
result or a recorded run-time error. The prompt set, token budget, merges and
thresholds are not changed after any cohort is merged.

# Strengthening campaign, started 2026-09-23

After the JAIR desk reject (AE: "significant data execution and presentation
issues", paper not reviewed). Input: a 13-item review the user supplied. Venue
is decided LATER; this campaign only makes the paper stronger.

## Ground rules for this campaign

- **Compute runs on `STUDENT-ACCOUNT` only.** `sanjay.g` has the user's live
  experiments: its tree is read-only (adapters, models, old results). Work tree
  on the cluster: `~/projects/rdmerge_v2` (student account). Every job sets
  `PYTHONNOUSERSITE=1` (the student `~/.local` shadows torch/peft) and keeps
  the 25 GB free-VRAM gate, because these jobs share GPUs with sanjay.g's.
- **The discipline is unchanged.** Every new empirical claim gets a
  registration in `notes/prereg_*.md`, committed before its generator, which is
  committed before any cell. No partial reads. Amendments only make our own
  predictions harder. All branches written in advance.
- The public repo stays hash-stable (39 pinned hashes). New commits on top only.
- Voice: item 12 asks to drop "the authors" constructions. The user chose
  third person on 2026-08-28; the user has now supplied a list recommending
  reversal. Done in the restructure pass (P4), flagged in the final report.

## The 13 items, mapped

| # | Item | Kind | Compute | Registration | Status |
|---|------|------|---------|--------------|--------|
| 5 | Internal inconsistencies (5 listed) | text | none | none | P1 |
| 6 | Drift theory: A_t = A0 + E_t -> cosines, kappa, amplification | theory + CPU test | CPU, drift cohort checkpoints + A0 | prereg_drift_theory | P1 |
| 1 | Real RegMean with activation Grams (X^T X), both arms | GPU | Gram collection (4 bases x 4 tasks x 2 distinct inputs/layer) + merges + NLL + downstream | prereg_regmean_activation | P2 |
| 2 | RegMean shipped alpha=0.9 end to end, GSM8K + HumanEval, both arms | GPU | 4 bases x 2 arms x 2 bench (+NLL) | same doc as 1 | P2 |
| 3 | Merge public cohorts (collapsed vs separated) + Part B | GPU + CPU audit | depends on which public cohorts share a base and have scorable tasks | prereg_public_merge | P2 |
| 4 | Fix translation adapter (shuffled stream), retrain, rerun paired cells | GPU training | 16 adapters minimum (shared + indep1 on 2 bases = 16) | prereg_translation_fix | P3 |
| 11 | Heuristics null at n=3 per arm + TOST equivalence | GPU | 5 methods x 4 bases x 4 extra cohorts x 2 bench | prereg_heuristics_tost | P3 |
| 8 | Third solver: MaTS (CG) + Fisher boundary case | GPU | as item 1 | prereg_third_solver | P3 |
| 9 | Data-free lambda rule from the audit | CPU design + GPU eval | small | prereg_lambda_rule | P3 |
| 10 | T in {2,4,8}, r in {8,16,64} on one base | GPU training + CPU geometry | training new adapters is the cost | prereg_tr_sweep | P4 |
| 7 | Broaden: PiSSA/MiLoRA/CorDA, LoRA-FA, FedSA; library defaults audit; related work | CPU audit + text | small | prereg_prevalence_amendment | P3 |
| 12 | Restructure: claims lead, audit trail to one appendix, ~25 pp, voice | text | none | none | P4 |
| 13 | Abstract to the five-step story | text | none | none | P4 (last) |

Order is by what a reviewer would use against us (Tier 1 first), adjusted so
GPU queues never sit idle while text work proceeds.

## Log

- 2026-09-23: plan written. Verified item-5 inconsistencies (see below).
- 2026-09-23, day 1, state at end of session:
  - **Item 5 done** (5 fixes; the 9/16 MDE turned out to be an arithmetic error
    in the registration itself, now disclosed in App. F.4).
  - **Item 6 done, all 5 registered predictions hold** (PT1, PT2 ratio 1.000 at
    4 checkpoints; PT3 dial slopes -2.00 / -1.01, review's 1/eps^2 form
    excluded; PT4 1.000 on Llama+Mistral, Qwen/Yi pending in
    logs/drift_theory.log; PT5 monotone). Drafts: paper/sections/drift_theory.tex,
    app_drift_theory.tex (not yet in main.tex).
  - **Execution error found and fixed**: adapter-only RegMean ran the
    Delta^T Delta fallback in every published cell (PeftModelView had no
    get_layer). Amendment 2b8fee5, fix 81fc3bf, Table 4 re-run queued (rdm_agram).
  - **Library audit (item 7b)**: Unsloth (random_state=3407, seeded before
    LoRA creation) and LLaMA-Factory (seed 42 before model load) give every
    run the same A by default. notes/library_defaults_audit_2026-09-23.md.
  - **Part B**: only KnOTS releases LoRA cohorts among 22 cited merging papers,
    so Part B cannot be discharged by its rule; KnOTS audit re-running.
  - Registrations committed: drift_theory, regmean_real (+amendment),
    heuristics_tost, third_solver, new_cohorts, public_merge, lambda_rule.
  - Running/queued on STUDENT-ACCOUNT (keeper resubmits; logs/keeper_jobs.json):
    rdm_tost0/1 (160 DS), rdm_rmr0/1 (72 NLL + 144 DS), rdm_train0/1 (32
    adapters), rdm_agram (56 NLL), rdm_third (72 NLL + 64 DS), rdm_pub (20
    public cohorts). After training: run gen_new_cohorts_eval.py, queue its
    manifest, then all analyzers.
  - Throughput: ~5 lanes (one cell per GPU, shared with sanjay.g jobs at 100%
    util); GSM8K cells ~56 min. Compute ETA ~3.5-4 days from 2026-09-23.
- 2026-09-25, results so far (registered verdicts first, exploratory after):
  - **Drift theory: branch 1**, PT1-PT5 all hold (PT4 12/12 cohorts).
  - **Table 4 re-run with the described A^T A Gram**: both originally
    registered predictions hold (ratios 4.1, 65.6, 5.8, 1.8; gain larger on
    shared 4/4).
  - **Real RegMean NLL half**: Q1 (immune to init) holds 4/4; Q1m FAILS
    (amplification 2.2-6.5, but equal across arms: it comes from the
    activation Grams, kappa up to 1e7); Q2 and Q3 hold 3/4 (Qwen misses the
    gate through one outlier cohort). Downstream half pending.
  - **Part B**: KnOTS Llama-3-8B NLI cohort COLLAPSED (0.968), ViT cohorts
    separated. One paper only, so Part B not discharged.
  - **Public-cohort merges: branch 3 (against us).** U1 fails on the
    separated side (unregularised adapter-only RegMean F >= 2 on 5/6
    collapsed AND 11/13 separated; Fisher p = 0.78); U2 fails (1/6); U3 holds
    (5/6); V4 fails (2/6). hallisky cohort recorded as a run-time error
    (vocab 128257 vs 128256), not repaired. The paper must not attribute
    public-cohort solver failures to the regime.
  - EXPLORATORY, not registered, reported as such: adapter-norm spread does
    not explain the separated-side failures (r = -0.35). In absolute terms,
    unregularised excess > 10 nats on 4/6 collapsed cohorts versus 1/13
    separated (and there task arithmetic itself is 11.9), > 1 nat on 4/6 vs
    4/13. The ratio-to-TA measure was a poor registered choice where TA's own
    excess is ~0.002-0.004. The kappa-target rule rescues 3 of the 4
    catastrophic collapsed cohorts (15.3 -> 0.009, 10.4 -> 0.058,
    11.7 -> 1.1) but not Satya-Sri (17.6 -> 11.0).
  - Infra: claims of walltime-killed jobs blocked GPUs 1/2/4 for hours; keeper
    now clears claims of dead jobs and counts only no-progress resubmits.
- 2026-09-27, new-cohort verdicts (analyze_new_cohorts.py):
  - **Part R: R1-R5 all hold.** Shared cos 0.990-0.999, independent
    0.032-0.099 at every T and r. kappa grows with T (3.9e3, 1.1e4, 3.9e4 at
    T = 2, 4, 8) and with r (5.6e3, 1.1e4, 4.1e4 at r = 8, 16, 64);
    shared amplification 17.9, 45.3, 90.4 at T = 2, 4, 8. First-order formula
    ratio 1.000 on every shared configuration. Independent amplification =
    T (1.99, 4.00, 7.91). Unregularised RegMean shared/indep 33.5, 6.0, 2.5,
    4.8, 65.0; task arithmetic |d| <= 0.008.
  - **Part P: P1, P2 hold.** PiSSA with independent seeds: cos 0.999, trained
    row spaces at 0.9996 of the base's top-r right singular vectors.
    (Amplification 1.6 on raw PiSSA factors is not comparable: raw D = BA
    includes the shared B0A0 principal part.)
  - **Rule: branch 2 (against us).** V1 fails 0/6 (the kappa-target rule is
    worse than the fixed lambda = 0.01 on every shared cohort), V2 holds 5/6
    (better than alpha = 0.9), V3 fails 4/6 by the letter (the two misses are
    the rule doing BETTER than lambda = 1e-6 by > 0.005). Paper keeps
    "start at 0.01 and sweep"; the rule is reported as tested and not better.
  - Part T pending 5 downstream cells.
- 2026-09-27, **Part T (translation fix): T1 FAILS, T2-T5 hold.** The
  shuffled-stream translation adapter now improves on the base model's
  translation NLL (Llama 11.7% / 12.0%, Mistral 5.6% / 6.4%; was -1.6% on
  Llama) but is still not the best adapter on its own task (gsm8k on Llama,
  magicoder on Mistral beat it), so by the registered first branch the
  "T = 4 nominal, 3 effective" caveat stays as written. On the fixed cohorts
  the replication holds anyway: cos 0.995/0.994 vs 0.047; RegMean 1e-6
  shared/indep 6.0 and 64.6 with larger ridge gain on shared; heuristics
  |d| < 0.02 on 10/10; encoder lambda* over lambda = 0 clears the accuracy gate
  on both bases and both benchmarks (Mistral GSM8K 0.000 -> 0.468, HumanEval
  0.000 -> 0.244, reproducing the headline zero on a cohort with a fixed
  translation adapter). These are reported as a replication, not as removing
  the caveat.
- 2026-09-28, **third solver: branch 2 (S1 fails).** TSV-M is not exposed:
  effect past the gate on Mistral only (d +0.0115, ratio 1.19), Yi in the
  opposite direction (-0.0054), Llama and Qwen none; no accuracy cell past the
  gate (S4 fails). MaTS immune 4/4 (ratios 0.93-0.99; one accuracy cell,
  Mistral GSM8K, favours SHARED by 0.074 past the binomial gate). Diagonal
  Fisher immune 3/4 (Llama +0.0119, ratio 1.08). As registered, the exposure
  rule narrows to methods that SOLVE against a matrix built from the adapters'
  factors (the encoder; adapter-only RegMean incl. its shipped default); the
  whitening case was predicted and did not hold.
- Last 14 cells (7 heuristics, 7 real-RegMean downstream) doubled up with
  reversed-order helper jobs rdm_htost / rdm_hrmr.

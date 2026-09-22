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

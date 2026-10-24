# Neurocomputing rewrite: outline for approval (2026-10-24)

Inputs: the current draft (paper/neurocomputing.tex, 18 pp), the red-team
campaign verdicts (notes/campaign_2026-10-19_redteam.md), the professor's plan
(notes/neurocomputing_presubmission_plan_2026-10-06.md). Title and authors
unchanged (the title is the user's choice). RD theory stays out.

## The claim, in one sentence

Whether the adapters of a LoRA cohort started from one random draw of the A
factor decides whether merges that solve a system built from the adapters work,
and that class includes a published data-free method, WUDI-Merging, run exactly
as its authors run it; averaging is unaffected, and a CPU-minute audit finds
the collapsed regime in 40% of public cohorts.

What changes from the draft: the professor's main objection ("both failing
methods are yours") is answered by WUDI (branch 2 of its registration, which
names this the lead result). The two methods of ours stay, now as the second
and third members of the exposed class.

## Abstract (rebuilt, <= 250 words)

1. Setting: LoRA cohorts are merged; whether they share an A draw is rarely
   reported.
2. Geometry: shared draw -> median principal cosine 0.99 vs 0.05; the init seed
   alone sets it (data order and dropout do not); Unsloth and LLaMA-Factory
   defaults, continued training from one parent, DoRA and PiSSA all produce it;
   it holds for CLIP ViT-B/32 too.
3. **Lead: WUDI-Merging as published fails on shared cohorts** (worst-task
   excess 4.7x to 48x the independent arm on four bases; GSM8K e.g. 0.03 vs
   0.70 on Llama); so do adapter-only RegMean and our encoder.
4. Averaging and sparsifying merges and data-built solvers are unaffected
   (math, code, translation; truncation tested and excluded as a cause).
5. Theory: first-order law; out-of-sample on 8 new training regimes, kappa
   within 2x on 8/8 ("predicts" now earned, per the regimes registration).
6. Audit: maximum pairwise cosine; 18 of 45 public cohorts (40%).
7. Every confirmatory test pre-registered; failures reported.

## Highlights (5, <= 85 chars each, checked by nc/check.py)

- A published data-free merge, WUDI, breaks when LoRA adapters share an init
- One shared LoRA initialisation collapses the task subspaces of a cohort
- Unsloth and LLaMA-Factory defaults produce collapsed cohorts
- Averaging is unaffected; solvers built from the adapters' factors break
- A CPU-minute audit finds the collapse in 18 of 45 public LoRA cohorts

## Section map (main text)

| # | Section | Keeps | Adds / changes |
|---|---|---|---|
| 1 | Introduction | motivation, evidence paragraph | rebuilt around WUDI; contributions list; C1 (name the exposed class precisely), C8 (why audit instead of sweeping a ridge) |
| 2 | Background | LoRA init, merging taxonomy, notation | WUDI in the taxonomy (task-update Gram); C7 related work (LMC, Git Re-Basin, model soups, LoRI, ZipLoRA); C6 ill-posed vs ill-conditioned |
| 3 | Shared initialisation collapses the task subspaces | Fig. 1, Table 1 | seed ablation (init seed alone, S1-S4); library defaults measured (L1, L2); continued training (0.997) and DoRA (0.995); ViT-B/32 geometry (0.96 vs 0.11) |
| 4 | A first-order theory | proposition, Fig. 2 | out-of-sample test on training regimes (K1-K5: kappa within 2x on 8/8, amplification within 1.25x on 7/8, rho = -0.976, t* 27x the largest shared t; 39x in the log was a slip); C2 wording; B2 replaced by K |
| 5 | Where the collapse occurs (audit) | sampling frame, Fig. 3 | Algorithm 1 restated with the maximum pairwise cosine (Part M branch); partial-collapse experiment M1 as its motivation; re-read 18/45 with 5 cohorts unreadable; B3 classification of collapsed cohorts; C9 |
| 6 | Which merges the collapse affects | Table 2, Fig. 4-6 | **opens with WUDI** (NLL and accuracy, published vs closed form vs 3k); then adapter-only RegMean and our encoder; heuristics unaffected incl. untruncated (H2: truncation excluded) and B4 statistics; vision replication (V1-V6); all-task (translation extends; IFEval collapse invisible, with reference rows); ridge selection (kappa rule, GCV, L-curve all fail); scale mismatch (G3) as an observation; B1 precision sentence; the rank-r realisation stated as it ran (randomised SVD, q = r, 11-18% from exact; norms within ~1%) |
| 7 | Discussion and limitations | practice, failed predictions, limitations | failed list grows (W3, W4, G1, G3, G4, H2, T1 on IFEval, lambda selection); the vision WUDI reversal (shared merged better there, unexplained); public failures stay unexplained (gauge excluded) |
| 8 | Conclusion | availability | D1 deposit link when the user has made it |

Appendices: A theory (as now); B registrations and commit table (7 new
registrations plus the B1 rule); C campaign detail tables (regimes,
heterogeneity, lambda selection, vision, all-task reference rows) so the main
text stays short.

## Length

User decision 2026-10-24: a longer paper, so more of the campaign stays in the
main text. The ~8,000-word short-rebuild target of 2026-10-06 is dropped. Main
text was ~6,400 words (texcount: 5,565 text + 142 headings + 677 captions);
the outline as written puts it near 9,500 to 10,500, about 22 to 24 pp in
cas-sc with Appendix C. Approved as the outline above.

## Open items that feed the text

- B1 float64 check: rerunning with exact truncation on both sides (456afe2); rule b82b81e unchanged.
- B3, B4, B5: done while writing their sections.
- User: D1 deposit, [TO CONFIRM] declarations, a bio (<= 100 words) and a
  passport-type photo for each of the three authors, public repo push (last).
- Template: Elsevier CAS bundle v2.4 (2024/05/04) in paper/nc/NCTemplate, cas-sc;
  builds in 16 s.

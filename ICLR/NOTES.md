# ICLR 2027 submission build — notes

Built 2026-07-02 from the canonical drafts in `../paper/sections/`.
Entry point: **`iclr2027_conference.tex`**. Compile on Overleaf (no local
LaTeX on this machine). Uses the ICLR **2026** style files with the running
header patched to "ICLR 2027" — swap in the official iclr2027 files when the
CFP releases them (historically a drop-in rename; update `\usepackage{...}`
and `\bibliographystyle{...}` too).

## Structure: main text (~9–10 pp target) + appendices

| New file | Built from | What changed |
|---|---|---|
| `sections/abstract.tex` | abstract.tex | Cut ~450 → ~230 words; numbers preserved |
| `sections/intro.tex` | intro.tex | Cut worked example + formal-problem para (§3 covers); contributions 7 → 5; figure path → `figures/` |
| `sections/related_work.tex` | related_work.tex | Each paragraph compressed; `\cite` → `\citep`/`\citet` |
| `sections/setup.tex` | setup.tex | Problem environment folded into prose; trimmed |
| `sections/lower_bound.tex` | lower_bound.tex | Lemmas/theorem statements **verbatim**; proof sketch compressed to one paragraph; limiting cases inline |
| `sections/achievability.tex` | achievability.tex | Theorem/remarks kept; algorithm compressed; shared-V compressed |
| `sections/experiments_main.tex` | experiments.tex + 6_2 + 6_5 + 6_6 + 6_7 (+ pointers) | One consolidated section; keeps tab:e10-baselines, tab:e6-worst, combined downstream table, fig:salvage |
| `sections/discussion.tex` | discussion.tex | CE bridge compressed; 8 limitations kept as one-liners |
| `sections/reproducibility.tex` | reproducibility.tex | Compressed; manifest table → Appendix K |
| `sections/app_synthetic.tex` | experiments.tex (synthetic ¶s) | Full detail |
| `sections/app_e5_floor.tex` | 6_3 + 6_3b | Merged; "Arm 2/3" renamed "Arm 1/2"; "Sidebar" → subsection |
| `sections/app_b2_mechanism.tex` | 6_4 | + fig:methods, fig:tvq moved here from main |
| `sections/app_tscaling_details.tex` | 6_6 (details) | Probe + slopes tables, artifacts A/B/C |
| `sections/app_downstream_details.tex` | 6_5 (details) | H1/H2/H3 falsifications, seed stability, power notes |
| `sections/app_baselines_bridge.tex` | 6_8 | E10 narrative + E11 bridge (both tables) |
| `sections/app_robustness.tex` | 6_2 (sweep, heldout, Fisher) + 6_8b | Full tuning-fairness story |
| `sections/app_multiseed.tex` | 6_9 | Full |
| `sections/app_td2.tex` | appendix_td2 | Cleaned; "5-minute CPU pass" → "a few CPU minutes" (see below) |
| `sections/app_manifest.tex` | 6_7 (table) + reproducibility (table) | Decision tree + manifest |

Checked programmatically: 0 duplicate labels, 0 undefined `\ref`/`\eqref`,
all 32 cited bib keys present, all 19 `\input` files and 5 figure files exist.

## Deliberate editorial changes (do not silently revert)

1. **Audit timing made consistent.** Old text said "1-CPU-minute audit"
   (abstract/intro) but "5-minute CPU pass" (Td2 appendix) and "10 min CPU"
   (manifest). The sign-election probe is now "CPU-only, minutes-long"
   everywhere; the *floor recipe* keeps its measured "≲1 CPU-minute per task".
   If the probe really is ≤1 min, restore the sharper claim consistently.
2. **Method-count harmonized.** "7-method matrix" (old intro) vs "ten
   methods" (abstract) → "ten methods / ten-method head-to-head" everywhere,
   matching tab:e10-baselines (9 baselines + rd-encoder ridge).
3. **"Default hyperparameters" qualifier** now attached to every
   best-on-all-bases claim, with the tuned-RegMean Yi loss ($0.033$ vs
   $0.037$) surfaced in the abstract, intro contribution 1, §6.3, and App G.
4. **All retractions preserved** (all-8-cells downstream claim; four-base
   tuned-RegMean claim). Lemma 2 floor form B²(1−d_eff/(Tr)) everywhere —
   never revert to B²/(4·d_eff).
5. **Lab-notebook artifacts stripped**: version headers, "what this changes
   for the paper", internal pre-registration plumbing. The pre-registration
   *claim* stays, now phrased against "the anonymized audit-trail bundle in
   the supplementary material" + commit `3582799`.

## Anonymity / pre-registration / Zenodo (IMPORTANT)

- The paper never links the GitHub repo or the **Zenodo record
  (posted 2026-06-02, under the author's real name)**. ICLR permits
  non-anonymous preprints (reviewers are instructed not to search), but the
  submission itself must not reference it de-anonymizingly. Do NOT cite the
  Zenodo DOI in the submission; if it must be discussed post-review, do it in
  the camera-ready.
- The Zenodo record does NOT establish the pre-registration: it predates the
  Td2 pre-registration commit (2026-06-25). The commit trail does.
- **Required before submission**: build the anonymized audit-trail bundle the
  paper now promises. Suggested:
  `git bundle create audit_trail.bundle phase3-bootstrap` after rewriting
  author fields (e.g. `git filter-repo --email-callback` / `--name-callback`
  to "Anonymous"), verify `3582799` (pre-reg) and the Mistral result commits
  survive with timestamps, and drop it in the supplementary zip. Verify no
  identifying strings (usernames, cluster paths, emails) remain:
  `git log --format='%an %ae' | sort -u` must show only Anonymous.

## 2026-07-04 compile-fix + trim session (main text now EXACTLY 10 pages)

First Overleaf compile (27 pp, refs started p13 → main ~11.5 pp) exposed
three bugs and the expected overage. All fixed locally (verified with a
local Tectonic 0.16.9 build, standalone exe from GitHub releases —
matches Overleaf's Times/Nimbus metrics once T1 fontenc is loaded):

1. **`\cellcolor` was silently undefined** → tab:downstream printed
   literal "green!18"/"red!10" in every `\best`/`\worst` cell. Cause:
   `iclr2026_conference.sty` line 18 loads `eso-pic`, which loads
   `xcolor`, so the later `\usepackage[table]{xcolor}` option-clashed
   and `[table]` (→ colortbl) was dropped. Fix: `\PassOptionsToPackage
   {table}{xcolor}` BEFORE the sty include. Do not remove.
2. **Double subscript** `w_\deff` in achievability.tex → `w_{\deff}`
   (Overleaf plowed past it; Tectonic halts).
3. **Three tables overflowed the right margin**: tab:downstream
   (restructured: Metric column → per-metric group rows),
   tab:multiseed-bootstrap (footnotesize + tabcolsep 2pt),
   tab:practical-tree (p{} columns). Zero Overfull warnings remain.
4. **`\usepackage[T1]{fontenc}` added** (also required for faithful
   local XeTeX/Tectonic builds; harmless-to-good under pdfLaTeX).

**Trims to fit the 10-page limit** (ICLR 2025/2026 CFP: 10 pp main text
strictly enforced, 11th page = desk reject; repro statement + refs +
appendices don't count; 2027 CFP still unreleased — re-check):
- All four designated targets applied: intro lens+regime merged; Yi
  alpaca-OUT rows moved to App D; R1–R5 compressed; related-work last
  paragraph compressed.
- **tab:e6-worst moved to Appendix D in full** (Figure 1 carries the
  T-scaling story in the main text; §6.4 points to the appendix table).
- **Intro "Two deliverables, one analysis" paragraph DELETED** — it
  triple-stated content already in the one-sentence contribution +
  contribution 1 + Fig 1 caption. Contributions 2–4 merged into one
  theorem item. Contribution 3 (real-LLM) lost its TIES/KnOTS clause
  (kept in the lens paragraph).
- §5 "Empirical calibration" paragraph folded into §6.1 (ratio ≤ 13
  fact preserved there).
- Limitations enumerate → run-in paragraph (8 items, all preserved).
- Proof sketches, Remark 5, setup, captions, §6/§7 prose tightened;
  hero figure 0.92→0.78\textwidth, salvage 0.96→0.85.
- **Nothing retracted or weakened**: Lemma 2 floor form, all theorem
  statements, the §6.5 retraction text, tuned-RegMean-Yi loss, Td2
  caveats, and pre-registration language are intact.
- Local build check: main text ends bottom of p10; p11 = repro stmt +
  refs; 25 pp total; 0 overfull; 0 undefined refs/citations; colors
  render. Only warning: `T1/ptm/m/scit` falls back to plain small caps
  (cosmetic, standard for Times).
- Note: theorem numbering is shared-counter (Lemmas 1–2, Theorems 3–4,
  Remarks 5–6); intro/prose references match automatically.

**Style rule (author preference, 2026-07-04): no em/en dashes as
mid-sentence punctuation anywhere in the paper.** All 81 prose `---`
occurrences were rewritten with commas, colons, semicolons, or
parentheses (R1--R5 and Step/H1--H3 run-in headers now use colons).
Numeric-range en dashes (`$0.10$--$0.22$`, `12--18 min`) and table
empty-cell `---` placeholders are NOT punctuation and stay. Do not
reintroduce dash asides in future edits.

**2026-07-05: abstract rewritten in plain language (do not re-jargonize).**
The abstract no longer contains the floor formula (it used the
paper-local `\deff` macro, which would not even render in OpenReview's
abstract field), "per-coordinate NLL excess", "Tikhonov",
"worst-of-the-matrix", "ambiguity window", "floor-zero", "nats",
"cohort", or the 0.84→0.52 / 0.10–0.22 micro-numbers. Every claim and
honesty qualifier survives: the closed-form bound (now in words:
"irreducible floor plus a compression cost"), the log-T matching
encoder, "at default hyperparameters" on the ten-method/four-base win,
the pre-registered third-base (Mistral-7B) confirmation, floor-is-zero
= loss avoidable in principle, 0.22→0.094 Llama 3-seed salvage, and
7-of-8 downstream ranking. The formula itself still appears in §1 and
Lemma 2; only the abstract presentation changed. When pasting the
abstract into OpenReview, the remaining math is just $T$ and
$0.22 \to 0.094$ (MathJax-safe).

**Same date: intro de-jargonized (same rules).** Dropped
"worst-of-the-matrix", "Tikhonov" (now "a single ridge term"),
"cohort" (now "set of trained adapters"), "upper-estimates"; the
one-sentence contribution was simplified (per-base predictions "held
on all three bases we tested", which is accurate since the untested
below-window regime is disclosed in §6.6/App F); the hero caption now
says the 0.84→0.52 / 1.38→0.47 numbers are ratios of rd-ridge excess
to Task Arithmetic's ("falls from $0.84\times$ to $0.52\times$ TA's"),
which the old caption never stated; NLL is now expanded at first use
in the lens paragraph. ALL formulas in contribution 2 kept verbatim
(only the sentence was split in three); "match in their rate exponent"
qualifier kept; "tuned RegMean edges it on one saturated base" kept;
pre-registration footnote kept verbatim. Page boundary re-verified:
main text still ends p10.

## Remaining pre-submission to-dos

1. **Re-upload to Overleaf and recompile** (whole folder — preamble AND
   sections changed). Verify refs start ≤ p11 and Table 2 cells are
   green/red, not literal "green!18".
2. **Page limit**: re-confirm when the ICLR 2027 CFP appears (2025/2026
   were 10 pages). Reproducibility statement + refs + appendices don't
   count.
3. **Supplementary zip**: full proofs (currently referenced as "supplementary
   material" — pull from `../theory/`), code, per-example NLL arrays, the
   audit-trail bundle above.
4. **references.bib**: add and differentiate 2026 concurrent work before
   submission — especially arXiv:2603.09463 (merging-collapse, reportedly
   uses rate-distortion), arXiv:2601.22285 (mergeability prediction),
   arXiv:2606.19549 (PEFT mergeability). Also fill real venues for
   placeholder entries if any.
5. Figures: hero + salvage + td2 PDFs are vector (fine); the two appendix
   PNGs (`headline_*_ci.png`) should be regenerated ≥300 dpi or as PDF.
6. The old `updates/` variant of the paper was ignored (paper/sections/ is
   canonical per decisions.md).

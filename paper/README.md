# paper/ — the single source of truth

"Initialisation Sets the Conditioning: A Rate-Distortion View of LoRA Merging"

Both build roots carry that title as of 2026-08-14. The earlier
"Initialisation Sets the Floor" contradicted the paper's own abstract, which
reports the floor as zero in both regimes, and "Predicting When LoRA Merging
Fails" belonged to the version whose downstream case was withdrawn. Neither
should come back without the evidence coming back with it.

Everything that used to live in `ICLR/`, `arxiv/v1/`, `arxiv/v2/`,
`overleaf/iclr2027/` and `paper/updates/` is now here, once. Those
directories were deleted on 2026-08-02; the state just before deletion is
commit `d15fa8e` on branch `paper-consolidation`, so nothing is lost.

## Layout

```
paper/
  tmlr.tex          TMLR build root: official style, anonymous  <- SUBMISSION
  arxiv.tex         arXiv build root: plain article + real names, no venue named
  iclr2027.tex      ICLR build root: documentclass + style + anonymous byline
  preamble.tex      packages, macros, theorem counters   (shared)
  main.tex          the document body: \maketitle -> appendices (shared)
  sections/         19 section files                     (shared)
  figures/          3 vector PDFs + 2 PNGs               (shared)
  references.bib                                          (shared)
  tmlr.sty/.bst, fancyhdr.sty    vendored TMLR assets, from
                                 github.com/JmlrOrg/tmlr-style-file
  iclr2026_conference.sty/.bst   vendored conference assets
  build.sh          ./build.sh {tmlr|arxiv|iclr|all}  ->  out/*.pdf
```

**TMLR is the submission target and `tmlr` is now the build.sh default.**
`\usepackage{tmlr}` with no option produces the anonymous version: the byline
becomes "Anonymous authors / Paper under double-blind review" and the running
header reads "Under review as submission to TMLR". Do not add `[accepted]` or
`[preprint]` before acceptance; TMLR rejects non-anonymous submissions without
review. The real author block stays in `tmlr.tex` and is simply not rendered,
so camera-ready is a one-option change plus filling in `\month`, `\year` and
`\openreview`.

Two TMLR constraints shaped the shared sources: **the abstract must be one
paragraph** (it is, at about 310 words), and there is **no page limit**, so the
ICLR 10-page problem does not apply here. Supplementary material may be up to
100 MB, which comfortably fits the 19 MB audit-trail bundle.

The two roots differ **only** in documentclass, style packages, title block
and `\bibliographystyle`. All prose lives in `sections/`, so a content edit
lands in both builds automatically. There is no second copy to keep in sync.

## Building

```sh
./build.sh iclr      # -> out/iclr2027.pdf
./build.sh arxiv     # -> out/arxiv.pdf
./build.sh both
```

Uses `tectonic` when present (self-contained, no TeX install), else
`pdflatex` + `bibtex`. Verified 2026-08-02 with tectonic 0.16.9.

**Overleaf:** upload the whole `paper/` folder and set the main document to
`iclr2027.tex` (or `arxiv.tex`). The flat layout means no path configuration
is needed. `out/` is build output and should not be uploaded.

## Verified state as of 2026-08-14

Built with tectonic 0.17.0 on Windows.

| | arXiv build | ICLR build |
|---|---|---|
| pages | 30 (main text to p17, refs p18, appendices p20+) | 26 |
| overfull hboxes | 6, all 1.7--4.8pt | 1 |
| undefined refs / citations | 0 / 0 | 0 / 0 |

**arXiv is the target.** The six remaining overfulls are sub-2mm prose lines
in `setup`, `lower_bound`, `related_work` and `app_manifest`; they are
cosmetic and predate this pass. The three that mattered (15--22pt, in
`method_tests`, `reproducibility` and `app_synthetic`) are fixed.

**The ICLR build compiles but does not fit ICLR.** The Reproducibility
Statement lands on p15 against a 10-page main-text limit, so that target
needs roughly five pages cut before it is a submission rather than a
compile. The 2026-07-04 page budget referred to a different, shorter paper.

## Load-bearing details, do not undo

1. **`\PassOptionsToPackage{table}{xcolor}` must stay the first line of
   `iclr2027.tex`.** The ICLR style file loads eso-pic, which loads xcolor;
   without the early pass, `\usepackage[table]{xcolor}` option-clashes,
   `[table]` is silently dropped, `\cellcolor` is never defined, and every
   `\best`/`\worst` cell prints a literal "green!18".
2. **`\usepackage[T1]{fontenc}`** is required for faithful tectonic/XeTeX
   builds and is harmless under pdfLaTeX.
3. **Shared theorem counter** (Lemmas 1--2, Theorems 3--4, Remarks 5--6).
   Prose cross-references depend on it; do not split the counters.
4. **Lemma 2 floor form is $B^2(1 - \deff/(Tr))$.** Never revert to
   $B^2/(4\deff)$, which is nowhere derived and contradicts Lemma 2.
5. **No em/en dashes as mid-sentence punctuation** anywhere in the prose.
   Numeric ranges (`$0.10$--$0.22$`) and empty table cells are not
   punctuation and stay.
6. **Anonymity:** never cite the Zenodo record or the GitHub repo in the
   ICLR build. The commit trail, not Zenodo, is what establishes the
   pre-registration.
7. The style files are ICLR **2026** with the running header patched to
   2027. Swap in the official 2027 files when the CFP lands, updating
   `\usepackage` and `\bibliographystyle` together.
8. **`\belowcaptionskip` is set to 6pt in `preamble.tex`.** `article.cls`
   ships 0pt, which drops the first `\toprule` onto the baseline of the
   caption's last line. It only bites on some tables, so it looks like a
   printing error rather than a systematic one. Do not set it back to 0.
9. **Do not add `microtype`.** Measured, not assumed: under the XeTeX engine
   tectonic uses, only protrusion is available for these Type 1 faces, and
   enabling it moved the arXiv build from six overfull lines to eight.
10. **The release URL lives in one place**, `\repohost` / `\repourl` in
    `preamble.tex`. `sections/reproducibility.tex` is the only consumer.

## State after the 2026-08-14 referee response

Every blocking item in the report now has a measurement behind it, run under a
pre-registration committed before any cell existed (`acebd1a`, amended
`7ce15b1`). The campaign was 280 GPU cells across four sweeps plus five
analyses that needed no compute. `decisions.md` carries the full record; the
short version is that the paper's positive claim got stronger and its claim
about our own method got weaker.

Strengthened: the conditioning effect holds on RegMean, a solver we did not
build (4/4); it survives removing the rank truncation and is larger there,
25-117x against 2.1-65x; the mechanism is measured rather than asserted; the
ridge sweep's noise gate now exists and clears on all four bases.

Weakened: worst-task NLL excess does not predict downstream accuracy and
inverts on HumanEval; the encoder's last clear win becomes a tie once the
untrained adapter is dropped; Table 3 mixed two inference paths.

## Before this is posted anywhere

**For a TMLR submission the repository does NOT need to be public.** TMLR
reviews double-blind and requires supplementary material to be anonymised, so
a public repo URL in the PDF would de-anonymise you three ways over: the
account name, the commit authorship, and the author emails in `arxiv.tex`.
Code release is encouraged, not required.

The audit trail travels as an anonymised **git bundle** instead, built by
`code/phase3/scripts/build_anon_bundle.sh` and currently at
`../rdmerge-audit-trail.bundle` (19 MB). A bundle carries real history, so
`git clone` works on it and `git merge-base --is-ancestor` runs, which a
snapshot service like anonymous.4open.science cannot offer. Rebuild it after
any new commits, and regenerate Table 6's hashes from the mapping the script
prints, because anonymisation changes every hash (not the topology).

Blocking, in the order they bite:

1. **Rebuild the bundle before submitting** if the repo has moved since it was
   last built, and re-run the ancestry checks. The script verifies no
   identifying string survives in any blob, commit message or identity, and
   refuses to write the bundle if one does.
2. **Upload the bundle as supplementary material**, not a URL. The
   Reproducibility Statement now describes the bundle; `\repourl` is unused by
   the submission build and is kept only for a camera-ready that names the
   repository.
3. **Decide what "released" covers at camera-ready.** The statement says the
   base weights and trained adapters are not redistributed and the configs
   are. Make that true of whatever eventually goes public.
4. **arXiv endorsement**, only if arXiv is the route. As of 2026-06 there was
   none, which is why the earlier snapshot went to Zenodo. TMLR permits
   preprints, but the submission must not link to a named version, so the
   paper must not cite the Zenodo DOI while under review.

Non-blocking, worth doing:

- Regenerate `figures/headline_*_ci.png` at >=300 dpi or as PDF.
- Six sub-2mm overfull lines remain; they need manual rewrapping, not a
  package.
- If ICLR ever becomes the target again, budget the five-page cut noted
  above and re-check the page boundary after every content edit.

Closed on 2026-08-14: the `[REPOSITORY URL]` placeholder, the anonymous
`concurrent2026merging` bib stub (replaced by the real concurrent work,
`cao2026collapse`, now cited and differentiated in `related_work.tex`), the
author emails, and the three large overfull boxes.

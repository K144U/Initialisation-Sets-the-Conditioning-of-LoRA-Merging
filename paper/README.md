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
  iclr2027.tex      ICLR build root: documentclass + style + anonymous byline
  arxiv.tex         arXiv build root: plain article + real names, no venue named
  preamble.tex      packages, macros, theorem counters   (shared)
  main.tex          the document body: \maketitle -> appendices (shared)
  sections/         19 section files                     (shared)
  figures/          3 vector PDFs + 2 PNGs               (shared)
  references.bib                                          (shared)
  iclr2026_conference.sty/.bst   vendored conference assets
  build.sh          ./build.sh {iclr|arxiv|both}  ->  out/*.pdf
```

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

## Before this is posted anywhere

Blocking, in the order they bite:

1. **The repository must exist publicly.** `\repourl` points at
   `github.com/K144U/rdmerge`, which currently 404s for anonymous visitors,
   and the branch carrying the audit trail (`paper-consolidation`, 53
   commits ahead of `origin/phase3-bootstrap`) has never been pushed. The
   paper's central integrity claim is that the commit ordering is checkable
   by a reader. Until that push happens the claim is unverifiable, and the
   Reproducibility Statement is describing something that does not exist.
2. **Decide what "released" covers.** The statement now says the base
   weights and trained adapters are not redistributed and the configs are.
   Make that true of whatever is actually pushed.
3. **arXiv endorsement.** As of 2026-06 there was none, which is why the
   earlier snapshot went to Zenodo. Confirm before assuming arXiv is
   available. If Zenodo is used again, the stale 2026-06-02 record should be
   updated rather than left alongside this version.

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

# paper/ — the single source of truth

"Predicting When LoRA Merging Fails: A Rate-Distortion View"

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

## Verified state as of 2026-08-02

| | ICLR build | arXiv build |
|---|---|---|
| pages | 25 (main text ends p10, refs p11) | 29 |
| overfull hboxes | **0** | 6 (five 1.7--4.8pt, one 15.6pt) |
| undefined refs / citations | 0 / 0 | 0 / 0 |

The ICLR build reproduces the 2026-07-04 page budget exactly: main text ends
at the bottom of p10, which is the hard limit (an 11th page is a desk
reject). Re-check the page boundary after any content edit.

The arXiv overfulls are pre-existing, not a regression: that build was
assembled on 2026-07-07 and never compiled, because there was no local LaTeX
toolchain at the time. They come from the wider 1in/11pt geometry pushing
tables that were tuned for the ICLR text width. Worth fixing before upload.

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

## Open to-dos

- Page limit: reconfirm when the ICLR 2027 CFP appears (2025/2026 were 10pp).
- `arxiv.tex`: fill the two `[EMAIL PLACEHOLDER]`s; decide on an affiliation
  line; `sections/reproducibility.tex` still says "accompanies the
  camera-ready version", which presumes acceptance.
- `references.bib`: drop the uncited `concurrent2026merging` placeholder
  (`journal = {Anonymous preprint}`); fill missing venues/eprints for
  `lorm2024`, `regmeanpp2025`, `tspa2025`; `tara2025` has a 2025 key but
  `year = {2026}`. Add and differentiate the 2026 concurrent work,
  especially arXiv:2603.09463, which reportedly uses the same lens.
- Build the anonymized audit-trail bundle the paper promises in the intro
  footnote and the reproducibility statement. It does not exist yet.
- Regenerate `figures/headline_*_ci.png` at >=300 dpi or as PDF.
- The content itself has open corrections pending; see
  `../notes/audit_2026-08-02_code_and_claims.md`.

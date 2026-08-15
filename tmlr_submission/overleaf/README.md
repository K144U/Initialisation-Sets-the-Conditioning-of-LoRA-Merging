# TMLR submission source

Standalone, self-contained LaTeX source for *Initialisation Sets the
Conditioning of LoRA Merging*. Everything needed to
compile is in this directory; nothing outside it is referenced.

## Overleaf

Upload the zip. `main.tex` is the root and Overleaf selects it automatically,
because it is the only file here with a `\documentclass` and it carries the
name Overleaf prefers. There is nothing to configure. Compile with pdfLaTeX or
XeLaTeX; both work.

## Locally

```sh
tectonic --keep-logs --outdir out main.tex     # or ./build.sh
```

Last verified with tectonic 0.16.9: **51 pages, 0 overfull boxes, 0 undefined
references, 0 undefined citations.** The underfull `\vbox` warnings are
vertical stretch on pages carrying wide tables or breaking between the
appendix listings, and are cosmetic.

## Layout

| file | role |
|---|---|
| `main.tex` | build root: documentclass, `\usepackage{tmlr}`, title block, bib style |
| `body.tex` | the document body, from `\maketitle` through the appendices |
| `preamble.tex` | shared packages, macros, theorem counter |
| `sections/` | the 15 section files `body.tex` inputs |
| `prereg/` | the eight pre-registration documents, included verbatim by Appendix D |
| `figures/` | the one figure the text draws |
| `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` | the official TMLR style files, unmodified |
| `references.bib` | bibliography |

**Two files are renamed** relative to the `paper/` tree in the project
repository, so that Overleaf needs no configuration:

```
main.tex   <-  paper/tmlr.tex
body.tex   <-  paper/main.tex
```

Everything else keeps its name, so an edit made on Overleaf can be copied
straight back. If you copy `main.tex` or `body.tex` back, rename accordingly
and restore `\input{main}` in place of `\input{body}`.

This directory carries only the TMLR build. The arXiv and legacy ICLR roots,
the nine appendix sections the current body does not input, and the six unused
figures are all still in `paper/` in the project repository.

## Anonymity

The build is the anonymous submission form: `\usepackage{tmlr}` with **no
option**. Do not add `[accepted]` or `[preprint]` before the paper is accepted,
as TMLR rejects non-anonymous submissions without review.

The source was scanned for author names, emails, the repository URL,
institutional strings, cluster hostnames and absolute paths; none are present.
`\repohost` and `\repourl` in `preamble.tex` are defined to an anonymous
placeholder. No section in this build uses either macro, so nothing renders
from them; restore the real values from `paper/preamble.tex` at camera-ready,
together with `[accepted]`, the real author block, and the month, year and
OpenReview forum id at the bottom of `main.tex`.

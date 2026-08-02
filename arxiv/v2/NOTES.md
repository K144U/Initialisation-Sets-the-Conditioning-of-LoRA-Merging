# arXiv preprint build (v2) — notes

Built 2026-07-07 from `../../ICLR/` (the ICLR 2027 double-blind
submission build). `sections/`, `figures/`, and `references.bib` are
byte-identical copies of the ICLR/ build's — content is intentionally
unchanged. **The preamble/style is NOT copied from ICLR/**: this build
uses a plain `\documentclass{article}` with generic packages
(geometry, times, amsmath, natbib w/ `plainnat`), no vendored
conference style or bib-style file, and no running head. Per explicit
instruction (2026-07-07): no mention of "ICLR" or its template/format
anywhere in this build — not in the title page, not in vendored
filenames, not in the compiled output.

This supersedes `arxiv/v1/` (the original 2026-05-17 draft, pre-reframe:
old title "A Rate-Distortion Function for Model Merging", 8 sections,
no appendices). `v1/` is left in place as history, not deleted.

## Title page

- **Real author names**: Sankalp Pathak, Sanjay Garg (co-authors, per
  `context.md`'s "Author: Sankalp Pathak (solo), Prof. Sanjay Garg
  reviewing/advising" — promoted to co-author for this preprint at the
  user's explicit choice, 2026-07-07). Standard `\and`-separated
  two-author block (no custom macros).
- **Email placeholders** (`[EMAIL PLACEHOLDER]` x2) — left as-is per
  the user's choice not to commit to contact info in this pass. Note:
  `arxiv/v1/main.tex` (the old draft) already used real addresses with
  no institution line (`pathaksankalp04@gmail.com` / `gargsv@gmail.com`)
  — worth checking whether those still apply before filling this in.
- No affiliation line (v1 had none either).
- `\date{\today}` — shows the compile date, not tied to any venue.

## What's the same as the ICLR submission (deliberate, per user choice)

- Full content match, not a restore of material trimmed for the
  ICLR page limit. If a longer arXiv-only version is wanted later, the
  natural candidates to restore are the ones `ICLR/NOTES.md` already
  flags as compressed (tab:e6-worst moved to appendix, R1-R5
  compressed, related-work last paragraph compressed, etc.).
- Citations use `\citep`/`\citet` (natbib) throughout — compatible
  with the plain `plainnat` bibliography style used here.

## Flagged, not changed

- `sections/reproducibility.tex` line ~25 ends "...accompanies the
  camera-ready version." That phrase presumes an eventual conference
  acceptance/camera-ready step; it doesn't name ICLR, so it was left
  untouched, but it's adjacent to the thing you asked to scrub. Say
  the word if you want it reworded (e.g. "accompanies the released
  code" instead).
- `sections/intro.tex` (~line 28) and `sections/reproducibility.tex`
  (~line 17) both still say "anonymized...bundle" — literally accurate
  (the git-history bundle itself has authors scrubbed regardless of
  this paper's byline) but reads a little oddly next to a named-author
  preprint. Left as-is; flag if you want it reworded too.
- No GitHub/Zenodo links were added even though this version doesn't
  need anonymity. Linking the repo/Zenodo record here is a legitimate
  option now, just not applied automatically.

## Removed from this build (present in a prior pass, now deleted)

`iclr2026_conference.sty`, `iclr2026_conference.bst`, `fancyhdr.sty`,
`natbib.sty`, `math_commands.tex` — all vendored ICLR-template assets
from an earlier iteration of this folder. Removed so no ICLR-named
file ships in the arXiv source tarball. Bibliography now uses
`plainnat` (ships with every standard LaTeX/TeX Live installation, no
vendored file needed); `natbib` itself is a standard CTAN package, also
no vendoring needed.

## Remaining before this can actually go to arXiv

1. Fill `[EMAIL PLACEHOLDER]` x2 in `main.tex` (see note above).
2. Decide on an affiliation line (v1 had none; still fine to omit).
3. Compile and sanity-check (no local LaTeX toolchain on this machine —
   use Overleaf or similar): title page, cellcolor tables render,
   citations resolve with `plainnat`.
4. Pick an arXiv category (likely cs.LG, cross-list cs.CL).
5. Decide on the two flagged phrases above ("camera-ready version",
   "anonymized...bundle").
6. This has NOT been uploaded anywhere. No submission action has been
   taken; this is source prep only.

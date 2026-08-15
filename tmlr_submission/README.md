# TMLR submission package

*Initialisation Sets the Conditioning of LoRA Merging*

Everything needed to submit, in the order you will need it. Nothing here
depends on anything outside this directory.

| what | where it goes |
|---|---|
| `paper.pdf` | the PDF field of the OpenReview submission form |
| `supplementary.zip` | the supplementary material field of the same form |
| `overleaf_tmlr.zip` | upload to Overleaf if you want to edit or recompile |

The two folders `overleaf/` and `supplementary/` are the unzipped contents of
the two zips, kept alongside them so you can read and edit without unpacking.
**If you change anything in a folder, re-run `make_zips.sh` before uploading**,
or the zip and the folder will disagree.

## paper.pdf

The submission, built from `overleaf/`. 51 pages, anonymous, 0 overfull boxes,
0 undefined references, 0 undefined citations. Verified byte-identical in
extracted text to the reference build in `paper/out/tmlr.pdf`, and to a build
from a clean extraction of `overleaf_tmlr.zip`.

34 of the 51 pages are the paper. The rest is Appendix D.4, which
reproduces all eight pre-registration documents in full. TMLR has no page limit,
and §1 and the Reproducibility Statement both promise those documents verbatim,
so they are included from the source files at build time rather than retyped:
the appendix cannot drift from what the commits contain, because it is the
files.

## overleaf_tmlr.zip

Upload the zip to Overleaf as a new project. `main.tex` is the root and
Overleaf selects it automatically: there is nothing to configure, which is the
reason two files are renamed relative to the `paper/` tree (`paper/tmlr.tex`
became `main.tex`, `paper/main.tex` became `body.tex`). `overleaf/README.md`
records the mapping and how to copy edits back.

The zip carries only the TMLR build: one root, the 15 section files the body
inputs, the eight pre-registration documents Appendix D includes verbatim, the
one figure the text draws, the three official TMLR style files, and the
bibliography. The arXiv and legacy ICLR roots, the nine appendix sections this
version does not input, and the six unused figures stay in `paper/`.

One anonymity note on `prereg/`. Those files are byte-for-byte copies of
`notes/prereg_*.md` with exactly one alteration: the first document records a
completion time with a timezone abbreviation, which would narrow the authors'
geography under double-blind review, and it appears as `[tz]`. The alteration
is declared in the appendix text at the point of reproduction, so the word
"verbatim" in §1 and the Reproducibility Statement is accurate as written.

## supplementary.zip

The anonymised audit-trail bundle plus a reader-facing README. TMLR allows 100
MB; this is 19 MB.

It is a git bundle rather than an anonymised repository link on purpose. The
paper's claim about its own procedure is that a reader can verify **commit
ordering**, and snapshot anonymisers serve files with no history, which removes
exactly the thing being claimed. A bundle is one file carrying real history:
`git clone` works on it and `git merge-base --is-ancestor` runs.

Verified before packaging, on a fresh clone of the exact file being shipped:

- `git bundle verify` reports a complete, self-contained history, 159 commits.
- All **9 ancestry rows of Table 12 pass**: every rules commit is an ancestor of
  the result commit on its row.
- All 10 intermediate analyzer and implementation hashes Table 12 cites resolve.
- All **pre-registration files have exactly one commit each**, so none was
  edited after it was first committed. That is the stronger form of the claim
  and it holds.
- Author and committer identity is uniformly `Anonymous Author
  <anon@anonymous.invalid>` across all 159 commits, and a scan of every blob in
  every revision, every commit message and every path name found no author
  name, email, institution string, cluster address or home directory.

## Before you submit

1. **Do not add `[accepted]` or `[preprint]`** to `\usepackage{tmlr}`. The
   no-option form is the anonymous one, and TMLR rejects non-anonymous
   submissions without review.
2. **Do not cite the Zenodo DOI** while under review. Preprints are allowed,
   but the submission must not link to a named version. Nothing in the current
   sources cites it; this is a note for anything you add.
3. The repository stays private until camera-ready. `\repohost` and `\repourl`
   are anonymous placeholders in this copy and are unused by the build.

## Two things in the bundle worth a decision

Neither is an anonymity breach and neither blocks submission, but both are
visible to a reviewer who browses the bundle, so decide deliberately rather
than by default.

**The bundle's own `README.md` and `CITATION.cff` carry the old title and
venue** ("A Rate-Distortion Function for Model Merging", ICLR 2027). They
predate the restructure and were never updated. The `paper/` sources inside the
bundle are current.

**The bundle carries the working files of the project as well as its results**:
`CLAUDE.md`, `master_plan_iclr2027.md`, `handoff.md`, `plan.md`, `target.md`,
`prompts/`. These are anonymised, but they are candid working documents rather
than research artifacts, and `CLAUDE.md` in particular contains a table of what
would raise and lower the paper's score. A reviewer could read that as
outcome-oriented. The argument for leaving it is that the paper's whole claim is
an unedited trail, and pruning it further would need disclosing in the
Reproducibility Statement, which already discloses the removals that were made.

Changing either means rebuilding the bundle. Note the cost: **anonymisation
rewrites every commit hash, so Table 12's hashes would all have to be
regenerated**, and `build_anon_bundle.sh` prints the old-to-new mapping for
exactly the commits the table cites. Adding a commit on top instead, to fix the
stale title, leaves every hash in Table 12 intact, since they are all ancestors.

## Rebuilding

`make_zips.sh` regenerates both zips from the two folders. To rebuild the PDF,
run `overleaf/build.sh`, which needs tectonic on the path or a TeX install.

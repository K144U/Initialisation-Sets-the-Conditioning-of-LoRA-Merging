# JAIR submission, what to upload and what to type

Everything in this folder is generated. Rebuild it with:

    python JAIR/sync_from_paper.py && bash JAIR/build_local.sh
    python JAIR/make_submission.py

## Files

| File | Where it goes |
|---|---|
| `Pathak_Garg_Singh_Initialisation_LoRA_Merging.pdf` | the manuscript, 49 pages. This is the file under review. |
| `cover_letter.txt` | paste into the cover-letter or comments field. Plain text on purpose: wizard fields do not render markdown. |
| `Pathak_Garg_Singh_Initialisation_LoRA_Merging_latex_source.zip` | LaTeX sources. Not needed at submission. Keep it for the camera-ready. |

## Wizard fields

- **Title:** Initialisation Sets the Conditioning of LoRA Merging
- **Authors, in order:**
  1. Sankalp Pathak, ORCID 0009-0006-5666-8271, pathaksankalp04@gmail.com, **corresponding author**
  2. Sanjay Garg, ORCID 0000-0002-2279-9373, gargsv@gmail.com
  3. Piyush Kumar Singh, ORCID 0009-0000-8033-3777, Piyushsingh5629@gmail.com
- **Affiliation, all three:** Department of Computer Science and Engineering,
  Jaypee University of Engineering and Technology, Guna
- **Abstract:** structured, on page 1 of the PDF under Background / Objectives /
  Methods / Results / Conclusions. Copy it from there if the wizard wants it
  separately.
- **Keywords / CCS concepts:** none. JAIR does not use them; do not invent any.
- **Excluded reviewers:** none.
- **Suggested reviewers:** none given.

## Declarations

- The work is **not** under review at any other journal or forum.
- **All three authors approve** submission.
- One disclosure, already in the cover letter: an earlier and superseded
  version is public on Zenodo, doi 10.5281/zenodo.21238820. It is cited in the
  introduction, which also says which two of its claims this paper does not
  make.

## Two things reviewers are told to check

- **No online appendix,** deliberately. JAIR does not review them, and the
  audit trail is the part most wanted under review. Appendix H of the PDF, the
  reproducibility checklist, explains which clause of items 1 and 4 is
  satisfied instead.
- **The commit trail is public** at
  https://github.com/K144U/Initialisation-Sets-the-Conditioning-of-LoRA-Merging
  A reviewer can clone it and check that every pre-registration precedes the
  compute it governs, with `git merge-base --is-ancestor`.

## Before uploading

- [ ] Open the PDF and confirm it is 49 pages and the author block is on page 1.
- [ ] Confirm Appendix H, the reproducibility checklist, is present and complete.
      A submission without it is desk-rejected.
- [ ] Paste `cover_letter.txt`, then read it once in the field to confirm no
      stray markdown survived.

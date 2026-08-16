# Supplementary material

*Initialisation Sets the Conditioning of LoRA Merging*

This directory contains two things:

| path | what |
|---|---|
| `audit-trail.bundle` | the project's real version history, as a git bundle (23 MB) |
| `prereg/` | the twelve pre-registration documents, complete and unedited |

`prereg/` is here for convenience, so the documents can be read without
cloning. It is **not** the authoritative copy. The same twelve files are inside
the bundle at the commits Appendix D names, and only the copies in the bundle
carry the evidence that they were written when they claim to have been. If the
two ever disagree, the bundle is right.

The bundle is a **git bundle**: a single file carrying the real version
history, not a snapshot of it. The distinction is the point. The paper asks you
to check that each pre-registration was committed *before* the compute it
governs, and that check is only meaningful against a commit graph. A directory
of files, however complete, cannot support it.

```
sha256  c0fb02bb683a8ea8e96972abff245e3542d2546237e36aff5232b6dec43fbc91
md5     df1f2a64e52079127222b73d4c04f758
```

## Open it

```sh
git clone audit-trail.bundle rdmerge
cd rdmerge
```

You now have an ordinary git repository: 202 commits on three branches,
`paper-consolidation` (checked out), `phase3-bootstrap` and `main`. Every git command
works normally. If you only want to read the files and do not care about the
history, the clone's working tree is already the snapshot you want.

`git bundle verify audit-trail.bundle` confirms the history is complete and
self-contained before you clone.

## Check the claim the paper actually makes

Appendix D, Table 12 lists, for each pre-registered test, the commit that fixed
the decision rules and the commit that recorded the result. The caption asks you
to verify the ordering yourself. From inside the clone:

```sh
git merge-base --is-ancestor 5693d9a ee90491 && echo ordering holds
```

which exits zero exactly when the rules commit precedes the result commit, and
non-zero otherwise. The 14 rows of Table 12 are:

| test | rules | result |
|---|---|---|
| Replication, step 0 | `5693d9a` | `ee90491` |
| Replication, n=3 | `44c0d39` | `3723f96` |
| DARE with TIES | `0e0898d` | `b1613fc` |
| Conditioning and the ridge | `f48c4ef` | `34a6c69` |
| Rate exponent | `f48c4ef` | `6ae3937` |
| Solver replication | `1eabc27` | `758892a` |
| Untruncated and gate | `1eabc27` | `f5323e2` |
| Repaired KnOTS | `1eabc27` | `86000cf` |
| Merge matrix at T=3 | `1eabc27` | `2a97774` |
| Margin-aware control | `1eabc27` | `a9cf9c7` |
| Shared arm at n=3 | `41d8adc` | `a9cf9c7` |
| Public-cohort prevalence | `5e8e736` | `a9cf9c7` |
| Downstream accuracy | `8845efa` | `a9cf9c7` |
| Heuristics null in accuracy | `9cc2394` | `77a7f37` |

The stronger check is that the rules files were not edited after they were
first committed, since a pre-registration that can be revised afterwards is
worth nothing:

```sh
git log --follow --oneline -- notes/prereg_tmlr_2026-08-14.md
```

should show exactly one commit.

## What is in it

| path | contents |
|---|---|
| `notes/prereg_*.md` | every pre-registration document, and the two amendments |
| `code/phase3/merging/` | all merge method implementations |
| `code/phase3/eval/` | the evaluation harness, both the NLL and the downstream scorers |
| `code/phase3/scripts/` | generators, analyzers, the geometry and conditioning audit, the cluster orchestrator |
| `code/phase3/configs/` | per-cell configurations and per-experiment manifests, including every seed |
| `results/phase3/` | per-cell result JSON and the aggregate summaries the tables are built from |
| `paper/` | the LaTeX sources |
| `theory/`, `paper_artifacts/` | derivations and figure sources |

The base model checkpoints and the trained LoRA adapters are **not** included.
The bases are public checkpoints, and the adapter training configurations,
including every initialisation seed, are in `code/phase3/configs/`.

## Anonymisation

The history is rewritten: author and committer identity, machine names,
absolute paths and institutional identifiers are replaced throughout every
commit message and every file. This changes every commit hash, which is why the
hashes above are the bundle's rather than the working repository's. It does not
change the commit graph, and the graph is what the verification depends on.

A small number of files are removed entirely, because they identify people and
bear on no result: correspondence, external review notes, and drafts carrying
author names. Nothing removed is cited anywhere in the paper.

Two consequences of the rewrite are visible and worth naming so they are not
mistaken for something else. The repository's own `README.md` and
`CITATION.cff` carry an earlier title and venue for this work, because they
were written before the paper was restructured and were never updated; the
paper's own sources under `paper/` are current. And a docstring in
`code/phase3/scripts/gpu_opportunity.py` still says `gpu01` where the code says
`gpu-node-01`, a leftover of the hostname rewrite. Commit timestamps are
normalised to UTC: the epoch second of each commit is unchanged, only the
recorded offset, so ordering is exactly as it was.

# Supplementary material

*Initialisation Sets the Conditioning of LoRA Merging*

This directory contains one file, `audit-trail.bundle` (19 MB). It is a **git
bundle**: a single file carrying the project's real version history, not a
snapshot of it. The distinction is the point. The paper asks you to check that
each pre-registration was committed *before* the compute it governs, and that
check is only meaningful against a commit graph. A directory of files, however
complete, cannot support it.

```
sha256  b35b5e4f01d1fed1b4143dd82161d2fa92a044e07b49997a16502e557cb11ece
md5     25a0f7c0e5c689a14b6882002015eea6
```

## Open it

```sh
git clone audit-trail.bundle rdmerge
cd rdmerge
```

You now have an ordinary git repository: 159 commits on two branches,
`paper-consolidation` (checked out) and `phase3-bootstrap`. Every git command
works normally. If you only want to read the files and do not care about the
history, the clone's working tree is already the snapshot you want.

`git bundle verify audit-trail.bundle` confirms the history is complete and
self-contained before you clone.

## Check the claim the paper actually makes

Appendix D, Table 12 lists, for each pre-registered test, the commit that fixed
the decision rules and the commit that recorded the result. The caption asks you
to verify the ordering yourself. From inside the clone:

```sh
git merge-base --is-ancestor 0d9924b 175453f && echo ordering holds
```

which exits zero exactly when the rules commit precedes the result commit, and
non-zero otherwise. The nine rows of Table 12 are:

| test | rules | result |
|---|---|---|
| Replication, step 0 | `0d9924b` | `175453f` |
| Replication, n = 3 | `d503347` | `7edf68a` |
| DARE with TIES | `100cd43` | `3a0822e` |
| Conditioning and the ridge | `fa5f3c3` | `25d7de0` |
| Rate exponent | `fa5f3c3` | `dc2991a` |
| Solver replication | `0c8b9e8` | `a5e1ea1` |
| Untruncated and gate | `0c8b9e8` | `23c251b` |
| Repaired KnOTS | `0c8b9e8` | `30d2c04` |
| Merge matrix at T = 3 | `0c8b9e8` | `0d35789` |

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
`gpu-node-01`, a leftover of the hostname rewrite.

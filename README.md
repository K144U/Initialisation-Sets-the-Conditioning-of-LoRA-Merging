# Initialisation Sets the Conditioning of LoRA Merging

Code, pre-registrations, and per-cell experimental results for the paper of the
same name, by **Sankalp Pathak**, **Sanjay Garg** and **Piyush Kumar Singh**
(Department of Computer Science and Engineering, Jaypee University of
Engineering and Technology, Guna). The manuscript is under review at the
*Journal of Artificial Intelligence Research*.

This repository exists to be **checked**, not just read. Every empirical claim
in the paper is governed by a pre-registration that was committed to version
control before the compute it governs was dispatched. The commit graph here is
intact, so that ordering can be verified with `git merge-base` rather than
taken on trust. See [Checking the audit trail](#checking-the-audit-trail).

## What the paper claims

LoRA factorises each update as `BA`, draws `A` at random and starts `B` at
zero. Whether the `T` adapters of a cohort drew **one** `A` or `T` independent
ones is almost never reported, and a single global training seed silently
decides it. Sharing the draw leaves the tasks in nearly the same subspace:
across four base models the median principal cosine between task row spaces
moves from 0.047 to 0.995.

1. **A one-CPU-minute audit.** The collapse is measurable from the adapter
   weight files alone: no forward passes, no evaluation data, no merging, about
   a minute of CPU time for a cohort of four rank-16 adapters. It runs before
   any merge is attempted and is independent of which merging method follows.
2. **A pre-registered prevalence estimate.** Under a sampling frame fixed
   before any download, **17 of 45** public LoRA cohorts (37.8%) are in the
   collapsed regime. The distribution is bimodal rather than continuous: one
   cohort of the 45 lies between the two populations.
3. **The consequence splits cleanly by method.** Merges that average or
   sparsify task vectors are unaffected: 15 of 20 base-by-method cells show no
   effect at 0.01 nat resolution, and 0 of 40 show one on GSM8K or HumanEval.
   Merges that solve a linear system are decisive: run unregularised they are
   2.1 to 65 times worse on a collapsed cohort, 25 to 117 times with rank
   truncation removed. The mechanism is measured rather than inferred: the
   heuristics never place more than a rounding error of their norm in the
   ill-conditioned directions, and the unregularised solvers place three
   quarters of it there. Shown for our own encoder **and** for RegMean, so it
   is a property of the family rather than of our construction.
4. **It reaches downstream accuracy, not just our loss metric.** On Mistral the
   unregularised shared-cohort merge scores **zero** on both GSM8K and
   HumanEval, with all 664 generations non-empty and none discarded, and
   recovers to 0.46 and 0.24 with one Tikhonov term.

Regularising a least-squares solve is standard practice, so we checked what the
standard practice actually does. RegMean's shipped alpha-shrinkage default of
0.9 is equivalent, on a collapsed cohort, to a ridge of about 5e-4: roughly 20
times smaller than the value the sweep selects, leaving the system at condition
number 6.6e3 to 9.1e3 against 1.5e2 with the ridge. It applies the same dose on
a well-conditioned cohort, where it is ample. **The default regularisation does
not know which regime it is in, and the audit is what tells you.**

## What did not survive

Several pre-registered tests went against us. They are here for the same reason
they are in the paper.

- **Our encoder loses its advantage** in the move from shared-initialisation
  cohorts to properly initialised ones. The paper claims no win for it.
- **An earlier headline claim of ours is withdrawn and stays withdrawn**: that
  published merging benchmarks are confounded by initialisation geometry. A
  control we designed ourselves refuted it. A later margin-aware control would
  reinstate it, but that control was written after its data had been seen, and
  we do not use it that way.
- **The rate-distortion bound is provenance, not a contribution.** It is why we
  looked at the conditioning at all, and no more than that. Its floor term is
  exactly zero on every cohort we measured, its rate term rests on an
  assumption stated but not proved, and it derives none of the four results
  above. The rate exponent it predicts is **unresolved**: our sweep lacks the
  dynamic range to measure it.
- **One of our four adapters never learned its task**, so the cohorts are `T=4`
  nominal and 3 effective. Every geometric quantity is reported at both.
- **The original scorers were broken.** The HumanEval extractor returned an
  empty completion for markdown-fenced output and the GSM8K extractor required
  the answer at end-of-string, discarding 61% to 81% of generations on some
  method and base combinations and near zero on others, which makes
  cross-method comparison meaningless. Both are fixed, all affected cells were
  re-scored, and the discard rate is now zero.
- **Training and evaluation splits are not disjoint** on two of the four tasks:
  12.9% to 16.2% overlap on the instruction-following task and 9.2% to 11.6% on
  the code task. Absolute numbers here are therefore optimistic and are not
  held-out performance. Whether that affects comparisons *between* methods was
  tested rather than asserted: on the disjoint subset the method ordering is
  unchanged in 19 of 20 rows and the worst task is the same in all 144 cells.

## Checking the audit trail

The twelve pre-registrations are in `notes/prereg_*.md`, unedited. Their stale
internal cross-references are deliberate: they are the evidence that nothing
was rewritten after the fact. The paper's appendix "Pre-Registration Documents
and Commit Ordering" lists, per row, the commit that fixed the rules and the
commit carrying the result they govern. For each row:

```
git merge-base --is-ancestor <rules-commit> <result-commit>
```

exits zero exactly when the ordering claimed holds, and fails on the reverse.

**One row is expected to fail, and the paper says so.** The solver replication
appears as two rows because its amendment (`9b3f57e`) does not precede the
step-0 result (`1dcadd4`). The ancestry check on that pair fails, and should.
Reporting it is cheaper than the alternative.

The published history is **filtered**, which rewrote every commit hash, so
those hashes are this repository's rather than our working repository's. Three
things are absent, none cited anywhere in the paper and none bearing on any
result: private correspondence and external review notes, which were never ours
alone to publish; large build artifacts; and text identifying a manuscript
still under anonymous review elsewhere. Filtering changes hashes but not the
commit graph, and the graph is what the verification depends on.

## Repository layout

```
paper/                   LaTeX source. Several build roots share one body:
                           jair.tex   JAIR submission, named authors
                           tmlr.tex   anonymous
                           arxiv.tex, iclr2027.tex
  sections/              section fragments
  references.bib
JAIR/                    JAIR build. overleaf/ is DERIVED from paper/ by
                         sync_from_paper.py; do not edit it by hand
code/phase3/
  merging/               merge implementations: task_arithmetic, ties, dare,
                         dare_ties, della, knots, tvq, regmean, adamerging,
                         fisher_avg, magnitude_prune, rd_encoder
  eval/                  evaluation drivers, NLL excess and downstream accuracy
  training/              LoRA training pipeline
  scripts/               generators, analyzers, figure makers, PBS launchers
  configs/               per-cell YAML configs and orchestrator manifests
notes/prereg_*.md        the twelve pre-registration documents
results/phase3/          per-cell JSON outputs and summary JSONs
paper_artifacts/figures/ generated figures
```

## Reproducing

```
conda create -n rdmerge python=3.11 -y
conda activate rdmerge
pip install -r requirements.txt
export PYTHONNOUSERSITE=1
export RDMERGE_ROOT=$(pwd)
```

### The audit, which needs no GPU

This is the measurement the paper asks you to run first, and it is deliberately
cheap: CPU only, from adapter weight files, about a minute per cohort.

```
python code/phase3/scripts/measure_subspace_geometry.py --cohort seed1
python code/phase3/scripts/measure_subspace_geometry.py --cohort indep1
```

`seed1` is a shared-initialisation cohort and `indep1` an independently
initialised one. The paper's geometry and conditioning tables are built from
this script and its siblings `geometry_T4_vs_T3.py` and `floor_conditioning.py`,
so every number in them is reproducible without GPU access.

The prevalence study over public cohorts is `audit_public_cohorts.py`, which
runs its metadata phase and fixes every inclusion decision before downloading
any weights. The library-source condition under which two adapters receive
identical `A` factors is worked out in `audit_peft_init_condition.py`.

### A single evaluation cell

```
python code/phase3/eval/run_eval_cell.py --config code/phase3/configs/eval_matrix_seeds/llama31_8b__ties__seed1.yaml
```

Larger sweeps are driven from the manifests in `code/phase3/configs/*.json`
through the orchestrator, one worker per pinned GPU. Analyzers are
`code/phase3/scripts/analyze_*.py`, and each refuses to report until every cell
it needs exists. That is a pre-registration constraint, not a convenience.

## Compute footprint

All merging and evaluation ran on a single node with A100-80GB GPUs, one
evaluation cell per card at 18 to 21 GiB. The project trained 112 LoRA adapters
and completed roughly 1,200 evaluation cells at 17 to 26 minutes each, on the
order of **400 GPU-hours**, of which the experiments reported in the paper are
a subset. That total includes superseded runs: these results were not obtained
on the first attempt, and several of the corrections above required re-running
cells that had already been analysed.

The base model weights and the trained adapters are **not** redistributed. The
bases are public checkpoints, and the adapter training configurations,
including every seed, are here.

## Citation

The paper is under review, so there is no volume, article number or DOI yet.

```bibtex
@unpublished{pathak2026initialisation,
  title  = {Initialisation Sets the Conditioning of {LoRA} Merging},
  author = {Pathak, Sankalp and Garg, Sanjay and Singh, Piyush Kumar},
  year   = {2026},
  note   = {Manuscript under review at the Journal of Artificial
            Intelligence Research}
}
```

An earlier and now **superseded** version of this work is public as a preprint,
[10.5281/zenodo.21238820](https://doi.org/10.5281/zenodo.21238820), titled *A
Rate-Distortion Function for Model Merging*. It predates the pre-registered
replication reported here and still presents as headline results two claims the
current manuscript does not make. Please cite the entry above instead; where
the two documents disagree, the current manuscript is the one that holds.

## License

MIT, see [`LICENSE`](LICENSE).

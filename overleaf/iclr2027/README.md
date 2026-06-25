# Overleaf-ready bundle for ICLR 2027 submission

This folder contains everything Overleaf needs to compile the paper.
Paths are already adjusted so the figures resolve whether Overleaf
compiles from the project root or from `paper/`.

## Folder

```
paper/
  main.tex                          # Overleaf main document
  references.bib
  sections/                         # 18 .tex section fragments
  paper_artifacts/figures/          # 4 figures (Figure 1 PDF + 3 PNG)
```

## Upload workflow

1. From your laptop: `git pull` (this folder is now in the repo at
   `overleaf/iclr2027/paper/`).
2. Drag the `paper/` directory into a new Overleaf project (or zip it
   and upload via "New Project → Upload Project").
3. In Overleaf, **Menu → Main document → `paper/main.tex`** if it
   doesn't auto-detect.
4. Recompile.

## Diff vs. `~/projects/rdmerge/paper/`

The canonical paper source lives at `~/projects/rdmerge/paper/`. This
Overleaf bundle is a **derived snapshot** with two differences:

1. The `paper_artifacts/` directory has been moved inside `paper/`
   (originally a sibling at the repo root).
2. All `../paper_artifacts/` paths in `main.tex` and
   `sections/{intro,experiments}.tex` have been rewritten to
   `paper_artifacts/`.

If you edit anything here, **also propagate the change back to
`paper/`** in the repo root so the canonical source stays the source
of truth. To rebuild this bundle from the canonical source:

```bash
STG=~/_overleaf_staging
DST=overleaf/iclr2027
rm -rf $STG && mkdir -p $STG/paper
cp -r paper/main.tex paper/references.bib paper/sections $STG/paper/
mkdir -p $STG/paper/paper_artifacts/figures
cp paper_artifacts/figures/*.pdf paper_artifacts/figures/*.png \
  $STG/paper/paper_artifacts/figures/
sed -i 's|\.\./paper_artifacts/|paper_artifacts/|g' \
  $STG/paper/main.tex $STG/paper/sections/*.tex
rm -rf $DST && mkdir -p $DST && cp -r $STG/paper $DST/
```

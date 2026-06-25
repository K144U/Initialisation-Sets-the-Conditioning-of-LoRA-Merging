# A Rate-Distortion Function for Model Merging

Code, configs, and experimental artifacts for the paper
*A Rate-Distortion Function for Model Merging*, submitted to ICLR 2027.

> **Anonymous review version** — author identities are redacted from this
> README pending review. The paper PDF is at
> `A_Rate_Distortion_Lower_Bound_for_Model_Merging.pdf`.

## What the paper does

1. Proves the first **rate-distortion lower bound** for LoRA model
   merging: worst-task NLL excess splits into an *irreducible
   task-overlap floor* and an *exponentially-decaying compression term*.
2. Provides a matching achievability via **Hadamard-incoherent
   orthogonal mixing + scalar quantization**, with a regularized
   variant (`rd-encoder ridge`) that ties or beats published methods on
   every base we test.
3. Validates the result on a **4-base × 7-method × 2-downstream-metric**
   matrix spanning Llama-3.1-Instruct, Mistral-7B-v0.3, Qwen-2.5-7B,
   and Yi-1.5-9B-Chat.
4. Derives a **5-rule practitioner decision tree** (§6.7) with a
   measurable base-saturation diagnostic that predicts when classical
   methods like TIES invert from best to worst.

## Repository layout

```
paper/
  main.tex              # paper entry point (compiles with pdflatex)
  sections/             # section fragments, including §6.2–§6.8 + Td2 appendix
  references.bib
code/phase3/
  merging/              # 9 merge methods incl. rd_encoder, fisher_avg, della
  eval/                 # evaluation drivers (NLL excess + downstream metrics)
  training/             # LoRA training pipeline
  scripts/              # analyzers, figure makers, PBS launchers
  configs/              # per-cell yaml configs for every experiment
results/phase3/         # per-cell JSON outputs + summary jsons
paper_artifacts/figures/# generated figures including Figure 1 (hero)
```

## Reproducing the empirical results

Every experiment in the paper has a one-line reproducer.
Table~tab:repro-manifest in `paper/sections/reproducibility.tex` lists
the per-experiment reproducer script + per-cell wallclock.

### Quick start (single cell)

```bash
# Set up environment
conda create -n rdmerge python=3.11 -y
conda activate rdmerge
pip install -r requirements.txt
export PYTHONNOUSERSITE=1
export PROJECT_ROOT=$(pwd)

# Run one cell: Llama-3.1-8B + TIES on the T=4 matrix at seed 1
python code/phase3/eval/run_eval_cell.py \
    --config code/phase3/configs/eval_matrix_seeds/llama31_8b__ties__seed1.yaml
```

### Headline experiments

| Experiment | Reproducer |
|---|---|
| §6.1 multi-seed matrix (40 cells) | `for f in code/phase3/configs/eval_matrix_seeds/*.yaml; do python code/phase3/eval/run_eval_cell.py --config $f; done` |
| §6.5 GSM8K downstream (20 cells)  | `for f in code/phase3/configs/eval_e3_gsm8k/*.yaml; do python code/phase3/eval/run_downstream_cell.py --config $f; done` |
| §6.5 HumanEval pass@1 (20 cells)  | `for f in code/phase3/configs/eval_b4_humaneval/*.yaml; do python code/phase3/eval/run_downstream_cell.py --config $f; done` |
| §6.6 cross-model T-scaling (108 cells) | orchestrator manifest `configs/e6_pilot_eval_manifest.json` |
| §6.8 added baselines (8 cells)    | orchestrator manifest `configs/e10_baselines_manifest.json` |
| §6.8 quadratic-bridge (16+10 cells) | manifests `e11_quadbridge_manifest.json` + `e11b_finer_alpha_manifest.json` |

### Analyzing results

After all cells in an experiment complete, run the analyzer scripts:

```bash
python code/phase3/scripts/analyze_e3_gsm8k.py        # → results/phase3/e3_gsm8k_summary.json
python code/phase3/scripts/analyze_b4_humaneval.py    # → b4_humaneval_summary.json
python code/phase3/scripts/analyze_e6_T_scaling.py    # → e6_T_scaling_summary.json
python code/phase3/scripts/analyze_e10_baselines.py
python code/phase3/scripts/analyze_e11_quadbridge.py
python code/phase3/scripts/compute_bootstrap_cis.py
python code/phase3/scripts/make_figure1_hero.py       # → paper_artifacts/figures/figure1_*
```

## Compute footprint

The full reproduction takes approximately **80–120 GPU-hours** on a
single NVIDIA A100-80GB, plus **~12 GPU-hours** of LoRA training of
the 16+12 task adapters at $T = 4$ and the 6 pilot adapters at $T = 7$.

## Citation

```bibtex
@inproceedings{anonymous2027ratedistortion,
  title  = {A Rate-Distortion Function for Model Merging},
  author = {Anonymous},
  booktitle = {Proceedings of the International Conference on Learning
              Representations (ICLR)},
  year   = {2027}
}
```

## License

This code is released under the MIT License (see `LICENSE`).

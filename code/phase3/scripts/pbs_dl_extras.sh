#!/bin/bash
#PBS -N rdm_dl_extras
#PBS -q workq
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=24:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR
source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh

MODELS_DIR=$PROJECT_ROOT/models
mkdir -p "$MODELS_DIR"
echo "[$(date)] extras download on $(hostname)"

for repo_outdir in \
    "mistralai/Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.3" \
    "01-ai/Yi-1.5-9B-Chat Yi-1.5-9B-Chat" \
    "google/gemma-3-12b-it gemma-3-12b-it" \
    "Qwen/Qwen2.5-14B-Instruct Qwen2.5-14B-Instruct"; do
    read repo outdir <<< "$repo_outdir"
    echo ""
    echo "=== $(date) — $repo -> $MODELS_DIR/$outdir ==="
    bash $PROJECT_ROOT/code/phase3/scripts/dl_model_curl.sh "$repo" "$MODELS_DIR/$outdir" \
        || { echo "FAILED on $repo; continuing"; continue; }
done

echo "[$(date)] EXTRAS_DOWNLOAD_DONE"

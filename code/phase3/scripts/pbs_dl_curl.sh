#!/bin/bash
#PBS -N rdm_dl_curl
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=08:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR
source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh

MODELS_DIR=$PROJECT_ROOT/models
mkdir -p "$MODELS_DIR"

echo "[$(date)] hostname=$(hostname)"

bash $PROJECT_ROOT/code/phase3/scripts/dl_model_curl.sh \
    meta-llama/Llama-3.1-8B-Instruct \
    "$MODELS_DIR/Llama-3.1-8B-Instruct"

bash $PROJECT_ROOT/code/phase3/scripts/dl_model_curl.sh \
    Qwen/Qwen2.5-7B-Instruct \
    "$MODELS_DIR/Qwen2.5-7B-Instruct"

echo "[$(date)] ALL_MODELS_DONE"

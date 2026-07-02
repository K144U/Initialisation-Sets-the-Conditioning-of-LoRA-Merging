#!/bin/bash
#PBS -N rdm_e3smk
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=02:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E3 smoke: one cell — qwen25_7b + TIES + GSM8K em, n=100.
# Validates the downstream-cell pipeline end-to-end before scaling
# to the full E3 matrix.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1

PICKED_GPU=$(python $PROJECT_ROOT/code/phase3/utils/gpu_picker.py --min-free-gb 25)
[ -z "$PICKED_GPU" ] && { echo "no GPU with >=25GB free" >&2; exit 87; }
export CUDA_VISIBLE_DEVICES=$PICKED_GPU
echo "[$(date '+%F %T')] e3 smoke pinned to GPU $PICKED_GPU"

python $PROJECT_ROOT/code/phase3/eval/run_downstream_cell.py \
    --config code/phase3/configs/eval_e3_smoke/qwen25_7b__ties__gsm8k_em.yaml

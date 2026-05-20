#!/bin/bash
#PBS -N rdm_merge_smoke
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=00:15:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -e /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR
source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge

PICKED_GPU=$(python $PROJECT_ROOT/code/phase3/utils/gpu_picker.py --min-free-gb 20)
[ -z "$PICKED_GPU" ] && { echo "no GPU with >=20GB free; aborting" >&2; exit 87; }
export CUDA_VISIBLE_DEVICES=$PICKED_GPU
echo "[$(date)] merge_smoke pinned to GPU $PICKED_GPU"

python $PROJECT_ROOT/code/phase3/merging/smoke_gpu.py

#!/bin/bash
#PBS -N rdm_cuda_probe
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:05:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -e /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR
source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge

PICKED_GPU=$(python $PROJECT_ROOT/code/phase3/utils/gpu_picker.py --min-free-gb 4)
[ -z "$PICKED_GPU" ] && { echo "no GPU free" >&2; exit 87; }
export CUDA_VISIBLE_DEVICES=$PICKED_GPU

python $PROJECT_ROOT/code/phase3/scripts/cuda_probe.py

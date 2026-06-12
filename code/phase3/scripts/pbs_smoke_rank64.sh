#!/bin/bash
#PBS -N rdm_smoke64
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=40gb
#PBS -l walltime=01:30:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1

# GPU6 has the most free VRAM and only light tenants; fixed pin, no picker.
export CUDA_VISIBLE_DEVICES=6
echo "[$(date '+%F %T')] smoke64 job=$PBS_JOBID gpu=$CUDA_VISIBLE_DEVICES"
python $PROJECT_ROOT/code/phase3/scripts/smoke_rank64.py
echo "[$(date '+%F %T')] smoke64 exit=$?"

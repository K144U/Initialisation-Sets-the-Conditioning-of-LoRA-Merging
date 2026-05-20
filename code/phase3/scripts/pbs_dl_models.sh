#!/bin/bash
#PBS -N rdm_dl_models
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=02:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -e /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR
source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda
source activate $PROJECT_ROOT/.conda/envs/rdmerge

# Doesn't need GPU pinning — this is a network-bound job.
# We're on the gpu queue only because compute nodes have better network access.
echo "[$(date)] starting download from $(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/dl_models.py
echo "[$(date)] download exit code $?"

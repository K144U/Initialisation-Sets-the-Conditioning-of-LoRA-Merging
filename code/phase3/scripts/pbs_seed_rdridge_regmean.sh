#!/bin/bash
#PBS -N rdm_s3rr
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=03:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# 3-seed apples-to-apples: rd-encoder ridge (l=0.13) vs tuned RegMean (l=0.01)
# on Qwen-2.5-7B + Yi-1.5-9B, on the matched seed1/2/3 adapters.
# 2 bases x 3 seeds x 2 methods = 12 cells. Wide GPU set (user-authorized).

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_seed_rdridge_regmean.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/seed_rdridge_regmean_manifest.json}"
echo "[$(date '+%F %T')] rdm_s3rr job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

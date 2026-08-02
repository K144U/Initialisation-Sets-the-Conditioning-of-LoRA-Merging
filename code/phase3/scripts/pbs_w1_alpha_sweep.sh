#!/bin/bash
#PBS -N rdm_w1a
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=06:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# W1: the merge-coefficient control the review demands.
#   28 cells  TA with weights [alpha]*T, alpha in {0.10 .. 1.00}, 4 bases, seed1
#   12 cells  rd-encoder ridge at lambda*, renorm="ta" (TA's norm, same direction)
#   12 cells  rd-encoder ridge at lambda*, realize="rank_r" (rank 16, storage parity)
# 52 cells at ~12-18 min => ~4-5 h wallclock over 6 lanes. Walltime 6 h; the
# orchestrator skips done cells on requeue, so a walltime kill is safe.
#
# Generate the cells first (idempotent):
#   python code/phase3/scripts/gen_w1_alpha_sweep.py
# Then:
#   qsub code/phase3/scripts/pbs_w1_alpha_sweep.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_w1_alpha.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/w1_alpha_manifest.json}"
echo "[$(date '+%F %T')] rdm_w1a job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

#!/bin/bash
#PBS -N rdm_a1rt
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# A1 replication stage 1: train the indep2 and indep3 cohorts, 32 adapters
# (2 cohorts x 4 bases x 4 tasks), each with an independent per-task LoRA A
# draw. Gives the step-0 matrix three seeds instead of one.
#
# ~25 min per adapter over 6 lanes => ~14 h, so expect one walltime kill and a
# keeper requeue. Done-file resume makes that free.
#
#   python code/phase3/scripts/gen_a1_indep_replicate.py
#   qsub code/phase3/scripts/pbs_a1_rep_train.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_a1_rep_train.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/a1_rep_train_manifest.json}"
echo "[$(date '+%F %T')] rdm_a1rt job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

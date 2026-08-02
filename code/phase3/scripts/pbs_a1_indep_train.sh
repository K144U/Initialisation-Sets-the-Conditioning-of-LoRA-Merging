#!/bin/bash
#PBS -N rdm_a1tr
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# A1 stage 1: retrain 16 adapters (4 bases x 4 tasks) with a DIFFERENT LoRA
# init seed per task, so the task subspaces are no longer forced to coincide
# by a shared A initialisation. The data-shuffle seed is held fixed at
# 20260518 via seeds.data, which both removes the confound and matches the
# eval cells, closing the train/eval overlap on alpaca and magicoder.
#
# ~25 min per adapter => ~7 h over 6 lanes. Walltime 12 h; done-file resume
# makes a walltime kill safe.
#
#   python code/phase3/scripts/gen_a1_indep_init.py
#   qsub code/phase3/scripts/pbs_a1_indep_train.sh
# then stage 2 (CPU):
#   python code/phase3/scripts/measure_subspace_geometry.py --cohort indep1

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_a1_indep_train.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/a1_indep_train_manifest.json}"
echo "[$(date '+%F %T')] rdm_a1tr job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

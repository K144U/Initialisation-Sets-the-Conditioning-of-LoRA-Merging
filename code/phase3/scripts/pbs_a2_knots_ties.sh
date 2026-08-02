#!/bin/bash
#PBS -N rdm_a2kt
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=03:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# A2: KnOTS with inner_combination="ties", the variant that is not
# algebraically Task Arithmetic. 4 bases x 3 seeds = 12 cells, ~1 h over
# 6 lanes.
#
#   python code/phase3/scripts/gen_a2_knots_ties.py
#   qsub code/phase3/scripts/pbs_a2_knots_ties.sh
#   python code/phase3/scripts/analyze_a2_knots_ties.py

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_a2_knots_ties.json
export GPUS=0,1,2,3,4,6

# The identity that motivates this run must still hold on this checkout.
python code/phase3/merging/tests/test_knots.py || {
  echo "[a2] ABORT: KnOTS tests failed"; exit 2; }

MANIFEST="${MANIFEST:-code/phase3/configs/a2_knots_ties_manifest.json}"
echo "[$(date '+%F %T')] rdm_a2kt job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

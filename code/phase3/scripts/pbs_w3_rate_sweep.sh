#!/bin/bash
#PBS -N rdm_w3rt
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=06:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# W3: rd-encoder ridge at finite rate, b in {1,2,3,4,8,16} at each base's
# lambda*, realize=rank_deff, seed1. 6 rates x 4 bases = 24 cells, ~2 h over
# 6 lanes. b=32 (b -> inf) already exists in the published cells.
#
# This is the experiment the review asks for and that has never been run:
# every rd-ridge cell in the paper is bits=32, so the rate axis is untested
# with the ridge on. The published lambda=0 sweep in eval_e1/ is non-monotone
# on 4 of 4 bases, but it conflates the sliver blow-up with the rate axis.
#
#   python code/phase3/scripts/gen_w3_rate_sweep.py
#   qsub code/phase3/scripts/pbs_w3_rate_sweep.sh
#   python code/phase3/scripts/analyze_w3_rate.py

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_w3_rate.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/w3_rate_manifest.json}"
echo "[$(date '+%F %T')] rdm_w3rt job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

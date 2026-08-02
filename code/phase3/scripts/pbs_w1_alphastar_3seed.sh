#!/bin/bash
#PBS -N rdm_w1s
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=04:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# W1 follow-up: TA at each base's alpha* for seeds 2 and 3, so the decisive
# Llama comparison becomes 3-seed vs 3-seed instead of seed1 TA vs 3-seed
# rd-ridge. 8 cells.
#
#   python code/phase3/scripts/gen_w1_alphastar_3seed.py
#   qsub code/phase3/scripts/pbs_w1_alphastar_3seed.sh
#   python code/phase3/scripts/w1_verdict_3seed.py

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_w1_alphastar.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/w1_alphastar_3seed_manifest.json}"
echo "[$(date '+%F %T')] rdm_w1s job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

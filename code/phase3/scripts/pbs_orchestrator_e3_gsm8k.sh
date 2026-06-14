#!/bin/bash
#PBS -N rdm_e3gsm
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=24:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E3 GSM8K em sweep: 20 cells = 4 models x 5 methods at n=500.
# Methods: TA, TIES, DARE, KnOTS, TVQ b=2 (the dip champion).
# Isolated sentinel/state. GPUs 2,4,6 default (or expanded set).
# Each cell ~1.5h on 7B model with greedy gen on 500 GSM8K prompts.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E3_GSM8K_COMPLETE
export ORCH_STATE=orchestrator_state_e3_gsm8k.json

MANIFEST="${MANIFEST:-code/phase3/configs/e3_gsm8k_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E3" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E3)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e3gsm job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

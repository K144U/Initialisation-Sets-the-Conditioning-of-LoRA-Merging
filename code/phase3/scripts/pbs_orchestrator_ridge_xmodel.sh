#!/bin/bash
#PBS -N rdm_xmodel
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=24:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Cross-model ridge sweep orchestrator. 12 cells = 3 models x 4 lambdas.
# Isolated sentinel + state file so it cannot clobber the matrix or
# ridge orchestrators. GPUs from _ORCH_GPUS_XMODEL if present, else 2,4,6.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_RIDGE_XMODEL_COMPLETE
export ORCH_STATE=orchestrator_state_ridge_xmodel.json

MANIFEST="${MANIFEST:-code/phase3/configs/ridge_xmodel_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_XMODEL" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_XMODEL)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_xmodel job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

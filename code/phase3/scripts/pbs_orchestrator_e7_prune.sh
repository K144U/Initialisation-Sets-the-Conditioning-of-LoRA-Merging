#!/bin/bash
#PBS -N rdm_e7prune
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E7 Phase 2b orchestrator: 8 eval cells (4 models x 2 densities) for
# the matched-sparsity magnitude pruning test. density=0.5 matches b=2
# zero-bucket sparsity; density=0.13 matches destroyed-signal fraction.
# Isolated sentinel/state, GPUs 2,4,6 default (or expanded set).

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E7_PRUNE_COMPLETE
export ORCH_STATE=orchestrator_state_e7_prune.json

MANIFEST="${MANIFEST:-code/phase3/configs/e7_prune_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E7" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E7)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e7prune job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

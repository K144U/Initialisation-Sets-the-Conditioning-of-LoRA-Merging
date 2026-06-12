#!/bin/bash
#PBS -N rdm_fisher
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=24:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Fisher-diag smoke orchestrator. Uses its own sentinel + state file so it
# cannot clobber the matrix (orchestrator.py default state) or the ridge
# orchestrator (pbs_orchestrator_ridge.sh). GPU comes from _ORCH_GPUS_FISHER
# if present, else GPU6 (shared with ridge fine sweep until that ends).

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1   # LMCA user-site torch 2.4.1 shadows the conda stack
export ORCH_SENTINEL=_FISHER_COMPLETE
export ORCH_STATE=orchestrator_state_fisher.json

MANIFEST="${MANIFEST:-code/phase3/configs/fisher_smoke_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_FISHER" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_FISHER)
else
    export GPUS=6
fi

echo "[$(date '+%F %T')] rdm_fisher job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

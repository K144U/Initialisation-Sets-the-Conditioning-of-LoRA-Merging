#!/bin/bash
#PBS -N rdm_orch
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=24:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Multi-GPU orchestrator wrapper. MANIFEST may be passed via qsub -v;
# defaults to the E2 manifest. GPU set comes from the _ORCH_GPUS file at
# the project root (edit + requeue to change), default 0,1,2,3,5.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1   # LMCA user-site torch 2.4.1 shadows the conda stack

MANIFEST="${MANIFEST:-code/phase3/configs/e2_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS" ]; then
    export GPUS=$(cat "$PROJECT_ROOT/_ORCH_GPUS" | tr -d '[:space:]')
else
    export GPUS="0,1,2,3,5"
fi

echo "[$(date '+%F %T')] rdm_orch job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

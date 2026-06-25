#!/bin/bash
#PBS -N rdm_e12bl
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=4:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Phase 1 steps 5+6 — RegMean + AdaMerging on the T=4 §6.1 matrix.
# 4 bases x 2 methods = 8 cells. Defensive baselines for the two
# closest published structural relatives of rd-encoder ridge.
#
# Compute: ~8 cells x ~15 min each / 3 lanes = ~45 min wallclock.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E12_REGMEAN_ADAMERGING_COMPLETE
export ORCH_STATE=orchestrator_state_e12.json

MANIFEST="${MANIFEST:-code/phase3/configs/e12_regmean_adamerging_manifest.json}"
# Pinned to GPU 6 by default for parallel dispatch alongside the seed3
# chain on GPUs 2,4. Override via _ORCH_GPUS_E12 if needed.
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E12" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E12)
else
    export GPUS=6
fi

echo "[$(date '+%F %T')] rdm_e12bl job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

#!/bin/bash
#PBS -N rdm_e11qb
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=6:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E11 — Quadratic-bridge: verify rd-encoder ridge approaches the
# Fisher-quadratic (diagonal Fisher proxy) merge as adapter norms
# shrink. 2 bases (Llama-3.1, Yi-1.5-Chat) x 4 delta_scale values
# {0.1, 0.25, 0.5, 1.0} x 2 methods (rd_encoder ridge, fisher_avg)
# = 16 cells, NLL excess metric.
# Compute: ~16 cells x ~15 min each / 3 lanes = ~80 min.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E11_QUADBRIDGE_COMPLETE
export ORCH_STATE=orchestrator_state_e11_quadbridge.json

MANIFEST="${MANIFEST:-code/phase3/configs/e11_quadbridge_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E6" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E6)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e11qb job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

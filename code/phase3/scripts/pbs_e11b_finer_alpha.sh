#!/bin/bash
#PBS -N rdm_e11b
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=4:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E11b — Finer alpha scan on Llama-3.1 for the quadratic-bridge probe.
# Closes the "Llama partial" caveat in §6.8 by adding alpha in
# {0.05, 0.075, 0.125, 0.15, 0.20} on Llama-3.1 only (Yi already has
# clean bridge convergence at alpha=0.10).
# 10 cells total. Compute: ~10 cells x ~10 min each / 3 lanes = ~35 min.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E11B_FINER_ALPHA_COMPLETE
export ORCH_STATE=orchestrator_state_e11b_finer_alpha.json

MANIFEST="${MANIFEST:-code/phase3/configs/e11b_finer_alpha_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E6" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E6)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e11b job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

#!/bin/bash
#PBS -N rdm_misT7
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=8:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Phase 1 step 3 — Mistral T-scaling sweep matching the Yi/Llama E6
# structure. 54 cells (T=2 × 4 subsets + T=4 × 4 subsets + T=7 × 1
# subset) × 6 methods on Mistral-7B-Instruct-v0.3. Tests the
# pre-registered TIES sign-election threshold prediction at T=7
# (Mistral R = 0.0326, inside Td2 window [0.025, 0.075] -> predicts
# ambiguous TIES; commit 537796a anchors the prediction).
#
# PRE-REQUISITE: rdm_mpilt (pbs_mistral_pilot_train.sh) must have
# completed — the 3 Mistral pilot adapter dirs (codealpaca, dolly,
# xsum at artifacts/lora/mistral_7b/e6_pilot/{task}/v1) must exist.
#
# Compute: ~54 cells × ~12 min each / 3 lanes = ~3.5h wallclock.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_MISTRAL_T7_COMPLETE
export ORCH_STATE=orchestrator_state_mistral_t7.json

MANIFEST="${MANIFEST:-code/phase3/configs/mistral_t7_eval_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E6" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E6)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_misT7 job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

#!/bin/bash
#PBS -N rdm_rgmlam
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=04:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Experiment B -- RegMean ridge_lambda sweep (defend/retract centroid claim).
# 4 bases x 5 lambdas {1e-4,1e-2,1e-1,1,10} = 20 cells.
# (lambda=1e-3 reused from the existing e12 cells, not re-run.)
# Finds RegMean's best regularizer; rd-encoder ridge is compared against it.
# Compute: ~20 cells x ~15 min / 3 lanes = ~1.7h wallclock.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_REGMEAN_LAMBDA_SWEEP_COMPLETE
export ORCH_STATE=orchestrator_state_regmean_lambda.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/regmean_lambda_sweep_manifest.json}"
echo "[$(date '+%F %T')] rdm_rgmlam job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

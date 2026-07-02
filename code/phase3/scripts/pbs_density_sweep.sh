#!/bin/bash
#PBS -N rdm_densw
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=06:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Experiment A -- TIES/DARE density sweep (fairness robustness check).
# 4 bases x 2 methods x 4 densities {0.05,0.1,0.3,0.5} = 32 cells.
# (density=0.2 reused from the existing matrix cells, not re-run.)
# NLL-excess metric, seed1, identical eval pipeline to the main matrix.
# Compute: ~32 cells x ~15 min / 3 lanes = ~2.5h wallclock.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_DENSITY_SWEEP_COMPLETE
export ORCH_STATE=orchestrator_state_density_sweep.json
export GPUS=2,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/density_sweep_manifest.json}"
echo "[$(date '+%F %T')] rdm_densw job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"

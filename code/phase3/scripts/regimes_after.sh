#!/bin/bash
#PBS -N r_regev
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V
# After the regime trainings: generate and run their 17 evaluation cells, then
# measure every cohort's geometry (notes/prereg_training_regimes_2026-10-19.md).
# Submitted held behind the last training job (depend=afterany), so it does
# not depend on a watcher on the local machine. If the 60 adapters are not all
# there it exits without launching anything.
#
#   qsub -W depend=afterany:<last reg job> -l select=1:ncpus=3:ngpus=2:mem=60gb \
#        -o ~/projects/rdmerge_v2/logs/pbs/ code/phase3/scripts/regimes_after.sh
set -uo pipefail
cd "$HOME/projects/rdmerge_v2"
export PYTHONNOUSERSITE=1 RDMERGE_ROOT=$HOME/projects/rdmerge_v2
P=$RDMERGE_ROOT/.conda/envs/rdmerge/bin/python
n=$(ls results/phase3/lora_train_regimes/ | wc -l)
echo "[regev $(date '+%F %T')] $n of 60 regime adapters trained"
[ "$n" -ge 60 ] || { echo "[regev] training incomplete, nothing launched"; exit 3; }
(cd code/phase3/scripts && $P gen_training_regimes.py eval) || exit 4
export MANIFEST=code/phase3/configs/regimes_eval_manifest.json ORCH_STATE=state_regev_0.json \
       ORCH_SHARD=0/1 ORCH_SENTINEL=_DONE_regev_0 ORCH_WORKERS=2
bash code/phase3/scripts/pbs_v2.sh
echo "[regev $(date '+%F %T')] evaluation finished; measuring geometry"
$P -u code/phase3/scripts/measure_regimes.py --cohorts
echo "[regev $(date '+%F %T')] done"

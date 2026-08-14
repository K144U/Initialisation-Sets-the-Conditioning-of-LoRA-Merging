#!/bin/bash
# Keeper for the R6 overnight sweep.
#
# The pattern that recovered two walltime deaths in the August campaign: a
# detached loop that resubmits the job whenever the queue is empty and the
# manifest is not yet satisfied. The orchestrator skips cells whose done
# marker exists, so a resubmission resumes rather than restarts.
#
# R6 is projected at roughly 5 to 6 hours against a 12 hour walltime, so this
# should never fire. It exists because the projection is a projection, and
# because an overnight run that dies at hour 11 with nobody watching costs a
# day.
#
#   setsid nohup bash code/phase3/scripts/keeper_r6.sh > logs/keeper_r6.log 2>&1 &
#
# Stop it by deleting the stop file's absence, i.e. create it:
#   touch _KEEPER_R6_STOP

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT

TARGET=72
MANIFEST=code/phase3/configs/r6_t3_manifest.json
RESDIR=results/phase3/eval_r6_t3
MAX_RESUBMITS=6
n_resub=0

while true; do
  if [ -f _KEEPER_R6_STOP ]; then
    echo "[$(date '+%F %T')] stop file present, keeper exiting"
    exit 0
  fi

  done_n=$(ls $RESDIR/*.json 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$TARGET" ]; then
    echo "[$(date '+%F %T')] $done_n/$TARGET cells, R6 complete, keeper exiting"
    exit 0
  fi

  running=$(qstat -u sanjay.g 2>/dev/null | grep -c " R \| Q ")
  if [ "$running" -eq 0 ]; then
    if [ "$n_resub" -ge "$MAX_RESUBMITS" ]; then
      echo "[$(date '+%F %T')] $done_n/$TARGET but $n_resub resubmits already;"
      echo "  refusing to loop. Something is failing that a resubmit will not fix."
      exit 2
    fi
    n_resub=$((n_resub + 1))
    echo "[$(date '+%F %T')] queue empty at $done_n/$TARGET, resubmit #$n_resub"
    qsub -v ARM=r6,GPUS_FILE=_ORCH_GPUS_R6 code/phase3/scripts/pbs_wave2.sh
  else
    echo "[$(date '+%F %T')] $done_n/$TARGET cells, $running job(s) live"
  fi
  sleep 900
done

#!/bin/bash
# Keeper for the drift-cohort run (80 cells).
#
# Same pattern that recovered two walltime deaths in the August campaign: a
# detached loop that resubmits whenever the queue is empty and the manifest is
# not yet satisfied. The orchestrator skips cells whose done marker exists, so
# a resubmission resumes rather than restarts.
#
# Unlike R6, this one is expected to fire. Three to four cells run in parallel
# and a HumanEval cell takes tens of minutes, so 80 cells will not fit in one
# 12 hour walltime.
#
#   setsid nohup bash code/phase3/scripts/keeper_heur_downstream.sh \
#     > logs/keeper_heur_downstream.log 2>&1 &
#
# Stop it with:  touch _KEEPER_DRIFT_STOP

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT

TARGET=8
MAX_RESUBMITS=8
n_resub=0
last_done=-1
stalls=0

while true; do
  if [ -f _KEEPER_DRIFT_STOP ]; then
    echo "[$(date '+%F %T')] stop file present, keeper exiting"
    exit 0
  fi

  done_n=$(ls artifacts/lora/llama31_8b/*/drift_*/adapter_A0.safetensors 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$TARGET" ]; then
    echo "[$(date '+%F %T')] $done_n/$TARGET cells complete."
    echo "  Rules: notes/prereg_drift_2026-08-16.md (9202d3b)."
    echo "  Now run: python code/phase3/scripts/analyze_drift.py"
    echo "  Do NOT read individual cells first; binding constraint 1."
    exit 0
  fi

  running=$(qstat -u sanjay.g 2>/dev/null | grep -c " R \| Q ")
  if [ "$running" -eq 0 ]; then
    if [ "$n_resub" -ge "$MAX_RESUBMITS" ]; then
      echo "[$(date '+%F %T')] $done_n/$TARGET after $n_resub resubmits;"
      echo "  refusing to loop. Check logs/orch for a cell that fails every time."
      exit 2
    fi
    n_resub=$((n_resub + 1))
    echo "[$(date '+%F %T')] queue empty at $done_n/$TARGET, resubmit #$n_resub"
    qsub code/phase3/scripts/pbs_drift_cohort.sh
  else
    # A live job that stops producing cells is worth flagging: it means a cell
    # is hanging rather than running, which a resubmit will not fix.
    if [ "$done_n" -eq "$last_done" ]; then
      stalls=$((stalls + 1))
    else
      stalls=0
    fi
    if [ "$stalls" -ge 8 ]; then
      echo "[$(date '+%F %T')] WARNING $done_n/$TARGET unchanged for 2 hours"
      echo "  with a job live. Look for a hung cell in logs/orch."
      stalls=0
    fi
    echo "[$(date '+%F %T')] $done_n/$TARGET cells, $running job(s) live"
  fi
  last_done=$done_n
  sleep 900
done

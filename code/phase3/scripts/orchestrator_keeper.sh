#!/bin/bash
# rdmerge orchestrator keeper — login-node nohup loop (LMCA/MOOLoRa pattern).
# Every 30 min: if no rdm_orch job is queued/running and the queue is not
# complete, requeue the orchestrator (resume-safe: done cells are skipped).
# Stop: touch _KEEPER_STOP. Completion: orchestrator writes _QUEUE_COMPLETE.

PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs
echo $$ > logs/orch_keeper.pid
LOG=logs/orch_keeper.log

log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "keeper started pid $$"

while true; do
    if [ -f _KEEPER_STOP ]; then
        log "stop sentinel found — exiting"; exit 0
    fi
    if [ -f _QUEUE_COMPLETE ]; then
        log "queue complete — exiting"; exit 0
    fi
    if ! /opt/pbs/bin/qstat -u sanjay.g 2>/dev/null | grep -q rdm_orch; then
        JID=$(/opt/pbs/bin/qsub code/phase3/scripts/pbs_orchestrator.sh 2>>$LOG)
        log "no orchestrator running and queue incomplete -> resubmitted: $JID"
    fi
    sleep 1800
done

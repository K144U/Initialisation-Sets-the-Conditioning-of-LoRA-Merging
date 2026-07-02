#!/bin/bash
# rdmerge orchestrator keeper — login-node nohup loop (LMCA/MOOLoRa pattern).
# Every 30 min: if no rdm_orch job is queued/running and the queue is not
# complete, requeue the orchestrator (resume-safe: done cells are skipped).
# Stop: touch _KEEPER_STOP. Completion: orchestrator writes _QUEUE_COMPLETE.
#
# _QUEUE_COMPLETE is VERIFIED against manifest done-files before being
# honored: jobs launched before the ORCH_SENTINEL patch (e.g. the ridge
# sweep 41521) write the same default file on completion and must not stop
# the keeper mid-matrix. If the sentinel is present but done-files < total,
# it is treated as stray and removed.

PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
MANIFEST=code/phase3/configs/all_manifest.json
cd $PROJECT_ROOT
mkdir -p logs
echo $$ > logs/orch_keeper.pid
LOG=logs/orch_keeper.log

log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "keeper v2 started pid $$ (sentinel verified against $MANIFEST done-files)"

while true; do
    if [ -f _KEEPER_STOP ]; then
        log "stop sentinel found — exiting"; exit 0
    fi
    if [ -f _QUEUE_COMPLETE ]; then
        CNT=$(python3 -c 'import json,os; c=json.load(open("'$MANIFEST'")); print(sum(1 for x in c if os.path.exists(x["done"])), len(c))' 2>/dev/null)
        if [ -n "$CNT" ] && [ "${CNT% *}" = "${CNT#* }" ]; then
            log "queue complete (verified ${CNT% *}/${CNT#* } done-files) — exiting"; exit 0
        fi
        rm -f _QUEUE_COMPLETE
        log "stray _QUEUE_COMPLETE removed (done-files ${CNT:-unreadable} of total — matrix incomplete); continuing"
    fi
    if ! /opt/pbs/bin/qstat -u sanjay.g 2>/dev/null | grep -q rdm_orch; then
        JID=$(/opt/pbs/bin/qsub code/phase3/scripts/pbs_orchestrator.sh 2>>$LOG)
        log "no orchestrator running and queue incomplete -> resubmitted: $JID"
    fi
    sleep 1800
done

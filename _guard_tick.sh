#!/bin/bash
# Sentinel guard + status for the rdmerge seed matrix.
# Matrix job is named rdm_orch; ridge job (rdm_ridge) SHARES
# logs/orchestrator_state.json AND writes the default _QUEUE_COMPLETE, so we
# judge matrix completeness ONLY by all_manifest done-files (ridge-proof) and
# grep job presence by name (job-agnostic across keeper requeues).
cd /home/sanjay.g/projects/rdmerge || exit 0
CNT=$(python3 -c 'import json,os; c=json.load(open("code/phase3/configs/all_manifest.json")); print(sum(1 for x in c if os.path.exists(x["done"])), len(c))' 2>/dev/null)
MDONE=${CNT% *}; MTOT=${CNT#* }
ORCH=$(/opt/pbs/bin/qstat -u sanjay.g 2>/dev/null | grep -c rdm_orch)
RIDGE=$(/opt/pbs/bin/qstat -u sanjay.g 2>/dev/null | grep -c rdm_ridge)
GUARD=none
if [ -f _QUEUE_COMPLETE ]; then
  if [ -n "$MDONE" ] && [ "$MDONE" -lt "$MTOT" ] 2>/dev/null; then
    rm -f _QUEUE_COMPLETE; GUARD=removed
  else
    GUARD=legit
  fi
fi
QF=no; [ -f _QUEUE_FAILED ] && QF=yes
echo "MDONE=${MDONE:-NA} MTOT=${MTOT:-NA} ORCH=$ORCH RIDGE=$RIDGE GUARD=$GUARD QF=$QF"

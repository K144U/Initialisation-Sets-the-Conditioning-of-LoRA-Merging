#!/bin/bash
# Sentinel guard + status tick for matrix job 41533 (re-keyed after 41524 qdel).
# Removes a STRAY _QUEUE_COMPLETE (written by the pre-patch ridge job 41521)
# while the matrix still has pending cells, so the keeper is not tricked into
# exiting. A legit matrix completion has pending==0 -> sentinel is preserved.
cd /home/sanjay.g/projects/rdmerge || exit 0
ST=logs/orchestrator_state.json
PEND=$(python3 -c "import json;print(json.load(open(\"$ST\"))[\"pending\"])" 2>/dev/null)
DONE=$(python3 -c "import json;print(len(json.load(open(\"$ST\"))[\"done\"]))" 2>/dev/null)
FAILED=$(python3 -c "import json;print(\",\".join(json.load(open(\"$ST\"))[\"failed\"]))" 2>/dev/null)
ORCH=$(/opt/pbs/bin/qstat -u sanjay.g 2>/dev/null | grep -c rdm_orch)
RIDGE=$(/opt/pbs/bin/qstat -u sanjay.g 2>/dev/null | grep -c rdm_ridge)
GUARD=none
if [ -f _QUEUE_COMPLETE ]; then
  if [ "$PEND" != "0" ] && [ -n "$PEND" ]; then rm -f _QUEUE_COMPLETE; GUARD=removed; else GUARD=legit; fi
fi
echo "PEND=${PEND:-NA} DONE=${DONE:-NA} ORCH=$ORCH RIDGE=$RIDGE GUARD=$GUARD FAILED=[$FAILED]"

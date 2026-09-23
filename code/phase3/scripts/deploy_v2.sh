#!/bin/bash
# Deploy the local code tree to the student-account work tree.
#
# Run from the repo root on the Windows machine. core.autocrlf is true here,
# so the working copy can hold CRLF files; a CRLF shebang kills every PBS job
# at start ("/bin/bash^M: bad interpreter", exit 126), which happened once on
# 2026-09-23. Carriage returns are therefore stripped on the cluster side on
# every deploy, not left to whatever git did to the working copy.
set -euo pipefail
HOST=STUDENT-ACCOUNT@CLUSTER-HOST
tar --exclude='__pycache__' -czf /tmp/rdm_code.tgz code
scp -q -o BatchMode=yes /tmp/rdm_code.tgz "$HOST:rdm_code.tgz"
ssh -o BatchMode=yes "$HOST" 'cd ~/projects/rdmerge_v2 && tar xzf ~/rdm_code.tgz && rm ~/rdm_code.tgz &&
  grep -rlI $'"'"'\r'"'"' code | xargs -r sed -i "s/\r$//" &&
  bash -n code/phase3/scripts/pbs_v2.sh && echo "deployed, CR stripped, launcher syntax ok"'

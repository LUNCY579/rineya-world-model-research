#!/usr/bin/env bash
# Follow-up chain: once PVR stage1 (sparse) exists, capture retrospective verifier features with
# Titan-Sparse and with Supa on the SAME Titan candidates, then analyse on CPU. Each GPU job takes
# the shared lease itself.
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/bank_pilot/followup_log.txt
S1=runs/bank_pilot/pvr_sparse/stage1.jsonl
echo "=== $(date -Iseconds) waiting for $S1" >> "$LOG"
until [ -f "$S1" ]; do sleep 30; done
sleep 5
step() { echo "=== $(date -Iseconds) start $*" >> "$LOG"; "$PY" -u scripts/run_pvr.py "$@" >> "$LOG" 2>&1; local rc=$?; echo "=== $(date -Iseconds) exit rc=$rc $*" >> "$LOG"; }
[ -f runs/bank_pilot/pvr_sparse/retro_sparse.npy ] || step retro --bank bank_pilot --arm sparse --source-arm sparse
[ -f runs/bank_pilot/pvr_sparse/retro_supa.npy ] || step retro --bank bank_pilot --arm supa --source-arm sparse
step retro_analyze --bank bank_pilot --source-arm sparse --verifier-arms sparse,supa
echo "=== $(date -Iseconds) followup finished" >> "$LOG"

#!/usr/bin/env bash
# Sequential pilot GPU queue. One model per process; each job takes the shared GPU lease itself.
# Resumable: run_arm.py refuses arms that already have DONE.json; PVR steps skip if outputs exist.
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/bank_pilot/queue_log.txt
mkdir -p runs/bank_pilot
run() { echo "=== $(date -Iseconds) start $*" >> "$LOG"; "$PY" -u scripts/run_arm.py --bank data/bank_pilot.json "$@" >> "$LOG" 2>&1; local rc=$?; echo "=== $(date -Iseconds) exit rc=$rc $*" >> "$LOG"; }
pvr() { echo "=== $(date -Iseconds) start pvr $*" >> "$LOG"; "$PY" -u scripts/run_pvr.py "$@" >> "$LOG" 2>&1; local rc=$?; echo "=== $(date -Iseconds) exit rc=$rc pvr $*" >> "$LOG"; }
run --arm sparse --parity --gen
run --arm supa --parity --gen
[ -f runs/bank_pilot/pvr_sparse/stage1.jsonl ] || pvr stage1 --bank bank_pilot --arm sparse --splits dev,test
[ -f runs/bank_pilot/pvr_sparse/hstates_supa.npy ] || pvr hstates --bank bank_pilot --arm supa --source-arm sparse
run --arm randdelta
run --arm exp
run --arm transition --gen
run --arm qwen3
run --arm sparse_a0.5
run --arm sparse_a2.0
echo "=== $(date -Iseconds) queue finished" >> "$LOG"

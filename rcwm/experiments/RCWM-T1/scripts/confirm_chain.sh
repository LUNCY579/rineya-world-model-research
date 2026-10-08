#!/usr/bin/env bash
# PREREG-1 §7 step 2: confirmation GPU runs (frozen code). Then the remaining pilot GPU work.
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/confirm_log.txt
step() { echo "=== $(date -Iseconds) start $*" >> "$LOG"; "$PY" -u "$@" >> "$LOG" 2>&1; local rc=$?; echo "=== $(date -Iseconds) exit rc=$rc $*" >> "$LOG"; }
[ -f runs/bank_confirm/sparse/DONE.json ] || step scripts/run_arm.py --bank data/bank_confirm.json --arm sparse --gen --gen-splits test --no-offtask
[ -f runs/bank_confirm/supa/DONE.json ]   || step scripts/run_arm.py --bank data/bank_confirm.json --arm supa --gen --gen-splits test --no-offtask --gen-only
echo "=== $(date -Iseconds) confirmation GPU runs finished" >> "$LOG"
# afterwards: non-binding pilot regeneration, then the rest of the pilot GPU queue
bash scripts/regen_chain.sh
bash scripts/master_chain.sh

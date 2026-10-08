#!/usr/bin/env bash
# Off-task captures for arms whose main capture finished before run_arm gained the off-task step.
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/bank_pilot/followup_offtask_log.txt
for arm in supa sparse; do
  echo "=== $(date -Iseconds) start offtask $arm" >> "$LOG"
  "$PY" -u scripts/run_offtask.py --bank bank_pilot --arm "$arm" >> "$LOG" 2>&1
  echo "=== $(date -Iseconds) exit rc=$? offtask $arm" >> "$LOG"
done
echo "=== $(date -Iseconds) followup_offtask finished" >> "$LOG"

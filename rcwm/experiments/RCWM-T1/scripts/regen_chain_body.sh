#!/usr/bin/env bash
# PREREG-1 M2: regenerate pilot TEST generations under the frozen instrument (cap 160, lenient parse).
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/regen_log.txt
for arm in supa sparse; do
  [ -f "runs/bank_pilot_regen/$arm/DONE.json" ] && continue
  echo "=== $(date -Iseconds) start regen $arm" >> "$LOG"
  "$PY" -u scripts/run_arm.py --bank data/bank_pilot.json --arm "$arm" --gen --gen-splits test --gen-only --out-stem bank_pilot_regen >> "$LOG" 2>&1; rc=$?
  echo "=== $(date -Iseconds) exit rc=$rc regen $arm" >> "$LOG"
done
echo "=== $(date -Iseconds) regen chain finished" >> "$LOG"

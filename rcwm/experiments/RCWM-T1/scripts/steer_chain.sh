#!/usr/bin/env bash
# Waits for the likelihood chain to finish, then runs the mode-vector steering test (leased).
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/bank_pilot/steer_log.txt
until grep -q "lik chain finished" runs/bank_pilot/lik_log.txt 2>/dev/null; do sleep 30; done
if [ ! -f runs/bank_pilot/steer/steer_results.json ]; then
  echo "=== $(date -Iseconds) start steer" >> "$LOG"
  "$PY" -u scripts/run_steer.py --bank bank_pilot --layers 16,20,24 --alphas 1.0 >> "$LOG" 2>&1; rc=$?
  echo "=== $(date -Iseconds) exit rc=$rc steer" >> "$LOG"
fi

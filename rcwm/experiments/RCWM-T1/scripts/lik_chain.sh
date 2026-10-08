#!/usr/bin/env bash
# Likelihood-ranking captures (idempotent; each job takes the lease).
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/bank_pilot/lik_log.txt
for arm in "$@"; do
  [ -f "runs/bank_pilot/$arm/lik_rank.jsonl" ] && continue
  echo "=== $(date -Iseconds) start lik $arm" >> "$LOG"
  "$PY" -u scripts/run_lik.py --bank bank_pilot --arm "$arm" --splits dev,test >> "$LOG" 2>&1; rc=$?
  echo "=== $(date -Iseconds) exit rc=$rc lik $arm" >> "$LOG"
done
echo "=== $(date -Iseconds) lik chain finished" >> "$LOG"

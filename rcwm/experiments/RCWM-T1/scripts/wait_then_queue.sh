#!/usr/bin/env bash
# Yield the GPU to a peer job before starting the pilot queue: wait until the shared lock's label
# matches $1 (the peer job we promised to go first) or until the epoch deadline $2, then run.
set -u
cd "$(dirname "$0")/.."
LOCK=/c/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock
PATTERN="${1:-P-actions}"
DEADLINE="${2:-0}"
echo "wait_then_queue: waiting for lock label ~ '$PATTERN' or deadline $DEADLINE ($(date -Iseconds))"
while true; do
  if [ -f "$LOCK" ] && grep -q "$PATTERN" "$LOCK" 2>/dev/null; then echo "peer job holds lease ($(date -Iseconds))"; break; fi
  if [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "deadline reached ($(date -Iseconds))"; break; fi
  sleep 20
done
exec bash scripts/pilot_queue.sh

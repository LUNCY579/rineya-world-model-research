#!/usr/bin/env bash
# RCX-E01R training chain (frozen PREREG_E01R, CX D-41; SEALED commitment D-42 minted before any training).
# Trains E-R2 / E-R3 seeds 0-2 one adapter at a time, each in its own CPU-lane block (rcx/lane.py, CX D-36): acquire
# (needs >= 12.5 GB free RAM; waiting marker only if another party holds the lane), export the lane token for
# native_e01r's lane check, train, release. Order interleaves the arms by seed so matched pairs complete early.
# Smoke, open-sealed and the SEALED evals are a separate chain, after all six adapters exist (PREREG section 10).
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/RCX-E01R/cells/train_chain.log
WINPID=$(cat /proc/$$/winpid)
trap 'python -B -m rcx.lane release --pid '"$WINPID"' >> '"$LOG"' 2>&1' EXIT
echo "$(date +%T) train chain start; winpid ${WINPID}" >> $LOG
for spec in "E-R2 0" "E-R3 0" "E-R2 1" "E-R3 1" "E-R2 2" "E-R3 2"; do
  set -- $spec
  arm=$1; seed=$2; tag="${arm}_s${seed}"
  if [ -f "runs/RCX-E01R/cells/${tag}/TRAIN_META.json" ]; then echo "$(date +%T) skip ${tag} (done)" >> $LOG; continue; fi
  python -B -m rcx.lane acquire --job "RCX E01R train ${tag}" --pid "$WINPID" --expected-min 200 --need-ram 12.5 \
    >> $LOG 2>&1 || { echo "$(date +%T) lane acquire failed" >> $LOG; exit 5; }
  export RCX_LANE_TOKEN=$(python -B -c "from rcx import lane; print(lane.read_lock()['token'])")
  echo "$(date +%T) START train ${tag}" >> $LOG
  RCX_THREADS=4 python -B experiments/RCX-E01R/native_e01r.py train --arm "$arm" --seed "$seed" \
    > "runs/RCX-E01R/cells/train_${tag}.stdout.txt" 2>&1
  echo "$(date +%T) END train ${tag} rc=$?" >> $LOG
  python -B -m rcx.lane release --pid "$WINPID" >> $LOG 2>&1
done
echo "$(date +%T) TRAIN CHAIN DONE" >> $LOG

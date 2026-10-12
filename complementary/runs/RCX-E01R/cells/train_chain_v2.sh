#!/usr/bin/env bash
# RCX-E01R training chain v2 (frozen PREREG_E01R, CX D-41; SEALED commitment D-42 minted before any training).
# v2 (BWM D-84 request; replaces train_chain.sh before any block was taken): a block is taken only inside the agreed
# quiet windows (runs/RCX-E01R/quiet_windows.json; BWM D-84, RCWM D-51) and only if its expected length ends >= 5 min
# before the window closes, so no CPU model job ever runs beside a GPU bundle. One adapter per lane block (rcx/lane.py,
# CX D-36): acquire (>= 12.5 GB free RAM), export the lane token, train, release. Arms interleaved by seed.
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/RCX-E01R/cells/train_chain.log
WIN=runs/RCX-E01R/quiet_windows.json
WINPID=$(cat /proc/$$/winpid)
trap 'python -B -m rcx.lane release --pid '"$WINPID"' >> '"$LOG"' 2>&1' EXIT
echo "$(date +%T) train chain v2 start (quiet windows); winpid ${WINPID}" >> $LOG
for spec in "E-R2 0" "E-R3 0" "E-R2 1" "E-R3 1" "E-R2 2" "E-R3 2"; do
  set -- $spec
  arm=$1; seed=$2; tag="${arm}_s${seed}"
  EXP=170; [ "$arm" = "E-R3" ] && EXP=200
  if [ -f "runs/RCX-E01R/cells/${tag}/TRAIN_META.json" ]; then echo "$(date +%T) skip ${tag} (done)" >> $LOG; continue; fi
  python -B -m rcx.lane acquire --job "RCX E01R train ${tag}" --pid "$WINPID" --expected-min "$EXP" --need-ram 12.5 \
    --windows "$WIN" >> $LOG 2>&1 || { echo "$(date +%T) lane acquire failed" >> $LOG; exit 5; }
  export RCX_LANE_TOKEN=$(python -B -c "from rcx import lane; print(lane.read_lock()['token'])")
  echo "$(date +%T) START train ${tag}" >> $LOG
  RCX_THREADS=4 python -B experiments/RCX-E01R/native_e01r.py train --arm "$arm" --seed "$seed" \
    > "runs/RCX-E01R/cells/train_${tag}.stdout.txt" 2>&1
  echo "$(date +%T) END train ${tag} rc=$?" >> $LOG
  python -B -m rcx.lane release --pid "$WINPID" >> $LOG 2>&1
done
echo "$(date +%T) TRAIN CHAIN DONE" >> $LOG

#!/usr/bin/env bash
# RCX-E01R step A (DESIGN.md section 3; CX D-34), v2: same evals as step_a_chain.sh, now under the shared CPU-lane lock
# agreed with BWM (D-76 / CX D-36; rcx/lane.py). Acquires the lane (waiting marker if BWM holds it; no marker while
# the lane is free but RAM is short), keeps it for the 4 evals + analysis, releases it on exit (trap).
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/RCX-E01R/dev2/step_a_chain.log
WINPID=$(cat /proc/$$/winpid)
echo "$(date +%T) v2 chain start (lane lock); winpid ${WINPID}" >> $LOG
python -B -m rcx.lane acquire --job "RCX E01R step A (4 DEV2 evals + analysis)" --pid "$WINPID" --expected-min 110 \
  --need-ram 12.5 >> $LOG 2>&1 || { echo "$(date +%T) lane acquire failed" >> $LOG; exit 5; }
trap 'python -B -m rcx.lane release --pid '"$WINPID"' >> '"$LOG"' 2>&1' EXIT
for spec in "N-CFSD 0" "N-CFSD 1" "N-CFSD 2" "N-VR 0"; do
  set -- $spec
  arm=$1; seed=$2; tag="${arm}_s${seed}"
  if [ -f "runs/RCX-E01R/dev2/eval_${tag}.json" ]; then echo "$(date +%T) skip ${tag} (done)" >> $LOG; continue; fi
  until python -B -c "from rcx.guard import available_ram_gb as r; import sys; sys.exit(0 if (r() or 0) >= 12.5 else 1)"; do
    echo "$(date +%T) waiting for RAM before ${tag} (lane held)" >> $LOG; sleep 300; done
  echo "$(date +%T) START ${tag}" >> $LOG
  RCX_THREADS=4 python -B experiments/RCX-E01R/run_e01r.py eval --arm "$arm" --seed "$seed" \
    > "runs/RCX-E01R/dev2/eval_${tag}.stdout.txt" 2>&1
  echo "$(date +%T) END ${tag} rc=$?" >> $LOG
done
PYTHONIOENCODING=utf-8 python -B experiments/RCX-E01R/run_e01r.py analyze > runs/RCX-E01R/dev2/analyze.stdout.txt 2>&1
echo "$(date +%T) STEP A DONE rc=$?" >> $LOG

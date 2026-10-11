#!/usr/bin/env bash
# RCX-E01N confirmation training chain (PREREG_E01N §6 step 2). Single CPU lane; waits for >= 12 GB free RAM.
cd "$(dirname "$0")/../../.." || exit 1
for cell in "N-VR 1" "N-CFSD 1" "N-VR 2" "N-CFSD 2"; do
  set -- $cell
  arm=$1; seed=$2
  if [ -f "runs/RCX-E01N/cells/${arm}_s${seed}/TRAIN_META.json" ]; then echo "$(date +%T) skip ${arm}_s${seed} (done)"; continue; fi
  until python -B -c "from rcx.guard import available_ram_gb as r; import sys; sys.exit(0 if (r() or 0) >= 12.5 else 1)"; do
    echo "$(date +%T) waiting for RAM before ${arm}_s${seed}"; sleep 300; done
  echo "$(date +%T) START ${arm}_s${seed}"
  RCX_THREADS=4 python -B experiments/RCX-E01N/native_e01.py train --out runs/RCX-E01N/cells --arm "$arm" --seed "$seed" \
    > "runs/RCX-E01N/cells/train_${arm}_s${seed}.stdout.txt" 2>&1
  echo "$(date +%T) END ${arm}_s${seed} rc=$?"
done
echo "$(date +%T) CHAIN DONE"

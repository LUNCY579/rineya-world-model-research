#!/usr/bin/env bash
# RCX-E01N SEALED evaluation chain (PREREG_E01N §6 step 4). Run ONLY after open-sealed and after SEALED_OPEN.json is
# committed with its sha256 logged in DECISIONS. Single CPU lane; waits for >= 12.5 GB free RAM before each model load.
cd "$(dirname "$0")/../../.." || exit 1
[ -f runs/RCX-E01N/cells/SEALED_OPEN.json ] || { echo "SEALED battery not open; refusing"; exit 2; }
for spec in "N-VR 0" "N-CFSD 0" "N-VR 1" "N-CFSD 1" "N-VR 2" "N-CFSD 2" "OFF 0" "T2 0"; do
  set -- $spec
  arm=$1; seed=$2
  tag=$arm; case $arm in N-*) tag="${arm}_s${seed}";; esac
  if [ -f "runs/RCX-E01N/cells/eval_sealed_${tag}.json" ]; then echo "$(date +%T) skip ${tag} (done)"; continue; fi
  until python -B -c "from rcx.guard import available_ram_gb as r; import sys; sys.exit(0 if (r() or 0) >= 12.5 else 1)"; do
    echo "$(date +%T) waiting for RAM before ${tag}"; sleep 300; done
  echo "$(date +%T) START sealed ${tag}"
  RCX_THREADS=4 python -B experiments/RCX-E01N/native_e01.py eval --battery sealed --out runs/RCX-E01N/cells \
    --arm "$arm" --seed "$seed" > "runs/RCX-E01N/cells/eval_sealed_${tag}.stdout.txt" 2>&1
  echo "$(date +%T) END sealed ${tag} rc=$?"
done
echo "$(date +%T) SEALED CHAIN DONE"

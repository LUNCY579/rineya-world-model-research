#!/usr/bin/env bash
# RCX-E01R step A (DESIGN.md section 3; CX D-34): DEV2 evals of the existing E01N adapters on P2_v2 wordings, then the
# descriptive analysis. Single CPU lane: starts only after E11 has ended; waits for >= 12.5 GB free RAM per model load.
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/RCX-E01R/dev2/step_a_chain.log
until grep -q "END E11 full" runs/RCX-E01N/cells/confirm_post_sealed_chain.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) E11 ended; starting step A" >> $LOG
for spec in "N-CFSD 0" "N-CFSD 1" "N-CFSD 2" "N-VR 0"; do
  set -- $spec
  arm=$1; seed=$2; tag="${arm}_s${seed}"
  if [ -f "runs/RCX-E01R/dev2/eval_${tag}.json" ]; then echo "$(date +%T) skip ${tag} (done)" >> $LOG; continue; fi
  until python -B -c "from rcx.guard import available_ram_gb as r; import sys; sys.exit(0 if (r() or 0) >= 12.5 else 1)"; do
    echo "$(date +%T) waiting for RAM before ${tag}" >> $LOG; sleep 300; done
  echo "$(date +%T) START ${tag}" >> $LOG
  RCX_THREADS=4 python -B experiments/RCX-E01R/run_e01r.py eval --arm "$arm" --seed "$seed" \
    > "runs/RCX-E01R/dev2/eval_${tag}.stdout.txt" 2>&1
  echo "$(date +%T) END ${tag} rc=$?" >> $LOG
done
PYTHONIOENCODING=utf-8 python -B experiments/RCX-E01R/run_e01r.py analyze > runs/RCX-E01R/dev2/analyze.stdout.txt 2>&1
echo "$(date +%T) STEP A DONE rc=$?" >> $LOG

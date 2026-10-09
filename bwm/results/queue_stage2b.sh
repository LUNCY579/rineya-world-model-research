#!/bin/bash
# Stage 2b: does variation in the rule parameter (limit) during F training make the limit rule extrapolate?
cd "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
until grep -q STAGE2_DONE runs/queue_stage2.log 2>/dev/null; do sleep 60; done
run() { local label=$1; shift; python scripts/gpu_run.py "$label" -- "$@"; }
for spec in "A:16,24" "B:8,16,24,40,56"; do
  name=${spec%%:*}; lims=${spec#*:}
  d=runs/E2/F_titan_xlim${name}_s0
  [ -f $d/metrics.json ] || run e2-xlim$name scripts/train_forward.py --core titan-sparse --rows 4000 --steps 300 --extra-limits $lims --seed 0 --out $d > $d.stdout 2>&1
done
echo STAGE2B_DONE

#!/bin/bash
# Stage 3 (launch only after the pilot passes review): proposer seeds 2-5 for both families + controls.
cd "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
source runs/lib.sh
STEPS=${STEPS:-800}
for s in 2 3 4 5; do
  for mode in actions states; do
    train_job train-P-$mode-s$s runs/P_${mode}_s$s scripts/train_proposer.py --mode $mode --seed $s --data-seed $s --families 5121 --steps $STEPS
  done
done
# controls: shuffled claimed states (actions intact), seeds 1-2; success-only filtered BC, seed 1 (step-matched)
for s in 1 2; do
  train_job train-P-shuf-s$s runs/P_shuf_s$s scripts/train_proposer.py --mode states --shuffle-states --seed $s --data-seed $s --families 5121 --steps $STEPS
done
train_job train-P-fbc-s1 runs/P_fbc_s1 scripts/train_proposer.py --mode actions --filter-success --seed 1 --data-seed 1 --families 5121 --steps $STEPS
echo STAGE3_DONE

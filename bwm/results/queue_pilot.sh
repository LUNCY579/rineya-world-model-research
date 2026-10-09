#!/bin/bash
# Pilot proposer trainings, each under its own GPU lease.
cd "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
for mode in actions states; do
  python scripts/gpu_run.py train-P-$mode-s1 -- scripts/train_proposer.py --mode $mode --seed 1 --data-seed 1 --families 5121 --steps 800 --save-at 400 --out runs/P_${mode}_s1 > runs/P_${mode}_s1.stdout 2>&1
done

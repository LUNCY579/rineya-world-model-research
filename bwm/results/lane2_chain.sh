#!/bin/bash
# CPU lane 2 chain: wait for A1 pass -> A1 analysis -> vlab labelling pass -> emit vlab rows/SFT.
cd /c/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1
export BWM_DEVICE=cpu BWM_DUTY_RATIO=0 BWM_YIELD_S=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
until grep -q "^done" runs/A1_factorized.stdout; do sleep 60; done
[ -f runs/A1_factorized/analysis.json ] || python scripts/f_factorized.py --out runs/A1_factorized --analyze > runs/A1_factorized.analysis.stdout 2>&1
[ -f runs/A2_vlab_labels/vlab_sft.jsonl ] || {
  python scripts/f_factorized.py --keyset vlab --out runs/A2_vlab_labels > runs/A2_vlab_labels.stdout 2>&1
  python scripts/f_factorized.py --emit-vlab --out runs/A2_vlab_labels >> runs/A2_vlab_labels.stdout 2>&1
}
echo LANE2_DONE

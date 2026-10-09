#!/bin/bash
cd "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
source runs/lib.sh
# wait for the states run to finish (adapter written or its process gone)
while powershell -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_proposer.py --mode states --seed 1*' }) { exit 0 } else { exit 1 }"; do sleep 30; done
train_job train-P-states-s1 runs/P_states_s1 scripts/train_proposer.py --mode states --seed 1 --data-seed 1 --families 5121 --steps 800 --save-at 400
train_job train-P-actions-s1 runs/P_actions_s1 scripts/train_proposer.py --mode actions --seed 1 --data-seed 1 --families 5121 --steps 800 --save-at 400
echo RETRY_DONE

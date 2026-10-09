#!/bin/bash
# Stage 2: pilot validation eval, then E2 forward-generalization study. Idempotent; each job holds its own lease.
cd "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
until [ -f runs/P_actions_s1/adapter/adapter_model.safetensors ] && [ -f runs/P_states_s1/adapter/adapter_model.safetensors ]; do sleep 30; done
FW=runs/F_titan_main/adapter
run() { local label=$1; shift; python scripts/gpu_run.py "$label" -- "$@"; }
# 2a. pilot validation (selection data only)
[ -f runs/EVAL_pilot_s1/.done_core ] || { run eval-pilot-core scripts/run_eval.py --fwd $FW --pa runs/P_actions_s1/adapter --ps runs/P_states_s1/adapter \
   --arms A00,A10,A01,A11,A01g,A11g --baselines G1F,G2F,BFS16F,BFS64F,BFS256F --splits id_val,long_val,range_val --tag pilot --out runs/EVAL_pilot_s1 > runs/EVAL_pilot_s1.stdout 2>&1 && touch runs/EVAL_pilot_s1/.done_core; }
[ -f runs/EVAL_pilot_s1_step400/.done ] || { run eval-pilot-400 scripts/run_eval.py --fwd $FW --pa runs/P_actions_s1/adapter_step400 --ps runs/P_states_s1/adapter_step400 \
   --arms A00,A10,A01,A11 --splits id_val --tag pilot400 --out runs/EVAL_pilot_s1_step400 > runs/EVAL_pilot_s1_step400.stdout 2>&1 && touch runs/EVAL_pilot_s1_step400/.done; }
[ -f runs/EVAL_pilot_s1/.done_diag ] || { run eval-pilot-diag scripts/run_eval.py --fwd $FW --pa runs/P_actions_s1/adapter --ps runs/P_states_s1/adapter \
   --arms A01x,A11x,A00flip,A10flip,A01k12,BASE_a,BASE_aF --splits id_val,range_val --tag pilotdiag --out runs/EVAL_pilot_s1 >> runs/EVAL_pilot_s1.stdout 2>&1 && touch runs/EVAL_pilot_s1/.done_diag; }
# 2b. E2: forward-model generalization by backbone (structured hold-out x%5==0; limits 40/56 never trained)
mkdir -p runs/E2
for core in titan-sparse supa qwen3-0.6b; do
  [ -f runs/E2/zs_$core.json ] || run e2-zs-$core scripts/grade_forward.py --core $core --out runs/E2/zs_$core.json > runs/E2/zs_$core.stdout 2>&1
  for rows in 256 1024; do
    d=runs/E2/F_${core}_r${rows}_s0
    [ -f $d/metrics.json ] || run e2-$core-$rows scripts/train_forward.py --core $core --rows $rows --steps 200 --holdout-mod5 --seed 0 --out $d > $d.stdout 2>&1
  done
done
echo STAGE2_DONE

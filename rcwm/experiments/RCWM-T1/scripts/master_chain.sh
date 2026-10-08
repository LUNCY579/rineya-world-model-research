#!/usr/bin/env bash
# Single idempotent GPU chain for the RCWM-T1 pilot. Safe to relaunch after a reboot: every step
# checks for its own completed output. Every GPU step takes the shared lease itself and is
# throttled (runtime.breathe; DECISIONS I-10). Ordered by scientific priority.
set -u
cd "$(dirname "$0")/.."
PY=C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
LOG=runs/bank_pilot/master_log.txt
R=runs/bank_pilot
mkdir -p "$R"
step() { echo "=== $(date -Iseconds) start $*" >> "$LOG"; "$PY" -u "$@" >> "$LOG" 2>&1; local rc=$?; echo "=== $(date -Iseconds) exit rc=$rc $*" >> "$LOG"; return $rc; }
arm() { [ -f "$R/$1/DONE.json" ] || step scripts/run_arm.py --bank data/bank_pilot.json --arm "$@"; }
lik() { [ -f "$R/$1/lik_rank.jsonl" ] || step scripts/run_lik.py --bank bank_pilot --arm "$1" --splits dev,test; }
echo "=== $(date -Iseconds) master chain start" >> "$LOG"
arm sparse --gen                                                   # capture done; gen + off-task
# PREREG-1 §7 step 3 (CPU): once both confirmation arms are DONE, verify the freeze, run the
# registered analysis once (thread count fixed by §7), render it mechanically. Freeze failure -> no analysis.
if [ -f runs/bank_confirm/sparse/DONE.json ] && [ -f runs/bank_confirm/supa/DONE.json ] \
   && [ ! -f runs/bank_confirm/analysis/PREREG_1_RESULTS.json ]; then
  step scripts/verify_freeze.py && RCWM_THREADS=8 step scripts/confirm_analysis.py && step scripts/render_confirm.py
fi
# Per-item activation patching (R9 follow-up; user request 2026-10-07): GPU, under the lease
[ -f runs/bank_pilot/patch/patch_results.json ] || step scripts/run_patch.py --device cuda --out-name patch --layers 8,16,24 --hybrid-layers 12,16,20,24 --dmis-layers 24
[ -f runs/t2/t2_s0/adapter/adapter_model.safetensors ] || step scripts/run_t2.py train --tag t2_s0 --steps 190 --seed 0
[ -f runs/t2/t2_s0/eval.json ]                         || step scripts/run_t2.py eval --tag t2_s0
if [ -f runs/t2/t2_s0/eval.json ] && "$PY" -u scripts/t2_screen.py --tag t2_s0 >> "$LOG" 2>&1; then   # D-19 screen
  for s in 1 2; do
    [ -f runs/t2/t2_s$s/adapter/adapter_model.safetensors ] || step scripts/run_t2.py train --tag t2_s$s --steps 190 --seed $s
    [ -f runs/t2/t2_s$s/eval.json ]                         || step scripts/run_t2.py eval --tag t2_s$s
  done
fi
# D-24/D-25 PREREG-2. Freeze is mechanical (CPU): it runs the frozen t2_analysis on s0-s2 and applies the §9 gate
# (D-23's off-task numbers HAVE been seen, R14 at 09:10). If frozen: the registered GPU job (may span two turns;
# the second call resumes), then the registered analysis once.
if [ -f runs/t2/t2_s1/eval.json ] && [ -f runs/t2/t2_s2/eval.json ] && [ ! -f PREREG_2_FREEZE.json ] \
   && [ ! -f docs/prereg2/FREEZE_GATE_FAILED.json ]; then
  step scripts/confirm2_gen.py --freeze --d23-offtask-seen yes
fi
if [ -f PREREG_2_FREEZE.json ]; then
  [ -f runs/bank_confirm2/DONE.json ] || step scripts/confirm2_gen.py
  [ -f runs/bank_confirm2/DONE.json ] || step scripts/confirm2_gen.py
  [ -f runs/bank_confirm2/DONE.json ] && [ ! -f runs/bank_confirm2/analysis/PREREG_2_RESULTS.json ] \
    && step scripts/confirm2_analysis.py
fi
# Queue hooks (D-26): sourced in name order so they can use step(); each must be idempotent. This lets a frozen
# job join the single RCWM queue without editing this running script (one-waiter rule, R14 addendum).
for h in scripts/queue_hooks/*.sh; do [ -f "$h" ] && source "$h"; done
lik sparse
lik supa
[ -f "$R/pvr_sparse/stage1.jsonl" ]      || step scripts/run_pvr.py stage1 --bank bank_pilot --arm sparse --splits dev,test
[ -f "$R/supa/offtask_meta.json" ]       || step scripts/run_offtask.py --bank bank_pilot --arm supa
[ -f "$R/pvr_sparse/retro_sparse.npy" ]  || step scripts/run_pvr.py retro --bank bank_pilot --arm sparse --source-arm sparse
[ -f "$R/pvr_sparse/retro_supa.npy" ]    || step scripts/run_pvr.py retro --bank bank_pilot --arm supa --source-arm sparse
[ -f "$R/steer/steer_results.json" ]     || step scripts/run_steer.py --bank bank_pilot --layers 16,20,24 --alphas 1.0
arm randdelta
arm exp
arm transition --gen
lik transition
arm sparse_a0.5
arm sparse_a2.0
arm qwen3
[ -f "$R/pvr_sparse/hstates_supa.npy" ]  || step scripts/run_pvr.py hstates --bank bank_pilot --arm supa --source-arm sparse
bash scripts/regen_chain_body.sh                                  # non-binding pilot regeneration (D-17)
echo "=== $(date -Iseconds) master chain finished" >> "$LOG"

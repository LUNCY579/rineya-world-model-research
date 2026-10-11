#!/usr/bin/env bash
# RCX single CPU lane, after the SEALED eval chain: (1) the one PREREG_E01N SEALED analysis, then (2) the frozen E11 run.
# Replaces the original wrapper's "chain; E11" tail so the analysis never overlaps E11 (CX D-32).
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/RCX-E01N/cells/confirm_post_sealed_chain.log
until grep -q "SEALED CHAIN DONE" runs/RCX-E01N/cells/confirm_sealed_eval_chain.log; do sleep 60; done
for tag in N-VR_s0 N-CFSD_s0 N-VR_s1 N-CFSD_s1 N-VR_s2 N-CFSD_s2 OFF T2; do
  [ -f "runs/RCX-E01N/cells/eval_sealed_${tag}.json" ] || { echo "$(date +%T) missing eval_sealed_${tag}.json; stopping" >> $LOG; exit 3; }
done
echo "$(date +%T) START SEALED analysis" >> $LOG
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES= python -B experiments/RCX-E01N/analyze_native.py --battery sealed \
  --cells runs/RCX-E01N/cells > runs/RCX-E01N/ANALYSIS_SEALED.stdout.txt 2>&1
rc=$?
echo "$(date +%T) END SEALED analysis rc=$rc" >> $LOG
[ $rc -eq 0 ] || exit 4
echo "$(date +%T) START E11 full" >> $LOG
CUDA_VISIBLE_DEVICES= python -B experiments/RCX-E11/run_e11.py --threads 4 > runs/RCX-E11/main.stdout.txt 2>&1
echo "$(date +%T) END E11 full rc=$?" >> $LOG

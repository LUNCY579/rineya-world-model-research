# Queue hook (D-47): seeds 1 and 2 of the arm the frozen 5a screen named (runs/t2bp/sd_s0/analysis/T2BP_SCREEN.json:
# WINNER=bbits, rule revision 4). Same rows, T2 recipe, surface-diversity build; frozen runner scripts/t2bp_run.py.
# Runs only if the screen's verdict file names B-bits. Exit codes: 0 complete; 3 deterministic stop (never retried);
# 5 deferred by the cap: sleep 60 s (BWM's yield loop sees the release) and call once more.
if [ -f runs/t2bp/sd_s0/analysis/T2BP_SCREEN.json ] && grep -q '"verdict": "WINNER=bbits"' runs/t2bp/sd_s0/analysis/T2BP_SCREEN.json; then
  for t2bp_seed in 1 2; do
    t2bp_tag=sd_s$t2bp_seed
    if [ ! -f runs/t2bp/$t2bp_tag/DONE.json ] && [ ! -f runs/t2bp/$t2bp_tag/STOPPED.json ]; then
      step scripts/t2bp_run.py --tag $t2bp_tag --seed $t2bp_seed --arms bbits
      t2bp_rc=$?
      if [ "$t2bp_rc" -eq 5 ]; then
        sleep 60
        step scripts/t2bp_run.py --tag $t2bp_tag --seed $t2bp_seed --arms bbits
      fi
    fi
  done
fi

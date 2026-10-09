# Queue hook (D-36): retention suite 5c, the Supa-only DEV run on the frozen dev draw, then the create-only dev selection.
# Sourced by master_chain.sh in name order. Idempotent: every line checks its own output. The run takes the shared lease
# itself (label rcwm-t1:retention) and refuses when another RCWM job holds or waits for the lease (one-waiter rule).
# ret_run.py exit codes: 0 complete; 3 deterministic stop (never retried); 5 deferred by the turn cap (one resume call).
RET_DEV=dev_20261121
if [ -f docs/retention/RET_FREEZE.json ] && [ -f docs/retention/draws/$RET_DEV.jsonl ] && [ ! -f docs/retention/DEV_SELECTION.json ]; then
  if [ ! -f runs/retention/$RET_DEV/DONE.json ]; then
    step scripts/ret_run.py --mode dev --draw docs/retention/draws/$RET_DEV.jsonl
    ret_rc=$?
    if [ "$ret_rc" -ne 3 ] && [ ! -f runs/retention/$RET_DEV/DONE.json ]; then
      step scripts/ret_run.py --mode dev --draw docs/retention/draws/$RET_DEV.jsonl
    fi
  fi
  # CPU, once, create-only: Supa's dev accuracies -> DEV_SELECTION.json (before any confirmation draw is minted)
  [ -f runs/retention/$RET_DEV/DONE.json ] && step scripts/ret_analysis.py dev-select --run runs/retention/$RET_DEV
fi

# DRAFT queue hook for PREREG-LB (joint roadmap item 3, the ledger beam test; docs/ledger_beam/PREREG_LB_DRAFT.md
# section 7). NOT INSTALLED. Its sha256 is recorded in the freeze, and scripts/lbeam_run.py refuses to run (exit 3, not
# persisted) if scripts/queue_hooks/70_lbeam.sh exists and is not this file byte for byte. Install only AFTER the freeze,
# by the RCWM-T1 operator on the user's go-ahead, logged in DECISIONS.md:
#   cp docs/ledger_beam/queue_hook_lbeam.sh.draft scripts/queue_hooks/70_lbeam.sh   (then compare the two sha256 values)
# WHEN IT RUNS. master_chain.sh expands `for h in scripts/queue_hooks/*.sh` ONCE, when execution ENTERS that loop, so a
# chain that entered the loop before the install never sources this hook; it then runs only in a FRESH chain
# (scripts/launch_master.cmd, hidden via WMI) started when no master_chain.sh / confirm_chain.sh process is alive. Never
# start a second chain while one is alive (R14 one-waiter rule); lbeam_run.py also refuses (exit 3, not persisted) when
# another RCWM job holds or waits for the GPU lease.
# IDEMPOTENT: every line checks its own output. The run step takes the shared lease itself (rcwmt.runtime.gpu_lease,
# label rcwm-t1:lbeam: alternation, thermal flag, holder check) and holds a one-runner lock (runs/lbeam/.run.lock); its
# CPU stage C1 (ORACLE / IGNORE_STATUS) never takes the lease and runs at the start of the call after the one that
# recorded A0, before that call's lease.
# Exit codes of scripts/lbeam_run.py: 0 complete; 3 deterministic stop (freeze / hook identity / one-waiter / runtime /
# hash / fingerprint / A0 anchor / batch invariance / no progress / sanity) - NEVER retried, and persisted as
# runs/lbeam/STOPPED.json when it concerns the run (an operator must investigate, log in DECISIONS.md and move it aside);
# 5 deferred by the 4.5 h turn cap at a stage boundary, or C1 still to run - exactly ONE resume call (it runs C1 if
# pending, waits for the next lease turn, re-runs A0 and continues); anything else is a crash (the lease is released) -
# exactly one retry. At most two calls per chain pass.
# The registered analysis (CPU, once, create-only) runs after DONE.json, or after a STOPPED.json that the runner marked
# "verdict_eligible": true (A0, G1 and C1 recorded and unchanged; the missing report-only stages are 'not evaluable').
if [ -f PREREG_LB_FREEZE.json ]; then
  if [ ! -f runs/lbeam/DONE.json ] && [ ! -f runs/lbeam/STOPPED.json ]; then
    step scripts/lbeam_run.py
    lb_rc=$?
    if [ "$lb_rc" -ne 0 ] && [ "$lb_rc" -ne 3 ] && [ ! -f runs/lbeam/DONE.json ] && [ ! -f runs/lbeam/STOPPED.json ]; then
      step scripts/lbeam_run.py                       # rc 5: the one resume call; any other rc: the one crash retry
    fi
  fi
  if [ ! -f runs/lbeam/analysis/PREREG_LB_RESULTS.json ]; then
    if [ -f runs/lbeam/DONE.json ] || grep -qs '"verdict_eligible": true' runs/lbeam/STOPPED.json; then
      step scripts/lbeam_analysis.py
    fi
  fi
fi

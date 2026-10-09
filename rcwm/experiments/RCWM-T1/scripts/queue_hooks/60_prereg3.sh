# DRAFT queue hook for PREREG-3 (rule or trigger; PREREG_3_DRAFT.md §7.5). NOT INSTALLED. Its sha256 is recorded in the
# freeze, and scripts/p3_run.py refuses to run (exit 3, not persisted) if scripts/queue_hooks/60_prereg3.sh exists and is
# not this file byte for byte. Install only after the freeze: cp docs/prereg3/queue_hook_p3.sh.draft
# scripts/queue_hooks/60_prereg3.sh (then compare the two sha256 values).
# WHEN IT RUNS. master_chain.sh expands `for h in scripts/queue_hooks/*.sh` ONCE, when execution ENTERS that loop. The
# chain launched at 11:03 on 2026-10-08 (master_chain.sh pid 28516 under confirm_chain.sh pid 40892) entered the loop at
# 18:38:31, when only 50_l1_screen.sh existed, so it will never source a hook installed later. After L1 it runs the
# remaining pilot GPU steps (lik, pvr, steer, arms, regen) and exits, and nothing relaunches it. This hook therefore
# runs only in a FRESH chain, started with scripts/launch_master.cmd (hidden, via WMI) after
# runs/bank_pilot/master_log.txt shows "master chain finished" AND no master_chain.sh or confirm_chain.sh process is
# alive. In the fresh chain every earlier step whose output exists is skipped (if L1 has not finished, hook 50 runs it
# first). Never start a second chain while one is alive (R14 one-waiter rule); p3_run.py also refuses (exit 3, not
# persisted) when another RCWM job holds or waits for the GPU lease. Relaunching before "master chain finished" means
# stopping the live chain at a safe point first (as R14 did): an operator decision, logged in DECISIONS.md. The RCWM-T1
# operator (the session that runs RCWM-T1, with the user's go-ahead) does the relaunch and logs it in DECISIONS.md.
# Idempotent: every line checks its own output. The run step takes the shared lease itself (rcwmt.runtime.gpu_lease,
# label rcwm-t1:prereg3: alternation, thermal flag, holder check) and holds a one-runner lock (runs/bank_p3_oor/.run.lock).
# Exit codes of scripts/p3_run.py: 0 complete; 3 deterministic stop (freeze / hook identity / one-waiter / runtime /
# hash / fingerprint / anchor / sanity) - NEVER retried, and persisted as runs/bank_p3_oor/STOPPED.json when it concerns
# the run (an operator must investigate, log in DECISIONS.md and move it aside); 5 deferred by the 4.5 h turn cap (the
# second call waits for the next turn and resumes); anything else is a crash (the lease is released) - one retry.
if [ -f PREREG_3_FREEZE.json ] && [ ! -f runs/bank_p3_oor/STOPPED.json ]; then
  if [ ! -f runs/bank_p3_oor/DONE.json ]; then
    step scripts/p3_run.py
    p3_rc=$?
    if [ "$p3_rc" -ne 3 ] && [ ! -f runs/bank_p3_oor/DONE.json ] && [ ! -f runs/bank_p3_oor/STOPPED.json ]; then
      step scripts/p3_run.py                            # resumes from the recorded stages; never re-runs a finished one
    fi
  fi
  # CPU, once, create-only: freeze check, scorer self-test BEFORE any output is read, validation, verdicts, render
  if [ -f runs/bank_p3_oor/DONE.json ] && [ ! -f runs/bank_p3_oor/analysis/PREREG_3_RESULTS.json ]; then
    step scripts/p3_analysis.py
  fi
fi

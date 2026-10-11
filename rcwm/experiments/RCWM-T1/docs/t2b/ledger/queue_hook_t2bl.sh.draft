# DRAFT queue hook for the T2b-ledger dev screen (docs/t2b/ledger/DESIGN.md; the pre-stated gate docs/t2b/ledger/GATE_RULE.md).
# NOT INSTALLED and NOT FROZEN. Install only AFTER `python -B scripts/t2bl_gate.py --freeze-rule --approval "..."` has
# written docs/t2b/ledger/GATE_RULE.json, by the RCWM-T1 operator on the user's go-ahead, logged in DECISIONS.md:
#   cp docs/t2b/ledger/queue_hook_t2bl.sh.draft scripts/queue_hooks/88_t2bl.sh   (then compare the two sha256 values)
# scripts/t2bl_run.py refuses (exit 3, not persisted) if any installed scripts/queue_hooks/*t2bl*.sh is not this draft.
# WHEN IT RUNS. master_chain.sh expands `for h in scripts/queue_hooks/*.sh` ONCE, when execution enters that loop, so a
# chain that entered the loop before the install never sources this hook; it runs only in a FRESH chain started when no
# master_chain.sh / confirm_chain.sh process is alive (R14 one-waiter rule; scripts/t2bl_run.py also refuses, exit 3 not
# persisted, while another RCWM job holds or waits for the GPU lease). Joint turn order (D-45): after 5b and BWM's turn.
# IDEMPOTENT: every line checks its own output. The run step takes the shared lease itself (rcwmt.runtime.gpu_lease,
# label rcwm-t1:t2b-ledger: alternation, thermal flag, holder check, 4.5 h cap with D-22 pairing) and holds a one-runner
# lock (runs/t2bl/dev/.run.lock).
# Exit codes of scripts/t2bl_run.py: 0 complete; 3 deterministic stop - NEVER retried (persisted as
# runs/t2bl/dev/STOPPED.json when it concerns the run, e.g. two consecutive turns without a recorded stage); 5 deferred by
# the turn cap: SLEEP 60 s FIRST (so BWM's yield loop sees the release; agreed with BWM), then call again (the next call
# waits for the next turn, re-anchors and resumes); anything else is a crash (the lease is released) - exactly one retry.
# At most three calls per chain pass. The plan is one turn (3.38 h at the planning rate, 2.56 h observed;
# docs/t2b/ledger/GEN_PLAN.json).
# The registered dev gate (CPU, once, create-only) runs after DONE.json, or after a persisted STOPPED.json (it then writes
# a NO VERDICT record): scorer self-test first, then the pre-stated rule -> runs/t2bl/dev/analysis/GATE.{md,json}. It
# decides only whether PREREG-5 may be frozen; nothing here freezes it.
if [ -f docs/t2b/ledger/GATE_RULE.json ] && [ -f data/t2bl_train.SOURCE.json ] && [ -f data/t2bl_dev.SOURCE.json ]; then
  t2bl_calls=0
  t2bl_crashes=0
  while [ "$t2bl_calls" -lt 3 ] && [ ! -f runs/t2bl/dev/DONE.json ] && [ ! -f runs/t2bl/dev/STOPPED.json ]; do
    step scripts/t2bl_run.py
    t2bl_rc=$?
    t2bl_calls=$((t2bl_calls + 1))
    if [ "$t2bl_rc" -eq 0 ] || [ "$t2bl_rc" -eq 3 ]; then
      break
    elif [ "$t2bl_rc" -eq 5 ]; then
      sleep 60
    else
      t2bl_crashes=$((t2bl_crashes + 1))
      [ "$t2bl_crashes" -ge 2 ] && break
    fi
  done
  if { [ -f runs/t2bl/dev/DONE.json ] || [ -f runs/t2bl/dev/STOPPED.json ]; } && [ ! -f runs/t2bl/dev/analysis/GATE.json ]; then
    step scripts/t2bl_gate.py
  fi
fi

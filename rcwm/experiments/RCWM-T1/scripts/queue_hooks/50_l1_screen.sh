# DRAFT queue hook for PREREG-L1 (D-26, D-27). NOT INSTALLED. Its sha256 is recorded in the freeze.
# Install only after the freeze, by copying this file byte for byte to scripts/queue_hooks/50_l1_screen.sh.
# master_chain.sh sources every scripts/queue_hooks/*.sh in name order AFTER its PREREG-2 steps, and only when execution
# reaches that loop. Before installing, read runs/bank_pilot/master_log.txt: if the running chain has already passed the
# loop, the hook runs only when the queue is relaunched, at a safe point and with no second RCWM waiter (R14).
# The hook uses step() and is idempotent (every line checks its own output). It adds no second GPU waiter: the run step
# takes the shared lease itself (rcwmt.runtime.gpu_lease: alternation, thermal flag, holder check) and holds a
# create-only run lock (runs/l1_screen/.run.lock).
# Exit codes of `l1_screen.py run`: 0 finished (DONE.json) or stopped by the parity gate (PARITY_FAIL.json);
# 3 deterministic failure (sanity / identity / freeze / environment) - never retried; 5 resumable host stop (RAM,
# watchdog, RAM-caused pause; the lease was released) - one retry; anything else (crash, LeaseError) - one retry.
if [ -f docs/l1/PREREG_L1_FREEZE.json ]; then
  # CPU, no model: materialise arm-l8_15 (two derivations, disk re-read, gates B1-B7); re-verifies if present
  [ -f runs/l1_screen/arm-l8_15/MATERIALIZATION.json ] || step scripts/l1_screen.py build
  if [ -f runs/l1_screen/arm-l8_15/MATERIALIZATION.json ] && [ ! -f runs/l1_screen/DONE.json ] \
     && [ ! -f runs/l1_screen/PARITY_FAIL.json ]; then
    # GPU, one lease (rcwm-t1:l1screen): supa, sparse, the anchor parity gate (CPU), then l8_15 only if it passed
    step scripts/l1_screen.py run --device cuda
    l1_rc=$?
    if [ "$l1_rc" -ne 3 ] && [ ! -f runs/l1_screen/DONE.json ] && [ ! -f runs/l1_screen/PARITY_FAIL.json ]; then
      step scripts/l1_screen.py run --device cuda       # resumes from the saved stages; never re-runs a finished one
    fi
  fi
  # CPU, once, create-only: binds the run to the freeze, re-derives the parity gate, every endpoint, the rule
  if { [ -f runs/l1_screen/DONE.json ] || [ -f runs/l1_screen/PARITY_FAIL.json ]; } \
     && [ ! -f runs/l1_screen/L1_RESULTS.json ]; then
    step scripts/l1_screen.py analyze
  fi
fi

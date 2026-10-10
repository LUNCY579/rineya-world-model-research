# DRAFT queue hook for the T2b-register screen, revision 6 (docs/t2b/register/DESIGN.md; roadmap item 3, joint item 5b;
# DECISIONS D-36, D-42; revision 6 = the extended shortcut family H, the adversarial identification stratum cond_adv and
# the report-only shortcut-agreement analysis; the stage names and this hook's logic are unchanged since revision 4, only
# these comments). NOT INSTALLED. Its sha256 is a frozen input of the screen (scripts/t2br_run.py --freeze). Install only
# after the freeze, as an operator decision logged in DECISIONS.md:
#   cp docs/t2b/register/queue_hook_t2br.sh.draft scripts/queue_hooks/80_t2br.sh
# (name order after the installed hooks: L1's 50, PREREG-3's 60, the retention dev run's 65, PREREG-LB's 70 and the
# prerequisite screen's 75). master_chain.sh globs scripts/queue_hooks/*.sh ONCE when it enters that loop, so a new hook
# needs a FRESH chain, launched only after the live chain has finished or been stopped at a safe point (R14 one-waiter
# rule). The runner also refuses (exit 3, not persisted) while another RCWM job holds or waits for the GPU lease, and if
# an installed *t2br*.sh is not this draft byte for byte.
# DECISION POINT 1 HAS FIRED (PREREG-3 STATUS: TRIGGER; operator decision C1): the screen trains ONLY the
# surface-diversity build 'sd' (data/t2br_train_sd_<arm>.jsonl); the runner refuses any other variant. Outputs:
# runs/t2b_register/s0_sd/; the analysis writes runs/t2b_register/analysis/T2BR_RESULTS_s0_sd.{json,md}.
# GPU plan (docs/t2b/register/GEN_PLAN.json): about 4.99 GPU-h in TWO turns at the registered planning rate (turn 1:
# training + A0 + E1 + E1H, 3.73 h; turn 2: re-anchor + E2, 1.26 h), ONE turn of 4.06 h at the observed rate.
# Generations: 15,216 with the anchor once; 16,048 with the planning rate's re-anchor; at most 16,880 (3 calls).
# Idempotent: every line checks its own output. The run step takes the shared lease itself (rcwmt.runtime.gpu_lease,
# label rcwm-t1:t2b-register: alternation, thermal flag, holder check, 4.5 h cap with D-22 pairing) and holds a
# one-runner lock (<out>/.run.lock). Exit codes of scripts/t2br_run.py: 0 complete; 3 deterministic stop - NEVER retried
# (persisted as <out>/STOPPED.json when it concerns the run); 5 deferred by the turn cap (the next call waits for the
# next RCWM turn and resumes from the recorded stages, re-anchoring first); anything else is a crash (the lease is
# released). At most 3 calls per chain pass: room for two deferrals, or one deferral and one crash retry.
T2BR_VARIANT=sd
T2BR_OUT="runs/t2b_register/s0_$T2BR_VARIANT"
if [ -f docs/t2b/register/SCREEN_FREEZE.json ] && [ ! -f "$T2BR_OUT/STOPPED.json" ]; then
  t2br_calls=0
  while [ ! -f "$T2BR_OUT/DONE.json" ] && [ ! -f "$T2BR_OUT/STOPPED.json" ] && [ "$t2br_calls" -lt 3 ]; do
    step scripts/t2br_run.py --seed 0 --train-variant "$T2BR_VARIANT"
    t2br_rc=$?
    t2br_calls=$((t2br_calls + 1))
    if [ "$t2br_rc" -eq 3 ]; then break; fi
  done
  # CPU, once, create-only: freeze + provenance verified, outputs validated, the pre-stated rule applied, report rendered
  # (with the report-only shortcut-agreement flag and the always-quoted limits beside the verdict)
  if [ -f "$T2BR_OUT/DONE.json" ] && [ ! -f "runs/t2b_register/analysis/T2BR_RESULTS_s0_$T2BR_VARIANT.json" ]; then
    step scripts/t2br_analysis.py --run-dir "$T2BR_OUT"
  fi
fi

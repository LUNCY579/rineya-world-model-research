# NEXT_STEPS (RCX)

Status: 2026-10-11, about 15:40 CDT. Items are listed in single-CPU-lane order.

1. **RCM-001 next recipe, RCX-E01R (follows the R-08 REVISE; CX D-33).** The target is unseen restrictive *positive*
   rule phrasings, which N-CFSD reads as "apply" (14% correct no-ops).
   - **a. Diagnostic first (descriptive; no training; about 1.6 h CPU). Pools fixed (CX D-34); the DEV2 battery is built;
     the step A chain is queued (waiting for ≥ 12.5 GB free RAM).**
     - Arms: N-CFSD s0–s2 and N-VR s0 on DEV2 (1,320 records; `in` plus `p2`, with 45 blind-authored P2_v2 wordings).
     - Readings fixed in DESIGN.md §3:
       - **H-alt:** POS no-op accuracy ≥ 0.80 in every form.
       - **H-E01R:** POS < 0.50 while NEG ≥ 0.80.
   - **b. Candidate recipes.** Choose by the diagnostic and preregister before training (DESIGN.md §4):
     - R1, complement coverage: a separately blind-authored training pool, audited against P2/P3;
     - R2, complement-focused twins: unlisted-status no-op twins under positive rules;
     - R3, a predicate-first trace (`cmp` before `after`). RCWM R23 and BWM D-54 favour it.
   - **c. Confirmation.**
     - A new salt-committed SEALED battery on P3_v2 (48 wordings; sha `88dd432b…`; never read or analysed).
     - 3 seeds per arm.
     - Control: the existing N-CFSD s0–s2 adapters, re-evaluated on the new battery. This saves about 8 h of training.
2. **Joint B4 with BWM.**
   - Agreed: the Pa_v3 cross is a secondary factor, and the primary Holm family stays under Pa_v2.
   - Waiting for BWM's freeze.
   - RCX supplies the arm (a) module shas: `b4_hook.py`, the GNN model file and its sidecar.
3. **SUPA probe arm** (R-03 completeness), at a quiet RAM moment (needs ≥ 12 GB free).
4. **RVM-001.**
   - E04 legality critic on `traces.bw_parts` minimal pairs.
   - E05 successor features for goal-weight re-ranking on RCWM-style decisions.
5. **RSM-001 follow-ups (R-09).**
   - Offer BWM the mined macro library as a B5 action-abstraction arm (BWM's call).
   - E11c: execution-aware subgoal labels.
6. **SymLM mechanism tier** (fixed vs resampled symbols × data regime). Low priority now that native results exist.

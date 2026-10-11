# NEXT_STEPS (RCX)

Status: 2026-10-11, about 14:30 CDT. Items are listed in single-CPU-lane order.

1. **RSM-001 E11 full run (frozen PREREG_E11, CX D-31). Running since 14:07; expected 2–3 h (≤ 4 h).**
   - Then record R-09 from `runs/RCX-E11/main/REPORT_E11.md`, mechanically.
   - Send BWM the shas of the B4 exclusion key lists:
     - `runs/RCX-E11/main/a/instances.json` (E11a);
     - `runs/RCX-E11/main/b/data/instances.json` (E11b, including the 128 exec instances).
2. **RCM-001 next recipe, RCX-E01R (follows the R-08 REVISE; CX D-33).** The target is unseen restrictive *positive*
   rule phrasings, which N-CFSD reads as "apply" (14% correct no-ops).
   - **a. Diagnostic first (descriptive; no training; about 2–3 h CPU).**
     - Blind-author a fresh wording pool P2 for every polarity and composition, then audit it against the training and
       `ho_tpl` pools for near-duplicates. The author must not see the training templates.
     - Evaluate N-CFSD s0–s2 and N-VR s0 on a small DEV2 battery that uses P2.
     - Question: does the positive-rule failure hold across new positive phrasings, or is it tied to the two `ho_tpl`
       surfaces?
   - **b. Candidate recipes.** Choose by the diagnostic and preregister before training:
     - R1, positive-phrasing coverage: grammar-generated restrictive positives ("only", "solely", "just",
       "otherwise nothing", conditionals), balanced against negatives, at matched rows;
     - R2, polarity-paired paraphrase twins: one rule written positively and negatively with identical targets, to
       force a semantic reading;
     - R3, a predicate-first trace (`cmp` before `after`). This tests the RCX-E01 hypothesis that exposing the
       comparison helps.
   - **c. Confirmation.**
     - A new salt-committed SEALED battery, with a new blind held-out wording pool (P3, never analysed).
     - 3 seeds per arm.
     - Control: the existing N-CFSD s0–s2 adapters, re-evaluated on the new battery. This saves about 8 h of training.
3. **Joint B4 with BWM.**
   - Agreed: the Pa_v3 cross is a secondary factor, and the primary Holm family stays under Pa_v2.
   - Waiting for BWM's freeze.
   - RCX supplies the arm (a) module shas: `b4_hook.py`, the GNN model file and its sidecar.
4. **SUPA probe arm** (R-03 completeness), at a quiet RAM moment (needs ≥ 12 GB free).
5. **RVM-001.**
   - E04 legality critic on `traces.bw_parts` minimal pairs.
   - E05 successor features for goal-weight re-ranking on RCWM-style decisions.
6. **SymLM mechanism tier** (fixed vs resampled symbols × data regime). Low priority now that native results exist.

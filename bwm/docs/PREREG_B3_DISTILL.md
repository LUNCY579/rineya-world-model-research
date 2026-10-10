# Addendum B3: search distillation into the proposer (joint roadmap item 4)

Written 2026-10-09 22:30 CDT. At that point no distillation search had run and no distillation row existed. Nothing
in this file has been tuned on any data. B3 lives in its own file so `PREREG_PLANBENCH.md` stays byte-identical to
its frozen SHA-256 (adfc4354…, recorded in `runs/FREEZE_B2_*.json`). All B2 rules carry over unless changed here:

- success = an exact or VAL-accepted plan, one plan per instance, no execution feedback;
- Clopper-Pearson 95% CIs;
- exact two-sided McNemar for paired contrasts.

## Question

Can the model's own world model teach its proposer? The frozen B2 system solves hard-style problems with search
(S-beam: hard 73/110), but the proposer alone barely does (S-direct: hard 5/110, mirror 6/110). If the proposer is
trained on plans that the *learned* world model F verified, does the model plan further **without search**? The
exact simulator never supplies a label here.

## Pipeline (fixed now; no selection anywhere)

1. **Search** (`scripts/bw_distill.py search --n 240`, as recorded in D-56):
   - 240 fresh generated hard-style problems, 6-10 blocks, whole-configuration tall-tower goals, seed 20261010;
   - hard-set signatures/invariants and val_large/mirror invariants are excluded;
   - solved by the frozen B2 planner with K = 4 (W 4, E 24, R 12, M 80, P 1; F = `BW_Fdelta_s0`, Pa = `BW_Pa_v2_s1`);
   - only **F-verified** plans are kept. The exact simulator measures label precision for analysis only.
2. **Rows** (`bw_distill.py rows`): every suffix of each verified plan becomes a proposer row. The row's state is
   F's predicted state and its target is the remaining actions.
3. **Mixes** (`scripts/bw_b3_prep.py mix`, seed 0). Let D be the set of distillation rows.
   - `mix_v3` = D + |D| rows sampled from `runs/BW_data_v2/proposer_actions_train.jsonl`.
   - `mix_v2c` = 2|D| rows sampled from the same v2 file. Its first |D| rows are exactly the v2 half of `mix_v3`.
   - Distillation rows over 640 tokens (prompt + target + EOS) are dropped explicitly before sampling, and their
     count is recorded in `MIX_META.json`. Long plans make rows longer than the 256-token v2 cap: the naive
     unstack-then-build plan on 10 blocks is already 345 tokens.
4. **Training.** Both arms continue from the `BW_Pa_v2_s1` adapter with `train_generic.py --init-adapter`, using
   400 steps × batch 16, lr 1e-4 (same schedule as D-03), seed 1 and max-len 640 (both arms).
   - **Pa_v3** trains on `mix_v3`.
   - **Pa_v2c** trains on `mix_v2c`. This is the "same extra training, no search plans" control.
5. **Fresh test set** (`bw_b3_prep.py fresh`): `runs/BW_val_large/instances_b3_fresh.json`.
   - 110 instances with exactly the hard set's block-count distribution (6-15), built by the mirror generator.
   - Seed 20261011.
   - Excludes the invariants of hard, val_large, mirror and all 240 distillation problems.
   - Generated before training. No system is evaluated on it before both adapters are frozen in
     `runs/FREEZE_B3.json`, which records the adapter SHA-256s and this file's SHA-256.

## Primary test

S-direct (frozen B2 direct settings: greedy, K 1, M 256) on `b3fresh`, **Pa_v3 vs Pa_v2c**, paired exact McNemar.

- **"Self-distillation improves the model's own planning"** requires Pa_v3 − Pa_v2c ≥ +10 pp and p < 0.05.
- Anything else is reported as "partial" (positive, but either below threshold or p ≥ 0.05) or "no".

## Secondary (report-only; no gate)

- Pa_v3 vs Pa_v2 (the original, no extra training): S-direct on `b3fresh`, paired.
- The same two contrasts on the mirror set. Mirror was a B2 test set and was never used for selection or training.
- Success by block count, in-range (6-10) vs out-of-range (11-15). Also the truncation rate at M = 256, plan length
  against the exact optimum where it can be computed, and invalid-action rate.
- Distillation statistics:
  - verified count;
  - rows;
  - label precision (exact simulator, analysis only);
  - share of F-verified plans that are exactly invalid. These are labels the student learns from F's mistakes.

## Hard-set runs: conditional, once each, only if the primary reading is met

- **H1:** S-direct(Pa_v3) on the 110 hard instances vs the frozen B2 S-direct (5/110), paired by instance. The B2 run
  used the same instances and per-instance seeds.
- **H2:** S-beam with the frozen B2 beam configuration, swapping in only the Pa adapter (Pa_v3), vs the frozen B2
  S-beam (73/110), paired McNemar. This is about 10 GPU-h and is run sharded as in D-50.
- If the primary reading is not met, B3 makes no hard-set runs and the null is reported.

## Allowed claims

- "Trained on plans its own world model verified (no simulator labels), the proposer alone solves X/110 fresh
  hard-style problems vs Y/110 for the same extra training without them (paired)."
- On hard, only the H1/H2 results, under B2's regime caveats.
- **Not allowed:**
  - any claim from a mix, step count or adapter chosen after looking at `b3fresh`, mirror or hard results;
  - calling the student "simulator-free" if label precision < 1. In that case, report what share of labels were
    wrong.

# PREREG-3 results

Rendered mechanically by `scripts/p3_analysis.py` from the results JSON. Bank sha256 0fd18295d6ea...

## Verdict: STATUS: TRIGGER

> In all three T2 seeds the adapted model's no-op did not extend beyond 'void': accuracy on the seven unseen-status strata that contain no form of 'void' (the five not-applied words, the neutral and nonce words, the look-alikes of 'posted') was 0.01-0.18 (each <= 0.30), against C_void s0 0.99, s1 1.00, s2 1.00 (each >= 0.80); 1605 of the 2030 errors on those rows applied the event as if posted. Near-miss spellings of 'void' left unchanged: s0 0.95 (unchanged), s1 0.94 (unchanged), s2 0.99 (unchanged). The adapted model treats 'void' and its near-miss spellings (Void, VOID, voided), not 'any status other than posted', as the no-op cue.

> Base semantics (descriptive): unseen-status no-op accuracy OFF 0.02-0.06, Supa 1.7 0.10-0.14; status-sensitive pairs (unseen row unchanged while its posted twin was changed) OFF 0.00-0.03, Supa 1.7 0.00-0.03, adapted seeds 0.00-0.97. Raw no-op accuracy cannot separate status semantics from copy bias; the paired rate can.

> MAGNITUDE (add / transfer): DOES NOT GENERALISE, with a partial gain over OFF (M_bal|at: OFF 0.22, s0 0.46, s1 0.49, s2 0.61; M_amt|at: OFF 0.09, s0 0.12, s1 0.21, s2 0.06; the same bases in distribution, T_posted|at: OFF 0.39, s0 0.84, s1 0.87, s2 0.91; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 -0.31 [-0.46, -0.17]; s1 -0.28 [-0.40, -0.17]; s2 -0.34 [-0.48, -0.20]). OFF is also below the floor on every cell: P3 cannot separate an in-range adapter shortcut from a capacity limit shared with the base.
> MAGNITUDE (scale: multi-digit multiplication): DOES NOT GENERALISE, with a partial gain over OFF (M_bal+M_amt|sc: OFF 0.26, s0 0.45, s1 0.53, s2 0.44; the same bases in distribution, T_posted|sc: OFF 0.64, s0 0.94, s1 0.97, s2 0.91; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 -0.11 [-0.30, +0.09]; s1 -0.06 [-0.27, +0.15]; s2 -0.09 [-0.29, +0.11]). OFF is also below the floor on every cell: P3 cannot separate an in-range adapter shortcut from a capacity limit shared with the base.
> NAMING: MIXED (A_renamed|all: OFF 0.37, s0 0.71, s1 0.76, s2 0.77; A_many|all: OFF 0.41, s0 0.84, s1 0.90, s2 0.90; the same bases in distribution, T_posted|all: OFF 0.47, s0 0.87, s1 0.90, s2 0.91; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 -0.01 [-0.09, +0.07]; s1 +0.01 [-0.06, +0.08]; s2 +0.01 [-0.07, +0.08]). Cells below the floor: A_renamed|all|s0, A_renamed|all|s1, A_renamed|all|s2.
> FORMAT: DOES NOT GENERALISE, with a partial gain over OFF (F_keyorder|all: OFF 0.21, s0 0.37, s1 0.47, s2 0.37; the same bases in distribution, T_posted|all: OFF 0.47, s0 0.87, s1 0.90, s2 0.91; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 -0.24 [-0.38, -0.11]; s1 -0.17 [-0.31, -0.03]; s2 -0.28 [-0.42, -0.14]). OFF is also below the floor on every cell: P3 cannot separate an in-range adapter shortcut from a capacity limit shared with the base.

> For PREREG-2: the void gain is a token cue, not the stated rule; PREREG-2's limit 'fresh at payload level, not out-of-distribution' binds, and rule data must be surface-diversified before reuse.

Limits that always accompany the verdicts:

- Ledger only: Titan's one-step frame (1,500 rows: 100 bases x 15 strata) plus 80 four-step chain_one chains; T2 adapters s0 (PREREG-2's registered artifact), s1 and s2, unmerged on frozen Titan-Sparse.
- 'Rule' is tested on 16 non-posted strings: five words that mean 'not applied', five neutral or nonce words, three near-miss spellings of 'void' and three look-alikes of 'posted'. It is not a claim about every possible status string.
- Case variants of 'posted' ('Posted', 'POSTED') and synonyms of it (booked, settled, ...) are not tested: the frame's rule text does not settle them.
- Out-of-range means outside T2's training rows (statuses posted/void, amounts -9..12, |before| <= 98, accounts A-D, sorted keys); Titan's own 384 training rows share these ranges; Supa's pretraining data are unknown. T_posted and C_void are fresh in-distribution controls, and CH_S events are in range.
- One factor at a time: every one-step stratum is the same 100 bases with one factor changed (status string, balances, amount, account names, number of accounts, key order), so each out-of-range effect is a paired, kind-matched contrast with the posted twins; magnitude is judged separately for add/transfer and for scale (multi-digit multiplication); the CH_X chains mix factors and are descriptive.
- No training and no self-distillation or untargeted-LoRA control: a verdict describes what the adapted model does, not why; PREREG-2's other limits (horizon <= 4, one frame, no non-LM comparator at matched labels) still apply.
- Thresholds are absolute (the bar: accuracy 0.80 with Wilson lower bound 0.70; twins copied <= 0.10; trigger 0.30) and were fixed before any output. Under independence a true accuracy of 0.85 passes one cell with probability about 0.93 but all 24 unseen-status cells with about 0.19; 0.90 passes all 24 with about 0.98 (power table in the dry run; the strata share bases, so the joint probabilities are lower bounds).
- Chains are descriptive: K = 4 candidates per parent (PREREG-2: 8), so normalised regret is not comparable to PREREG-2's 0.544 -> 0.056; phi clips balances at +-300, so regret is partly blind to large-magnitude errors in CH_X (exact@k is the primary chain measure there); 10 parents per stratum, so bootstrap intervals are indicative.

## One-step accuracy by stratum (accuracy [Wilson 95%])

| stratum | group | gold | OFF (Titan-Sparse) | T2 s0 | T2 s1 | T2 s2 | Supa 1.7 | copy floor |
|---|---|---|---|---|---|---|---|---|
| T_posted | status | apply | 0.47 [0.38, 0.57] | 0.87 [0.79, 0.92] | 0.90 [0.83, 0.94] | 0.91 [0.84, 0.95] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.04] |
| C_void | status | noop | 0.05 [0.02, 0.11] | 0.99 [0.95, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 0.11 [0.06, 0.19] | 1.00 [0.96, 1.00] |
| U_pending | status | noop | 0.04 [0.02, 0.10] | 0.02 [0.01, 0.07] | 0.01 [0.00, 0.05] | 0.02 [0.01, 0.07] | 0.11 [0.06, 0.19] | 1.00 [0.96, 1.00] |
| U_draft | status | noop | 0.03 [0.01, 0.08] | 0.01 [0.00, 0.05] | 0.01 [0.00, 0.05] | 0.02 [0.01, 0.07] | 0.10 [0.06, 0.17] | 1.00 [0.96, 1.00] |
| U_cancelled | status | noop | 0.04 [0.02, 0.10] | 0.12 [0.07, 0.20] | 0.05 [0.02, 0.11] | 0.18 [0.12, 0.27] | 0.12 [0.07, 0.20] | 1.00 [0.96, 1.00] |
| U_reversed | status | noop | 0.04 [0.02, 0.10] | 0.03 [0.01, 0.08] | 0.02 [0.01, 0.07] | 0.01 [0.00, 0.05] | 0.14 [0.09, 0.22] | 1.00 [0.96, 1.00] |
| U_held | status | noop | 0.03 [0.01, 0.08] | 0.02 [0.01, 0.07] | 0.01 [0.00, 0.05] | 0.02 [0.01, 0.07] | 0.11 [0.06, 0.19] | 1.00 [0.96, 1.00] |
| U_neutral | status | noop | 0.02 [0.01, 0.07] | 0.02 [0.01, 0.07] | 0.01 [0.00, 0.05] | 0.02 [0.01, 0.07] | 0.10 [0.06, 0.17] | 1.00 [0.96, 1.00] |
| N_voidcase | status | noop | 0.06 [0.03, 0.12] | 0.95 [0.89, 0.98] | 0.94 [0.88, 0.97] | 0.99 [0.95, 1.00] | 0.13 [0.08, 0.21] | 1.00 [0.96, 1.00] |
| N_postedlike | status | noop | 0.03 [0.01, 0.08] | 0.03 [0.01, 0.08] | 0.02 [0.01, 0.07] | 0.05 [0.02, 0.11] | 0.13 [0.08, 0.21] | 1.00 [0.96, 1.00] |
| M_bal | magnitude | apply | 0.28 [0.20, 0.37] | 0.50 [0.40, 0.60] | 0.57 [0.47, 0.66] | 0.60 [0.50, 0.69] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.04] |
| M_amt | magnitude | apply | 0.10 [0.06, 0.17] | 0.19 [0.13, 0.28] | 0.25 [0.18, 0.34] | 0.14 [0.09, 0.22] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.04] |
| A_renamed | naming | apply | 0.37 [0.28, 0.47] | 0.71 [0.61, 0.79] | 0.76 [0.67, 0.83] | 0.77 [0.68, 0.84] | 0.06 [0.03, 0.12] | 0.00 [0.00, 0.04] |
| A_many | naming | apply | 0.41 [0.32, 0.51] | 0.84 [0.76, 0.90] | 0.90 [0.83, 0.94] | 0.90 [0.83, 0.94] | 0.04 [0.02, 0.10] | 0.00 [0.00, 0.04] |
| F_keyorder | format | apply | 0.21 [0.14, 0.30] | 0.37 [0.28, 0.47] | 0.47 [0.38, 0.57] | 0.37 [0.28, 0.47] | 0.03 [0.01, 0.08] | 0.00 [0.00, 0.04] |

## V1 status: per seed

| seed | unseen strata meet the bar | failing unseen strata | twin acc | twins copied | twin guard | trigger-set <= 0.30 | C_void | errors applied / wrong (trigger set) |
|---|---|---|---|---|---|---|---|---|
| s0 | False | U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_postedlike | 0.87 | 0.02 | True | True | 0.99 | 496 / 675 |
| s1 | False | U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_postedlike | 0.90 | 0.01 | True | True | 1.00 | 570 / 687 |
| s2 | False | U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_postedlike | 0.91 | 0.02 | True | True | 1.00 | 539 / 668 |

Verdict TRIGGER; attribution None; PARTIAL patterns -; adapter gain Holm-significant in all 24 cells: False (in any: True); OFF meets the bar on every unseen status: False (OFF twins copied 0.03). Expectations: E1 C_void ON >= 0.95 in every seed: holds; E2 C_void OFF <= 0.30: holds

## McNemar ON vs OFF (exact, two-sided; Holm within each family: status 24, gen 18)

| family | cell | ON only | OFF only | p | Holm p | significant | direction |
|---|---|---|---|---|---|---|---|
| status | U_pending|s0 | 2 | 4 | 0.6875 | 1 | False | off>on |
| status | U_pending|s1 | 0 | 3 | 0.25 | 1 | False | off>on |
| status | U_pending|s2 | 2 | 4 | 0.6875 | 1 | False | off>on |
| status | U_draft|s0 | 1 | 3 | 0.625 | 1 | False | off>on |
| status | U_draft|s1 | 0 | 2 | 0.5 | 1 | False | off>on |
| status | U_draft|s2 | 2 | 3 | 1 | 1 | False | off>on |
| status | U_cancelled|s0 | 11 | 3 | 0.05737 | 1 | False | on>off |
| status | U_cancelled|s1 | 3 | 2 | 1 | 1 | False | on>off |
| status | U_cancelled|s2 | 16 | 2 | 0.001312 | 0.02756 | True | on>off |
| status | U_reversed|s0 | 3 | 4 | 1 | 1 | False | off>on |
| status | U_reversed|s1 | 1 | 3 | 0.625 | 1 | False | off>on |
| status | U_reversed|s2 | 1 | 4 | 0.375 | 1 | False | off>on |
| status | U_held|s0 | 2 | 3 | 1 | 1 | False | off>on |
| status | U_held|s1 | 0 | 2 | 0.5 | 1 | False | off>on |
| status | U_held|s2 | 2 | 3 | 1 | 1 | False | off>on |
| status | U_neutral|s0 | 2 | 2 | 1 | 1 | False | tie |
| status | U_neutral|s1 | 0 | 1 | 1 | 1 | False | off>on |
| status | U_neutral|s2 | 2 | 2 | 1 | 1 | False | tie |
| status | N_voidcase|s0 | 89 | 0 | 3.231e-27 | 7.432e-26 | True | on>off |
| status | N_voidcase|s1 | 88 | 0 | 6.462e-27 | 1.422e-25 | True | on>off |
| status | N_voidcase|s2 | 93 | 0 | 2.019e-28 | 4.847e-27 | True | on>off |
| status | N_postedlike|s0 | 3 | 3 | 1 | 1 | False | tie |
| status | N_postedlike|s1 | 1 | 2 | 1 | 1 | False | off>on |
| status | N_postedlike|s2 | 4 | 2 | 0.6875 | 1 | False | on>off |
| gen | M_bal|at|s0 | 22 | 6 | 0.003719 | 0.02603 | True | on>off |
| gen | M_bal|at|s1 | 23 | 5 | 0.0009122 | 0.00821 | True | on>off |
| gen | M_bal|at|s2 | 29 | 3 | 2.556e-06 | 3.067e-05 | True | on>off |
| gen | M_amt|at|s0 | 4 | 2 | 0.6875 | 1 | False | on>off |
| gen | M_amt|at|s1 | 8 | 0 | 0.007812 | 0.03125 | True | on>off |
| gen | M_amt|at|s2 | 0 | 2 | 0.5 | 1 | False | off>on |
| gen | M_bal+M_amt|sc|s0 | 16 | 3 | 0.004425 | 0.02655 | True | on>off |
| gen | M_bal+M_amt|sc|s1 | 20 | 2 | 0.0001211 | 0.001211 | True | on>off |
| gen | M_bal+M_amt|sc|s2 | 16 | 4 | 0.01182 | 0.03545 | True | on>off |
| gen | A_renamed|all|s0 | 36 | 2 | 5.399e-09 | 7.018e-08 | True | on>off |
| gen | A_renamed|all|s1 | 39 | 0 | 3.638e-12 | 5.821e-11 | True | on>off |
| gen | A_renamed|all|s2 | 43 | 3 | 4.622e-10 | 6.471e-09 | True | on>off |
| gen | A_many|all|s0 | 46 | 3 | 6.981e-11 | 1.047e-09 | True | on>off |
| gen | A_many|all|s1 | 49 | 0 | 3.553e-15 | 6.395e-14 | True | on>off |
| gen | A_many|all|s2 | 51 | 2 | 3.18e-13 | 5.405e-12 | True | on>off |
| gen | F_keyorder|all|s0 | 23 | 7 | 0.005223 | 0.02655 | True | on>off |
| gen | F_keyorder|all|s1 | 30 | 4 | 6.165e-06 | 6.781e-05 | True | on>off |
| gen | F_keyorder|all|s2 | 21 | 5 | 0.002494 | 0.01995 | True | on>off |

## Magnitude / naming / format cells

| group | verdict | cell classes |
|---|---|---|
| magnitude_at | DOES NOT GENERALISE, with a partial gain over OFF | M_bal|at|s0 GAIN_BELOW_FLOOR, M_bal|at|s1 GAIN_BELOW_FLOOR, M_bal|at|s2 GAIN_BELOW_FLOOR, M_amt|at|s0 LOW_NOGAIN, M_amt|at|s1 GAIN_BELOW_FLOOR, M_amt|at|s2 LOW_NOGAIN |
| magnitude_scale | DOES NOT GENERALISE, with a partial gain over OFF | M_bal+M_amt|sc|s0 GAIN_BELOW_FLOOR, M_bal+M_amt|sc|s1 GAIN_BELOW_FLOOR, M_bal+M_amt|sc|s2 GAIN_BELOW_FLOOR |
| naming | MIXED | A_renamed|all|s0 GAIN_BELOW_FLOOR, A_renamed|all|s1 GAIN_BELOW_FLOOR, A_renamed|all|s2 GAIN_BELOW_FLOOR, A_many|all|s0 HIGH_GAIN, A_many|all|s1 HIGH_GAIN, A_many|all|s2 HIGH_GAIN |
| format | DOES NOT GENERALISE, with a partial gain over OFF | F_keyorder|all|s0 GAIN_BELOW_FLOOR, F_keyorder|all|s1 GAIN_BELOW_FLOOR, F_keyorder|all|s2 GAIN_BELOW_FLOOR |

### Cell accuracies (kind classes: at = add + transfer, sc = scale)

| cell | OFF (Titan-Sparse) | T2 s0 | T2 s1 | T2 s2 | Supa 1.7 | copy floor |
|---|---|---|---|---|---|---|
| T_posted|all | 0.47 [0.38, 0.57] | 0.87 [0.79, 0.92] | 0.90 [0.83, 0.94] | 0.91 [0.84, 0.95] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.04] |
| T_posted|at | 0.39 [0.28, 0.51] | 0.84 [0.73, 0.91] | 0.87 [0.76, 0.93] | 0.91 [0.82, 0.96] | 0.01 [0.00, 0.08] | 0.00 [0.00, 0.05] |
| T_posted|sc | 0.64 [0.47, 0.78] | 0.94 [0.80, 0.98] | 0.97 [0.85, 0.99] | 0.91 [0.76, 0.97] | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] |
| M_bal|at | 0.22 [0.14, 0.34] | 0.46 [0.35, 0.58] | 0.49 [0.38, 0.61] | 0.61 [0.49, 0.72] | 0.01 [0.00, 0.08] | 0.00 [0.00, 0.05] |
| M_amt|at | 0.09 [0.04, 0.18] | 0.12 [0.06, 0.22] | 0.21 [0.13, 0.32] | 0.06 [0.02, 0.14] | 0.01 [0.00, 0.08] | 0.00 [0.00, 0.05] |
| M_bal+M_amt|sc | 0.26 [0.17, 0.37] | 0.45 [0.34, 0.57] | 0.53 [0.41, 0.65] | 0.44 [0.33, 0.56] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] |
| A_renamed|all | 0.37 [0.28, 0.47] | 0.71 [0.61, 0.79] | 0.76 [0.67, 0.83] | 0.77 [0.68, 0.84] | 0.06 [0.03, 0.12] | 0.00 [0.00, 0.04] |
| A_many|all | 0.41 [0.32, 0.51] | 0.84 [0.76, 0.90] | 0.90 [0.83, 0.94] | 0.90 [0.83, 0.94] | 0.04 [0.02, 0.10] | 0.00 [0.00, 0.04] |
| F_keyorder|all | 0.21 [0.14, 0.30] | 0.37 [0.28, 0.47] | 0.47 [0.38, 0.57] | 0.37 [0.28, 0.47] | 0.03 [0.01, 0.08] | 0.00 [0.00, 0.04] |
| M_bal|sc | 0.39 [0.25, 0.56] | 0.58 [0.41, 0.73] | 0.73 [0.56, 0.85] | 0.58 [0.41, 0.73] | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] |
| M_amt|sc | 0.12 [0.05, 0.27] | 0.33 [0.20, 0.50] | 0.33 [0.20, 0.50] | 0.30 [0.17, 0.47] | 0.00 [0.00, 0.10] | 0.00 [0.00, 0.10] |

### In distribution vs out of range within each arm (paired by base; exact McNemar, descriptive)

| arm | cell | acc out | acc in (twins) | out only | in only | p |
|---|---|---|---|---|---|---|
| OFF (Titan-Sparse) | M_bal|at | 0.22 | 0.39 | 6 | 17 | 0.0347 |
| OFF (Titan-Sparse) | M_bal|sc | 0.39 | 0.64 | 1 | 9 | 0.0215 |
| OFF (Titan-Sparse) | M_amt|at | 0.09 | 0.39 | 1 | 21 | 1.1e-05 |
| OFF (Titan-Sparse) | M_amt|sc | 0.12 | 0.64 | 2 | 19 | 0.000221 |
| OFF (Titan-Sparse) | A_renamed|all | 0.37 | 0.47 | 6 | 16 | 0.0525 |
| OFF (Titan-Sparse) | A_many|all | 0.41 | 0.47 | 2 | 8 | 0.109 |
| OFF (Titan-Sparse) | F_keyorder|all | 0.21 | 0.47 | 6 | 32 | 2.43e-05 |
| T2 s0 | M_bal|at | 0.46 | 0.84 | 4 | 29 | 1.09e-05 |
| T2 s0 | M_bal|sc | 0.58 | 0.94 | 0 | 12 | 0.000488 |
| T2 s0 | M_amt|at | 0.12 | 0.84 | 1 | 49 | 9.06e-14 |
| T2 s0 | M_amt|sc | 0.33 | 0.94 | 0 | 20 | 1.91e-06 |
| T2 s0 | A_renamed|all | 0.71 | 0.87 | 3 | 19 | 0.000855 |
| T2 s0 | A_many|all | 0.84 | 0.87 | 3 | 6 | 0.508 |
| T2 s0 | F_keyorder|all | 0.37 | 0.87 | 0 | 50 | 1.78e-15 |
| T2 s1 | M_bal|at | 0.49 | 0.87 | 0 | 25 | 5.96e-08 |
| T2 s1 | M_bal|sc | 0.73 | 0.97 | 0 | 8 | 0.00781 |
| T2 s1 | M_amt|at | 0.21 | 0.87 | 0 | 44 | 1.14e-13 |
| T2 s1 | M_amt|sc | 0.33 | 0.97 | 0 | 21 | 9.54e-07 |
| T2 s1 | A_renamed|all | 0.76 | 0.90 | 2 | 16 | 0.00131 |
| T2 s1 | A_many|all | 0.90 | 0.90 | 2 | 2 | 1 |
| T2 s1 | F_keyorder|all | 0.47 | 0.90 | 0 | 43 | 2.27e-13 |
| T2 s2 | M_bal|at | 0.61 | 0.91 | 3 | 23 | 8.8e-05 |
| T2 s2 | M_bal|sc | 0.58 | 0.91 | 1 | 12 | 0.00342 |
| T2 s2 | M_amt|at | 0.06 | 0.91 | 0 | 57 | 1.39e-17 |
| T2 s2 | M_amt|sc | 0.30 | 0.91 | 0 | 20 | 1.91e-06 |
| T2 s2 | A_renamed|all | 0.77 | 0.91 | 2 | 16 | 0.00131 |
| T2 s2 | A_many|all | 0.90 | 0.91 | 1 | 2 | 1 |
| T2 s2 | F_keyorder|all | 0.37 | 0.91 | 0 | 54 | 1.11e-16 |
| Supa 1.7 | M_bal|at | 0.01 | 0.01 | 1 | 1 | 1 |
| Supa 1.7 | M_bal|sc | 0.00 | 0.00 | 0 | 0 | 1 |
| Supa 1.7 | M_amt|at | 0.01 | 0.01 | 1 | 1 | 1 |
| Supa 1.7 | M_amt|sc | 0.00 | 0.00 | 0 | 0 | 1 |
| Supa 1.7 | A_renamed|all | 0.06 | 0.01 | 5 | 0 | 0.0625 |
| Supa 1.7 | A_many|all | 0.04 | 0.01 | 4 | 1 | 0.375 |
| Supa 1.7 | F_keyorder|all | 0.03 | 0.01 | 3 | 1 | 0.625 |
| copy floor | M_bal|at | 0.00 | 0.00 | 0 | 0 | 1 |
| copy floor | M_bal|sc | 0.00 | 0.00 | 0 | 0 | 1 |
| copy floor | M_amt|at | 0.00 | 0.00 | 0 | 0 | 1 |
| copy floor | M_amt|sc | 0.00 | 0.00 | 0 | 0 | 1 |
| copy floor | A_renamed|all | 0.00 | 0.00 | 0 | 0 | 1 |
| copy floor | A_many|all | 0.00 | 0.00 | 0 | 0 | 1 |
| copy floor | F_keyorder|all | 0.00 | 0.00 | 0 | 0 | 1 |

### Retained adapter gain [(ON - OFF) out of range minus (ON - OFF) on the same bases' twins; base bootstrap]

| group | arm | gain out | gain in (same bases) | retained | 95% CI |
|---|---|---|---|---|---|
| magnitude_at | s0 | +0.134 | +0.448 | -0.313 | [-0.455, -0.172] |
| magnitude_at | s1 | +0.194 | +0.478 | -0.284 | [-0.396, -0.172] |
| magnitude_at | s2 | +0.179 | +0.522 | -0.343 | [-0.478, -0.201] |
| magnitude_at | supa | -0.142 | -0.373 | +0.231 | [+0.112, +0.351] |
| magnitude_scale | s0 | +0.197 | +0.303 | -0.106 | [-0.303, +0.091] |
| magnitude_scale | s1 | +0.273 | +0.333 | -0.061 | [-0.273, +0.152] |
| magnitude_scale | s2 | +0.182 | +0.273 | -0.091 | [-0.288, +0.106] |
| magnitude_scale | supa | -0.258 | -0.636 | +0.379 | [+0.212, +0.530] |
| naming | s0 | +0.385 | +0.400 | -0.015 | [-0.090, +0.065] |
| naming | s1 | +0.440 | +0.430 | +0.010 | [-0.060, +0.080] |
| naming | s2 | +0.445 | +0.440 | +0.005 | [-0.070, +0.080] |
| naming | supa | -0.340 | -0.460 | +0.120 | [+0.045, +0.195] |
| format | s0 | +0.160 | +0.400 | -0.240 | [-0.380, -0.110] |
| format | s1 | +0.260 | +0.430 | -0.170 | [-0.310, -0.030] |
| format | s2 | +0.160 | +0.440 | -0.280 | [-0.420, -0.140] |
| format | supa | -0.180 | -0.460 | +0.280 | [+0.170, +0.390] |

## Pairs and status sensitivity (descriptive)

Both correct: the unseen row and its posted twin. Sensitive: the unseen row unchanged while the twin was changed (correct or wrong value, not copied).

| stratum | OFF (Titan-Sparse) both / sensitive | T2 s0 both / sensitive | T2 s1 both / sensitive | T2 s2 both / sensitive | Supa 1.7 both / sensitive | copy floor both / sensitive |
|---|---|---|---|---|---|---|
| U_pending | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| U_draft | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| U_cancelled | 0.00 / 0.01 | 0.10 / 0.11 | 0.02 / 0.04 | 0.14 / 0.16 | 0.00 / 0.01 | 0.00 / 0.00 |
| U_reversed | 0.00 / 0.02 | 0.03 / 0.03 | 0.01 / 0.01 | 0.00 / 0.00 | 0.00 / 0.03 | 0.00 / 0.00 |
| U_held | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 |
| U_neutral | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| N_voidcase | 0.02 / 0.03 | 0.83 / 0.93 | 0.85 / 0.92 | 0.90 / 0.97 | 0.00 / 0.01 | 0.00 / 0.00 |
| N_postedlike | 0.00 / 0.00 | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.03 | 0.00 / 0.03 | 0.00 / 0.00 |
| C_void | 0.01 / 0.02 | 0.87 / 0.97 | 0.90 / 0.98 | 0.91 / 0.98 | 0.00 / 0.00 | 0.00 / 0.00 |

Copied rate over the magnitude / naming / format rows: off 0.12, s0 0.01, s1 0.02, s2 0.03, supa 0.16, copy 1.00

## Error classes (descriptive)

| stratum | OFF (Titan-Sparse) | T2 s0 | T2 s1 | T2 s2 | Supa 1.7 | copy floor |
|---|---|---|---|---|---|---|
| T_posted | copied 3, wrong_value 50 | copied 2, wrong_value 11 | copied 1, unparsed 1, wrong_value 8 | copied 2, wrong_value 7 | copied 11, unparsed 5, wrong_value 83 | copied 100 |
| C_void | applied_as_posted 42, other_change 53 | other_change 1 | - | - | applied_as_posted 4, other_change 81, unparsed 4 | - |
| U_pending | applied_as_posted 46, other_change 50 | applied_as_posted 84, other_change 14 | applied_as_posted 90, other_change 8, unparsed 1 | applied_as_posted 87, other_change 11 | applied_as_posted 3, other_change 82, unparsed 4 | - |
| U_draft | applied_as_posted 46, other_change 51 | applied_as_posted 86, other_change 13 | applied_as_posted 90, other_change 8, unparsed 1 | applied_as_posted 90, other_change 8 | applied_as_posted 3, other_change 82, unparsed 5 | - |
| U_cancelled | applied_as_posted 44, other_change 52 | applied_as_posted 34, other_change 54 | applied_as_posted 59, other_change 36 | applied_as_posted 36, other_change 46 | applied_as_posted 3, other_change 82, unparsed 3 | - |
| U_reversed | applied_as_posted 40, other_change 56 | applied_as_posted 40, other_change 57 | applied_as_posted 60, other_change 38 | applied_as_posted 61, other_change 38 | applied_as_posted 4, other_change 78, unparsed 4 | - |
| U_held | applied_as_posted 44, other_change 53 | applied_as_posted 87, other_change 11 | applied_as_posted 91, other_change 7, unparsed 1 | applied_as_posted 91, other_change 7 | applied_as_posted 2, other_change 82, unparsed 5 | - |
| U_neutral | applied_as_posted 45, other_change 53 | applied_as_posted 87, other_change 11 | applied_as_posted 89, other_change 9, unparsed 1 | applied_as_posted 90, other_change 7, unparsed 1 | applied_as_posted 2, other_change 82, unparsed 6 | - |
| N_voidcase | applied_as_posted 43, other_change 51 | other_change 5 | other_change 6 | other_change 1 | applied_as_posted 5, other_change 79, unparsed 3 | - |
| N_postedlike | applied_as_posted 46, other_change 51 | applied_as_posted 78, other_change 19 | applied_as_posted 91, other_change 7 | applied_as_posted 84, other_change 11 | applied_as_posted 2, other_change 81, unparsed 4 | - |
| M_bal | copied 20, unparsed 1, wrong_value 51 | unparsed 3, wrong_value 47 | unparsed 2, wrong_value 41 | copied 3, unparsed 2, wrong_value 35 | copied 22, unparsed 32, wrong_value 45 | copied 100 |
| M_amt | copied 13, wrong_value 77 | unparsed 1, wrong_value 80 | copied 1, wrong_value 74 | wrong_value 86 | copied 6, unparsed 10, wrong_value 83 | copied 100 |
| A_renamed | copied 8, wrong_value 55 | wrong_value 29 | copied 3, wrong_value 21 | copied 2, wrong_value 21 | copied 8, unparsed 11, wrong_value 75 | copied 100 |
| A_many | copied 5, wrong_value 54 | copied 1, wrong_value 15 | copied 1, wrong_value 9 | copied 2, wrong_value 8 | copied 10, unparsed 5, wrong_value 81 | copied 100 |
| F_keyorder | copied 15, unparsed 1, wrong_value 63 | copied 2, unparsed 1, wrong_value 60 | copied 5, unparsed 1, wrong_value 47 | copied 7, unparsed 2, wrong_value 54 | copied 34, unparsed 24, wrong_value 39 | copied 100 |

## Breakdowns (descriptive; registered sub-variants)

- U_neutral: off: active 0.00, approved 0.05, dax 0.00, ok 0.00, wug 0.05; s0: active 0.00, approved 0.05, dax 0.00, ok 0.00, wug 0.05; s1: active 0.00, approved 0.00, dax 0.00, ok 0.00, wug 0.05; s2: active 0.00, approved 0.05, dax 0.00, ok 0.00, wug 0.05; supa: active 0.15, approved 0.05, dax 0.15, ok 0.05, wug 0.10; copy: active 1.00, approved 1.00, dax 1.00, ok 1.00, wug 1.00
- N_voidcase: off: VOID 0.03, Void 0.03, voided 0.12; s0: VOID 0.97, Void 0.88, voided 1.00; s1: VOID 1.00, Void 0.82, voided 1.00; s2: VOID 1.00, Void 0.97, voided 1.00; supa: VOID 0.12, Void 0.06, voided 0.21; copy: VOID 1.00, Void 1.00, voided 1.00
- N_postedlike: off: not posted 0.06, not_posted 0.03, unposted 0.00; s0: not posted 0.06, not_posted 0.03, unposted 0.00; s1: not posted 0.03, not_posted 0.03, unposted 0.00; s2: not posted 0.06, not_posted 0.03, unposted 0.06; supa: not posted 0.21, not_posted 0.09, unposted 0.09; copy: not posted 1.00, not_posted 1.00, unposted 1.00
- M_amt: off: 100..500 0.10, 13..99 0.10; s0: 100..500 0.16, 13..99 0.22; s1: 100..500 0.18, 13..99 0.32; s2: 100..500 0.12, 13..99 0.16; supa: 100..500 0.02, 13..99 0.00; copy: 100..500 0.00, 13..99 0.00
- A_renamed: off: letters 0.38, words 0.36; s0: letters 0.82, words 0.60; s1: letters 0.90, words 0.62; s2: letters 0.88, words 0.66; supa: letters 0.02, words 0.10; copy: letters 0.00, words 0.00
- A_many: off: 5 0.40, 6 0.40, 7 0.40, 8 0.44; s0: 5 0.88, 6 0.84, 7 0.84, 8 0.80; s1: 5 0.84, 6 0.92, 7 0.96, 8 0.88; s2: 5 0.96, 6 0.92, 7 0.88, 8 0.84; supa: 5 0.08, 6 0.00, 7 0.04, 8 0.04; copy: 5 0.00, 6 0.00, 7 0.00, 8 0.00
- F_keyorder: off: all_orders 0.16, top_only 0.26; s0: all_orders 0.28, top_only 0.46; s1: all_orders 0.30, top_only 0.64; s2: all_orders 0.28, top_only 0.46; supa: all_orders 0.02, top_only 0.04; copy: all_orders 0.00, top_only 0.00

## Chains (descriptive)

K = 4 candidates per parent (PREREG-2: 8), so normalised regret is not comparable to PREREG-2's; phi clips balances at +-300, so regret is partly blind to large-magnitude errors (exact@k is the primary chain measure for CH_X); 10 parents per stratum, so the bootstrap intervals are indicative only.

### CH_S (10 parents x 4 candidates): accounts A-D, balances and amounts in range, sorted keys; each event posted (p .50), an unseen status word (p .35, 10 equally: 5 not-applied, 5 neutral / nonce) or a near-miss / look-alike (p .15, 6 equally); >= 1 posted and >= 1 non-posted event

Gold states beyond the phi clip (|balance| > 300): k1 0 of 40, k2 0 of 40, k4 0 of 40

| arm | exact k1 | exact k2 | exact k4 | regret k1 | regret k2 | regret k4 |
|---|---|---|---|---|---|---|
| OFF (Titan-Sparse) | 0.200 | 0.050 | 0.000 | 0.491 | 0.734 | 0.869 |
| T2 s0 | 0.625 | 0.325 | 0.100 | 0.293 | 0.360 | 0.482 |
| T2 s1 | 0.600 | 0.300 | 0.125 | 0.307 | 0.354 | 0.527 |
| T2 s2 | 0.600 | 0.300 | 0.050 | 0.297 | 0.330 | 0.523 |
| Supa 1.7 | 0.100 | 0.025 | 0.000 | 0.617 | 0.793 | 0.844 |
| copy floor | 0.475 | 0.225 | 0.000 | 1.002 | 0.981 | 0.969 |

Floors (regret): k1: F_copy_s0 1.002, F_first_only 0.000, F_ignore_status 0.412, F_last_only 0.000, F_touched 0.026; k2: F_copy_s0 0.981, F_first_only 0.439, F_ignore_status 0.409, F_last_only 0.421, F_touched 0.189; k4: F_copy_s0 0.969, F_first_only 0.667, F_ignore_status 0.585, F_last_only 0.615, F_touched 0.260; copy-chain exact floor k1 0.475, k2 0.225, k4 0.000

Step accuracy on correct-input steps: off: near_void 0.00 (n 6), neutral 0.00 (n 10), posted 0.44 (n 25), posted_like 0.00 (n 4), unseen 0.00 (n 6); s0: near_void 1.00 (n 10), neutral 0.00 (n 14), posted 0.95 (n 41), posted_like 0.00 (n 7), unseen 0.00 (n 13); s1: near_void 1.00 (n 10), neutral 0.00 (n 14), posted 0.95 (n 40), posted_like 0.00 (n 6), unseen 0.00 (n 13); s2: near_void 1.00 (n 7), neutral 0.00 (n 14), posted 0.90 (n 41), posted_like 0.00 (n 7), unseen 0.00 (n 13); supa: near_void 0.17 (n 6), neutral 0.11 (n 9), posted 0.04 (n 23), posted_like 0.33 (n 3), unseen 0.25 (n 4); copy: near_void 1.00 (n 6), neutral 1.00 (n 11), posted 0.00 (n 40), posted_like 1.00 (n 4), unseen 1.00 (n 8)

### CH_X (10 parents x 4 candidates): accounts A-D, start in range; each event posted (p .60) or non-posted (p .40; the 16 non-posted strings equally); |amount| 13..500 on each add/transfer event w.p. .40 (scale keeps a trained amount, so products do not compound); each step's keys reordered w.p. .50; >= 1 posted, >= 1 non-posted, >= 1 posted OOR-amount event and >= 1 reordered step

Gold states beyond the phi clip (|balance| > 300): k1 3 of 40, k2 7 of 40, k4 13 of 40

| arm | exact k1 | exact k2 | exact k4 | regret k1 | regret k2 | regret k4 |
|---|---|---|---|---|---|---|
| OFF (Titan-Sparse) | 0.100 | 0.000 | 0.000 | 0.522 | 0.648 | 0.857 |
| T2 s0 | 0.250 | 0.075 | 0.000 | 0.418 | 0.471 | 0.445 |
| T2 s1 | 0.400 | 0.125 | 0.000 | 0.457 | 0.578 | 0.547 |
| T2 s2 | 0.375 | 0.100 | 0.000 | 0.403 | 0.537 | 0.519 |
| Supa 1.7 | 0.175 | 0.000 | 0.000 | 0.773 | 0.846 | 0.999 |
| copy floor | 0.475 | 0.150 | 0.000 | 1.017 | 1.008 | 1.009 |

Floors (regret): k1: F_copy_s0 1.017, F_first_only 0.000, F_ignore_status 0.383, F_last_only 0.000, F_touched 0.283; k2: F_copy_s0 1.008, F_first_only 0.448, F_ignore_status 0.346, F_last_only 0.385, F_touched 0.474; k4: F_copy_s0 1.009, F_first_only 0.775, F_ignore_status 0.220, F_last_only 0.634, F_touched 0.694; copy-chain exact floor k1 0.475, k2 0.150, k4 0.000

Step accuracy on correct-input steps: off: near_void 0.00 (n 3), near_void+reordered 0.00 (n 1), neutral 0.00 (n 1), neutral+reordered 0.00 (n 3), posted 0.33 (n 3), posted+reordered 0.14 (n 7), posted_like 0.00 (n 2), posted_like+reordered 1.00 (n 1), posted_oor_amount 0.00 (n 7), posted_oor_amount+reordered 0.00 (n 7), unseen 0.00 (n 4), unseen+reordered 0.20 (n 5); s0: near_void 0.67 (n 3), near_void+reordered 1.00 (n 1), neutral 0.00 (n 1), neutral+reordered 0.00 (n 4), posted 1.00 (n 7), posted+reordered 0.50 (n 6), posted_like 0.00 (n 3), posted_like+reordered 0.00 (n 2), posted_oor_amount 0.12 (n 8), posted_oor_amount+reordered 0.00 (n 10), unseen 0.00 (n 4), unseen+reordered 0.17 (n 6); s1: near_void 1.00 (n 3), near_void+reordered 1.00 (n 1), neutral 0.00 (n 2), neutral+reordered 0.25 (n 4), posted 1.00 (n 10), posted+reordered 0.57 (n 7), posted_like 0.00 (n 3), posted_like+reordered 0.50 (n 2), posted_oor_amount 0.20 (n 10), posted_oor_amount+reordered 0.00 (n 12), unseen 0.00 (n 4), unseen+reordered 0.33 (n 6); s2: near_void 1.00 (n 3), near_void+reordered 1.00 (n 1), neutral 0.00 (n 2), neutral+reordered 0.00 (n 3), posted 0.89 (n 9), posted+reordered 0.43 (n 7), posted_like 0.00 (n 3), posted_like+reordered 0.50 (n 2), posted_oor_amount 0.10 (n 10), posted_oor_amount+reordered 0.00 (n 11), unseen 0.00 (n 4), unseen+reordered 0.71 (n 7); supa: near_void 0.00 (n 3), near_void+reordered 0.50 (n 2), neutral 0.00 (n 1), neutral+reordered 0.33 (n 3), posted 0.00 (n 4), posted+reordered 0.00 (n 6), posted_like 0.33 (n 3), posted_like+reordered 0.00 (n 2), posted_oor_amount 0.00 (n 7), posted_oor_amount+reordered 0.00 (n 7), unseen 0.67 (n 3), unseen+reordered 0.33 (n 6); copy: near_void 1.00 (n 3), near_void+reordered 1.00 (n 2), neutral 1.00 (n 2), neutral+reordered 1.00 (n 3), posted 0.00 (n 8), posted+reordered 0.00 (n 8), posted_like 1.00 (n 3), posted_like+reordered 1.00 (n 2), posted_oor_amount 0.00 (n 13), posted_oor_amount+reordered 0.00 (n 11), unseen 1.00 (n 4), unseen+reordered 1.00 (n 7)

| chain contrast (k4; 10 parents, indicative) | point | 95% CI |
|---|---|---|
| CH_S|s0-off|k4|exact | +0.100 | [+0.025, +0.175] |
| CH_S|s0-off|k4|regret | -0.386 | [-0.703, -0.042] |
| CH_S|s1-off|k4|exact | +0.125 | [+0.050, +0.200] |
| CH_S|s1-off|k4|regret | -0.342 | [-0.660, -0.002] |
| CH_S|s2-off|k4|exact | +0.050 | [+0.000, +0.125] |
| CH_S|s2-off|k4|regret | -0.346 | [-0.666, -0.008] |
| CH_S|supa-off|k4|exact | +0.000 | [+0.000, +0.000] |
| CH_S|supa-off|k4|regret | -0.025 | [-0.261, +0.190] |
| CH_X|s0-off|k4|exact | +0.000 | [+0.000, +0.000] |
| CH_X|s0-off|k4|regret | -0.412 | [-0.628, -0.205] |
| CH_X|s1-off|k4|exact | +0.000 | [+0.000, +0.000] |
| CH_X|s1-off|k4|regret | -0.310 | [-0.529, -0.104] |
| CH_X|s2-off|k4|exact | +0.000 | [+0.000, +0.000] |
| CH_X|s2-off|k4|regret | -0.338 | [-0.583, -0.092] |
| CH_X|supa-off|k4|exact | +0.000 | [+0.000, +0.000] |
| CH_X|supa-off|k4|regret | +0.142 | [-0.014, +0.325] |

Inputs: {"code_sha256": {"scripts/p3_analysis.py": "24b50b9ccadd5fe89808d2ac89eb03cd0174dca4e8154421bd734d8d626a70e9", "scripts/p3_common.py": "737d7984f118c248f8169ad55806a2f5852f5e0b06bf046868a15fa8e764c648", "scripts/p3_run.py": "88ab385cbcca9435317b1502c02da4ffc6c7dc8ea68486b40cf4eaa474e0bbbc", "scripts/confirm2_analysis.py": "1c158a4e72bab2e8638ff03b526fa56a1d1b0dc75c266f1ea8ef820f685f7b1a", "rcwmt/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "rcwmt/bank.py": "b08b0229c566f9f5b50c7c0c5738f7fff90b768a827e7e32c9880ff69352592f", "rcwmt/capture.py": "7b0a19ee4aefda183f22336521fa39d5d3694bf1d60b5b950d769fa942ce255a", "rcwmt/controls.py": "08a9e4479fe621d5bb2d2a6e9d970eef122ae6850e7b894206b6a7c57002a702", "rcwmt/data.py": "f77270ee04d64985e5b6fa9a88f989a04b0b7ed8cc75f69a4984ca7e7593beb4", "rcwmt/decide.py": "148e98ab528dcd283f6896f07f01135ede658e124d09d08d293e7d3645463643", "rcwmt/decisions.py": "bd6a4531622d2479f1b7fe35fcb11338d831c1ab2e9402b515f9a35ef5237188", "rcwmt/env.py": "92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2", "rcwmt/fidelity.py": "da03bbca4962c05b13eb13dfa0414578344bbe2674015dc93070465ca4e1bb0e", "rcwmt/generation.py": "a4052088a3b6e4d1bc0b59c947599ebce76cdd0bb37230761cb817fa92df7b19", "rcwmt/heads.py": "b3bbb95696fc40fa69cb1d3f60e2547f2fde8305862af2e5b934b073170d58b6", "rcwmt/lease_vendored.py": "d1576bb9296c6bd033dd677c0f8cbe18c9bc24939f75bba30d740429b2ec36c9", "rcwmt/offtask.py": "5fcc70500d21b9043ff93a054b53bdcf29fb2e052eeb722a99fb388fdd41ba0e", "rcwmt/prereg1.py": "1068d66dab6e1c87a255440eae376c1a884ac7dfc3cedac85478e30775cacd35", "rcwmt/pvr.py": "f7b1717fb276252900e75c3a6bf2f835d772498f9b75762aa82faea799e94b94", "rcwmt/runtime.py": "9360541286dc5648e6fd71c217135914b80fd4cef665364c47688cb07f0a12ca"}, "numpy": "2.4.6", "torch_imported": false, "seconds": 1.0}


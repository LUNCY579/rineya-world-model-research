# PREREG-3 DRY RUN on synthetic fixtures (scenario 'rule_learned'; NOT results)

Rendered mechanically by `scripts/p3_analysis.py` from the results JSON. Bank sha256 0fd18295d6ea...

## Verdict: STATUS: RULE (learned by the adapter)

> In all three T2 seeds, Titan-Sparse with the adapter left unseen-status events unchanged on 0.95-0.97 of rows in every seed and stratum (each >= 0.80 with Wilson lower bound >= 0.70): five words that mean 'not applied' (pending, draft, cancelled, reversed, held), neutral and nonce words without that meaning (approved, active, ok, dax, wug), near-miss spellings of 'void' (Void, VOID, voided) and look-alikes of 'posted' (unposted, not_posted, 'not posted'), as the frame's rule states. It applied the same events when posted (twin accuracy s0 0.92, s1 0.92, s2 0.92; twins copied s0 0.06, s1 0.06, s2 0.06). Without the adapter the same rows scored 0.10-0.15; ON > OFF in all 24 stratum x seed McNemar tests (Holm).

> Base semantics (descriptive): unseen-status no-op accuracy OFF 0.10-0.15, Supa 1.7 0.20-0.20; status-sensitive pairs (unseen row unchanged while its posted twin was changed) OFF 0.06-0.10, Supa 1.7 0.03-0.14, adapted seeds 0.89-0.92. Raw no-op accuracy cannot separate status semantics from copy bias; the paired rate can.

> MAGNITUDE (add / transfer): GENERALISES (M_bal|at: OFF 0.40, s0 0.93, s1 0.93, s2 0.93; M_amt|at: OFF 0.40, s0 0.93, s1 0.93, s2 0.93; the same bases in distribution, T_posted|at: OFF 0.51, s0 0.93, s1 0.93, s2 0.93; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 +0.10 [-0.03, +0.25]; s1 +0.10 [-0.03, +0.24]; s2 +0.10 [-0.05, +0.27]).
> MAGNITUDE (scale: multi-digit multiplication): GENERALISES (M_bal+M_amt|sc: OFF 0.39, s0 0.91, s1 0.91, s2 0.91; the same bases in distribution, T_posted|sc: OFF 0.48, s0 0.91, s1 0.91, s2 0.91; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 +0.09 [-0.14, +0.35]; s1 +0.09 [-0.12, +0.32]; s2 +0.09 [-0.15, +0.33]).
> NAMING: GENERALISES (A_renamed|all: OFF 0.40, s0 0.92, s1 0.92, s2 0.92; A_many|all: OFF 0.40, s0 0.92, s1 0.92, s2 0.92; the same bases in distribution, T_posted|all: OFF 0.50, s0 0.92, s1 0.92, s2 0.92; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 +0.10 [-0.03, +0.23]; s1 +0.10 [-0.04, +0.23]; s2 +0.10 [-0.04, +0.24]).
> FORMAT: GENERALISES (F_keyorder|all: OFF 0.40, s0 0.92, s1 0.92, s2 0.92; the same bases in distribution, T_posted|all: OFF 0.50, s0 0.92, s1 0.92, s2 0.92; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 +0.10 [-0.06, +0.25]; s1 +0.10 [-0.05, +0.25]; s2 +0.10 [-0.07, +0.27]).

> For PREREG-2: within the tested status vocabulary (16 non-posted strings in eight strata) the adapter's status handling follows the stated rule, not the token 'void'; PREREG-2's limit 'fresh at payload level, not out-of-distribution' can be narrowed for the status rule only, as far as the magnitude / naming / format verdicts allow.

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
| T_posted | status | apply | 0.50 [0.40, 0.60] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.20 [0.13, 0.29] | 0.00 [0.00, 0.04] |
| C_void | status | noop | 0.08 [0.04, 0.15] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| U_pending | status | noop | 0.15 [0.09, 0.23] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| U_draft | status | noop | 0.15 [0.09, 0.23] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| U_cancelled | status | noop | 0.15 [0.09, 0.23] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| U_reversed | status | noop | 0.15 [0.09, 0.23] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| U_held | status | noop | 0.15 [0.09, 0.23] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| U_neutral | status | noop | 0.15 [0.09, 0.23] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.97 [0.92, 0.99] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| N_voidcase | status | noop | 0.10 [0.06, 0.17] | 0.95 [0.89, 0.98] | 0.95 [0.89, 0.98] | 0.95 [0.89, 0.98] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| N_postedlike | status | noop | 0.10 [0.06, 0.17] | 0.95 [0.89, 0.98] | 0.95 [0.89, 0.98] | 0.95 [0.89, 0.98] | 0.20 [0.13, 0.29] | 1.00 [0.96, 1.00] |
| M_bal | magnitude | apply | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| M_amt | magnitude | apply | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| A_renamed | naming | apply | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| A_many | naming | apply | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| F_keyorder | format | apply | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |

## V1 status: per seed

| seed | unseen strata meet the bar | failing unseen strata | twin acc | twins copied | twin guard | trigger-set <= 0.30 | C_void | errors applied / wrong (trigger set) |
|---|---|---|---|---|---|---|---|---|
| s0 | True | - | 0.92 | 0.06 | True | False | 1.00 | 16 / 23 |
| s1 | True | - | 0.92 | 0.06 | True | False | 1.00 | 16 / 23 |
| s2 | True | - | 0.92 | 0.06 | True | False | 1.00 | 16 / 23 |

Verdict RULE; attribution learned; PARTIAL patterns -; adapter gain Holm-significant in all 24 cells: True (in any: True); OFF meets the bar on every unseen status: False (OFF twins copied 0.35). Expectations: E1 C_void ON >= 0.95 in every seed: holds; E2 C_void OFF <= 0.30: holds

## McNemar ON vs OFF (exact, two-sided; Holm within each family: status 24, gen 18)

| family | cell | ON only | OFF only | p | Holm p | significant | direction |
|---|---|---|---|---|---|---|---|
| status | U_pending|s0 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_pending|s1 | 83 | 1 | 8.789e-24 | 6.152e-23 | True | on>off |
| status | U_pending|s2 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_draft|s0 | 83 | 1 | 8.789e-24 | 6.152e-23 | True | on>off |
| status | U_draft|s1 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_draft|s2 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_cancelled|s0 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_cancelled|s1 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_cancelled|s2 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_reversed|s0 | 83 | 1 | 8.789e-24 | 6.152e-23 | True | on>off |
| status | U_reversed|s1 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_reversed|s2 | 84 | 2 | 9.673e-23 | 9.673e-23 | True | on>off |
| status | U_held|s0 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_held|s1 | 83 | 1 | 8.789e-24 | 6.152e-23 | True | on>off |
| status | U_held|s2 | 83 | 1 | 8.789e-24 | 6.152e-23 | True | on>off |
| status | U_neutral|s0 | 83 | 1 | 8.789e-24 | 6.152e-23 | True | on>off |
| status | U_neutral|s1 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | U_neutral|s2 | 82 | 0 | 4.136e-25 | 8.685e-24 | True | on>off |
| status | N_voidcase|s0 | 85 | 0 | 5.17e-26 | 1.241e-24 | True | on>off |
| status | N_voidcase|s1 | 85 | 0 | 5.17e-26 | 1.241e-24 | True | on>off |
| status | N_voidcase|s2 | 85 | 0 | 5.17e-26 | 1.241e-24 | True | on>off |
| status | N_postedlike|s0 | 86 | 1 | 1.137e-24 | 1.137e-23 | True | on>off |
| status | N_postedlike|s1 | 86 | 1 | 1.137e-24 | 1.137e-23 | True | on>off |
| status | N_postedlike|s2 | 86 | 1 | 1.137e-24 | 1.137e-23 | True | on>off |
| gen | M_bal|at|s0 | 37 | 2 | 2.841e-09 | 1.421e-08 | True | on>off |
| gen | M_bal|at|s1 | 37 | 2 | 2.841e-09 | 1.421e-08 | True | on>off |
| gen | M_bal|at|s2 | 38 | 3 | 1.048e-08 | 3.144e-08 | True | on>off |
| gen | M_amt|at|s0 | 36 | 1 | 5.53e-10 | 4.424e-09 | True | on>off |
| gen | M_amt|at|s1 | 36 | 1 | 5.53e-10 | 4.424e-09 | True | on>off |
| gen | M_amt|at|s2 | 36 | 1 | 5.53e-10 | 4.424e-09 | True | on>off |
| gen | M_bal+M_amt|sc|s0 | 38 | 4 | 5.653e-08 | 5.653e-08 | True | on>off |
| gen | M_bal+M_amt|sc|s1 | 37 | 3 | 1.947e-08 | 3.893e-08 | True | on>off |
| gen | M_bal+M_amt|sc|s2 | 34 | 0 | 1.164e-10 | 1.048e-09 | True | on>off |
| gen | A_renamed|all|s0 | 56 | 4 | 9.085e-13 | 1.272e-11 | True | on>off |
| gen | A_renamed|all|s1 | 53 | 1 | 6.106e-15 | 1.099e-13 | True | on>off |
| gen | A_renamed|all|s2 | 56 | 4 | 9.085e-13 | 1.272e-11 | True | on>off |
| gen | A_many|all|s0 | 56 | 4 | 9.085e-13 | 1.272e-11 | True | on>off |
| gen | A_many|all|s1 | 56 | 4 | 9.085e-13 | 1.272e-11 | True | on>off |
| gen | A_many|all|s2 | 53 | 1 | 6.106e-15 | 1.099e-13 | True | on>off |
| gen | F_keyorder|all|s0 | 55 | 3 | 2.26e-13 | 3.616e-12 | True | on>off |
| gen | F_keyorder|all|s1 | 55 | 3 | 2.26e-13 | 3.616e-12 | True | on>off |
| gen | F_keyorder|all|s2 | 56 | 4 | 9.085e-13 | 1.272e-11 | True | on>off |

## Magnitude / naming / format cells

| group | verdict | cell classes |
|---|---|---|
| magnitude_at | GENERALISES | M_bal|at|s0 HIGH_GAIN, M_bal|at|s1 HIGH_GAIN, M_bal|at|s2 HIGH_GAIN, M_amt|at|s0 HIGH_GAIN, M_amt|at|s1 HIGH_GAIN, M_amt|at|s2 HIGH_GAIN |
| magnitude_scale | GENERALISES | M_bal+M_amt|sc|s0 HIGH_GAIN, M_bal+M_amt|sc|s1 HIGH_GAIN, M_bal+M_amt|sc|s2 HIGH_GAIN |
| naming | GENERALISES | A_renamed|all|s0 HIGH_GAIN, A_renamed|all|s1 HIGH_GAIN, A_renamed|all|s2 HIGH_GAIN, A_many|all|s0 HIGH_GAIN, A_many|all|s1 HIGH_GAIN, A_many|all|s2 HIGH_GAIN |
| format | GENERALISES | F_keyorder|all|s0 HIGH_GAIN, F_keyorder|all|s1 HIGH_GAIN, F_keyorder|all|s2 HIGH_GAIN |

### Cell accuracies (kind classes: at = add + transfer, sc = scale)

| cell | OFF (Titan-Sparse) | T2 s0 | T2 s1 | T2 s2 | Supa 1.7 | copy floor |
|---|---|---|---|---|---|---|
| T_posted|all | 0.50 [0.40, 0.60] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.20 [0.13, 0.29] | 0.00 [0.00, 0.04] |
| T_posted|at | 0.51 [0.39, 0.62] | 0.93 [0.84, 0.97] | 0.93 [0.84, 0.97] | 0.93 [0.84, 0.97] | 0.19 [0.12, 0.30] | 0.00 [0.00, 0.05] |
| T_posted|sc | 0.48 [0.33, 0.65] | 0.91 [0.76, 0.97] | 0.91 [0.76, 0.97] | 0.91 [0.76, 0.97] | 0.21 [0.11, 0.38] | 0.00 [0.00, 0.10] |
| M_bal|at | 0.40 [0.29, 0.52] | 0.93 [0.84, 0.97] | 0.93 [0.84, 0.97] | 0.93 [0.84, 0.97] | 0.10 [0.05, 0.20] | 0.00 [0.00, 0.05] |
| M_amt|at | 0.40 [0.29, 0.52] | 0.93 [0.84, 0.97] | 0.93 [0.84, 0.97] | 0.93 [0.84, 0.97] | 0.10 [0.05, 0.20] | 0.00 [0.00, 0.05] |
| M_bal+M_amt|sc | 0.39 [0.29, 0.51] | 0.91 [0.82, 0.96] | 0.91 [0.82, 0.96] | 0.91 [0.82, 0.96] | 0.09 [0.04, 0.18] | 0.00 [0.00, 0.06] |
| A_renamed|all | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| A_many|all | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| F_keyorder|all | 0.40 [0.31, 0.50] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.92 [0.85, 0.96] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.04] |
| M_bal|sc | 0.39 [0.25, 0.56] | 0.91 [0.76, 0.97] | 0.91 [0.76, 0.97] | 0.91 [0.76, 0.97] | 0.09 [0.03, 0.24] | 0.00 [0.00, 0.10] |
| M_amt|sc | 0.39 [0.25, 0.56] | 0.91 [0.76, 0.97] | 0.91 [0.76, 0.97] | 0.91 [0.76, 0.97] | 0.09 [0.03, 0.24] | 0.00 [0.00, 0.10] |

### In distribution vs out of range within each arm (paired by base; exact McNemar, descriptive)

| arm | cell | acc out | acc in (twins) | out only | in only | p |
|---|---|---|---|---|---|---|
| OFF (Titan-Sparse) | M_bal|at | 0.40 | 0.51 | 11 | 18 | 0.265 |
| OFF (Titan-Sparse) | M_bal|sc | 0.39 | 0.48 | 7 | 10 | 0.629 |
| OFF (Titan-Sparse) | M_amt|at | 0.40 | 0.51 | 11 | 18 | 0.265 |
| OFF (Titan-Sparse) | M_amt|sc | 0.39 | 0.48 | 6 | 9 | 0.607 |
| OFF (Titan-Sparse) | A_renamed|all | 0.40 | 0.50 | 16 | 26 | 0.164 |
| OFF (Titan-Sparse) | A_many|all | 0.40 | 0.50 | 23 | 33 | 0.229 |
| OFF (Titan-Sparse) | F_keyorder|all | 0.40 | 0.50 | 21 | 31 | 0.212 |
| T2 s0 | M_bal|at | 0.93 | 0.93 | 4 | 4 | 1 |
| T2 s0 | M_bal|sc | 0.91 | 0.91 | 3 | 3 | 1 |
| T2 s0 | M_amt|at | 0.93 | 0.93 | 5 | 5 | 1 |
| T2 s0 | M_amt|sc | 0.91 | 0.91 | 2 | 2 | 1 |
| T2 s0 | A_renamed|all | 0.92 | 0.92 | 7 | 7 | 1 |
| T2 s0 | A_many|all | 0.92 | 0.92 | 5 | 5 | 1 |
| T2 s0 | F_keyorder|all | 0.92 | 0.92 | 7 | 7 | 1 |
| T2 s1 | M_bal|at | 0.93 | 0.93 | 4 | 4 | 1 |
| T2 s1 | M_bal|sc | 0.91 | 0.91 | 3 | 3 | 1 |
| T2 s1 | M_amt|at | 0.93 | 0.93 | 5 | 5 | 1 |
| T2 s1 | M_amt|sc | 0.91 | 0.91 | 3 | 3 | 1 |
| T2 s1 | A_renamed|all | 0.92 | 0.92 | 8 | 8 | 1 |
| T2 s1 | A_many|all | 0.92 | 0.92 | 8 | 8 | 1 |
| T2 s1 | F_keyorder|all | 0.92 | 0.92 | 7 | 7 | 1 |
| T2 s2 | M_bal|at | 0.93 | 0.93 | 5 | 5 | 1 |
| T2 s2 | M_bal|sc | 0.91 | 0.91 | 2 | 2 | 1 |
| T2 s2 | M_amt|at | 0.93 | 0.93 | 5 | 5 | 1 |
| T2 s2 | M_amt|sc | 0.91 | 0.91 | 3 | 3 | 1 |
| T2 s2 | A_renamed|all | 0.92 | 0.92 | 8 | 8 | 1 |
| T2 s2 | A_many|all | 0.92 | 0.92 | 8 | 8 | 1 |
| T2 s2 | F_keyorder|all | 0.92 | 0.92 | 8 | 8 | 1 |
| Supa 1.7 | M_bal|at | 0.10 | 0.19 | 6 | 12 | 0.238 |
| Supa 1.7 | M_bal|sc | 0.09 | 0.21 | 2 | 6 | 0.289 |
| Supa 1.7 | M_amt|at | 0.10 | 0.19 | 7 | 13 | 0.263 |
| Supa 1.7 | M_amt|sc | 0.09 | 0.21 | 2 | 6 | 0.289 |
| Supa 1.7 | A_renamed|all | 0.10 | 0.20 | 7 | 17 | 0.0639 |
| Supa 1.7 | A_many|all | 0.10 | 0.20 | 8 | 18 | 0.0755 |
| Supa 1.7 | F_keyorder|all | 0.10 | 0.20 | 8 | 18 | 0.0755 |
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
| magnitude_at | s0 | +0.522 | +0.418 | +0.104 | [-0.030, +0.246] |
| magnitude_at | s1 | +0.522 | +0.418 | +0.104 | [-0.030, +0.239] |
| magnitude_at | s2 | +0.522 | +0.418 | +0.104 | [-0.052, +0.269] |
| magnitude_at | supa | -0.299 | -0.313 | +0.015 | [-0.142, +0.164] |
| magnitude_scale | s0 | +0.515 | +0.424 | +0.091 | [-0.136, +0.348] |
| magnitude_scale | s1 | +0.515 | +0.424 | +0.091 | [-0.121, +0.318] |
| magnitude_scale | s2 | +0.515 | +0.424 | +0.091 | [-0.152, +0.333] |
| magnitude_scale | supa | -0.303 | -0.273 | -0.030 | [-0.273, +0.197] |
| naming | s0 | +0.520 | +0.420 | +0.100 | [-0.030, +0.230] |
| naming | s1 | +0.520 | +0.420 | +0.100 | [-0.040, +0.235] |
| naming | s2 | +0.520 | +0.420 | +0.100 | [-0.040, +0.240] |
| naming | supa | -0.300 | -0.300 | +0.000 | [-0.140, +0.140] |
| format | s0 | +0.520 | +0.420 | +0.100 | [-0.060, +0.250] |
| format | s1 | +0.520 | +0.420 | +0.100 | [-0.050, +0.250] |
| format | s2 | +0.520 | +0.420 | +0.100 | [-0.070, +0.270] |
| format | supa | -0.300 | -0.300 | +0.000 | [-0.170, +0.170] |

## Pairs and status sensitivity (descriptive)

Both correct: the unseen row and its posted twin. Sensitive: the unseen row unchanged while the twin was changed (correct or wrong value, not copied).

| stratum | OFF (Titan-Sparse) both / sensitive | T2 s0 both / sensitive | T2 s1 both / sensitive | T2 s2 both / sensitive | Supa 1.7 both / sensitive | copy floor both / sensitive |
|---|---|---|---|---|---|---|
| U_pending | 0.09 / 0.10 | 0.89 / 0.91 | 0.89 / 0.91 | 0.89 / 0.91 | 0.05 / 0.08 | 0.00 / 0.00 |
| U_draft | 0.07 / 0.09 | 0.89 / 0.91 | 0.89 / 0.91 | 0.90 / 0.91 | 0.04 / 0.06 | 0.00 / 0.00 |
| U_cancelled | 0.07 / 0.10 | 0.89 / 0.91 | 0.89 / 0.91 | 0.89 / 0.91 | 0.02 / 0.06 | 0.00 / 0.00 |
| U_reversed | 0.08 / 0.09 | 0.89 / 0.91 | 0.89 / 0.91 | 0.89 / 0.91 | 0.06 / 0.09 | 0.00 / 0.00 |
| U_held | 0.09 / 0.10 | 0.90 / 0.92 | 0.89 / 0.91 | 0.89 / 0.91 | 0.09 / 0.14 | 0.00 / 0.00 |
| U_neutral | 0.03 / 0.07 | 0.89 / 0.91 | 0.89 / 0.91 | 0.90 / 0.92 | 0.04 / 0.04 | 0.00 / 0.00 |
| N_voidcase | 0.06 / 0.07 | 0.87 / 0.89 | 0.87 / 0.89 | 0.87 / 0.89 | 0.05 / 0.10 | 0.00 / 0.00 |
| N_postedlike | 0.05 / 0.06 | 0.89 / 0.90 | 0.87 / 0.89 | 0.87 / 0.89 | 0.01 / 0.03 | 0.00 / 0.00 |
| C_void | 0.03 / 0.04 | 0.92 / 0.94 | 0.92 / 0.94 | 0.92 / 0.94 | 0.03 / 0.07 | 0.00 / 0.00 |

Copied rate over the magnitude / naming / format rows: off 0.42, s0 0.06, s1 0.06, s2 0.06, supa 0.63, copy 1.00

## Error classes (descriptive)

| stratum | OFF (Titan-Sparse) | T2 s0 | T2 s1 | T2 s2 | Supa 1.7 | copy floor |
|---|---|---|---|---|---|---|
| T_posted | copied 35, unparsed 5, wrong_value 10 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 56, unparsed 8, wrong_value 16 | copied 100 |
| C_void | applied_as_posted 64, other_change 19, unparsed 9 | - | - | - | applied_as_posted 56, other_change 16, unparsed 8 | - |
| U_pending | applied_as_posted 59, other_change 17, unparsed 9 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 56, other_change 16, unparsed 8 | - |
| U_draft | applied_as_posted 59, other_change 17, unparsed 9 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 56, other_change 16, unparsed 8 | - |
| U_cancelled | applied_as_posted 59, other_change 17, unparsed 9 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 56, other_change 16, unparsed 8 | - |
| U_reversed | applied_as_posted 59, other_change 17, unparsed 9 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 56, other_change 16, unparsed 8 | - |
| U_held | applied_as_posted 59, other_change 17, unparsed 9 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 2, other_change 1 | applied_as_posted 56, other_change 16, unparsed 8 | - |
| U_neutral | applied_as_posted 60, other_change 15, unparsed 10 | applied_as_posted 3 | applied_as_posted 3 | applied_as_posted 3 | applied_as_posted 55, other_change 15, unparsed 10 | - |
| N_voidcase | applied_as_posted 63, other_change 18, unparsed 9 | applied_as_posted 3, other_change 2 | applied_as_posted 3, other_change 2 | applied_as_posted 3, other_change 2 | applied_as_posted 56, other_change 15, unparsed 9 | - |
| N_postedlike | applied_as_posted 63, other_change 18, unparsed 9 | applied_as_posted 3, other_change 2 | applied_as_posted 3, other_change 2 | applied_as_posted 3, other_change 2 | applied_as_posted 56, other_change 15, unparsed 9 | - |
| M_bal | copied 42, unparsed 6, wrong_value 12 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 63, unparsed 9, wrong_value 18 | copied 100 |
| M_amt | copied 42, unparsed 6, wrong_value 12 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 63, unparsed 9, wrong_value 18 | copied 100 |
| A_renamed | copied 42, unparsed 6, wrong_value 12 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 63, unparsed 9, wrong_value 18 | copied 100 |
| A_many | copied 42, unparsed 6, wrong_value 12 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 63, unparsed 9, wrong_value 18 | copied 100 |
| F_keyorder | copied 42, unparsed 6, wrong_value 12 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 6, wrong_value 2 | copied 63, unparsed 9, wrong_value 18 | copied 100 |

## Breakdowns (descriptive; registered sub-variants)

- U_neutral: off: active 0.15, approved 0.15, dax 0.15, ok 0.15, wug 0.15; s0: active 1.00, approved 1.00, dax 0.95, ok 0.95, wug 0.95; s1: active 1.00, approved 1.00, dax 0.95, ok 0.95, wug 0.95; s2: active 1.00, approved 1.00, dax 0.95, ok 0.95, wug 0.95; supa: active 0.20, approved 0.20, dax 0.20, ok 0.20, wug 0.20; copy: active 1.00, approved 1.00, dax 1.00, ok 1.00, wug 1.00
- N_voidcase: off: VOID 0.09, Void 0.12, voided 0.09; s0: VOID 0.97, Void 0.94, voided 0.94; s1: VOID 0.97, Void 0.94, voided 0.94; s2: VOID 0.97, Void 0.94, voided 0.94; supa: VOID 0.21, Void 0.21, voided 0.18; copy: VOID 1.00, Void 1.00, voided 1.00
- N_postedlike: off: not posted 0.09, not_posted 0.09, unposted 0.12; s0: not posted 0.97, not_posted 0.94, unposted 0.94; s1: not posted 0.97, not_posted 0.94, unposted 0.94; s2: not posted 0.97, not_posted 0.94, unposted 0.94; supa: not posted 0.21, not_posted 0.18, unposted 0.21; copy: not posted 1.00, not_posted 1.00, unposted 1.00
- M_amt: off: 100..500 0.38, 13..99 0.42; s0: 100..500 0.92, 13..99 0.92; s1: 100..500 0.92, 13..99 0.92; s2: 100..500 0.94, 13..99 0.90; supa: 100..500 0.04, 13..99 0.16; copy: 100..500 0.00, 13..99 0.00
- A_renamed: off: letters 0.44, words 0.36; s0: letters 0.88, words 0.96; s1: letters 0.94, words 0.90; s2: letters 0.90, words 0.94; supa: letters 0.08, words 0.12; copy: letters 0.00, words 0.00
- A_many: off: 5 0.28, 6 0.52, 7 0.40, 8 0.40; s0: 5 0.88, 6 0.92, 7 0.92, 8 0.96; s1: 5 0.92, 6 1.00, 7 0.92, 8 0.84; s2: 5 0.96, 6 0.88, 7 0.96, 8 0.88; supa: 5 0.16, 6 0.08, 7 0.12, 8 0.04; copy: 5 0.00, 6 0.00, 7 0.00, 8 0.00
- F_keyorder: off: all_orders 0.38, top_only 0.42; s0: all_orders 0.92, top_only 0.92; s1: all_orders 0.90, top_only 0.94; s2: all_orders 0.92, top_only 0.92; supa: all_orders 0.06, top_only 0.14; copy: all_orders 0.00, top_only 0.00

## Chains (descriptive)

K = 4 candidates per parent (PREREG-2: 8), so normalised regret is not comparable to PREREG-2's; phi clips balances at +-300, so regret is partly blind to large-magnitude errors (exact@k is the primary chain measure for CH_X); 10 parents per stratum, so the bootstrap intervals are indicative only.

### CH_S (10 parents x 4 candidates): accounts A-D, balances and amounts in range, sorted keys; each event posted (p .50), an unseen status word (p .35, 10 equally: 5 not-applied, 5 neutral / nonce) or a near-miss / look-alike (p .15, 6 equally); >= 1 posted and >= 1 non-posted event

Gold states beyond the phi clip (|balance| > 300): k1 0 of 40, k2 0 of 40, k4 0 of 40

| arm | exact k1 | exact k2 | exact k4 | regret k1 | regret k2 | regret k4 |
|---|---|---|---|---|---|---|
| OFF (Titan-Sparse) | 0.400 | 0.225 | 0.050 | 0.478 | 0.489 | 0.629 |
| T2 s0 | 0.950 | 0.900 | 0.825 | 0.071 | 0.136 | 0.112 |
| T2 s1 | 1.000 | 1.000 | 0.900 | 0.000 | 0.000 | 0.051 |
| T2 s2 | 1.000 | 0.950 | 0.875 | 0.000 | 0.078 | 0.113 |
| Supa 1.7 | 0.250 | 0.000 | 0.000 | 0.496 | 0.474 | 0.783 |
| copy floor | 0.475 | 0.225 | 0.000 | 1.002 | 0.981 | 0.969 |

Floors (regret): k1: F_copy_s0 1.002, F_first_only 0.000, F_ignore_status 0.412, F_last_only 0.000, F_touched 0.026; k2: F_copy_s0 0.981, F_first_only 0.439, F_ignore_status 0.409, F_last_only 0.421, F_touched 0.189; k4: F_copy_s0 0.969, F_first_only 0.667, F_ignore_status 0.585, F_last_only 0.615, F_touched 0.260; copy-chain exact floor k1 0.475, k2 0.225, k4 0.000

Step accuracy on correct-input steps: off: near_void 0.71 (n 7), neutral 0.60 (n 15), posted 0.40 (n 30), posted_like 0.43 (n 7), unseen 0.20 (n 10); s0: near_void 0.94 (n 18), neutral 0.96 (n 24), posted 0.94 (n 72), posted_like 1.00 (n 14), unseen 0.95 (n 21); s1: near_void 0.94 (n 17), neutral 0.92 (n 26), posted 0.99 (n 78), posted_like 1.00 (n 14), unseen 1.00 (n 24); s2: near_void 1.00 (n 18), neutral 1.00 (n 24), posted 0.95 (n 76), posted_like 0.93 (n 14), unseen 1.00 (n 24); supa: near_void 0.00 (n 6), neutral 0.30 (n 10), posted 0.12 (n 25), posted_like 0.67 (n 3), unseen 0.33 (n 6); copy: near_void 1.00 (n 6), neutral 1.00 (n 11), posted 0.00 (n 40), posted_like 1.00 (n 4), unseen 1.00 (n 8)

### CH_X (10 parents x 4 candidates): accounts A-D, start in range; each event posted (p .60) or non-posted (p .40; the 16 non-posted strings equally); |amount| 13..500 on each add/transfer event w.p. .40 (scale keeps a trained amount, so products do not compound); each step's keys reordered w.p. .50; >= 1 posted, >= 1 non-posted, >= 1 posted OOR-amount event and >= 1 reordered step

Gold states beyond the phi clip (|balance| > 300): k1 3 of 40, k2 7 of 40, k4 13 of 40

| arm | exact k1 | exact k2 | exact k4 | regret k1 | regret k2 | regret k4 |
|---|---|---|---|---|---|---|
| OFF (Titan-Sparse) | 0.425 | 0.225 | 0.000 | 0.475 | 0.504 | 0.613 |
| T2 s0 | 0.925 | 0.900 | 0.800 | 0.073 | 0.124 | 0.067 |
| T2 s1 | 0.975 | 0.950 | 0.875 | 0.015 | 0.018 | 0.076 |
| T2 s2 | 1.000 | 0.950 | 0.825 | 0.000 | 0.039 | 0.067 |
| Supa 1.7 | 0.125 | 0.025 | 0.000 | 0.589 | 0.536 | 0.371 |
| copy floor | 0.475 | 0.150 | 0.000 | 1.017 | 1.008 | 1.009 |

Floors (regret): k1: F_copy_s0 1.017, F_first_only 0.000, F_ignore_status 0.383, F_last_only 0.000, F_touched 0.283; k2: F_copy_s0 1.008, F_first_only 0.448, F_ignore_status 0.346, F_last_only 0.385, F_touched 0.474; k4: F_copy_s0 1.009, F_first_only 0.775, F_ignore_status 0.220, F_last_only 0.634, F_touched 0.694; copy-chain exact floor k1 0.475, k2 0.150, k4 0.000

Step accuracy on correct-input steps: off: near_void 0.50 (n 4), near_void+reordered 0.50 (n 4), neutral 0.50 (n 2), neutral+reordered 0.17 (n 6), posted 0.30 (n 10), posted+reordered 0.57 (n 7), posted_like 1.00 (n 3), posted_like+reordered 0.00 (n 2), posted_oor_amount 0.36 (n 11), posted_oor_amount+reordered 0.40 (n 10), unseen 0.20 (n 5), unseen+reordered 0.83 (n 6); s0: near_void 0.83 (n 6), near_void+reordered 1.00 (n 5), neutral 1.00 (n 4), neutral+reordered 1.00 (n 14), posted 0.95 (n 22), posted+reordered 0.94 (n 18), posted_like 0.88 (n 8), posted_like+reordered 0.75 (n 4), posted_oor_amount 1.00 (n 23), posted_oor_amount+reordered 1.00 (n 25), unseen 1.00 (n 8), unseen+reordered 0.75 (n 12); s1: near_void 1.00 (n 7), near_void+reordered 0.83 (n 6), neutral 1.00 (n 5), neutral+reordered 1.00 (n 14), posted 1.00 (n 23), posted+reordered 0.95 (n 19), posted_like 0.83 (n 6), posted_like+reordered 1.00 (n 4), posted_oor_amount 0.96 (n 24), posted_oor_amount+reordered 0.96 (n 26), unseen 1.00 (n 7), unseen+reordered 1.00 (n 11); s2: near_void 1.00 (n 7), near_void+reordered 1.00 (n 6), neutral 0.83 (n 6), neutral+reordered 0.92 (n 13), posted 0.96 (n 23), posted+reordered 1.00 (n 17), posted_like 1.00 (n 8), posted_like+reordered 1.00 (n 4), posted_oor_amount 1.00 (n 23), posted_oor_amount+reordered 0.89 (n 27), unseen 1.00 (n 8), unseen+reordered 0.92 (n 12); supa: near_void 0.00 (n 3), near_void+reordered 0.00 (n 2), neutral 0.00 (n 1), neutral+reordered 0.25 (n 4), posted 0.00 (n 2), posted+reordered 0.00 (n 7), posted_like 0.50 (n 2), posted_like+reordered 0.50 (n 2), posted_oor_amount 0.12 (n 8), posted_oor_amount+reordered 0.00 (n 7), unseen 0.33 (n 3), unseen+reordered 0.33 (n 6); copy: near_void 1.00 (n 3), near_void+reordered 1.00 (n 2), neutral 1.00 (n 2), neutral+reordered 1.00 (n 3), posted 0.00 (n 8), posted+reordered 0.00 (n 8), posted_like 1.00 (n 3), posted_like+reordered 1.00 (n 2), posted_oor_amount 0.00 (n 13), posted_oor_amount+reordered 0.00 (n 11), unseen 1.00 (n 4), unseen+reordered 1.00 (n 7)

| chain contrast (k4; 10 parents, indicative) | point | 95% CI |
|---|---|---|
| CH_S|s0-off|k4|exact | +0.775 | [+0.625, +0.900] |
| CH_S|s0-off|k4|regret | -0.516 | [-0.745, -0.291] |
| CH_S|s1-off|k4|exact | +0.850 | [+0.725, +0.975] |
| CH_S|s1-off|k4|regret | -0.578 | [-0.795, -0.373] |
| CH_S|s2-off|k4|exact | +0.825 | [+0.650, +0.975] |
| CH_S|s2-off|k4|regret | -0.516 | [-0.774, -0.274] |
| CH_S|supa-off|k4|exact | -0.050 | [-0.125, +0.000] |
| CH_S|supa-off|k4|regret | +0.154 | [+0.060, +0.253] |
| CH_X|s0-off|k4|exact | +0.800 | [+0.650, +0.925] |
| CH_X|s0-off|k4|regret | -0.546 | [-0.717, -0.362] |
| CH_X|s1-off|k4|exact | +0.875 | [+0.800, +0.950] |
| CH_X|s1-off|k4|regret | -0.537 | [-0.683, -0.401] |
| CH_X|s2-off|k4|exact | +0.825 | [+0.650, +0.975] |
| CH_X|s2-off|k4|regret | -0.546 | [-0.726, -0.361] |
| CH_X|supa-off|k4|exact | +0.000 | [+0.000, +0.000] |
| CH_X|supa-off|k4|regret | -0.242 | [-0.391, -0.096] |


## All dry-run scenarios (synthetic; expected = designed verdicts)

| scenario | expected | got | ok |
|---|---|---|---|
| rule_learned | STATUS: RULE (learned by the adapter); []; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | STATUS: RULE (learned by the adapter); []; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | True |
| trigger | STATUS: TRIGGER; []; {'magnitude_at': 'DOES NOT GENERALISE', 'magnitude_scale': 'DOES NOT GENERALISE', 'naming': 'ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN', 'format': 'DOES NOT GENERALISE, with a partial gain over OFF'} | STATUS: TRIGGER; []; {'magnitude_at': 'DOES NOT GENERALISE', 'magnitude_scale': 'DOES NOT GENERALISE', 'naming': 'ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN', 'format': 'DOES NOT GENERALISE, with a partial gain over OFF'} | True |
| partial_seeds_disagree | STATUS: PARTIAL; []; {'magnitude_at': 'HARMED', 'magnitude_scale': 'DOES NOT GENERALISE', 'naming': 'MIXED', 'format': 'GENERALISES'} | STATUS: PARTIAL; []; {'magnitude_at': 'HARMED', 'magnitude_scale': 'DOES NOT GENERALISE', 'naming': 'MIXED', 'format': 'GENERALISES'} | True |
| rule_base_already | STATUS: RULE (already present without the adapter); []; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | STATUS: RULE (already present without the adapter); []; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | True |
| rule_strengthened | STATUS: RULE (present in the base at the bar, strengthened by the adapter); []; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | STATUS: RULE (present in the base at the bar, strengthened by the adapter); []; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | True |
| twins_fail | STATUS: PARTIAL; ['twins']; {'magnitude_at': 'DOES NOT GENERALISE', 'magnitude_scale': 'DOES NOT GENERALISE', 'naming': 'DOES NOT GENERALISE', 'format': 'DOES NOT GENERALISE'} | STATUS: PARTIAL; ['twins']; {'magnitude_at': 'DOES NOT GENERALISE', 'magnitude_scale': 'DOES NOT GENERALISE', 'naming': 'DOES NOT GENERALISE', 'format': 'DOES NOT GENERALISE'} | ['twins'] |
| semantic | STATUS: PARTIAL; ['semantic']; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | STATUS: PARTIAL; ['semantic']; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | ['semantic'] |
| trigger_not_interpretable | STATUS: PARTIAL; ['trigger_not_interpretable']; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | STATUS: PARTIAL; ['trigger_not_interpretable', 'no_adapter_gain']; {'magnitude_at': 'GENERALISES', 'magnitude_scale': 'GENERALISES', 'naming': 'GENERALISES', 'format': 'GENERALISES'} | ['trigger_not_interpretable'] |

## Wording self-test (real summary, McNemar and Holm on synthetic per-row classes)

- RULE learned: ok
- RULE strengthened: ok
- RULE base already: ok
- RULE mixed: copy-biased base: ok
- RULE mixed: weak gain in one seed: ok
- TRIGGER exact token: ok
- TRIGGER with void-like spellings: ok
- PARTIAL seeds disagree: ok
- PARTIAL intermediate: ok
- PARTIAL twins: copying: ok
- PARTIAL twins: arithmetic: ok
- PARTIAL twin copy guard: ok
- PARTIAL semantic: ok
- PARTIAL posted substring: ok
- PARTIAL near-miss only: ok
- PARTIAL trigger not interpretable: C_void: ok
- PARTIAL trigger not interpretable: errors: ok
- PARTIAL no adapter gain: ok
- boundary 0.80 passes: ok
- boundary 0.79 fails: ok
- boundary 0.30 trigger: ok
- boundary 0.31 partial: ok
- boundary twins copied 0.10 passes: ok
- boundary twins copied 0.11 fails: ok
- qualifier E1: ok
- qualifier E2: ok
- qualifier sub-variant: ok
- qualifier novelty fallback: ok
- GENERALISES: ok
- ABOVE FLOOR: ok
- DOES NOT GENERALISE: shared limit: ok
- DOES NOT GENERALISE: partial gain: ok
- DOES NOT GENERALISE: OFF meets: ok
- HARMED: ok
- MIXED: ok
- boundary floor n=100 0.80: ok
- boundary floor n=100 0.79: ok
- boundary floor n=67 55: ok
- boundary floor n=67 54: ok
- boundary floor n=66 54: ok
- boundary floor n=66 53: ok

## Statistics unit tests

- mcnemar b=10 c=0: ok
- mcnemar b=9 c=1: ok
- mcnemar b=5 c=5: ok
- mcnemar b=0 c=0: ok
- mcnemar b=0 c=7: ok
- mcnemar b=15 c=0: ok
- mcnemar b=4 c=0: ok
- holm case 0: ok
- holm case 1: ok
- holm case 2: ok
- holm case 3: ok
- holm case 4: ok
- holm case 5: ok
- holm case 6: ok

## Power (independent cells (lower bounds for the joint probabilities: the strata share the 100 bases))

| true acc per unseen cell | P(cell meets bar) | P(all 24 cells) | P(RULE), twins 0.90 copied 0.03 | P(RULE), twins at the same acc, copied min(0.03, 1 - acc) |
|---|---|---|---|---|
| 0.8 | 0.5595 | 0.0 | 0.0 | 0.0 |
| 0.85 | 0.9337 | 0.1926 | 0.1921 | 0.1568 |
| 0.9 | 0.9992 | 0.9808 | 0.9779 | 0.9779 |
| 0.95 | 1.0 | 1.0 | 0.997 | 0.9994 |
| 0.98 | 1.0 | 1.0 | 0.997 | 1.0 |

| true acc (trigger set) | P(cell <= 0.30) | P(all 21 cells) |
|---|---|---|
| 0.1 | 1.0 | 1.0 |
| 0.2 | 0.9939 | 0.8802 |
| 0.25 | 0.8962 | 0.1001 |
| 0.3 | 0.5491 | 0.0 |

| group | rows per cell | cells | min correct | true acc | P(cell meets floor) | P(all cells) |
|---|---|---|---|---|---|---|
| magnitude_at | [67, 67] | 6 | [55, 55] | 0.85 | [0.8026, 0.8026] | 0.2674 |
| magnitude_at | [67, 67] | 6 | [55, 55] | 0.9 | [0.9855, 0.9855] | 0.9163 |
| magnitude_at | [67, 67] | 6 | [55, 55] | 0.95 | [1.0, 1.0] | 0.9999 |
| magnitude_scale | [66] | 3 | [54] | 0.85 | [0.8174] | 0.5462 |
| magnitude_scale | [66] | 3 | [54] | 0.9 | [0.9872] | 0.9621 |
| magnitude_scale | [66] | 3 | [54] | 0.95 | [1.0] | 0.9999 |
| naming | [100, 100] | 6 | [80, 80] | 0.85 | [0.9337, 0.9337] | 0.6625 |
| naming | [100, 100] | 6 | [80, 80] | 0.9 | [0.9992, 0.9992] | 0.9952 |
| naming | [100, 100] | 6 | [80, 80] | 0.95 | [1.0, 1.0] | 1.0 |
| format | [100] | 3 | [80] | 0.85 | [0.9337] | 0.8139 |
| format | [100] | 3 | [80] | 0.9 | [0.9992] | 0.9976 |
| format | [100] | 3 | [80] | 0.95 | [1.0] | 1.0 |
| unpooled scale cells (NOT registered; why the scale cell pools M_bal and M_amt) | [33, 33] | 6 | [29, 29] | 0.85 | [0.4355, 0.4355] | 0.0068 |
| unpooled scale cells (NOT registered; why the scale cell pools M_bal and M_amt) | [33, 33] | 6 | [29, 29] | 0.9 | [0.7697, 0.7697] | 0.2079 |

| McNemar rows | ON | OFF | worst-case Holm m | power (20,000 simulations, independent errors) |
|---|---|---|---|---|
| 100 | 0.9 | 0.6 | 24 | 0.963 |
| 100 | 0.85 | 0.7 | 24 | 0.226 |
| 100 | 0.95 | 0.8 | 24 | 0.476 |
| 100 | 0.9 | 0.6 | 18 | 0.971 |
| 100 | 0.85 | 0.7 | 18 | 0.256 |
| 67 | 0.9 | 0.6 | 18 | 0.824 |
| 66 | 0.9 | 0.6 | 18 | 0.813 |
| 66 | 0.85 | 0.6 | 18 | 0.518 |

Dry run ok: True

# PREREG-1 confirmation bank: replication of pilot findings R8 and R10

**Status: post-hoc exploratory replication on the confirmation bank; not preregistered; no tuning on these data.** Every definition, hyper-parameter, head, selection rule, seed and checkpoint rule comes from the pilot (`scripts/s_roll.py`, `s_roll_select*_{L,P,R}.json`, RESULTS_PILOT R8/R10). Confirmation data: 150 parents per family, 8 non-twin candidates each (twins excluded everywhere, as in the decision set).

Sanity (both pass): Titan chain_one k=4 regret through this path = L 0.5390 / R 0.3560 / P 0.8993 (confirmatory 0.539 / 0.356 / 0.899; every per-parent value equal to `PREREG_1_RESULTS_PER_PARENT.npz`); an oracle one-step model through the same rollout gives regret 0 and exact state 1 for every family and k. Retrained S-roll models are bit-identical to the pilot models (DEV exact, best step, pilot-TEST per-parent regret).

## (A) R8: Titan's one-step accuracy by operation (chain_one)

R8 rule = chain steps whose input state is correct; 1st step = first step from gold s0. Cells: accuracy (n); confirmation 95% parent-cluster CI. Error = most common type among wrong outputs under the R8 rule (same typology on both banks), pilot → confirmation.

| Fam | Operation | R8 rule: pilot | R8 rule: confirm [CI] | 1st step: pilot | 1st step: confirm | Most common error (R8 rule), pilot → confirm |
|---|---|---|---|---|---|---|
| L | add, posted | 0.529 (68) | 0.539 (356) [0.486, 0.592] | 0.568 (37) | 0.561 (212) | one balance wrong 27/32 → one balance wrong 149/164 |
| L | scale, posted | 0.640 (100) | 0.582 (622) [0.540, 0.622] | 0.554 (56) | 0.585 (376) | one balance wrong 36/36 → one balance wrong 258/260 |
| L | transfer, posted | 0.315 (146) | 0.332 (802) [0.296, 0.366] | 0.305 (95) | 0.345 (467) | one balance wrong 64/100 → one balance wrong 333/536 |
| L | any void event | 0.073 (41) | 0.102 (245) [0.069, 0.139] | 0.100 (20) | 0.124 (145) | void applied as posted 16/38 → void applied as posted 89/220 |
| P | applicable | 0.319 (166) | 0.451 (1127) [0.413, 0.486] | 0.368 (125) | 0.515 (748) | output = state + add 64/113 → output = state + add 442/619 |
| P | inapplicable | 0.009 (117) | 0.008 (598) [0.000, 0.020] | 0.010 (99) | 0.009 (452) | applied anyway 89/116 → applied anyway 461/593 |
| R | swap | 1.000 (133) | 0.998 (657) [0.995, 1.000] | 1.000 (67) | 0.997 (343) | — → swap wrong 1/1 |
| R | add | 0.842 (95) | 0.888 (500) [0.859, 0.916] | 0.851 (47) | 0.882 (238) | wrong value 8/15 → wrong value 44/56; missing mod 97 7/15 → 12/56 |
| R | add, no wrap | 0.920 (87) | 0.923 (481) [0.899, 0.945] | 0.952 (42) | 0.929 (226) | wrong value 7/7 → wrong value 34/37 |
| R | add, wrap | 0.000 (8) | 0.000 (19) [0.000, 0.000] | 0.000 (5) | 0.000 (12) | upward wrap (r + v ≥ 97): missing mod 7/7 → 9/9, of which exactly r + v 6 → 9; downward wrap (r + v < 0): wrong value 1/1 → 10/10 |
| R | conditional | 0.474 (95) | 0.465 (645) [0.425, 0.504] | 0.450 (40) | 0.464 (293) | wrong branch 40/50 → wrong branch 260/345 |
| R | mix | 0.008 (126) | 0.011 (699) [0.004, 0.020] | 0.019 (54) | 0.006 (326) | wrong value 84/125 → wrong value 516/691; missing mod 97 41/125 → 175/691 |

Notes. Void first steps, wrong outputs equal to the posted application: 8/18 → 51/127 (posted application is a no-op for 2 pilot and 14 confirmation void steps; those are excluded from this count). P applicable first steps are higher on confirmation (0.368 → 0.515). Post hoc and descriptive only (not a pilot definition): the confirmation bank has more steps where state + add is already the gold answer (42% → 55%); re-weighted to the pilot cell mix the confirmation accuracy is 0.394, so the rise is largely consistent with case mix, with a smaller within-cell rise (0.868 → 0.923, see (B)).

**Verdict (A): replicates, except the R add error label, which replicates only partially.** Every operation keeps its pilot accuracy profile (swap ≈ 1, mix ≈ 0, conditional 0.474 → 0.465 with mostly wrong-branch errors, void events and inapplicable actions applied anyway, posted ledger operations 0.32–0.64 → 0.33–0.58, add fails at every wrap 0/8 → 0/19); but R8's "missing mod 97" is not the most common add error on either bank (wrong value 8/15 → 44/56): the unreduced sum replicates at upward wraps, but downward wraps (rare on the pilot) give other wrong values and wraps are a smaller share of confirmation add steps (8.4% → 3.8%).

## (B) R10: Titan's prerequisite rule is integer addition (applicable first steps, gold s0)

| Quantity | Pilot | Confirm |
|---|---|---|
| Applicable first steps: right / wrong (unparseable) | 46 / 79 (2) | 385 / 363 (26) |
| Parsed wrong outputs = state + add (R10 denominator) | 50/77 (65%) | 261/337 (77%; CI 71%–83%) |
| … add bit already set (unambiguous carry); of those, nothing deleted | 32; 29 | 163; 121 |
| … add bit clear (consistent with ignoring delete) | 18 | 98 |
| Titan right where state + add = gold | 46/53 (0.868) | 383/415 (0.923, CI 0.884–0.957) |
| Titan right where state + add ≠ gold | 0/72 | 2/333 |
| Inapplicable first steps right; output = state + add | 1/99; 78 | 4/452; 351 |

**Verdict (B): replicates.** Titan is right almost only where integer addition happens to give the gold state (383/415 vs 2/333; pilot 46/53 vs 0/72), and the unambiguous-carry errors recur (32 → 163).

## (C) R10: S-roll vs Titan chain_one (k=4 normalised regret)

Cells: Titan − S-roll [95% paired parent bootstrap]; negative = Titan better. Pilot n = 26 parents, confirmation n = 150. spec / DEV-regret are the two pilot selection rules.

| Fam | Titan pilot → confirm | S-roll arm | spec: pilot | spec: confirm | DEV-regret: pilot | DEV-regret: confirm |
|---|---|---|---|---|---|---|
| L | 0.507 → 0.539 | FULL | +0.199 [+0.051, +0.345] | +0.216 [+0.156, +0.279] | +0.456 [+0.332, +0.588] | +0.457 [+0.404, +0.514] |
| L |  | MATCHED128 | −0.185 [−0.317, −0.051] | −0.167 [−0.221, −0.111] | −0.204 [−0.329, −0.075] | −0.146 [−0.201, −0.089] |
| L |  | TITANROWS (suppl.) | −0.158 [−0.305, −0.008] | −0.104 [−0.166, −0.039] | −0.199 [−0.371, −0.022] | −0.152 [−0.218, −0.085] |
| L |  | N512 (suppl.) | −0.093 [−0.213, +0.031] | −0.071 [−0.131, −0.010] | +0.053 [−0.076, +0.189] | +0.065 [+0.012, +0.121] |
| L |  | N2000 (suppl.) | +0.116 [−0.029, +0.264] | +0.099 [+0.036, +0.163] | +0.373 [+0.244, +0.509] | +0.382 [+0.326, +0.441] |
| P | 0.879 → 0.899 | FULL | +0.626 [+0.491, +0.766] | +0.579 [+0.532, +0.626] | +0.627 [+0.491, +0.766] | +0.579 [+0.530, +0.625] |
| P |  | MATCHED128 | +0.333 [+0.220, +0.450] | +0.306 [+0.267, +0.345] | +0.318 [+0.205, +0.431] | +0.288 [+0.249, +0.326] |
| P |  | TITANROWS (suppl.) | +0.127 [−0.025, +0.278] | +0.150 [+0.103, +0.198] | +0.137 [−0.025, +0.300] | +0.152 [+0.106, +0.199] |
| P |  | N512 (suppl.) | +0.599 [+0.481, +0.720] | +0.515 [+0.469, +0.560] | +0.599 [+0.477, +0.722] | +0.510 [+0.464, +0.554] |
| R | 0.386 → 0.356 | FULL | −0.239 [−0.296, −0.182] | −0.234 [−0.258, −0.209] | −0.235 [−0.284, −0.185] | −0.231 [−0.256, −0.206] |
| R |  | MATCHED128 | −0.607 [−0.650, −0.565] | −0.624 [−0.644, −0.602] | −0.391 [−0.438, −0.342] | −0.393 [−0.415, −0.371] |
| R |  | TITANROWS (suppl.) | −0.598 [−0.650, −0.549] | −0.626 [−0.646, −0.604] | −0.394 [−0.457, −0.326] | −0.389 [−0.416, −0.362] |
| R |  | N512 (suppl.) | −0.387 [−0.442, −0.330] | −0.369 [−0.392, −0.346] | −0.387 [−0.442, −0.330] | −0.369 [−0.392, −0.346] |
| R |  | N2000 (suppl.) | −0.293 [−0.348, −0.237] | −0.312 [−0.336, −0.288] | −0.287 [−0.348, −0.223] | −0.315 [−0.338, −0.292] |

All 84 pilot-vs-confirmation contrasts (k = 1, 2, 4; main and supplementary arms): 68 significant on both banks with the same sign; 10 go from a pilot CI including 0 to a confirmation CI excluding 0 in the direction of the pilot point estimate (L TITANROWS spec k2 −0.085 → −0.081; L N512 spec k4 −0.093 → −0.071; L N2000 spec k2 +0.111 → +0.128; L N2000 spec k4 +0.116 → +0.099; P TITANROWS spec k4 +0.127 → +0.150; L MATCHED128 DEV-regret k1 −0.050 → −0.098; L TITANROWS DEV-regret k1 −0.004 → −0.087; L N512 DEV-regret k2 +0.068 → +0.050; L N512 DEV-regret k4 +0.053 → +0.065; P TITANROWS DEV-regret k4 +0.137 → +0.152); 1 sign change(s) with a significant confirmation CI (L MATCHED128 spec k1 +0.002 [−0.102, +0.109] → −0.095 [−0.142, −0.047]); 0 lose significance; 5 are n.s. on both. P leave-out-overlap (pilot check; 55 confirmation parents have a transition whose learner input occurs in pilot TRAIN): without them, k=4 FULL +0.513 [+0.455, +0.572], MATCHED128 +0.261 [+0.216, +0.308], TITANROWS +0.126 [+0.072, +0.182] (spec; n = 95); 24 of the 24 P contrasts (both rules, all arms, k = 1, 2, 4) keep the sign and the CI-excludes-0 status they have on all 150 parents.

**Verdict (C): replicates.** All 12 main k=4 contrasts (FULL and MATCHED128 × spec and DEV-regret × L, P, R) keep the pilot sign with confirmation CIs excluding 0 (12/12 significant on both banks): at k=4 S-roll beats Titan on P at every training budget and on L with all TRAIN transitions, and Titan beats S-roll on R at every budget and on L at 128 rows; the pilot's two supplementary "no difference" readings (P TITANROWS, L N512 DEV-regret) become same-direction differences favouring S-roll, a gain in precision rather than a reversal.

Deviations: none forced by the data. Analysis choices are listed in `confirm_replicate.json` (`deviations`) and `confirm_replicate_fixes.json` (`deviations_added`): twins excluded from (A)/(B) too; pilot (A) values are the `s_roll.onestep_stats` recomputation of R8 (R8's own L/P counts differ by 1–5 steps); the error typology and parent-cluster CIs are new; S-roll models retrained (bit-identical to the pilot's); the wrap-direction split and the case-mix re-weighting are post hoc and descriptive.

## Files

- `scripts/confirm_replicate.py`, `scripts/confirm_replicate_fixes.py`
- `runs/bank_pilot/analysis/confirm_replicate.json` (all numbers; `tables_markdown`), `confirm_replicate_fixes.json` (fixes F1–F5), `confirm_replicate_part_{exact,regret}_{L,P,R}.{json,npz}`, logs
- Fixes after verification: F1 R add error label and wrap-direction split; F2 full contrast-status scan; F3 L void "applied as posted" counted among wrong outputs; F4 P leave-out-overlap check; F5 P share over parsed wrong outputs (R10 denominator) with CI, re-weighting labelled post hoc.

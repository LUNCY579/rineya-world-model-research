# RCX-E07b report (full)

Preregistered in `experiments/RCX-E07/PREREG_E07B.md` (frozen, sha256 verified at start; CX D-20). CPU mechanism tier, exact dynamics; a learned heuristic is an external specialist (tier B), so nothing here supports a Rineya-core claim. One primary test; everything else is descriptive.

## PRIMARY decision (computed by the prereg rule)

Scope: TEST-B n in [12, 14, 15] (75 instances). `BEST` = **A3** (`A3_l1`), chosen on VAL-B; A3's λ = 1.0. Paired tests use seed 1 of each arm (A0 = `A0_s20261276`).

| quantity | value |
|---|---|
| BEST success@2000 | 55/75 |
| misplaced_blocks success@2000 | 54/75 |
| A0 success@2000 | 35/75 |
| BEST success@200 vs misplaced_blocks@200 | 39 vs 19 (misplaced-only 4, BEST-only 24, exact McNemar p = 0.00018) |
| BEST vs misplaced_blocks @2000 | misplaced-only 8, BEST-only 9, p = 1 |
| BEST vs A0 @2000 | A0-only 0, BEST-only 20, p = 1.907e-06 |
| CLOSED (a): BEST@2000 ≥ misplaced@2000 − 3 | 55 >= 54 - 3 → True |
| CLOSED (b): BEST@200 > misplaced@200, McNemar p < 0.05 | 39 > 19 and McNemar p 0.00018 < 0.05 → True |
| IMPROVED: BEST@2000 − A0@2000 ≥ +8, McNemar p < 0.05, not CLOSED | 55 - 35 = 20 >= 8 and McNemar p 1.907e-06 < 0.05 and not CLOSED → False |
| **Outcome** | **CLOSED** |

## Selection on VAL-B (λ and BEST only)

VAL-B: 30 instances (n [12, 14]). Rules: λ — max success@2000, then success@200, then smaller lambda; BEST — max success@2000, then success@200, then order A1, A2, A3; seed 1 of trained arms. Rows marked context are not candidates.

| heuristic | n=12 @2000 | n=14 @2000 | all @50 | all @200 | all @500 | all @2000 |
|---|---|---|---|---|---|---|
| A3_l0.25 | 8 | 7 | 11 | 11 | 11 | 15 |
| A3_l0.5 | 9 | 7 | 11 | 11 | 12 | 16 |
| A3_l1 | 12 | 9 | 11 | 13 | 17 | 21 |
| A1_s20261292 | 8 | 3 | 11 | 11 | 11 | 11 |
| A2_s20261295 | 9 | 7 | 14 | 14 | 16 | 16 |
| A0_s20261276 (context) | 8 | 3 | 11 | 11 | 11 | 11 |
| misplaced_blocks (context) | 11 | 10 | 2 | 8 | 14 | 21 |

Selected λ = **1.0**; BEST = **A3** (`A3_l1`).

## TEST-B — GBFS success, budget 2000

| heuristic | 10 | 12 | 14 | 15 | n 12–15 | all | rate n 12–15 [Wilson 95%] | median exp (all) | mean plan len | mean subopt (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_s20261276 | 19/25 | 15/25 | 10/25 | 10/25 | 35/75 | 54/100 | 0.467 [0.358, 0.578] | 535 | 27.0 | 1.044 (53) |
| A0_s20261277 | 17/25 | 15/25 | 8/25 | 9/25 | 32/75 | 49/100 | 0.427 [0.321, 0.539] | 2000 | 26.6 | 1.037 (48) |
| A0_s20261278 | 17/25 | 14/25 | 8/25 | 7/25 | 29/75 | 46/100 | 0.387 [0.285, 0.500] | 2000 | 25.9 | 1.029 (45) |
| A1_s20261292 | 19/25 | 14/25 | 8/25 | 9/25 | 31/75 | 50/100 | 0.413 [0.309, 0.526] | 1386 | 26.1 | 1.021 (49) |
| A1_s20261293 | 20/25 | 15/25 | 13/25 | 8/25 | 36/75 | 56/100 | 0.480 [0.371, 0.591] | 464 | 26.8 | 1.028 (55) |
| A1_s20261294 | 20/25 | 16/25 | 14/25 | 11/25 | 41/75 | 61/100 | 0.547 [0.434, 0.654] | 268 | 27.0 | 1.018 (59) |
| A2_s20261295 | 20/25 | 20/25 | 15/25 | 14/25 | 49/75 | 69/100 | 0.653 [0.541, 0.751] | 44 | 28.3 | 1.042 (66) |
| A2_s20261296 | 20/25 | 15/25 | 15/25 | 11/25 | 41/75 | 61/100 | 0.547 [0.434, 0.654] | 230 | 28.3 | 1.053 (58) |
| A2_s20261297 | 19/25 | 15/25 | 8/25 | 8/25 | 31/75 | 50/100 | 0.413 [0.309, 0.526] | 1472 | 27.0 | 1.061 (49) |
| A3_l1 | 25/25 | 21/25 | 16/25 | 18/25 | 55/75 | 80/100 | 0.733 [0.624, 0.820] | 96 | 29.0 | 1.045 (74) |
| misplaced_blocks | 21/25 | 23/25 | 19/25 | 12/25 | 54/75 | 75/100 | 0.720 [0.610, 0.809] | 483 | 27.1 | 1.007 (72) |
| h_add | 25/25 | 25/25 | 21/25 | 20/25 | 66/75 | 91/100 | 0.880 [0.787, 0.936] | 364 | 60.8 | 1.965 (77) |
| unmet_goal_facts | 16/25 | 2/25 | 0/25 | 1/25 | 3/75 | 19/100 | 0.040 [0.014, 0.111] | 2000 | 40.6 | 1.658 (17) |

## TEST-B — GBFS success, budget 500 (derived from the full-budget runs)

| heuristic | 10 | 12 | 14 | 15 | n 12–15 | all | rate n 12–15 [Wilson 95%] | median exp (all) | mean plan len | mean subopt (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_s20261276 | 18/25 | 15/25 | 9/25 | 8/25 | 32/75 | 50/100 | 0.427 [0.321, 0.539] | 484 | 26.2 | 1.041 (50) |
| A0_s20261277 | 17/25 | 13/25 | 8/25 | 8/25 | 29/75 | 46/100 | 0.387 [0.285, 0.500] | 500 | 26.2 | 1.031 (45) |
| A0_s20261278 | 17/25 | 13/25 | 7/25 | 5/25 | 25/75 | 42/100 | 0.333 [0.237, 0.446] | 500 | 25.1 | 1.025 (42) |
| A1_s20261292 | 17/25 | 14/25 | 8/25 | 8/25 | 30/75 | 47/100 | 0.400 [0.297, 0.513] | 500 | 26.0 | 1.022 (46) |
| A1_s20261293 | 18/25 | 14/25 | 10/25 | 8/25 | 32/75 | 50/100 | 0.427 [0.321, 0.539] | 421 | 26.1 | 1.025 (49) |
| A1_s20261294 | 19/25 | 14/25 | 11/25 | 9/25 | 34/75 | 53/100 | 0.453 [0.346, 0.566] | 268 | 26.1 | 1.017 (52) |
| A2_s20261295 | 20/25 | 20/25 | 15/25 | 14/25 | 49/75 | 69/100 | 0.653 [0.541, 0.751] | 44 | 28.3 | 1.042 (66) |
| A2_s20261296 | 20/25 | 15/25 | 13/25 | 8/25 | 36/75 | 56/100 | 0.480 [0.371, 0.591] | 230 | 26.9 | 1.032 (55) |
| A2_s20261297 | 17/25 | 14/25 | 8/25 | 8/25 | 30/75 | 47/100 | 0.400 [0.297, 0.513] | 500 | 26.9 | 1.052 (46) |
| A3_l1 | 19/25 | 19/25 | 15/25 | 13/25 | 47/75 | 66/100 | 0.627 [0.514, 0.727] | 96 | 28.0 | 1.041 (63) |
| misplaced_blocks | 16/25 | 15/25 | 12/25 | 8/25 | 35/75 | 51/100 | 0.467 [0.358, 0.578] | 483 | 25.0 | 1.010 (51) |
| h_add | 24/25 | 22/25 | 8/25 | 4/25 | 34/75 | 58/100 | 0.453 [0.346, 0.566] | 364 | 49.3 | 1.873 (55) |
| unmet_goal_facts | 6/25 | 1/25 | 0/25 | 0/25 | 1/75 | 7/100 | 0.013 [0.002, 0.072] | 500 | 36.3 | 1.442 (7) |

## TEST-B — GBFS success, budget 200 (derived from the full-budget runs)

| heuristic | 10 | 12 | 14 | 15 | n 12–15 | all | rate n 12–15 [Wilson 95%] | median exp (all) | mean plan len | mean subopt (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_s20261276 | 17/25 | 13/25 | 7/25 | 8/25 | 28/75 | 45/100 | 0.373 [0.273, 0.486] | 200 | 26.0 | 1.027 (45) |
| A0_s20261277 | 17/25 | 13/25 | 8/25 | 7/25 | 28/75 | 45/100 | 0.373 [0.273, 0.486] | 200 | 26.0 | 1.031 (44) |
| A0_s20261278 | 17/25 | 10/25 | 7/25 | 5/25 | 22/75 | 39/100 | 0.293 [0.202, 0.404] | 200 | 25.0 | 1.025 (39) |
| A1_s20261292 | 17/25 | 13/25 | 8/25 | 8/25 | 29/75 | 46/100 | 0.387 [0.285, 0.500] | 200 | 25.9 | 1.016 (45) |
| A1_s20261293 | 17/25 | 12/25 | 7/25 | 8/25 | 27/75 | 44/100 | 0.360 [0.261, 0.473] | 200 | 25.6 | 1.008 (43) |
| A1_s20261294 | 17/25 | 13/25 | 8/25 | 8/25 | 29/75 | 46/100 | 0.387 [0.285, 0.500] | 200 | 25.7 | 1.009 (45) |
| A2_s20261295 | 20/25 | 16/25 | 15/25 | 13/25 | 44/75 | 64/100 | 0.587 [0.474, 0.691] | 44 | 27.9 | 1.036 (61) |
| A2_s20261296 | 19/25 | 14/25 | 9/25 | 8/25 | 31/75 | 50/100 | 0.413 [0.309, 0.526] | 194 | 26.0 | 1.021 (49) |
| A2_s20261297 | 17/25 | 12/25 | 7/25 | 4/25 | 23/75 | 40/100 | 0.307 [0.214, 0.418] | 200 | 24.6 | 1.015 (40) |
| A3_l1 | 19/25 | 15/25 | 14/25 | 10/25 | 39/75 | 58/100 | 0.520 [0.409, 0.629] | 96 | 27.2 | 1.027 (56) |
| misplaced_blocks | 14/25 | 10/25 | 5/25 | 4/25 | 19/75 | 33/100 | 0.253 [0.169, 0.362] | 200 | 22.9 | 1.010 (33) |
| h_add | 17/25 | 11/25 | 2/25 | 0/25 | 13/75 | 30/100 | 0.173 [0.104, 0.274] | 200 | 36.9 | 1.589 (30) |
| unmet_goal_facts | 2/25 | 0/25 | 0/25 | 0/25 | 0/75 | 2/100 | 0.000 [0.000, 0.049] | 200 | 36.0 | 1.358 (2) |

## TEST-B — GBFS success, budget 50 (derived from the full-budget runs)

| heuristic | 10 | 12 | 14 | 15 | n 12–15 | all | rate n 12–15 [Wilson 95%] | median exp (all) | mean plan len | mean subopt (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_s20261276 | 17/25 | 12/25 | 7/25 | 5/25 | 24/75 | 41/100 | 0.320 [0.225, 0.432] | 50 | 24.8 | 1.012 (41) |
| A0_s20261277 | 17/25 | 12/25 | 7/25 | 5/25 | 24/75 | 41/100 | 0.320 [0.225, 0.432] | 50 | 25.0 | 1.022 (41) |
| A0_s20261278 | 16/25 | 10/25 | 5/25 | 3/25 | 18/75 | 34/100 | 0.240 [0.158, 0.348] | 50 | 24.1 | 1.022 (34) |
| A1_s20261292 | 17/25 | 13/25 | 7/25 | 7/25 | 27/75 | 44/100 | 0.360 [0.261, 0.473] | 50 | 25.2 | 1.005 (43) |
| A1_s20261293 | 16/25 | 12/25 | 7/25 | 5/25 | 24/75 | 40/100 | 0.320 [0.225, 0.432] | 50 | 24.9 | 1.006 (39) |
| A1_s20261294 | 17/25 | 13/25 | 8/25 | 7/25 | 28/75 | 45/100 | 0.373 [0.273, 0.486] | 50 | 25.4 | 1.005 (44) |
| A2_s20261295 | 18/25 | 14/25 | 12/25 | 10/25 | 36/75 | 54/100 | 0.480 [0.371, 0.591] | 44 | 26.8 | 1.018 (52) |
| A2_s20261296 | 17/25 | 12/25 | 6/25 | 6/25 | 24/75 | 41/100 | 0.320 [0.225, 0.432] | 50 | 25.0 | 1.000 (40) |
| A2_s20261297 | 15/25 | 11/25 | 6/25 | 4/25 | 21/75 | 36/100 | 0.280 [0.191, 0.390] | 50 | 24.4 | 1.007 (36) |
| A3_l1 | 17/25 | 13/25 | 7/25 | 6/25 | 26/75 | 43/100 | 0.347 [0.249, 0.459] | 50 | 25.1 | 1.013 (43) |
| misplaced_blocks | 3/25 | 3/25 | 1/25 | 4/25 | 8/75 | 11/100 | 0.107 [0.055, 0.197] | 50 | 22.4 | 1.000 (11) |
| h_add | 0/25 | 0/25 | 0/25 | 0/25 | 0/75 | 0/100 | 0.000 [0.000, 0.049] | 50 | – | – (0) |
| unmet_goal_facts | 0/25 | 0/25 | 0/25 | 0/25 | 0/75 | 0/100 | 0.000 [0.000, 0.049] | 50 | – | – (0) |

## TEST-B — seed mean ± sd (success counts; A0 = R-04 seeds, A1/A2 = E07b seeds)

| arm | budget | 10 | 12 | 14 | 15 | n 12–15 | all | mean subopt (all) | median exp (all) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 2000 | 17.67 ± 1.15 | 14.67 ± 0.58 | 8.67 ± 1.15 | 8.67 ± 1.53 | 32.00 ± 3.00 | 49.67 ± 4.04 | 1.037 ± 0.007 | 1511.7 ± 845.8 |
| A0 | 500 | 17.33 ± 0.58 | 13.67 ± 1.15 | 8.00 ± 1.00 | 7.00 ± 1.73 | 28.67 ± 3.51 | 46.00 ± 4.00 | 1.033 ± 0.008 | 494.5 ± 9.5 |
| A0 | 200 | 17.00 ± 0.00 | 12.00 ± 1.73 | 7.33 ± 0.58 | 6.67 ± 1.53 | 26.00 ± 3.46 | 43.00 ± 3.46 | 1.028 ± 0.003 | 200.0 ± 0.0 |
| A0 | 50 | 16.67 ± 0.58 | 11.33 ± 1.15 | 6.33 ± 1.15 | 4.33 ± 1.15 | 22.00 ± 3.46 | 38.67 ± 4.04 | 1.019 ± 0.006 | 50.0 ± 0.0 |
| A1 | 2000 | 19.67 ± 0.58 | 15.00 ± 1.00 | 11.67 ± 3.21 | 9.33 ± 1.53 | 36.00 ± 5.00 | 55.67 ± 5.51 | 1.022 ± 0.005 | 706.0 ± 596.5 |
| A1 | 500 | 18.00 ± 1.00 | 14.00 ± 0.00 | 9.67 ± 1.53 | 8.33 ± 0.58 | 32.00 ± 2.00 | 50.00 ± 3.00 | 1.021 ± 0.004 | 396.5 ± 117.7 |
| A1 | 200 | 17.00 ± 0.00 | 12.67 ± 0.58 | 7.67 ± 0.58 | 8.00 ± 0.00 | 28.33 ± 1.15 | 45.33 ± 1.15 | 1.011 ± 0.004 | 200.0 ± 0.0 |
| A1 | 50 | 16.67 ± 0.58 | 12.67 ± 0.58 | 7.33 ± 0.58 | 6.33 ± 1.15 | 26.33 ± 2.08 | 43.00 ± 2.65 | 1.005 ± 0.001 | 50.0 ± 0.0 |
| A2 | 2000 | 19.67 ± 0.58 | 16.67 ± 2.89 | 12.67 ± 4.04 | 11.00 ± 3.00 | 40.33 ± 9.02 | 60.00 ± 9.54 | 1.052 ± 0.010 | 581.8 ± 776.9 |
| A2 | 500 | 19.00 ± 1.73 | 16.33 ± 3.21 | 12.00 ± 3.61 | 10.00 ± 3.46 | 38.33 ± 9.71 | 57.33 ± 11.06 | 1.042 ± 0.010 | 257.7 ± 229.5 |
| A2 | 200 | 18.67 ± 1.53 | 14.00 ± 2.00 | 10.33 ± 4.16 | 8.33 ± 4.51 | 32.67 ± 10.60 | 51.33 ± 12.06 | 1.024 ± 0.011 | 145.8 ± 88.7 |
| A2 | 50 | 16.67 ± 1.53 | 12.33 ± 1.53 | 8.00 ± 3.46 | 6.67 ± 3.06 | 27.00 ± 7.94 | 43.67 ± 9.29 | 1.008 ± 0.009 | 47.8 ± 3.8 |

## TEST-B — paired exact McNemar (descriptive; seed 1 unless seed-matched)

| kind | baseline | candidate | budget | scope | base succ | cand succ | base only | cand only | n | p |
|---|---|---|---|---|---|---|---|---|---|---|
| A0_vs_arm | A0_s20261276 | A1_s20261292 | 2000 | primary | 35 | 31 | 5 | 1 | 75 | 0.2188 |
| A0_vs_arm | A0_s20261276 | A2_s20261295 | 2000 | primary | 35 | 49 | 0 | 14 | 75 | 0.0001 |
| A0_vs_arm | A0_s20261276 | A3_l1 | 2000 | primary | 35 | 55 | 0 | 20 | 75 | 0.0000 |
| misplaced_vs_arm | misplaced_blocks | A0_s20261276 | 2000 | primary | 54 | 35 | 22 | 3 | 75 | 0.0002 |
| misplaced_vs_arm | misplaced_blocks | A1_s20261292 | 2000 | primary | 54 | 31 | 26 | 3 | 75 | 0.0000 |
| misplaced_vs_arm | misplaced_blocks | A2_s20261295 | 2000 | primary | 54 | 49 | 11 | 6 | 75 | 0.3323 |
| misplaced_vs_arm | misplaced_blocks | A3_l1 | 2000 | primary | 54 | 55 | 8 | 9 | 75 | 1.0000 |
| seed_matched | A0_s20261276 | A1_s20261292 | 2000 | primary | 35 | 31 | 5 | 1 | 75 | 0.2188 |
| seed_matched | A0_s20261276 | A2_s20261295 | 2000 | primary | 35 | 49 | 0 | 14 | 75 | 0.0001 |
| seed_matched | A0_s20261277 | A1_s20261293 | 2000 | primary | 32 | 36 | 2 | 6 | 75 | 0.2891 |
| seed_matched | A0_s20261277 | A2_s20261296 | 2000 | primary | 32 | 41 | 1 | 10 | 75 | 0.0117 |
| seed_matched | A0_s20261278 | A1_s20261294 | 2000 | primary | 29 | 41 | 0 | 12 | 75 | 0.0005 |
| seed_matched | A0_s20261278 | A2_s20261297 | 2000 | primary | 29 | 31 | 0 | 2 | 75 | 0.5000 |
| A0_vs_arm | A0_s20261276 | A1_s20261292 | 2000 | all | 54 | 50 | 6 | 2 | 100 | 0.2891 |
| A0_vs_arm | A0_s20261276 | A2_s20261295 | 2000 | all | 54 | 69 | 0 | 15 | 100 | 0.0001 |
| A0_vs_arm | A0_s20261276 | A3_l1 | 2000 | all | 54 | 80 | 0 | 26 | 100 | 0.0000 |
| misplaced_vs_arm | misplaced_blocks | A0_s20261276 | 2000 | all | 75 | 54 | 25 | 4 | 100 | 0.0001 |
| misplaced_vs_arm | misplaced_blocks | A1_s20261292 | 2000 | all | 75 | 50 | 30 | 5 | 100 | 0.0000 |
| misplaced_vs_arm | misplaced_blocks | A2_s20261295 | 2000 | all | 75 | 69 | 14 | 8 | 100 | 0.2863 |
| misplaced_vs_arm | misplaced_blocks | A3_l1 | 2000 | all | 75 | 80 | 8 | 13 | 100 | 0.3833 |
| seed_matched | A0_s20261276 | A1_s20261292 | 2000 | all | 54 | 50 | 6 | 2 | 100 | 0.2891 |
| seed_matched | A0_s20261276 | A2_s20261295 | 2000 | all | 54 | 69 | 0 | 15 | 100 | 0.0001 |
| seed_matched | A0_s20261277 | A1_s20261293 | 2000 | all | 49 | 56 | 2 | 9 | 100 | 0.0654 |
| seed_matched | A0_s20261277 | A2_s20261296 | 2000 | all | 49 | 61 | 1 | 13 | 100 | 0.0018 |
| seed_matched | A0_s20261278 | A1_s20261294 | 2000 | all | 46 | 61 | 0 | 15 | 100 | 0.0001 |
| seed_matched | A0_s20261278 | A2_s20261297 | 2000 | all | 46 | 50 | 0 | 4 | 100 | 0.1250 |
| A0_vs_arm | A0_s20261276 | A1_s20261292 | 200 | primary | 28 | 29 | 1 | 2 | 75 | 1.0000 |
| A0_vs_arm | A0_s20261276 | A2_s20261295 | 200 | primary | 28 | 44 | 0 | 16 | 75 | 0.0000 |
| A0_vs_arm | A0_s20261276 | A3_l1 | 200 | primary | 28 | 39 | 0 | 11 | 75 | 0.0010 |
| misplaced_vs_arm | misplaced_blocks | A0_s20261276 | 200 | primary | 19 | 28 | 5 | 14 | 75 | 0.0636 |
| misplaced_vs_arm | misplaced_blocks | A1_s20261292 | 200 | primary | 19 | 29 | 5 | 15 | 75 | 0.0414 |
| misplaced_vs_arm | misplaced_blocks | A2_s20261295 | 200 | primary | 19 | 44 | 3 | 28 | 75 | 0.0000 |
| misplaced_vs_arm | misplaced_blocks | A3_l1 | 200 | primary | 19 | 39 | 4 | 24 | 75 | 0.0002 |
| seed_matched | A0_s20261276 | A1_s20261292 | 200 | primary | 28 | 29 | 1 | 2 | 75 | 1.0000 |
| seed_matched | A0_s20261276 | A2_s20261295 | 200 | primary | 28 | 44 | 0 | 16 | 75 | 0.0000 |
| seed_matched | A0_s20261277 | A1_s20261293 | 200 | primary | 28 | 27 | 3 | 2 | 75 | 1.0000 |
| seed_matched | A0_s20261277 | A2_s20261296 | 200 | primary | 28 | 31 | 4 | 7 | 75 | 0.5488 |
| seed_matched | A0_s20261278 | A1_s20261294 | 200 | primary | 22 | 29 | 0 | 7 | 75 | 0.0156 |
| seed_matched | A0_s20261278 | A2_s20261297 | 200 | primary | 22 | 23 | 1 | 2 | 75 | 1.0000 |
| A0_vs_arm | A0_s20261276 | A1_s20261292 | 200 | all | 45 | 46 | 1 | 2 | 100 | 1.0000 |
| A0_vs_arm | A0_s20261276 | A2_s20261295 | 200 | all | 45 | 64 | 0 | 19 | 100 | 0.0000 |
| A0_vs_arm | A0_s20261276 | A3_l1 | 200 | all | 45 | 58 | 0 | 13 | 100 | 0.0002 |
| misplaced_vs_arm | misplaced_blocks | A0_s20261276 | 200 | all | 33 | 45 | 5 | 17 | 100 | 0.0169 |
| misplaced_vs_arm | misplaced_blocks | A1_s20261292 | 200 | all | 33 | 46 | 5 | 18 | 100 | 0.0106 |
| misplaced_vs_arm | misplaced_blocks | A2_s20261295 | 200 | all | 33 | 64 | 3 | 34 | 100 | 0.0000 |
| misplaced_vs_arm | misplaced_blocks | A3_l1 | 200 | all | 33 | 58 | 4 | 29 | 100 | 0.0000 |
| seed_matched | A0_s20261276 | A1_s20261292 | 200 | all | 45 | 46 | 1 | 2 | 100 | 1.0000 |
| seed_matched | A0_s20261276 | A2_s20261295 | 200 | all | 45 | 64 | 0 | 19 | 100 | 0.0000 |
| seed_matched | A0_s20261277 | A1_s20261293 | 200 | all | 45 | 44 | 4 | 3 | 100 | 1.0000 |
| seed_matched | A0_s20261277 | A2_s20261296 | 200 | all | 45 | 50 | 4 | 9 | 100 | 0.2668 |
| seed_matched | A0_s20261278 | A1_s20261294 | 200 | all | 39 | 46 | 0 | 7 | 100 | 0.0156 |
| seed_matched | A0_s20261278 | A2_s20261297 | 200 | all | 39 | 40 | 1 | 2 | 100 | 1.0000 |

## Budget monotonicity

Each (instance, heuristic) runs once at budget 2000; success at a smaller budget b is derived as (found a valid plan and expansions ≤ b), as R-04 did. This is exact because `search.gbfs` checks the budget only before each pop, applies the goal test at generation, and its pop order and heuristic calls do not depend on the budget, so a b-budget run is a prefix of the larger run. Direct re-runs at every smaller budget: 30/30 identical (heuristics ['A0_s20261276', 'misplaced_blocks']).

## Policy accuracy on R-04's E07 test states (n [8, 10, 12]; E07 definition)

| arm | 8 | 10 | 12 | OOD (pooled states) |
|---|---|---|---|---|
| A0 (mean ± sd over 3 seeds) | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.995 ± 0.002 | 0.998 ± 0.001 |
| A1 (mean ± sd over 3 seeds) | 1.000 ± 0.000 | 0.998 ± 0.001 | 0.995 ± 0.004 | 0.997 ± 0.002 |
| A2 (mean ± sd over 3 seeds) | 0.999 ± 0.001 | 0.997 ± 0.002 | 0.997 ± 0.002 | 0.998 ± 0.001 |
| A3_l1 (A0 s1 + λ·misplaced; descriptive extra) | 1.000 | 0.998 | 0.997 | 0.998 |

| comparison | scope | mean A0 | mean arm | diff | 95% CI | instances |
|---|---|---|---|---|---|---|
| A1_minus_A0 | 8 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 15 |
| A2_minus_A0 | 8 | 1.000 | 1.000 | -0.000 | [-0.001, +0.000] | 15 |
| A1_minus_A0 | 10 | 0.998 | 0.998 | -0.000 | [-0.003, +0.003] | 15 |
| A2_minus_A0 | 10 | 0.998 | 0.998 | -0.000 | [-0.004, +0.003] | 15 |
| A1_minus_A0 | 12 | 0.994 | 0.995 | +0.000 | [-0.002, +0.003] | 15 |
| A2_minus_A0 | 12 | 0.994 | 0.997 | +0.003 | [-0.001, +0.009] | 15 |
| A1_minus_A0 | OOD | 0.998 | 0.998 | -0.000 | [-0.001, +0.001] | 45 |
| A2_minus_A0 | OOD | 0.998 | 0.998 | +0.001 | [-0.001, +0.003] | 45 |

Instance-clustered bootstrap, seeds averaged within instance, 5000 resamples, seed 20261299. A0 reproduction against R-04's stored E07 scores: `{"A0_s20261276": {"max_abs_pred_diff": 0.0, "policy_disagreements": 0, "states": 2017}, "A0_s20261277": {"max_abs_pred_diff": 0.0, "policy_disagreements": 0, "states": 2017}, "A0_s20261278": {"max_abs_pred_diff": 0.0, "policy_disagreements": 0, "states": 2017}}`.

## Training (A1, A2; R-04 recipe: same epochs, batch, AdamW lr, time cap, val-MAE checkpoint)

| arm | seed | train rows | epochs | steps | best epoch | best val MAE | cap hit | wall s | ms/step | last-epoch hinge / misorder |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 20261292 | 181230 | 12 | 33984 | 11 | 0.0486 | False | 202.3 | 5.3 | – |
| A1 | 20261293 | 181230 | 12 | 33984 | 11 | 0.0578 | False | 226.6 | 5.5 | – |
| A1 | 20261294 | 181230 | 12 | 33984 | 11 | 0.0668 | False | 183.6 | 5.4 | – |
| A2 | 20261295 | 110507 | 12 | 20724 | 8 | 0.0951 | False | 255.2 | 12.3 | 0.0011 / 0.0006 |
| A2 | 20261296 | 110507 | 12 | 20724 | 9 | 0.0841 | False | 252.7 | 12.2 | 0.0011 / 0.0005 |
| A2 | 20261297 | 110507 | 12 | 20724 | 8 | 0.0768 | False | 244.0 | 11.7 | 0.0012 / 0.0005 |

A2 rank data: `{"succ_refs": 304331, "succ_excluded_signature": 421, "succ_is_row": 72497, "rows_no_pairs_no_optimal": 8849, "succ_excluded_selection_signature": 553, "rows_ranked": 79947, "pairs": 170705, "ranked_succ": 240690, "ranked_opt_succ": 109553, "rows_no_pairs_all_optimal": 20449, "rows_h0": 1262, "succ_excluded_invariant": 28175, "extra_graphs": 172619, "pairs_per_ranked_row": 2.135227087945764, "succ_per_ranked_row": 3.0106195354422303, "build_s": 5.687326700004633}`

## Data

TEST-B {'10': 25, '12': 25, '14': 25, '15': 25}, VAL-B {'12': 15, '14': 15} (full goals; seeds 20261290, 20261291). Disjointness checks: `{"testb_unique": true, "valb_unique": true, "testb_x_r04_e08": 0, "valb_x_r04_e08": 0, "testb_x_valb": 0, "testb_valb_x_r04_e07_test": 0, "goal_classes_x_r04_train_val": 0, "goals_all_on_facts": true}`.
Generation filters: `{"testb": {"draws": 115, "excluded_invariant": 15, "kept": 100}, "valb": {"draws": 37, "excluded_invariant": 7, "kept": 30}, "a1": {"draws": 870, "excluded_invariant": 510, "rejected_class": 186, "generate_none": 2, "duplicate": 162, "kept": 10}}`.
Optimal-length coverage (A* within 300000 expansions): {'10': '25/25', '12': '24/25', '14': '20/25', '15': '16/25'}.

A1 n = 8 goal classes. Availability = discovery count in 50000 draws (seed `20261298:discovery`, filters {'excluded_invariant': 28711, 'generate_none': 87, 'counted': 21202}); fresh = not a goal class of R-04's E07 test states at n = 8; chosen = the 10 most frequent fresh classes with ≥ 5 draws.

| goal class | discovery count | status |
|---|---|---|
| chains:8:3,5 | 3075 | R-04 E07 test class (not fresh) · in R-04 E08 n=8 |
| chains:8:1,7 | 2532 | R-04 E07 test class (not fresh) · in R-04 E08 n=8 |
| chains:8:1,3,4 | 1915 | R-04 E07 test class (not fresh) · in R-04 E08 n=8 |
| chains:8:4,4 | 1901 | chosen · in R-04 E08 n=8 |
| chains:8:1,2,5 | 1612 | chosen · in R-04 E08 n=8 |
| chains:8:1,1,6 | 1465 | chosen · in R-04 E08 n=8 |
| chains:8:2,6 | 1375 | chosen · in R-04 E08 n=8 |
| chains:8:2,2,4 | 1303 | chosen |
| chains:8:8 | 1260 | chosen · in R-04 E08 n=8 |
| chains:8:2,3,3 | 1252 | R-04 E07 test class (not fresh) |
| chains:8:1,2,2,3 | 1044 | R-04 E07 test class (not fresh) · in R-04 E08 n=8 |
| chains:8:1,1,2,4 | 887 | chosen · in R-04 E08 n=8 |
| chains:8:1,1,3,3 | 523 | R-04 E07 test class (not fresh) · in R-04 E08 n=8 |
| chains:8:1,1,1,5 | 378 | chosen |
| chains:8:1,1,1,2,3 | 288 | R-04 E07 test class (not fresh) |
| chains:8:1,1,2,2,2 | 156 | chosen |
| chains:8:2,2,2,2 | 97 | chosen |
| chains:8:1,1,1,1,4 | 72 | fresh, not chosen |
| chains:8:1,1,1,1,2,2 | 44 | R-04 E07 test class (not fresh) |
| chains:8:1,1,1,1,1,3 | 19 | R-04 E07 test class (not fresh) |
| chains:8:1,1,1,1,1,1,2 | 4 | R-04 E07 test class (not fresh) |

A1 instances: seed 20261298; states: 10000 sampled per instance (seed `20261298:states`), exact h* from the n = 8 space; rows kept 70723; build `{"dead": 0, "rows": 70723, "rows_excluded_invariant": 29277, "instances": 10, "dead_states_in_space": 0}`; checks `{"classes_as_chosen": true, "x_r04_e08_instances": 0, "x_r04_e07_test_instances": 0, "x_r04_e07_test_classes": 0, "x_r04_train_val_classes": 0}`.

## Resolutions of prereg ambiguities (see README, section RCX-E07b)

- A1 'fresh goal classes' = n = 8 full-goal classes that are not goal classes of R-04's E07 test states at n = 8 (the only n = 8 states E07b evaluates). 11 such classes exist, so '10' is satisfiable. The stricter reading that also excludes the classes of R-04's E08 n = 8 instances (not used by E07b) leaves only 5 classes and would make the prereg impossible; it is recorded in data/meta.json as `strict_reading_fresh_classes`.
- A1 'largest-available' = highest frequency among discovery draws of the generator (B.generate, full goals) with seed '20261298:discovery' (50,000 draws; generate()=None and invariant filters as R-04's allocate_classes); a class must be drawn at least 5 times (R-04's min_class_count); ties by class key. One instance per class (R-04's unique='goal' scheme), drawn with the spare seed 20261298; 10,000 states per instance sampled with seed '20261298:states' (R-04's include_init and row-exclusion rules). A1's checkpoint selection uses R-04's val rows.
- A2 hinge: for every training row with h* > 0, every pair (optimal successor o, non-optimal successor q) with relu(0.5 + f(o) - f(q)); mean over a row's pairs, mean over rows with pairs, added to the Huber loss with weight 1. Successors whose (state, goal) is a reserved instance (row-level exclusion) are dropped. One forward pass per batch over the rows plus their successors.
- Paired tests (primary and secondary) use seed 1 of each arm (A0 = 20261276, A1 = 20261292, A2 = 20261295); A3 is built on A0 seed 1. Seed mean ± sd is descriptive. 'A0 success@2000' in IMPROVED is A0 seed 1.
- Remaining ties: λ by smaller λ (least departure from A0); BEST by prereg listing order A1, A2, A3.
- VAL-B also runs A0 seed 1 and misplaced_blocks as context; they are never candidates.
- Smaller budgets are derived from one 2,000-expansion run per (instance, heuristic) (budget-monotone GBFS; verified by direct re-runs).
- TEST-B/VAL-B are also kept disjoint from R-04's E07 OOD test instances (n > 7), as R-04's E08 was.

## Provenance

```json
{
 "prereg": {
  "path": "experiments/RCX-E07/PREREG_E07B.md",
  "sha256_verified": "8a4cbde047f17eeb415ad9353459cc17f99e4f81c3a6008ba090db074af29cfb"
 },
 "code_sha256": {
  "experiments/RCX-E07/run_e07b.py": "b91fdd4f88486f391b27e6c33ef55c4ed56b77d7d547ebbb63a603c5f8ed4769",
  "experiments/RCX-E07/run_e07.py": "b1f6bb2215aa78e3ba39b0abdc59b36cea9f0befa2015a15e08dbdc8fb07a43e",
  "rcx/stats.py": "06a30c10acb6998033619560aac1b1130542a4344d0edf53e85e63386acf24b2",
  "rcx/guard.py": "08fa782d848528a318b0a6fd9699e4d7c301aacdd03e9fdf7c0e6d2e6ca26054",
  "rcx/bw/__init__.py": "4f403acd75709867f8dc7f6952a4e63e702ebaa40001dc2cb7024a6feb1440aa",
  "rcx/bw/bwm_import.py": "84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e",
  "rcx/bw/excl.py": "eb31d9eb5cd98e9b5372248ff03104cd4016254e472aaa8b07e8a6abc6f4e252",
  "rcx/bw/space.py": "29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4",
  "rcx/bw/data.py": "3f85401cd0bbe1d35fe49fb93678b11541768588953071975ba5e2f7e11b6b9d",
  "rcx/bw/encode.py": "32980f141eaf0fac4757ee253d22c59c55f889ae98c26eddd37eef055b928a8a",
  "rcx/bw/models.py": "4c2e3242eadc4ee155d8ddedb5f4188ac3aa5f70a19061145ff0e2fbee118f52",
  "rcx/bw/heuristics.py": "2f80f7a751e23589cf25c9e57342685bcb2c516cc8492d95d8ad3ada3e03191d",
  "rcx/bw/search.py": "10a1b499446e732ff860f479f30639270296a8ac2443abb4024ed50def9b758f"
 },
 "exclusions_json_sha256": "2f211de379352ef53450dd0a091987408e19fb9dad1db4d8ea24f549c6178167",
 "selection_set_exclusions": {
  "runs/BW_data_v2/instances_val.json": {
   "sha256": "ca1dd0e027343862763fd09452390861616426a955e8c6ec8836d3659c8d0488",
   "instances": 400,
   "unique_signatures": 181,
   "already_in_exclusions": 0,
   "block_counts": {
    "4": 67,
    "5": 284,
    "6": 49
   }
  },
  "runs/BW_data/instances_val.json": {
   "sha256": "baa6fc7665e194bd84ba7dc3d312dc93c1e896194373694f23c3fad0bb3b17b6",
   "instances": 400,
   "unique_signatures": 136,
   "already_in_exclusions": 0,
   "block_counts": {
    "4": 49,
    "5": 288,
    "6": 63
   }
  }
 },
 "r04_inputs": {
  "prereg_sha256": "8a4cbde047f17eeb415ad9353459cc17f99e4f81c3a6008ba090db074af29cfb",
  "r04_dir": "runs/RCX-E07/main",
  "r04_files_sha256": {
   "config.json": "6660e816d5178b3e03a2f0f00cc153dacae57af7c8dc4104f9ec108d2412e107",
   "data/meta.json": "ac3fc5f680ea6f50e105557c15b6ede9737ba607a19ecc26d664c270b4521316",
   "data/instances.json": "1e39f9bbc25a7e857ccbfee97c996638fed63d606c334efef88fc3b7a32f4e31",
   "data/train.jsonl.gz": "14860383211467fea49a6d968092129a5baf6f05447df6cb7befca34b85f2698",
   "data/val.jsonl.gz": "502c90f60f9d0e8df035a6c87f63335a2f8b2a56e26600a20b791ec6e38616dd",
   "data/test.jsonl.gz": "81d34ad136f48fac48af003fd2c4b24fb891d25737ca65a7462dff24dbf01be3",
   "e07_scores.jsonl.gz": "51e907231fb789c3be195f174dc3b11142dd687219794c50f1142f3c83d8103b",
   "e08_instances.json": "2b84da7be546de2f8723caf640a2bccfa4084917d4678588aef13a3b61a84dc6",
   "models/gnn_s20261276.pt": "9cc417a3075391e4fc8022f0dddd811e90fb4647c6a88dcd81361e9bb2c13f04",
   "models/gnn_s20261277.pt": "a05e559d77afd5a62576b9e02b8d4521bb4cbb0b662e0849d5eb3b69fef77694",
   "models/gnn_s20261278.pt": "575d76379f424e645eab617f24979543848cf7ac809cc371819dd169abef54b5"
  }
 },
 "bwm_pins": {
  "bwmn/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "bwmn/blocks.py": "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83",
  "scripts/blocks_val_large.py": "6a66d489944150b50294a75f8c07671ad722b51ede09409ddee1d2cfc91f2356",
  "scripts/bw_distill.py": "2b2d518206b2e36c252ac9b87f991a73ff701b54404758e0363670d573d49fa1"
 },
 "seeds": {
  "testb_instances": 20261290,
  "valb_instances": 20261291,
  "a1_models": [
   20261292,
   20261293,
   20261294
  ],
  "a2_models": [
   20261295,
   20261296,
   20261297
  ],
  "spare_a1_n8": 20261298,
  "bootstrap": 20261299,
  "a1_discovery": "20261298:discovery",
  "a1_states": "20261298:states",
  "a0_models_r04": [
   20261276,
   20261277,
   20261278
  ],
  "r04_seeds": {
   "train_instances": 20261270,
   "val_instances": 20261271,
   "test_ood_instances": 20261272,
   "e08_instances": 20261273,
   "state_sampling": 20261274,
   "split": 20261275,
   "model_init": 20261276,
   "tests": 20261280
  }
 },
 "config": {
  "testb_sizes": [
   10,
   12,
   14,
   15
  ],
  "testb_per_size": 25,
  "valb_sizes": [
   12,
   14
  ],
  "valb_per_size": 15,
  "primary_sizes": [
   12,
   14,
   15
  ],
  "a1_n": 8,
  "a1_classes": 10,
  "a1_states": 10000,
  "a1_discovery_draws": 50000,
  "a1_min_class_count": 5,
  "a1_max_draws": 50000,
  "r04_train_stride": 1,
  "a0_seeds": [
   20261276,
   20261277,
   20261278
  ],
  "a1_seeds": [
   20261292,
   20261293,
   20261294
  ],
  "a2_seeds": [
   20261295,
   20261296,
   20261297
  ],
  "epochs": 12,
  "batch_size": 64,
  "lr": 0.002,
  "time_cap_s": 3600.0,
  "rank_margin": 0.5,
  "rank_weight": 1.0,
  "lambdas": [
   0.25,
   0.5,
   1.0
  ],
  "budgets": [
   50,
   200,
   500,
   2000
  ],
  "b_primary": 2000,
  "b_low": 200,
  "opt_budget": 300000,
  "policy_sizes": [
   8,
   10,
   12
  ],
  "policy_inst_per_size": null,
  "tie_tol": 1e-05,
  "closed_margin": 3,
  "improved_delta": 8,
  "alpha": 0.05,
  "bootstrap_resamples": 5000,
  "mono_check": [
   3,
   2
  ]
 },
 "smoke": false,
 "threads": 4,
 "python": "3.11.15",
 "torch": "2.6.0+cu124",
 "numpy": "2.4.6",
 "platform": "Windows-10-10.0.26200-SP0",
 "wall_s": {
  "data_b": 160.63,
  "train_b": 1376.13,
  "select": 162.97,
  "e08b": 855.79,
  "policy": 4.7
 },
 "guard_wait_s": 80.02999999999884,
 "bwm_bytecode_check": {
  "new_pycache_dirs": [],
  "new_or_changed_pyc": [],
  "ours": []
 }
}
```

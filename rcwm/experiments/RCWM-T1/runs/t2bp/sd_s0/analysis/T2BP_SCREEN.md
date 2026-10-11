# T2b prerequisite representation screen: C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1\runs\t2bp\sd_s0

**Verdict (pre-stated rule revision 4, scripts/t2bp_analysis.py):** WINNER=bbits

WINNER=B-bits: seeds 1 and 2 run B-bits (same rows, T2 recipe).

Frozen inputs unchanged: True. Anchors passed in every evaluating turn: True.

| Arm | IAW | OOR | K4 | HS | FC | FA | WF | ID | R |
|---|---|---|---|---|---|---|---|---|---|
| D-int | 0.100 | 0.447 | 0.133 | 0.462 | 0.731 | 0.490 | 1.000 | 0.447 | 0.286 |
| S-trace | 0.587 | 0.557 | 0.125 | 0.537 | 0.289 | 0.210 | 1.000 | 0.584 | 0.451 |
| B-bits | 0.833 | 0.800 | 0.842 | 0.741 | 0.074 | 0.140 | 1.000 | 0.918 | 0.804 |

Excluded: D-int: IAW 0.100 < 0.2; false copy 0.731 > 0.25; false apply 0.490 > 0.25; S-trace: false copy 0.289 > 0.25

Per-stratum accuracy (OFF -> ON):

| Stratum | D-int | S-trace | B-bits |
|---|---|---|---|
| D_app_add | 0.80 -> 0.46 | 0.66 -> 0.44 | 0.00 -> 0.94 |
| D_app_unchanged | 0.04 -> 0.59 | 0.14 -> 0.43 | 0.00 -> 1.00 |
| D_app_delete | 0.04 -> 0.02 | 0.01 -> 0.04 | 0.00 -> 0.95 |
| D_int_add_wrong | 0.01 -> 0.10 | 0.00 -> 0.59 | 0.00 -> 0.83 |
| D_inapp_pre_on | 0.01 -> 0.51 | 0.04 -> 0.80 | 0.00 -> 0.97 |
| D_inapp_pre_off | 0.02 -> 0.52 | 0.04 -> 1.00 | 0.00 -> 0.86 |
| D_ood10 | 0.12 -> 0.44 | 0.19 -> 0.57 | 0.00 -> 0.80 |
| D_ood12 | 0.22 -> 0.45 | 0.25 -> 0.55 | 0.00 -> 0.80 |
| D_inapp_addset | 0.05 -> 0.93 | 0.11 -> 0.79 | 0.00 -> 0.87 |

Held-out surface (stage E3, ON; paired by row with the in-surface ON answers):

| Stratum | D-int | S-trace | B-bits |
|---|---|---|---|
| D_app_add | 0.46 -> 0.16 | 0.44 -> 0.19 | 0.94 -> 0.34 |
| D_app_unchanged | 0.59 -> 0.69 | 0.43 -> 0.64 | 1.00 -> 1.00 |
| D_app_delete | 0.02 -> 0.00 | 0.04 -> 0.01 | 0.95 -> 0.43 |
| D_int_add_wrong | 0.10 -> 0.13 | 0.59 -> 0.08 | 0.83 -> 0.43 |
| D_inapp_pre_on | 0.51 -> 0.67 | 0.80 -> 0.91 | 0.97 -> 1.00 |
| D_inapp_pre_off | 0.52 -> 0.78 | 1.00 -> 1.00 | 0.86 -> 0.98 |
| D_ood10 | 0.44 -> 0.43 | 0.57 -> 0.47 | 0.80 -> 0.63 |
| D_ood12 | 0.45 -> 0.35 | 0.55 -> 0.45 | 0.80 -> 0.64 |
| D_inapp_addset | 0.93 -> 0.81 | 0.79 -> 0.93 | 0.87 -> 1.00 |

Training dose: 150 of the 1,500 training rows were int-add-wrong (A4) rows.

Descriptive only below this line; nothing here changes the verdict. This is a dev-bank screen, not a confirmation.

Held-out-surface chains (stage E4, ON; REPORT-ONLY, not in R and not in eligibility; K4 in the ranking is the in-surface value):

| Arm | K4 (in-surface) | K4_HS (held-out surface) | K4 - K4_HS | 95% CI (chains resampled) |
|---|---|---|---|---|
| D-int | 0.133 | 0.087 | 0.046 | [0.004, 0.092] |
| S-trace | 0.125 | 0.125 | 0.000 | [-0.046, 0.046] |
| B-bits | 0.842 | 0.417 | 0.425 | [0.354, 0.496] |

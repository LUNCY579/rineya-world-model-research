# FREPLAY: registered CPU replay of every logged F query (PREREG_FREPLAY.md, F-09)

Written 2026-10-11 16:21:14 Central Daylight Time by `scripts/freplay.py` (sha256 `866ca5b696b465d0de77d0882b38323df50f1bb3bcc216ff68aaadbb0beb0a3b`). Prereg sha256 `994d74c0b721724a49cd24774b5ecf0fc12027d058718575f276b100b3e975f8` (verified at run time; every input sha256 verified against the prereg table before reading). No model and no torch were loaded (torch in sys.modules: False; bwmn.lm: False). Wall 7.2 s.

`core` sha256 (byte-compare target, prereg §8): `9413d13c18f40cac56049926b47839bbe2d468a92fc8fc1c4158779628c305f8`.

Not blind: the exploratory F-09 scratch numbers existed before this replay (prereg §0); see the last table.

## Per set: classes of unique (state, action) keys

| Set | Unique keys | Legal / illegal | ok | false_illegal | false_legal_accepted | false_legal_rejected_by_support_check | wrong_state | malformed | Conflicting keys |
|---|---|---|---|---|---|---|---|---|---|
| b2_hard | 20530 | 16464 / 4066 | 19726 | 6 | 0 | 798 | 0 | 0 | 0 |
| b2_mirror | 23752 | 19254 / 4498 | 22673 | 1 | 0 | 1078 | 0 | 0 | 0 |
| b2_pb602 | 871 | 794 / 77 | 861 | 0 | 0 | 10 | 0 | 0 | 0 |
| h2_hard | 22895 | 20532 / 2363 | 22233 | 5 | 0 | 657 | 0 | 0 | 0 |
| distill_r1 | 15030 | 13360 / 1670 | 14662 | 0 | 0 | 368 | 0 | 0 | 0 |
| all5 (pooled, secondary) | 77432 | 65083 / 12349 | 74578 | 8 | 0 | 2846 | 0 | 0 | 0 |

Subtypes (prereg §3):

| Set | ok_legal_exact | ok_illegal_flag | ok_illegal_flag_unsupported | wrong_state | fi_flag | fi_support | fi_malformed | false_legal_accepted | false_legal_rejected_by_support_check | malformed |
|---|---|---|---|---|---|---|---|---|---|---|
| b2_hard | 16458 | 3268 | 0 | 0 | 6 | 0 | 0 | 0 | 798 | 0 |
| b2_mirror | 19253 | 3420 | 0 | 0 | 1 | 0 | 0 | 0 | 1078 | 0 |
| b2_pb602 | 794 | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| h2_hard | 20527 | 1706 | 0 | 0 | 5 | 0 | 0 | 0 | 657 | 0 |
| distill_r1 | 13360 | 1302 | 0 | 0 | 0 | 0 | 0 | 0 | 368 | 0 |
| all5 | 65075 | 9503 | 0 | 0 | 8 | 0 | 0 | 0 | 2846 | 0 |

## Rates (exact Clopper-Pearson 95% CIs)

| Set | harmful / keys | false_illegal / legal | wrong_state / legal | false_legal_accepted / illegal | **R2** support_share / illegal | F flag says legal / illegal | F-flag accuracy (schema-valid) | delta-exact |
|---|---|---|---|---|---|---|---|---|
| b2_hard | 6/20530 = 0.03% [0.01, 0.06] | 6/16464 = 0.04% [0.01, 0.08] | 0/16464 = 0.00% [0.00, 0.02] | 0/4066 = 0.00% [0.00, 0.09] | 798/4066 = 19.63% [18.42, 20.88] | 798/4066 = 19.63% [18.42, 20.88] | 19726/20530 = 96.08% [95.81, 96.34] | 19726/20530 = 96.08% [95.81, 96.34] |
| b2_mirror | 1/23752 = 0.00% [0.00, 0.02] | 1/19254 = 0.01% [0.00, 0.03] | 0/19254 = 0.00% [0.00, 0.02] | 0/4498 = 0.00% [0.00, 0.08] | 1078/4498 = 23.97% [22.72, 25.24] | 1078/4498 = 23.97% [22.72, 25.24] | 22673/23752 = 95.46% [95.18, 95.72] | 22673/23752 = 95.46% [95.18, 95.72] |
| b2_pb602 | 0/871 = 0.00% [0.00, 0.42] | 0/794 = 0.00% [0.00, 0.46] | 0/794 = 0.00% [0.00, 0.46] | 0/77 = 0.00% [0.00, 4.68] | 10/77 = 12.99% [6.41, 22.59] | 10/77 = 12.99% [6.41, 22.59] | 861/871 = 98.85% [97.90, 99.45] | 861/871 = 98.85% [97.90, 99.45] |
| h2_hard | 5/22895 = 0.02% [0.01, 0.05] | 5/20532 = 0.02% [0.01, 0.06] | 0/20532 = 0.00% [0.00, 0.02] | 0/2363 = 0.00% [0.00, 0.16] | 657/2363 = 27.80% [26.00, 29.66] | 657/2363 = 27.80% [26.00, 29.66] | 22233/22895 = 97.11% [96.88, 97.32] | 22233/22895 = 97.11% [96.88, 97.32] |
| distill_r1 | 0/15030 = 0.00% [0.00, 0.02] | 0/13360 = 0.00% [0.00, 0.03] | 0/13360 = 0.00% [0.00, 0.03] | 0/1670 = 0.00% [0.00, 0.22] | 368/1670 = 22.04% [20.07, 24.10] | 368/1670 = 22.04% [20.07, 24.10] | 14662/15030 = 97.55% [97.29, 97.79] | 14662/15030 = 97.55% [97.29, 97.79] |
| all5 | 8/77432 = 0.01% [0.00, 0.02] | 8/65083 = 0.01% [0.01, 0.02] | 0/65083 = 0.00% [0.00, 0.01] | 0/12349 = 0.00% [0.00, 0.03] | 2846/12349 = 23.05% [22.31, 23.80] | 2846/12349 = 23.05% [22.31, 23.80] | 74578/77432 = 96.31% [96.18, 96.45] | 74578/77432 = 96.31% [96.18, 96.45] |

## R1 (oracle-equivalence; prereg §6)

| Set | Instances | Observed success | Flagged (harmful key in forest) | Flagged solved | Exact-F success bounds [L, U] | Ceiling <= observed | Support-exposed instances | Plans not covered | Verified plans not goal under F |
|---|---|---|---|---|---|---|---|---|---|
| b2_hard | 110 | 73 | 1 (hard/instance-102) | 1 | [72, 73] | True | 104 | 0 | 0 |
| b2_mirror | 110 | 61 | 1 (mirror/103) | 0 | [61, 62] | False | 103 | 0 | 0 |
| b2_pb602 | 602 | 602 | 0  | 0 | [602, 602] | True | 452 | 0 | 0 |
| h2_hard | 110 | 88 | 1 (hard/instance-102) | 1 | [87, 88] | True | 93 | 0 | 0 |
| distill_r1 | 240 | 211 | 0  | 0 | [211, 211] | True | 134 | 0 | 0 |

Reading (registered): an unflagged instance's exact-F run is identical to the logged run by construction (D-50 CRN assumption). Where every flagged instance was solved, U equals the observed success, so the exact-F ceiling is AT MOST the observed success; exact equality needs an exact-F re-run of the flagged instances only (GPU; not part of this replay). Advisory consequence for the B4 draft (D-70 answer 5): an exact-F oracle arm can show no headroom on such a set; drop it or restrict it to flagged instances.

## Harmful keys (all sets; unit level)

| Set | key_id | Class / subtype | Blocks | Action | F text |
|---|---|---|---|---|---|
| b2_hard | `0d9d585497980d01` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| b2_hard | `4e379c835b54b071` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| b2_hard | `7a75f93a24135c5f` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| b2_hard | `b76d80490b32df8d` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| b2_hard | `c3e1dda12b1fd008` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| b2_hard | `d2b3db76a2138414` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| b2_mirror | `2c70a635dcb5caca` | false_illegal / fi_flag | 15 | stack h m | `{"add":[],"del":[],"legal":false}` |
| h2_hard | `4e379c835b54b071` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| h2_hard | `7a75f93a24135c5f` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| h2_hard | `b76d80490b32df8d` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| h2_hard | `caf764b82d789409` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |
| h2_hard | `d2b3db76a2138414` | false_illegal / fi_flag | 12 | unstack c j | `{"add":[],"del":[],"legal":false}` |

## R2 (support-check dependence; prereg §7)

| Set | Claim | R2 | > 5%: wording change |
|---|---|---|---|
| b2_hard | P-08/P-10 hard: 'every plan S-beam verified in imagination was valid' | 798/4066 = 19.63% [18.42, 20.88] | True |
| b2_mirror | P-10 mirror verification statement | 1078/4498 = 23.97% [22.72, 25.24] | True |
| b2_pb602 | P-10 PB-602 verification statement | 10/77 = 12.99% [6.41, 22.59] | True |
| h2_hard | P-14: 'every plan the beam verified in imagination was valid (88/88)' | 657/2363 = 27.80% [26.00, 29.66] | True |
| distill_r1 | P-11 / B3: label precision 1.000 (211/211), 'no simulator labels' | 368/1670 = 22.04% [20.07, 24.10] | True |

R2 > 5% in any set: **True**.

Registered consequence: each claim whose set is triggered must say the plan was verified "by the learned F plus a domain-agnostic delta-consistency check (a predicted deletion must be a fact of the state)"; "the domain rules are known only to the learned F" is withdrawn from every document that states or quotes it (the pinned blocks_planner.py docstring is not edited).

## Strata by block count (unique keys)

### b2_hard

| Blocks | Keys | Legal / illegal | harmful | false_illegal | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|---|
| 6 | 28 | 25 / 3 | 0 | 0 | 1/3 = 33.33% [0.84, 90.57] | 1/3 = 33.33% [0.84, 90.57] |
| 7 | 64 | 59 / 5 | 0 | 0 | 0/5 = 0.00% [0.00, 52.18] | 0/5 = 0.00% [0.00, 52.18] |
| 8 | 457 | 404 / 53 | 0 | 0 | 13/53 = 24.53% [13.76, 38.28] | 13/53 = 24.53% [13.76, 38.28] |
| 9 | 1136 | 991 / 145 | 0 | 0 | 29/145 = 20.00% [13.82, 27.44] | 29/145 = 20.00% [13.82, 27.44] |
| 10 | 1807 | 1507 / 300 | 0 | 0 | 51/300 = 17.00% [12.93, 21.74] | 51/300 = 17.00% [12.93, 21.74] |
| 11 | 1972 | 1599 / 373 | 0 | 0 | 67/373 = 17.96% [14.20, 22.24] | 67/373 = 17.96% [14.20, 22.24] |
| 12 | 2319 | 1848 / 471 | 6 | 6 | 112/471 = 23.78% [20.00, 27.89] | 112/471 = 23.78% [20.00, 27.89] |
| 13 | 4829 | 3867 / 962 | 0 | 0 | 173/962 = 17.98% [15.61, 20.56] | 173/962 = 17.98% [15.61, 20.56] |
| 14 | 4999 | 3921 / 1078 | 0 | 0 | 232/1078 = 21.52% [19.10, 24.10] | 232/1078 = 21.52% [19.10, 24.10] |
| 15 | 2919 | 2243 / 676 | 0 | 0 | 120/676 = 17.75% [14.94, 20.84] | 120/676 = 17.75% [14.94, 20.84] |

| Action | Keys | Legal / illegal | harmful | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|
| pick-up | 5787 | 3780 / 2007 | 0 | 735/2007 = 36.62% [34.51, 38.77] | 735/2007 = 36.62% [34.51, 38.77] |
| put-down | 4077 | 4000 / 77 | 0 | 0/77 = 0.00% [0.00, 4.68] | 0/77 = 0.00% [0.00, 4.68] |
| stack | 4449 | 4151 / 298 | 0 | 3/298 = 1.01% [0.21, 2.91] | 3/298 = 1.01% [0.21, 2.91] |
| unstack | 6217 | 4533 / 1684 | 6 | 60/1684 = 3.56% [2.73, 4.56] | 60/1684 = 3.56% [2.73, 4.56] |

### b2_mirror

| Blocks | Keys | Legal / illegal | harmful | false_illegal | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|---|
| 6 | 18 | 18 / 0 | 0 | 0 | 0/0 (n/a) | 0/0 (n/a) |
| 7 | 21 | 20 / 1 | 0 | 0 | 0/1 = 0.00% [0.00, 97.50] | 0/1 = 0.00% [0.00, 97.50] |
| 8 | 457 | 402 / 55 | 0 | 0 | 10/55 = 18.18% [9.08, 30.90] | 10/55 = 18.18% [9.08, 30.90] |
| 9 | 1373 | 1196 / 177 | 0 | 0 | 33/177 = 18.64% [13.19, 25.17] | 33/177 = 18.64% [13.19, 25.17] |
| 10 | 2005 | 1708 / 297 | 0 | 0 | 56/297 = 18.86% [14.57, 23.78] | 56/297 = 18.86% [14.57, 23.78] |
| 11 | 2013 | 1667 / 346 | 0 | 0 | 90/346 = 26.01% [21.47, 30.97] | 90/346 = 26.01% [21.47, 30.97] |
| 12 | 3390 | 2703 / 687 | 0 | 0 | 155/687 = 22.56% [19.49, 25.87] | 155/687 = 22.56% [19.49, 25.87] |
| 13 | 4949 | 3992 / 957 | 0 | 0 | 224/957 = 23.41% [20.76, 26.22] | 224/957 = 23.41% [20.76, 26.22] |
| 14 | 6271 | 4961 / 1310 | 0 | 0 | 340/1310 = 25.95% [23.60, 28.42] | 340/1310 = 25.95% [23.60, 28.42] |
| 15 | 3255 | 2587 / 668 | 1 | 1 | 170/668 = 25.45% [22.19, 28.93] | 170/668 = 25.45% [22.19, 28.93] |

| Action | Keys | Legal / illegal | harmful | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|
| pick-up | 7316 | 4724 / 2592 | 0 | 1015/2592 = 39.16% [37.27, 41.07] | 1015/2592 = 39.16% [37.27, 41.07] |
| put-down | 4421 | 4375 / 46 | 0 | 0/46 = 0.00% [0.00, 7.71] | 0/46 = 0.00% [0.00, 7.71] |
| stack | 5486 | 5204 / 282 | 1 | 1/282 = 0.35% [0.01, 1.96] | 1/282 = 0.35% [0.01, 1.96] |
| unstack | 6529 | 4951 / 1578 | 0 | 62/1578 = 3.93% [3.03, 5.01] | 62/1578 = 3.93% [3.03, 5.01] |

### b2_pb602

| Blocks | Keys | Legal / illegal | harmful | false_illegal | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|---|
| 3 | 44 | 42 / 2 | 0 | 0 | 0/2 = 0.00% [0.00, 84.19] | 0/2 = 0.00% [0.00, 84.19] |
| 4 | 317 | 272 / 45 | 0 | 0 | 6/45 = 13.33% [5.05, 26.79] | 6/45 = 13.33% [5.05, 26.79] |
| 5 | 510 | 480 / 30 | 0 | 0 | 4/30 = 13.33% [3.76, 30.72] | 4/30 = 13.33% [3.76, 30.72] |

| Action | Keys | Legal / illegal | harmful | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|
| pick-up | 212 | 161 / 51 | 0 | 10/51 = 19.61% [9.82, 33.12] | 10/51 = 19.61% [9.82, 33.12] |
| put-down | 156 | 155 / 1 | 0 | 0/1 = 0.00% [0.00, 97.50] | 0/1 = 0.00% [0.00, 97.50] |
| stack | 251 | 245 / 6 | 0 | 0/6 = 0.00% [0.00, 45.93] | 0/6 = 0.00% [0.00, 45.93] |
| unstack | 252 | 233 / 19 | 0 | 0/19 = 0.00% [0.00, 17.65] | 0/19 = 0.00% [0.00, 17.65] |

### h2_hard

| Blocks | Keys | Legal / illegal | harmful | false_illegal | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|---|
| 6 | 43 | 43 / 0 | 0 | 0 | 0/0 (n/a) | 0/0 (n/a) |
| 7 | 42 | 40 / 2 | 0 | 0 | 0/2 = 0.00% [0.00, 84.19] | 0/2 = 0.00% [0.00, 84.19] |
| 8 | 398 | 374 / 24 | 0 | 0 | 4/24 = 16.67% [4.74, 37.38] | 4/24 = 16.67% [4.74, 37.38] |
| 9 | 978 | 907 / 71 | 0 | 0 | 17/71 = 23.94% [14.61, 35.54] | 17/71 = 23.94% [14.61, 35.54] |
| 10 | 1673 | 1520 / 153 | 0 | 0 | 42/153 = 27.45% [20.55, 35.24] | 42/153 = 27.45% [20.55, 35.24] |
| 11 | 2075 | 1903 / 172 | 0 | 0 | 46/172 = 26.74% [20.29, 34.02] | 46/172 = 26.74% [20.29, 34.02] |
| 12 | 3273 | 2934 / 339 | 5 | 5 | 110/339 = 32.45% [27.49, 37.72] | 110/339 = 32.45% [27.49, 37.72] |
| 13 | 5132 | 4610 / 522 | 0 | 0 | 163/522 = 31.23% [27.27, 35.40] | 163/522 = 31.23% [27.27, 35.40] |
| 14 | 5637 | 5063 / 574 | 0 | 0 | 167/574 = 29.09% [25.41, 33.00] | 167/574 = 29.09% [25.41, 33.00] |
| 15 | 3644 | 3138 / 506 | 0 | 0 | 108/506 = 21.34% [17.85, 25.18] | 108/506 = 21.34% [17.85, 25.18] |

| Action | Keys | Legal / illegal | harmful | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|
| pick-up | 5245 | 4211 / 1034 | 0 | 600/1034 = 58.03% [54.95, 61.06] | 600/1034 = 58.03% [54.95, 61.06] |
| put-down | 5512 | 5503 / 9 | 0 | 0/9 = 0.00% [0.00, 33.63] | 0/9 = 0.00% [0.00, 33.63] |
| stack | 4934 | 4678 / 256 | 0 | 2/256 = 0.78% [0.09, 2.79] | 2/256 = 0.78% [0.09, 2.79] |
| unstack | 7204 | 6140 / 1064 | 5 | 55/1064 = 5.17% [3.92, 6.68] | 55/1064 = 5.17% [3.92, 6.68] |

### distill_r1

| Blocks | Keys | Legal / illegal | harmful | false_illegal | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|---|
| 6 | 486 | 464 / 22 | 0 | 0 | 4/22 = 18.18% [5.19, 40.28] | 4/22 = 18.18% [5.19, 40.28] |
| 7 | 951 | 891 / 60 | 0 | 0 | 15/60 = 25.00% [14.72, 37.86] | 15/60 = 25.00% [14.72, 37.86] |
| 8 | 3094 | 2809 / 285 | 0 | 0 | 66/285 = 23.16% [18.39, 28.50] | 66/285 = 23.16% [18.39, 28.50] |
| 9 | 3907 | 3483 / 424 | 0 | 0 | 85/424 = 20.05% [16.34, 24.18] | 85/424 = 20.05% [16.34, 24.18] |
| 10 | 6592 | 5713 / 879 | 0 | 0 | 198/879 = 22.53% [19.80, 25.43] | 198/879 = 22.53% [19.80, 25.43] |

| Action | Keys | Legal / illegal | harmful | R2 support_share / illegal | F flag says legal / illegal |
|---|---|---|---|---|---|
| pick-up | 4545 | 3488 / 1057 | 0 | 365/1057 = 34.53% [31.66, 37.49] | 365/1057 = 34.53% [31.66, 37.49] |
| put-down | 2760 | 2750 / 10 | 0 | 0/10 = 0.00% [0.00, 30.85] | 0/10 = 0.00% [0.00, 30.85] |
| stack | 4018 | 3898 / 120 | 0 | 0/120 = 0.00% [0.00, 3.03] | 0/120 = 0.00% [0.00, 3.03] |
| unstack | 3707 | 3224 / 483 | 0 | 3/483 = 0.62% [0.13, 1.80] | 3/483 = 0.62% [0.13, 1.80] |

Set x block count x action type strata with all rates and CIs: `FREPLAY.json` core.sets.<set>.strata.by_blocks_action.

## Input files

| File | Lines | Torn lines | Keys | Duplicate lines (same / different text) | sha256 |
|---|---|---|---|---|---|
| `runs/BW_TEST_B2/hard_beam/shard_0of8/fcache.jsonl` | 2417 | 0 | 2417 | 0 / 0 | `c84563c20fec05028b1a28a271848a6081b98c54f919bf379c242d44c07283f1` |
| `runs/BW_TEST_B2/hard_beam/shard_1of8/fcache.jsonl` | 2551 | 0 | 2551 | 0 / 0 | `8931869e07489758ebc2df55855691367ac6690de6dc5562922ff7a06c41a73c` |
| `runs/BW_TEST_B2/hard_beam/shard_2of8/fcache.jsonl` | 2534 | 0 | 2534 | 0 / 0 | `54db54c9ab01fd753b494376959b630d8bec7fb1464052c9b8dcc5a43d3ad170` |
| `runs/BW_TEST_B2/hard_beam/shard_3of8/fcache.jsonl` | 2493 | 0 | 2493 | 0 / 0 | `ba4868f2b19fb708457e1bda95858f7ddc380b1accb39049f918e15d43ea7123` |
| `runs/BW_TEST_B2/hard_beam/shard_4of8/fcache.jsonl` | 2685 | 0 | 2685 | 0 / 0 | `18b5135e49756cd46c92aef130797f2727c76d3a0aa6eb72cccca3647b5776c3` |
| `runs/BW_TEST_B2/hard_beam/shard_5of8/fcache.jsonl` | 3149 | 0 | 3149 | 0 / 0 | `5d2f833fa7e535294df0c8f5f26457cfdbbee80c82fc350f9aeb2defefb15f00` |
| `runs/BW_TEST_B2/hard_beam/shard_6of8/fcache.jsonl` | 2130 | 0 | 2130 | 0 / 0 | `7baf775dbe026ac1675a9eecbcbdcc718151fbaee860824c931572e411ec7098` |
| `runs/BW_TEST_B2/hard_beam/shard_7of8/fcache.jsonl` | 2595 | 0 | 2595 | 0 / 0 | `8481815e1735f2f01b9565eb4aafa552f8c2b39f7fa2edaf597913d2606b6ce1` |
| `runs/BW_TEST_B2/mirror_beam/shard_0of8/fcache.jsonl` | 3014 | 0 | 3014 | 0 / 0 | `54a71e26ecc30963ba38a26444c4adfaa95aed317845632adf85e77fa9342df3` |
| `runs/BW_TEST_B2/mirror_beam/shard_1of8/fcache.jsonl` | 2647 | 0 | 2647 | 0 / 0 | `0d3380fb00276c27c842605821a72e57b70bc9c51334b5d8962e8223c5142e41` |
| `runs/BW_TEST_B2/mirror_beam/shard_2of8/fcache.jsonl` | 2356 | 0 | 2356 | 0 / 0 | `1b5568e43421f0d636b9b402d937345df9e8d28ab9f8453c7302334d88da59c0` |
| `runs/BW_TEST_B2/mirror_beam/shard_3of8/fcache.jsonl` | 2747 | 0 | 2747 | 0 / 0 | `e70a17d9d058f06d6b73d3378cffb956efdcd832a1be95474a3f70878d52c7d7` |
| `runs/BW_TEST_B2/mirror_beam/shard_4of8/fcache.jsonl` | 3686 | 0 | 3686 | 0 / 0 | `99416006d1f298fc29fd9cbe72fecd48c97771437e7869acc3fed22b08261a08` |
| `runs/BW_TEST_B2/mirror_beam/shard_5of8/fcache.jsonl` | 3082 | 0 | 3082 | 0 / 0 | `0fdfe929af2ec772909da5c66d26bdfddfca5d68f01a00b31e411950dfec7421` |
| `runs/BW_TEST_B2/mirror_beam/shard_6of8/fcache.jsonl` | 3378 | 0 | 3378 | 0 / 0 | `6a2e8f086c94f7b5fa83007c89fe7b52687b4b0d644388f1e03d8ce2dd5812c7` |
| `runs/BW_TEST_B2/mirror_beam/shard_7of8/fcache.jsonl` | 2842 | 0 | 2842 | 0 / 0 | `97563de12c847669849be906b93db169ad9da6961751c85f20f9a1b3654ad600` |
| `runs/BW_TEST_B2/planbench_beam/shard_0of4/fcache.jsonl` | 466 | 0 | 466 | 0 / 0 | `2c9cb1f6553423ecf613ec7fe225190ba687b92ad38a44912ccdd99f86f033c4` |
| `runs/BW_TEST_B2/planbench_beam/shard_1of4/fcache.jsonl` | 424 | 0 | 424 | 0 / 0 | `1fd852d9e16d0c23a2d54cea1cd8ae23d4f3ca121ad835ab034a3edf1baddfbf` |
| `runs/BW_TEST_B2/planbench_beam/shard_2of4/fcache.jsonl` | 454 | 0 | 454 | 0 / 0 | `7c115df45577c3222cb7235a66857f450cf6b8515f988770172d6b4c96cbcc75` |
| `runs/BW_TEST_B2/planbench_beam/shard_3of4/fcache.jsonl` | 442 | 0 | 442 | 0 / 0 | `f8ca5d7190c6da98aa08b0aa753090140ea2dfc69a88852003aec162b7c51287` |
| `runs/BW_B3_HARD/h2_beam/shard_0of8/fcache.jsonl` | 2642 | 0 | 2642 | 0 / 0 | `c7b9dfb605868e53f1d0b6bac6068a80e5bbc87779bf53f44ec89e51b809e58a` |
| `runs/BW_B3_HARD/h2_beam/shard_1of8/fcache.jsonl` | 3036 | 0 | 3036 | 0 / 0 | `28c5e6ee0d08ebd4a0895add7e2a88f5a6d98dce5ccc87edca17042242b8ca4b` |
| `runs/BW_B3_HARD/h2_beam/shard_2of8/fcache.jsonl` | 2988 | 0 | 2988 | 0 / 0 | `13ccb217f8a54a56bb15c7035cb9e643a55ca0290d38885b162ae657cb2c2717` |
| `runs/BW_B3_HARD/h2_beam/shard_3of8/fcache.jsonl` | 2890 | 0 | 2890 | 0 / 0 | `f1f2bf098263f283bde582aec94290a0f4bee0ab3f04852884b72f87a710054a` |
| `runs/BW_B3_HARD/h2_beam/shard_4of8/fcache.jsonl` | 2919 | 0 | 2919 | 0 / 0 | `52b312b8722e2e1476cf4a82fdc8093c1895687229d4c50076859d8d0753bb4f` |
| `runs/BW_B3_HARD/h2_beam/shard_5of8/fcache.jsonl` | 3057 | 0 | 3057 | 0 / 0 | `39c361eed0eff68fa78b3820f2f1cad957a6465f2e2db16401e6427b17275cc0` |
| `runs/BW_B3_HARD/h2_beam/shard_6of8/fcache.jsonl` | 2395 | 0 | 2395 | 0 / 0 | `5b5418bd565da7f4036ace70bda362188cf28fad2cd918b6ec66f56809dd5bab` |
| `runs/BW_B3_HARD/h2_beam/shard_7of8/fcache.jsonl` | 2983 | 0 | 2983 | 0 / 0 | `2c354964f8f58556313482fdf96d9708486d7121659db2150aeadbf7cf6bd998` |
| `runs/BW_DISTILL_r1/fcache.jsonl` | 15030 | 0 | 15030 | 0 / 0 | `055d3b667f74e1675f92c0ea9855d300b08fbd99f67c6acb40ac62b0e7b78b46` |

- `runs/BW_TEST_B2/hard_beam/plans.jsonl` `85812972a2e5e8bcedee245673dee98a0f0fdb93f02e2e852e44fa7c037fb3ee`
- `runs/BW_TEST_B2/mirror_beam/plans.jsonl` `22147f0750f7ceb1992504b20a3d44999d06ce48af357282ffb2b151ab913cfd`
- `runs/BW_TEST_B2/planbench_beam/plans.jsonl` `839142ab787be6d573c43c4f8876195b48f322dc3e3b4cbe2d2f2a45bf062db1`
- `runs/BW_B3_HARD/h2_beam/plans.jsonl` `3cf3cd01b8a73f3e036fda126a5d6b562b38f63e458197a188639f2df6e2b91b`
- `runs/BW_DISTILL_r1/solutions.jsonl` `fdf80151a52d5728c27d2bb7136f4ea1caa17ba3241487af9553a77dac86244c`
- `runs/BW_val_large/instances_mirror_hard.json` `2f61e38bf2f99e2c5f7524d8642618d51d1e2d83573123524b3d8e41155d53f2`
- `bench/planbench/blocksworld_hard/generated/*.pddl` 110 files, digest `d7b7e446c1cea76da344830ab750138d89a44d311de0b8d0fb4a629018100b93`
- `bench/planbench/blocksworld/generated_basic_3/*.pddl` 101 files, digest `46aceac767835c09360801823b4ac8ac0d2076e9b88120408c5bcc01bfccd3fe`
- `bench/planbench/blocksworld/generated_basic/*.pddl` 501 files, digest `4807eb8bd6cb17d8eacdfc3fa0b05e49c79d4f4de0b5684ea0f566c4024d565e`
- `bwmn/blocks.py` `b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83`; `bwmn/blocks_planner.py` (reference, not imported) `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960`
- Per-shard plans.jsonl shas: `FREPLAY.json` meta.inputs.shard_plans.

## Registered vs exploratory (F-09 scratch; prereg §0)

| Set | Quantity | Exploratory | Registered |
|---|---|---|---|
| b2_hard | unique_keys | 20530 | 20530 |
| b2_hard | false_legal_accepted | 0 | 0 |
| b2_hard | wrong_state | 0 | 0 |
| b2_hard | false_illegal | 6 | 6 |
| b2_hard | flag_legal_on_illegal | 798 | 798 |
| b2_hard | illegal_keys | 4066 | 4066 |
| h2_hard | unique_keys | 22895 | 22895 |
| h2_hard | false_legal_accepted | 0 | 0 |
| h2_hard | wrong_state | 0 | 0 |
| h2_hard | false_illegal | 5 | 5 |
| h2_hard | flag_legal_on_illegal | 657 | 657 |
| h2_hard | illegal_keys | 2363 | 2363 |
| b2_mirror | flag_legal_on_illegal | 1078 | 1078 |
| b2_mirror | illegal_keys | 4498 | 4498 |
| b2_pb602 | flag_legal_on_illegal | 10 | 10 |
| b2_pb602 | illegal_keys | 77 | 77 |
| distill_r1 | flag_legal_share_approx | 0.22 | 0.2204 |

Definitions differ slightly from the scratch scripts (prereg §0 a-d); differences are reported, not reconciled. Independent second implementation from the prereg text alone (D-61/D-71 practice): not yet done; its target is the `core` sha256 above.

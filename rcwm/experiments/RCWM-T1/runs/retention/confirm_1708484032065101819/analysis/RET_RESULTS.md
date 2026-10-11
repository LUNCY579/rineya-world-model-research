# Retention suite results: confirm_1708484032065101819

Draw sha256 `864481a3adffbb90d620a3caea6bc37c6b1862f8b7f0338535affeb1e463b19b`; selection sha256 `2a4d15febf147ccc14ddf67a8df2c82360314e0208c48dab30a0f368b6da0836`; surface diversity False; registered: True.

## Pre-stated rule

> For each gating comparison of an arm A with its reference R (T2 s0 ON vs adapter OFF = Titan-Sparse; each extra Titan-Sparse adapter vs OFF; each extra Supa adapter vs Supa 1.7) on a sealed confirmation draw minted after the dev selection of its surface-diversity setting, for each of the two graders g (the primary typed grader and the lenient extraction grader) and each selected category c (75 items): lost = items R answers correctly and A does not; gained = items A answers correctly and R does not; net = gained - lost (an unknown_twin item is correct only if both twins are). p = two-sided exact McNemar p-value (binomial on the lost + gained discordant items, 1/2); p_Holm = Holm-Bonferroni adjustment of p over the selected categories of that comparison and grader. Category c has a DETECTED LOSS under g iff lost > gained and p_Holm < 0.05, and EXCEEDS ITS MARGIN under g iff net < -11 (more than 11 of its 75 items lost net). The pooled net under g is the sum of net over the selected categories; it EXCEEDS ITS MARGIN iff it is below -0.04 x the number of selected items (-24 for 600 items). Comparison verdict: LOSS DETECTED iff some category has a detected loss under either grader; otherwise MARGIN EXCEEDED iff some category or the pooled net exceeds its margin under either grader; otherwise RETAINED. GO (the retention guard passes for arm A) iff RETAINED, reported as GO (k/m binding), where k of the m selected categories are binding, i.e. R answers at least the fewest items whose loss with no gain would fail the category alone (9 when 8 categories are selected); LOSS DETECTED and MARGIN EXCEEDED are NO-GO. Report-only, never gating: the per-category flag net < -5 of 75 (a point estimate, not a non-inferiority test), Titan-Sparse (OFF) vs Supa 1.7 and T2 s0 vs Supa 1.7 (core and cumulative drift, verdicts computed the same way), excluded categories, accuracies with Wilson 95% intervals, EOS-termination rates, abstention components, the operating characteristics, and the AUROC of the per-item first-token KL(R || A) for predicting correctness flips. The suite is templated and narrow: a RETAINED verdict is a floor, not a proof of general retention.

## Verdicts

| comparison | gating | verdict | GO | pooled net primary / lenient (margin) | categories over -11 | flags net < -5 (report-only) |
|---|---|---|---|---|---|---|
| t2s0_vs_off (t2s0 vs off) | yes | RETAINED | GO (5/5 binding) | +22 / +0 (-15) | - | - |
| x_bwf_vs_off (x_bwf vs off) | yes | RETAINED | GO (5/5 binding) | -3 / -5 (-15) | - | unit_conv, unknown_twin |
| x_bwpa_vs_off (x_bwpa vs off) | yes | RETAINED | GO (5/5 binding) | +9 / -10 (-15) | - | - |
| sparse_vs_supa (off vs supa) | report-only | LOSS DETECTED | NO-GO | -29 / -22 (-15) | unit_conv | unit_conv |
| t2s0_vs_supa (t2s0 vs supa) | report-only | LOSS DETECTED | NO-GO | -7 / -22 (-15) | unit_conv | unit_conv |

### t2s0_vs_off: primary grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 30/75 | 26/75 | 8 | 4 | -4 | 0.3877 | 0.7754 | no | no | yes |
| pct_ratio | [1] | 37/75 | 42/75 | 0 | 5 | +5 | 0.0625 | 0.1875 | no | no | yes |
| kv_recall | [3] | 52/75 | 63/75 | 0 | 11 | +11 | 0.0009766 | 0.004883 | no | no | yes |
| unknown_twin | [2] | 54/75 | 63/75 | 2 | 11 | +9 | 0.02246 | 0.08984 | no | no | yes |
| format_follow | [1] | 52/75 | 53/75 | 0 | 1 | +1 | 1 | 1 | no | no | yes |

### t2s0_vs_off: lenient grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 30/75 | 26/75 | 8 | 4 | -4 | 0.3877 | 1 | no | no | yes |
| pct_ratio | [1] | 37/75 | 42/75 | 0 | 5 | +5 | 0.0625 | 0.3125 | no | no | yes |
| kv_recall | [3] | 65/75 | 67/75 | 0 | 2 | +2 | 0.5 | 1 | no | no | yes |
| unknown_twin | [2] | 75/75 | 72/75 | 3 | 0 | -3 | 0.25 | 1 | no | no | yes |
| format_follow | [1] | 53/75 | 53/75 | 1 | 1 | +0 | 1 | 1 | no | no | yes |

Report-only: first-token KL(off || t2s0) mean 0.0761; AUROC for correctness flips 0.625 [0.5301476232283958, 0.7184160279377007] (n 375, flips 42).

### x_bwf_vs_off: primary grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 30/75 | 23/75 | 8 | 1 | -7 | 0.03906 | 0.1953 | no | no | yes |
| pct_ratio | [1] | 37/75 | 38/75 | 0 | 1 | +1 | 1 | 1 | no | no | yes |
| kv_recall | [3] | 52/75 | 59/75 | 7 | 14 | +7 | 0.1892 | 0.5677 | no | no | yes |
| unknown_twin | [2] | 54/75 | 48/75 | 7 | 1 | -6 | 0.07031 | 0.2812 | no | no | yes |
| format_follow | [1] | 52/75 | 54/75 | 1 | 3 | +2 | 0.625 | 1 | no | no | yes |

### x_bwf_vs_off: lenient grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 30/75 | 23/75 | 8 | 1 | -7 | 0.03906 | 0.1953 | no | no | yes |
| pct_ratio | [1] | 37/75 | 38/75 | 0 | 1 | +1 | 1 | 1 | no | no | yes |
| kv_recall | [3] | 65/75 | 68/75 | 2 | 5 | +3 | 0.4531 | 1 | no | no | yes |
| unknown_twin | [2] | 75/75 | 72/75 | 3 | 0 | -3 | 0.25 | 1 | no | no | yes |
| format_follow | [1] | 53/75 | 54/75 | 2 | 3 | +1 | 1 | 1 | no | no | yes |

Report-only: first-token KL(off || x_bwf) mean 0.2451; AUROC for correctness flips 0.759 [0.6704322988707082, 0.8383794888942612] (n 375, flips 43).

### x_bwpa_vs_off: primary grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 30/75 | 25/75 | 6 | 1 | -5 | 0.125 | 0.5 | no | no | yes |
| pct_ratio | [1] | 37/75 | 38/75 | 0 | 1 | +1 | 1 | 1 | no | no | yes |
| kv_recall | [3] | 52/75 | 61/75 | 5 | 14 | +9 | 0.06357 | 0.3178 | no | no | yes |
| unknown_twin | [2] | 54/75 | 58/75 | 3 | 7 | +4 | 0.3438 | 1 | no | no | yes |
| format_follow | [1] | 52/75 | 52/75 | 1 | 1 | +0 | 1 | 1 | no | no | yes |

### x_bwpa_vs_off: lenient grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 30/75 | 28/75 | 6 | 4 | -2 | 0.7539 | 1 | no | no | yes |
| pct_ratio | [1] | 37/75 | 38/75 | 0 | 1 | +1 | 1 | 1 | no | no | yes |
| kv_recall | [3] | 65/75 | 62/75 | 6 | 3 | -3 | 0.5078 | 1 | no | no | yes |
| unknown_twin | [2] | 75/75 | 70/75 | 5 | 0 | -5 | 0.0625 | 0.3125 | no | no | yes |
| format_follow | [1] | 53/75 | 52/75 | 2 | 1 | -1 | 1 | 1 | no | no | yes |

Report-only: first-token KL(off || x_bwpa) mean 0.2019; AUROC for correctness flips 0.613 [0.5129707550325366, 0.7131929396597764] (n 375, flips 39).

### sparse_vs_supa: primary grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 49/75 | 30/75 | 22 | 3 | -19 | 0.0001565 | 0.0007826 | YES | YES | yes |
| pct_ratio | [1] | 37/75 | 37/75 | 0 | 0 | +0 | 1 | 1 | no | no | yes |
| kv_recall | [3] | 56/75 | 52/75 | 6 | 2 | -4 | 0.2891 | 0.8672 | no | no | yes |
| unknown_twin | [2] | 58/75 | 54/75 | 4 | 0 | -4 | 0.125 | 0.5 | no | no | yes |
| format_follow | [1] | 54/75 | 52/75 | 2 | 0 | -2 | 0.5 | 1 | no | no | yes |

### sparse_vs_supa: lenient grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 50/75 | 30/75 | 22 | 2 | -20 | 3.588e-05 | 0.0001794 | YES | YES | yes |
| pct_ratio | [1] | 37/75 | 37/75 | 0 | 0 | +0 | 1 | 1 | no | no | yes |
| kv_recall | [3] | 66/75 | 65/75 | 1 | 0 | -1 | 1 | 1 | no | no | yes |
| unknown_twin | [2] | 75/75 | 75/75 | 0 | 0 | +0 | 1 | 1 | no | no | yes |
| format_follow | [1] | 54/75 | 53/75 | 2 | 1 | -1 | 1 | 1 | no | no | yes |

Report-only: first-token KL(supa || off) mean 0.0740; AUROC for correctness flips 0.501 [0.4066443707165864, 0.5954177993707593] (n 375, flips 39).

### t2s0_vs_supa: primary grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 49/75 | 26/75 | 28 | 5 | -23 | 6.619e-05 | 0.0003309 | YES | YES | yes |
| pct_ratio | [1] | 37/75 | 42/75 | 0 | 5 | +5 | 0.0625 | 0.1875 | no | no | yes |
| kv_recall | [3] | 56/75 | 63/75 | 0 | 7 | +7 | 0.01562 | 0.0625 | no | no | yes |
| unknown_twin | [2] | 58/75 | 63/75 | 2 | 7 | +5 | 0.1797 | 0.3594 | no | no | yes |
| format_follow | [1] | 54/75 | 53/75 | 1 | 0 | -1 | 1 | 1 | no | no | yes |

### t2s0_vs_supa: lenient grader

| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | binding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit_conv | [1] | 50/75 | 26/75 | 28 | 4 | -24 | 1.93e-05 | 9.651e-05 | YES | YES | yes |
| pct_ratio | [1] | 37/75 | 42/75 | 0 | 5 | +5 | 0.0625 | 0.25 | no | no | yes |
| kv_recall | [3] | 66/75 | 67/75 | 0 | 1 | +1 | 1 | 1 | no | no | yes |
| unknown_twin | [2] | 75/75 | 72/75 | 3 | 0 | -3 | 0.25 | 0.75 | no | no | yes |
| format_follow | [1] | 54/75 | 53/75 | 1 | 0 | -1 | 1 | 1 | no | no | yes |

Report-only: first-token KL(supa || t2s0) mean 0.1340; AUROC for correctness flips 0.676 [0.5984558982079082, 0.7539601822674549] (n 375, flips 55).

## Accuracy (primary grader; report-only)

| side | unit_conv | pct_ratio | kv_recall | unknown_twin | format_follow | EOS rate |
|---|---|---|---|---|---|---|
| off | 0.40 (30) | 0.49 (37) | 0.69 (65) | 0.72 (75) | 0.69 (53) | 1.000 |
| t2s0 | 0.35 (26) | 0.56 (42) | 0.84 (67) | 0.84 (72) | 0.71 (53) | 1.000 |
| x_bwf | 0.31 (23) | 0.51 (38) | 0.79 (68) | 0.64 (72) | 0.72 (54) | 1.000 |
| x_bwpa | 0.33 (28) | 0.51 (38) | 0.81 (62) | 0.77 (70) | 0.69 (52) | 1.000 |
| supa | 0.65 (50) | 0.49 (37) | 0.75 (66) | 0.77 (75) | 0.72 (54) | 1.000 |

Cells: primary accuracy (lenient-grader correct count).


<!-- synthetic dry run: NOT model output -->
# T2b-register dry run, rule revision 7 (synthetic sides; NOT model output)

| scenario | expected | verdict | ok |
|---|---|---|---|
| go | GO | GO | yes |
| filler_equals_trace | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) | yes |
| nothing_helps | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) - neither format helped (trace ~ filler ~ plain) | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) - neither format helped (trace ~ filler ~ plain) | yes |
| controls_lost | NO-GO (failed: G4 [swap]) | NO-GO (failed: G4 [swap]) | yes |
| in_dist_worse | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_sametens, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_sametens HS]; G2 [mix, mix HS]; G5 [conditional_in, mix_in]) | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_sametens, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_sametens HS]; G2 [mix, mix HS]; G5 [conditional_in, mix_in]) | yes |
| g3_only | NO-GO (failed: G3 [mix]) | NO-GO (failed: G3 [mix]) | yes |
| filler_invalid | INCONCLUSIVE: filler control invalid (G1-G5 hold, but filler format_ok < 0.95 on a primary) | INCONCLUSIVE: filler control invalid (G1-G5 hold, but filler format_ok < 0.95 on a primary) | yes |
| malformed_traces | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) - trace format failed (trace format_ok < 0.95 on a primary) | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) - trace format failed (trace format_ok < 0.95 on a primary) | yes |
| surface_locked | NO-GO (failed: G1 [conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS, cond_adv predicate HS]; G2 [mix HS]) | NO-GO (failed: G1 [conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS, cond_adv predicate HS]; G2 [mix HS]) | yes |
| coarse_comparison | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_sametens, cond_sametens predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_sametens, cond_sametens predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS]) | yes |
| tens_or_eq_plus1 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS]) | yes |
| tens_ge_not_eq | NO-GO (failed: G1 [cond_sametens, cond_sametens predicate, cond_sametens HS, cond_sametens predicate HS]) | NO-GO (failed: G1 [cond_sametens, cond_sametens predicate, cond_sametens HS, cond_sametens predicate HS]) | yes |
| true_rule_digitwise | GO | GO | yes |
| weak_plain_tens_or_eq_plus1 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS]) | yes |
| weak_plain_tens_ge_not_eq | NO-GO (failed: G1 [cond_sametens predicate, cond_sametens predicate HS]) | NO-GO (failed: G1 [cond_sametens predicate, cond_sametens predicate HS]) | yes |
| weak_plain_go | GO | GO | yes |
| obo_m1 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | yes |
| obo_0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | yes |
| obo_p1 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | yes |
| obo_m1_0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | yes |
| obo_m1_p1 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | yes |
| obo_0_p1 | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | yes |
| obo_m1_0_p1 | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]; G5 [conditional_in]) | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]; G5 [conditional_in]) | yes |
| weak_plain_obo_m1 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | yes |
| weak_plain_obo_0 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | yes |
| weak_plain_obo_p1 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | yes |
| weak_plain_obo_m1_0 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | yes |
| weak_plain_obo_m1_p1 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | yes |
| weak_plain_obo_0_p1 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | yes |
| weak_plain_obo_m1_0_p1 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]; G5 [conditional_in]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]; G5 [conditional_in]) | yes |
| weak_plain_lex_signed | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_magnitude | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_digitsum_sl | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_units | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_exc_m3 | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_exc_m2 | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_exc_p3 | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_exc_p4 | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| weak_plain_exc_m4 | GO | GO | yes |
| lex_signed | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | yes |
| engineered boundary_go | GO | GO | yes |
| engineered boundary_below | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_sametens, cond_sametens predicate, cond_adv predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS, cond_adv predicate HS]; G2 [mix, mix HS]; G3 [conditional]; G4 [swap]; G5 [conditional_in, mix_in]) | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_sametens, cond_sametens predicate, cond_adv predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS, cond_adv predicate HS]; G2 [mix, mix HS]; G3 [conditional]; G4 [swap]; G5 [conditional_in, mix_in]) | yes |
| engineered boundary_hs_cond_oor_val | NO-GO (failed: G1 [cond_oor_val HS]) | NO-GO (failed: G1 [cond_oor_val HS]) | yes |
| engineered boundary_hs_cond_sametens | NO-GO (failed: G1 [cond_sametens HS]) | NO-GO (failed: G1 [cond_sametens HS]) | yes |
| engineered boundary_in_cond_sametens | NO-GO (failed: G1 [cond_sametens]) | NO-GO (failed: G1 [cond_sametens]) | yes |
| engineered boundary_hs_cond_oor_edge | NO-GO (failed: G1 [cond_oor_edge HS]) | NO-GO (failed: G1 [cond_oor_edge HS]) | yes |
| engineered boundary_in_cond_oor_edge | NO-GO (failed: G1 [cond_oor_edge]) | NO-GO (failed: G1 [cond_oor_edge]) | yes |
| engineered boundary_in_pred_cond_oor_edge | NO-GO (failed: G1 [cond_oor_edge predicate]) | NO-GO (failed: G1 [cond_oor_edge predicate]) | yes |
| engineered boundary_hs_pred_cond_oor_edge | NO-GO (failed: G1 [cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate HS]) | yes |
| engineered boundary_in_pred_cond_sametens | NO-GO (failed: G1 [cond_sametens predicate]) | NO-GO (failed: G1 [cond_sametens predicate]) | yes |
| engineered boundary_hs_pred_cond_sametens | NO-GO (failed: G1 [cond_sametens predicate HS]) | NO-GO (failed: G1 [cond_sametens predicate HS]) | yes |
| engineered boundary_in_pred_cond_adv | NO-GO (failed: G1 [cond_adv predicate]) | NO-GO (failed: G1 [cond_adv predicate]) | yes |
| engineered boundary_hs_pred_cond_adv | NO-GO (failed: G1 [cond_adv predicate HS]) | NO-GO (failed: G1 [cond_adv predicate HS]) | yes |
| engineered boundary_dp3 | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) | STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format) | yes |

G1 parts of the rule scenarios (exact; plain = the scenario's plain side: 'chance' = exactly 1/2 on every G1 part, 'weak_plain' = 0 on every out-of-range row, 'go' = the coin-flip plain):

| scenario | trace conditional | cond_oor_val | cond_oor_edge | cond_oor_edge predicate | cond_sametens | cond_sametens predicate | cond_adv predicate | plain cond_oor_val / cond_oor_edge / cond_sametens | shortcut agreement |
|---|---|---|---|---|---|---|---|---|---|
| go | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 37/100 / 43/80 / 49/100 | - |
| true_rule_digitwise | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | - |
| tens_or_eq_plus1 | 1 | 1 | 63/80 | 63/80 | 1/2 | 1/2 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| tens_ge_not_eq | 1 | 1 | 4/5 | 4/5 | 1/2 | 1/2 | 99/100 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| coarse_comparison | 253/300 | 53/100 | 11/20 | 11/20 | 1/2 | 1/2 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| weak_plain_tens_or_eq_plus1 | 1 | 1 | 63/80 | 63/80 | 1/2 | 1/2 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_tens_ge_not_eq | 1 | 1 | 4/5 | 4/5 | 1/2 | 1/2 | 99/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_go | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 / 0 / 0 | - |
| obo_m1 | 1 | 1 | 3/4 | 3/4 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| obo_0 | 5/6 | 1/2 | 3/4 | 3/4 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| obo_p1 | 5/6 | 1/2 | 3/4 | 3/4 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| obo_m1_0 | 5/6 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| obo_m1_p1 | 5/6 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| obo_0_p1 | 2/3 | 0 | 1/2 | 1/2 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| obo_m1_0_p1 | 2/3 | 0 | 1/4 | 1/4 | 1 | 1 | 1 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| weak_plain_obo_m1 | 1 | 1 | 3/4 | 3/4 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_obo_0 | 5/6 | 1/2 | 3/4 | 3/4 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_obo_p1 | 5/6 | 1/2 | 3/4 | 3/4 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_obo_m1_0 | 5/6 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_obo_m1_p1 | 5/6 | 1/2 | 1/2 | 1/2 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_obo_0_p1 | 2/3 | 0 | 1/2 | 1/2 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_obo_m1_0_p1 | 2/3 | 0 | 1/4 | 1/4 | 1 | 1 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| lex_signed | 41/50 | 99/100 | 79/80 | 79/80 | 1 | 1 | 67/100 | 1/2 / 1/2 / 1/2 | SHORTCUT_CONSISTENT |
| weak_plain_lex_signed | 41/50 | 99/100 | 79/80 | 79/80 | 1 | 1 | 67/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_magnitude | 287/300 | 1 | 1 | 1 | 1 | 1 | 67/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_digitsum_sl | 283/300 | 49/50 | 73/80 | 73/80 | 1 | 1 | 67/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_units | 203/300 | 97/100 | 9/10 | 9/10 | 1 | 1 | 1/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_exc_m3 | 1 | 1 | 1 | 1 | 22/25 | 22/25 | 79/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_exc_m2 | 1 | 1 | 1 | 1 | 41/50 | 41/50 | 79/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_exc_p3 | 1 | 1 | 1 | 1 | 43/50 | 43/50 | 79/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_exc_p4 | 1 | 1 | 1 | 1 | 22/25 | 22/25 | 79/100 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |
| weak_plain_exc_m4 | 1 | 1 | 1 | 1 | 22/25 | 22/25 | 1 | 0 / 0 / 0 | SHORTCUT_CONSISTENT |

Off-by-one family (decision 2): share of each G1-gated stratum's rows that contradicts the variant, and its verdict in the full dry run:

| variant | form | conditional | cond_oor_val | cond_oor_edge | cond_sametens | verdict, chance plain | verdict, weak plain |
|---|---|---|---|---|---|---|---|
| obo_m1 | `x > t or x == t-1` | 0 | 0 | 1/4 | 0 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) |
| obo_0 | `x >= t` | 1/6 | 1/2 | 1/4 | 0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) |
| obo_p1 | `x > t+1` | 1/6 | 1/2 | 1/4 | 0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) |
| obo_m1_0 | `x >= t-1` | 1/6 | 1/2 | 1/2 | 0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) |
| obo_m1_p1 | `(x > t or x == t-1) and x != t+1` | 1/6 | 1/2 | 1/2 | 0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) |
| obo_0_p1 | `x == t or x > t+1` | 1/3 | 1 | 1/2 | 0 | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]) |
| obo_m1_0_p1 | `x in {t-1, t} or x > t+1` | 1/3 | 1 | 3/4 | 0 | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS]; G5 [conditional_in]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, cond_oor_val HS, cond_oor_edge predicate HS]; G5 [conditional_in]) |

Enumeration over the extended shortcut family H: 3519 predicates (3483 enumerated + 36 extension; 1004 distinct on the dev conditional rows), G1 decided by verdict() on exact counts, both surfaces. Target: no member of H that differs from the comparison passes G1 on either surface, under any baseline; met: False.

- plain baseline `chance`: passed revision-5 G1 71 (41 not the true rule, 15 of them in the extension); pass revision-6 G1 31, of which NOT the true rule on the domain 1 in 1 classes; caught by revision 6 (cond_adv) 40; off-by-one family members passing 0; extension members passing 1.
  - still passes: `exception d=-4: x>t or x==t-4` (+0 equivalent on the dev rows; flips at d = [-4]; wrong on 0 cond_adv rows); acc {'primary_conditional': '1', 'cond_oor_val': '1', 'cond_oor_edge': '1', 'cond_sametens': '22/25', 'cond_oor_edge_pred': '1', 'cond_sametens_pred': '22/25', 'cond_adv_pred': '1'}
- plain baseline `dryrun_plainlike`: passed revision-5 G1 114 (84 not the true rule, 16 of them in the extension); pass revision-6 G1 31, of which NOT the true rule on the domain 1 in 1 classes; caught by revision 6 (cond_adv) 83; off-by-one family members passing 0; extension members passing 1.
  - still passes: `exception d=-4: x>t or x==t-4` (+0 equivalent on the dev rows; flips at d = [-4]; wrong on 0 cond_adv rows); acc {'primary_conditional': '1', 'cond_oor_val': '1', 'cond_oor_edge': '1', 'cond_sametens': '22/25', 'cond_oor_edge_pred': '1', 'cond_sametens_pred': '22/25', 'cond_adv_pred': '1'}
- plain baseline `weak_plain`: passed revision-5 G1 156 (126 not the true rule, 16 of them in the extension); pass revision-6 G1 31, of which NOT the true rule on the domain 1 in 1 classes; caught by revision 6 (cond_adv) 125; off-by-one family members passing 0; extension members passing 1.
  - still passes: `exception d=-4: x>t or x==t-4` (+0 equivalent on the dev rows; flips at d = [-4]; wrong on 0 cond_adv rows); acc {'primary_conditional': '1', 'cond_oor_val': '1', 'cond_oor_edge': '1', 'cond_sametens': '22/25', 'cond_oor_edge_pred': '1', 'cond_sametens_pred': '22/25', 'cond_adv_pred': '1'}
- survivor `exception d=-4: x>t or x==t-4`: an isolated exception at other - threshold = -4: it is wrong only on rows at that distance; the predicate-gated strata have {'cond_oor_edge': 0, 'cond_sametens': 12, 'cond_adv': 0} such rows (it needs > 1/5 of a stratum's rows there to fail the 0.80 bar); no 100-row branch-balanced stratum can catch the exceptions at -4, -3 and -2 together (3 x 21 = 63 > 50 ok-false rows), and cond_adv catches -3 and -2 (21 rows each); no relative clause catches it.

Members of the extension (revision 6): wrong rows on cond_adv and G1 failures against the weak plain arm:

| member | cond_adv wrong | passed revision 5's G1 (weak plain) | G1 failed parts (weak plain) |
|---|---|---|---|
| `lex signed: str(x)>str(t)` | 33 | yes | cond_adv predicate, cond_adv predicate HS |
| `lex unsigned: str(\|x\|)>str(\|t\|)` | 66 | yes | cond_adv predicate, cond_adv predicate HS |
| `lex signed: str(x)>=str(t)` | 33 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `lex unsigned: str(\|x\|)>=str(\|t\|)` | 61 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `units plain: units(\|x\|)>units(\|t\|)` | 99 | yes | cond_adv predicate, cond_adv predicate HS |
| `units sign-aware: >` | 66 | yes | cond_adv predicate, cond_adv predicate HS |
| `units length-aware: >` | 66 | yes | cond_adv predicate, cond_adv predicate HS |
| `units sign+length-aware: >` | 33 | yes | cond_adv predicate, cond_adv predicate HS |
| `units plain: units(\|x\|)>=units(\|t\|)` | 94 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `units sign-aware: >=` | 66 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `units length-aware: >=` | 61 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `units sign+length-aware: >=` | 33 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `tens plain: tens(\|x\|)>tens(\|t\|)` | 33 | no | cond_oor_edge predicate, cond_sametens predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_sametens predicate HS, cond_adv predicate HS |
| `tens sign-aware: >` | 0 | no | cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS |
| `tens length-aware: >` | 33 | no | cond_oor_edge predicate, cond_sametens predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_sametens predicate HS, cond_adv predicate HS |
| `tens sign+length-aware: >` | 0 | no | cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS |
| `tens plain: tens(\|x\|)>=tens(\|t\|)` | 1 | no | cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS |
| `tens sign-aware: >=` | 1 | no | cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS |
| `tens length-aware: >=` | 1 | no | cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS |
| `tens sign+length-aware: >=` | 1 | no | cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge predicate HS, cond_sametens predicate HS |
| `digitsum plain: digitsum(\|x\|)>digitsum(\|t\|)` | 91 | yes | cond_adv predicate, cond_adv predicate HS |
| `digitsum sign-aware: >` | 58 | yes | cond_adv predicate, cond_adv predicate HS |
| `digitsum length-aware: >` | 66 | yes | cond_adv predicate, cond_adv predicate HS |
| `digitsum sign+length-aware: >` | 33 | yes | cond_adv predicate, cond_adv predicate HS |
| `digitsum plain: digitsum(\|x\|)>=digitsum(\|t\|)` | 86 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `digitsum sign-aware: >=` | 58 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `digitsum length-aware: >=` | 61 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `digitsum sign+length-aware: >=` | 33 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `magnitude: \|x\|>\|t\|` | 33 | yes | cond_adv predicate, cond_adv predicate HS |
| `magnitude: \|x\|>=\|t\|` | 28 | no | cond_oor_edge predicate, cond_adv predicate, cond_oor_edge predicate HS, cond_adv predicate HS |
| `exception d=-4: x>t or x==t-4` | 0 | yes | - (passes) |
| `exception d=-3: x>t or x==t-3` | 21 | yes | cond_adv predicate, cond_adv predicate HS |
| `exception d=-2: x>t or x==t-2` | 21 | yes | cond_adv predicate, cond_adv predicate HS |
| `exception d=+2: x>t and x!=t+2` | 0 | no | cond_oor_edge predicate, cond_oor_edge predicate HS |
| `exception d=+3: x>t and x!=t+3` | 21 | yes | cond_adv predicate, cond_adv predicate HS |
| `exception d=+4: x>t and x!=t+4` | 21 | yes | cond_adv predicate, cond_adv predicate HS |

cond_adv targets: ok True; broad revision-5 survivors 119 wrong on >= 33 rows; exceptions caught: `x>t-3 | and x!=t-1, and x!=t` 21, `x>t-3 | and x!=t-1, and x!=t, or x==t+1` 21, `exception d=-3: x>t or x==t-3` 21, `exception d=-2: x>t or x==t-2` 21, `exception d=+3: x>t and x!=t+3` 21, `exception d=+4: x>t and x!=t+4` 21; not caught: `exception d=-4: x>t or x==t-4` 0; the operator's 40-row target met by 2444 of 3489 predicates (infeasible for all: five members with pairwise-disjoint error sets).

OVERALL headline (revision 7: it carries the report-only flag and the LIMITS pointer whenever the flag is set; otherwise revision 6's headline):

- weak_plain_exc_m4: `SCREEN (seed 0): GO | SHORTCUT_CONSISTENT in s0 (report-only shortcut-agreement flag; it never changes the verdict: a member of H agrees with the trace's stated cmp.ok at least as well as the true predicate) | read the LIMITS quoted with the verdict (G1 excludes only the members of H, and not the exception 'other > threshold or other == threshold - 4'; heuristics outside H are not excluded)`
- lex_signed: `SCREEN (seed 0): NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) | SHORTCUT_CONSISTENT in s0 (report-only shortcut-agreement flag; it never changes the verdict: a member of H agrees with the trace's stated cmp.ok at least as well as the true predicate) | read the LIMITS quoted with the verdict (G1 excludes only the members of H, and not the exception 'other > threshold or other == threshold - 4'; heuristics outside H are not excluded)`
- go: `SCREEN (seed 0): GO`
- multi-seed, flag in s1 only: `GO in every analysed seed | SHORTCUT_CONSISTENT in s1 (report-only shortcut-agreement flag; it never changes the verdict: a member of H agrees with the trace's stated cmp.ok at least as well as the true predicate) | read the LIMITS quoted with the verdict (G1 excludes only the members of H, and not the exception 'other > threshold or other == threshold - 4'; heuristics outside H are not excluded)`
- multi-seed, no flag: `GO in every analysed seed`
- multi-seed, flag in both, one NO-GO: `NOT GO in every analysed seed | SHORTCUT_CONSISTENT in s0, s1 (report-only shortcut-agreement flag; it never changes the verdict: a member of H agrees with the trace's stated cmp.ok at least as well as the true predicate) | read the LIMITS quoted with the verdict (G1 excludes only the members of H, and not the exception 'other > threshold or other == threshold - 4'; heuristics outside H are not excluded)`


# T2b-register screen results (s0; training variant sd)

**Overall:** SCREEN (seed 0): GO [SYNTHETIC DRY RUN]

**Limits (quoted with every statement of the verdict):**

- G1 excludes the members of the enumerated extended shortcut family H (3,519 predicates of other and threshold; scripts/t2br_shortcuts.py, DESIGN.md section 15) that differ from the comparison, as the enumeration over H against three plain baselines documents (DRYRUN.json shortcut_enumeration), EXCEPT the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part (on any 100-row branch-balanced stratum one of the exceptions at other - threshold = -4, -3, -2 passes the 0.80 predicate bar; cond_adv catches -3 and -2).
- Heuristics OUTSIDE H are NOT excluded, and natural ones pass every G1 part (the revision-6 independent check, confirmed with verdict() on exact counts, both surfaces; DESIGN.md section 14): isolated single-distance exceptions at |other - threshold| >= 5 (H has them only at 2..4; e.g. 'other > threshold except at other - threshold = +5') and flipped distance bands (e.g. 5..9); three-digit-threshold slips (the hundreds digit dropped, only the first two digits read, a threshold above 96 reduced modulo 97, string comparison only against three-digit thresholds, always adding when the threshold is above 96); a register of 0 never counted above a negative threshold; the sign of a threshold at or below -10 dropped; a circular comparison ((other - threshold) mod 97 in 1..48); string comparison only for one-digit thresholds or one-digit registers; guessing on the far rows (|other - threshold| > 15) by string comparison or a prior (other > 48, threshold < 48), and, against a plain arm well below chance on the pooled conditional (the dry run's coin-flip plain at 5/12, or a weak plain arm), also by always adding, always subtracting or the units digit; and any rule that departs from the comparison only on cells that no G1 stratum contains (e.g. in-range cross-decade cells at |other - threshold| 5..15, decided by the units digit or with the branch flipped). The report-only shortcut-agreement flag compares the members of H only, so its absence does not rule these out.
- cond_adv catches its targets only at narrow margins: the exceptions at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 of its 100 rows (79/100, one row below the 0.80 bar), and the signed string comparison, the magnitude comparison and the sign+length-aware digit-sum and units comparisons on exactly 33 (67/100). A trace that follows a caught member on only part of its rows can pass: following it everywhere except on any 1 of those 21 rows, or any 13 of those 33, passes G1; and a trace that departs from the comparison on at most 1/5 of each predicate-gated stratum's rows passes the predicate clauses.
- Per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice: a held-out-surface loss cannot be attributed to one factor.
- A GO is a seed-0 screen result: the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

Pre-stated rule, revision 7 (verbatim):

> PRE-STATED SCREEN GO RULE, REVISION 7 (T2b-register; supersedes revisions 1-6, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on the three identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; none is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows; 'cond_adv' (revision 6) = 100 rows, 50 ok true and 50 ok false, drawn by seeded rejection sampling over valid conditional rows (other 0..96, threshold -40..136, |value| 12..40, neither branch wraps) to fixed quotas of 15 design types: 13 signature classes of the extended shortcut family H (sets of (other, threshold) cells on which exactly the same members of H are wrong), one of them split by digit lengths, covering sign-crossing cells, cross-length cells (1 vs 2, 2 vs 1, 1 vs 3 and 2 vs 3 digits), cross-decade cells whose units comparison disagrees with the tens comparison and cells at other - threshold = -3, -2, +3, +4; H (scripts/t2br_shortcuts.py, DESIGN.md section 15; 3,519 predicates) = the enumerated digit and off-by-one family with its modifiers at other = threshold - 1, threshold and threshold + 1 (3,483 predicates, the branch-blind constants included) + lexicographic string comparison with and without sign handling, digit-sum, units-only and tens-only comparisons with sign- and length-aware variants, magnitude-only |other| > |threshold| and the isolated single-distance exceptions at |other - threshold| = 2..4; on cond_adv every member of H that differs from the comparison somewhere on that domain and passes revision 5's G1 against a plain arm at 0 is wrong on at least 33 rows, except the isolated distance exceptions, of which those at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 (the operator's target, every member of H wrong on at least 40 rows, is infeasible on any 100-row branch-balanced stratum: five members of H have pairwise-disjoint error sets). PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone, on cond_sametens alone AND on cond_adv alone (cond_adv has no trace - plain clause), each both in-surface and on the held-out surface (the predicate clauses have no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let through a member of H that they catch; G1 excludes the members of the enumerated family H that differ from the comparison, except the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part, as documented by the enumeration over H against three plain baselines (DRYRUN.json shortcut_enumeration); it does NOT exclude heuristics outside H, and natural ones pass every G1 part (LIMITS)); (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; the shortcut-agreement analysis (the trace side's cmp.ok against the env's predicate and against every member of H over all conditional rows of the dev bank, in each rendering: 'SHORTCUT_CONSISTENT' if a member of H that differs from the predicate on those rows agrees with the trace's cmp.ok at least as well as the predicate does; it never changes the verdict; it is quoted with every statement of the verdict, so with any GO, and whenever it is set the OVERALL headline carries it beside the verdict, with a pointer to the LIMITS); per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). LIMITS, quoted with every statement of the verdict (t2br_common.ALWAYS_QUOTED_LIMITS; DESIGN.md section 14): (1) G1 excludes the members of the enumerated family H that differ from the comparison EXCEPT the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part (on any 100-row branch-balanced stratum one of the exceptions at other - threshold = -4, -3, -2 passes the 0.80 bar); (2) heuristics OUTSIDE H are NOT excluded, and natural ones pass every G1 part (the revision-6 independent check, confirmed by verdict() on exact counts): isolated single-distance exceptions at |other - threshold| >= 5 (H has them only at 2..4) and flipped distance bands (e.g. 5..9); three-digit-threshold slips (the hundreds digit dropped, only the first two digits read, a threshold above 96 reduced modulo 97, string comparison only against three-digit thresholds, always adding when the threshold is above 96); a register of 0 never counted above a negative threshold; the sign of a threshold at or below -10 dropped; a circular comparison ((other - threshold) mod 97 in 1..48); string comparison only for one-digit thresholds or one-digit registers; guessing on the far rows (|other - threshold| > 15) by string comparison or a prior (other > 48, threshold < 48), and, against a plain arm well below chance on the pooled conditional (the dry run's coin-flip plain at 5/12, or a weak plain arm), also by always adding, always subtracting or the units digit; any rule that departs from the comparison only on cells that no G1 stratum contains (e.g. in-range cross-decade cells at |other - threshold| 5..15, decided by the units digit or with the branch flipped); the report-only shortcut-agreement flag compares the members of H only, so its absence does not rule these out; (3) cond_adv catches its targets only at narrow margins: the exceptions at other - threshold = -3, -2, +3, +4 are wrong on exactly 21 of its 100 rows (79/100, one row below the bar) and the signed string, magnitude, and sign+length-aware digit-sum and units comparisons on exactly 33 (67/100), so a trace that follows a caught member on only part of its rows can pass (following it everywhere except on any 1 of those 21 rows, or any 13 of those 33, passes G1), and a trace that departs from the comparison on at most 1/5 of each predicate-gated stratum's rows passes the predicate clauses; (4) per-factor held-out renderings are omitted; (5) a GO is a seed-0 screen result. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

## Seed s0: GO

**Quoted with the verdict** (a GO): shortcut agreement (report-only; never changes the verdict): no member of H agrees with the trace's cmp.ok as well as the true predicate - in-surface: true predicate 860/860, best member of H 'exception d=-4: x>t or x==t-4' 846/860; held-out surface: true predicate 580/580, best member of H 'exception d=-4: x>t or x==t-4' 568/580.

- in: rows 860, stated ok 860, true predicate 860, top members of H: `exception d=-4: x>t or x==t-4` 846; `exception d=+4: x>t and x!=t+4` 825; `exception d=-3: x>t or x==t-3` 824; `exception d=+2: x>t and x!=t+2` 821; `x>t+2 | and x!=t, or x==t+1` 821 (0 members equal the predicate on these rows; flag: none)
- hs: rows 580, stated ok 580, true predicate 580, top members of H: `exception d=-4: x>t or x==t-4` 568; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>=units(t)) | or x==t-1, and x!=t` 560; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>=units(t)) | or x==t-1, and x!=t, or x==t+1` 560; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>units(t)) | or x==t-1` 560; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>units(t)) | or x==t-1, and x!=t` 560 (0 members equal the predicate on these rows; flag: none)

| condition | holds | failed parts |
|---|---|---|
| G1 trace-plain >= +0.25 on conditional (OOR pooled), on cond_oor_val, cond_oor_edge and cond_sametens alone, and trace predicate accuracy >= 0.80 on cond_oor_edge, cond_sametens and cond_adv, in-surface and held-out surface | yes | - |
| G2 trace-plain >= +0.30 on mix (OOR), in-surface and held-out surface | yes | - |
| G3 filler below trace: trace-filler >= +0.10 on each in-surface primary (4b filler) | yes | - |
| G4 controls: trace ON-OFF >= -0.05 on swap and add_nowrap | yes | - |
| G5 in-distribution non-inferiority: trace-plain >= -0.05 on conditional_in and mix_in | yes | - |
| V filler side valid: filler in-surface format_ok >= 0.95 on each primary | yes | - |

Validity (in-surface): trace format_ok {'primary_conditional': 1.0, 'primary_mix': 1.0}, filler format_ok {'primary_conditional': 1.0, 'primary_mix': 1.0} (bar 0.95 on each primary).

Trace predicate accuracy (G1's absolute clause, bar 0.80; strict parser, cmp keys x, thr, ok in order, boolean ok equal to the env's predicate): cond_oor_edge 1.000 in-surface / 1.000 held-out; cond_sametens 1.000 in-surface / 1.000 held-out; cond_adv 1.000 in-surface / 1.000 held-out.

| component | surface | contrast | outcome | value (exact) | bar | holds | SE | within 1 SE |
|---|---|---|---|---|---|---|---|---|
| G1 | in | trace-plain | primary_conditional | +0.5833 (7/12) | >= +0.25 | yes | 0.02722 |  |
| G1 | in | trace-plain | cond_oor_val | +0.6300 (63/100) | >= +0.25 | yes | 0.040817 |  |
| G1 | in | trace-plain | cond_oor_edge | +0.4625 (37/80) | >= +0.25 | yes | 0.047833 |  |
| G1 | in | trace | cond_oor_edge_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | in | trace-plain | cond_sametens | +0.5100 (51/100) | >= +0.25 | yes | 0.045919 |  |
| G1 | in | trace | cond_sametens_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | in | trace | cond_adv_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | hs | trace-plain | primary_conditional | +0.5833 (7/12) | >= +0.25 | yes | 0.02722 |  |
| G1 | hs | trace-plain | cond_oor_val | +0.6300 (63/100) | >= +0.25 | yes | 0.040817 |  |
| G1 | hs | trace-plain | cond_oor_edge | +0.4625 (37/80) | >= +0.25 | yes | 0.047833 |  |
| G1 | hs | trace | cond_oor_edge_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | hs | trace-plain | cond_sametens | +0.5100 (51/100) | >= +0.25 | yes | 0.045919 |  |
| G1 | hs | trace | cond_sametens_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | hs | trace | cond_adv_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G2 | in | trace-plain | primary_mix | +0.5000 (1/2) | >= +0.30 | yes | 0.051021 |  |
| G2 | hs | trace-plain | primary_mix | +0.5000 (1/2) | >= +0.30 | yes | 0.051021 |  |
| G3 | in | trace-filler | primary_conditional | +0.5833 (7/12) | >= +0.10 | yes | 0.02722 |  |
| G3 | in | trace-filler | primary_mix | +0.5000 (1/2) | >= +0.10 | yes | 0.051021 |  |
| G4 | in | trace-off | swap | +0.0000 (0) | >= -0.05 | yes | 0.0 |  |
| G4 | in | trace-off | add_nowrap | +0.0000 (0) | >= -0.05 | yes | 0.0 |  |
| G5 | in | trace-plain | conditional_in | +0.5045 (111/220) | >= -0.05 | yes | 0.031608 |  |
| G5 | in | trace-plain | mix_in | +0.0000 (0) | >= -0.05 | yes | 0.0 |  |
| DP3 | in | trace-filler | primary_conditional | +0.5833 (7/12) | < +0.10 | NO | 0.02722 |  |
| DP3 | in | trace-filler | primary_mix | +0.5000 (1/2) | < +0.10 | NO | 0.051021 |  |

In-surface (Titan's verbatim frame):

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.417 | 0.417 | 1.000 | 0.417 | +0.583 [+0.530, +0.637] | +0.583 [+0.530, +0.637] |
| primary_mix | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| conditional_in | 0.485 | 0.495 | 1.000 | 0.495 | +0.504 [+0.441, +0.565] | +0.504 [+0.441, +0.565] |
| mix_in | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mix | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| wrap | 0.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| cond_xm1 | 0.517 | 0.517 | 1.000 | 0.517 | +0.483 [+0.350, +0.617] | +0.483 [+0.350, +0.617] |
| cond_x0 | 0.483 | 0.517 | 1.000 | 0.517 | +0.483 [+0.350, +0.617] | +0.483 [+0.350, +0.617] |
| cond_xp1 | 0.483 | 0.483 | 1.000 | 0.483 | +0.517 [+0.383, +0.650] | +0.517 [+0.383, +0.650] |
| cond_mid | 0.470 | 0.480 | 1.000 | 0.480 | +0.520 [+0.410, +0.630] | +0.520 [+0.410, +0.630] |
| swap | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| add_nowrap | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mix_oor | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| wrap_oor | 0.000 | 0.000 | 1.000 | 0.000 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] |
| cond_oor_val | 0.370 | 0.370 | 1.000 | 0.370 | +0.630 [+0.550, +0.710] | +0.630 [+0.550, +0.710] |
| cond_oor_thr | 0.420 | 0.420 | 1.000 | 0.420 | +0.580 [+0.490, +0.670] | +0.580 [+0.490, +0.670] |
| cond_far | 0.460 | 0.460 | 1.000 | 0.460 | +0.540 [+0.450, +0.640] | +0.540 [+0.450, +0.640] |
| cond_sametens | 0.490 | 0.490 | 1.000 | 0.490 | +0.510 [+0.420, +0.600] | +0.510 [+0.420, +0.600] |
| cond_oor_edge | 0.537 | 0.537 | 1.000 | 0.537 | +0.463 [+0.375, +0.562] | +0.463 [+0.375, +0.562] |
| cond_adv | 0.520 | 0.520 | 1.000 | 0.520 | +0.480 [+0.375, +0.577] | +0.480 [+0.375, +0.577] |

Held-out surface (paraphrase 4 + field names 4 + held-out key orders), the primary rows and the identification strata:

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.417 | 0.417 | 1.000 | 0.417 | +0.583 [+0.530, +0.637] | +0.583 [+0.530, +0.637] |
| cond_oor_val | 0.370 | 0.370 | 1.000 | 0.370 | +0.630 [+0.550, +0.710] | +0.630 [+0.550, +0.710] |
| cond_oor_thr | 0.420 | 0.420 | 1.000 | 0.420 | +0.580 [+0.490, +0.670] | +0.580 [+0.490, +0.670] |
| cond_far | 0.460 | 0.460 | 1.000 | 0.460 | +0.540 [+0.450, +0.640] | +0.540 [+0.450, +0.640] |
| primary_mix | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| mix_oor | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| cond_sametens | 0.490 | 0.490 | 1.000 | 0.490 | +0.510 [+0.420, +0.600] | +0.510 [+0.420, +0.600] |
| cond_oor_edge | 0.537 | 0.537 | 1.000 | 0.537 | +0.463 [+0.375, +0.562] | +0.463 [+0.375, +0.562] |
| cond_adv | 0.520 | 0.520 | 1.000 | 0.520 | +0.480 [+0.375, +0.577] | +0.480 [+0.375, +0.577] |

Surface audit (held-out minus in-surface, paired by row; rows right only in-surface / only held-out):

| outcome | off | plain | trace | filler |
|---|---|---|---|---|
| primary_conditional | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_oor_val | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_oor_thr | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_far | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| primary_mix | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| mix_oor | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_sametens | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_oor_edge | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_adv | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |

Controls (a G4 loss shared by every trained arm is a property of the data: every training add wraps):

- swap: trace-off +0.000, plain-off +0.000, filler-off +0.000, trace-plain +0.000
- add_nowrap: trace-off +0.000, plain-off +0.000, filler-off +0.000, trace-plain +0.000

Trace fields (trace side, in-surface):

| stratum | format | predicate (pred_ok) | cmp_ok | ok_ok | calc_ok | after follows calc |
|---|---|---|---|---|---|---|
| mix | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| wrap | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| cond_xm1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_x0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_xp1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_mid | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| swap | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| add_nowrap | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| mix_oor | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| wrap_oor | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| cond_oor_val | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_oor_thr | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_far | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_sametens | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_oor_edge | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_adv | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Filler residual covariate (reported, not gating; amendment 4b: the fill is constant per action kind, so its length carries no answer information; residual = filler tokens - that row's trace tokens):

| cell | n | residual mean / median / range | filler acc, residual <= median (n) | > median (n) | held-out: <= median | > median |
|---|---|---|---|---|---|---|
| cond_oor_val | 100 | 4.19 / 4.0 / [3, 7] | 0.3855 (83) | 0.2941 (17) | 0.3855 | 0.2941 |
| cond_oor_thr | 100 | 3.91 / 4.0 / [3, 7] | 0.4625 (80) | 0.25 (20) | 0.4625 | 0.25 |
| cond_far | 100 | 4.38 / 4.0 / [4, 7] | 0.5143 (70) | 0.3333 (30) | 0.5143 | 0.3333 |
| mix_oor | 100 | 3.61 / 4.0 / [3, 6] | 0.4792 (96) | 1.0 (4) | 0.4792 | 1.0 |
| cond_sametens | 100 | 4.48 / 4.0 / [4, 6] | 0.4605 (76) | 0.5833 (24) | 0.4605 | 0.5833 |
| cond_oor_edge | 80 | 4.4875 / 4.0 / [4, 8] | 0.5484 (62) | 0.5 (18) | 0.5484 | 0.5 |
| cond_adv | 100 | 4.95 / 5.0 / [3, 8] | 0.5441 (68) | 0.4688 (32) | 0.5441 | 0.4688 |
| primary_conditional | 300 | 4.16 / 4.0 / [3, 7] | 0.4506 (233) | 0.2985 (67) | 0.4506 | 0.2985 |
| primary_mix | 100 | 3.61 / 4.0 / [3, 6] | 0.4792 (96) | 1.0 (4) | 0.4792 | 1.0 |

Branch choice read from 'after' (correct / wrong / neither), conditional strata (in-surface):

- off: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_mid 0.48/0.52/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0; cond_far 0.49/0.51/0.0; cond_sametens 0.49/0.51/0.0; cond_oor_edge 0.5375/0.4625/0.0; cond_adv 0.52/0.48/0.0
- plain: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_mid 0.48/0.52/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0; cond_far 0.49/0.51/0.0; cond_sametens 0.49/0.51/0.0; cond_oor_edge 0.5375/0.4625/0.0; cond_adv 0.52/0.48/0.0
- trace: cond_xm1 1.0/0.0/0.0; cond_x0 1.0/0.0/0.0; cond_xp1 1.0/0.0/0.0; cond_mid 1.0/0.0/0.0; cond_oor_val 1.0/0.0/0.0; cond_oor_thr 1.0/0.0/0.0; cond_far 1.0/0.0/0.0; cond_sametens 1.0/0.0/0.0; cond_oor_edge 1.0/0.0/0.0; cond_adv 1.0/0.0/0.0
- filler: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_mid 0.48/0.52/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0; cond_far 0.49/0.51/0.0; cond_sametens 0.49/0.51/0.0; cond_oor_edge 0.5375/0.4625/0.0; cond_adv 0.52/0.48/0.0

Chains (k=4; normalised regret / exact; mean [95% CI]):

| stratum | side | regret k4 | exact k4 | product-law predicted k4 | observed |
|---|---|---|---|---|---|
| CH_IN | off | 0.296 [+0.254, +0.339] | 0.234 | 0.2287 | 0.2344 |
| CH_IN | plain | 0.044 [+0.027, +0.063] | 0.660 | 0.6501 | 0.6602 |
| CH_IN | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| CH_IN | filler | 0.044 [+0.027, +0.063] | 0.660 | 0.6501 | 0.6602 |
| CH_X | off | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| CH_X | plain | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| CH_X | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| CH_X | filler | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| ALL | off | 0.379 [+0.328, +0.433] | 0.190 | 0.1758 | 0.1901 |
| ALL | plain | 0.213 [+0.143, +0.290] | 0.474 | 0.4404 | 0.474 |
| ALL | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| ALL | filler | 0.213 [+0.143, +0.290] | 0.474 | 0.4404 | 0.474 |


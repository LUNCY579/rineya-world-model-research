# B4 screen: post-hoc sensitivity for seeds_fresh/0 (BWM P-18)

Post-hoc sensitivity (BWM P-18, found after item 7 and before any screen output): excluding seeds_fresh/0, a block-renamed suffix of a Pa_v2 training row shared by both arms. The registered decision uses all 40 instances.

- seeds_fresh/0: arm a solved, R0 solved: **concordant**.
- All 40 (registered): a 33, R0 27, a_only 7, R0_only 1, net +6.
- Concordant, so excluding it leaves a_only, R0_only and the net unchanged.
- Registered decision (quoted from B4S_REPORT.json, unchanged): {"status": "GO", "decision": {"arm_only": 7, "control_only": 1, "net": 6, "p_one_sided_sign": 0.03515625, "p_mcnemar_two_sided": 0.0703125, "reported_only": ["p_one_sided_sign", "p_mcnemar_two_sided"], "decision": "GO", "go": true, "because": "net 6 >= 2"}, "n_paired": 40, "paired": {"a_only": 7, "R0_only": 1, "both": 26, "neither": 6}}

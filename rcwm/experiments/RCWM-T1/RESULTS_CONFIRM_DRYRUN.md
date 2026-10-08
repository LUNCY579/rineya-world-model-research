# RCWM-T1 PREREG-1 results (DRY RUN on pilot TEST — not confirmatory)

Source: `runs/bank_pilot/analysis/PREREG_1_DRYRUN.json`. Rendered mechanically by `scripts/render_confirm.py`. Analysis plan: `PREREG_1.md` (frozen; `PREREG_1_FREEZE.json`).

**Claim confirmed (C0–C4 all PASS): True**

## Primary contrasts (L and R, k = 4, pooled with equal family weight; normalised regret, negative favours G_chain(Titan))

| | Contrast | Pooled point | 95% CI | One-sided p | Holm p | PASS | Holds in both families | Per-family points | Floor qualifier |
|---|---|---|---|---|---|---|---|---|---|
| C0 | Gtitan_chain_one - F_touched | -0.195 | [-0.264, -0.123] | 9.999e-05 | 0.0005 | PASS | False | ledger -0.092, regist -0.298 |  |
| C1 | Gtitan_chain_one - B3 | -0.363 | [-0.433, -0.291] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.273, regist -0.453 | comparator at/below answer-free floor |
| C2 | Gtitan_chain_one - S | -0.352 | [-0.424, -0.278] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.257, regist -0.446 | comparator at/below answer-free floor |
| C3 | Gtitan_chain_one - Gsupa_best | -0.440 | [-0.516, -0.361] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.329, regist -0.550 |  |
| C4 | Gtitan_chain_one - Gtitan_e2e | -0.389 | [-0.458, -0.319] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.340, regist -0.438 |  |

## Mean normalised regret per method (confirmation parents)

| family / k | n | Gtitan_chain_one | Gtitan_chain_m1 | Gtitan_e2e | Gsupa_best | Gsupa_e2e | Gsupa_chain_one | B3 | S | S_abundant | B2_bilinear@full16 | B2_bilinear@sub4 | F_touched | F_copy_s0 | F_first_only | F_ignore_status | F_ignore_pre |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ledger_replay|k1 | 26 | 0.357 | 0.323 | 0.323 | 0.606 | 0.636 | 0.606 | 0.238 | 0.274 |  | 0.286 | 0.820 | 0.280 | 1.009 | 0.000 | 0.060 |  |
| ledger_replay|k2 | 26 | 0.366 | 0.345 | 0.501 | 0.676 | 0.753 | 0.676 | 0.522 | 0.437 |  | 0.605 | 0.858 | 0.420 | 1.015 | 0.369 | 0.063 |  |
| ledger_replay|k4 | 26 | 0.507 | 0.509 | 0.848 | 0.836 | 0.909 | 0.836 | 0.780 | 0.765 | 0.203 | 0.884 | 0.940 | 0.599 | 1.011 | 0.595 | 0.099 |  |
| prerequisite_plan|k1 | 28 | 0.505 | 0.532 | 0.532 | 0.857 | 0.857 | 0.885 | 0.540 | 0.148 |  | 0.422 | 0.667 |  | 0.885 | 0.000 |  | 0.085 |
| prerequisite_plan|k2 | 28 | 0.681 | 0.699 | 0.776 | 0.909 | 0.909 | 0.923 | 0.535 | 0.218 |  | 0.595 | 0.813 |  | 0.923 | 0.311 |  | 0.093 |
| prerequisite_plan|k4 | 28 | 0.879 | 0.884 | 0.885 | 0.955 | 0.959 | 0.955 | 0.594 | 0.437 |  | 0.647 | 0.802 |  | 0.955 | 0.512 |  | 0.137 |
| register_machine|k1 | 26 | 0.132 | 0.126 | 0.126 | 0.893 | 0.893 | 0.997 | 0.303 | 0.300 |  | 0.319 | 0.537 | 0.401 | 0.997 | 0.000 |  |  |
| register_machine|k2 | 26 | 0.208 | 0.213 | 0.480 | 0.979 | 0.979 | 1.003 | 0.504 | 0.481 |  | 0.549 | 0.742 | 0.511 | 1.003 | 0.497 |  |  |
| register_machine|k4 | 26 | 0.386 | 0.391 | 0.824 | 0.936 | 0.936 | 1.007 | 0.839 | 0.832 | 0.583 | 0.900 | 0.946 | 0.683 | 1.007 | 0.803 |  |  |

## Selections (pilot DEV only) and parse rates

- ledger_replay|k1: B3 chosen {'layer': 16, 'head': 'mlp'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.244, parse {"Gtitan_e2e": 0.995, "Gtitan_chain_one": 0.995, "Gtitan_chain_m1": 0.995, "Gsupa_e2e": 0.986, "Gsupa_chain_one": 0.904, "Gsupa_chain_m1": 0.966}
- ledger_replay|k2: B3 chosen {'layer': 28, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.498, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 0.995, "Gtitan_chain_m1": 0.995, "Gsupa_e2e": 1.0, "Gsupa_chain_one": 0.788, "Gsupa_chain_m1": 0.923}
- ledger_replay|k4: B3 chosen {'layer': 8, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.754, parse {"Gtitan_e2e": 0.995, "Gtitan_chain_one": 0.995, "Gtitan_chain_m1": 0.995, "Gsupa_e2e": 0.957, "Gsupa_chain_one": 0.596, "Gsupa_chain_m1": 0.846}
- prerequisite_plan|k1: B3 chosen {'layer': 20, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_e2e, B3 best-on-test upper bound 0.461, parse {"Gtitan_e2e": 0.955, "Gtitan_chain_one": 0.969, "Gtitan_chain_m1": 0.955, "Gsupa_e2e": 0.839, "Gsupa_chain_one": 0.0, "Gsupa_chain_m1": 0.0}
- prerequisite_plan|k2: B3 chosen {'layer': 20, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_e2e, B3 best-on-test upper bound 0.492, parse {"Gtitan_e2e": 0.991, "Gtitan_chain_one": 0.826, "Gtitan_chain_m1": 0.839, "Gsupa_e2e": 0.348, "Gsupa_chain_one": 0.0, "Gsupa_chain_m1": 0.0}
- prerequisite_plan|k4: B3 chosen {'layer': 24, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.570, parse {"Gtitan_e2e": 0.996, "Gtitan_chain_one": 0.594, "Gtitan_chain_m1": 0.616, "Gsupa_e2e": 0.96, "Gsupa_chain_one": 0.0, "Gsupa_chain_m1": 0.0}
- register_machine|k1: B3 chosen {'layer': 20, 'head': 'mlp'}, S chosen mlp, Supa best mode Gsupa_e2e, B3 best-on-test upper bound 0.292, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 1.0, "Gsupa_e2e": 0.466, "Gsupa_chain_one": 0.0, "Gsupa_chain_m1": 0.0}
- register_machine|k2: B3 chosen {'layer': 20, 'head': 'mlp'}, S chosen mlp, Supa best mode Gsupa_e2e, B3 best-on-test upper bound 0.512, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 1.0, "Gsupa_e2e": 1.0, "Gsupa_chain_one": 0.0, "Gsupa_chain_m1": 0.0}
- register_machine|k4: B3 chosen {'layer': 28, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_e2e, B3 best-on-test upper bound 0.839, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 1.0, "Gsupa_e2e": 1.0, "Gsupa_chain_one": 0.0, "Gsupa_chain_m1": 0.0}

## Secondary contrasts (descriptive; BH q = 0.10)

| contrast | point | 95% CI | one-sided p | BH p | BH sig |
|---|---|---|---|---|---|
| ledger_replay|k1|G_chain - B3 | 0.119 | [0.030, 0.218] | 0.9961 | 1 | False |
| ledger_replay|k1|G_chain - S | 0.083 | [-0.027, 0.200] | 0.9246 | 1 | False |
| ledger_replay|k1|G_chain - F_touched | 0.077 | [-0.031, 0.193] | 0.917 | 1 | False |
| ledger_replay|k1|G_chain - G_e2e | 0.035 | [-0.045, 0.140] | 0.7501 | 0.9377 | False |
| ledger_replay|k1|G_chain - Gsupa_best | -0.249 | [-0.350, -0.141] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k1|G_chain - F_first_only (partial simulator) | 0.357 | [0.250, 0.480] | 1 | 1 | False |
| ledger_replay|k1|B3 - B2 out-of-span (sub4) | -0.581 | [-0.637, -0.520] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k1|B3 - B2 in-span (full16) | -0.048 | [-0.066, -0.032] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k1|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.251 | [-0.348, -0.146] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k2|G_chain - B3 | -0.156 | [-0.248, -0.061] | 0.0008999 | 0.0016 | True |
| ledger_replay|k2|G_chain - S | -0.071 | [-0.149, 0.014] | 0.0489 | 0.0738 | True |
| ledger_replay|k2|G_chain - F_touched | -0.054 | [-0.139, 0.034] | 0.1152 | 0.1589 | False |
| ledger_replay|k2|G_chain - G_e2e | -0.135 | [-0.234, -0.029] | 0.007599 | 0.01216 | True |
| ledger_replay|k2|G_chain - Gsupa_best | -0.310 | [-0.408, -0.205] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k2|G_chain - F_first_only (partial simulator) | -0.003 | [-0.109, 0.105] | 0.4801 | 0.6096 | False |
| ledger_replay|k2|B3 - B2 out-of-span (sub4) | -0.336 | [-0.389, -0.283] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k2|B3 - B2 in-span (full16) | -0.083 | [-0.112, -0.052] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k2|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.309 | [-0.414, -0.198] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k4|G_chain - B3 | -0.273 | [-0.390, -0.149] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k4|G_chain - S | -0.257 | [-0.388, -0.122] | 0.0003 | 0.0005581 | True |
| ledger_replay|k4|G_chain - F_touched | -0.092 | [-0.213, 0.033] | 0.07689 | 0.1098 | False |
| ledger_replay|k4|G_chain - G_e2e | -0.340 | [-0.456, -0.219] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k4|G_chain - Gsupa_best | -0.329 | [-0.469, -0.181] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k4|G_chain - S_abundant | 0.305 | [0.171, 0.440] | 1 | 1 | False |
| ledger_replay|k4|G_chain - F_first_only (partial simulator) | -0.088 | [-0.238, 0.055] | 0.1199 | 0.1626 | False |
| ledger_replay|k4|B3 - B2 out-of-span (sub4) | -0.159 | [-0.218, -0.101] | 9.999e-05 | 0.0001951 | True |
| ledger_replay|k4|B3 - B2 in-span (full16) | -0.104 | [-0.150, -0.057] | 0.0002 | 0.0003809 | True |
| ledger_replay|k4|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.281 | [-0.426, -0.134] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k1|G_chain - B3 | -0.034 | [-0.169, 0.086] | 0.306 | 0.4013 | False |
| prerequisite_plan|k1|G_chain - S | 0.357 | [0.267, 0.451] | 1 | 1 | False |
| prerequisite_plan|k1|G_chain - G_e2e | -0.027 | [-0.105, 0.054] | 0.2482 | 0.3309 | False |
| prerequisite_plan|k1|G_chain - Gsupa_best | -0.351 | [-0.446, -0.254] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k1|G_chain - F_first_only (partial simulator) | 0.505 | [0.401, 0.608] | 1 | 1 | False |
| prerequisite_plan|k1|B3 - B2 out-of-span (sub4) | -0.128 | [-0.270, 0.024] | 0.0467 | 0.07184 | True |
| prerequisite_plan|k1|B3 - B2 in-span (full16) | 0.118 | [0.021, 0.239] | 0.9927 | 1 | False |
| prerequisite_plan|k1|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.392 | [-0.486, -0.293] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k2|G_chain - B3 | 0.146 | [0.012, 0.279] | 0.9847 | 1 | False |
| prerequisite_plan|k2|G_chain - S | 0.462 | [0.343, 0.582] | 1 | 1 | False |
| prerequisite_plan|k2|G_chain - G_e2e | -0.095 | [-0.238, 0.046] | 0.09409 | 0.1321 | False |
| prerequisite_plan|k2|G_chain - Gsupa_best | -0.229 | [-0.331, -0.125] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k2|G_chain - F_first_only (partial simulator) | 0.370 | [0.234, 0.516] | 1 | 1 | False |
| prerequisite_plan|k2|B3 - B2 out-of-span (sub4) | -0.278 | [-0.372, -0.181] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k2|B3 - B2 in-span (full16) | -0.060 | [-0.107, -0.012] | 0.007199 | 0.01175 | True |
| prerequisite_plan|k2|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.282 | [-0.389, -0.173] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k4|G_chain - B3 | 0.284 | [0.179, 0.396] | 1 | 1 | False |
| prerequisite_plan|k4|G_chain - S | 0.442 | [0.322, 0.565] | 1 | 1 | False |
| prerequisite_plan|k4|G_chain - G_e2e | -0.006 | [-0.142, 0.131] | 0.4596 | 0.593 | False |
| prerequisite_plan|k4|G_chain - Gsupa_best | -0.077 | [-0.169, 0.016] | 0.05409 | 0.08014 | True |
| prerequisite_plan|k4|G_chain - F_first_only (partial simulator) | 0.366 | [0.217, 0.523] | 1 | 1 | False |
| prerequisite_plan|k4|B3 - B2 out-of-span (sub4) | -0.208 | [-0.290, -0.129] | 9.999e-05 | 0.0001951 | True |
| prerequisite_plan|k4|B3 - B2 in-span (full16) | -0.053 | [-0.100, -0.006] | 0.0142 | 0.02227 | True |
| prerequisite_plan|k4|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.149 | [-0.240, -0.054] | 0.0011 | 0.001913 | True |
| register_machine|k1|G_chain - B3 | -0.171 | [-0.218, -0.127] | 9.999e-05 | 0.0001951 | True |
| register_machine|k1|G_chain - S | -0.168 | [-0.215, -0.121] | 9.999e-05 | 0.0001951 | True |
| register_machine|k1|G_chain - F_touched | -0.269 | [-0.306, -0.232] | 9.999e-05 | 0.0001951 | True |
| register_machine|k1|G_chain - G_e2e | 0.006 | [-0.003, 0.015] | 0.9028 | 1 | False |
| register_machine|k1|G_chain - Gsupa_best | -0.761 | [-0.830, -0.684] | 9.999e-05 | 0.0001951 | True |
| register_machine|k1|G_chain - F_first_only (partial simulator) | 0.132 | [0.099, 0.166] | 1 | 1 | False |
| register_machine|k1|B3 - B2 out-of-span (sub4) | -0.233 | [-0.292, -0.179] | 9.999e-05 | 0.0001951 | True |
| register_machine|k1|B3 - B2 in-span (full16) | -0.016 | [-0.035, 0.004] | 0.05919 | 0.0861 | True |
| register_machine|k1|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.865 | [-0.898, -0.833] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|G_chain - B3 | -0.296 | [-0.339, -0.250] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|G_chain - S | -0.272 | [-0.321, -0.220] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|G_chain - F_touched | -0.302 | [-0.355, -0.249] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|G_chain - G_e2e | -0.271 | [-0.328, -0.218] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|G_chain - Gsupa_best | -0.770 | [-0.838, -0.695] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|G_chain - F_first_only (partial simulator) | -0.289 | [-0.366, -0.207] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|B3 - B2 out-of-span (sub4) | -0.238 | [-0.281, -0.194] | 9.999e-05 | 0.0001951 | True |
| register_machine|k2|B3 - B2 in-span (full16) | -0.044 | [-0.071, -0.018] | 0.0005 | 0.000909 | True |
| register_machine|k2|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.794 | [-0.843, -0.743] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - B3 | -0.453 | [-0.526, -0.378] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - S | -0.446 | [-0.510, -0.380] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - F_touched | -0.298 | [-0.363, -0.229] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - G_e2e | -0.438 | [-0.509, -0.364] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - Gsupa_best | -0.550 | [-0.611, -0.491] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - S_abundant | -0.198 | [-0.256, -0.136] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|G_chain - F_first_only (partial simulator) | -0.417 | [-0.484, -0.355] | 9.999e-05 | 0.0001951 | True |
| register_machine|k4|B3 - B2 out-of-span (sub4) | -0.107 | [-0.179, -0.037] | 0.0015 | 0.0025 | True |
| register_machine|k4|B3 - B2 in-span (full16) | -0.061 | [-0.104, -0.019] | 0.0012 | 0.002042 | True |
| register_machine|k4|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.622 | [-0.664, -0.578] | 9.999e-05 | 0.0001951 | True |

## Exact-state accuracy (lenient parse)

| arm | family / k / mode | exact | parse | n |
|---|---|---|---|---|
| sparse | ledger_replay|k1|e2e | 0.389 | 0.995 | 208 |
| sparse | ledger_replay|k1|chain_one | 0.399 | 0.995 | 208 |
| sparse | ledger_replay|k1|chain_m1 | 0.389 | 0.995 | 208 |
| sparse | ledger_replay|k2|e2e | 0.072 | 1.000 | 208 |
| sparse | ledger_replay|k2|chain_one | 0.202 | 0.995 | 208 |
| sparse | ledger_replay|k2|chain_m1 | 0.192 | 0.995 | 208 |
| sparse | ledger_replay|k4|e2e | 0.000 | 0.995 | 208 |
| sparse | ledger_replay|k4|chain_one | 0.024 | 0.995 | 208 |
| sparse | ledger_replay|k4|chain_m1 | 0.024 | 0.995 | 208 |
| sparse | prerequisite_plan|k1|e2e | 0.196 | 0.955 | 224 |
| sparse | prerequisite_plan|k1|chain_one | 0.210 | 0.969 | 224 |
| sparse | prerequisite_plan|k1|chain_m1 | 0.196 | 0.955 | 224 |
| sparse | prerequisite_plan|k2|e2e | 0.094 | 0.991 | 224 |
| sparse | prerequisite_plan|k2|chain_one | 0.049 | 0.826 | 224 |
| sparse | prerequisite_plan|k2|chain_m1 | 0.036 | 0.839 | 224 |
| sparse | prerequisite_plan|k4|e2e | 0.143 | 0.996 | 224 |
| sparse | prerequisite_plan|k4|chain_one | 0.000 | 0.594 | 224 |
| sparse | prerequisite_plan|k4|chain_m1 | 0.018 | 0.616 | 224 |
| sparse | register_machine|k1|e2e | 0.611 | 1.000 | 208 |
| sparse | register_machine|k1|chain_one | 0.606 | 1.000 | 208 |
| sparse | register_machine|k1|chain_m1 | 0.611 | 1.000 | 208 |
| sparse | register_machine|k2|e2e | 0.091 | 1.000 | 208 |
| sparse | register_machine|k2|chain_one | 0.361 | 1.000 | 208 |
| sparse | register_machine|k2|chain_m1 | 0.365 | 1.000 | 208 |
| sparse | register_machine|k4|e2e | 0.005 | 1.000 | 208 |
| sparse | register_machine|k4|chain_one | 0.087 | 1.000 | 208 |
| sparse | register_machine|k4|chain_m1 | 0.087 | 1.000 | 208 |
| supa | ledger_replay|k1|e2e | 0.106 | 0.986 | 208 |
| supa | ledger_replay|k1|chain_one | 0.067 | 0.904 | 208 |
| supa | ledger_replay|k1|chain_m1 | 0.106 | 0.966 | 208 |
| supa | ledger_replay|k2|e2e | 0.005 | 1.000 | 208 |
| supa | ledger_replay|k2|chain_one | 0.000 | 0.788 | 208 |
| supa | ledger_replay|k2|chain_m1 | 0.010 | 0.923 | 208 |
| supa | ledger_replay|k4|e2e | 0.000 | 0.957 | 208 |
| supa | ledger_replay|k4|chain_one | 0.000 | 0.596 | 208 |
| supa | ledger_replay|k4|chain_m1 | 0.000 | 0.846 | 208 |
| supa | prerequisite_plan|k1|e2e | 0.112 | 0.839 | 224 |
| supa | prerequisite_plan|k1|chain_one | 0.000 | 0.000 | 224 |
| supa | prerequisite_plan|k1|chain_m1 | 0.000 | 0.000 | 224 |
| supa | prerequisite_plan|k2|e2e | 0.143 | 0.348 | 224 |
| supa | prerequisite_plan|k2|chain_one | 0.000 | 0.000 | 224 |
| supa | prerequisite_plan|k2|chain_m1 | 0.000 | 0.000 | 224 |
| supa | prerequisite_plan|k4|e2e | 0.263 | 0.960 | 224 |
| supa | prerequisite_plan|k4|chain_one | 0.000 | 0.000 | 224 |
| supa | prerequisite_plan|k4|chain_m1 | 0.000 | 0.000 | 224 |
| supa | register_machine|k1|e2e | 0.048 | 0.466 | 208 |
| supa | register_machine|k1|chain_one | 0.000 | 0.000 | 208 |
| supa | register_machine|k1|chain_m1 | 0.000 | 0.000 | 208 |
| supa | register_machine|k2|e2e | 0.043 | 1.000 | 208 |
| supa | register_machine|k2|chain_one | 0.000 | 0.000 | 208 |
| supa | register_machine|k2|chain_m1 | 0.000 | 0.000 | 208 |
| supa | register_machine|k4|e2e | 0.000 | 1.000 | 208 |
| supa | register_machine|k4|chain_one | 0.000 | 0.000 | 208 |
| supa | register_machine|k4|chain_m1 | 0.000 | 0.000 | 208 |

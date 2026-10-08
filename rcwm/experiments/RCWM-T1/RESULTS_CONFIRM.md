# RCWM-T1 PREREG-1 results (confirmation)

Source: `runs/bank_confirm/analysis/PREREG_1_RESULTS.json`. Rendered mechanically by `scripts/render_confirm.py`. Analysis plan: `PREREG_1.md` (frozen; `PREREG_1_FREEZE.json`).

**Claim confirmed (C0–C4 all PASS): True**

## Primary contrasts (L and R, k = 4, pooled with equal family weight; normalised regret, negative favours G_chain(Titan))

| | Contrast | Pooled point | 95% CI | One-sided p | Holm p | PASS | Holds in both families | Per-family points | Floor qualifier |
|---|---|---|---|---|---|---|---|---|---|
| C0 | Gtitan_chain_one - F_touched | -0.195 | [-0.224, -0.165] | 9.999e-05 | 0.0005 | PASS | False | ledger -0.072, regist -0.317 |  |
| C1 | Gtitan_chain_one - B3 | -0.352 | [-0.384, -0.319] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.222, regist -0.483 | comparator at/below answer-free floor |
| C2 | Gtitan_chain_one - S | -0.340 | [-0.372, -0.308] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.198, regist -0.483 | comparator at/below answer-free floor |
| C3 | Gtitan_chain_one - Gsupa_best | -0.440 | [-0.473, -0.408] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.306, regist -0.575 |  |
| C4 | Gtitan_chain_one - Gtitan_e2e | -0.397 | [-0.428, -0.364] | 9.999e-05 | 0.0005 | PASS | True | ledger -0.326, regist -0.468 |  |

## Mean normalised regret per method (confirmation parents)

| family / k | n | Gtitan_chain_one | Gtitan_chain_m1 | Gtitan_e2e | Gsupa_best | Gsupa_e2e | Gsupa_chain_one | B3 | S | S_abundant | B2_bilinear@full16 | B2_bilinear@sub4 | F_touched | F_copy_s0 | F_first_only | F_ignore_status | F_ignore_pre |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ledger_replay|k1 | 150 | 0.305 | 0.315 | 0.315 | 0.648 | 0.680 | 0.648 | 0.211 | 0.274 |  | 0.244 | 0.804 | 0.310 | 0.998 | 0.000 | 0.081 |  |
| ledger_replay|k2 | 150 | 0.388 | 0.406 | 0.512 | 0.720 | 0.785 | 0.720 | 0.532 | 0.457 |  | 0.596 | 0.883 | 0.416 | 1.003 | 0.365 | 0.088 |  |
| ledger_replay|k4 | 150 | 0.539 | 0.537 | 0.865 | 0.845 | 0.920 | 0.845 | 0.761 | 0.737 | 0.273 | 0.854 | 0.936 | 0.611 | 1.005 | 0.656 | 0.115 |  |
| prerequisite_plan|k1 | 150 | 0.576 | 0.607 | 0.607 | 0.911 | 0.929 | 0.911 | 0.539 | 0.189 |  | 0.469 | 0.802 |  | 0.981 | 0.000 |  | 0.178 |
| prerequisite_plan|k2 | 150 | 0.734 | 0.763 | 0.813 | 0.947 | 0.985 | 0.947 | 0.556 | 0.267 |  | 0.624 | 0.847 |  | 0.989 | 0.393 |  | 0.170 |
| prerequisite_plan|k4 | 150 | 0.899 | 0.913 | 0.897 | 0.967 | 0.996 | 0.967 | 0.622 | 0.482 |  | 0.681 | 0.845 |  | 1.000 | 0.682 |  | 0.224 |
| register_machine|k1 | 150 | 0.135 | 0.140 | 0.140 | 0.751 | 0.931 | 0.751 | 0.321 | 0.332 |  | 0.324 | 0.537 | 0.401 | 0.994 | 0.000 |  |  |
| register_machine|k2 | 150 | 0.199 | 0.203 | 0.491 | 0.838 | 0.948 | 0.838 | 0.523 | 0.480 |  | 0.554 | 0.771 | 0.506 | 0.995 | 0.488 |  |  |
| register_machine|k4 | 150 | 0.356 | 0.352 | 0.824 | 0.931 | 0.935 | 0.931 | 0.839 | 0.839 | 0.594 | 0.890 | 0.957 | 0.673 | 0.997 | 0.810 |  |  |

## Selections (pilot DEV only) and parse rates

- ledger_replay|k1: B3 chosen {'layer': 16, 'head': 'mlp'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.219, parse {"Gtitan_e2e": 0.999, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 0.999, "Gsupa_e2e": 0.978, "Gsupa_chain_one": 0.951, "Gsupa_chain_m1": 0.978}
- ledger_replay|k2: B3 chosen {'layer': 28, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.524, parse {"Gtitan_e2e": 0.999, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 0.999, "Gsupa_e2e": 0.996, "Gsupa_chain_one": 0.899, "Gsupa_chain_m1": 0.964}
- ledger_replay|k4: B3 chosen {'layer': 8, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.761, parse {"Gtitan_e2e": 0.999, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 0.999, "Gsupa_e2e": 0.972, "Gsupa_chain_one": 0.815, "Gsupa_chain_m1": 0.936}
- prerequisite_plan|k1: B3 chosen {'layer': 20, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.527, parse {"Gtitan_e2e": 0.973, "Gtitan_chain_one": 0.976, "Gtitan_chain_m1": 0.973, "Gsupa_e2e": 0.812, "Gsupa_chain_one": 0.978, "Gsupa_chain_m1": 0.812}
- prerequisite_plan|k2: B3 chosen {'layer': 20, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.541, parse {"Gtitan_e2e": 0.987, "Gtitan_chain_one": 0.935, "Gtitan_chain_m1": 0.943, "Gsupa_e2e": 0.342, "Gsupa_chain_one": 0.96, "Gsupa_chain_m1": 0.684}
- prerequisite_plan|k4: B3 chosen {'layer': 24, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.622, parse {"Gtitan_e2e": 0.988, "Gtitan_chain_one": 0.779, "Gtitan_chain_m1": 0.833, "Gsupa_e2e": 0.904, "Gsupa_chain_one": 0.938, "Gsupa_chain_m1": 0.544}
- register_machine|k1: B3 chosen {'layer': 20, 'head': 'mlp'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.312, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 1.0, "Gsupa_e2e": 0.511, "Gsupa_chain_one": 0.798, "Gsupa_chain_m1": 0.511}
- register_machine|k2: B3 chosen {'layer': 20, 'head': 'mlp'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.525, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 1.0, "Gtitan_chain_m1": 1.0, "Gsupa_e2e": 0.999, "Gsupa_chain_one": 0.663, "Gsupa_chain_m1": 0.346}
- register_machine|k4: B3 chosen {'layer': 28, 'head': 'ridge'}, S chosen mlp, Supa best mode Gsupa_chain_one, B3 best-on-test upper bound 0.839, parse {"Gtitan_e2e": 1.0, "Gtitan_chain_one": 0.999, "Gtitan_chain_m1": 1.0, "Gsupa_e2e": 1.0, "Gsupa_chain_one": 0.466, "Gsupa_chain_m1": 0.166}

## Secondary contrasts (descriptive; BH q = 0.10)

| contrast | point | 95% CI | one-sided p | BH p | BH sig |
|---|---|---|---|---|---|
| ledger_replay|k1|G_chain - B3 | 0.094 | [0.046, 0.145] | 0.9999 | 1 | False |
| ledger_replay|k1|G_chain - S | 0.031 | [-0.017, 0.079] | 0.8986 | 1 | False |
| ledger_replay|k1|G_chain - F_touched | -0.005 | [-0.052, 0.042] | 0.4165 | 0.5288 | False |
| ledger_replay|k1|G_chain - G_e2e | -0.010 | [-0.039, 0.018] | 0.2603 | 0.3413 | False |
| ledger_replay|k1|G_chain - Gsupa_best | -0.343 | [-0.393, -0.291] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k1|G_chain - F_first_only (partial simulator) | 0.305 | [0.262, 0.351] | 1 | 1 | False |
| ledger_replay|k1|B3 - B2 out-of-span (sub4) | -0.593 | [-0.623, -0.564] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k1|B3 - B2 in-span (full16) | -0.034 | [-0.043, -0.025] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k1|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.348 | [-0.398, -0.297] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k2|G_chain - B3 | -0.145 | [-0.193, -0.093] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k2|G_chain - S | -0.070 | [-0.117, -0.019] | 0.0036 | 0.005142 | True |
| ledger_replay|k2|G_chain - F_touched | -0.028 | [-0.076, 0.022] | 0.1276 | 0.1701 | False |
| ledger_replay|k2|G_chain - G_e2e | -0.124 | [-0.174, -0.070] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k2|G_chain - Gsupa_best | -0.333 | [-0.383, -0.281] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k2|G_chain - F_first_only (partial simulator) | 0.023 | [-0.031, 0.082] | 0.788 | 0.9699 | False |
| ledger_replay|k2|B3 - B2 out-of-span (sub4) | -0.350 | [-0.373, -0.327] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k2|B3 - B2 in-span (full16) | -0.064 | [-0.078, -0.050] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k2|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.337 | [-0.387, -0.285] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|G_chain - B3 | -0.222 | [-0.278, -0.162] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|G_chain - S | -0.198 | [-0.255, -0.138] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|G_chain - F_touched | -0.072 | [-0.126, -0.017] | 0.005399 | 0.007578 | True |
| ledger_replay|k4|G_chain - G_e2e | -0.326 | [-0.382, -0.267] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|G_chain - Gsupa_best | -0.306 | [-0.362, -0.248] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|G_chain - S_abundant | 0.266 | [0.206, 0.328] | 1 | 1 | False |
| ledger_replay|k4|G_chain - F_first_only (partial simulator) | -0.117 | [-0.174, -0.058] | 0.0003 | 0.0004444 | True |
| ledger_replay|k4|B3 - B2 out-of-span (sub4) | -0.176 | [-0.201, -0.150] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|B3 - B2 in-span (full16) | -0.094 | [-0.114, -0.074] | 9.999e-05 | 0.0001538 | True |
| ledger_replay|k4|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.305 | [-0.360, -0.249] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k1|G_chain - B3 | 0.037 | [-0.006, 0.080] | 0.9501 | 1 | False |
| prerequisite_plan|k1|G_chain - S | 0.387 | [0.344, 0.431] | 1 | 1 | False |
| prerequisite_plan|k1|G_chain - G_e2e | -0.031 | [-0.063, 0.001] | 0.0275 | 0.03793 | True |
| prerequisite_plan|k1|G_chain - Gsupa_best | -0.335 | [-0.389, -0.281] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k1|G_chain - F_first_only (partial simulator) | 0.576 | [0.536, 0.616] | 1 | 1 | False |
| prerequisite_plan|k1|B3 - B2 out-of-span (sub4) | -0.262 | [-0.304, -0.219] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k1|B3 - B2 in-span (full16) | 0.070 | [0.047, 0.095] | 1 | 1 | False |
| prerequisite_plan|k1|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.345 | [-0.397, -0.290] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k2|G_chain - B3 | 0.177 | [0.133, 0.222] | 1 | 1 | False |
| prerequisite_plan|k2|G_chain - S | 0.467 | [0.423, 0.511] | 1 | 1 | False |
| prerequisite_plan|k2|G_chain - G_e2e | -0.079 | [-0.126, -0.032] | 0.0004 | 0.0005818 | True |
| prerequisite_plan|k2|G_chain - Gsupa_best | -0.214 | [-0.262, -0.165] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k2|G_chain - F_first_only (partial simulator) | 0.341 | [0.285, 0.395] | 1 | 1 | False |
| prerequisite_plan|k2|B3 - B2 out-of-span (sub4) | -0.291 | [-0.333, -0.251] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k2|B3 - B2 in-span (full16) | -0.068 | [-0.095, -0.041] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k2|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.237 | [-0.285, -0.188] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k4|G_chain - B3 | 0.277 | [0.236, 0.319] | 1 | 1 | False |
| prerequisite_plan|k4|G_chain - S | 0.417 | [0.376, 0.457] | 1 | 1 | False |
| prerequisite_plan|k4|G_chain - G_e2e | 0.002 | [-0.036, 0.040] | 0.5441 | 0.6802 | False |
| prerequisite_plan|k4|G_chain - Gsupa_best | -0.067 | [-0.105, -0.030] | 0.0002 | 0.0003019 | True |
| prerequisite_plan|k4|G_chain - F_first_only (partial simulator) | 0.217 | [0.176, 0.258] | 1 | 1 | False |
| prerequisite_plan|k4|B3 - B2 out-of-span (sub4) | -0.223 | [-0.256, -0.191] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k4|B3 - B2 in-span (full16) | -0.059 | [-0.080, -0.037] | 9.999e-05 | 0.0001538 | True |
| prerequisite_plan|k4|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.101 | [-0.139, -0.063] | 9.999e-05 | 0.0001538 | True |
| register_machine|k1|G_chain - B3 | -0.185 | [-0.206, -0.165] | 9.999e-05 | 0.0001538 | True |
| register_machine|k1|G_chain - S | -0.197 | [-0.216, -0.177] | 9.999e-05 | 0.0001538 | True |
| register_machine|k1|G_chain - F_touched | -0.266 | [-0.282, -0.248] | 9.999e-05 | 0.0001538 | True |
| register_machine|k1|G_chain - G_e2e | -0.005 | [-0.011, 0.002] | 0.07679 | 0.1041 | False |
| register_machine|k1|G_chain - Gsupa_best | -0.615 | [-0.650, -0.579] | 9.999e-05 | 0.0001538 | True |
| register_machine|k1|G_chain - F_first_only (partial simulator) | 0.135 | [0.121, 0.151] | 1 | 1 | False |
| register_machine|k1|B3 - B2 out-of-span (sub4) | -0.216 | [-0.240, -0.194] | 9.999e-05 | 0.0001538 | True |
| register_machine|k1|B3 - B2 in-span (full16) | -0.003 | [-0.014, 0.008] | 0.2906 | 0.3749 | False |
| register_machine|k1|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.617 | [-0.652, -0.582] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain - B3 | -0.324 | [-0.345, -0.304] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain - S | -0.281 | [-0.300, -0.261] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain - F_touched | -0.307 | [-0.325, -0.290] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain - G_e2e | -0.291 | [-0.315, -0.268] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain - Gsupa_best | -0.638 | [-0.664, -0.611] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain - F_first_only (partial simulator) | -0.289 | [-0.316, -0.262] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|B3 - B2 out-of-span (sub4) | -0.248 | [-0.271, -0.224] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|B3 - B2 in-span (full16) | -0.031 | [-0.044, -0.018] | 9.999e-05 | 0.0001538 | True |
| register_machine|k2|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.625 | [-0.651, -0.600] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - B3 | -0.483 | [-0.512, -0.454] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - S | -0.483 | [-0.510, -0.456] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - F_touched | -0.317 | [-0.340, -0.294] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - G_e2e | -0.468 | [-0.496, -0.439] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - Gsupa_best | -0.575 | [-0.606, -0.543] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - S_abundant | -0.238 | [-0.261, -0.214] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain - F_first_only (partial simulator) | -0.454 | [-0.483, -0.424] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|B3 - B2 out-of-span (sub4) | -0.117 | [-0.144, -0.090] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|B3 - B2 in-span (full16) | -0.051 | [-0.072, -0.029] | 9.999e-05 | 0.0001538 | True |
| register_machine|k4|G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed) | -0.565 | [-0.595, -0.534] | 9.999e-05 | 0.0001538 | True |

## Exact-state accuracy (lenient parse)

| arm | family / k / mode | exact | parse | n |
|---|---|---|---|---|
| sparse | ledger_replay|k1|e2e | 0.426 | 0.999 | 1200 |
| sparse | ledger_replay|k1|chain_one | 0.432 | 1.000 | 1200 |
| sparse | ledger_replay|k1|chain_m1 | 0.426 | 0.999 | 1200 |
| sparse | ledger_replay|k2|e2e | 0.092 | 0.999 | 1200 |
| sparse | ledger_replay|k2|chain_one | 0.181 | 1.000 | 1200 |
| sparse | ledger_replay|k2|chain_m1 | 0.164 | 0.999 | 1200 |
| sparse | ledger_replay|k4|e2e | 0.006 | 0.999 | 1200 |
| sparse | ledger_replay|k4|chain_one | 0.024 | 1.000 | 1200 |
| sparse | ledger_replay|k4|chain_m1 | 0.020 | 0.999 | 1200 |
| sparse | prerequisite_plan|k1|e2e | 0.291 | 0.973 | 1200 |
| sparse | prerequisite_plan|k1|chain_one | 0.324 | 0.976 | 1200 |
| sparse | prerequisite_plan|k1|chain_m1 | 0.291 | 0.973 | 1200 |
| sparse | prerequisite_plan|k2|e2e | 0.139 | 0.987 | 1200 |
| sparse | prerequisite_plan|k2|chain_one | 0.094 | 0.935 | 1200 |
| sparse | prerequisite_plan|k2|chain_m1 | 0.079 | 0.943 | 1200 |
| sparse | prerequisite_plan|k4|e2e | 0.058 | 0.988 | 1200 |
| sparse | prerequisite_plan|k4|chain_one | 0.006 | 0.779 | 1200 |
| sparse | prerequisite_plan|k4|chain_m1 | 0.008 | 0.833 | 1200 |
| sparse | register_machine|k1|e2e | 0.573 | 1.000 | 1200 |
| sparse | register_machine|k1|chain_one | 0.575 | 1.000 | 1200 |
| sparse | register_machine|k1|chain_m1 | 0.573 | 1.000 | 1200 |
| sparse | register_machine|k2|e2e | 0.065 | 1.000 | 1200 |
| sparse | register_machine|k2|chain_one | 0.330 | 1.000 | 1200 |
| sparse | register_machine|k2|chain_m1 | 0.332 | 1.000 | 1200 |
| sparse | register_machine|k4|e2e | 0.005 | 1.000 | 1200 |
| sparse | register_machine|k4|chain_one | 0.092 | 0.999 | 1200 |
| sparse | register_machine|k4|chain_m1 | 0.098 | 1.000 | 1200 |
| supa | ledger_replay|k1|e2e | 0.089 | 0.978 | 1200 |
| supa | ledger_replay|k1|chain_one | 0.048 | 0.951 | 1200 |
| supa | ledger_replay|k1|chain_m1 | 0.089 | 0.978 | 1200 |
| supa | ledger_replay|k2|e2e | 0.017 | 0.996 | 1200 |
| supa | ledger_replay|k2|chain_one | 0.004 | 0.899 | 1200 |
| supa | ledger_replay|k2|chain_m1 | 0.007 | 0.964 | 1200 |
| supa | ledger_replay|k4|e2e | 0.002 | 0.972 | 1200 |
| supa | ledger_replay|k4|chain_one | 0.000 | 0.815 | 1200 |
| supa | ledger_replay|k4|chain_m1 | 0.000 | 0.936 | 1200 |
| supa | prerequisite_plan|k1|e2e | 0.356 | 0.812 | 1200 |
| supa | prerequisite_plan|k1|chain_one | 0.368 | 0.978 | 1200 |
| supa | prerequisite_plan|k1|chain_m1 | 0.356 | 0.812 | 1200 |
| supa | prerequisite_plan|k2|e2e | 0.086 | 0.342 | 1200 |
| supa | prerequisite_plan|k2|chain_one | 0.178 | 0.960 | 1200 |
| supa | prerequisite_plan|k2|chain_m1 | 0.164 | 0.684 | 1200 |
| supa | prerequisite_plan|k4|e2e | 0.081 | 0.904 | 1200 |
| supa | prerequisite_plan|k4|chain_one | 0.052 | 0.938 | 1200 |
| supa | prerequisite_plan|k4|chain_m1 | 0.048 | 0.544 | 1200 |
| supa | register_machine|k1|e2e | 0.041 | 0.511 | 1200 |
| supa | register_machine|k1|chain_one | 0.101 | 0.798 | 1200 |
| supa | register_machine|k1|chain_m1 | 0.041 | 0.511 | 1200 |
| supa | register_machine|k2|e2e | 0.029 | 0.999 | 1200 |
| supa | register_machine|k2|chain_one | 0.016 | 0.663 | 1200 |
| supa | register_machine|k2|chain_m1 | 0.003 | 0.346 | 1200 |
| supa | register_machine|k4|e2e | 0.002 | 1.000 | 1200 |
| supa | register_machine|k4|chain_one | 0.002 | 0.466 | 1200 |
| supa | register_machine|k4|chain_m1 | 0.001 | 0.166 | 1200 |

## Twin consistency (k = 4)

- sparse ledger_replay|e2e: n 150, both exact 0.000, predictions differ 0.667
- sparse ledger_replay|chain_one: n 150, both exact 0.020, predictions differ 0.967
- sparse prerequisite_plan|e2e: n 150, both exact 0.000, predictions differ 0.560
- sparse prerequisite_plan|chain_one: n 150, both exact 0.000, predictions differ 0.347
- sparse register_machine|e2e: n 150, both exact 0.000, predictions differ 0.667
- sparse register_machine|chain_one: n 150, both exact 0.053, predictions differ 0.920
- supa ledger_replay|e2e: n 150, both exact 0.000, predictions differ 0.733
- supa ledger_replay|chain_one: n 150, both exact 0.000, predictions differ 0.633
- supa prerequisite_plan|e2e: n 150, both exact 0.000, predictions differ 0.053
- supa prerequisite_plan|chain_one: n 150, both exact 0.000, predictions differ 0.100
- supa register_machine|e2e: n 150, both exact 0.000, predictions differ 0.407
- supa register_machine|chain_one: n 150, both exact 0.000, predictions differ 0.127

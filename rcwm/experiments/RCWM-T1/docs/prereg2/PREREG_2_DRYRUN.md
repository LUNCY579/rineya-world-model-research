# PREREG-2 DRY RUN (code check on pilot TEST, T2 seed 0; NOT confirmatory)

Rendered mechanically by `scripts/confirm2_analysis.py` from the results JSON. Bank: bank_pilot (TEST split only) (sha 7c9b432c30e1...).

The dry run uses pilot TEST, which was used to target the failures T2 fixes (R13), so every number below is exploratory and only checks the code path. n = 26 ledger parents, not 150.

## Verdict: DRY RUN ONLY (pilot TEST, not fresh parents; code check, not evidence); the mechanical wording would read: CONFIRMED (strong pass: both one-sided 95% bounds beyond the thresholds)

> On 26 fresh ledger parents, adding a LoRA trained on 1,501 oracle-labelled one-step rows stratified toward rule cases (T2 seed 0, the fixed artifact) to frozen Titan-Sparse lowered the normalised regret of its chained four-step self-rollout decisions from 0.507 to 0.038 (ON - OFF -0.469, 95% CI [-0.593, -0.354], one-sided 95% upper bound -0.370, Holm one-sided p 0.0002; 0 = oracle, 1 = random) and raised four-step exact-state accuracy from 0.024 to 0.683 (+0.659, [+0.587, +0.726], one-sided 95% lower bound +0.601), against the same model with the adapter off in the same process. Both point estimates are beyond the pre-set thresholds (-0.15, +0.25); strong pass: yes.

> P1 ledger chain_one k=4 normalised regret, ON - OFF: -0.469 (OFF 0.507 -> ON 0.038), 95% CI [-0.593, -0.354], one-sided 95% upper bound -0.370, Holm one-sided p 0.0002; threshold -0.15: PASS (strong).

> P2 ledger chain_one k=4 exact-state accuracy, ON - OFF: +0.659 (OFF 0.024 -> ON 0.683), 95% CI [+0.587, +0.726], one-sided 95% lower bound +0.601, Holm one-sided p 0.0002; threshold +0.25: PASS (strong).

Off-task: Forgetting guard PENDING (no Supa reference in this run).

Limits that always accompany the claim:

- Ledger only, horizon <= 4, one adapter (T2 seed 0), Titan-Sparse only, the chain_one frame.
- The plans and the adapter's training rows come from the same event generator and marginals: the bank is fresh at payload level, not out-of-distribution.
- Rule learning is not separated from frame adaptation (no self-distillation or untargeted-LoRA control); posted-arithmetic gains count toward the effect, and the targeting of the training rows was not shown to cause the gain.
- No comparison with non-LM learners at the same label budget: S@abundant (4,000 parents) reached 0.273 on ledger in PREREG-1 and S-roll matched Titan on ledger at about 512 transitions (R10); the adapted model used 384 + 1,501 oracle-labelled rows.
- Register (mix, wrap) and prerequisite are not claimed fixed; the prerequisite prediction is a copy shortcut, and the prerequisite family is fresh at payload level only.
- Retention evidence is limited to the first-token guard and 64-token agreement on 248 prompts, a re-measurement of quantities partly known before the freeze (§5b).

Split halves (random, exchangeable; descriptive only): P1 A -0.410 / B -0.528; P2 A +0.692 / B +0.625


## Primary contrasts (ledger, chain_one, k = 4; Holm m = 2)

| | contrast | OFF | ON | ON - OFF | 95% CI | one-sided 95% bound | one-sided p | Holm p | threshold | PASS | strong | reversed | halves A / B (descriptive) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | ledger chain_one k=4 normalised regret, ON - OFF | 0.507 | 0.038 | -0.469 | [-0.593, -0.354] | -0.370 | <0.0001 | 0.0002 | -0.15 | True | True | False | -0.410 / -0.528 |
| P2 | ledger chain_one k=4 exact-state accuracy, ON - OFF | 0.024 | 0.683 | +0.659 | [+0.587, +0.726] | +0.601 | <0.0001 | 0.0002 | +0.25 | True | True | False | +0.692 / +0.625 |

Floors (expected never to bind on ledger): F_touched 0.599 (ON below it: True); copy-chain exact floor 0.000 (ON exceeds it by >= 0.10: True). Rule-ablated F_ignore_status 0.099 (diagnostic). Ledger chain_one false-copy ON - OFF -0.030 [-0.044, -0.018] (increased: False).

## Forgetting guard (registered; not part of the claim)

- Status: **PENDING** (PASS iff the one-sided 95% upper bound of r < 1; FAIL iff the lower bound > 1)
- KL(OFF||ON) mean 0.0639 (median 0.0325); top-1 agreement 0.823
- KL(Supa||Sparse): pending (no Supa capture in this run)
- 64-token greedy agreement OFF/ON 0.540; Supa/Sparse n/a; Supa/ON n/a; JSON bleed (non-numeric classes) OFF 0.010 ON 0.015
- Guard code self-test on synthetic distributions: ok = True
- Wording self-test (12 synthetic outcome branches): ok = True

## Cells (OFF -> ON; per-parent paired bootstrap)

| cell | role | k | parents | regret OFF | regret ON | ON - OFF [95% CI] | exact OFF | exact ON | ON - OFF [95% CI] | parse OFF / ON |
|---|---|---|---|---|---|---|---|---|---|---|
| L|chain_one | primary | 1 | 26 | 0.357 | 0.037 | -0.320 [-0.444, -0.212] | 0.399 | 0.880 | +0.481 [+0.409, +0.548] | 0.995 / 0.995 |
| L|chain_one | primary | 2 | 26 | 0.366 | 0.027 | -0.339 [-0.431, -0.252] | 0.202 | 0.817 | +0.615 [+0.543, +0.688] | 0.995 / 0.995 |
| L|chain_one | primary | 4 | 26 | 0.507 | 0.038 | -0.469 [-0.593, -0.354] | 0.024 | 0.683 | +0.659 [+0.587, +0.726] | 0.995 / 0.995 |
| R|chain_one | secondary | 1 | 26 | 0.132 | 0.103 | -0.029 [-0.059, +0.000] | 0.606 | 0.606 | +0.000 [-0.034, +0.043] | 1.000 / 1.000 |
| R|chain_one | secondary | 2 | 26 | 0.208 | 0.188 | -0.021 [-0.060, +0.013] | 0.361 | 0.370 | +0.010 [-0.024, +0.043] | 1.000 / 1.000 |
| R|chain_one | secondary | 4 | 26 | 0.386 | 0.323 | -0.063 [-0.108, -0.016] | 0.087 | 0.091 | +0.005 [-0.029, +0.038] | 1.000 / 1.000 |
| P|chain_m1 | secondary | 1 | 28 | 0.532 | 0.603 | +0.071 [-0.071, +0.215] | 0.196 | 0.638 | +0.442 [+0.326, +0.554] | 0.955 / 1.000 |
| P|chain_m1 | secondary | 2 | 28 | 0.699 | 0.650 | -0.049 [-0.217, +0.123] | 0.036 | 0.460 | +0.424 [+0.299, +0.549] | 0.839 / 1.000 |
| P|chain_m1 | secondary | 4 | 28 | 0.884 | 0.688 | -0.196 [-0.307, -0.093] | 0.018 | 0.330 | +0.312 [+0.201, +0.433] | 0.616 / 1.000 |
| L|chain_m1 | transfer | 1 | 26 | 0.323 | 0.020 | -0.302 [-0.395, -0.215] | 0.389 | 0.918 | +0.529 [+0.466, +0.587] | 0.995 / 1.000 |
| L|chain_m1 | transfer | 2 | 26 | 0.345 | 0.016 | -0.329 [-0.418, -0.249] | 0.192 | 0.885 | +0.692 [+0.635, +0.750] | 0.995 / 1.000 |
| L|chain_m1 | transfer | 4 | 26 | 0.509 | 0.034 | -0.475 [-0.588, -0.368] | 0.024 | 0.779 | +0.755 [+0.697, +0.808] | 0.995 / 1.000 |

## False copy (own-input convention) and sensitivity

| cell | false-copy OFF | ON | ON - OFF [95% CI] | k4 regret, last-parsed fill | k4 regret, goals seed 1004 | k4 regret, t2_train-overlap parents dropped (n) | k4 regret, any-overlap parents dropped (n) |
|---|---|---|---|---|---|---|---|
| L|chain_one | 0.030 | 0.000 | -0.030 [-0.044, -0.018] | -0.470 | -0.451 | -0.469 (0) | -0.469 (0) |
| R|chain_one | 0.049 | 0.022 | -0.028 [-0.043, -0.014] | -0.063 | -0.063 | -0.063 (0) | -0.063 (0) |
| P|chain_m1 | 0.015 | 0.677 | +0.628 [+0.490, +0.758] | -0.103 | -0.198 | -0.205 (3) | -0.196 (0) |
| L|chain_m1 | 0.037 | 0.006 | -0.031 [-0.045, -0.019] | -0.477 | -0.467 | -0.475 (0) | -0.475 (0) |

## Secondary contrasts (two-sided, BH q = 0.10)

Two-sided parent-bootstrap p for 'ON - OFF != 0'; BH q = 0.10 across every listed contrast (28). About half are ledger contrasts expected to be near-certain, which loosens the BH thresholds for register and prerequisite. False-copy and step-level rates condition on arm-specific step sets, so their ON - OFF differences are paired by parent only.

| contrast | ON - OFF | 95% CI | p (two-sided) | BH p | BH sig |
|---|---|---|---|---|---|
| L|chain_one|k1|regret_on_minus_off | -0.320 | [-0.444, -0.212] | 0.0002 | 0.0003 | True |
| L|chain_one|k1|exact_on_minus_off | +0.481 | [+0.409, +0.548] | 0.0002 | 0.0003 | True |
| L|chain_one|k2|regret_on_minus_off | -0.339 | [-0.431, -0.252] | 0.0002 | 0.0003 | True |
| L|chain_one|k2|exact_on_minus_off | +0.615 | [+0.543, +0.688] | 0.0002 | 0.0003 | True |
| L|chain_one|false_copy_on_minus_off | -0.030 | [-0.044, -0.018] | 0.0002 | 0.0003 | True |
| R|chain_one|k1|regret_on_minus_off | -0.029 | [-0.059, +0.000] | 0.0542 | 0.0690 | True |
| R|chain_one|k1|exact_on_minus_off | +0.000 | [-0.034, +0.043] | 1.0000 | 1.0000 | False |
| R|chain_one|k2|regret_on_minus_off | -0.021 | [-0.060, +0.013] | 0.2522 | 0.3070 | False |
| R|chain_one|k2|exact_on_minus_off | +0.010 | [-0.024, +0.043] | 0.6773 | 0.7294 | False |
| R|chain_one|k4|regret_on_minus_off | -0.063 | [-0.108, -0.016] | 0.0098 | 0.0131 | True |
| R|chain_one|k4|exact_on_minus_off | +0.005 | [-0.029, +0.038] | 0.8669 | 0.8990 | False |
| R|chain_one|false_copy_on_minus_off | -0.028 | [-0.043, -0.014] | 0.0004 | 0.0006 | True |
| P|chain_m1|k1|regret_on_minus_off | +0.071 | [-0.071, +0.215] | 0.3216 | 0.3752 | False |
| P|chain_m1|k1|exact_on_minus_off | +0.442 | [+0.326, +0.554] | 0.0002 | 0.0003 | True |
| P|chain_m1|k2|regret_on_minus_off | -0.049 | [-0.217, +0.123] | 0.5699 | 0.6383 | False |
| P|chain_m1|k2|exact_on_minus_off | +0.424 | [+0.299, +0.549] | 0.0002 | 0.0003 | True |
| P|chain_m1|k4|regret_on_minus_off | -0.196 | [-0.307, -0.093] | 0.0002 | 0.0003 | True |
| P|chain_m1|k4|exact_on_minus_off | +0.312 | [+0.201, +0.433] | 0.0002 | 0.0003 | True |
| P|chain_m1|false_copy_on_minus_off | +0.628 | [+0.490, +0.758] | 0.0002 | 0.0003 | True |
| L|chain_m1|k1|regret_on_minus_off | -0.302 | [-0.395, -0.215] | 0.0002 | 0.0003 | True |
| L|chain_m1|k1|exact_on_minus_off | +0.529 | [+0.466, +0.587] | 0.0002 | 0.0003 | True |
| L|chain_m1|k2|regret_on_minus_off | -0.329 | [-0.418, -0.249] | 0.0002 | 0.0003 | True |
| L|chain_m1|k2|exact_on_minus_off | +0.692 | [+0.635, +0.750] | 0.0002 | 0.0003 | True |
| L|chain_m1|k4|regret_on_minus_off | -0.475 | [-0.588, -0.368] | 0.0002 | 0.0003 | True |
| L|chain_m1|k4|exact_on_minus_off | +0.755 | [+0.697, +0.808] | 0.0002 | 0.0003 | True |
| L|chain_m1|false_copy_on_minus_off | -0.031 | [-0.045, -0.019] | 0.0002 | 0.0003 | True |
| L|chain_one|k4|exact_on_minus_off|all_posted_plans | +0.588 | [+0.495, +0.676] | 0.0002 | 0.0003 | True |
| L|chain_one|k4|exact_on_minus_off|any_void_plans | +0.759 | [+0.670, +0.843] | 0.0002 | 0.0003 | True |

## Pre-declared directional predictions

- D1 prerequisite copy shortcut persists: P chain_m1 false-copy ON - OFF > 0 with the 95% CI above 0: **holds** ({"point": 0.6276560847366949, "ci95": [0.4900387328015325, 0.7577009289931403]})
- D2 prerequisite chain_m1 k=4 exact(ON) <= copy-chain floor + 0.10: **holds** ({"on": 0.33035714285714285, "copy_floor": 0.3080357142857143})
- D3 prerequisite chain_m1 applicable state-changing step accuracy ON <= OFF (correct-input steps; descriptive point; arm-specific step sets): **holds** ({"off": 0.46078431372549017, "on": 0.30864197530864196})
- D4 register: no large change (chain_one k=4: |exact ON - OFF| <= 0.05 and |regret ON - OFF| <= 0.1; can hold alongside a small BH-significant change): **holds** ({"exact_delta": 0.004807692307692308, "regret_delta": -0.06255844779885732, "regret_delta_bh_significant": true})

## Step accuracy on correct-input chain steps (descriptive; arm-specific step sets)

- L|chain_one: add/posted 0.529->0.884 (n 68/155); add/void 0.000->1.000 (n 10/20); scale/posted 0.640->0.971 (n 100/207); scale/void 0.071->1.000 (n 14/25); transfer/posted 0.315->0.845 (n 146/277); transfer/void 0.118->1.000 (n 17/30)
- R|chain_one: add/nowrap 0.920->0.875 (n 87/88); add/wrap 0.000->0.000 (n 8/8); conditional/gt 0.438->0.420 (n 48/50); conditional/le 0.511->0.674 (n 47/43); mix/nowrap 0.050->0.048 (n 20/21); mix/wrap 0.000->0.019 (n 106/107); swap 1.000->0.978 (n 133/134)
- P|chain_m1: applicable, state-changing 0.461->0.309 (n 102/162); applicable, unchanged 0.000->0.826 (n 56/121); inapplicable 0.034->0.923 (n 118/273)
- L|chain_m1: add/posted 0.426->0.932 (n 68/161); add/void 0.000->1.000 (n 10/21); scale/posted 0.649->0.961 (n 94/228); scale/void 0.077->1.000 (n 13/26); transfer/posted 0.333->0.906 (n 144/288); transfer/void 0.056->1.000 (n 18/32)
- Ledger chain_one k=4 exact by plan type (pooled; parent-paired ON - OFF): all_posted: 0.039->0.636 (n 129; +0.588 [+0.495, +0.676], 26 parents); any_void: 0.000->0.759 (n 79; +0.759 [+0.670, +0.843], 25 parents)

## Floors (normalised regret; copy-chain exact floor)

- L|k1: F_copy_s0 1.009, F_first_only 0.000, F_ignore_status 0.060, F_last_only 0.000, F_touched 0.280, copy_chain_exact_floor 0.144
- L|k2: F_copy_s0 1.015, F_first_only 0.369, F_ignore_status 0.063, F_last_only 0.362, F_touched 0.420, copy_chain_exact_floor 0.010
- L|k4: F_copy_s0 1.011, F_first_only 0.595, F_ignore_status 0.099, F_last_only 0.658, F_touched 0.599, copy_chain_exact_floor 0.000
- R|k1: F_copy_s0 0.997, F_first_only 0.000, F_last_only 0.000, F_touched 0.401, copy_chain_exact_floor 0.038
- R|k2: F_copy_s0 1.003, F_first_only 0.497, F_last_only 0.539, F_touched 0.511, copy_chain_exact_floor 0.034
- R|k4: F_copy_s0 1.007, F_first_only 0.803, F_last_only 0.859, F_touched 0.683, copy_chain_exact_floor 0.000
- P|k1: F_copy_s0 0.885, F_first_only 0.000, F_ignore_pre 0.085, F_last_only 0.000, copy_chain_exact_floor 0.612
- P|k2: F_copy_s0 0.923, F_first_only 0.311, F_ignore_pre 0.093, F_last_only 0.322, copy_chain_exact_floor 0.451
- P|k4: F_copy_s0 0.955, F_first_only 0.512, F_ignore_pre 0.137, F_last_only 0.632, copy_chain_exact_floor 0.308

## Dry-run reproduction of R13 / T2_RESULTS (code check)

- L chain_one k4 regret OFF/ON (R13: 0.507 / 0.038): [0.5073, 0.0385]
- T2 drop via decide.paired_bootstrap (R13: +0.469 [+0.354, +0.593]): [0.4688, [0.3539, 0.5929]]
- P1 equals minus the T2 drop, CI mirrored: True
- L chain_one k4 exact OFF/ON (R13: 0.024 / 0.683): [0.024, 0.6827]
- R chain_one k4 regret OFF/ON (R13: 0.386 / 0.323): [0.3856, 0.323]
- P chain_m1 k4 exact OFF/ON (R13: 0.018 / 0.330): [0.0179, 0.3304]
- P chain_m1 k4 regret OFF/ON, t2_analysis two-mode g_only call (R13: 0.878 / 0.685): [0.8777, 0.6847]
- P chain_m1 k4 regret OFF/ON, registered single-mode call (differs from R13 only by the tie-break stream order): [0.8845, 0.6885]
- reproduces R13 to the reported 3 decimals (L, R, P exact; P regret via the two-mode call): True

## Power at the registered sizes

(a) Normal approximation from pilot per-parent SDs (homogeneous effects):

- P1 (L regret ON-OFF): pilot n 26, pilot effect +0.469, per-parent SD 0.314, threshold -0.15
  - pilot effect (mu 0.469), SD x1.0: pass (n 150) 1.000, strong 1.000; SE 0.0257
  - pilot effect (mu 0.469), SD x1.25: pass (n 150) 1.000, strong 1.000; SE 0.0321
  - pilot effect (mu 0.469), SD x2.0: pass (n 150) 1.000, strong 1.000; SE 0.0513
  - pilot x 0.5 (mu 0.234), SD x1.0: pass (n 150) 0.999, strong 0.950; SE 0.0257
  - pilot x 0.5 (mu 0.234), SD x1.25: pass (n 150) 0.996, strong 0.838; SE 0.0321
  - pilot x 0.5 (mu 0.234), SD x2.0: pass (n 150) 0.950, strong 0.500; SE 0.0513
  - threshold + 0.05 (mu 0.200), SD x1.0: pass (n 150) 0.974, strong 0.619; SE 0.0257
  - threshold + 0.05 (mu 0.200), SD x1.25: pass (n 150) 0.941, strong 0.466; SE 0.0321
  - threshold + 0.05 (mu 0.200), SD x2.0: pass (n 150) 0.835, strong 0.251; SE 0.0513
- P2 (L exact ON-OFF): pilot n 26, pilot effect +0.659, per-parent SD 0.182, threshold +0.25
  - pilot effect (mu 0.659), SD x1.0: pass (n 150) 1.000, strong 1.000; SE 0.0149
  - pilot effect (mu 0.659), SD x1.25: pass (n 150) 1.000, strong 1.000; SE 0.0186
  - pilot effect (mu 0.659), SD x2.0: pass (n 150) 1.000, strong 1.000; SE 0.0298
  - pilot x 0.5 (mu 0.329), SD x1.0: pass (n 150) 1.000, strong 1.000; SE 0.0149
  - pilot x 0.5 (mu 0.329), SD x1.25: pass (n 150) 1.000, strong 0.996; SE 0.0186
  - pilot x 0.5 (mu 0.329), SD x2.0: pass (n 150) 0.996, strong 0.846; SE 0.0298
  - threshold + 0.05 (mu 0.300), SD x1.0: pass (n 150) 1.000, strong 0.957; SE 0.0149
  - threshold + 0.05 (mu 0.300), SD x1.25: pass (n 150) 0.996, strong 0.852; SE 0.0186
  - threshold + 0.05 (mu 0.300), SD x2.0: pass (n 150) 0.954, strong 0.514; SE 0.0298
- S (R regret ON-OFF): pilot n 26, pilot effect +0.063, per-parent SD 0.122, threshold +0.00
  - pilot effect (mu 0.063), SD x1.0: pass (n 150) 1.000; SE 0.0100
  - pilot effect (mu 0.063), SD x1.25: pass (n 150) 0.999; SE 0.0124
  - pilot effect (mu 0.063), SD x2.0: pass (n 150) 0.881; SE 0.0199
  - pilot x 0.5 (mu 0.031), SD x1.0: pass (n 150) 0.881; SE 0.0100
  - pilot x 0.5 (mu 0.031), SD x1.25: pass (n 150) 0.710; SE 0.0124
  - pilot x 0.5 (mu 0.031), SD x2.0: pass (n 150) 0.349; SE 0.0199
  - threshold + 0.05 (mu 0.050), SD x1.0: pass (n 150) 0.999; SE 0.0100
  - threshold + 0.05 (mu 0.050), SD x1.25: pass (n 150) 0.980; SE 0.0124
  - threshold + 0.05 (mu 0.050), SD x2.0: pass (n 150) 0.709; SE 0.0199
- S (P regret ON-OFF, chain_m1): pilot n 26, pilot effect +0.196, per-parent SD 0.279, threshold +0.00
  - pilot effect (mu 0.196), SD x1.0: pass (n 150) 1.000; SE 0.0228
  - pilot effect (mu 0.196), SD x1.25: pass (n 150) 1.000; SE 0.0285
  - pilot effect (mu 0.196), SD x2.0: pass (n 150) 0.990; SE 0.0456
  - pilot x 0.5 (mu 0.098), SD x1.0: pass (n 150) 0.990; SE 0.0228
  - pilot x 0.5 (mu 0.098), SD x1.25: pass (n 150) 0.930; SE 0.0285
  - pilot x 0.5 (mu 0.098), SD x2.0: pass (n 150) 0.574; SE 0.0456
  - threshold + 0.05 (mu 0.050), SD x1.0: pass (n 150) 0.592; SE 0.0228
  - threshold + 0.05 (mu 0.050), SD x1.25: pass (n 150) 0.418; SE 0.0285
  - threshold + 0.05 (mu 0.050), SD x2.0: pass (n 150) 0.194; SE 0.0456

(b) Joint power by simulation ({'n': 150, 'n_sim': 200, 'n_boot': 2000, 'seed': 20261008, 'note': 'registered boot_mean/holm with n_boot reduced for speed; pilot parents resampled with replacement'}):

| model / effects | P1 PASS | P2 PASS | claim confirmed | both strong |
|---|---|---|---|---|
| scaled: P1 effect 0.469, P2 effect 0.659 | 1.000 | 1.000 | 1.000 | 1.000 |
| scaled: P1 effect 0.234, P2 effect 0.329 | 1.000 | 1.000 | 1.000 | 1.000 |
| scaled: P1 effect 0.200, P2 effect 0.281 | 1.000 | 1.000 | 1.000 | 0.995 |
| scaled: P1 effect 0.169, P2 effect 0.237 | 0.995 | 0.005 | 0.005 | 0.000 |
| mixture: P1 effect 0.469, P2 effect 0.659 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixture: P1 effect 0.234, P2 effect 0.329 | 1.000 | 0.990 | 0.990 | 0.860 |
| mixture: P1 effect 0.200, P2 effect 0.281 | 0.980 | 0.880 | 0.880 | 0.260 |
| mixture: P1 effect 0.169, P2 effect 0.237 | 0.820 | 0.340 | 0.340 | 0.035 |
| sd_upper: P1 effect 0.469, P2 effect 0.659 | 1.000 | 1.000 | 1.000 | 1.000 |
| sd_upper: P1 effect 0.234, P2 effect 0.329 | 0.985 | 1.000 | 0.985 | 0.780 |
| sd_upper: P1 effect 0.200, P2 effect 0.281 | 0.945 | 0.950 | 0.895 | 0.180 |
| sd_upper: P1 effect 0.169, P2 effect 0.237 | 0.720 | 0.220 | 0.150 | 0.000 |

Inputs: {"code_sha256": {"scripts/confirm2_analysis.py": "1c158a4e72bab2e8638ff03b526fa56a1d1b0dc75c266f1ea8ef820f685f7b1a", "scripts/t2_analysis.py": "d699dd5afa432dfef8b1c39cd14af26b92750237c7108328f7a61403decbcaf0", "scripts/confirm2_gen.py": "4253a42039c4a1bb95f412666ad4bc1804244d5b10e6e9fa393c99a488659380", "rcwmt/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "rcwmt/bank.py": "b08b0229c566f9f5b50c7c0c5738f7fff90b768a827e7e32c9880ff69352592f", "rcwmt/capture.py": "7b0a19ee4aefda183f22336521fa39d5d3694bf1d60b5b950d769fa942ce255a", "rcwmt/controls.py": "08a9e4479fe621d5bb2d2a6e9d970eef122ae6850e7b894206b6a7c57002a702", "rcwmt/data.py": "f77270ee04d64985e5b6fa9a88f989a04b0b7ed8cc75f69a4984ca7e7593beb4", "rcwmt/decide.py": "148e98ab528dcd283f6896f07f01135ede658e124d09d08d293e7d3645463643", "rcwmt/decisions.py": "bd6a4531622d2479f1b7fe35fcb11338d831c1ab2e9402b515f9a35ef5237188", "rcwmt/env.py": "92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2", "rcwmt/fidelity.py": "da03bbca4962c05b13eb13dfa0414578344bbe2674015dc93070465ca4e1bb0e", "rcwmt/generation.py": "a4052088a3b6e4d1bc0b59c947599ebce76cdd0bb37230761cb817fa92df7b19", "rcwmt/heads.py": "b3bbb95696fc40fa69cb1d3f60e2547f2fde8305862af2e5b934b073170d58b6", "rcwmt/lease_vendored.py": "d1576bb9296c6bd033dd677c0f8cbe18c9bc24939f75bba30d740429b2ec36c9", "rcwmt/offtask.py": "5fcc70500d21b9043ff93a054b53bdcf29fb2e052eeb722a99fb388fdd41ba0e", "rcwmt/prereg1.py": "1068d66dab6e1c87a255440eae376c1a884ac7dfc3cedac85478e30775cacd35", "rcwmt/pvr.py": "f7b1717fb276252900e75c3a6bf2f835d772498f9b75762aa82faea799e94b94", "rcwmt/runtime.py": "9360541286dc5648e6fd71c217135914b80fd4cef665364c47688cb07f0a12ca"}, "numpy": "2.4.6", "torch_imported": false, "seconds": 9.9}


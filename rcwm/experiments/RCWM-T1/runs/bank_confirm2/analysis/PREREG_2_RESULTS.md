# PREREG-2 results

Rendered mechanically by `scripts/confirm2_analysis.py` from the results JSON. Bank: bank_confirm2 (sha fafb13e573a2...).

## Verdict: CONFIRMED (strong pass: both one-sided 95% bounds beyond the thresholds)

> On 150 fresh ledger parents, adding a LoRA trained on 1,501 oracle-labelled one-step rows stratified toward rule cases (T2 seed 0, the fixed artifact) to frozen Titan-Sparse lowered the normalised regret of its chained four-step self-rollout decisions from 0.544 to 0.056 (ON - OFF -0.488, 95% CI [-0.535, -0.443], one-sided 95% upper bound -0.450, Holm one-sided p 0.0002; 0 = oracle, 1 = random) and raised four-step exact-state accuracy from 0.029 to 0.698 (+0.668, [+0.642, +0.695], one-sided 95% lower bound +0.646), against the same model with the adapter off in the same process. Both point estimates are beyond the pre-set thresholds (-0.15, +0.25); strong pass: yes.

> P1 ledger chain_one k=4 normalised regret, ON - OFF: -0.488 (OFF 0.544 -> ON 0.056), 95% CI [-0.535, -0.443], one-sided 95% upper bound -0.450, Holm one-sided p 0.0002; threshold -0.15: PASS (strong).

> P2 ledger chain_one k=4 exact-state accuracy, ON - OFF: +0.668 (OFF 0.029 -> ON 0.698), 95% CI [+0.642, +0.695], one-sided 95% lower bound +0.646, Holm one-sided p 0.0002; threshold +0.25: PASS (strong).

Off-task: Forgetting guard PASS: off-task first-token KL(OFF||ON) is 0.50x KL(Supa||Sparse), the Titan delta's own drift (95% CI 0.34-0.70; one-sided 95% bounds 0.37 / 0.67); cumulative KL(Supa||ON) is 1.94x KL(Supa||Sparse) (reading bound 1.25); 64-token greedy agreement OFF/ON 0.54 vs Supa/Sparse 0.50 vs Supa/ON 0.37. A re-measurement of quantities partly known before the freeze (§5b).

Limits that always accompany the claim:

- Ledger only, horizon <= 4, one adapter (T2 seed 0), Titan-Sparse only, the chain_one frame.
- The plans and the adapter's training rows come from the same event generator and marginals: the bank is fresh at payload level, not out-of-distribution.
- Rule learning is not separated from frame adaptation (no self-distillation or untargeted-LoRA control); posted-arithmetic gains count toward the effect, and the targeting of the training rows was not shown to cause the gain.
- No comparison with non-LM learners at the same label budget: S@abundant (4,000 parents) reached 0.273 on ledger in PREREG-1 and S-roll matched Titan on ledger at about 512 transitions (R10); the adapted model used 384 + 1,501 oracle-labelled rows.
- Register (mix, wrap) and prerequisite are not claimed fixed; the prerequisite prediction is a copy shortcut, and the prerequisite family is fresh at payload level only.
- Retention evidence is limited to the first-token guard and 64-token agreement on 248 prompts, a re-measurement of quantities partly known before the freeze (§5b).

Split halves (random, exchangeable; descriptive only): P1 A -0.504 / B -0.473; P2 A +0.657 / B +0.680


## Primary contrasts (ledger, chain_one, k = 4; Holm m = 2)

| | contrast | OFF | ON | ON - OFF | 95% CI | one-sided 95% bound | one-sided p | Holm p | threshold | PASS | strong | reversed | halves A / B (descriptive) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | ledger chain_one k=4 normalised regret, ON - OFF | 0.544 | 0.056 | -0.488 | [-0.535, -0.443] | -0.450 | <0.0001 | 0.0002 | -0.15 | True | True | False | -0.504 / -0.473 |
| P2 | ledger chain_one k=4 exact-state accuracy, ON - OFF | 0.029 | 0.698 | +0.668 | [+0.642, +0.695] | +0.646 | <0.0001 | 0.0002 | +0.25 | True | True | False | +0.657 / +0.680 |

Floors (expected never to bind on ledger): F_touched 0.577 (ON below it: True); copy-chain exact floor 0.002 (ON exceeds it by >= 0.10: True). Rule-ablated F_ignore_status 0.120 (diagnostic). Ledger chain_one false-copy ON - OFF -0.025 [-0.030, -0.020] (increased: False).

## Forgetting guard (registered; not part of the claim)

- Status: **PASS** (PASS iff the one-sided 95% upper bound of r < 1; FAIL iff the lower bound > 1)
- KL(OFF||ON) mean 0.0639 (median 0.0325); top-1 agreement 0.823
- KL(Supa||Sparse) mean 0.1285; r = 0.497 [0.344, 0.703], one-sided bounds 0.366 / 0.673 (reply-class cluster bootstrap, 62 classes)
- Cumulative KL(Supa||ON) mean 0.2492; ratio to KL(Supa||Sparse) 1.940 [1.766, 2.183] (reading bound 1.25: beyond)
- 64-token greedy agreement OFF/ON 0.540; Supa/Sparse 0.504; Supa/ON 0.371; JSON bleed (non-numeric classes) OFF 0.010 ON 0.015

## Cells (OFF -> ON; per-parent paired bootstrap)

| cell | role | k | parents | regret OFF | regret ON | ON - OFF [95% CI] | exact OFF | exact ON | ON - OFF [95% CI] | parse OFF / ON |
|---|---|---|---|---|---|---|---|---|---|---|
| L|chain_one | primary | 1 | 150 | 0.306 | 0.029 | -0.276 [-0.316, -0.239] | 0.397 | 0.905 | +0.508 [+0.479, +0.537] | 1.000 / 0.998 |
| L|chain_one | primary | 2 | 150 | 0.396 | 0.040 | -0.356 [-0.397, -0.317] | 0.156 | 0.834 | +0.678 [+0.650, +0.707] | 0.999 / 0.998 |
| L|chain_one | primary | 4 | 150 | 0.544 | 0.056 | -0.488 [-0.535, -0.443] | 0.029 | 0.698 | +0.668 [+0.642, +0.695] | 0.998 / 0.998 |
| R|chain_one | secondary | 1 | 150 | 0.161 | 0.138 | -0.023 [-0.037, -0.008] | 0.574 | 0.578 | +0.003 [-0.015, +0.022] | 1.000 / 1.000 |
| R|chain_one | secondary | 2 | 150 | 0.215 | 0.195 | -0.019 [-0.036, -0.003] | 0.342 | 0.343 | +0.001 [-0.019, +0.021] | 1.000 / 1.000 |
| R|chain_one | secondary | 4 | 150 | 0.348 | 0.329 | -0.019 [-0.039, +0.001] | 0.101 | 0.112 | +0.011 [-0.002, +0.024] | 1.000 / 1.000 |
| P|chain_m1 | secondary | 1 | 150 | 0.626 | 0.586 | -0.040 [-0.100, +0.019] | 0.307 | 0.605 | +0.297 [+0.243, +0.349] | 0.962 / 0.997 |
| P|chain_m1 | secondary | 2 | 150 | 0.770 | 0.663 | -0.107 [-0.162, -0.053] | 0.065 | 0.398 | +0.333 [+0.288, +0.376] | 0.907 / 0.994 |
| P|chain_m1 | secondary | 4 | 150 | 0.921 | 0.716 | -0.204 [-0.253, -0.157] | 0.004 | 0.205 | +0.201 [+0.168, +0.235] | 0.765 / 0.990 |
| L|chain_m1 | transfer | 1 | 150 | 0.322 | 0.028 | -0.294 [-0.334, -0.256] | 0.388 | 0.937 | +0.548 [+0.520, +0.577] | 1.000 / 0.998 |
| L|chain_m1 | transfer | 2 | 150 | 0.399 | 0.045 | -0.354 [-0.396, -0.314] | 0.143 | 0.876 | +0.733 [+0.708, +0.758] | 1.000 / 0.998 |
| L|chain_m1 | transfer | 4 | 150 | 0.554 | 0.060 | -0.494 [-0.542, -0.447] | 0.029 | 0.773 | +0.744 [+0.717, +0.771] | 1.000 / 0.998 |

## False copy (own-input convention) and sensitivity

| cell | false-copy OFF | ON | ON - OFF [95% CI] | k4 regret, last-parsed fill | k4 regret, goals seed 1004 | k4 regret, t2_train-overlap parents dropped (n) | k4 regret, any-overlap parents dropped (n) |
|---|---|---|---|---|---|---|---|
| L|chain_one | 0.026 | 0.001 | -0.025 [-0.030, -0.020] | -0.488 | -0.486 | -0.488 (0) | -0.488 (0) |
| R|chain_one | 0.071 | 0.020 | -0.050 [-0.059, -0.042] | -0.019 | -0.027 | -0.019 (0) | -0.019 (11) |
| P|chain_m1 | 0.039 | 0.624 | +0.558 [+0.511, +0.605] | -0.162 | -0.184 | -0.198 (27) | -0.140 (121) |
| L|chain_m1 | 0.034 | 0.005 | -0.029 [-0.035, -0.022] | -0.493 | -0.488 | -0.494 (0) | -0.494 (0) |

## Secondary contrasts (two-sided, BH q = 0.10)

Two-sided parent-bootstrap p for 'ON - OFF != 0'; BH q = 0.10 across every listed contrast (28). About half are ledger contrasts expected to be near-certain, which loosens the BH thresholds for register and prerequisite. False-copy and step-level rates condition on arm-specific step sets, so their ON - OFF differences are paired by parent only.

| contrast | ON - OFF | 95% CI | p (two-sided) | BH p | BH sig |
|---|---|---|---|---|---|
| L|chain_one|k1|regret_on_minus_off | -0.276 | [-0.316, -0.239] | 0.0002 | 0.0003 | True |
| L|chain_one|k1|exact_on_minus_off | +0.508 | [+0.479, +0.537] | 0.0002 | 0.0003 | True |
| L|chain_one|k2|regret_on_minus_off | -0.356 | [-0.397, -0.317] | 0.0002 | 0.0003 | True |
| L|chain_one|k2|exact_on_minus_off | +0.678 | [+0.650, +0.707] | 0.0002 | 0.0003 | True |
| L|chain_one|false_copy_on_minus_off | -0.025 | [-0.030, -0.020] | 0.0002 | 0.0003 | True |
| R|chain_one|k1|regret_on_minus_off | -0.023 | [-0.037, -0.008] | 0.0036 | 0.0046 | True |
| R|chain_one|k1|exact_on_minus_off | +0.003 | [-0.015, +0.022] | 0.7413 | 0.7688 | False |
| R|chain_one|k2|regret_on_minus_off | -0.019 | [-0.036, -0.003] | 0.0234 | 0.0285 | True |
| R|chain_one|k2|exact_on_minus_off | +0.001 | [-0.019, +0.021] | 0.9543 | 0.9543 | False |
| R|chain_one|k4|regret_on_minus_off | -0.019 | [-0.039, +0.001] | 0.0614 | 0.0716 | True |
| R|chain_one|k4|exact_on_minus_off | +0.011 | [-0.002, +0.024] | 0.1192 | 0.1335 | False |
| R|chain_one|false_copy_on_minus_off | -0.050 | [-0.059, -0.042] | 0.0002 | 0.0003 | True |
| P|chain_m1|k1|regret_on_minus_off | -0.040 | [-0.100, +0.019] | 0.1896 | 0.2042 | False |
| P|chain_m1|k1|exact_on_minus_off | +0.297 | [+0.243, +0.349] | 0.0002 | 0.0003 | True |
| P|chain_m1|k2|regret_on_minus_off | -0.107 | [-0.162, -0.053] | 0.0002 | 0.0003 | True |
| P|chain_m1|k2|exact_on_minus_off | +0.333 | [+0.288, +0.376] | 0.0002 | 0.0003 | True |
| P|chain_m1|k4|regret_on_minus_off | -0.204 | [-0.253, -0.157] | 0.0002 | 0.0003 | True |
| P|chain_m1|k4|exact_on_minus_off | +0.201 | [+0.168, +0.235] | 0.0002 | 0.0003 | True |
| P|chain_m1|false_copy_on_minus_off | +0.558 | [+0.511, +0.605] | 0.0002 | 0.0003 | True |
| L|chain_m1|k1|regret_on_minus_off | -0.294 | [-0.334, -0.256] | 0.0002 | 0.0003 | True |
| L|chain_m1|k1|exact_on_minus_off | +0.548 | [+0.520, +0.577] | 0.0002 | 0.0003 | True |
| L|chain_m1|k2|regret_on_minus_off | -0.354 | [-0.396, -0.314] | 0.0002 | 0.0003 | True |
| L|chain_m1|k2|exact_on_minus_off | +0.733 | [+0.708, +0.758] | 0.0002 | 0.0003 | True |
| L|chain_m1|k4|regret_on_minus_off | -0.494 | [-0.542, -0.447] | 0.0002 | 0.0003 | True |
| L|chain_m1|k4|exact_on_minus_off | +0.744 | [+0.717, +0.771] | 0.0002 | 0.0003 | True |
| L|chain_m1|false_copy_on_minus_off | -0.029 | [-0.035, -0.022] | 0.0002 | 0.0003 | True |
| L|chain_one|k4|exact_on_minus_off|all_posted_plans | +0.623 | [+0.584, +0.661] | 0.0002 | 0.0003 | True |
| L|chain_one|k4|exact_on_minus_off|any_void_plans | +0.721 | [+0.676, +0.766] | 0.0002 | 0.0003 | True |

## Pre-declared directional predictions

- D1 prerequisite copy shortcut persists: P chain_m1 false-copy ON - OFF > 0 with the 95% CI above 0: **holds** ({"point": 0.558437358017214, "ci95": [0.5107401341339123, 0.6050130812972828]})
- D2 prerequisite chain_m1 k=4 exact(ON) <= copy-chain floor + 0.10: **holds** ({"on": 0.205, "copy_floor": 0.11666666666666667})
- D3 prerequisite chain_m1 applicable state-changing step accuracy ON <= OFF (correct-input steps; descriptive point; arm-specific step sets): **holds** ({"off": 0.5867208672086721, "on": 0.3529929577464789})
- D4 register: no large change (chain_one k=4: |exact ON - OFF| <= 0.05 and |regret ON - OFF| <= 0.1; can hold alongside a small BH-significant change): **holds** ({"exact_delta": 0.010833333333333334, "regret_delta": -0.018937742138902345, "regret_delta_bh_significant": true})

## Step accuracy on correct-input chain steps (descriptive; arm-specific step sets)

- L|chain_one: add/posted 0.533->0.912 (n 366/829); add/void 0.116->1.000 (n 43/106); scale/posted 0.566->0.966 (n 572/1214); scale/void 0.084->1.000 (n 83/171); transfer/posted 0.302->0.849 (n 769/1662); transfer/void 0.037->1.000 (n 109/230)
- R|chain_one: add/nowrap 0.904->0.897 (n 499/496); add/wrap 0.000->0.318 (n 22/22); conditional/gt 0.472->0.536 (n 288/302); conditional/le 0.510->0.512 (n 308/303); mix/nowrap 0.022->0.022 (n 185/178); mix/wrap 0.013->0.009 (n 537/543); swap 0.996->0.993 (n 672/679)
- P|chain_m1: applicable, state-changing 0.587->0.353 (n 738/1136); applicable, unchanged 0.027->0.767 (n 299/473); inapplicable 0.032->0.879 (n 632/1131)
- L|chain_m1: add/posted 0.423->0.948 (n 352/844); add/void 0.128->1.000 (n 47/106); scale/posted 0.616->0.957 (n 565/1271); scale/void 0.145->1.000 (n 76/179); transfer/posted 0.288->0.899 (n 761/1726); transfer/void 0.054->1.000 (n 111/237)
- Ledger chain_one k=4 exact by plan type (pooled; parent-paired ON - OFF): all_posted: 0.043->0.669 (n 703; +0.623 [+0.584, +0.661], 150 parents); any_void: 0.010->0.738 (n 497; +0.721 [+0.676, +0.766], 148 parents)

## Floors (normalised regret; copy-chain exact floor)

- L|k1: F_copy_s0 1.001, F_first_only 0.000, F_ignore_status 0.095, F_last_only 0.000, F_touched 0.284, copy_chain_exact_floor 0.191
- L|k2: F_copy_s0 1.001, F_first_only 0.372, F_ignore_status 0.100, F_last_only 0.368, F_touched 0.380, copy_chain_exact_floor 0.032
- L|k4: F_copy_s0 1.002, F_first_only 0.648, F_ignore_status 0.120, F_last_only 0.677, F_touched 0.577, copy_chain_exact_floor 0.002
- R|k1: F_copy_s0 1.000, F_first_only 0.000, F_last_only 0.000, F_touched 0.416, copy_chain_exact_floor 0.048
- R|k2: F_copy_s0 1.001, F_first_only 0.497, F_last_only 0.526, F_touched 0.508, copy_chain_exact_floor 0.037
- R|k4: F_copy_s0 1.000, F_first_only 0.802, F_last_only 0.818, F_touched 0.670, copy_chain_exact_floor 0.003
- P|k1: F_copy_s0 0.993, F_first_only 0.000, F_ignore_pre 0.188, F_last_only 0.000, copy_chain_exact_floor 0.513
- P|k2: F_copy_s0 0.999, F_first_only 0.369, F_ignore_pre 0.168, F_last_only 0.351, copy_chain_exact_floor 0.298
- P|k4: F_copy_s0 1.004, F_first_only 0.625, F_ignore_pre 0.254, F_last_only 0.618, copy_chain_exact_floor 0.117

Inputs: {"code_sha256": {"scripts/confirm2_analysis.py": "1c158a4e72bab2e8638ff03b526fa56a1d1b0dc75c266f1ea8ef820f685f7b1a", "scripts/t2_analysis.py": "d699dd5afa432dfef8b1c39cd14af26b92750237c7108328f7a61403decbcaf0", "scripts/confirm2_gen.py": "4253a42039c4a1bb95f412666ad4bc1804244d5b10e6e9fa393c99a488659380", "rcwmt/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "rcwmt/bank.py": "b08b0229c566f9f5b50c7c0c5738f7fff90b768a827e7e32c9880ff69352592f", "rcwmt/capture.py": "7b0a19ee4aefda183f22336521fa39d5d3694bf1d60b5b950d769fa942ce255a", "rcwmt/controls.py": "08a9e4479fe621d5bb2d2a6e9d970eef122ae6850e7b894206b6a7c57002a702", "rcwmt/data.py": "f77270ee04d64985e5b6fa9a88f989a04b0b7ed8cc75f69a4984ca7e7593beb4", "rcwmt/decide.py": "148e98ab528dcd283f6896f07f01135ede658e124d09d08d293e7d3645463643", "rcwmt/decisions.py": "bd6a4531622d2479f1b7fe35fcb11338d831c1ab2e9402b515f9a35ef5237188", "rcwmt/env.py": "92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2", "rcwmt/fidelity.py": "da03bbca4962c05b13eb13dfa0414578344bbe2674015dc93070465ca4e1bb0e", "rcwmt/generation.py": "a4052088a3b6e4d1bc0b59c947599ebce76cdd0bb37230761cb817fa92df7b19", "rcwmt/heads.py": "b3bbb95696fc40fa69cb1d3f60e2547f2fde8305862af2e5b934b073170d58b6", "rcwmt/lease_vendored.py": "d1576bb9296c6bd033dd677c0f8cbe18c9bc24939f75bba30d740429b2ec36c9", "rcwmt/offtask.py": "5fcc70500d21b9043ff93a054b53bdcf29fb2e052eeb722a99fb388fdd41ba0e", "rcwmt/prereg1.py": "1068d66dab6e1c87a255440eae376c1a884ac7dfc3cedac85478e30775cacd35", "rcwmt/pvr.py": "f7b1717fb276252900e75c3a6bf2f835d772498f9b75762aa82faea799e94b94", "rcwmt/runtime.py": "9360541286dc5648e6fd71c217135914b80fd4cef665364c47688cb07f0a12ca"}, "numpy": "2.4.6", "torch_imported": false, "seconds": 6.1}


# RCX-E07/E08 report (SMOKE — plumbing check only, numbers are not results)

CPU mechanism tier. A learned heuristic is an external specialist (tier B); no Rineya-core claim.
Descriptive unless a frozen preregistration names the comparison.

## Data

| split | instances by n | rows by n |
|---|---|---|
| train | {'4': 2, '5': 2, '6': 2} | {'4': 121, '5': 1713, '6': 471} |
| val | {'4': 1, '5': 1, '6': 1} | {'4': 30, '5': 49, '6': 44} |
| test | {'5': 2, '8': 2, '10': 2, '12': 2} | {'5': 12, '8': 11, '10': 10, '12': 14} |

Goal-class allocation (classes per size): n=4: train 3, val 1, test 0; n=5: train 3, val 1, test 2; n=6: train 8, val 2, test 0
Instance generation filters: `{"train": {"draws": 39, "generate_none": 6, "rejected_class": 5, "excluded_signature": 14, "duplicate": 8, "kept": 6}, "val": {"draws": 9, "rejected_class": 4, "excluded_invariant": 2, "kept": 3}, "test": {"draws": 25, "rejected_class": 14, "generate_none": 3, "kept": 8}}`
Row-level exclusions (train/val builds): `{"train:4": {"rows_excluded_signature": 129, "dead": 0, "rows": 121, "instances": 2, "dead_states_in_space": 0}, "train:5": {"dead": 0, "rows": 1713, "rows_excluded_signature": 19, "instances": 2, "dead_states_in_space": 0}, "train:6": {"dead": 0, "rows": 471, "instances": 2, "dead_states_in_space": 0, "rows_excluded_invariant": 129}, "val:4": {"dead": 0, "rows": 30, "rows_excluded_signature": 20, "instances": 1, "dead_states_in_space": 0}, "val:5": {"dead": 0, "rows": 49, "rows_excluded_signature": 1, "instances": 1, "dead_states_in_space": 0}, "val:6": {"dead": 0, "rows": 44, "rows_excluded_invariant": 6, "instances": 1, "dead_states_in_space": 0}}`
Test labelling: `{"test_states_sampled": 56, "test_states_deduped": 52, "test_rows_missing": 1, "test_rows_excluded_invariant": 5}`

## Training

| model | seed | params | epochs | steps | best epoch | best val MAE | cap hit | wall s |
|---|---|---|---|---|---|---|---|---|
| gnn | 20261276 | 123969 | 2 | 74 | 0 | 1.1924 | False | 0.9 |
| mlp | 20261276 | 158721 | 2 | 74 | 0 | 1.6833 | False | 20.2 |

Parameters: RelGNN 123969, FlatMLP 158721.

## E07 — Policy accuracy (argmin successor optimal; ties strict)

| model | 5 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|
| GNN (mean ± sd over 1 seeds) | 0.667 | 0.909 | 1.000 | 0.846 | 0.667 | 0.912 |
| MLP (mean ± sd over 1 seeds) | 0.667 | 0.636 | 0.300 | 0.538 | 0.667 | 0.500 |
| unmet_goal_facts | 0.083 | 0.091 | 0.100 | 0.077 | 0.083 | 0.088 |
| misplaced_blocks | 0.750 | 0.273 | 0.400 | 0.615 | 0.750 | 0.441 |
| misplaced_lb | 0.750 | 0.273 | 0.400 | 0.615 | 0.750 | 0.441 |
| h_add | 0.917 | 0.909 | 0.900 | 0.846 | 0.917 | 0.882 |
| h0 | 0.083 | 0.091 | 0.100 | 0.000 | 0.083 | 0.059 |

## E07 — Spearman rank correlation with h*

| model | 5 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|
| GNN (mean ± sd over 1 seeds) | 0.700 | 0.883 | 0.902 | 0.982 | 0.700 | 0.975 |
| MLP (mean ± sd over 1 seeds) | 0.343 | 0.613 | -0.018 | 0.308 | 0.343 | 0.232 |
| unmet_goal_facts | – | 0.721 | – | 0.835 | – | 0.792 |
| misplaced_blocks | 0.955 | 0.830 | 0.525 | 0.951 | 0.955 | 0.875 |
| misplaced_lb | 0.936 | 0.773 | 0.546 | 0.952 | 0.936 | 0.880 |
| h_add | 0.967 | 0.772 | 0.652 | 0.871 | 0.967 | 0.868 |
| h0 | – | – | – | – | – | – |

## E07 — Tie rate among argmin successors

| model | 5 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|
| GNN (mean ± sd over 1 seeds) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MLP (mean ± sd over 1 seeds) | 0.000 | 0.000 | 0.000 | 0.077 | 0.000 | 0.029 |
| unmet_goal_facts | 0.917 | 0.818 | 0.900 | 0.769 | 0.917 | 0.824 |
| misplaced_blocks | 0.417 | 0.909 | 0.700 | 0.769 | 0.417 | 0.794 |
| misplaced_lb | 0.417 | 0.909 | 0.700 | 0.769 | 0.417 | 0.794 |
| h_add | 0.333 | 0.273 | 0.100 | 0.462 | 0.333 | 0.294 |
| h0 | 0.917 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 |

## E07 — MAE vs h*

| model | 5 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|
| GNN (mean ± sd over 1 seeds) | 1.335 | 2.391 | 2.797 | 3.627 | 1.335 | 2.983 |
| MLP (mean ± sd over 1 seeds) | 4.586 | 6.771 | 16.509 | 10.479 | 4.586 | 11.053 |

## E07 — RMSE vs h*

| model | 5 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|
| GNN (mean ± sd over 1 seeds) | 1.750 | 2.682 | 2.960 | 3.924 | 1.750 | 3.285 |
| MLP (mean ± sd over 1 seeds) | 4.956 | 7.864 | 16.770 | 11.486 | 4.956 | 12.376 |

## E07 — Calibrated exact (round(pred) == h*)

| model | 5 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|
| GNN (mean ± sd over 1 seeds) | 0.333 | 0.000 | 0.000 | 0.077 | 0.333 | 0.029 |
| MLP (mean ± sd over 1 seeds) | 0.000 | 0.091 | 0.000 | 0.000 | 0.000 | 0.029 |

States per group (labelled / policy-evaluable): 5: 12/12, 8: 11/11, 10: 10/10, 12: 13/13, ID: 12/12, OOD: 34/34
Missing labels (A* budget): {'5': 0, '8': 0, '10': 0, '12': 1, 'ID': 0, 'OOD': 1}

## E07 — paired policy-accuracy differences (instance-clustered bootstrap, seeds averaged within instance)

| scope | comparison | baseline | mean base | mean GNN | diff | 95% CI | instances |
|---|---|---|---|---|---|---|---|
| 8 | gnn_minus_mlp | MLP | 0.650 | 0.917 | +0.267 | [+0.200, +0.333] | 2 |
| 8 | gnn_minus_best_handcrafted | h_add | 0.917 | 0.917 | +0.000 | [+0.000, +0.000] | 2 |
| 10 | gnn_minus_mlp | MLP | 0.300 | 1.000 | +0.700 | [+0.600, +0.800] | 2 |
| 10 | gnn_minus_best_handcrafted | h_add | 0.900 | 1.000 | +0.100 | [+0.000, +0.200] | 2 |
| 12 | gnn_minus_mlp | MLP | 0.548 | 0.845 | +0.298 | [+0.167, +0.429] | 2 |
| 12 | gnn_minus_best_handcrafted | h_add | 0.857 | 0.845 | -0.012 | [-0.167, +0.143] | 2 |
| OOD | gnn_minus_mlp | MLP | 0.499 | 0.921 | +0.421 | [+0.249, +0.617] | 6 |
| OOD | gnn_minus_best_handcrafted | h_add | 0.891 | 0.921 | +0.029 | [-0.060, +0.124] | 6 |
| ID (context) | gnn_minus_mlp | MLP | 0.714 | 0.714 | +0.000 | [+0.000, +0.000] | 2 |
| ID (context) | gnn_minus_best_handcrafted | h_add | 0.900 | 0.714 | -0.186 | [-0.571, +0.200] | 2 |

Instance-macro means (each instance weighted equally) differ from the state-pooled table above.

## E08 — GBFS success, max_expansions = 100

| heuristic | 8 | 10 | all | rate [Wilson 95%] | median exp | mean plan len | mean subopt | mean s |
|---|---|---|---|---|---|---|---|---|
| gnn | 0/2 | 0/2 | 0/4 | 0.000 [0.000, 0.490] | 100 | – | – (n=0) | 0.186 |
| mlp | 0/2 | 0/2 | 0/4 | 0.000 [0.000, 0.490] | 100 | – | – (n=0) | 0.048 |
| unmet_goal_facts | 0/2 | 0/2 | 0/4 | 0.000 [0.000, 0.490] | 100 | – | – (n=0) | 0.004 |
| misplaced_blocks | 1/2 | 1/2 | 2/4 | 0.500 [0.150, 0.850] | 94 | 20.0 | 1.000 (n=2) | 0.015 |
| h_add | 1/2 | 2/2 | 3/4 | 0.750 [0.301, 0.954] | 88 | 31.3 | 1.458 (n=3) | 0.074 |
| h0 | 0/2 | 0/2 | 0/4 | 0.000 [0.000, 0.490] | 100 | – | – (n=0) | 0.005 |

## E08 — GBFS success, max_expansions = 300

| heuristic | 8 | 10 | all | rate [Wilson 95%] | median exp | mean plan len | mean subopt | mean s |
|---|---|---|---|---|---|---|---|---|
| gnn | 0/2 | 0/2 | 0/4 | 0.000 [0.000, 0.490] | 300 | – | – (n=0) | 0.565 |
| mlp | 0/2 | 0/2 | 0/4 | 0.000 [0.000, 0.490] | 300 | – | – (n=0) | 0.131 |
| unmet_goal_facts | 1/2 | 0/2 | 1/4 | 0.250 [0.046, 0.699] | 300 | 22.0 | 1.375 (n=1) | 0.014 |
| misplaced_blocks | 2/2 | 2/2 | 4/4 | 1.000 [0.510, 1.000] | 142 | 20.5 | 1.000 (n=4) | 0.024 |
| h_add | 2/2 | 2/2 | 4/4 | 1.000 [0.510, 1.000] | 88 | 31.0 | 1.510 (n=4) | 0.094 |

## E08 — paired success, exact McNemar (GNN seed-1 vs baseline)

| comparison | budget | size | baseline only | GNN only | n | p |
|---|---|---|---|---|---|---|
| gnn_vs_unmet_goal_facts | 100 | 8 | 0 | 0 | 2 | 1.0000 |
| gnn_vs_unmet_goal_facts | 100 | 10 | 0 | 0 | 2 | 1.0000 |
| gnn_vs_unmet_goal_facts | 100 | all | 0 | 0 | 4 | 1.0000 |
| gnn_vs_misplaced_blocks | 100 | 8 | 1 | 0 | 2 | 1.0000 |
| gnn_vs_misplaced_blocks | 100 | 10 | 1 | 0 | 2 | 1.0000 |
| gnn_vs_misplaced_blocks | 100 | all | 2 | 0 | 4 | 0.5000 |
| gnn_vs_unmet_goal_facts | 300 | 8 | 1 | 0 | 2 | 1.0000 |
| gnn_vs_unmet_goal_facts | 300 | 10 | 0 | 0 | 2 | 1.0000 |
| gnn_vs_unmet_goal_facts | 300 | all | 1 | 0 | 4 | 1.0000 |
| gnn_vs_misplaced_blocks | 300 | 8 | 2 | 0 | 2 | 0.5000 |
| gnn_vs_misplaced_blocks | 300 | 10 | 2 | 0 | 2 | 0.5000 |
| gnn_vs_misplaced_blocks | 300 | all | 4 | 0 | 4 | 0.1250 |

Optimal-length coverage (A* within 20000 expansions): {'8': '2/2', '10': '2/2'}; mean A* seconds {"8": 0.006, "10": 0.011}.
FlatMLP cannot encode n > 12 (n_max = 12): its 14- and 15-block cells are n/a by construction.

## Provenance

```json
{
 "code_sha256": {
  "experiments/RCX-E07/run_e07.py": "816dd795f4687b8e22ebf8948d9f76f940b6006eb335863f75077ead9afebad3",
  "rcx/stats.py": "ad1c38527f3b7ee9f95685d4e982ff7aacf99bc0cd73ed839601fd6d40aea259",
  "rcx/guard.py": "08fa782d848528a318b0a6fd9699e4d7c301aacdd03e9fdf7c0e6d2e6ca26054",
  "rcx/bw/__init__.py": "4f403acd75709867f8dc7f6952a4e63e702ebaa40001dc2cb7024a6feb1440aa",
  "rcx/bw/bwm_import.py": "84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e",
  "rcx/bw/excl.py": "b985a6635c842aa25ef3bb6b56f422139ed4a8ae37169762570ed9b808e9265b",
  "rcx/bw/space.py": "29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4",
  "rcx/bw/data.py": "43d10ede8b8a4ae6c926ae900a97923c0ec83be4c17fbbaf152cf340ba99921c",
  "rcx/bw/encode.py": "32980f141eaf0fac4757ee253d22c59c55f889ae98c26eddd37eef055b928a8a",
  "rcx/bw/models.py": "4c2e3242eadc4ee155d8ddedb5f4188ac3aa5f70a19061145ff0e2fbee118f52",
  "rcx/bw/heuristics.py": "2f80f7a751e23589cf25c9e57342685bcb2c516cc8492d95d8ad3ada3e03191d",
  "rcx/bw/search.py": "10a1b499446e732ff860f479f30639270296a8ac2443abb4024ed50def9b758f"
 },
 "exclusions_json_sha256": "acb56fd970098b7bb30ea16177d66e5d503a30e7db3059c4a5703b4e95096407",
 "bwm_pins": {
  "bwmn/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "bwmn/blocks.py": "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83",
  "scripts/blocks_val_large.py": "6a66d489944150b50294a75f8c07671ad722b51ede09409ddee1d2cfc91f2356",
  "scripts/bw_distill.py": "2b2d518206b2e36c252ac9b87f991a73ff701b54404758e0363670d573d49fa1"
 },
 "seeds": {
  "train_instances": 20261270,
  "val_instances": 20261271,
  "test_ood_instances": 20261272,
  "e08_instances": 20261273,
  "state_sampling": 20261274,
  "split": 20261275,
  "model_init": 20261276,
  "tests": 20261280,
  "model_seeds": [
   20261276
  ]
 },
 "config": {
  "train_sizes": [
   4,
   5,
   6
  ],
  "train_per_size": 2,
  "train_all_states_max_n": 5,
  "train_states_large": 300,
  "goal_frac": 0.5,
  "val_sizes": [
   4,
   5,
   6
  ],
  "val_per_size": 1,
  "val_states": 50,
  "id_sizes": [
   5
  ],
  "ood_sizes": [
   8,
   10,
   12
  ],
  "test_per_size": 2,
  "k_states": 6,
  "discovery_draws": 600,
  "min_class_count": 5,
  "val_class_frac": 0.2,
  "test_class_frac": 0.25,
  "space_max_n": 8,
  "astar_budget": 60000,
  "seeds": [
   20261276
  ],
  "epochs": 2,
  "batch_size": 64,
  "lr": 0.002,
  "time_cap_s": 25.0,
  "e08_sizes": [
   8,
   10
  ],
  "e08_per_size": 2,
  "budgets": [
   100,
   300
  ],
  "h0_budgets": [
   100
  ],
  "opt_budget": 20000,
  "tie_tol": 1e-05,
  "bootstrap_resamples": 200
 },
 "smoke": true,
 "threads": 2,
 "python": "3.11.15",
 "torch": "2.6.0+cu124",
 "numpy": "2.4.6",
 "platform": "Windows-10-10.0.26200-SP0",
 "wall_s": {
  "data": 18.03,
  "train": 24.16,
  "e07": 2.09,
  "e08": 4.81
 },
 "guard_wait_s": 80.01600000000008,
 "bwm_bytecode_check": {
  "new_pycache_dirs": [],
  "new_or_changed_pyc": [],
  "ours": []
 }
}
```

## Estimated full-run CPU time (extrapolated from this smoke run; indicative)

```json
{
 "data_train_val_labels_s": 83.9397617712523,
 "data_test_labels_s": 1192.5889694902594,
 "data_test_labels_note": "per-state A* seconds measured on few smoke states: {5: 4.5833333333333334e-05, 8: 2.636363636363637e-05, 10: 0.25176600000000005, 12: 0.5963571428571427}; max {5: 0.00017, 8: 6e-05, 10: 0.70159, 12: 2.58136}; the n=12 cost is heavy-tailed (budget 300000 expansions caps it)",
 "train_s": 715.3907796488566,
 "train_steps_per_model": 17340,
 "train_rows_full_upper": 92429,
 "e07_s": 23.62041545739585,
 "e08_search_upper_s": 1295.5564624168512,
 "e08_opt_astar_upper_s": 863.9075342484394,
 "total_expected_s": 2922.4904178507213,
 "assumptions": "expected: half of the E08 GBFS runs exhaust their budget, 30% of optimal-length A* runs hit the budget; upper: every search exhausts its budget and test labelling costs 3x the smoke per-state A* time; expansion cost scales with n^2 (ground actions)",
 "total_upper_s": 6560.181862013574
}
```

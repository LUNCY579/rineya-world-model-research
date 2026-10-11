# RCX-E07/E08 report (full)

CPU mechanism tier. A learned heuristic is an external specialist (tier B); no Rineya-core claim.
Descriptive unless a frozen preregistration names the comparison.

## Data

| split | instances by n | rows by n |
|---|---|---|
| train | {'4': 3, '5': 3, '6': 5, '7': 11} | {'4': 181, '5': 2413, '6': 17380, '7': 90533} |
| val | {'5': 1, '6': 2, '7': 3} | {'5': 186, '6': 571, '7': 782} |
| test | {'5': 15, '6': 15, '8': 15, '10': 15, '12': 15} | {'5': 359, '6': 411, '8': 548, '10': 673, '12': 796} |

Goal-class allocation (classes per size): n=4: train 3, val 1, test 0; n=5: train 3, val 1, test 2; n=6: train 5, val 2, test 3; n=7: train 11, val 3, test 0
Instance generation filters: `{"train": {"draws": 293, "generate_none": 16, "excluded_signature": 38, "duplicate": 73, "rejected_class": 78, "excluded_selection_signature": 4, "excluded_invariant": 62, "kept": 22}, "val": {"draws": 2465, "rejected_class": 1413, "generate_none": 262, "excluded_signature": 447, "excluded_selection_signature": 102, "exhausted": 1, "excluded_invariant": 158, "duplicate": 77, "kept": 6}, "test": {"draws": 173, "rejected_class": 62, "generate_none": 8, "excluded_signature": 1, "excluded_selection_signature": 1, "excluded_invariant": 26, "kept": 75}}`
Row-level exclusions (train/val builds): `{"train:4": {"rows_excluded_signature": 183, "dead": 0, "rows": 181, "rows_excluded_selection_signature": 11, "instances": 3, "dead_states_in_space": 0}, "train:5": {"rows_excluded_selection_signature": 148, "dead": 0, "rows": 2413, "rows_excluded_signature": 37, "instances": 3, "dead_states_in_space": 0}, "train:6": {"dead": 0, "rows": 17380, "rows_excluded_selection_signature": 55, "rows_excluded_invariant": 17850, "instances": 5, "dead_states_in_space": 0}, "train:7": {"dead": 0, "rows": 90533, "rows_excluded_invariant": 19467, "instances": 11, "dead_states_in_space": 0}, "val:5": {"dead": 0, "rows": 186, "rows_excluded_selection_signature": 106, "rows_excluded_signature": 8, "instances": 1, "dead_states_in_space": 0}, "val:6": {"dead": 0, "rows": 571, "rows_excluded_selection_signature": 29, "instances": 2, "dead_states_in_space": 0}, "val:7": {"dead": 0, "rows": 782, "rows_excluded_invariant": 118, "instances": 3, "dead_states_in_space": 0}}`
Test labelling: `{"test_states_sampled": 4575, "test_states_deduped": 3063, "test_rows_missing": 53, "test_rows_excluded_signature": 11, "test_rows_excluded_selection_signature": 21, "test_rows_excluded_invariant": 244}`

## Training

| model | seed | params | epochs | steps | best epoch | best val MAE | cap hit | wall s |
|---|---|---|---|---|---|---|---|---|
| gnn | 20261276 | 123969 | 12 | 20724 | 8 | 0.0848 | False | 163.0 |
| gnn | 20261277 | 123969 | 12 | 20724 | 11 | 0.0716 | False | 157.1 |
| gnn | 20261278 | 123969 | 12 | 20724 | 10 | 0.0716 | False | 158.8 |
| mlp | 20261276 | 158721 | 12 | 20724 | 10 | 0.5586 | False | 26.1 |
| mlp | 20261277 | 158721 | 12 | 20724 | 11 | 0.5412 | False | 46.5 |
| mlp | 20261278 | 158721 | 12 | 20724 | 11 | 0.5505 | False | 66.1 |

Parameters: RelGNN 123969, FlatMLP 158721.

## E07 — Policy accuracy (argmin successor optimal; ties strict)

| model | 5 | 6 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|---|
| GNN (mean ± sd over 3 seeds) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.995 ± 0.002 | 1.000 ± 0.000 | 0.998 ± 0.001 |
| MLP (mean ± sd over 3 seeds) | 0.990 ± 0.003 | 0.988 ± 0.002 | 0.709 ± 0.012 | 0.513 ± 0.012 | 0.499 ± 0.019 | 0.989 ± 0.003 | 0.563 ± 0.008 |
| unmet_goal_facts | 0.126 | 0.151 | 0.082 | 0.054 | 0.054 | 0.139 | 0.062 |
| misplaced_blocks | 0.419 | 0.683 | 0.600 | 0.633 | 0.377 | 0.560 | 0.526 |
| misplaced_lb | 0.419 | 0.683 | 0.600 | 0.633 | 0.377 | 0.560 | 0.526 |
| h_add | 0.729 | 0.868 | 0.819 | 0.903 | 0.821 | 0.803 | 0.848 |
| h0 | 0.084 | 0.112 | 0.038 | 0.029 | 0.028 | 0.099 | 0.031 |

## E07 — Spearman rank correlation with h* (within size; ID/OOD = mean over sizes)

| model | 5 | 6 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|---|
| GNN (mean ± sd over 3 seeds) | 0.989 ± 0.000 | 0.993 ± 0.000 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.988 ± 0.001 | 0.991 ± 0.000 | 0.995 ± 0.000 |
| MLP (mean ± sd over 3 seeds) | 0.962 ± 0.002 | 0.983 ± 0.001 | 0.883 ± 0.007 | 0.793 ± 0.009 | 0.711 ± 0.012 | 0.973 ± 0.001 | 0.796 ± 0.005 |
| unmet_goal_facts | -0.187 | 0.463 | 0.712 | 0.890 | 0.837 | 0.138 | 0.813 |
| misplaced_blocks | 0.666 | 0.873 | 0.948 | 0.987 | 0.869 | 0.770 | 0.935 |
| misplaced_lb | 0.662 | 0.889 | 0.951 | 0.988 | 0.861 | 0.775 | 0.933 |
| h_add | 0.458 | 0.707 | 0.847 | 0.961 | 0.788 | 0.583 | 0.865 |
| h0 | – | – | – | – | – | – | – |

## E07 — Tie rate among argmin successors

| model | 5 | 6 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|---|
| GNN (mean ± sd over 3 seeds) | 0.000 ± 0.000 | 0.011 ± 0.001 | 0.042 ± 0.027 | 0.027 ± 0.013 | 0.022 ± 0.007 | 0.006 ± 0.001 | 0.030 ± 0.014 |
| MLP (mean ± sd over 3 seeds) | 0.000 ± 0.000 | 0.007 ± 0.002 | 0.021 ± 0.006 | 0.049 ± 0.006 | 0.037 ± 0.004 | 0.004 ± 0.001 | 0.037 ± 0.005 |
| unmet_goal_facts | 0.771 | 0.890 | 0.889 | 0.929 | 0.887 | 0.835 | 0.902 |
| misplaced_blocks | 0.687 | 0.602 | 0.768 | 0.699 | 0.815 | 0.642 | 0.763 |
| misplaced_lb | 0.687 | 0.602 | 0.768 | 0.699 | 0.815 | 0.642 | 0.763 |
| h_add | 0.131 | 0.239 | 0.370 | 0.367 | 0.233 | 0.189 | 0.317 |
| h0 | 0.964 | 0.971 | 0.995 | 0.997 | 0.996 | 0.967 | 0.996 |

## E07 — MAE vs h*

| model | 5 | 6 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|---|
| GNN (mean ± sd over 3 seeds) | 0.048 ± 0.010 | 0.042 ± 0.007 | 0.055 ± 0.012 | 0.242 ± 0.031 | 0.625 ± 0.036 | 0.044 ± 0.007 | 0.335 ± 0.017 |
| MLP (mean ± sd over 3 seeds) | 0.436 ± 0.004 | 0.405 ± 0.043 | 2.820 ± 0.211 | 6.753 ± 0.247 | 13.161 ± 0.174 | 0.419 ± 0.024 | 8.080 ± 0.182 |

## E07 — RMSE vs h*

| model | 5 | 6 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|---|
| GNN (mean ± sd over 3 seeds) | 0.061 ± 0.012 | 0.077 ± 0.011 | 0.099 ± 0.003 | 0.558 ± 0.100 | 1.130 ± 0.084 | 0.070 ± 0.009 | 0.770 ± 0.067 |
| MLP (mean ± sd over 3 seeds) | 0.580 ± 0.015 | 0.554 ± 0.047 | 3.452 ± 0.189 | 7.981 ± 0.197 | 14.114 ± 0.139 | 0.567 ± 0.018 | 10.026 ± 0.132 |

## E07 — Calibrated exact (round(pred) == h*)

| model | 5 | 6 | 8 | 10 | 12 | ID | OOD |
|---|---|---|---|---|---|---|---|
| GNN (mean ± sd over 3 seeds) | 1.000 ± 0.000 | 0.997 ± 0.001 | 0.993 ± 0.002 | 0.874 ± 0.017 | 0.703 ± 0.013 | 0.998 ± 0.001 | 0.842 ± 0.004 |
| MLP (mean ± sd over 3 seeds) | 0.670 ± 0.018 | 0.711 ± 0.072 | 0.116 ± 0.042 | 0.041 ± 0.014 | 0.011 ± 0.004 | 0.692 ± 0.042 | 0.050 ± 0.018 |

States per group (labelled / policy-evaluable): 5: 359/358, 6: 411/410, 8: 548/548, 10: 673/662, 12: 743/741, ID: 770/768, OOD: 1964/1951
Missing labels (A* budget): {'5': 0, '6': 0, '8': 0, '10': 0, '12': 53, 'ID': 0, 'OOD': 53}

## E07 — paired policy-accuracy differences (instance-clustered bootstrap, seeds averaged within instance)

| scope | comparison | baseline | mean base | mean GNN | diff | 95% CI | instances |
|---|---|---|---|---|---|---|---|
| 8 | gnn_minus_mlp | MLP | 0.730 | 1.000 | +0.270 | [+0.220, +0.317] | 15 |
| 8 | gnn_minus_best_handcrafted | h_add | 0.818 | 1.000 | +0.182 | [+0.123, +0.243] | 15 |
| 10 | gnn_minus_mlp | MLP | 0.539 | 0.998 | +0.459 | [+0.387, +0.535] | 15 |
| 10 | gnn_minus_best_handcrafted | h_add | 0.908 | 0.998 | +0.090 | [+0.051, +0.133] | 15 |
| 12 | gnn_minus_mlp | MLP | 0.505 | 0.994 | +0.489 | [+0.394, +0.573] | 15 |
| 12 | gnn_minus_best_handcrafted | h_add | 0.824 | 0.994 | +0.170 | [+0.108, +0.236] | 15 |
| OOD | gnn_minus_mlp | MLP | 0.591 | 0.998 | +0.406 | [+0.355, +0.456] | 45 |
| OOD | gnn_minus_best_handcrafted | h_add | 0.850 | 0.998 | +0.148 | [+0.113, +0.184] | 45 |
| ID (context) | gnn_minus_mlp | MLP | 0.990 | 1.000 | +0.010 | [+0.004, +0.016] | 30 |
| ID (context) | gnn_minus_best_handcrafted | h_add | 0.803 | 1.000 | +0.197 | [+0.157, +0.236] | 30 |

Instance-macro means (each instance weighted equally) differ from the state-pooled table above.

## E08 — GBFS success, max_expansions = 500

| heuristic | 8 | 10 | 12 | 14 | 15 | all | rate [Wilson 95%] | median exp | mean plan len | mean subopt | mean s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 20/20 | 15/20 | 9/20 | 6/20 | 11/20 | 61/100 | 0.610 [0.512, 0.700] | 52 | 23.5 | 1.043 (n=60) | 0.223 |
| mlp | 14/20 | 0/20 | 0/20 | n/a | n/a | 14/60* | 0.233 [0.144, 0.354] | 500 | 28.6 | 1.703 (n=14) | 0.139 |
| unmet_goal_facts | 8/20 | 5/20 | 0/20 | 0/20 | 1/20 | 14/100 | 0.140 [0.085, 0.221] | 500 | 24.7 | 1.327 (n=14) | 0.027 |
| misplaced_blocks | 19/20 | 15/20 | 10/20 | 11/20 | 13/20 | 68/100 | 0.680 [0.583, 0.763] | 219 | 23.1 | 1.012 (n=67) | 0.049 |
| h_add | 20/20 | 18/20 | 10/20 | 6/20 | 9/20 | 63/100 | 0.630 [0.532, 0.718] | 390 | 46.1 | 1.858 (n=60) | 0.574 |
| h0 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/100 | 0.000 [0.000, 0.037] | 500 | – | – (n=0) | 0.029 |

## E08 — GBFS success, max_expansions = 2000

| heuristic | 8 | 10 | 12 | 14 | 15 | all | rate [Wilson 95%] | median exp | mean plan len | mean subopt | mean s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn | 20/20 | 16/20 | 11/20 | 6/20 | 12/20 | 65/100 | 0.650 [0.553, 0.736] | 52 | 24.2 | 1.056 (n=64) | 0.737 |
| mlp | 16/20 | 3/20 | 0/20 | n/a | n/a | 19/60* | 0.317 [0.213, 0.442] | 2000 | 37.8 | 2.001 (n=19) | 0.526 |
| unmet_goal_facts | 14/20 | 13/20 | 1/20 | 2/20 | 1/20 | 31/100 | 0.310 [0.228, 0.406] | 2000 | 33.4 | 1.595 (n=31) | 0.104 |
| misplaced_blocks | 20/20 | 19/20 | 15/20 | 17/20 | 16/20 | 87/100 | 0.870 [0.790, 0.922] | 219 | 24.6 | 1.012 (n=85) | 0.115 |
| h_add | 20/20 | 20/20 | 18/20 | 16/20 | 15/20 | 89/100 | 0.890 [0.814, 0.937] | 390 | 53.1 | 2.024 (n=85) | 0.942 |

## E08 — paired success, exact McNemar (GNN seed-1 vs baseline)

| comparison | budget | size | baseline only | GNN only | n | p |
|---|---|---|---|---|---|---|
| gnn_vs_unmet_goal_facts | 500 | 8 | 0 | 12 | 20 | 0.0005 |
| gnn_vs_unmet_goal_facts | 500 | 10 | 0 | 10 | 20 | 0.0020 |
| gnn_vs_unmet_goal_facts | 500 | 12 | 0 | 9 | 20 | 0.0039 |
| gnn_vs_unmet_goal_facts | 500 | 14 | 0 | 6 | 20 | 0.0313 |
| gnn_vs_unmet_goal_facts | 500 | 15 | 0 | 10 | 20 | 0.0020 |
| gnn_vs_unmet_goal_facts | 500 | all | 0 | 47 | 100 | 0.0000 |
| gnn_vs_misplaced_blocks | 500 | 8 | 0 | 1 | 20 | 1.0000 |
| gnn_vs_misplaced_blocks | 500 | 10 | 3 | 3 | 20 | 1.0000 |
| gnn_vs_misplaced_blocks | 500 | 12 | 4 | 3 | 20 | 1.0000 |
| gnn_vs_misplaced_blocks | 500 | 14 | 7 | 2 | 20 | 0.1797 |
| gnn_vs_misplaced_blocks | 500 | 15 | 4 | 2 | 20 | 0.6875 |
| gnn_vs_misplaced_blocks | 500 | all | 18 | 11 | 100 | 0.2649 |
| gnn_vs_unmet_goal_facts | 2000 | 8 | 0 | 6 | 20 | 0.0313 |
| gnn_vs_unmet_goal_facts | 2000 | 10 | 3 | 6 | 20 | 0.5078 |
| gnn_vs_unmet_goal_facts | 2000 | 12 | 1 | 11 | 20 | 0.0063 |
| gnn_vs_unmet_goal_facts | 2000 | 14 | 2 | 6 | 20 | 0.2891 |
| gnn_vs_unmet_goal_facts | 2000 | 15 | 0 | 11 | 20 | 0.0010 |
| gnn_vs_unmet_goal_facts | 2000 | all | 6 | 40 | 100 | 0.0000 |
| gnn_vs_misplaced_blocks | 2000 | 8 | 0 | 0 | 20 | 1.0000 |
| gnn_vs_misplaced_blocks | 2000 | 10 | 3 | 0 | 20 | 0.2500 |
| gnn_vs_misplaced_blocks | 2000 | 12 | 6 | 2 | 20 | 0.2891 |
| gnn_vs_misplaced_blocks | 2000 | 14 | 11 | 0 | 20 | 0.0010 |
| gnn_vs_misplaced_blocks | 2000 | 15 | 4 | 0 | 20 | 0.1250 |
| gnn_vs_misplaced_blocks | 2000 | all | 24 | 2 | 100 | 0.0000 |

Optimal-length coverage (A* within 300000 expansions): {'8': '20/20', '10': '20/20', '12': '20/20', '14': '17/20', '15': '18/20'}; mean A* seconds {"8": 0.002, "10": 0.032, "12": 0.168, "14": 2.187, "15": 1.186}.
FlatMLP cannot encode n > 12 (n_max = 12): its 14- and 15-block cells are n/a by construction. A `*` marks a pooled cell that therefore covers n <= 12 only; that row's rate, median expansions, plan length, subopt and seconds are over the same reduced set and are not comparable with the other rows' pooled values.

## Provenance

```json
{
 "code_sha256": {
  "experiments/RCX-E07/run_e07.py": "b1f6bb2215aa78e3ba39b0abdc59b36cea9f0befa2015a15e08dbdc8fb07a43e",
  "rcx/stats.py": "ad1c38527f3b7ee9f95685d4e982ff7aacf99bc0cd73ed839601fd6d40aea259",
  "rcx/guard.py": "08fa782d848528a318b0a6fd9699e4d7c301aacdd03e9fdf7c0e6d2e6ca26054",
  "rcx/bw/__init__.py": "4f403acd75709867f8dc7f6952a4e63e702ebaa40001dc2cb7024a6feb1440aa",
  "rcx/bw/bwm_import.py": "84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e",
  "rcx/bw/excl.py": "eb31d9eb5cd98e9b5372248ff03104cd4016254e472aaa8b07e8a6abc6f4e252",
  "rcx/bw/space.py": "29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4",
  "rcx/bw/data.py": "3f85401cd0bbe1d35fe49fb93678b11541768588953071975ba5e2f7e11b6b9d",
  "rcx/bw/encode.py": "32980f141eaf0fac4757ee253d22c59c55f889ae98c26eddd37eef055b928a8a",
  "rcx/bw/models.py": "4c2e3242eadc4ee155d8ddedb5f4188ac3aa5f70a19061145ff0e2fbee118f52",
  "rcx/bw/heuristics.py": "2f80f7a751e23589cf25c9e57342685bcb2c516cc8492d95d8ad3ada3e03191d",
  "rcx/bw/search.py": "10a1b499446e732ff860f479f30639270296a8ac2443abb4024ed50def9b758f"
 },
 "exclusions_json_sha256": "2f211de379352ef53450dd0a091987408e19fb9dad1db4d8ea24f549c6178167",
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
   20261276,
   20261277,
   20261278
  ],
  "bootstrap": "20261275:bootstrap"
 },
 "selection_set_exclusions": {
  "runs/BW_data_v2/instances_val.json": {
   "sha256": "ca1dd0e027343862763fd09452390861616426a955e8c6ec8836d3659c8d0488",
   "instances": 400,
   "unique_signatures": 181,
   "already_in_exclusions": 0,
   "block_counts": {
    "4": 67,
    "5": 284,
    "6": 49
   }
  },
  "runs/BW_data/instances_val.json": {
   "sha256": "baa6fc7665e194bd84ba7dc3d312dc93c1e896194373694f23c3fad0bb3b17b6",
   "instances": 400,
   "unique_signatures": 136,
   "already_in_exclusions": 0,
   "block_counts": {
    "4": 49,
    "5": 288,
    "6": 63
   }
  }
 },
 "config": {
  "train_sizes": [
   4,
   5,
   6,
   7
  ],
  "train_per_size": 40,
  "train_all_states_max_n": 6,
  "train_states_large": 10000,
  "goal_frac": 0.5,
  "val_sizes": [
   4,
   5,
   6,
   7
  ],
  "val_per_size": 10,
  "val_states": 300,
  "id_sizes": [
   5,
   6
  ],
  "ood_sizes": [
   8,
   10,
   12
  ],
  "test_per_size": 15,
  "k_states": 60,
  "discovery_draws": 3000,
  "min_class_count": 5,
  "val_class_frac": 0.2,
  "test_class_frac": 0.25,
  "space_max_n": 8,
  "astar_budget": 300000,
  "seeds": [
   20261276,
   20261277,
   20261278
  ],
  "epochs": 12,
  "batch_size": 64,
  "lr": 0.002,
  "time_cap_s": 3600.0,
  "e08_sizes": [
   8,
   10,
   12,
   14,
   15
  ],
  "e08_per_size": 20,
  "budgets": [
   500,
   2000
  ],
  "h0_budgets": [
   500
  ],
  "opt_budget": 300000,
  "tie_tol": 1e-05,
  "bootstrap_resamples": 5000
 },
 "smoke": false,
 "threads": 4,
 "python": "3.11.15",
 "torch": "2.6.0+cu124",
 "numpy": "2.4.6",
 "platform": "Windows-10-10.0.26200-SP0",
 "wall_s": {
  "data": 647.82,
  "train": 625.67,
  "e07": 7.87,
  "e08": 365.37
 },
 "guard_wait_s": 60.01499999999942,
 "bwm_bytecode_check": {
  "new_pycache_dirs": [],
  "new_or_changed_pyc": [],
  "ours": []
 }
}
```

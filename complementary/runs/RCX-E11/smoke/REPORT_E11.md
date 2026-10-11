# RCX-E11 report (SMOKE: plumbing only, not a result)

Prereg: {'draft': 'experiments\\RCX-E11\\PREREG_E11_DRAFT.md', 'draft_sha256': '2b95f42fc8d61777d3b8c8ac7d8f42aeb7a53805b956d670820dc7b517678fa2', 'frozen': 'experiments\\RCX-E11\\PREREG_E11.md', 'frozen_sha256': None, 'pinned_sha256': None, 'status': 'DRAFT'}

Custody: BWM bytecode checks of earlier invocations 0, all clean: True

## E11a macro discovery and reuse

Decision: **INCONCLUSIVE** ({"outcome": "INCONCLUSIVE", "primary_significant": false, "M_better": true, "attribution_ok": false, "compute_matched_secondary": {"h_calls": {"threshold": 1000, "exact": false, "p_exact": null, "M_better": true, "evaluable": false}, "prim_steps": {"threshold": 1000, "exact": false, "p_exact": null, "M_better": true, "evaluable": false}, "saving_claim_allowed": false}, "M_minus_G_primary_cell": 1, "g_clause": false})

Library M: `pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?0 | on(?0,?1),on(?2,?0)`; `stack ?0 ?1; pick-up ?2; stack ?2 ?0 | on(?0,?1),on(?2,?0)`; `pick-up ?0; stack ?0 ?1; pick-up ?2 | on(?0,?1),on(?2,?0)`

Library R: `unstack ?0 ?1; stack ?0 ?2; pick-up ?1; stack ?1 ?0 | on(?0,?2),on(?1,?0)`; `unstack ?0 ?1; put-down ?0; unstack ?1 ?2 | on(?0,?1),on(?2,?0)`; `put-down ?0; unstack ?1 ?2; put-down ?1 | on(?0,?2),on(?1,?0)`

Library S: `unstack ?0 ?1; stack ?0 ?2; pick-up ?3; stack ?3 ?1 | on(?0,?2),on(?3,?1)`; `stack ?0 ?1; pick-up ?2; stack ?2 ?3 | on(?0,?1),on(?2,?3)`; `unstack ?0 ?1; stack ?0 ?2; pick-up ?3 | on(?0,?2)`

Library G: `pick-up ?0; stack ?0 ?1 | on(?0,?1)`; `unstack ?0 ?1; stack ?0 ?2 | on(?0,?2)`

| arm \| heur \| algo | @20 | @50 | @100 | @200 | children/exp @max | subopt (solved) |
|---|---|---|---|---|---|---|
| P \| A3 \| gbfs | 0/4 | 2/4 | 2/4 | 2/4 | 4.4 | 1.000 |
| M \| A3 \| gbfs | 1/4 | 3/4 | 3/4 | 3/4 | 5.4 | 1.000 |
| R \| A3 \| gbfs | 1/4 | 2/4 | 3/4 | 3/4 | 5.2 | 1.000 |
| S \| A3 \| gbfs | 0/4 | 2/4 | 2/4 | 2/4 | 5.3 | 1.108 |
| G \| A3 \| gbfs | 1/4 | 2/4 | 3/4 | 3/4 | 5.8 | 1.000 |
| P \| A3 \| beam4 | 0/4 | 0/4 | 1/4 | 2/4 | 5.1 | 1.000 |
| M \| A3 \| beam4 | 0/4 | 1/4 | 2/4 | 2/4 | 5.5 | 1.000 |
| R \| A3 \| beam4 | 0/4 | 0/4 | 1/4 | 2/4 | 5.6 | 1.000 |
| S \| A3 \| beam4 | 0/4 | 0/4 | 2/4 | 3/4 | 5.5 | 1.508 |
| G \| A3 \| beam4 | 0/4 | 0/4 | 3/4 | 3/4 | 6.3 | 1.051 |
| P \| misplaced_blocks \| gbfs | 0/4 | 0/4 | 1/4 | 1/4 | 5.6 | 1.000 |
| M \| misplaced_blocks \| gbfs | 0/4 | 1/4 | 1/4 | 2/4 | 5.8 | 1.000 |
| R \| misplaced_blocks \| gbfs | 0/4 | 0/4 | 1/4 | 1/4 | 5.7 | 1.000 |
| S \| misplaced_blocks \| gbfs | 0/4 | 0/4 | 2/4 | 2/4 | 5.2 | 1.385 |
| G \| misplaced_blocks \| gbfs | 0/4 | 1/4 | 2/4 | 2/4 | 5.9 | 1.038 |
| P \| misplaced_blocks \| beam4 | 0/4 | 0/4 | 0/4 | 0/4 | 4.6 | - |
| M \| misplaced_blocks \| beam4 | 0/4 | 0/4 | 0/4 | 0/4 | 4.5 | - |
| R \| misplaced_blocks \| beam4 | 0/4 | 0/4 | 0/4 | 0/4 | 4.6 | - |
| S \| misplaced_blocks \| beam4 | 0/4 | 0/4 | 0/4 | 2/4 | 5.1 | 2.022 |
| G \| misplaced_blocks \| beam4 | 0/4 | 0/4 | 0/4 | 1/4 | 4.7 | 2.385 |

PRIMARY (M vs P, M|A3|gbfs @50): 3 vs 2 of 4; discordant M-only 1 / P-only 0; exact McNemar p = 1
Attribution M_vs_R: 3 vs 2; p = 1, Holm 1
Attribution M_vs_S: 3 vs 2; p = 1, Holm 1

Compute-matched view (A3, gbfs; descriptive; derived from the 200-expansion runs; `*` = censored for an arm, a lower bound, no test):

| unit | threshold | P | M | R | S | G | M vs P exact McNemar p |
|---|---|---|---|---|---|---|---|
| h_calls | 500 | 2 | 3 | 3 | 2 | 3 | 1 |
| h_calls | 1000 | 2* | 3* | 3* | 2* | 3* | censored |
| h_calls | 2500 | 2* | 3* | 3* | 2* | 3* | censored |
| h_calls | 5000 | 2* | 3* | 3* | 2* | 3* | censored |
| prim_steps | 500 | 2 | 3 | 3 | 2 | 2 | 1 |
| prim_steps | 1000 | 2* | 3 | 3 | 2 | 3 | censored |
| prim_steps | 2500 | 2* | 3* | 3* | 2* | 3* | censored |
| prim_steps | 5000 | 2* | 3* | 3* | 2* | 3* | censored |

A3 hook accounting (per-run hooks): {"runs": 40, "without_counters": 0, "calls_ne_h_calls": 0, "cache_hits": 0, "invalid_states": 0, "fallbacks": 0, "failsafe": 0}

Gates: {"G1_library_sizes": {"M": 3, "R": 3, "S": 3}, "G1_ok": true, "G2_invalid_found_plans": 0, "G3_sym_mismatch": 0, "ok": true}; monotonicity ok: True; A3 hook: {"model_sha256": "9cc417a3075391e4fc8022f0dddd811e90fb4647c6a88dcd81361e9bb2c13f04", "weights_sha256": "c9b1d376c14a810098e3f1c30605e99670acebfe4cd236e7511104135d61c8e0", "hybrid_lambda": 1.0, "invalid_policy": "fallback"}

## E11b compositional subgoal policy

Decision: **REVISE** ({"outcome": "REVISE", "learned_success": 3, "bu_success": 3, "replicates_all_negative": false, "vs_rnd_point_negative": true, "vs_rf_point_le_0": false, "rf_clause": false})

| stratum | method | success | mean len (solved) | subopt (A*-covered) | geo ratio vs BU |
|---|---|---|---|---|---|
| pool_OOD | learned_s20261259 | 3/3 | 25.3 | 1.449 (3) | 1.100 |
| pool_OOD | bottom_up | 3/3 | 22.7 | 1.275 (3) | 1.000 |
| pool_OOD | ready_first | 3/3 | 21.3 | 1.208 (3) | 0.941 |
| pool_OOD | random_order | 3/3 | 44.7 | 2.595 (3) | 1.833 |
| pool_OOD | gc_gbfs | 0/3 | - | - (0) | - |
| S | learned_s20261259 | 1/1 | 6.0 | 1.000 (1) | 1.000 |
| S | bottom_up | 1/1 | 6.0 | 1.000 (1) | 1.000 |
| S | ready_first | 1/1 | 6.0 | 1.000 (1) | 1.000 |
| S | random_order | 1/1 | 10.0 | 1.667 (1) | 1.667 |
| S | gc_gbfs | 0/1 | - | - (0) | - |
| T | learned_s20261259 | 1/1 | 20.0 | 1.000 (1) | 0.833 |
| T | bottom_up | 1/1 | 24.0 | 1.200 (1) | 1.000 |
| T | ready_first | 1/1 | 20.0 | 1.000 (1) | 0.833 |
| T | random_order | 1/1 | 26.0 | 1.300 (1) | 1.083 |
| T | gc_gbfs | 0/1 | - | - (0) | - |
| H | learned_s20261259 | 1/1 | 22.0 | 1.222 (1) | 1.222 |
| H | bottom_up | 1/1 | 18.0 | 1.000 (1) | 1.000 |
| H | ready_first | 1/1 | 18.0 | 1.000 (1) | 1.000 |
| H | random_order | 1/1 | 38.0 | 2.111 (1) | 2.111 |
| H | gc_gbfs | 0/1 | - | - (0) | - |
| TH | learned_s20261259 | 1/1 | 34.0 | 2.125 (1) | 1.308 |
| TH | bottom_up | 1/1 | 26.0 | 1.625 (1) | 1.000 |
| TH | ready_first | 1/1 | 26.0 | 1.625 (1) | 1.000 |
| TH | random_order | 1/1 | 70.0 | 4.375 (1) | 2.692 |
| TH | gc_gbfs | 0/1 | - | - (0) | - |

Goal-count GBFS success by budget: {"pool_OOD": {"20": 0, "50": 0, "100": 0, "200": 0}, "S": {"20": 0, "50": 0, "100": 0, "200": 0}, "T": {"20": 0, "50": 0, "100": 0, "200": 0}, "H": {"20": 0, "50": 0, "100": 0, "200": 0}, "TH": {"20": 0, "50": 0, "100": 0, "200": 0}}

PRIMARY (learned r0 vs BU, OOD pool n = 3): mean log length ratio 0.0955 [-0.1823, 0.2683] (geo ratio 1.100)
vs random order: -0.5103 [-0.7221, -0.2624]; vs ready-first: 0.1563
Replicates (log ratio vs BU): {"learned_s20261259": 0.0955377084209587}

Achiever-level ceiling OPT_A (descriptive): OOD-pool geo ratio vs OPT_A {"learned_s20261259": "1.236", "bottom_up": "1.124", "ready_first": "1.057", "random_order": "2.059"}; fraction of BU's excess over OPT_A closed: {"n": "3", "n_pool": "3", "learned_s20261259": "-0.820", "ready_first": "0.522", "random_order": "-5.203"}

Label accuracy (top-1 in the training-target set `achiever`; `|vs_plan` / `|vs_achiever` = in A(s) / A_A(s); `unseen` = rows whose renaming class is not in an earlier split, the in-distribution basis):
```
{
 "test_id": {
  "all": {
   "rows": 88,
   "learned_s20261259": 0.8977272727272727,
   "learned_s20261259|vs_plan": 0.9431818181818182,
   "learned_s20261259|vs_achiever": 0.8977272727272727,
   "bottom_up": 0.8409090909090909,
   "bottom_up|vs_plan": 0.9431818181818182,
   "bottom_up|vs_achiever": 0.8409090909090909,
   "ready_first": 0.8636363636363636,
   "ready_first|vs_plan": 0.9204545454545454,
   "ready_first|vs_achiever": 0.8636363636363636,
   "random_order": 0.7272727272727273,
   "random_order|vs_plan": 0.7954545454545454,
   "random_order|vs_achiever": 0.7272727272727273
  },
  "unseen": {
   "rows": 58,
   "learned_s20261259": 0.8448275862068966,
   "learned_s20261259|vs_plan": 0.9137931034482759,
   "learned_s20261259|vs_achiever": 0.8448275862068966,
   "bottom_up": 0.7931034482758621,
   "bottom_up|vs_plan": 0.9310344827586207,
   "bottom_up|vs_achiever": 0.7931034482758621,
   "ready_first": 0.8275862068965517,
   "ready_first|vs_plan": 0.896551724137931,
   "ready_first|vs_achiever": 0.8275862068965517,
   "random_order": 0.7413793103448276,
   "random_order|vs_plan": 0.7931034482758621,
   "random_order|vs_achiever": 0.7413793103448276
  }
 },
 "ood_small": {
  "all": {
   "rows": 50,
   "learned_s20261259": 0.76,
   "learned_s20261259|vs_plan": 0.94,
   "learned_s20261259|vs_achiever": 0.76,
   "bottom_up": 0.6,
   "bottom_up|vs_plan": 0.96,
   "bottom_up|vs_achiever": 0.6,
   "ready_first": 0.72,
   "ready_first|vs_plan": 0.94,
   "ready_first|vs_achiever": 0.72,
   "random_order": 0.58,
   "random_order|vs_plan": 0.56,
   "random_order|vs_achiever": 0.58
  },
  "unseen": {
   "rows": 50,
   "learned_s20261259": 0.76,
   "learned_s20261259|vs_plan": 0.94,
   "learned_s20261259|vs_achiever": 0.76,
   "bottom_up": 0.6,
   "bottom_up|vs_plan": 0.96,
   "bottom_up|vs_achiever": 0.6,
   "ready_first": 0.72,
   "ready_first|vs_plan": 0.94,
   "ready_first|vs_achiever": 0.72,
   "random_order": 0.58,
   "random_order|vs_plan": 0.56,
   "random_order|vs_achiever": 0.58
  }
 }
}
```


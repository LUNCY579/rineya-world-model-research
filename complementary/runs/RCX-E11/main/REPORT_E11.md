# RCX-E11 report (full run)

Prereg: {'draft': 'experiments\\RCX-E11\\PREREG_E11_DRAFT.md', 'draft_sha256': '2b95f42fc8d61777d3b8c8ac7d8f42aeb7a53805b956d670820dc7b517678fa2', 'frozen': 'experiments\\RCX-E11\\PREREG_E11.md', 'frozen_sha256': 'c67f036d530ed1f961f84caf2e9b2e598562190ccf1722069379807046b5f40c', 'pinned_sha256': 'c67f036d530ed1f961f84caf2e9b2e598562190ccf1722069379807046b5f40c', 'status': 'FROZEN'}

Custody: BWM bytecode checks of earlier invocations 0, all clean: True

## E11a macro discovery and reuse

Decision: **ADVANCE** ({"outcome": "ADVANCE", "primary_significant": true, "M_better": true, "attribution_ok": true, "compute_matched_secondary": {"h_calls": {"threshold": 1000, "exact": true, "p_exact": 7.629394531250002e-06, "M_better": true, "evaluable": true}, "prim_steps": {"threshold": 1000, "exact": true, "p_exact": 0.0013122558593749976, "M_better": true, "evaluable": true}, "saving_claim_allowed": true}, "M_minus_G_primary_cell": 14, "g_clause": false})

Library M: `pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?0 | on(?0,?1),on(?2,?0)`; `stack ?0 ?1; pick-up ?2; stack ?2 ?0 | on(?0,?1),on(?2,?0)`; `pick-up ?0; stack ?0 ?1; pick-up ?2 | on(?0,?1),on(?2,?0)`; `pick-up ?0; stack ?0 ?1 | on(?0,?1)`; `stack ?0 ?1; pick-up ?2; stack ?2 ?0; pick-up ?3 | on(?0,?1),on(?2,?0),on(?3,?2)`; `stack ?0 ?1; pick-up ?2 | on(?0,?1),on(?2,?0)`; `unstack ?0 ?1; put-down ?0 | -`; `pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?3 | on(?0,?1),on(?2,?3)`

Library R: `unstack ?0 ?1; stack ?0 ?2; pick-up ?1; stack ?1 ?0 | on(?0,?2),on(?1,?0)`; `unstack ?0 ?1; put-down ?0; unstack ?1 ?2 | on(?0,?1),on(?2,?0)`; `put-down ?0; unstack ?1 ?2; put-down ?1 | on(?0,?2),on(?1,?0)`; `put-down ?0; pick-up ?1 | on(?1,?0)`; `unstack ?0 ?1; stack ?0 ?2; pick-up ?1; stack ?1 ?3 | on(?1,?3),on(?2,?1),on(?3,?0)`; `put-down ?0; unstack ?1 ?2 | on(?1,?2),on(?2,?0)`; `pick-up ?0; stack ?0 ?1 | -`; `put-down ?0; unstack ?1 ?2; stack ?1 ?0; unstack ?2 ?3 | on(?2,?1),on(?3,?2)`

Library S: `unstack ?0 ?1; stack ?0 ?2; pick-up ?3; stack ?3 ?1 | on(?0,?2),on(?3,?1)`; `stack ?0 ?1; pick-up ?2; stack ?2 ?3 | on(?0,?1),on(?2,?3)`; `unstack ?0 ?1; stack ?0 ?2; pick-up ?3 | on(?0,?2)`; `put-down ?0; pick-up ?1 | -`; `put-down ?0; unstack ?1 ?2; put-down ?1; pick-up ?2 | -`; `put-down ?0; unstack ?1 ?2 | -`; `stack ?0 ?1; unstack ?2 ?3 | on(?0,?1)`; `put-down ?0; unstack ?1 ?2; put-down ?1; unstack ?2 ?3 | on(?1,?2)`

Library G: `pick-up ?0; stack ?0 ?1 | on(?0,?1)`; `unstack ?0 ?1; stack ?0 ?2 | on(?0,?2)`

| arm \| heur \| algo | @50 | @200 | @500 | @2000 | children/exp @max | subopt (solved) |
|---|---|---|---|---|---|---|
| P \| A3 \| gbfs | 34/120 | 52/120 | 63/120 | 76/120 | 4.1 | 1.062 |
| M \| A3 \| gbfs | 75/120 | 77/120 | 82/120 | 85/120 | 11.1 | 1.077 |
| R \| A3 \| gbfs | 53/120 | 69/120 | 74/120 | 83/120 | 11.2 | 1.131 |
| S \| A3 \| gbfs | 42/120 | 63/120 | 69/120 | 79/120 | 7.6 | 1.173 |
| G \| A3 \| gbfs | 49/120 | 63/120 | 71/120 | 84/120 | 5.4 | 1.037 |
| P \| A3 \| beam4 | 0/120 | 43/120 | 58/120 | 70/120 | 4.5 | 3.041 |
| M \| A3 \| beam4 | 41/120 | 69/120 | 70/120 | 73/120 | 9.1 | 1.520 |
| R \| A3 \| beam4 | 1/120 | 63/120 | 71/120 | 71/120 | 10.1 | 1.367 |
| S \| A3 \| beam4 | 18/120 | 59/120 | 68/120 | 74/120 | 8.4 | 2.368 |
| G \| A3 \| beam4 | 5/120 | 61/120 | 63/120 | 65/120 | 5.5 | 1.398 |
| P \| misplaced_blocks \| gbfs | 7/120 | 34/120 | 61/120 | 83/120 | 4.6 | 1.011 |
| M \| misplaced_blocks \| gbfs | 31/120 | 85/120 | 102/120 | 115/120 | 7.3 | 1.013 |
| R \| misplaced_blocks \| gbfs | 14/120 | 39/120 | 55/120 | 82/120 | 8.2 | 1.021 |
| S \| misplaced_blocks \| gbfs | 25/120 | 90/120 | 103/120 | 115/120 | 7.4 | 1.461 |
| G \| misplaced_blocks \| gbfs | 25/120 | 64/120 | 84/120 | 105/120 | 5.2 | 1.099 |
| P \| misplaced_blocks \| beam4 | 0/120 | 27/120 | 59/120 | 98/120 | 4.1 | 4.287 |
| M \| misplaced_blocks \| beam4 | 7/120 | 37/120 | 62/120 | 101/120 | 4.3 | 4.144 |
| R \| misplaced_blocks \| beam4 | 0/120 | 23/120 | 23/120 | 23/120 | 11.8 | 1.046 |
| S \| misplaced_blocks \| beam4 | 4/120 | 51/120 | 87/120 | 113/120 | 5.7 | 3.808 |
| G \| misplaced_blocks \| beam4 | 3/120 | 43/120 | 80/120 | 114/120 | 4.5 | 3.472 |

PRIMARY (M vs P, M|A3|gbfs @200): 77 vs 52 of 120; discordant M-only 25 / P-only 0; exact McNemar p = 5.96e-08
Attribution M_vs_R: 77 vs 69; p = 0.00781, Holm 0.00781
Attribution M_vs_S: 77 vs 63; p = 0.000122, Holm 0.000244

Compute-matched view (A3, gbfs; descriptive; derived from the 2000-expansion runs; `*` = censored for an arm, a lower bound, no test):

| unit | threshold | P | M | R | S | G | M vs P exact McNemar p |
|---|---|---|---|---|---|---|---|
| h_calls | 500 | 49 | 76 | 44 | 51 | 60 | 1.49e-08 |
| h_calls | 1000 | 58 | 76 | 64 | 65 | 64 | 7.63e-06 |
| h_calls | 2500 | 69 | 82 | 75 | 75 | 75 | 0.000244 |
| h_calls | 5000 | 76* | 84 | 82 | 77 | 84 | censored |
| prim_steps | 500 | 43 | 41 | 19 | 25 | 51 | 0.824 |
| prim_steps | 1000 | 52 | 66 | 50 | 46 | 60 | 0.00131 |
| prim_steps | 2500 | 63 | 76 | 67 | 63 | 65 | 0.000244 |
| prim_steps | 5000 | 70 | 79 | 70 | 65 | 74 | 0.00391 |

A3 hook accounting (per-run hooks): {"runs": 1200, "without_counters": 0, "calls_ne_h_calls": 0, "cache_hits": 0, "invalid_states": 0, "fallbacks": 0, "failsafe": 0}

Gates: {"G1_library_sizes": {"M": 8, "R": 8, "S": 8}, "G1_ok": true, "G2_invalid_found_plans": 0, "G3_sym_mismatch": 0, "ok": true}; monotonicity ok: True; A3 hook: {"model_sha256": "9cc417a3075391e4fc8022f0dddd811e90fb4647c6a88dcd81361e9bb2c13f04", "weights_sha256": "c9b1d376c14a810098e3f1c30605e99670acebfe4cd236e7511104135d61c8e0", "hybrid_lambda": 1.0, "invalid_policy": "fallback"}

## E11b compositional subgoal policy

Decision: **REVISE** ({"outcome": "REVISE", "learned_success": 96, "bu_success": 96, "replicates_all_negative": true, "vs_rnd_point_negative": true, "vs_rf_point_le_0": false, "rf_clause": false})

| stratum | method | success | mean len (solved) | subopt (A*-covered) | geo ratio vs BU |
|---|---|---|---|---|---|
| pool_OOD | learned_s20261259 | 96/96 | 30.7 | 1.216 (93) | 0.929 |
| pool_OOD | learned_s816062654 | 96/96 | 30.1 | 1.191 (93) | 0.920 |
| pool_OOD | learned_s250692302 | 96/96 | 31.7 | 1.265 (93) | 0.960 |
| pool_OOD | bottom_up | 96/96 | 31.0 | 1.319 (93) | 1.000 |
| pool_OOD | ready_first | 96/96 | 28.1 | 1.200 (93) | 0.912 |
| pool_OOD | random_order | 96/96 | 52.4 | 2.180 (93) | 1.568 |
| pool_OOD | gc_gbfs | 20/96 | 28.1 | 1.251 (19) | - |
| S | learned_s20261259 | 32/32 | 12.2 | 1.033 (32) | 0.945 |
| S | learned_s816062654 | 32/32 | 12.2 | 1.033 (32) | 0.945 |
| S | learned_s250692302 | 32/32 | 12.2 | 1.033 (32) | 0.945 |
| S | bottom_up | 32/32 | 13.3 | 1.113 (32) | 1.000 |
| S | ready_first | 32/32 | 12.9 | 1.080 (32) | 0.977 |
| S | random_order | 32/32 | 16.3 | 1.333 (32) | 1.188 |
| S | gc_gbfs | 8/32 | 10.2 | 1.062 (8) | - |
| T | learned_s20261259 | 32/32 | 21.8 | 1.053 (31) | 0.805 |
| T | learned_s816062654 | 32/32 | 21.9 | 1.064 (31) | 0.813 |
| T | learned_s250692302 | 32/32 | 21.9 | 1.058 (31) | 0.808 |
| T | bottom_up | 32/32 | 27.6 | 1.336 (31) | 1.000 |
| T | ready_first | 32/32 | 23.1 | 1.123 (31) | 0.855 |
| T | random_order | 32/32 | 34.6 | 1.642 (31) | 1.205 |
| T | gc_gbfs | 7/32 | 15.7 | 1.093 (7) | - |
| H | learned_s20261259 | 32/32 | 43.0 | 1.485 (30) | 1.144 |
| H | learned_s816062654 | 32/32 | 40.8 | 1.387 (30) | 1.091 |
| H | learned_s250692302 | 32/32 | 44.4 | 1.576 (30) | 1.210 |
| H | bottom_up | 32/32 | 33.1 | 1.305 (30) | 1.000 |
| H | ready_first | 32/32 | 31.6 | 1.277 (30) | 0.963 |
| H | random_order | 32/32 | 66.3 | 2.541 (30) | 1.848 |
| H | gc_gbfs | 5/32 | 44.4 | 1.454 (4) | - |
| TH | learned_s20261259 | 32/32 | 27.4 | 1.123 (32) | 0.870 |
| TH | learned_s816062654 | 32/32 | 27.6 | 1.132 (32) | 0.877 |
| TH | learned_s250692302 | 32/32 | 28.8 | 1.173 (32) | 0.904 |
| TH | bottom_up | 32/32 | 32.3 | 1.315 (32) | 1.000 |
| TH | ready_first | 32/32 | 29.8 | 1.203 (32) | 0.922 |
| TH | random_order | 32/32 | 56.2 | 2.363 (32) | 1.730 |
| TH | gc_gbfs | 8/32 | 28.8 | 1.289 (8) | - |

Goal-count GBFS success by budget: {"pool_OOD": {"50": 1, "200": 8, "500": 11, "2000": 20}, "S": {"50": 1, "200": 2, "500": 3, "2000": 8}, "T": {"50": 0, "200": 5, "500": 5, "2000": 7}, "H": {"50": 0, "200": 2, "500": 2, "2000": 5}, "TH": {"50": 1, "200": 1, "500": 4, "2000": 8}}

PRIMARY (learned r0 vs BU, OOD pool n = 96): mean log length ratio -0.0739 [-0.1393, 0.0006] (geo ratio 0.929)
vs random order: -0.5236 [-0.5987, -0.4537]; vs ready-first: 0.0182
Replicates (log ratio vs BU): {"learned_s20261259": -0.07389780573230371, "learned_s816062654": -0.08375682804740504, "learned_s250692302": -0.04099667040868025}

Achiever-level ceiling OPT_A (descriptive): OOD-pool geo ratio vs OPT_A {"learned_s20261259": "1.136", "learned_s816062654": "1.125", "learned_s250692302": "1.174", "bottom_up": "1.223", "ready_first": "1.116", "random_order": "1.918"}; fraction of BU's excess over OPT_A closed: {"n": "96", "n_pool": "96", "learned_s20261259": "0.367", "learned_s816062654": "0.415", "learned_s250692302": "0.203", "ready_first": "0.457", "random_order": "-2.230"}

Label accuracy (top-1 in the training-target set `achiever`; `|vs_plan` / `|vs_achiever` = in A(s) / A_A(s); `unseen` = rows whose renaming class is not in an earlier split, the in-distribution basis):
```
{
 "test_id": {
  "all": {
   "rows": 6630,
   "learned_s20261259": 0.9986425339366516,
   "learned_s20261259|vs_plan": 0.8559577677224736,
   "learned_s20261259|vs_achiever": 0.9986425339366516,
   "learned_s816062654": 0.9987933634992459,
   "learned_s816062654|vs_plan": 0.8562594268476621,
   "learned_s816062654|vs_achiever": 0.9987933634992459,
   "learned_s250692302": 0.9986425339366516,
   "learned_s250692302|vs_plan": 0.8547511312217194,
   "learned_s250692302|vs_achiever": 0.9986425339366516,
   "bottom_up": 0.7324283559577677,
   "bottom_up|vs_plan": 0.8755656108597285,
   "bottom_up|vs_achiever": 0.7324283559577677,
   "ready_first": 0.8063348416289593,
   "ready_first|vs_plan": 0.8600301659125189,
   "ready_first|vs_achiever": 0.8063348416289593,
   "random_order": 0.604524886877828,
   "random_order|vs_plan": 0.6577677224736048,
   "random_order|vs_achiever": 0.604524886877828
  },
  "unseen": {
   "rows": 2326,
   "learned_s20261259": 0.9965606190885641,
   "learned_s20261259|vs_plan": 0.788907996560619,
   "learned_s20261259|vs_achiever": 0.9965606190885641,
   "learned_s816062654": 0.9965606190885641,
   "learned_s816062654|vs_plan": 0.7901977644024075,
   "learned_s816062654|vs_achiever": 0.9965606190885641,
   "learned_s250692302": 0.9961306964746346,
   "learned_s250692302|vs_plan": 0.7876182287188306,
   "learned_s250692302|vs_achiever": 0.9961306964746346,
   "bottom_up": 0.6302665520206363,
   "bottom_up|vs_plan": 0.8095442820292348,
   "bottom_up|vs_achiever": 0.6302665520206363,
   "ready_first": 0.7467755803955288,
   "ready_first|vs_plan": 0.7803095442820293,
   "ready_first|vs_achiever": 0.7467755803955288,
   "random_order": 0.48108340498710234,
   "random_order|vs_plan": 0.5507308684436801,
   "random_order|vs_achiever": 0.48108340498710234
  }
 },
 "ood_small": {
  "all": {
   "rows": 4310,
   "learned_s20261259": 0.9788863109048724,
   "learned_s20261259|vs_plan": 0.7508120649651973,
   "learned_s20261259|vs_achiever": 0.9788863109048724,
   "learned_s816062654": 0.9754060324825986,
   "learned_s816062654|vs_plan": 0.7385150812064966,
   "learned_s816062654|vs_achiever": 0.9754060324825986,
   "learned_s250692302": 0.97122969837587,
   "learned_s250692302|vs_plan": 0.745707656612529,
   "learned_s250692302|vs_achiever": 0.97122969837587,
   "bottom_up": 0.5663573085846868,
   "bottom_up|vs_plan": 0.8232018561484918,
   "bottom_up|vs_achiever": 0.5663573085846868,
   "ready_first": 0.6663573085846868,
   "ready_first|vs_plan": 0.8136890951276102,
   "ready_first|vs_achiever": 0.6663573085846868,
   "random_order": 0.3310904872389791,
   "random_order|vs_plan": 0.3679814385150812,
   "random_order|vs_achiever": 0.3310904872389791
  },
  "unseen": {
   "rows": 4310,
   "learned_s20261259": 0.9788863109048724,
   "learned_s20261259|vs_plan": 0.7508120649651973,
   "learned_s20261259|vs_achiever": 0.9788863109048724,
   "learned_s816062654": 0.9754060324825986,
   "learned_s816062654|vs_plan": 0.7385150812064966,
   "learned_s816062654|vs_achiever": 0.9754060324825986,
   "learned_s250692302": 0.97122969837587,
   "learned_s250692302|vs_plan": 0.745707656612529,
   "learned_s250692302|vs_achiever": 0.97122969837587,
   "bottom_up": 0.5663573085846868,
   "bottom_up|vs_plan": 0.8232018561484918,
   "bottom_up|vs_achiever": 0.5663573085846868,
   "ready_first": 0.6663573085846868,
   "ready_first|vs_plan": 0.8136890951276102,
   "ready_first|vs_achiever": 0.6663573085846868,
   "random_order": 0.3310904872389791,
   "random_order|vs_plan": 0.3679814385150812,
   "random_order|vs_achiever": 0.3310904872389791
  }
 }
}
```


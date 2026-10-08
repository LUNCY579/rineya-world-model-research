Audit result: none of C1–C4 is produced by an alignment, layer, split, target, ridge, metric or bootstrap bug. Every number I re-ran with the current code came out the same. I did find three problems that bend how C1 and C3 read, plus one minor issue in the P targets.

## CONFIRMED issues

**B1 — C3's "CKA 0.93–0.99" and "median |z| ~1.6" come from pooling the three families.**
- **Where:** `scripts/mechanism_report.py:50` puts the TEST one-frame items of all families into one matrix. `cka()` (lines 28–30) and `shift()` (lines 33–39) then run on that pooled matrix, so the per-dimension SD at line 37 includes between-family spread.
- **Demonstration** (`audit_10`, `audit_06`): 92–97% of Supa's answer-slot variance at L12–L28 is between-family. That family structure is trivially kept, and it dominates both metrics.

| Layer | Pooled CKA | CKA with family means removed | Ledger-only CKA |
|---|---|---|---|
| 12 | 0.991 | 0.756 | 0.898 |
| 16 | 0.925 | 0.864 | 0.926 |
| 20 | 0.979 | 0.437 | 0.793 |
| 24 | 0.937 | 0.616 | 0.604 |
| 28 | 0.910 | 0.622 | 0.622 |

- Median |z| within a single family is 5–13 (ledger 6.0–7.4). Pooled it is 1.4–2.3.
- So R5's "keeps their geometry" and "one to two SDs per dimension" do not hold within a family, including ledger, the family behind C1 and C2.

**B2 — The R5 table (C3's numbers) cannot be reproduced from any code in the repo.**
- R5 says "TEST, one-step frame". Recomputing on that set with `mechanism_report.shift`/`cka` gives:

| Layer | Rel L2 w/o top-5 dims (recomputed / claimed) | Median \|z\| (recomputed / claimed) | CKA (recomputed / claimed) |
|---|---|---|---|
| 12 | 0.248 / 0.24 | 1.40 / 1.03 | 0.991 / 0.991 |
| 16 | 0.586 / 0.53 | 2.26 / 1.58 | 0.925 / 0.953 |
| 20 | 0.826 / 0.81 | 1.92 / 1.61 | 0.979 / 0.965 |
| 24 | 1.087 / 1.15 | 1.72 / 1.61 | 0.937 / 0.925 |
| 28 | 0.711 / 0.73 | 1.46 / 1.36 | 0.910 / 0.937 |

- I searched every split × format combination (`audit_06b`) and none matches exactly. Neither does the stored `state_shift` in `h2b_controls_v2_EARLY.json` (0.1645 at L12).
- `RESULTS_PILOT.md` is timestamped 12:58 and `mechanism_report.py` 13:03, and there is no `mechanism.json`, so the table came from an unsaved ad-hoc run. The qualitative pattern does hold.

**B3 — C1's "P and R unparseable at k=1" is caused by output length caps and a strict parser, not by what Supa answers.**
- **Truncation:** `rcwmt/generation.py:147-148` caps P answers at 12 tokens. 84% of Supa's P k1 outputs hit the cap mid-answer, e.g. `'{"before":146, "after":14'`. 24% of R one-frame outputs hit the 56-token cap.
- **Strict parser:** `rcwmt/env.py:316-332` (keys checked at line 328) rejects `'{"after":146}'` and `'{"before":{..},"after":{..}}'`.
- **Scoring:** `rcwmt/fidelity.py:206-208` then counts every unparseable answer as all features wrong.
- **Demonstration** (`audit_04`): reading the `"after"` field leniently changes Supa's exact-state accuracy as follows.
  - P one-frame: 0 → 0.406.
  - R one-frame: 0 → 0.067.
  - R k1: 0 → 0.048.
- All of the P gain is Supa copying the start state on items that don't change. On items whose state does change, exact accuracy is 0.000 for P and at most 0.043 for R.
- **Effect on C1:** the substantive point (Supa can't generate the consequence) survives. The "unparseable" wording is an artifact, and the generative exact-Δ comparator for P one-frame is understated (0 instead of about 0.41, roughly the copy floor).

**B4 (minor, P only) — The P one-frame targets don't match the one-step prompt.**
- `SYSTEM_ONE` for P (`env.py:44-48`) says the supplied action is applicable. But gold for the `-one` items (`bank.py:141-146`) uses `env.step`, which treats an inapplicable action as a no-op.
- In P TEST, 44.2% of first actions are inapplicable. For 24.6% of items, the gold state differs from what the prompt literally instructs.
- This only affects R3's "P one-frame at copy floor" and the P generative scoring, not C1–C4's ledger or R claims.

## Not bugs: checked and found correct
- **Row alignment:**
  - Both `index.json` files are byte-identical, match the bank SHA, and list items in the same order as `bank.items()` (0 mismatches out of 19,200). The event rows match.
  - Every `candidate_table` row agrees with its prompt's `before`/`events` and gold state (0 mismatches out of 14,400), including the `-one` mapping.
  - No last-state row is all zeros or non-finite.
- **CPU forward pass (FP32, 12 items per arm, every family × format):** stored rows match recomputed `hidden_states` within 4.5e-4 relative error, at all 8 layers and at the event positions. A sibling item from the same parent differs by about 1e-2, so this check can tell items apart. Each stored row best-matches `hidden_states[L]` among all 29 layers, so the layer indices are right; layer 28 is the final-norm output. Event tokens are `'},{"'` and `'}'`.
- **`X_last` / `X_event`:** with shuffled and duplicated ids they equal direct indexing exactly.
- **Layers and weight delta:** layers 0, 4 and 8 are bit-identical across arms over all rows. Exactly 220 weight tensors differ, all in blocks 8–27. (Peripheral: the delta's relative norm is 0.048% of all parameters or 0.052% of blocks 8–27, not 0.056%.)
- **Splits:**
  - No parent appears in more than one split.
  - No (start state, first event) pair crosses splits.
  - The standardizer is fit on train only, and λ, layer and MLP learning rate are chosen on dev only.
- **Targets:**
  - `delta_of` matches an independent formula on all 14,400 rows.
  - The ledger rotation in `step_cf` is correct and keeps base rates (true vs counterfactual change rates within about 0.03).
  - The counterfactual differs from the true target on 86.5% of items.
- **Ridge:** the closed form equals a reference with an unpenalized intercept to 7e-16.
- **Metric and bootstrap:** the exact-match metric, `_parent_mean` and `paired_bootstrap` all give the right answers on toy cases.
- **Re-standardized transfer:** the math recovers perfectly on a synthetic affine-shifted copy (1.0 vs 0.76 without re-standardizing). At L8, where states are identical, transfer equals within-arm. The failure at L16 is real: feature accuracy is 0.67, below the 0.81 you get by predicting "no change".
- **CKA formula:** matches an HSIC reference implementation. The problem is only the pooling in B1.
- **Reproduced with current code:**
  - C1: ridge 0.846/0.812, MLP 0.913/0.918, parent-bootstrap CIs [−0.005, 0.072] and [−0.024, 0.014].
  - C2: selectivity −0.022 / 0.000 on n=180. Parent-level CIs: Sparse [−0.055, 0.014], Supa [−0.032, 0.028].
  - C4: answer slot and event-boundary tracking match exactly (L 0.865/0.524/0.322/0.192, R 0.452/0.115/0.024/0.010, P at copy floor).
- **C4 is not caused by how the probe is tuned:** the ridge picks λ and layer by per-feature accuracy but is scored on exact match. Choosing by dev exact match instead, or widening the λ grid to 1e-2…1e7, changes test exact-Δ by ≤0.04 and the decay stays.
- **Operational:**
  - **Stale index risk:** `run_arm.py:53-59` rewrites `index.json` before its skip check. It is identical today, but a changed bank would silently desync it from `last.npy`.
  - **Recapture risk:** Sparse's `CAPTURED.json` was written by hand. Reloading the model on CPU gives the same fingerprint (818be9c6b6a3), so the running master chain will skip recapturing.

All audit scripts are in `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/verification/`, from `audit_01_alignment.py` through `audit_11_selectivity_ci.py`.
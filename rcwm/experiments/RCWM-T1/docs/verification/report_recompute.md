**Verification of C1–C4 by independent recomputation (sklearn venv, CPU only)**

I rebuilt everything from the bank and the raw `last.npy` files. Transitions were reimplemented and match `rcwmt.env.step` on all 19,200 transitions. Both arms' indexes list items in the same order, and their splits match the bank.

Probes are standardised on train, with the layer (4–28) and the regularisation strength picked on dev. Ridge was run with two dev-selection rules: by exact match, and by mean per-feature accuracy (the second is the rule the repo uses). Logistic regression is one model per feature with C picked on dev. Confidence intervals are 95% paired bootstraps over the 26 test parents. Discrepancies over 0.03 are marked **FLAG**.

**C1 — Sparse and Supa decode the one-step change mask equally well (test exact match)**

| Case | Claimed | Mine |
|---|---|---|
| Ledger one-frame, ridge (per-feature selection) | 0.846 vs 0.812 | 0.846 vs 0.812; diff +0.034 [−0.005, +0.072], identical to the claim |
| Ledger one-frame, ridge (exact selection) | — | 0.846 vs 0.812; diff +0.034 [−0.019, +0.091] |
| Ledger one-frame, MLP | 0.913 vs 0.918 | 0.913 vs 0.918; diff −0.005 [−0.024, +0.014] |
| Ledger one-frame, logistic | — | 0.885 vs 0.894; diff −0.010 [−0.034, +0.014] |
| Ledger k=1, ridge | 0.875 vs 0.851 | 0.875 vs 0.851; diff +0.024 [−0.010, +0.058] |
| Ledger k=1, logistic | — | 0.885 vs 0.865, not significant |
| Register one-frame, ridge (per-feature selection) | 0.385 vs 0.370 | 0.385 vs 0.370; diff +0.014 [−0.058, +0.087] |
| Register one-frame, ridge (exact selection) | — | 0.385 vs 0.332; diff +0.053 [−0.014, +0.120] |
| Register one-frame, logistic | — | **FLAG** 0.428 vs 0.346; diff +0.082 [+0.019, +0.144], p≈0.015 |
| Register k=1, ridge (exact selection) | — | 0.394 vs 0.341; diff +0.053 [+0.005, +0.106], significant |
| Register k=1, ridge (per-feature selection) | — | 0.433 vs 0.370, not significant |
| Register k=1, logistic | — | 0.418 vs 0.413, not significant |

- **Ledger:** equal decodability holds for every probe I tried.
- **Register:** the "indistinguishable" result depends on the probe and the selection rule. Sparse comes out significantly ahead under logistic regression (one-frame) and under exact-selected ridge (k=1).

Supa generative exact-state accuracy, test split:

| Item | Claimed | Mine |
|---|---|---|
| Ledger one-frame | 0.07–0.11 | 0.067 (parse rate 0.90) |
| Ledger k=1 | 0.07–0.11 | 0.106 (parse rate 0.97) |
| Prerequisite (P), one-frame and k=1 | unparseable | parse rate 0, exact 0 |
| Register (R), one-frame and k=1 | unparseable | parse rate 0, exact 0 |

My exact scores agree with the recorded `exact` flags on every item (0 disagreements).

Caveat on "unparseable": Supa wraps its answers as `{"before":…,"after":…}`. For P the 12-token generation budget also cuts many answers off mid-number. If I pull out the "after" field anyway:
- P one-frame scores 0.429, but it is pure copying: 0 of 87 changed items are right.
- R one-frame scores 0.067, with 9 of 200 changed items right.

So "Supa does not compute the consequence" stands. But "unparseable" is partly a formatting and truncation artifact.

**C2 — Counterfactual control (ledger accounts rotated A→B→C→D→A)**

- I find 180 discriminating test items, as claimed. True and counterfactual masks agree on 13.5% of items. Copy floors are matched (0.144 true vs 0.154 counterfactual).

| Arm, frame | Probe | Selectivity (true − counterfactual) |
|---|---|---|
| Sparse, one-frame | ridge | −0.022 [−0.056, +0.014]; true 0.878 vs counterfactual 0.900, as claimed |
| Sparse, one-frame | logistic | −0.006 |
| Supa, one-frame | ridge | +0.006 / 0.000 |
| Supa, one-frame | logistic | **FLAG** +0.039 [−0.024, +0.104], against a claimed ≈ −0.03 to 0; not significant |
| Sparse, k=1 | ridge | −0.022 |
| Supa, k=1 | ridge | −0.028 |
| Sparse / Supa, k=1 | logistic | +0.022 / −0.011 |

- The claim is confirmed: selectivity is about 0 for both arms and all probes.
- The control is limited in power. Its intervals only rule out selectivity above roughly 0.05–0.10.
- The rotation simply relabels which account is named, so selectivity of about 0 is what any input-linear probe would give.

**C3 — The delta shifts late-layer states; pooled test one-frame items, n=640**

| Measure | Layer | Claimed | Mine |
|---|---|---|---|
| Layers 0, 4, 8 bit-identical | — | yes | yes; max absolute difference is exactly 0 |
| Relative L2, full | 12 / 16 / 20 / 24 / 28 | 0.16 / 0.40 / 0.65 / 0.75 / 0.57 | 0.164 / 0.393 / 0.641 / 0.730 / 0.564 |
| Relative L2 without top-5 massive dims | 12 | 0.24 | 0.248 |
| | 16 | 0.53 | **FLAG** 0.586 |
| | 20 | 0.81 | 0.826 |
| | 24 | 1.15 | **FLAG** 1.087 |
| Median per-dim \|z\| | 12 | 1.03 | **FLAG** 1.40 |
| | 16 | 1.58 | **FLAG** 2.26 |
| | 20 | 1.61 | **FLAG** 1.92 |
| | 24 | 1.61 | **FLAG** 1.72 |
| | 28 | 1.36 | **FLAG** 1.46 |
| Linear CKA | 12 / 16 / 20 / 24 / 28 | 0.991 / 0.953 / 0.965 / 0.925 / 0.937 | 0.991 / 0.925 / 0.979 / 0.937 / 0.910 |

- **The R5 table can't be reproduced exactly.** No item set I tried gives all its columns at once. Using all test items (k1, k2, k4 and one-frame) comes closest on the median-|z| column (1.01 / 1.56 / 1.54 / 1.53 / 1.32). That suggests that column was not computed on the stated "TEST, one-step frame" set.
- **The CKA ≥ 0.93 figure comes from pooling the three families.** Within a single family, CKA is much lower:
  - Ledger: 0.79 at layer 20, 0.60 at layer 24.
  - Register: 0.87 at layer 20, 0.72 at layer 24.
  - Prerequisite: 0.73 at layer 16, 0.51 at layer 20, 0.64 at layer 24.

  "Geometry is preserved" therefore reflects the between-family structure, not the structure within a family.
- **Cross-arm transfer of ridge probes:**
  - Applying a probe to the other arm with no adjustment scores 0–0.005.
  - After re-standardisation: ledger k=1 gives 0.058 (Sparse→Supa) and 0.274 (Supa→Sparse), matching the claim. Ledger one-frame Supa→Sparse gives **FLAG** 0.034 against a claimed 0.139. That number depends on the ridge strength chosen (mine picked 1, theirs 10).
  - Conclusion: probes do not transfer, confirmed.

**C4 — Multi-step decay (my addition; ridge, Sparse arm)**

| Family | k=1 / k=2 / k=4 | Floors |
|---|---|---|
| Ledger | 0.875 / 0.500 / 0.221 (exact match to the claim) | k=4: copy 0.0, majority 0.091 |
| Register | 0.433 / 0.091 / 0.010 with per-feature selection (exact match); exact selection gives **FLAG** 0.394 at k=1 | — |
| Prerequisite | 0.585–0.612 / 0.438 / 0.299–0.313 | copy floor 0.612 / 0.451 / 0.308 |

- For P, the claimed k=1 value is 0.625. Mine sits at the copy floor, so "P at copy floor" is confirmed.
- I did not recompute the event-boundary tracking or the touched-entity floor (0.433). Ledger at k=4 (0.22) is still above the copy and majority floors.

Everything is in `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/verification/`:
- `indep_recompute.py`, `indep_recompute.log`, `indep_recompute_results.json`
- `gen_lenient.py`, `gen_lenient.log`, `gen_lenient_copy.log`
- `indep_shift_variants.py`, `indep_shift_variants.log`
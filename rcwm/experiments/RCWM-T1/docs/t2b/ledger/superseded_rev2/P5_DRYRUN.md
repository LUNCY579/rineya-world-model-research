# PREREG-5 dry run (synthetic outcomes; every wording branch)

## PASS: (a) all seven factors generalise (T2 s0 far below the bar on every primary cell)
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) count: T2 s0 already meets the bar -> PARTIAL no_headroom
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for an unseen account count (8 accounts)
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - PARTIAL for an unseen account count (8 accounts): accuracy 0.95 (T2 s0 0.90); no shown gain where T2 s0 already meets the bar (H_count T2 s0 0.90). Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) words: trained words learned (well above T2 s0), held-out words near T2 s0 -> TRIGGER WIDENED; combined PARTIAL
- PREREG-5 (a): GENERALISES TO unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts); TRIGGER WIDENED for unseen status words; PARTIAL for the combined unseen surface
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - TRIGGER WIDENED for unseen status words: on trained variants (S_words) the adapter reached 0.95 (each >= 0.80 and at least 0.10 above T2 s0's 0.10), but on H_words it stayed near T2 s0 (0.10 vs T2 s0 0.10; within 0.10 of it in every seed).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - PARTIAL for the combined unseen surface: accuracy 0.08 to 0.95 (T2 s0 0.05 to 0.20); below the bar in HS_words s0, HS_words s1, HS_words s2; not non-inferior to its own canonical accuracy in HS_words (seed-mean lower bound -0.937); no Holm-significant gain over T2 s0 in HS_words s0, HS_words s1, HS_words s2. Per-cell and per-variant table below.
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) words: held-out words near T2 s0 but the trained-word control is not 0.10 above T2 s0 -> PARTIAL
- PREREG-5 (a): GENERALISES TO unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts); TRIGGER WIDENED for none; PARTIAL for unseen status words, the combined unseen surface
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL for unseen status words: accuracy 0.10 (T2 s0 0.10); below the bar in H_words s0, H_words s1, H_words s2; not non-inferior to its own canonical accuracy in H_words (seed-mean lower bound -0.920); no Holm-significant gain over T2 s0 in H_words s0, H_words s1, H_words s2. Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - PARTIAL for the combined unseen surface: accuracy 0.08 to 0.95 (T2 s0 0.05 to 0.20); below the bar in HS_words s0, HS_words s1, HS_words s2; not non-inferior to its own canonical accuracy in HS_words (seed-mean lower bound -0.937); no Holm-significant gain over T2 s0 in HS_words s0, HS_words s1, HS_words s2. Per-cell and per-variant table below.
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) count: far BELOW T2 s0 (not 'near' it) -> PARTIAL below_t2s0, not TRIGGER WIDENED
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for an unseen account count (8 accounts)
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - PARTIAL for an unseen account count (8 accounts): accuracy 0.50 (T2 s0 0.90); below the bar in H_count s0, H_count s1, H_count s2; not non-inferior to its own canonical accuracy in H_count (seed-mean lower bound -0.483); no shown gain where T2 s0 already meets the bar (H_count T2 s0 0.90); more than 0.10 below T2 s0 in H_count s0, H_count s1, H_count s2. Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) magnitude: trained magnitude bands learned, held-out bands near T2 s0 -> TRIGGER WIDENED
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, the combined unseen surface; TRIGGER WIDENED for unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts); PARTIAL for none
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.05 per seed, seed-mean lower bound -0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.05 per seed, seed-mean lower bound 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.05 per seed, seed-mean lower bound -0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.05 per seed, seed-mean lower bound 0.00, each >= -0.10).
  - TRIGGER WIDENED for unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts): on trained variants (S_bal, S_amt) the adapter reached 0.96 (each >= 0.80 and at least 0.10 above T2 s0's 0.15 to 0.51), but on H_bal (add / transfer) and H_amt (add / transfer) it stayed near T2 s0 (0.16 to 0.52 vs T2 s0 0.15 to 0.52; within 0.10 of it in every seed).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.01, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.013): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) order below the bar but gaining -> PARTIAL below_bar
- PREREG-5 (a): GENERALISES TO unseen status words, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for unseen key orders
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - PARTIAL for unseen key orders: accuracy 0.75 (T2 s0 0.25); below the bar in H_order s0, H_order s1, H_order s2; not non-inferior to its own canonical accuracy in H_order (seed-mean lower bound -0.223). Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) names at the bar but far below own canonical -> PARTIAL not_noninferior
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for unseen account names
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.06, each >= -0.10).
  - PARTIAL for unseen account names: accuracy 0.82 (T2 s0 0.35); not non-inferior to its own canonical accuracy in H_names (seed-mean lower bound -0.210). Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to -0.01 per seed, seed-mean lower bound -0.06 to -0.05, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.06, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.060, s1 +0.060, s2 +0.060; seed mean +0.060 (one-sided 95% lower bound +0.033): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) paraphrase: bar met, T2 s0 just below the bar, gain not significant -> PARTIAL no_gain
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for an unseen rule paraphrase
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - PARTIAL for an unseen rule paraphrase: accuracy 0.86 (T2 s0 0.79); not non-inferior to its own canonical accuracy in H_para (seed-mean lower bound -0.107); no Holm-significant gain over T2 s0 in H_para s0, H_para s1, H_para s2. Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) magnitude: one seed fails -> PARTIAL seed_disagreement + below_bar
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts)
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - PARTIAL for unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts): accuracy 0.60 to 0.97 (T2 s0 0.15 to 0.52); below the bar in H_bal s2; not non-inferior to its own canonical accuracy in H_bal (seed-mean lower bound -0.134); no Holm-significant gain over T2 s0 in H_bal s2; the seeds disagree. Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) in-distribution cost qualifies every headline
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none (every part (a) result at an in-distribution cost relative to T2 s0)
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.05 per seed, seed-mean lower bound 0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.09 to 0.13 per seed, seed-mean lower bound 0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.09 to 0.13 per seed, seed-mean lower bound 0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS) (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.09 to 0.13 per seed, seed-mean lower bound 0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.09 to 0.13 per seed, seed-mean lower bound 0.06, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS) (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.19 to 0.28 per seed, seed-mean lower bound 0.18 to 0.19, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE (at an in-distribution cost relative to T2 s0): in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.05 to 0.25 per seed, seed-mean lower bound 0.01 to 0.20, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 -0.130, s1 -0.130, s2 -0.130; seed mean -0.130 (one-sided 95% lower bound -0.170): not shown non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS PARTIAL for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) order near T2 s0 but the trained-surface control also low -> PARTIAL, not TRIGGER WIDENED
- PREREG-5 (a): GENERALISES TO unseen status words, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for unseen key orders
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - PARTIAL for unseen key orders: accuracy 0.30 (T2 s0 0.25); below the bar in H_order s0, H_order s1, H_order s2; not non-inferior to its own canonical accuracy in H_order (seed-mean lower bound -0.673); no Holm-significant gain over T2 s0 in H_order s0, H_order s1, H_order s2. Per-cell and per-variant table below.
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) report-only: word-cluster intervals for H_words and HS_words on every side
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) secondary: T2b-ledger RULE vs T2 s0 TRIGGER
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) secondary: T2b-ledger TRIGGER on PREREG-3's strings
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS TRIGGER for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.05 to 0.90 across seeds (the seven without a form of 'void': 0.05); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 0 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) secondary: PARTIAL
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS PARTIAL for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.50 to 0.95 across seeds (the seven without a form of 'void': 0.50 to 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) GENERALISES
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger PLANS ON THE UNSEEN SURFACE: 120/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.639, discordant 115/0, exact McNemar p <0.0001; status PASS); 0.750 of ORACLE's success (>= 0.70) and within the margin of its own canonical success (125/180; difference -0.028, lower bound -0.050).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) GENERALISES + near-oracle qualifier
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger PLANS ON THE UNSEEN SURFACE: 150/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.806, discordant 145/0, exact McNemar p <0.0001; status PASS); 0.938 of ORACLE's success (>= 0.70) and within the margin of its own canonical success (152/180; difference -0.011, lower bound -0.028). Report-only: near-oracle on the held-out surface (ratio 0.938, 95% interval 0.898-0.974).
  - Report-only: on the canonical surface T2b-ledger 152/180 vs T2 s0 127/180 (difference +0.139, 95% paired bootstrap CI [+0.089, +0.194], exact McNemar p <0.0001); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) TRIGGER WIDENED
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): TRIGGER WIDENED
- PREREG-5 (c): retention guard GO (5/5 binding)
  - TRIGGER WIDENED (planning): T2b-ledger planned on the canonical surface (125/180 vs T2 s0 127/180) but stayed near T2 s0 on the held-out surface: 10/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.028, discordant 5/0, exact McNemar p 0.0625; status NS).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) PARTIAL short
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 35/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.167, discordant 30/0, exact McNemar p <0.0001; status SHORT); the gain is significant but below +0.20.
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) PARTIAL below_bar
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 100/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.528, discordant 95/0, exact McNemar p <0.0001; status PASS); its held-out success is below 0.70 of ORACLE's (0.625); its held-out success is not non-inferior to its own canonical success (lower bound -0.183 < -0.10).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) PARTIAL not_noninferior
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 115/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.611, discordant 110/0, exact McNemar p <0.0001; status PASS); its held-out success is not non-inferior to its own canonical success (lower bound -0.306 < -0.10).
  - Report-only: on the canonical surface T2b-ledger 160/180 vs T2 s0 127/180 (difference +0.183, 95% paired bootstrap CI [+0.128, +0.239], exact McNemar p <0.0001); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) PARTIAL canonical cost + near T2 s0 on held-out
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 8/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.017, discordant 3/0, exact McNemar p 0.2500; status NS); it plans worse than T2 s0 on the canonical surface (90/180 vs 127/180).
  - Report-only: on the canonical surface T2b-ledger 90/180 vs T2 s0 127/180 (difference -0.206, 95% paired bootstrap CI [-0.267, -0.150], exact McNemar p <0.0001); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) PARTIAL reversed
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 0/180 vs T2 s0's 40/180 on PREREG-LB's held-out combined rendering (delta -0.222, discordant 0/40, exact McNemar p <0.0001; status REVERSED); T2 s0 planned significantly better on the held-out surface.
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) PARTIAL reversed with canonical cost
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 0/180 vs T2 s0's 40/180 on PREREG-LB's held-out combined rendering (delta -0.222, discordant 0/40, exact McNemar p <0.0001; status REVERSED); T2 s0 planned significantly better on the held-out surface; it plans worse than T2 s0 on the canonical surface (60/180 vs 127/180).
  - Report-only: on the canonical surface T2b-ledger 60/180 vs T2 s0 127/180 (difference -0.372, 95% paired bootstrap CI [-0.444, -0.300], exact McNemar p <0.0001); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (b) NS but not near T2 s0 -> PARTIAL ns_with_gain
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PARTIAL
- PREREG-5 (c): retention guard GO (5/5 binding)
  - PARTIAL (planning): 100/180 vs T2 s0's 80/180 on PREREG-LB's held-out combined rendering (delta +0.111, discordant 60/40, exact McNemar p 0.0569; status NS); the held-out difference is not significant but not near T2 s0 either.
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (c) NO-GO
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- PREREG-5 (c): retention guard NO-GO
  - T2b-ledger PLANS ON THE UNSEEN SURFACE: 120/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.639, discordant 115/0, exact McNemar p <0.0001; status PASS); 0.750 of ORACLE's success (>= 0.70) and within the margin of its own canonical success (125/180; difference -0.028, lower bound -0.050).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): NO-GO (LOSS DETECTED): T2b-ledger s0 loses retention against Titan-Sparse on the 5c guard; it is not a packaging candidate without retention repair.

## PASS: (c) not evaluable
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- PREREG-5 (c): retention guard not evaluable
  - T2b-ledger PLANS ON THE UNSEEN SURFACE: 120/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.639, discordant 115/0, exact McNemar p <0.0001; status PASS); 0.750 of ORACLE's success (>= 0.70) and within the margin of its own canonical success (125/180; difference -0.028, lower bound -0.050).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c): not evaluable (no registered retention result); every headline is read without it.

## PASS: (c) GO from a ret_analysis result
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): not evaluable (this part's stages are not recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - Retention guard (5c, draw confirm_x): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: (a) and (b) not evaluable after an early stop
- PREREG-5 (a): not evaluable (the run ended by a deterministic stop before this part's stages were recorded).
- PREREG-5 (b): not evaluable (the run ended by a deterministic stop before this part's stages were recorded).
- PREREG-5 (c): retention guard not evaluable
  - Retention guard (5c): not evaluable (the retention run was not completed and bound); every headline is read without it.

## PASS: (b) not evaluable, (a) rendered (a stop after M1)
- PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders, unseen account names, an unseen account count (8 accounts), an unseen rule paraphrase, unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts), the combined unseen surface; TRIGGER WIDENED for none; PARTIAL for none
- PREREG-5 (b): not evaluable (the run ended by a deterministic stop before this part's stages were recorded).
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger GENERALISES TO UNSEEN STATUS WORDS: in all three seeds, accuracy on H_words 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.10) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 per seed, seed-mean lower bound -0.07, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN KEY ORDERS: in all three seeds, accuracy on H_order 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.25) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN ACCOUNT NAMES: in all three seeds, accuracy on H_names 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN ACCOUNT COUNT (8 ACCOUNTS): in all three seeds, accuracy on H_count 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.47) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO AN UNSEEN RULE PARAPHRASE: in all three seeds, accuracy on H_para 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.35) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.01 to 0.03 per seed, seed-mean lower bound -0.01, each >= -0.10).
  - T2b-ledger GENERALISES TO UNSEEN MAGNITUDES (POSTED ADD / TRANSFER EVENTS: 5-DIGIT BALANCES, 4-DIGIT AMOUNTS): in all three seeds, accuracy on H_bal (add / transfer) and H_amt (add / transfer) 0.96 to 0.97 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.15 to 0.52) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference 0.03 to 0.06 per seed, seed-mean lower bound -0.01 to 0.00, each >= -0.10).
  - T2b-ledger GENERALISES TO THE COMBINED UNSEEN SURFACE: in all three seeds, accuracy on HS_words and HS_posted 0.95 (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 (0.05 to 0.20) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference -0.04 to 0.05 per seed, seed-mean lower bound -0.06 to 0.02, each >= -0.10).
  - In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: s0 +0.015, s1 +0.015, s2 +0.015; seed mean +0.015 (one-sided 95% lower bound -0.015): non-inferior within 0.05.
  - Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS RULE for T2b-ledger (T2 s0 in this run: TRIGGER). Accuracy on the eight PREREG-3 strata 0.95 across seeds (the seven without a form of 'void': 0.95); T2 s0 0.05 to 0.95; Holm-significant gains over T2 s0 in 21 of 24 stratum x seed tests.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.

## PASS: rules and code unchanged since the dev gate
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger PLANS ON THE UNSEEN SURFACE: 120/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.639, discordant 115/0, exact McNemar p <0.0001; status PASS); 0.750 of ORACLE's success (>= 0.70) and within the margin of its own canonical success (125/180; difference -0.028, lower bound -0.050).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.
  - The decision rules of parts (a)-(c), the wording templates, the limits and the decision code are byte-identical to those recorded when the dev gate rule was frozen (2026-10-11 09:00:00 -0500), before any T2b-ledger adapter existed.

## PASS: rules or code changed after the dev gate
- PREREG-5 (a): not evaluable (this part's stages are not recorded).
- PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- PREREG-5 (c): retention guard GO (5/5 binding)
  - T2b-ledger PLANS ON THE UNSEEN SURFACE: 120/180 vs T2 s0's 5/180 on PREREG-LB's held-out combined rendering (delta +0.639, discordant 115/0, exact McNemar p <0.0001; status PASS); 0.750 of ORACLE's success (>= 0.70) and within the margin of its own canonical success (125/180; difference -0.028, lower bound -0.050).
  - Report-only: on the canonical surface T2b-ledger 125/180 vs T2 s0 127/180 (difference -0.011, 95% paired bootstrap CI [-0.028, +0.000], exact McNemar p 0.5000); ORACLE 160/180; IGNORE_STATUS 0/180.
  - Retention guard (5c, draw confirm_test): GO (5/5 binding): T2b-ledger s0 retains against Titan-Sparse.
  - The decision rules, wording templates, limits or decision code differ from those recorded when the dev gate rule was frozen (2026-10-11 09:00:00 -0500); the changes were made after the dev screen's outputs existed and are listed, with their reasons, in PREREG_5.md Appendix A.

## PASS: every wording template is realised by at least one case

## PASS: statistics units (McNemar, Holm, bootstrap, Wilson) vs reference implementations


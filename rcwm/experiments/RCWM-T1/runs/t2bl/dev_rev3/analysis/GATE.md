# T2b-ledger dev gate: DEV GATE: PROCEED

In all three T2b-ledger seeds the adapter left held-out non-posted statuses unchanged on 0.91-0.99 of rows on the combined held-out surface (rule paraphrase 4, held-out account names, held-out key orders) and on 0.96-1.00 on the canonical surface (each >= 0.80, every word kind >= 0.60; T2 s0 0.21 and 0.04, OFF 0.25), within 0.10 of its own canonical no-op accuracy; it applied the posted twins on the held-out surface correctly on 0.92-0.94 (each >= 0.80; copied 0.00, each <= 0.10), and its in-distribution accuracy was within the margin of T2 s0's (ON - T2 s0: s0 0.06, s1 0.06, s2 0.05; each >= -0.05).

The sealed confirmation PREREG-5 may be frozen (operator step). This is a dev screen, not a claim.

## Pre-stated rule (verbatim)

> T2b-ledger dev gate, revision 2 (pre-stated on 2026-10-10 before any T2b-ledger adapter or output exists). Inputs: the dev bank data/t2bl_dev.jsonl (seed 20261162; 100 bases; 19 strata; the DEV half of every split held-out partition: held-out words, names, key orders and paraphrase 4) generated in one run of scripts/t2bl_run.py (tag dev) with sides OFF (Titan-Sparse, adapter disabled), T2s0 (the PREREG-2 T2 seed-0 adapter) and ON (each T2b-ledger seed s = 0, 1, 2, unmerged), scored by p3_common.score_onestep (exact balances, any key order). For each seed s: G1 (held-out words on the combined held-out surface): the ON accuracy on HS_words (100 rows: a held-out non-posted status word in the combined held-out surface: rule paraphrase 4, held-out account names, every object in a held-out key order; gold = balances unchanged) is >= 0.80; G1b (held-out words on the canonical surface): the ON accuracy on H_words (100 rows) is >= 0.80; G1c (every word kind): on HS_words and on H_words, the ON accuracy of each of the five word kinds (not-applied, neutral, nonce, void variants, posted look-alikes; 20 rows each) is >= 0.60; G2 (in-distribution non-inferiority to T2 s0): the ON accuracy on ID (ID_posted and ID_void, 200 canonical-surface in-range rows) minus T2s0's accuracy on the same rows is >= -0.05 (point estimate; the paired base-bootstrap 95% lower bound is reported, not gated); G3 (the posted twins on the held-out surface): on HS_posted (the posted twins of HS_words' bases in the same surface) the ON accuracy is >= 0.80 and the share of ON answers equal to the input balances is <= 0.10; G4 (alignment with the seed's own canonical accuracy, as in PREREG-5's non-inferiority): the ON accuracies on HS_words and on H_words are each at least the seed's ON accuracy on ID_void minus 0.10. PROCEED to the sealed confirmation (PREREG-5) iff G1, G1b, G1c, G2, G3 and G4 hold in all three seeds, the run is complete (DONE.json, every stage recorded and unchanged) and the A0 anchor passed in every turn. Otherwise STOP, worded as: TRIGGER WIDENED iff in all three seeds the ON accuracy on S_words (training non-posted words, canonical surface) is >= 0.80, the ON accuracy on H_words (held-out words, canonical surface) is <= 0.30, and more than half of the H_words errors apply the event as if posted; PARTIAL otherwise, with the per-stratum, per-variant, per-seed table and the named patterns (heldout_below_bar, word_kind_below_floor, in_distribution_cost, heldout_posted_twins, not_aligned_with_own_canonical, trained_words_below_bar, seed_disagreement). NO VERDICT (instrument) iff the run is incomplete or stopped (STOPPED.json) or an anchor failed; the gate analysis then writes a NO VERDICT record. Revision path after a STOP (pre-stated): no confirmation is drafted from this kit; a revised kit may change only the training data (composition, vocabulary sizes, magnitude bands, paraphrases), must draw a fresh dev bank under a new seed (this dev bank and every adapter trained on the old data are then recorded as spent), and keeps these thresholds. T2b-ledger seed 0 is the registered artifact for PREREG-5's planning and retention parts; seeds 1 and 2 replicate; no seed is ever chosen by its outcome.

## Inputs of this kit revision

Kit revision 3 (2026-10-12; DECISIONS R29; the rule's pre-stated revision path). The rule text and every threshold are byte-identical to the revision-2 rule frozen on 2026-10-10 (D-48; docs/t2b/ledger/superseded_rev2/GATE_RULE.json). Only the training data changed (data/t2bl_train.jsonl, seed 20261165: both top-level key orders trained; half of the diversified slots on the stacked training surface; posted and non-posted rows exactly equal on the stacked-surface and on the reversed-top rows), and the dev bank was drawn FRESH under a new seed by the same builder and strata (data/t2bl_dev.jsonl, seed 20261166; the DEV half of every held-out partition, unchanged). It replaces the bank of seed 20261162 that the rule text names, which is spent together with every adapter trained on the revision-2 data (runs/t2bl/dev; seeds 20261161 and 20261162). The dev run is tag dev in runs/t2bl/dev_rev3. Since kit revision 3 the top-level {before, event} object is trained in both orders, so 'every object in a held-out key order' in the HS_words definition holds for the event and before objects, and the reversed top order of HS_words, HS_posted and H_order is a trained level there: a deliberate consequence of the revision path, not a new held-out factor. The rule text's 'pre-stated on 2026-10-10 before any T2b-ledger adapter or output exists' is likewise revision 2's: revision 2's adapters and outputs exist (spent), and for this kit revision the unchanged rule is frozen before any adapter trained on its data and before any output of its dev screen exists.

## Gate per seed

| seed | G1 HS_words | G1b H_words | G1c kinds | G2 ID diff vs T2 s0 (boot LB) | G3 HS_posted (copied) | G4 vs ID_void | S_words |
|---|---:|---:|---|---:|---:|---:|---:|
| s0 | 0.910 (pass) | 0.960 (pass) | min 0.70 (pass) | +0.060 (+0.035) (pass) | 0.920 (0.000) (pass) | 1.000 (pass) | 1.000 |
| s1 | 0.990 (pass) | 1.000 (pass) | min 0.95 (pass) | +0.065 (+0.035) (pass) | 0.920 (0.000) (pass) | 1.000 (pass) | 1.000 |
| s2 | 0.980 (pass) | 0.980 (pass) | min 0.90 (pass) | +0.055 (+0.030) (pass) | 0.940 (0.000) (pass) | 1.000 (pass) | 1.000 |

## Accuracy per stratum and side (descriptive)

| stratum | off | t2s0 | s0 | s1 | s2 |
|---|---:|---:|---:|---:|---:|
| ID_posted | 0.44 | 0.83 | 0.95 | 0.96 | 0.94 |
| ID_void | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 |
| S_words | 0.04 | 0.15 | 1.00 | 1.00 | 1.00 |
| S_surface_posted | 0.18 | 0.32 | 0.72 | 0.75 | 0.71 |
| S_surface_words | 0.17 | 0.17 | 1.00 | 0.98 | 0.99 |
| S_bal | 0.40 | 0.64 | 0.90 | 0.93 | 0.78 |
| S_amt | 0.04 | 0.13 | 0.61 | 0.70 | 0.69 |
| H_words | 0.03 | 0.04 | 0.96 | 1.00 | 0.98 |
| H_order | 0.20 | 0.25 | 0.95 | 0.97 | 0.97 |
| H_names | 0.12 | 0.42 | 0.90 | 0.93 | 0.87 |
| H_count | 0.22 | 0.46 | 0.95 | 0.99 | 0.98 |
| H_para | 0.20 | 0.49 | 0.96 | 0.98 | 0.97 |
| H_bal | 0.35 | 0.44 | 0.76 | 0.73 | 0.68 |
| H_amt | 0.00 | 0.00 | 0.37 | 0.28 | 0.19 |
| HS_words | 0.25 | 0.21 | 0.91 | 0.99 | 0.98 |
| HS_posted | 0.13 | 0.13 | 0.92 | 0.92 | 0.94 |
| LB_void | 0.27 | 0.21 | 1.00 | 1.00 | 0.95 |
| LB_posted | 0.17 | 0.14 | 0.92 | 0.76 | 0.55 |
| P3_words | 0.03 | 0.21 | 0.97 | 1.00 | 0.98 |

## Variants (descriptive)

- **S_words** by word_kind: neutral: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.00, t2s0 0.05, s0 1.00, s1 1.00, s2 1.00; posted_like: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.05, t2s0 0.70, s0 1.00, s1 1.00, s2 1.00
- **H_words** by word_kind: neutral: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.00, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; posted_like: off 0.05, t2s0 0.00, s0 0.80, s1 1.00, s2 0.90; void_variant: off 0.00, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00
- **HS_words** by word_kind: neutral: off 0.25, t2s0 0.15, s0 0.70, s1 1.00, s2 1.00; nonce: off 0.20, t2s0 0.15, s0 0.95, s1 0.95, s2 1.00; not_applied: off 0.40, t2s0 0.50, s0 1.00, s1 1.00, s2 0.95; posted_like: off 0.15, t2s0 0.00, s0 0.90, s1 1.00, s2 0.95; void_variant: off 0.25, t2s0 0.25, s0 1.00, s1 1.00, s2 1.00
- **S_surface_words** by word_kind: neutral: off 0.10, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.15, t2s0 0.00, s0 1.00, s1 0.95, s2 1.00; not_applied: off 0.25, t2s0 0.05, s0 1.00, s1 1.00, s2 1.00; posted_like: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.30, t2s0 0.70, s0 1.00, s1 0.95, s2 0.95
- **H_names** by names_kind: heldout_letters: off 0.04, t2s0 0.44, s0 0.92, s1 0.92, s2 0.86; heldout_words: off 0.20, t2s0 0.40, s0 0.88, s1 0.94, s2 0.88
- **HS_posted** by names_kind: heldout_letters: off 0.12, t2s0 0.08, s0 0.94, s1 0.90, s2 0.90; heldout_words: off 0.14, t2s0 0.18, s0 0.90, s1 0.94, s2 0.98
- **H_order** by event_order: 0: off 0.21, t2s0 0.32, s0 0.91, s1 0.94, s2 0.94; 1: off 0.18, t2s0 0.21, s0 1.00, s1 1.00, s2 1.00; 2: off 0.21, t2s0 0.21, s0 0.94, s1 0.97, s2 0.97
- **H_count** by status_kind: neutral: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; posted: off 0.42, t2s0 0.78, s0 0.90, s1 0.98, s2 0.96; posted_like: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.00, t2s0 0.70, s0 1.00, s1 1.00, s2 1.00
- **H_para** by status_kind: neutral: off 0.10, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; posted: off 0.36, t2s0 0.82, s0 0.92, s1 0.96, s2 0.94; posted_like: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.00, t2s0 0.70, s0 1.00, s1 1.00, s2 1.00
- **P3_words** by word: VOID: off 0.00, t2s0 1.00, s0 1.00, s1 1.00, s2 1.00; Void: off 0.00, t2s0 1.00, s0 1.00, s1 1.00, s2 1.00; active: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; approved: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; cancelled: off 0.14, t2s0 0.29, s0 1.00, s1 1.00, s2 1.00; dax: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; draft: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; held: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not posted: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_posted: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; ok: off 0.00, t2s0 0.00, s0 0.67, s1 1.00, s2 1.00; pending: off 0.14, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; reversed: off 0.00, t2s0 0.14, s0 0.86, s1 1.00, s2 0.71; unposted: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; voided: off 0.17, t2s0 1.00, s0 1.00, s1 1.00, s2 1.00; wug: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00
- **S_surface_posted** by names_kind: train_letters: off 0.20, t2s0 0.32, s0 0.74, s1 0.76, s2 0.72; train_words: off 0.16, t2s0 0.32, s0 0.70, s1 0.74, s2 0.70

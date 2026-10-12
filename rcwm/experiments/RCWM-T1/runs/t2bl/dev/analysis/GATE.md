# T2b-ledger dev gate: DEV GATE: STOP (PARTIAL)

Neither PROCEED nor TRIGGER WIDENED: the posted twins on the held-out surface below 0.80 or copied above 0.10 (accuracy s0 0.54, s1 0.40, s2 0.42; copied s0 0.16, s1 0.31, s2 0.35). The per-stratum, per-variant table follows.

No confirmation is drafted from this kit (revision path in the rule).

## Pre-stated rule (verbatim)

> T2b-ledger dev gate, revision 2 (pre-stated on 2026-10-10 before any T2b-ledger adapter or output exists). Inputs: the dev bank data/t2bl_dev.jsonl (seed 20261162; 100 bases; 19 strata; the DEV half of every split held-out partition: held-out words, names, key orders and paraphrase 4) generated in one run of scripts/t2bl_run.py (tag dev) with sides OFF (Titan-Sparse, adapter disabled), T2s0 (the PREREG-2 T2 seed-0 adapter) and ON (each T2b-ledger seed s = 0, 1, 2, unmerged), scored by p3_common.score_onestep (exact balances, any key order). For each seed s: G1 (held-out words on the combined held-out surface): the ON accuracy on HS_words (100 rows: a held-out non-posted status word in the combined held-out surface: rule paraphrase 4, held-out account names, every object in a held-out key order; gold = balances unchanged) is >= 0.80; G1b (held-out words on the canonical surface): the ON accuracy on H_words (100 rows) is >= 0.80; G1c (every word kind): on HS_words and on H_words, the ON accuracy of each of the five word kinds (not-applied, neutral, nonce, void variants, posted look-alikes; 20 rows each) is >= 0.60; G2 (in-distribution non-inferiority to T2 s0): the ON accuracy on ID (ID_posted and ID_void, 200 canonical-surface in-range rows) minus T2s0's accuracy on the same rows is >= -0.05 (point estimate; the paired base-bootstrap 95% lower bound is reported, not gated); G3 (the posted twins on the held-out surface): on HS_posted (the posted twins of HS_words' bases in the same surface) the ON accuracy is >= 0.80 and the share of ON answers equal to the input balances is <= 0.10; G4 (alignment with the seed's own canonical accuracy, as in PREREG-5's non-inferiority): the ON accuracies on HS_words and on H_words are each at least the seed's ON accuracy on ID_void minus 0.10. PROCEED to the sealed confirmation (PREREG-5) iff G1, G1b, G1c, G2, G3 and G4 hold in all three seeds, the run is complete (DONE.json, every stage recorded and unchanged) and the A0 anchor passed in every turn. Otherwise STOP, worded as: TRIGGER WIDENED iff in all three seeds the ON accuracy on S_words (training non-posted words, canonical surface) is >= 0.80, the ON accuracy on H_words (held-out words, canonical surface) is <= 0.30, and more than half of the H_words errors apply the event as if posted; PARTIAL otherwise, with the per-stratum, per-variant, per-seed table and the named patterns (heldout_below_bar, word_kind_below_floor, in_distribution_cost, heldout_posted_twins, not_aligned_with_own_canonical, trained_words_below_bar, seed_disagreement). NO VERDICT (instrument) iff the run is incomplete or stopped (STOPPED.json) or an anchor failed; the gate analysis then writes a NO VERDICT record. Revision path after a STOP (pre-stated): no confirmation is drafted from this kit; a revised kit may change only the training data (composition, vocabulary sizes, magnitude bands, paraphrases), must draw a fresh dev bank under a new seed (this dev bank and every adapter trained on the old data are then recorded as spent), and keeps these thresholds. T2b-ledger seed 0 is the registered artifact for PREREG-5's planning and retention parts; seeds 1 and 2 replicate; no seed is ever chosen by its outcome.

## Gate per seed

| seed | G1 HS_words | G1b H_words | G1c kinds | G2 ID diff vs T2 s0 (boot LB) | G3 HS_posted (copied) | G4 vs ID_void | S_words |
|---|---:|---:|---|---:|---:|---:|---:|
| s0 | 0.980 (pass) | 0.990 (pass) | min 0.90 (pass) | +0.020 (-0.010) (pass) | 0.540 (0.160) (FAIL) | 1.000 (pass) | 1.000 |
| s1 | 0.990 (pass) | 1.000 (pass) | min 0.95 (pass) | +0.040 (+0.020) (pass) | 0.400 (0.310) (FAIL) | 1.000 (pass) | 1.000 |
| s2 | 0.980 (pass) | 1.000 (pass) | min 0.95 (pass) | +0.040 (+0.015) (pass) | 0.420 (0.350) (FAIL) | 1.000 (pass) | 1.000 |

## Accuracy per stratum and side (descriptive)

| stratum | off | t2s0 | s0 | s1 | s2 |
|---|---:|---:|---:|---:|---:|
| ID_posted | 0.46 | 0.92 | 0.95 | 0.99 | 0.99 |
| ID_void | 0.04 | 0.99 | 1.00 | 1.00 | 1.00 |
| S_words | 0.03 | 0.14 | 1.00 | 1.00 | 1.00 |
| S_surface_posted | 0.22 | 0.42 | 0.85 | 0.84 | 0.86 |
| S_surface_words | 0.14 | 0.17 | 1.00 | 0.99 | 1.00 |
| S_bal | 0.42 | 0.64 | 0.91 | 0.85 | 0.94 |
| S_amt | 0.06 | 0.15 | 0.72 | 0.72 | 0.72 |
| H_words | 0.03 | 0.03 | 0.99 | 1.00 | 1.00 |
| H_order | 0.18 | 0.30 | 0.87 | 0.77 | 0.82 |
| H_names | 0.13 | 0.44 | 0.97 | 0.95 | 0.96 |
| H_count | 0.21 | 0.52 | 0.93 | 0.99 | 0.98 |
| H_para | 0.20 | 0.54 | 0.98 | 0.99 | 0.99 |
| H_bal | 0.25 | 0.51 | 0.75 | 0.69 | 0.74 |
| H_amt | 0.00 | 0.00 | 0.22 | 0.06 | 0.49 |
| HS_words | 0.30 | 0.18 | 0.98 | 0.99 | 0.98 |
| HS_posted | 0.10 | 0.23 | 0.54 | 0.40 | 0.42 |
| LB_void | 0.25 | 0.17 | 1.00 | 1.00 | 0.98 |
| LB_posted | 0.23 | 0.22 | 0.48 | 0.40 | 0.43 |
| P3_words | 0.03 | 0.18 | 0.90 | 0.94 | 0.99 |

## Variants (descriptive)

- **S_words** by word_kind: neutral: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; posted_like: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.00, t2s0 0.70, s0 1.00, s1 1.00, s2 1.00
- **H_words** by word_kind: neutral: off 0.00, t2s0 0.00, s0 0.95, s1 1.00, s2 1.00; nonce: off 0.05, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.00, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; posted_like: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.00, t2s0 0.05, s0 1.00, s1 1.00, s2 1.00
- **HS_words** by word_kind: neutral: off 0.25, t2s0 0.05, s0 0.90, s1 0.95, s2 0.95; nonce: off 0.20, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.35, t2s0 0.35, s0 1.00, s1 1.00, s2 0.95; posted_like: off 0.45, t2s0 0.20, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.25, t2s0 0.20, s0 1.00, s1 1.00, s2 1.00
- **S_surface_words** by word_kind: neutral: off 0.15, t2s0 0.05, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.15, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; posted_like: off 0.30, t2s0 0.10, s0 1.00, s1 0.95, s2 1.00; void_variant: off 0.00, t2s0 0.60, s0 1.00, s1 1.00, s2 1.00
- **H_names** by names_kind: heldout_letters: off 0.14, t2s0 0.50, s0 0.98, s1 0.98, s2 0.96; heldout_words: off 0.12, t2s0 0.38, s0 0.96, s1 0.92, s2 0.96
- **HS_posted** by names_kind: heldout_letters: off 0.04, t2s0 0.16, s0 0.58, s1 0.46, s2 0.44; heldout_words: off 0.16, t2s0 0.30, s0 0.50, s1 0.34, s2 0.40
- **H_order** by event_order: 0: off 0.24, t2s0 0.29, s0 0.88, s1 0.65, s2 0.74; 1: off 0.12, t2s0 0.36, s0 0.85, s1 0.82, s2 0.88; 2: off 0.18, t2s0 0.24, s0 0.88, s1 0.85, s2 0.85
- **H_count** by status_kind: neutral: off 0.10, t2s0 0.00, s0 0.80, s1 1.00, s2 1.00; nonce: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; posted: off 0.34, t2s0 0.88, s0 0.92, s1 0.98, s2 0.96; posted_like: off 0.10, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.10, t2s0 0.70, s0 0.90, s1 1.00, s2 1.00
- **H_para** by status_kind: neutral: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; nonce: off 0.10, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_applied: off 0.10, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; posted: off 0.30, t2s0 0.90, s0 0.96, s1 0.98, s2 0.98; posted_like: off 0.20, t2s0 0.10, s0 1.00, s1 1.00, s2 1.00; void_variant: off 0.10, t2s0 0.70, s0 1.00, s1 1.00, s2 1.00
- **P3_words** by word: VOID: off 0.17, t2s0 1.00, s0 1.00, s1 1.00, s2 1.00; Void: off 0.00, t2s0 1.00, s0 1.00, s1 1.00, s2 1.00; active: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; approved: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; cancelled: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; dax: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; draft: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; held: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not posted: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; not_posted: off 0.17, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; ok: off 0.00, t2s0 0.00, s0 0.00, s1 0.17, s2 1.00; pending: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; reversed: off 0.14, t2s0 0.00, s0 0.43, s1 0.86, s2 0.86; unposted: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00; voided: off 0.00, t2s0 1.00, s0 1.00, s1 1.00, s2 1.00; wug: off 0.00, t2s0 0.00, s0 1.00, s1 1.00, s2 1.00
- **S_surface_posted** by names_kind: train_letters: off 0.18, t2s0 0.46, s0 0.84, s1 0.82, s2 0.86; train_words: off 0.26, t2s0 0.38, s0 0.86, s1 0.86, s2 0.86

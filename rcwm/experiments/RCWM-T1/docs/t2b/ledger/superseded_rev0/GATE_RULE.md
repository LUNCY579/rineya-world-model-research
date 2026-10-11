# T2b-ledger dev gate rule (revision 1; pre-stated before any T2b-ledger output)

Status: PRE-STATED, NOT FROZEN. Freezing writes `GATE_RULE.json` (sha256 of every frozen input) via `python -B scripts/t2bl_gate.py --freeze-rule --approval "..."`, an operator step logged in DECISIONS.md; the runner refuses to start without it.

## Rule (verbatim; `t2bl_gate.RULE_TEXT`)

> T2b-ledger dev gate, revision 1 (pre-stated on 2026-10-10 before any T2b-ledger adapter or output exists). Inputs: the dev bank data/t2bl_dev.jsonl (seed 20261162; 100 bases; 17 strata) generated in one run of scripts/t2bl_run.py (tag dev) with sides OFF (Titan-Sparse, adapter disabled), T2s0 (the PREREG-2 T2 seed-0 adapter) and ON (each T2b-ledger seed s = 0, 1, 2, unmerged), scored by p3_common.score_onestep (exact balances, any key order). For each seed s: G1 (held-out words on the combined held-out surface): the ON accuracy on HS_words (100 rows: a held-out non-posted status word in the combined held-out surface: rule paraphrase 4, held-out account names, every object in a held-out key order; gold = balances unchanged) is >= 0.80; G2 (in-distribution non-inferiority to T2 s0): the ON accuracy on ID (ID_posted and ID_void, 200 canonical-surface in-range rows) minus T2s0's accuracy on the same rows is >= -0.05 (point estimate; the paired base-bootstrap 95% lower bound is reported, not gated); G3 (copy guard on the held-out surface): on HS_posted (the posted twins of HS_words' bases in the same surface) the share of ON answers equal to the input balances is <= 0.10. PROCEED to the sealed confirmation (PREREG-5) iff G1, G2 and G3 hold in all three seeds, the run is complete (DONE.json, every stage recorded and unchanged) and the A0 anchor passed in every turn. Otherwise STOP, worded as: TRIGGER WIDENED iff in all three seeds the ON accuracy on S_words (training non-posted words, canonical surface) is >= 0.80 and on HS_words is <= 0.30; PARTIAL otherwise, with the per-stratum, per-variant, per-seed table and the named patterns (heldout_below_bar, in_distribution_cost, copy_on_heldout_surface, trained_words_below_bar, seed_disagreement). NO VERDICT (instrument) iff the run is incomplete or an anchor failed. Revision path after a STOP (pre-stated): no confirmation is drafted from this kit; a revised kit may change only the training data (composition, vocabulary sizes, magnitude bands, paraphrases), must draw a fresh dev bank under a new seed (this dev bank and every adapter trained on the old data are then recorded as spent), and keeps these thresholds. T2b-ledger seed 0 is the registered artifact for PREREG-5's planning and retention parts; seeds 1 and 2 replicate; no seed is ever chosen by its outcome.

## Thresholds

| clause | quantity | threshold |
|---|---|---|
| G1 | ON accuracy, HS_words (held-out words, combined held-out surface) | >= 0.80 |
| G2 | ON - T2 s0 accuracy on ID (ID_posted + ID_void), point estimate | >= -0.05 |
| G3 | ON copied rate on HS_posted | <= 0.10 |
| TRIGGER WIDENED | S_words >= 0.80 and HS_words <= 0.30, every seed | |

## Why these choices

- **G1 at 0.80** is the programme's bar (PREREG-3). HS_words is the hardest dev stratum: every surface factor is held out at once, and the status word is held out too. T2 s0 is expected near 0 there (PREREG-3: unseen statuses 0.01-0.18 on the canonical surface; PREREG-LB: 5/180 plans on a held-out surface).
- **G2 at -0.05** is the task's margin against T2 s0 on T2's own surface. It is a point estimate, because this is a screen; PREREG-5 tests non-inferiority with an interval.
- **G3 (added to the task's two clauses)**: a no-op accuracy alone is met by a model that copies every unfamiliar input. The copy guard on the posted twins in the same surface rules that out, as PREREG-3's twin guard did (copied <= 0.10).
- **TRIGGER WIDENED** uses PREREG-3's trigger threshold (0.30) for 'stays near T2 s0'.
- **All three seeds.** The question is about the recipe, not one artifact.

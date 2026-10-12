# T2b-ledger: a surface-diversified ledger rule adapter, its dev screen and PREREG-5 (kit design, revision 3)

Status: KIT REVISION 3 BUILT ON THE CPU (2026-10-12), NOTHING FROZEN FOR IT, ITS HOOK NOT INSTALLED, NO GPU USED BY IT. Joint plan
item 5 (D-45, `docs/roadmap/JOINT_PLAN_2026-10-10.md`). Files: `scripts/t2bl_*.py`, `data/t2bl_*`, `docs/t2b/ledger/*`,
`PREREG_5_DRAFT.md`. Revision 3 follows the revision-2 dev screen's STOP (DECISIONS R29) under the gate rule's pre-stated revision
path; section 10 says what changed and why, section 10b the operator decisions applied to revision 3 on 2026-10-12 before
any freeze or training (exact posted / non-posted balance on the revised surfaces, the training set regenerated under the
same seed with the dev bank byte-identical; the first revision-3 build is kept unused in `superseded_rev3_preop/`).
Revision 2 (frozen D-48; its dev screen `runs/t2bl/dev`) is SPENT: its data are in
`data/t2bl_superseded_rev2/`, its records, rendered draft and code copy in `docs/t2b/ledger/superseded_rev2/`. Revisions 0 and 1
(never trained on) stay in `superseded_rev0/` and `superseded_rev1/`.

## 1. Why

- PREREG-3 (REPORT 3c): T2 s0's ledger no-op is the token 'void' (unseen statuses 0.01-0.18), and it fails out of range on
  magnitudes (amounts 13-490: 0.14-0.25; 4-5-digit balances 0.50-0.60), reordered keys (0.37-0.47) and word account names (0.60-0.66).
- PREREG-LB (REPORT 3d): as BWM's planner's world model T2 s0 plans at 0.706 on its training surface but 5/180 on the held-out
  combined surface.
- D-35 / BWM D-54: every claim gets an out-of-range and a surface-form audit; the shared wording is "generalises to unseen X"
  (held-out surfaces never in training: absolute bar, non-inferior to its own canonical accuracy, significantly above the
  unaugmented arm T2 s0), "trigger widened" (better only on trained surface variants, near T2 s0 on held-out ones), otherwise
  "partial" with the per-variant table.
- R19 (RCX): neither OFF nor T2 s0 reads a rule supplied in the prompt. This kit trains the frame's FIXED rule on diversified
  surfaces; at most it can show "follows the fixed rule on unseen surfaces".
- R29 (the revision-2 dev screen): held-out status words passed everywhere (H_words 0.99-1.00, HS_words 0.98-0.99, every kind
  >= 0.90, in distribution +0.02 to +0.04 over T2 s0), but the posted twins on the combined held-out surface fell to 0.40-0.54
  with 16-35% copied (G3 failed in all seeds). Revision 3 is the pre-stated response (section 10).

## 2. The partitions (fixed at build time; `scripts/t2bl_common.py`; recorded in `data/t2bl_train.SOURCE.json` "partitions")

Every held-out partition the dev gate reads is split into a DEV half (dev bank and dev gate only) and a CONFIRM half (PREREG-5's
sealed map bank only); both halves are excluded from training, so the confirmation never re-tests the items the gate conditioned on.
Unchanged since revision 2 except the top-level key order (section 10; the data test checks the partitions record against
revision 2's, the top-level entry aside).

| factor | TRAIN | HELD-OUT, DEV half | HELD-OUT, CONFIRM half |
|---|---|---|---|
| not-applied words | rejected declined failed blocked suspended withdrawn aborted revoked annulled deferred queued paused | rescinded refused stalled nullified | halted vetoed retracted denied |
| neutral words | open new flagged reviewed manual standard accepted checked pinned tagged | verified valid signed routine | marked linked normal seen |
| nonce words | blick zorp kiki mipa tove frell glorp nim | toma zib fep wux | blorf snik trav quib |
| void variants | voids voiding void_ voidz VOId vOid | voyd vooid v0id re_void | viod vo_id vo1d vojd |
| 'posted' look-alikes | postedx xposted posted_no zposted posted2 qposted | yposted mposted tposted posteda | aposted oposted sposted postedy |
| reserved statuses | PREREG-3's 16 strings: never trained, in neither half (a comparable secondary stratum of the confirmation) | | |
| excluded statuses (no defensible gold) | Posted POSTED "posted." " posted" "posted " booked recorded settled cleared completed confirmed logged entered applied executed processed done: never used | | |
| account names | A-D; letters I J K L M N P Q R S T U V; words bills bonus cafe coins dues farm gifts house media petty pool roof salary shop tips trust | W X Y Z; barn dock lamp mill nest pier silo yard | E F G H; fund loan rent wage fort pond vault gold |
| PREREG-LB names | bank card cash fees: reserved for PREREG-LB's rendering (report-only strata of both banks, planning part b) | | |
| account count | 4, 5, 6, 7 | 8 (not split: the gate does not read it) | 8 |
| balances | T2's pool (abs <= 98), 3-digit, 4-digit (0.40 / 0.30 / 0.30); scale rows t2 / 3-digit only | 5-digit (not split; fresh values) | 5-digit |
| add / transfer amounts | T2's 21 values, 2-digit 13-99, 3-digit 100-999 (0.45 / 0.30 / 0.25) | 4-digit (not split) | 4-digit |
| scale factors | T2's recorded marginal only (train and test; magnitude claims cover add / transfer) | none | none |
| key orders, event and before | every permutation except the 6 held-out ones per object (n = 4..8) | 3 random orders each, >= 2 adjacent swaps from sorted (indices 1-3) | the reversal + 2 random orders each (indices 0, 4, 5) |
| key order, top-level {before, event} | BOTH orders since revision 3 (revision 2: the sorted order only, the reversal held out) | none (the bank strata still reverse it: a trained level) | none (likewise) |
| rule text | paraphrases 0 (Titan's verbatim system text) - 3 | paraphrase 4 | paraphrase 5 |

**Token-novel held-out words (binding).** Under Titan's tokenizer, in the JSON context `"status":"<w>"`, every held-out word of
both halves contains neither the token 'void' nor the token 'posted' and has at least one token that no trained status has
(`t2bl_audit.token_audit`; 58 trained status tokens). The trained variants keep the anchor tokens on purpose (voids = [void, s],
xposted = [x, posted]); revision 1's voidd = [void, d], voidy, xvoid, postedq, posted_zero and posted9 were near-trained and are
gone. Look-alikes of 'posted' are affixed forms, never misspellings (a misspelt 'posted' has no defensible gold, as PREREG-3's
Posted / POSTED). PREREG-3's strings: Void and VOID are single tokens never trained; voided = [void, ed] and unposted = [un, posted]
are token-near-trained (a stated limit of the secondary stratum).

**Magnitudes.** Training widens T2's range by two decimal orders while keeping T2's own range as a class; no training prompt or
target contains a number with more than 4 digits (every posted gold is checked; scale rows keep balances of at most 3 digits),
so a 5-digit balance is a digit length never seen anywhere in training. Scale factors stay small: multi-digit x multi-digit
multiplication is a separate capacity question.

## 3. The training set (`scripts/t2bl_make_data.py`, kit revision 3, seed 20261165, generator t2bl-data-v3.1; `data/t2bl_train.jsonl`)

- 1,500 oracle-labelled one-step rows on Titan's ledger frame family (T2: 501 ledger rows inside 1,501 rows of three families).
  Target unchanged: `env.dumps(after)`. Gold = `p3_common.gold_of` (= `rcwmt.env.step`), asserted equal to the explicit rule and to
  an independent implementation in the audit.
- CANON block, 300 rows: the exact canonical T2 frame (byte-identical to `env.one_step_messages`), 100 posted / void twin pairs
  + 100 posted singles.
- DIV block, 1,200 rows: 300 twin pairs (posted + a non-posted member, same base and surface) + 250 posted and 350 non-posted
  singles. Each DIV slot has a SURFACE MODE by EXACT COUNTS within each slot type (`SURFACE_MODE_COUNTS`; canonical / stacked /
  free: pairs 45 / 150 / 105, posted singles 0 / 150 / 100, non-posted singles 100 / 150 / 100):
  - **canonical** (190 rows): the exact canonical surface with the slot's own status (non-void words also occur on the canonical frame);
  - **stacked** (600 rows; new in revision 3): every surface factor at a non-canonical TRAINING level at once: paraphrase 1-3,
    training letters or training words as names (4-7 accounts), non-sorted training orders of the event and before objects, the
    reversed top-level order;
  - **free** (410 rows): every factor drawn independently from its TRAINING partition: paraphrases 0-3, event / before orders
    (sorted w.p. 0.25), names A-D-based 0.35 / training letters 0.30 / training words 0.35; the top-level order reversed on an
    EXACT half of the free slots of each slot type (pairs 53 / 52, posted and non-posted singles 50 / 50 each; `FREE_TOP_QUOTA`,
    drawn from the plan stream); a free slot is never the full stack (paraphrase and inner orders redrawn), so the stacked
    training surface is exactly the stacked mode.
  Stacked and free slots draw 4-7 accounts (4 w.p. 0.55) and magnitudes by digit class exactly as revision 2. Non-posted DIV
  statuses: void 50 and 600 training words (not-applied 180, neutral 150, nonce 120, void variants 60, posted look-alikes 90; 15
  rows per word, 10 per void variant). The counts are STATUS-SYMMETRIC wherever the surface is revised (operator decision,
  section 10b): pairs share their surface, posted and non-posted singles get the same number of stacked (150) and free (100, half
  reversed) slots, and the 100 extra non-posted singles (the DIV block is 550 posted / 650 non-posted, offsetting the CANON
  block's 200 / 100) are on the canonical surface. Posted / non-posted per mode: stacked 300 / 300, free 205 / 205, DIV
  canonical 45 / 145, CANON block 200 / 100 (canonical surface overall 245 / 245).
- Totals: posted 750 / non-posted 750; kinds by PREREG-3's quota (transfer 677, scale 497, add 326); every posted event changes
  the state; max |value| in any prompt or target 9,962; max scale-row balance 999; 4 accounts 1,033 rows, 5-7: 467; 425 distinct
  event orders; 356 rows on the fully canonical surface with posted / void.
- **Revision-3 surface counts (independent read-back of the prompt text, binding where marked):** top-level order sorted 694 rows
  (347 posted / 347 non-posted), reversed 806 rows (403 / 403) [binding: each order on >= 300 rows]; the stacked training surface
  600 rows (exactly the stacked mode), 300 posted / 300 non-posted, so 0.40 of the 750 posted rows [binding: >= 1/3] and 0.500
  posted within [binding: 0.49-0.51]; the reversed-top rows 0.500 posted [binding: 0.49-0.51]. Revision 2 for comparison: 0
  reversed-top rows, and only 124 posted rows (0.165) even on the stacked surface without the top-order clause. The first
  revision-3 build (superseded unused, section 10b): stacked 656 rows (300 posted, 0.457), reversed 788 rows (360 posted,
  0.457), under a [0.40, 0.60] band.
- Heuristic training accuracies (binding bounds in brackets): copy 0.50 [0.48, 0.52]; apply-regardless = ignore-status 0.50
  [0.48, 0.52]; T2's void trigger 0.64 [<= 0.70]; string cues: contains 'posted' 0.94, starts with 'p' 0.95, 6 letters 0.94,
  ends in 'ed' 0.80, 'means not applied' 0.76 (each <= 0.95); the rule 1.00. Revision 3 (binding [0.49, 0.51]): on the stacked
  rows copy 0.500 / apply-regardless 0.500 (n 600), on the reversed-top rows 0.500 / 0.500 (n 806), so 'copy on an unfamiliar
  surface' and 'apply on an unfamiliar surface' are each wrong on exactly half of those rows (no surface-to-status cue there).
  Descriptive: the DIV block copy 0.54, void trigger 0.55; the canonical surface copy 0.50 and void trigger 0.76 (the dev gate's
  G1b tests the canonical surface).
- Token statistics: mean 160.78 tokens per row incl. EOS (revision 2: 160.73; T2's ledger rows 146), so 190 steps x 16 = 2.03
  epochs as in T2.

## 4. Audit (`scripts/t2bl_audit.py`; independent of the renderer)

Every training prompt is split by its own regular expression and parsed with `json.loads` (literal key order, duplicate keys
rejected); held-out orders are re-derived from the seed by the audit's own copy of the procedure; every factor level is
classified from the partition tables, with the dev and confirm halves told apart; the gold is recomputed independently.
Binding: no held-out factor of either half in any training prompt (0 hits); none of 107 held-out text fragments (both
paraphrases' phrases, every held-out / PREREG-3 / excluded status, every held-out and PREREG-LB name) in any prompt or target
(0); no number above 4 digits; the token audit; the read-back equals the row's payload and surface; the canonical block is
Titan's exact frame; pair members differ only in the status; the partitions are disjoint (train, dev half, confirm half,
PREREG-3; exactly and after lower-casing, except the lower-case form 'void' shared by the training case variants and PREREG-3's
reserved strings); names never collide with field names, kinds or statuses; since revision 3, both top-level orders trained on
>= 300 rows each, >= 1/3 of the posted rows on the stacked surface, and 0.49-0.51 posted on the stacked and on the reversed-top
rows (operator decision; built exactly 0.50). Bank audit (the dev bank here, the map bank at the freeze): every row is its stratum's declared change of its base in the
bank's own half and nothing else (the top-level reversal of H_order, HS_*, H_order_top and LB_* is declared as 'reversed', a
trained level since revision 3), mixed strata follow the base parity, golds independent, twins present. Freshness (binding): 0
payload and 0 operative-content overlaps with the union (338,271 payloads, 322,546 operative contents, 95 sources; 0 freshness
rejections); since revision 3 the union of the kit build includes every superseded kit file under `data/t2bl_superseded_rev*/`
(only this kit's own top-level `data/t2bl_*` files are skipped), so the fresh dev bank and the training set share nothing with the
spent revision-2 dev bank or training set (also checked directly); HSKILL_CONFIRM_V2* skipped by name, never opened;
`rcwmt.bank.load_exclusions()` never called; dev rows exclude each other across bases; training rows exclude the dev bank.
`--verify` re-derives both files byte for byte and rebuilds the union (ok).

Heuristics on the dev bank's held-out words (H_words, HS_words; descriptive, `AUDIT.json`): copy 1.00 by construction (all
no-op rows), T2's void substring cue 0.05 (re_void), the 'void' token 0.00, a trigger widened to the trained anchor tokens 0.00,
a trigger on any token shared with a trained no-op word 0.30, contains 'posted' 0.80 (it fails the posted look-alikes, 20%),
starts with 'p' 0.95, 6 letters 0.95. The mixed factor cells (H_order, H_names, H_count, H_para): copy and apply-regardless 0.50.

## 5. Dev bank (kit revision 3, seed 20261166; 100 bases; 19 strata; 1,801 rows; the DEV half)

Drawn FRESH under the new seed by revision 2's builder and strata, unchanged (the data test regenerates it byte for byte with the
revision-2 code copy in `superseded_rev2/code/` from the new seed and the recorded rejections; same per-stratum counts). Each
base (A-D, |b| <= 98, a trained amount, an effective posted event, kind quota 45/33/22) is rendered in every stratum with ONE
factor changed (or one combined surface): ID_posted, ID_void (in-distribution); S_words (training word, canonical surface);
S_surface_posted / S_surface_words (a training surface on the base's own in-range values: paraphrase 1-3, training names, 5-7
accounts, training orders, top sorted); S_bal / S_amt (the trained 3-4-digit balance and 2-3-digit amount bands; add / transfer
bases); H_words (held-out word); H_order (event and before in held-out orders, top reversed); H_names (W-Z or 4 of 8 held-out
words); H_count (8 accounts); H_para (paraphrase 4); H_order, H_names, H_count and H_para posted on even bases and the base's
S_words training word on odd bases; H_bal (5-digit balances); H_amt (4-digit amounts, add / transfer bases); HS_words / HS_posted
(the COMBINED held-out surface: paraphrase 4, held-out names, held-out event and before orders, top reversed, with a held-out
word / posted); LB_void / LB_posted (PREREG-LB's rendering, report-only); P3_words (PREREG-3's 16 strings). Since revision 3 the
top-level reversal in H_order, HS_* and LB_* is a trained level (section 10).

## 6. Runner (`scripts/t2bl_run.py`) and the dev gate (`scripts/t2bl_gate.py`)

- T2 recipe (D-15/D-16): the vendored BWM trainer invoked exactly as `run_t2.py` does (LoRA r16 / alpha 32 / dropout 0.05 on
  q,k,v,o,gate,up,down, FP32 Titan-Sparse, WSD, lr 1e-4, batch 16, 190 steps), seeds 0, 1, 2 on the same file; one adapter per
  seed, never merged (D-28).
- Outputs: `runs/t2bl/dev_rev3/` (`t2bl_common.DEV_RUN`; tag dev). The runner never reads or writes revision 2's spent
  `runs/t2bl/dev`. One lease `rcwm-t1:t2b-ledger`, 4.5 h cap (`--cap-h` may only lower it), D-22 pairing (a seed trains only if
  1.25 x its whole pair fits; never within 0.5 h of the cap), deferral and resume (exit 5), A0 anchor every turn, two consecutive
  turns without a recorded stage stop the job, create-only outputs with provenance, orphaning of partial stages, a stale
  DONE.json.partial moved aside, a run lock, the one-waiter rule, every installed `*t2bl*.sh` hook equal to the draft, the frozen
  gate rule and every frozen input checked before the lease.
- Evaluation per load: OFF (disable_adapter), T2 s0 (adapter 't2', PREREG-2 pins) and the seed's adapter ('t2bl'), unmerged,
  greedy, 160 tokens; E0 (OFF and T2 s0 on the whole dev bank) once, E per seed.
- GPU plan (`GEN_PLAN.json`): one turn, 3.38 h at the planning rate (2.57 h at the model-based observed rate): s0 T, A0, E0, s0 E,
  s1 T, s1 E, s2 T, s2 E; 10,669 generations. Revision 2's measured stages (provenance metadata, same row length): T 1,526 / 1,595 s
  (one outlier, s1 T 6,656 s, deferred s2 to a second turn of 0.67 h), E 457-687 s, E0 858 s, A0 309-330 s: about 2.2 h for one
  turn without such an outlier. Three train + eval pairs plus E0 under one lease extends D-22's "T2 pairs only" as approved in
  D-48 (5); the DECISIONS entry that freezes revision 3 should restate it.
- Dev gate rule: revision 2, text and thresholds UNCHANGED (byte-identical `t2bl_gate.RULE_TEXT`; verbatim in `GATE_RULE.md`,
  which adds an "Inputs of this kit revision" note): PROCEED iff in all three seeds G1 HS_words >= 0.80, G1b H_words >= 0.80, G1c
  every word kind >= 0.60 on both, G2 ID accuracy minus T2 s0's >= -0.05, G3 HS_posted >= 0.80 correct and copied <= 0.10, G4
  HS_words and H_words each within 0.10 of the seed's own ID_void accuracy, the run complete and every anchor passed; TRIGGER
  WIDENED iff S_words >= 0.80, H_words <= 0.30 and the H_words errors mostly applied as posted in every seed; PARTIAL otherwise
  (named patterns); NO VERDICT on an incomplete or stopped run (a stopped run gets a NO VERDICT record) or a failed anchor. The
  revision path after a STOP is pre-stated. Freezing the rule (`--freeze-rule`, an operator step) also pins the sha256 of
  PREREG-5's rules (`rules_digest`) and of its decision code (`code_digest`); since revision 3 the record also carries the kit
  revision, its seeds, its run directory and the spent revision-2 artifacts. `--freeze-ready` writes `GATE_FREEZE_READY.json`
  (what the freeze would pin now, its preconditions and the exact command; not a freeze). P(PROCEED) at a true held-out-word
  accuracy of 0.85 / 0.90 / 0.95: 0.00 / 0.12-0.28 / 0.96-0.97, against PREREG-5's P(GENERALISES) on those cells of 0.00 / 0.14 /
  0.91-0.99 (unchanged: the power records are identical to revision 2's).
- The registered analysis writes GATE.md (with the revision note), then GATE.json (the hook's completion marker); an orphan
  GATE.md or a stale .partial is moved aside, never deleted.

## 7. PREREG-5 (`PREREG_5_DRAFT.md`, draft revision 3; `scripts/t2bl_confirm_bank.py`, `t2bl_confirm_run.py`, `t2bl_confirm_analysis.py`)

(a) the one-step map on a fresh sealed bank (28 strata, 2,701 rows, the CONFIRM half) with OFF, T2 s0 and T2b-ledger s0-s2
(revision 3's adapters, `runs/t2bl/dev_rev3/s<k>`): per factor GENERALISES / TRIGGER WIDENED / PARTIAL with the D-54 wording,
'near T2 s0' two-sided (within 0.10), the trained-variant control at least 0.10 above T2 s0, a below_t2s0 pattern, word-cluster
intervals (report-only); (b) LB-style planning on the held-out combined surface and on canonical (T2BL s0 vs T2 s0, paired, with
ORACLE), BWM's planner pinned; (c) the 5c retention guard on a confirmation draw minted at run time. Bank and draw are minted at
the freeze from a commit-reveal seed (BWM's salt; the union digest published before the co-sign and quoted in it); spent seeds
20261161-20261166 and 20261141-20261143 are refused. The freeze refuses a rules or decision-code change since the gate freeze
unless it is listed (rendered into Appendix A). Revision 3 changed registered strings before any revision-3 gate freeze: the
limit on the shared top-level reversal (now: a trained level; H_order_top is a trained-variant report), a new limit naming the
revision, the 'rules unchanged' sentence ("before any adapter trained on this kit revision's data existed") and (operator
decision, section 10b) a new limit: the top-level key order is held out nowhere, so key-order claims cover the inner orders
only. The draft's Appendix A lists every registered-string and decision-code change made after the revision-2 dev outputs
existed, with old / new sha256 (rules 37b96032... -> 71dd5938..., 68 -> 70 strings; code 148d180c... -> 1408be8a...), and the
four rule-text statements kept verbatim from revision 2 that the revision-3 note supersedes. Stage
start rule and GPU plan as revision 2 (`P5_GEN_PLAN.json`: 2 turns at the observed rate, 3.19 + 2.14 h; 3 at the planning rate;
no simulated turn exceeds 3.94 h); part (c) needs one more RCWM turn. Full-size mint rehearsal against the real union (stand-in
tags and salt): 2,701 rows and 180 instances, 0 freshness rejections, 48 s, 339 MB.

## 8. Review response (revision 2)

Applied: token-novel held-out words with a binding tokenizer check (MUST-FIX); in-range trained-surface controls and separate
magnitude controls (MUST-FIX); the write_doc policy that rewrites PREREG-5's pre-freeze records in place until the freeze, with a
fake-root test of the documented path (MUST-FIX); gate clauses G1b, G1c, G3 accuracy and G4, the trigger-widened branch on
H_words with the error-class condition, P(PROCEED) against P(GENERALISES); DEV / CONFIRM halves of every gated held-out
partition and word-cluster intervals; seed-correlation and independent-within-ORACLE power regimes; two-sided 'near' with a
control gain and below_t2s0; mixed posted / training-word factor cells; state-dependent gate tests; the ceiling-bounded start
rule and a plan simulated with it; reboot-aware attempts; PREREG-5 decision code pinned at the gate freeze and a change list at
the freeze; a full-size mint rehearsal as a pre-freeze record; the union digest quoted in the co-sign; ret_mint recovery that
checks time and other consumers; stale .partial handling, GATE.md before GATE.json, an orphan results MD moved aside; NO VERDICT
record for a stopped dev run; hook globs; the cap clamp; a no-progress stop in the dev runner; the stale-lock race; held-out
orders at least two swaps from sorted with H_order_top / H_order_inner; no training value above 4 digits; the limits reworded
(near-trained PREREG-3 strings, shared items, thresholds set with knowledge of aggregates, row independence); T2S0_LIKE updated;
the 55 / 67 effective bar documented; the tolerance in the power simulation.
Declined (partly): mixing H_bal / H_amt with non-posted rows (the magnitude factor is an arithmetic claim; status reading is
tested by the mixed surface cells and the combined cell, and the label and a limit say "posted add / transfer events"); mixing
the factor cells with 'void' (T2 s0's own cue would hand it half of every cell; training words do the job without that).

## 9. Tests (`scripts/t2bl_test.py`; records `docs/t2b/ledger/TEST_*.json`)

Revision 3 after the operator decisions (section 10b; `--part all`, 2026-10-12 16:49-16:54): 236/236 checks (data 66, gate 23,
p5analysis 9, plan 11, devtiny 26, p5tiny 26, p5freeze 42, rehearsal 5, hook 28; peak RAM 993 MB, devtiny); the first
revision-3 build had 227/227 (data 59, hook 26). Parts: data (audit, token audit, halves, perturbation refusals, partitions,
renderer; revision 3: the seeds and the spent ones, both top orders trained with posted = non-posted in each, the
stacked-surface share of the posted rows against revision 2's, posted = non-posted EXACTLY on the stacked-surface and
reversed-top rows within the [0.49, 0.51] band, the 1/3 floor in the code, the exact mode counts and free top quotas, the stacked
surface exactly the stacked mode, the regeneration under the same seed (kinds and statuses per slot, the canon block and the pair
modes unchanged), the dev bank byte-identical to the first build's, the first build refused under the new band, unchanged
vocabularies / quotas / partitions, no payload shared with the spent revision-2 files, the fresh dev bank regenerated by revision
2's builder code, a reversed top order not refused, the audit refusing revision 2's composition, a posted-only-off-stacked set
and sets 0.513 posted on the stacked or on the reversed-top rows), gate (scorer self-test, every gate branch, frozen-state
checks, the gate freeze with its revision-3 fields, a stopped run, an orphan GATE.md, P(PROCEED), hook glob, the installed hook's
state, the rule text and every threshold byte-identical to revision 2's, the revision note, the freeze-ready record), p5analysis
(every PREREG-5 wording branch, verbatim strings, code digest, power records), plan (both `--plan` paths without torch; every
PREREG-5 schedule within 4.5 h; revision 3's run directory), devtiny (tiny random Qwen3 + LoRA: full run with the real vendored
trainer, crash / resume, deferral, no-progress stop, stale DONE partial, stops), p5tiny (the PREREG-5 job with tiny models and
the pinned BWM planner), p5freeze (fake root: freeze refusals, freeze, a listed change in Appendix A, verify, tamper, the
documented path to the freeze, the registered job, hook glob, retention binding and mint recovery, the registered analysis),
rehearsal (the full-size mint against the real union: 2,701 rows, 180 instances, 0 freshness rejections, 339 MB), hook (both
hook drafts under bash with a stub step, incl. the 60 s sleep before every exit-5 re-call; master_chain.sh's own hook loop with an
in-place replacement of an installed hook and a new name created mid-loop; the draft's WHEN IT RUNS text). Also run:
`t2bl_make_data.py --verify` (both files re-derived, union rebuilt: 338,271 payloads, 0 overlaps, audit ok), `t2bl_gate.py
--selftest` / `--dryrun`, `t2bl_confirm_analysis.py --dryrun` (35/35 cases), both `--plan` paths, `t2bl_gate.py --freeze-ready`
(ready). PREREG-5 pre-freeze checks on the real root: 19/24; the other 5 wait on the dev screen by design.

## 10. Revision 3 (2026-10-12): what changed and why (DECISIONS R29)

**Trigger.** The revision-2 dev gate (frozen D-48) returned STOP (PARTIAL), pattern heldout_posted_twins: G1, G1b, G1c, G2 and G4
passed in all three seeds, G3 failed in all three (HS_posted 0.54 / 0.40 / 0.42, copied 0.16 / 0.31 / 0.35); every no-op stratum
and every single held-out factor passed. The combined held-out surface also reverses the top-level JSON object order (event
before balances), which revision-2 training never varied; R29 names it the suspected main contributor (untested). Only these
gate-reported aggregates informed revision 3; no revision-2 model output was read to choose anything.

**The pre-stated revision path** (verbatim in the rule): "a revised kit may change only the training data (composition,
vocabulary sizes, magnitude bands, paraphrases), must draw a fresh dev bank under a new seed (this dev bank and every adapter
trained on the old data are then recorded as spent), and keeps these thresholds." Revision 3 does exactly that:

1. **Training data only (composition).** (a) The top-level key order of the user JSON is varied in a training subset, both
   orders (806 reversed / 694 sorted rows). Consequence, deliberate and stated here: the dev bank's combined held-out surface
   (and H_order, H_order_top, LB_*) no longer contains an untrained top-level factor; the reversed top order is a trained level
   there, not a new held-out factor. Every other held-out factor of the combined surface (paraphrase 4, held-out names, held-out
   event and before orders, held-out words) is unchanged and disjoint from training. (b) Posted rows under STACKED training-surface
   variation (training paraphrase + training names + training inner orders + reversed top order at once) rise from 0 (124 without
   the top clause) to 300 of the 750 posted rows (0.40), via a 'stacked' surface mode on half of the diversified slots, so the
   posted / non-posted balance (750 / 750) and the heuristic audit (copy = apply-regardless = 0.50) are kept, and posted and
   non-posted rows are EXACTLY equal on the stacked rows (300 / 300) and on the reversed-top rows (403 / 403) (section 10b). The
   first revision-3 build said "no surface-to-status cue exists (0.543 / 0.457 on the stacked rows)": that was wrong, 0.457
   posted on the stacked and on the reversed-top rows is a residual cue ('copy on an unfamiliar surface' right on 54% of them);
   it was corrected before any freeze or training. Row count 1,500, 190 steps, 2.03 epochs, the oracle, the frame family and the
   targets are unchanged.
   (c) **Left unchanged, with reasons.** Magnitude bands: the posted-twin failure is on in-range values (HS_posted carries the
   base's |b| <= 98 and a trained amount), so widening the bands would not address it and would confound the revision; the
   magnitude cells (H_bal, H_amt: 0.06-0.75) are not gated and stay a reported limit. Vocabularies: every word stratum passed
   (H_words 0.99-1.00, HS_words 0.98-0.99, every kind >= 0.90), so the status vocabulary and its quotas stay as they are (the data
   test checks the per-word row counts against revision 2). Held-out partitions: changing them would spend the confirm half's
   independence and break the dev / confirm split; they are byte-identical (orders regenerate from 20261163; the partitions record
   equals revision 2's except the top-level entry, which is the change in (a) and is not held out any more).
2. **A fresh dev bank under a new seed.** Seed 20261166 (training 20261165; 20261165-20261179 unused in every tree under
   WorldModel/ on 2026-10-12, HSKILL_CONFIRM_V2* excluded by name). Same strata and builder code path (regenerated byte for byte by
   the revision-2 code copy); the DEV halves are the same partitions, the bases and values are fresh, and the bank shares no payload
   with the spent revision-2 bank or training set. **Spent and recorded:** the revision-2 training set (seed 20261161, sha256
   89d6a01e...) and dev bank (seed 20261162, f710b17e...), now in `data/t2bl_superseded_rev2/`, and the three revision-2 adapters
   in `runs/t2bl/dev` (left where they are; never an input; their sha256 in `GATE_FREEZE_READY.json` "spent").
3. **The gate rule text and thresholds unchanged** (byte-identical to the frozen revision-2 text; every constant equal to revision
   2's). The rule text names its inputs as they were in revision 2 ("data/t2bl_dev.jsonl (seed 20261162 ...)", "tag dev"); under
   the revision path that names the spent bank, and the revision-3 inputs are stated in `GATE_RULE.md`'s "Inputs of this kit
   revision" note, in every GATE.md and in the freeze record (the operator may instead decide to update the seed in the text; that
   would change the rule text's sha256). The revision-3 run uses tag dev in `runs/t2bl/dev_rev3` (the spent directory is never
   reused). `GATE_FREEZE_READY.json` lists what `--freeze-rule` would pin now; nothing is frozen. The stale statements of the
   rule text are listed one by one in the PREREG-5 draft's Appendix A: "pre-stated on 2026-10-10 before any T2b-ledger adapter
   or output exists", "data/t2bl_dev.jsonl (seed 20261162 ...)", "(tag dev)" (the tag is still dev, the directory is
   `runs/t2bl/dev_rev3`) and "every object in a held-out key order" (event and before only); the revision note covers each
   (since section 10b it also states the first one explicitly).
4. **Hooks.** `queue_hook_t2bl.sh.draft` now drives `runs/t2bl/dev_rev3` (comments updated; it still sleeps 60 s before every
   exit-5 re-call, at most three calls per pass). The installed `scripts/queue_hooks/88_t2bl.sh` is revision 2's (byte-identical to
   `superseded_rev2/queue_hook_t2bl.sh.draft`); it is a no-op now (GATE_RULE.json moved aside; `runs/t2bl/dev/DONE.json` and
   GATE.json exist) and must be replaced by the revision-3 draft after the revision-3 freeze (the runner refuses otherwise).
   The replacement is IN PLACE (same name): a live chain that has not yet reached 88_t2bl.sh in its pass sources the new file,
   a chain inside or past it does not (section 10b; the draft's WHEN IT RUNS comment, tested with master_chain.sh's own loop).
   `queue_hook_prereg5.sh.draft` binds retention to `runs/t2bl/dev_rev3/s0/train/adapter`.

**Other code changes (no rule change):** `t2bl_common.run_outputs` / `write_doc` look at this revision's run directory only (the
spent `runs/t2bl/dev` no longer blocks records or the freeze); the union of the kit build includes `data/t2bl_superseded_rev*/`;
`t2bl_confirm_run` reads the dev screen from `runs/t2bl/dev_rev3`; `t2bl_confirm_bank.SPENT_SEEDS` adds 20261161-20261162.

## 10b. Operator decisions on revision 3 (2026-10-12, before any freeze, training or GPU use)

The independent check of the first revision-3 build found a residual surface-to-status cue that the kit's own records denied,
an inaccurate hook comment and a floor documented as 1/3 but coded as 0.33. The operator decided (applied here, nothing frozen):

1. **No surface-to-status cue on the revised surfaces.** Posted and non-posted rows are now EXACTLY equal on the stacked-surface
   rows (300 / 300, 600 rows: 0.40 of the posted rows, the stacked share kept) and on the reversed-top rows (403 / 403, 806 rows),
   so copy and apply-regardless score exactly 0.50 on both; the overall 750 / 750, the 1,500 rows and 2.03 epochs are kept. The
   new binding audit band for those subsets is [0.49, 0.51] (`t2bl_common.SURFACE_SUBSET_BOUNDS`, read by the independent audit
   and the heuristic audit; it was [0.40, 0.60]); the data test shows refusals just outside it (300 / 585 and 403 / 786 = 0.513).
   How (`t2bl_make_data`, generator t2bl-data-v3.1): status-symmetric exact mode counts per slot type (section 3), the free
   slots' top order by an exact half per slot type instead of a per-slot coin, and free slots kept off the full stack, so the
   stacked surface is exactly the stacked mode. Side effect, not required but welcome: the canonical surface is balanced too
   (245 / 245; its void trigger 0.76, was 0.83). Mode totals 190 / 600 / 410 (were 181 / 600 / 419).
   **Seed.** The SAME training seed 20261165: the generator allowed the quota change without changing the seed semantics. Every
   stream is derived from the seed exactly as before (`t2bl-train-plan|<seed>`, `t2bl-train|<seed>|<block>|<slot>|<attempt>`,
   `t2bl-train-shuffle|<seed>`); only the plan's quotas changed (the free slots' top quota is drawn last from the plan stream).
   So every slot keeps its event kind and statuses, the 300 canon-block rows are byte-identical and every pair keeps its mode
   (data test); 996 of the 1,500 rows are byte-identical to the first build's. The first build was never trained on, frozen or
   evaluated, so the seed is not spent and 20261167 was not needed. **The dev bank (seed 20261166) is byte-identical**
   (731bbc6a...; same builder, seed and union: 338,271 payloads, 322,546 operative contents, 95 sources, 0 rejections); its
   SOURCE record changed only in gen_version, the build-code hashes, built_local and a 'superseded_unused' note.
   New training set: sha256 88f04e6b17c2c0e50b0e655f4f59e747232cc2ce8db0f49f189b406fd4fc8ba4 (was 43646b0d...).
   **Kept, not deleted:** the first build's training set, SOURCE records, records, rendered draft, hook draft and code copy in
   `docs/t2b/ledger/superseded_rev3_preop/` (README.txt). It is outside every freshness union on purpose: a copy of the dev bank
   under data/ would collide with the identical current one, and the regenerated training set would be pushed off every slot it
   shares with the first build; neither file was ever seen by a model.
   **Wording corrected:** the first build's DESIGN (section 3 and 10), AUDIT.json note, `t2bl_make_data` / `t2bl_common`
   comments and the PREREG-5 draft's Appendix A / C said "no surface-to-status cue" or "posted and non-posted alike" while the
   revised subsets were 0.457 posted; they now state the exact balance and the band, and name the first build's numbers.
2. **PREREG-5 draft.** Appendix A lists every registered-string and decision-code change made after the revision-2 dev outputs
   existed, with old / new sha256 (section 7), adds the limit that the top-level key order is held out nowhere (a new registered
   limit; key-order claims cover the inner orders only) and lists the four stale statements of the gate rule text (seed, tag,
   'pre-stated ... before any adapter or output exists', 'every object in a held-out key order'), kept verbatim from revision 2
   and superseded by the revision-3 note (`t2bl_gate.revision_note`, which now also covers the dated preamble).
3. **Hook draft.** The WHEN IT RUNS comment of `queue_hook_t2bl.sh.draft` now describes the install as an in-place replacement
   of the installed 88_t2bl.sh: master_chain.sh expands its hook glob once but sources each hook by name when it reaches it, so a
   live chain that has not reached 88 sources the new file and a chain inside or past it does not; the operator replaces
   88_t2bl.sh at a waiting point or with the chain stopped, then starts ONE fresh chain only if no master_chain.sh /
   confirm_chain.sh process is alive. Logic unchanged (it still sleeps 60 s before every exit-5 re-call; the hook test also runs
   master_chain.sh's own loop line against an in-place replacement). `GATE_FREEZE_READY.json` "hook" now carries the install
   command and when to run it. **Stacked-share floor:** 1/3 in both the code (`MIN_STACKED_POSTED_SHARE = 1 / 3`, was 0.33) and
   the documents; the training set holds 0.40.

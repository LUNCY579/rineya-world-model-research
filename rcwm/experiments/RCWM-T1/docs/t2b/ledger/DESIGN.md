# T2b-ledger: a surface-diversified ledger rule adapter, its dev screen and PREREG-5 (kit design, revision 2)

Status: KIT BUILT ON THE CPU, NOTHING FROZEN, NO HOOK INSTALLED, NO GPU USED. Joint plan item 5 (D-45, `docs/roadmap/JOINT_PLAN_2026-10-10.md`).
Files: `scripts/t2bl_*.py`, `data/t2bl_*`, `docs/t2b/ledger/*`, `PREREG_5_DRAFT.md`. Revision 2 (2026-10-10 evening) applies two reviews
(validity / statistics and engineering) of revision 1; revision-1 data and records are kept, unread, in `data/t2bl_superseded_rev1/` and
`docs/t2b/ledger/superseded_rev1/` (no adapter was ever trained on them).

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

## 2. The partitions (fixed at build time; `scripts/t2bl_common.py`; recorded in `data/t2bl_train.SOURCE.json` "partitions")

Every held-out partition the dev gate reads is split into a DEV half (dev bank and dev gate only) and a CONFIRM half (PREREG-5's
sealed map bank only); both halves are excluded from training, so the confirmation never re-tests the items the gate conditioned on.

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
| key orders | every permutation except the 6 held-out ones per object; the identity w.p. 0.25; the 2-key top object never varied | event and before (n = 4..8): 3 random orders each, >= 2 adjacent swaps from sorted (indices 1-3) | the reversal + 2 random orders each (indices 0, 4, 5); top reversed (shared) |
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

## 3. The training set (`scripts/t2bl_make_data.py`, seed 20261161; `data/t2bl_train.jsonl`)

- 1,500 oracle-labelled one-step rows on Titan's ledger frame family (T2: 501 ledger rows inside 1,501 rows of three families).
  Target unchanged: `env.dumps(after)`. Gold = `p3_common.gold_of` (= `rcwmt.env.step`), asserted equal to the explicit rule and to
  an independent implementation in the audit.
- CANON block, 300 rows: the exact canonical T2 frame (byte-identical to `env.one_step_messages`), 100 posted / void twin pairs
  + 100 posted singles.
- DIV block, 1,200 rows: 300 twin pairs (posted + a non-posted member, same base and surface) + 250 posted and 350 non-posted
  singles. 15% of DIV slots use the exact canonical surface with the slot's own status (so non-void words also occur on the
  canonical frame). Otherwise every factor is drawn from its TRAINING partition. Non-posted DIV statuses: void 50 and 600
  training words (not-applied 180, neutral 150, nonce 120, void variants 60, posted look-alikes 90).
- Totals: posted 750 / non-posted 750; kinds by PREREG-3's quota (transfer 0.45, scale 0.33, add 0.22); every posted event
  changes the state; max |value| in any prompt or target 9,974; max scale-row balance 998; 4 accounts 1,043 rows, 5-7: 457;
  398 distinct event orders; 378 rows on the fully canonical surface.
- Heuristic training accuracies (binding bounds in brackets): copy 0.50 [0.48, 0.52]; apply-regardless = ignore-status 0.50
  [0.48, 0.52]; T2's void trigger 0.64 [<= 0.70]; string cues: contains 'posted' 0.94, starts with 'p' 0.95, 6 letters 0.94,
  ends in 'ed' 0.80, 'means not applied' 0.76 (each <= 0.95); the rule 1.00. The DIV block alone: copy 0.54, void trigger 0.55;
  on the canonical surface the void trigger is right on 0.86 of rows (the dev gate's G1b tests the canonical surface).
- Token statistics: mean 161 tokens per row incl. EOS (T2's ledger rows 146), so 190 steps x 16 = 2.03 epochs as in T2.

## 4. Audit (`scripts/t2bl_audit.py`; independent of the renderer)

Every training prompt is split by its own regular expression and parsed with `json.loads` (literal key order, duplicate keys
rejected); held-out orders are re-derived from the seed by the audit's own copy of the procedure; every factor level is
classified from the partition tables, with the dev and confirm halves told apart; the gold is recomputed independently.
Binding: no held-out factor of either half in any training prompt (0 hits); none of 107 held-out text fragments (both
paraphrases' phrases, every held-out / PREREG-3 / excluded status, every held-out and PREREG-LB name) in any prompt or target
(0); no number above 4 digits; the token audit; the read-back equals the row's payload and surface; the canonical block is
Titan's exact frame; pair members differ only in the status; the partitions are disjoint (train, dev half, confirm half,
PREREG-3; exactly and after lower-casing, except the lower-case form 'void' shared by the training case variants and PREREG-3's
reserved strings); names never collide with field names, kinds or statuses. Bank audit (the dev bank here, the map bank at the
freeze): every row is its stratum's declared change of its base in the bank's own half and nothing else, mixed strata follow
the base parity, golds independent, twins present. Freshness (binding): 0 payload and 0 operative-content overlaps with the
union (329,532 payloads, 320,334 operative contents, 85 sources; 0 freshness rejections), HSKILL_CONFIRM_V2* skipped by name,
never opened; `rcwmt.bank.load_exclusions()` never called; dev rows exclude each other across bases; training rows exclude the
dev bank. `--verify` re-derives both files byte for byte and rebuilds the union.

Heuristics on the dev bank's held-out words (H_words, HS_words; descriptive, `AUDIT.json`): copy 1.00 by construction (all
no-op rows), T2's void substring cue 0.05 (re_void), the 'void' token 0.00, a trigger widened to the trained anchor tokens 0.00,
a trigger on any token shared with a trained no-op word 0.30, contains 'posted' 0.80 (it fails the posted look-alikes, 20%),
starts with 'p' 0.95, 6 letters 0.95. The mixed factor cells (H_order, H_names, H_count, H_para): copy and apply-regardless 0.50.

## 5. Dev bank (seed 20261162; 100 bases; 19 strata; 1,801 rows; the DEV half)

Each base (A-D, |b| <= 98, a trained amount, an effective posted event, kind quota 45/33/22) is rendered in every stratum with
ONE factor changed (or one combined surface): ID_posted, ID_void (in-distribution); S_words (training word, canonical surface);
S_surface_posted / S_surface_words (a training surface on the base's own in-range values: paraphrase 1-3, training names, 5-7
accounts, training orders); S_bal / S_amt (the trained 3-4-digit balance and 2-3-digit amount bands; add / transfer bases);
H_words (held-out word); H_order (every object in a held-out order); H_names (W-Z or 4 of 8 held-out words); H_count (8
accounts); H_para (paraphrase 4); H_order, H_names, H_count and H_para posted on even bases and the base's S_words training word
on odd bases; H_bal (5-digit balances); H_amt (4-digit amounts, add / transfer bases); HS_words / HS_posted (the COMBINED
held-out surface: paraphrase 4, held-out names, held-out key orders, with a held-out word / posted); LB_void / LB_posted
(PREREG-LB's rendering, report-only); P3_words (PREREG-3's 16 strings).

## 6. Runner (`scripts/t2bl_run.py`) and the dev gate (`scripts/t2bl_gate.py`)

- T2 recipe (D-15/D-16): the vendored BWM trainer invoked exactly as `run_t2.py` does (LoRA r16 / alpha 32 / dropout 0.05 on
  q,k,v,o,gate,up,down, FP32 Titan-Sparse, WSD, lr 1e-4, batch 16, 190 steps), seeds 0, 1, 2 on the same file; one adapter per
  seed, never merged (D-28).
- One lease `rcwm-t1:t2b-ledger`, 4.5 h cap (`--cap-h` may only lower it), D-22 pairing (a seed trains only if 1.25 x its whole
  pair fits; never within 0.5 h of the cap), deferral and resume (exit 5), A0 anchor every turn, two consecutive turns without a
  recorded stage stop the job, create-only outputs with provenance, orphaning of partial stages, a stale DONE.json.partial moved
  aside, a run lock, the one-waiter rule, every installed `*t2bl*.sh` hook equal to the draft, the frozen gate rule and every
  frozen input checked before the lease.
- Evaluation per load: OFF (disable_adapter), T2 s0 (adapter 't2', PREREG-2 pins) and the seed's adapter ('t2bl'), unmerged,
  greedy, 160 tokens; E0 (OFF and T2 s0 on the whole dev bank) once, E per seed.
- GPU plan (`GEN_PLAN.json`): one turn, 3.38 h at the planning rate (2.56 h observed): s0 T, A0, E0, s0 E, s1 T, s1 E, s2 T, s2 E.
  This runs three train + eval pairs plus E0 under one lease, as t2bp / t2br did; D-22 says "Scope: T2 pairs only", so the
  DECISIONS entry that freezes the gate rule or installs hook 88 should state this extension.
- Dev gate rule (revision 2, pre-stated, verbatim in `GATE_RULE.md`): PROCEED iff in all three seeds G1 HS_words >= 0.80, G1b
  H_words >= 0.80, G1c every word kind >= 0.60 on both, G2 ID accuracy minus T2 s0's >= -0.05, G3 HS_posted >= 0.80 correct and
  copied <= 0.10, G4 HS_words and H_words each within 0.10 of the seed's own ID_void accuracy, the run complete and every anchor
  passed; TRIGGER WIDENED iff S_words >= 0.80, H_words <= 0.30 and the H_words errors mostly applied as posted in every seed;
  PARTIAL otherwise (named patterns); NO VERDICT on an incomplete or stopped run (a stopped run gets a NO VERDICT record) or a
  failed anchor. The revision path after a STOP is pre-stated. Freezing the rule (`--freeze-rule`, an operator step) also pins
  the sha256 of PREREG-5's rules (`rules_digest`) and of its decision code (`code_digest`). P(PROCEED) at a true held-out-word
  accuracy of 0.85 / 0.90 / 0.95: 0.00 / 0.12-0.28 / 0.96-0.97, against PREREG-5's P(GENERALISES) on those cells of 0.00 / 0.14 /
  0.91-0.99.
- The registered analysis writes GATE.md, then GATE.json (the hook's completion marker); an orphan GATE.md or a stale
  .partial is moved aside, never deleted.

## 7. PREREG-5 (`PREREG_5_DRAFT.md`; `scripts/t2bl_confirm_bank.py`, `t2bl_confirm_run.py`, `t2bl_confirm_analysis.py`)

(a) the one-step map on a fresh sealed bank (28 strata, 2,701 rows, the CONFIRM half) with OFF, T2 s0 and T2b-ledger s0-s2:
per factor GENERALISES / TRIGGER WIDENED / PARTIAL with the D-54 wording, 'near T2 s0' two-sided (within 0.10), the
trained-variant control at least 0.10 above T2 s0, a below_t2s0 pattern, word-cluster intervals (report-only); (b) LB-style
planning on the held-out combined surface and on canonical (T2BL s0 vs T2 s0, paired, with ORACLE), BWM's planner pinned;
(c) the 5c retention guard on a confirmation draw minted at run time. Bank and draw are minted at the freeze from a
commit-reveal seed (BWM's salt; the union digest published before the co-sign and quoted in it). The freeze refuses a rules or
decision-code change since the gate freeze unless it is listed (rendered into Appendix A). Stage start rule: a planning stage is
bounded by its ceiling generation count at the measured rate; attempts cut off by a reboot do not count as failures. GPU plan
(`P5_GEN_PLAN.json`, simulated with the runner's own start rule): 2 turns at the observed rate (3.19 + 2.14 h), 3 at the planning
rate (3.30 + 3.16 + 1.67 h), 4 at the planning-rate ceiling, 5 at 2.5 generations/s; no simulated turn exceeds 3.94 h. Part (c)
needs one more RCWM turn (lease `rcwm-t1:retention`, about 0.5-1.1 h). A full-size rehearsal of the mint against the real union
(stand-in tags and salt): 2,701 rows (2 freshness rejections), 180 instances, 63 s, 339 MB.

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

Parts: data (audit, token audit, halves, perturbation refusals, partitions, renderer), gate (scorer self-test, every gate
branch, frozen-state checks, the gate freeze, a stopped run, an orphan GATE.md, P(PROCEED), hook glob), p5analysis (every
PREREG-5 wording branch, verbatim strings, code digest, power records), plan (both `--plan` paths without torch; every PREREG-5
schedule within 4.5 h), devtiny (tiny random Qwen3 + LoRA: full run with the real vendored trainer, crash / resume, deferral,
no-progress stop, stale DONE partial, stops), p5tiny (the PREREG-5 job with tiny models and the pinned BWM planner: full run, C1,
crash / resume with the exception recorded, reboot-interrupted attempts, deferral, A0 failures, stops), p5freeze (fake root:
freeze refusals incl. the union digest and unlisted changes, freeze, a listed change in Appendix A, verify, tamper, the
documented path to the freeze with the real pre-freeze checks, the registered job, hook glob, retention binding and mint
recovery, the registered analysis), rehearsal (the full-size mint against the real union), hook (both hook drafts under bash with
a stub step).

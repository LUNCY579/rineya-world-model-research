# T2b-ledger: a surface-diversified ledger rule adapter, its dev screen and PREREG-5 (kit design, revision 1)

Status: KIT BUILT ON THE CPU, NOTHING FROZEN, NO HOOK INSTALLED, NO GPU USED. Joint plan item 5 (D-45, `docs/roadmap/JOINT_PLAN_2026-10-10.md`).
Files: `scripts/t2bl_*.py`, `data/t2bl_*`, `docs/t2b/ledger/*`, `PREREG_5_DRAFT.md`.

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

| factor | TRAIN | HELD-OUT (dev screen and PREREG-5 only) |
|---|---|---|
| status, not-applied words | rejected declined failed blocked suspended withdrawn aborted revoked annulled deferred queued paused | rescinded refused stalled nullified |
| status, neutral words | open new flagged reviewed manual standard accepted checked pinned tagged | verified valid signed routine |
| status, nonce words | blick zorp kiki mipa tove frell glorp nim | toma zib fep wux |
| status, case / spelling variants of void | voids voiding void_ voidz VOId vOid | voidd re_void voidy xvoid |
| status, posted look-alikes | postedx xposted posted_no zposted posted2 qposted | postedq yposted posted_zero posted9 |
| status, reserved | PREREG-3's 16 strings (pending draft cancelled reversed held approved active ok dax wug Void VOID voided unposted not_posted "not posted") are NEVER trained: a comparable secondary stratum | |
| status, excluded (no defensible gold) | Posted POSTED "posted." " posted" "posted " booked recorded settled cleared completed confirmed logged entered applied executed processed done: never used | |
| account names | A-D; letters I J K L M N P Q R S T U V W; words bills bonus cafe coins dues farm gifts house media petty pool roof salary shop tips trust | letters E F G H; words bank card cash fees fund loan rent wage barn dock lamp mill nest pier silo yard |
| account count | 4, 5, 6, 7 | 8 |
| balances | T2's pool (abs <= 98), 3-digit, 4-digit (classes 0.40 / 0.30 / 0.30) | 5-digit (10,000-99,999) |
| add / transfer amounts | T2's 21 trained values (-9..12 without -3), 2-digit 13-99, 3-digit 100-999 (0.45 / 0.30 / 0.25) | 4-digit (1,000-9,999) |
| scale factors | T2's recorded marginal only (train and test; magnitude claims cover add / transfer) | none |
| key orders (position permutations) | every permutation except the held-out ones; the identity w.p. 0.25; the 2-key top object never varied | top: reversed; event (6 keys) and before (n = 4..8 accounts): the reversal + 2 fixed random permutations each, from seed 20261163 |
| rule text | paraphrase 0 (Titan's verbatim system text) and paraphrases 1-3 | paraphrase 4 |

Why these magnitude bands. Training widens T2's range by two decimal orders while keeping T2's own range as a class (so the
canonical rows stay in distribution); the held-out band is the next digit count, so "unseen magnitude" means more digits than
any training input, not a value gap inside a trained band. Scale factors stay small: multi-digit times multi-digit
multiplication is a separate capacity question PREREG-3 already separated.

Case variants of 'void'. The task asked for case variants among the training words; held-out strings must differ from every
trained string even after lower-casing, so case variants of 'void' are training items only (VOId, vOid) and PREREG-3's Void /
VOID stay reserved (they become near-trained for T2b-ledger, a stated limit of the secondary stratum).

## 3. The training set (`scripts/t2bl_make_data.py`, seed 20261161)

- 1,500 oracle-labelled one-step rows on Titan's ledger frame family (T2: 501 ledger rows inside 1,501 rows of three families).
  Target unchanged: `env.dumps(after)`, the compact sorted-key balances object. Gold = `p3_common.gold_of` (= `rcwmt.env.step`),
  asserted equal to the explicit rule and to an independent implementation in the audit.
- CANON block, 300 rows: the exact canonical T2 frame (byte-identical to `env.one_step_messages`), 100 posted / void twin pairs
  + 100 posted singles.
- DIV block, 1,200 rows: 300 twin pairs (posted + a non-posted member, same base and surface) + 250 posted and 350 non-posted
  singles. 15% of DIV slots use the exact canonical surface with the slot's own status (so non-void words also occur on the
  canonical frame; otherwise a surface-conditioned 'void' cue would fit). Otherwise every factor is drawn from its TRAINING
  partition. Non-posted DIV statuses: void 50 and 600 training words (not-applied 180, neutral 150, nonce 120, void variants 60,
  posted look-alikes 90; 15 rows per word, 10 per void variant).
- Totals: posted 750 / non-posted 750; kinds by PREREG-3's quota (transfer 0.45, scale 0.33, add 0.22); every posted event
  changes the state.
- Heuristic training accuracies (binding bounds in brackets): copy 0.50 [0.48, 0.52]; apply-regardless = ignore-status 0.50
  [0.48, 0.52]; T2's void trigger 0.64 [<= 0.70]; string cues: contains 'posted' 0.94, starts with 'p' 0.95, 6 letters 0.94,
  ends in 'ed' 0.80, 'means not applied' 0.76 (each <= 0.95); the rule (exact and case-insensitive 'posted') 1.00. The DIV
  block alone: copy 0.54, void trigger 0.55. On the canonical surface the void trigger is right on about 0.86 of rows.
- Token statistics: mean 161 tokens per row incl. EOS (T2's ledger rows 146), so 190 steps x 16 = 2.03 epochs as in T2.

## 4. Audit (`scripts/t2bl_audit.py`; independent of the renderer)

Every training prompt is split by its own regular expression and parsed with `json.loads` (literal key order, duplicate keys
rejected); held-out orders are re-derived from the seed by the audit's own copy of the procedure; every factor level is
classified from the partition tables; the gold is recomputed independently. Binding: no held-out factor in any training
prompt (0 hits); none of 76 held-out text fragments in any prompt or target (0); the read-back equals the row's payload and
surface; the canonical block is Titan's exact frame; pair members differ only in the status; the status partitions are
disjoint exactly and after lower-casing (except the lower-case form 'void' itself, shared by design by the training case
variants and PREREG-3's reserved strings); no held-out string lower-cases to 'void'; names never collide with field names,
kinds or statuses. Dev bank: every row is its stratum's declared factor change of its base and nothing else (classified from
the rendered text); golds independent; twins present. Freshness (binding): 0 payload and 0 operative-content overlaps with
the union (329,532 payloads, 320,334 operative contents, 85 sources: `lbeam_problems.load_exclusions` with the name-skipping
historical harvest, every pinned source, PREREG-3's bank, every other data/**/*.json[l], the binding transitions of PREREG-LB's
spent dev sets and its spent confirmation draw, every {before, event} payload under docs/); HSKILL_CONFIRM_V2* skipped by
name, never opened; `rcwmt.bank.load_exclusions()` never called; dev rows exclude each other across bases; training rows
exclude the dev bank. `--verify` re-derives both files byte for byte from (seed, recorded rejections) and rebuilds the union.

## 5. Dev bank (seed 20261162; 100 bases; 17 strata; 1,667 rows)

Each base (A-D, abs(b) <= 98, a trained amount, an effective posted event, kind quota 45/33/22) is rendered in every stratum
with ONE factor changed (or one combined surface): ID_posted, ID_void (in-distribution); S_words (training word, canonical
surface); S_surface_posted / S_surface_words (a training surface: paraphrase 1-3, training names, 4-7 accounts, training key
orders, 3-4-digit balances, 2-3-digit amounts); H_words (held-out word); H_order (every object in a held-out order); H_names
(E-H or 4 held-out words); H_count (8 accounts); H_para (paraphrase 4; posted / void); H_bal (5-digit balances); H_amt (4-digit
amounts, add / transfer bases); HS_words / HS_posted (the COMBINED held-out surface: paraphrase 4, held-out names, held-out key
orders, with a held-out word / posted); LB_void / LB_posted (PREREG-LB's heldout_combined rendering, 'void' as 'cancelled');
P3_words (PREREG-3's 16 strings).

## 6. Runner (`scripts/t2bl_run.py`) and the dev gate (`scripts/t2bl_gate.py`)

- T2 recipe (D-15/D-16): the vendored BWM trainer invoked exactly as `run_t2.py` does (LoRA r16 / alpha 32 / dropout 0.05 on
  q,k,v,o,gate,up,down, FP32 Titan-Sparse, WSD, lr 1e-4, batch 16, 190 steps), seeds 0, 1, 2 on the same file; one adapter per
  seed, never merged (D-28).
- One lease `rcwm-t1:t2b-ledger`, 4.5 h cap, D-22 pairing (a seed trains only if 1.25 x its whole pair fits; never within
  0.5 h of the cap), deferral and resume (exit 5), A0 anchor every turn, create-only outputs with provenance, orphaning of
  partial stages, a run lock, the one-waiter rule, the frozen gate rule and every frozen input checked before the lease.
- Evaluation per load: OFF (disable_adapter), T2 s0 (adapter 't2', PREREG-2 pins) and the seed's adapter ('t2bl'), unmerged,
  greedy, 160 tokens; E0 (OFF and T2 s0 on the whole dev bank) once, E per seed.
- GPU plan (`GEN_PLAN.json`): one turn, 3.32 h at the planning rate (2.53 h observed): s0 T, A0, E0, s0 E, s1 T, s1 E, s2 T, s2 E.
- Dev gate rule (revision 1, pre-stated, verbatim in `GATE_RULE.md`): PROCEED iff in all three seeds HS_words >= 0.80, ID
  accuracy minus T2 s0's >= -0.05, HS_posted copied <= 0.10, the run complete and every anchor passed; TRIGGER WIDENED iff
  S_words >= 0.80 and HS_words <= 0.30 in every seed; PARTIAL otherwise (named patterns); NO VERDICT on an incomplete run or a
  failed anchor. The revision path after a STOP is pre-stated. Freezing the rule (`--freeze-rule`, an operator step) also pins
  the sha256 of PREREG-5's rules (`t2bl_confirm_analysis.rules_digest()`).

## 7. PREREG-5 (`PREREG_5_DRAFT.md`; `scripts/t2bl_confirm_bank.py`, `t2bl_confirm_run.py`, `t2bl_confirm_analysis.py`)

(a) the one-step map on a fresh sealed bank (24 strata, 2,367 rows) with OFF, T2 s0 and T2b-ledger s0-s2; (b) LB-style
planning on the held-out combined surface and on canonical (T2BL s0 vs T2 s0, paired, with ORACLE), BWM's planner pinned;
(c) the 5c retention guard on a confirmation draw minted at run time. Bank and draw are minted at the freeze from a
commit-reveal seed (BWM's salt). GPU plan (`P5_GEN_PLAN.json`): 2 turns at the observed rate (4.01 + 1.22 h), 3 at the
planning rate (3.12 + 3.16 + 1.67 h), 4 at the planning-rate ceiling; part (c) needs one more RCWM turn (lease
`rcwm-t1:retention`, about 0.5-1.1 h).

## 8. Tests (`scripts/t2bl_test.py`; records `docs/t2b/ledger/TEST_*.json`)

Parts: data (audit, perturbation refusals, partitions, renderer), gate (scorer self-test, every gate branch), p5analysis
(every PREREG-5 wording branch, verbatim strings, power), devtiny (tiny random Qwen3 + LoRA: full run with the real vendored
trainer, crash / resume, deferral, stops), p5tiny (the PREREG-5 job with tiny models and the pinned BWM planner: full run,
C1, crash / resume, deferral, A0 failures, stops), p5freeze (fake root: freeze refusals, freeze, verify, tamper, run, retention
binding, registered analysis), plan (both `--plan` paths without torch), hook (both hook drafts under bash with a stub step).

Result of the build (2026-10-10, CPU only): data 34/34, gate 14/14 (scorer self-test: 11,669 gold renderings, 15,003 negatives, parser identity 13,872/13,872; 15 gate cases), p5analysis 8/8 (31 PREREG-5 wording cases, 64 registered strings verbatim in the draft), plan 7/7, devtiny 24/24, p5tiny 21/21, p5freeze 27/27, hook 24/24: 159/159 checks; `t2bl_make_data.py --verify` with the rebuilt union: ok. Peak RAM 996 MB (devtiny).

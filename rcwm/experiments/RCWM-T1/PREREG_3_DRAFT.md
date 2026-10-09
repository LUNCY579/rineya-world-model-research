# RCWM-T1 PREREG-3 (DRAFT): does the T2 ledger adapter follow the frame's status rule out of range, or a cue? (one-step out-of-range strata and four-step chains, adapter ON vs OFF in three seeds, no training)

**Status:** DRAFT, not frozen. Freezing needs the user's explicit go-ahead, recorded by `scripts/p3_run.py --freeze --approval "..."` (§7, §9).

Template: `PREREG_2.md`. Where this document is silent, PREREG-2's conventions apply (runtime, decoding, parsing, lease and turn rules, create-only outputs, deviation policy). No model output of any kind was read to choose anything in this plan; `runs/t2/*/gen_*` files were hashed, never opened. This is revision 2 of the draft: it applies the pre-freeze review of 2026-10-09 (Appendix C); the revision-1 bank was superseded before any model saw it (§2.1).

## 1. Question

PREREG-2 confirmed that the T2 seed-0 LoRA, unmerged on frozen FP32 Titan-Sparse, makes ledger four-step rollout decisions near-oracle on a fresh bank (regret 0.544 → 0.056, exact 0.029 → 0.698). One of its registered limits: **"the bank is fresh at payload level, not out-of-distribution."** T2's 501 ledger training rows covered only:

| T2 training (data/t2_train.jsonl, recomputed by the bank builder) | values |
|---|---|
| statuses | posted 375, void 126 |
| amounts | −9..12 (the recorded marginal's 21 values) |
| balances | \|before\| ≤ 98 |
| accounts | exactly A, B, C, D |
| JSON | compact, keys sorted (`env.one_step_messages`) |
| tick | 1..99 |

Titan's own 384 training rows (`rrp015 … microfit-train.jsonl`) have the same ledger statuses (posted 111, void 17).

So step-level void accuracy 0.04–0.13 → 1.00 cannot tell apart what the adapter could have learned:

- **RULE:** what the frame's system text states, "If status is not posted, leave all balances unchanged", plus arithmetic at any magnitude, account naming or key order;
- **a semantic cue:** status words that *mean* "not applied" (void, cancelled, pending, ...) are no-ops; other words are applied;
- **a substring cue:** a status that contains "posted" is applied, anything else is a no-op;
- **TRIGGER:** the literal token `void` means no-op, and the arithmetic holds only in range.

BWM-001 F-01/F-06 found exactly such a data-forced shortcut in a forward LoRA, visible only out of range. The answer decides whether rule data must be surface-diversified before roadmap items 2–3, and how PREREG-2's result may be worded. **No training is involved.**

Verdicts (§5):
- **V1 status:** RULE / TRIGGER / PARTIAL (PARTIAL with named patterns: twins, semantic, posted_substring, trigger_not_interpretable, no_adapter_gain);
- **V2 magnitude (add/transfer), V2s magnitude (scale), V3 naming, V4 format:** GENERALISES / ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN / DOES NOT GENERALISE / HARMED / MIXED.

## 2. Data

### 2.1 Bank

- **File:** `data/bank_p3_oor.jsonl`, sha256 `0fd18295d6eaa84d0f451c94b81ea1ecf37ab88e8ab298b58820d80ac740bf12`.
- **Record:** `data/bank_p3_oor.SOURCE.json` (schema 2).
- **Builder:** `scripts/p3_make_bank.py`, with `scripts/p3_common.py` (both hashed in the SOURCE record and at the freeze).
- **Seed:** 20261103, which no earlier RCWM-T1 builder, analysis or bootstrap used.
- **No model, tokenizer or model output was used;** torch was never imported.
- **Builds.** Three rehearsals into the session scratchpad and three builds into `data/` (the second and third only re-recorded the SOURCE after code-only changes to `p3_common.py` and to the builder's census pass; the earlier files were moved aside, not used) all produced the identical sha. `p3_make_bank.py --verify` re-derives everything from the file (§2.6).
- **Contents:** 1,500 one-step rows (100 bases × 15 strata) and 80 four-step chains.
- **Superseded revision-1 bank.** The bank of the first draft (sha `9e7108df6aa681b8f08769b2e37b976933c8406ffd671034f542e4068881b545`, 14 strata with independently drawn magnitude, naming and format rows) was superseded on 2026-10-09 before any model saw it, because the review found two design gaps: no unseen status without a not-applied meaning, and magnitude/naming/format rows that could not be paired with the posted twins. The change is a design change made on input composition only. The file is kept, unused, in `docs/prereg3/superseded/` with its SOURCE record.

### 2.2 Frame and gold

- **Frame.** Titan's one-step frame, `rcwmt.env.one_step_messages(ledger_replay, before, event)`, which is exactly what T2 trained on. The system text is verbatim:
  > Apply one event to the supplied account balances. If status is not posted, leave all balances unchanged. add: add amount to account. scale: multiply account by amount. transfer: subtract amount from account and add it to other, simultaneously. Return only the complete resulting balances as a JSON object.
- **F_keyorder rows and reordered CH_X steps** use the same system text and the same compact JSON, with the recorded key order (`p3_common.user_json`). Nothing else in the frame changes.
- **Gold.** `rcwmt.env.step` treats every status other than the exact string `posted` as a no-op, which is what the rule text says. On every row and every chain transition the builder asserts:

  `env.step == p3_common.rule_gold == the validated FDS oracle (env.cross_check)`

  `rule_gold` is the rule text implemented explicitly and independently.

### 2.3 Strata, and how the stated rule gives each gold

**Design: one factor at a time on 100 shared bases.** A base is an `env.gen_start` balance set (|b| ≤ 99, accounts A–D) and an `env.gen_event` event (trained amounts) whose posted version changes the state. Event kinds are drawn to a fixed quota: transfer 45, scale 33, add 22 (the recorded marginal 58/42/28 of 128, rounded; the quota was fixed in the first draft after a first draw came out 63% transfer, p ≈ 0.0003 under the marginal; it was chosen on input composition only, with no model output involved). The 100 bases came from 165 draws (13 rejected because the posted event was a no-op, 52 because the kind quota was full). **Every one of the 15 strata is the same 100 bases with exactly one factor changed**, so every row has a posted, in-distribution twin (the T_posted row of its base), every stratum inherits the kind quota, and every out-of-range effect is a paired, kind-matched contrast. Row ids are `p3-<stratum>-<base index>`.

| stratum | n | group | the one factor changed (sub-variants by base index) | gold | why the frame's rule text gives that gold |
|---|---|---|---|---|---|
| T_posted | 100 | status | none: status `posted` (the twin) | apply | status is "posted": the arithmetic clauses |
| C_void | 100 | status | status `void` (trained) | unchanged | "If status is not posted, leave all balances unchanged" |
| U_pending | 100 | status | status `pending` | unchanged | same clause: "pending" is not "posted" |
| U_draft | 100 | status | status `draft` | unchanged | same |
| U_cancelled | 100 | status | status `cancelled` | unchanged | same |
| U_reversed | 100 | status | status `reversed` | unchanged | same. A reversal is not "posted"; its natural net effect (no change) also agrees with the gold, so this stratum does not set rule against semantics |
| U_held | 100 | status | status `held` | unchanged | same |
| U_neutral | 100 | status | status `approved` / `active` / `ok` / `dax` / `wug` (20 each; index mod 5) | unchanged | same clause. None of the five is "posted" under the literal or the case-insensitive reading, none is a synonym of "posted", and none means "not applied": approved and ok are positive, active is neutral, dax and wug are nonce words. Here the rule and world-knowledge semantics come apart (an "approved" transaction would naturally be applied) |
| N_voidcase | 100 | status | `Void` 34 / `VOID` 33 / `voided` 33 (index mod 3) | unchanged | same: none is the string "posted" |
| N_postedlike | 100 | status | `unposted` 34 / `not_posted` 33 / `not posted` 33 (index mod 3) | unchanged | same: each contains "posted" but none is "posted", and each literally negates it |
| M_bal | 100 | magnitude | balances: every balance redrawn with 4 or 5 digits (−99,704..99,419); the base's event | apply | arithmetic clauses (no magnitude limit is stated) |
| M_amt | 100 | magnitude | amount: \|amount\| 13..99 (even bases) or 100..500 (odd bases), random sign; the base's balances | apply | arithmetic clauses |
| A_renamed | 100 | naming | names: A–D relabelled to E–H (even) or to 4 of {bank, card, cash, fees, fund, loan, rent, wage} in sorted order (odd) | apply | "the supplied account balances" (no names are stated); "complete resulting balances" |
| A_many | 100 | naming | number of accounts: extra accounts E.. (5/6/7/8 accounts, index mod 4; extra balances in range); the event is the base's (touches A–D) | apply | same |
| F_keyorder | 100 | format | key order: `{"event":…,"before":…}` with inner keys sorted (even: top-level swap only) or with event and balance keys also permuted, never sorted (odd) | apply | key order is not part of the rule; answers are scored order-insensitively |

**Unseen-status strata** (the RULE conditions, §5): U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_voidcase, N_postedlike, i.e. every status stratum whose status string never occurs in T2's or Titan's training. The first five words mean "not applied"; U_neutral does not; N_voidcase are spellings of the trained token; N_postedlike contain the string "posted".

**What the rule text settles, and what is left out.** The frame's rule is stated for any status ("If status is not posted"), so every included stratum has an unambiguous gold under the documented frame, and no prompt variant is needed. Two candidate strata were deliberately left out:

- **Case variants and synonyms of `posted`** (`Posted`, `POSTED`; booked, recorded, settled, cleared, completed, confirmed). The literal reading (string ≠ "posted" → no-op) and the natural reading disagree, and the rule text does not settle which applies. They have no defensible gold (Appendix B, Q2).
- **"Inverted rule text"** (roadmap). It is not Titan's documented frame. Testing it would change the system prompt, not the input.

**Rule-consistency check** (`rule_consistency` in the SOURCE record): gold == before on every no-op row and gold == rule_gold ≠ before on every apply row, 100/100 in every stratum. This is a consistency check of the builder and holds by construction (gold_of asserts env.step == rule_gold); the argument that the rule *text* implies each gold is the table above.

**Chains** (accounts A–D, so PREREG-2's decision pipeline applies; K = 4 candidates per parent, H = 4):
- **CH_S (10 parents, 40 chains):** balances, amounts and keys in range and sorted; each event is posted (p .50), an unseen status word (p .35, the 10 of U_* equally) or a near-miss / look-alike (p .15, 6 equally); every chain has ≥ 1 posted and ≥ 1 non-posted event; step classes posted 78, unseen 24, neutral 26, near_void 18, posted_like 14; final |balance| ≤ 288.
- **CH_X (10 parents, 40 chains):** each event posted (p .60) or one of the 16 non-posted strings (p .40); |amount| 13..500 on add/transfer w.p. .40 (scale keeps a trained amount, so products do not compound); each step's keys reordered w.p. .50; every chain has ≥ 1 posted OOR-amount event and ≥ 1 reordered step; final |balance| ≤ 3,228, and 13 of 40 chains end beyond ±300 (the phi clip, §4).
- **Copy-chain exact floor at k = 4:** 0.000 in both strata.

### 2.4 Out-of-range checks (builder; all pass)

- No unseen, neutral, near-miss or look-alike status occurs in T2's or Titan's ledger training rows (both: posted / void only).
- T_posted and C_void are fresh **in-distribution controls** (trained statuses on in-range bases); CH_S events are in range.
- M_bal: every |b| ≥ 1,000 > 98. Scale golds reach 796,275 (median 91,145); add/transfer golds median 70,708.
- M_amt: every |amount| ≥ 13, outside −9..12. Scale golds median 1,806, max 16,286; add/transfer golds median 150.
- A_renamed: no account named A–D. A_many: 5–8 accounts.
- F_keyorder: the user JSON differs from the sorted frame and starts with the event.
- Every row outside M_amt, and every CH_S event, keeps a trained amount; every row outside M_bal has |before| ≤ 99.

### 2.5 Freshness

**Enforced by the builder (binding; re-verified from the file):**

1. 0 payload overlaps and 0 FDS-operative-content overlaps for every row and chain transition against each of 16 pinned sources: a union of 308,726 payloads and 301,783 operative contents. For a non-posted event the operative content is (before, not-posted), so an unseen-status row is also fresh against every non-posted row of every source with the same balances.

   | source | payloads | operative contents | overlap (payload / operative) |
   |---|---|---|---|
   | historical records tree (name-skipping harvest, below) | 6,464 | 6,258 | 0 / 0 |
   | bank_pilot | 15,629 | 15,278 | 0 / 0 |
   | bank_confirm | 13,552 | 13,247 | 0 / 0 |
   | **bank_confirm2** (PREREG-2) | 12,163 | 11,903 | 0 / 0 |
   | bank_s4000 | 255,745 | 250,614 | 0 / 0 |
   | t2_train / t2_eval / t2_eval_xframe / t2_eval_supp | 1,501 / 600 / 600 / 360 | 1,496 / 600 / 600 / 360 | 0 / 0 each |
   | SELECT-VIEWS-001 | 384 | 383 | 0 / 0 |
   | **L1 OOD bank** (`docs/l1/ood_bank.jsonl`) | 600 | 600 | 0 / 0 |
   | FDS T_BANK / T_BANK_V2 | 192 / 192 | 192 / 192 | 0 / 0 each |
   | composition_confirmation T_BANK_V3X / STAGE_A_T_V3X / T_BANK_V4 | 864 / 15 / 864 | 864 / 15 / 864 | 0 / 0 each |

2. 0 payload and 0 operative-content overlaps with every **multi-event** ledger prompt (`{before|s0, events}`, also JSON-in-string) in the records tree and the pinned sources, rolled out with env.step: 934 prompts, 1,068 transitions in 7 files (t2_eval_xframe and DMRF-001). The payload harvest above does not expand multi-event prompts; this pass does.

**Historical harvest.** The historical set is harvested by `scripts/l1_make_ood.harvest_historical`. That is FDS `harvest_exclusions` line for line, except that files named `HSKILL_CONFIRM_V2*` are skipped by name and never opened or hashed (DECISIONS I-28). It skipped exactly the four reserved files; the census and rollout pass skips the same four by name. `rcwmt.bank.load_exclusions()` is **not** called.

**Wide scan (descriptive pre-freeze record, `docs/prereg3/WIDE_SCAN.json`, `scripts/p3_wide_scan.py`):** every *.json / *.jsonl file under `C:/Users/clclc/BusinessProject` that mentions a ledger state (single-step objects, multi-event prompts and bank parents rolled out), with HSKILL_CONFIRM_V2* skipped by name, `runs/t2/*/gen_*` skipped by path and the PREREG-3 bank itself excluded: WIDE_SCAN_RESULT. The review's independent scan did the same for the superseded revision-1 bank.

**Within the bank:**
- payloads are unique, except that each F_keyorder row has its twin's payload (only the prompt's key order differs, by design);
- an operative content repeats only inside one base or inside one chain parent.

**Status census** (descriptive; every occurrence of a bank status string as a JSON "status" value in the records tree and the pinned sources, including multi-event prompts; "ledger event" = a field of an object that also has `kind` and `amount`):
- in ledger events: void 19,424; **held 293** (13 DMRF-001 files: matrix-CHECK prompts, predictions and MATRIX_B, scout; mostly multi-event evaluation prompts, which the single-step payload union sees only twice); **draft 2** (FDS T_BANK, a Titan-era evaluation bank); every other bank status 0 (pending, cancelled, reversed, approved, active, ok, dax, wug, Void, VOID, voided, unposted, not_posted, not posted);
- outside ledger events: "ok" 55,403 times, all as a generation record's status;
- none of the bank's non-posted strings except void occurs in any training set.

So the unseen statuses are unseen by T2 and by Titan's training, and nearly absent from every instrument.

### 2.6 Verification (`p3_make_bank.py --verify`, 17 checks)

SOURCE sha; strata and ids; frame and verbatim rule text; gold == env.step == rule == oracle; every row is its base with exactly its stratum's factor changed (re-derived from the row and its twin; negative-tested by tampering); 100 bases in all 15 strata; kind quota in every stratum; chains; payload uniqueness (F_keyorder == twin); 0 overlaps with every pinned source; source sizes; reserved files skipped by name; 0 overlaps with multi-event rollouts; census reproduces; OOR properties; rule consistency; SOURCE identity.

## 3. Subjects and instrument

| arm | what | role |
|---|---|---|
| OFF | Titan-Sparse (`…Titan-Sparse-Experimental-1.0.0-devfreeze`, fingerprint 818be9c6b6a3), all adapters disabled (`disable_adapter()`) | the comparator of every ON − OFF contrast |
| s0 | + T2 seed 0, `runs/t2/t2_s0/adapter` (sha 92bc7c74…, PREREG-2's registered artifact) | primary seed |
| s1, s2 | + T2 seeds 1, 2 (same recipe; hashed at the freeze and at load) | replication; RULE and TRIGGER require all three seeds |
| Supa | Supa 1.7 (fingerprint a9952c55f5a1), the model before Titan's delta | descriptive: base behaviour on the status words, read through the paired status-sensitivity rate (raw no-op accuracy cannot separate status semantics from copy bias) |
| COPY | answer = before (no model) | floor: 1.00 on every no-op stratum, 0.00 on every apply stratum |

**Process and loading:**
- The three adapters are loaded **unmerged** into one PeftModel on frozen Titan-Sparse (D-28: never merged). Sides are switched with `set_adapter` and disabled with `disable_adapter()`.
- All four sides of a stage are generated in one process and written together.
- Supa is loaded after Titan-Sparse is unloaded and VRAM is confirmed empty.

**Runtime (as PREREG-2):**
- FP32, TF32 off;
- torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0, **checked by the job after each model load and before anything is generated** (a drift stops the job, persisted);
- greedy decoding, `enable_thinking=False`, EOS [151645, 151643];
- 160-token cap; `generation.generate_texts` batching (budget 8,000 tokens, ≤ 32 per batch).

**Parsing:**
- A–D rows: `env.parse_state(…, lenient=True)`, PREREG-1/2's parser, unchanged.
- Other account sets: `p3_common.parse_general`. It is the same procedure with the row's account set, and is asserted equal to `env.parse_state` on every A–D text of the scorer self-test.
- An answer is correct iff the parsed balances equal the gold (order-insensitive). Unparseable means wrong.
- **Error classes:** no-op rows: applied_as_posted (the answer equals the posted twin's gold) / other_change / unparsed; apply rows: copied (answer == before) / wrong_value / unparsed.

**Chains:**
- chain_one self-rollout: each step's `before` is the model's own previous parsed state, which breaks to None at the first unparseable answer.
- Built by `p3_run.p3_chain`, which is `rcwmt.generation.chain`'s loop with the recorded key orders. On sorted-key A–D chains it gives the same texts and states as `generation.chain` (tiny test).

**Anchor A0 (instrument identity):**
- Before any bank output, the job regenerates the 208 pilot TEST ledger chain_one chains on all four sides.
- Each side must reproduce its T2 eval file with identical states on ≥ 99% of chains (identical texts are recorded as a diagnostic):
  - OFF: `t2_s0/gen_chain_one_base.jsonl`, pinned by PREREG-2;
  - sN: `t2_sN/gen_chain_one.jsonl`.
- This also shows that switching adapters inside one PeftModel equals loading each alone, as T2's eval did. The tiny test checks the same property, texts included. PREREG-2's anchor reproduced 100%.
- A resumed turn that runs S1/S2 re-anchors first. Any failure is persisted and final.

## 4. Endpoints

- **One-step accuracy** per (arm, stratum) and per (arm, cell), where a cell is a stratum (or, for magnitude, a stratum × kind class: `at` = add + transfer, 67 rows; `sc` = scale, 33 rows; the registered scale cell pools M_bal and M_amt, 66 rows): k/n with Wilson 95% interval (score interval, z 1.959964).
- **The bar:** accuracy ≥ 0.80 AND Wilson lower bound ≥ 0.70 (min correct: 80/100, 55/67, 54/66).
- **ON − OFF per cell:** exact two-sided McNemar on the paired rows: p = min(1, 2·P(X ≤ min(b, c))), X ~ Bin(b + c, ½).
- **Twin guard** per seed: T_posted accuracy ≥ 0.80 AND T_posted copied rate ≤ 0.10.
- **Descriptive, one-step:**
  - pairs (an unseen-status row and its twin both correct) and **status-sensitive pairs** (the unseen row unchanged while its twin was changed, i.e. the twin's answer is correct or a wrong value, not a copy);
  - in distribution vs out of range within each arm (paired by base, exact McNemar): each magnitude / naming / format single-stratum cell against its bases' twins;
  - **retained gain** per group and arm: per base, (ON − OFF) on the group's rows minus (ON − OFF) on the base's twin, base bootstrap (10,000 resamples, seed 17);
  - copied rate over the magnitude / naming / format rows (novelty fallback, §5);
  - error classes; sub-variants (status words, spellings, look-alikes, naming, number of accounts, amount band, key-order variant); stratum × kind.
- **Chains** (descriptive), per chain stratum and arm:
  - exact state at k = 1, 2, 4 (per-parent means) — the primary chain measure for CH_X;
  - normalised regret at k = 1, 2, 4, through PREREG-2's own path (`confirm2_analysis.gen_regret`, asserted equal to `rcwmt.decisions.g_only`): 256 goals with seed 1001, ties with seed 17, unparseable → parent-mean utility. **Not comparable to PREREG-2's numbers:** K = 4 candidates per parent here (PREREG-2: 8), and phi clips balances at ±300 (`data.phi_from_state`), so regret is partly blind to large-magnitude errors (13 of 40 CH_X chains end beyond ±300);
  - floors (F_copy_s0, F_first_only, F_last_only, F_ignore_status, F_touched) and the copy-chain exact floor;
  - step accuracy on correct-input steps by step class;
  - ON − OFF exact@4 and regret@4 by paired parent bootstrap (10,000 resamples, seed 17); with 10 parents per stratum the intervals are indicative only.

## 5. Decision rules (registered verbatim; applied mechanically by `scripts/p3_analysis.py`)

**Precondition.** The scorer self-test passes before any model output is read:
1. 100% correct on 10,500 gold renderings: compact, spaced, the row's key order, trailing period, pretty-printed, `{"before","after"}`-wrapped;
2. the copy answer scores exactly the copy floor: 1 on every no-op row, 0 on every apply row;
3. 0% on 12,000 negatives (value ±1; a key missing, extra, a string or a bool; empty answer; prose; the applied answer on a no-op row; the copied answer on an apply row), and the error classes are right;
4. `parse_general == env.parse_state` on every A–D text of the corpus (27,300 texts: the renderings, the negatives, arithmetic-annotated, truncated and malformed texts);
5. oracle chains give exact 1.0 and regret 0, and copy chains equal the copy-chain floor and F_copy_s0.

The registered analysis dies if the self-test fails. It passed on the sealed bank (`docs/prereg3/SCORER_SELFTEST.json`).

**V1 status (the primary question).** Unseen-status strata: U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_voidcase, N_postedlike.

> **RULE** iff, for EACH of s0, s1, s2: every unseen-status stratum (U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_voidcase, N_postedlike) has accuracy ≥ 0.80 AND Wilson 95% lower bound ≥ 0.70, and the twin guard holds: T_posted accuracy ≥ 0.80 AND T_posted copied rate ≤ 0.10.
> **TRIGGER** iff, for EACH of s0, s1, s2: every unseen-status stratum without a form of "void" (U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_postedlike) has accuracy ≤ 0.30, C_void has accuracy ≥ 0.80, and more than half of the wrong answers on those seven strata are the event applied as if posted (error class applied_as_posted).
> **PARTIAL** otherwise, reported with the per-stratum, per-seed table and the named patterns below.

*PARTIAL patterns* (wording only; several may apply):
- **twins:** every unseen-status cell met the bar in every seed, but the twin guard failed in at least one seed. The cause is named per seed from T_posted's error classes: "posted events left unchanged" if copied ≥ wrong_value, else "arithmetic errors on posted events".
- **semantic:** in every seed the five not-applied words met the bar, but U_neutral did not in at least one seed.
- **posted_substring:** in every seed all six U strata met the bar, but N_postedlike did not in at least one seed.
- **trigger_not_interpretable:** every TRIGGER accuracy condition held in every seed, but C_void < 0.80 or applied_as_posted ≤ half of the errors in some seed.
- **no_adapter_gain:** no unseen-status cell has a Holm-significant ON > OFF gain.

*Attribution* (wording only; RULE only):
- Exact McNemar ON(seed) vs OFF on the 24 unseen-status cells; Holm over these 24 at α = 0.05. all_sig = all 24 Holm-significant with ON > OFF.
- base_meets = OFF meets the bar on every unseen-status stratum AND OFF's T_posted copied rate ≤ 0.10 (so a high OFF no-op rate is not copy bias; OFF's twin *accuracy* is not required, because OFF's in-distribution posted arithmetic is weak, 0.30–0.57 in PREREG-2, which is not about status).
- Precedence: base_meets and all_sig → "present in the base at the bar, strengthened by the adapter"; base_meets only → "already present without the adapter"; all_sig only → "learned by the adapter"; neither → "attribution mixed" (the wording names the non-significant cells with OFF and ON accuracy).
- Holm is conservative for the conjunction "all 24 significant" (an intersection-union test needs no adjustment); it is kept so that each per-cell statement is familywise-controlled.

*Qualifiers* (wording only; never change a verdict):
- **Sub-variants:** in an adapted seed, any sub-variant (a U_neutral word, n 20; a N_voidcase spelling or N_postedlike look-alike, n 33/34) at or below 0.30 inside a stratum above 0.30 is named.
- **Novelty fallback:** if in some seed the adapted model's copied rate over the 500 magnitude / naming / format rows exceeds 0.20, the wording says that its no-op on unseen statuses may partly be a fallback to copying on unfamiliar input, and a RULE verdict then does **not** narrow PREREG-2's limit.
- **Instrument expectations** (§5b) that fail.
- **Base behaviour** (always, descriptive): OFF and Supa raw no-op accuracy on the unseen statuses and their status-sensitive pair rates, next to the adapted seeds'.

**V2 magnitude (add/transfer) {M_bal|at, M_amt|at}, V2s magnitude (scale) {M_bal+M_amt|sc}, V3 naming {A_renamed, A_many}, V4 format {F_keyorder}.** Per cell (cell × seed, 18 cells): exact McNemar ON vs OFF, Holm over the 18 at α = 0.05. Floor: the bar.

Cell classes:

| class | definition |
|---|---|
| HARMED | Holm-significant, OFF > ON |
| HIGH_GAIN | floor met, and Holm-significant ON > OFF |
| HIGH_NOGAIN | floor met, no significant gain |
| GAIN_BELOW_FLOOR | significant gain, floor not met |
| LOW_NOGAIN | neither |

Group verdict, in this order:

| verdict | condition |
|---|---|
| HARMED | any cell is HARMED |
| GENERALISES | every cell is HIGH_GAIN |
| ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN | every cell is HIGH_GAIN or HIGH_NOGAIN, at least one HIGH_NOGAIN |
| DOES NOT GENERALISE | no cell meets the floor (", with a partial gain over OFF" if any cell is GAIN_BELOW_FLOOR) |
| MIXED | otherwise |

Group wording adds, mechanically: the same bases' in-distribution accuracy (T_posted of the matching kind class) and the retained gain with its interval; for DOES NOT GENERALISE with OFF also below the floor on every cell, "P3 cannot separate an in-range adapter shortcut from a capacity limit shared with the base"; for DOES NOT GENERALISE with OFF meeting the floor on a cell where no adapted seed does, that fact; for HARMED, "the signature of an in-range shortcut (BWM-001 F-01/F-06)"; for MIXED, the cells below the floor.

**Why these thresholds and choices** (fixed before any output):

- **The bar, 0.80 / Wilson 0.70,** is the roadmap's. The adapter's in-distribution reference on the fresh PREREG-2 bank: void steps 1.00; posted steps 0.85–0.97 by kind. A model that has the rule should sit near those levels; 0.80 leaves about 0.05–0.15 of room for unfamiliar strings and numbers, while excluding behaviour that is merely better than OFF. At n = 100, accuracy ≥ 0.80 implies Wilson LB ≥ 0.71; the LB clause binds at 67 and 66 rows (55/67, 54/66) and for sub-variants.
- **All eight unseen-status strata in RULE.** The roadmap's RULE is "every unseen-status stratum ≥ 0.80"; all eight strata are unseen statuses. U_neutral separates the rule from a semantic "not applied" cue, N_postedlike from a "contains posted" cue, and N_voidcase from an exact-token trigger that happens to cover unseen words. Without them, RULE would be indistinguishable from those cues.
- **The twin guard.** Twins ≥ 0.80 is the roadmap's; the copied rate ≤ 0.10 targets the guard's purpose directly (to rule out a model that leaves everything unchanged; COPY scores 1.00 on every unseen stratum). With twins ≥ 0.80, the copy cap binds only when at least half of the twin errors are copies. In-distribution twin failures caused by arithmetic are reported as such.
- **TRIGGER 0.30, plus C_void ≥ 0.80 and errors that are applications.** OFF's in-distribution void accuracy was 0.04–0.13, so ≤ 0.30 means "base-like for this status". TRIGGER asserts a mechanism ("void is the no-op cue"), so it also requires that void itself is a no-op and that the misses on the other statuses are applications of the event, not unparseable or other answers. N_voidcase is not a TRIGGER condition, since a void cue may or may not cover case and inflection; it decides the TRIGGER wording.
- **All three seeds.** The question is about the recipe, not one artifact. Seeds 1 and 2 already exist, so requiring them costs no training.
- **Holm across 24 (status) and 18 (groups).** This keeps the attribution and gain statements familywise at 0.05 within each family.
- **Magnitude split by kind class.** Out-of-range scale rows test multi-digit multiplication (products up to 796,275), which a 0.6B model may lack with or without the adapter; pooling them with add/transfer would make "DOES NOT GENERALISE" close to predetermined. The scale cell pools M_bal and M_amt (66 rows per seed): six 33-row cells would all meet the floor with probability only about 0.21 at a true 0.90.
- **Novelty threshold 0.20.** It is twice the twin copy cap; in distribution the adapted model's posted errors were 0.03–0.15 in total (PREREG-2).

**Power** (exact binomial / multinomial, n = 100 per stratum, **under independence**, so the joint probabilities are lower bounds — the strata share the 100 bases and errors are positively associated; `docs/prereg3/P3_DRYRUN.md`):

| true accuracy per unseen cell | P(cell meets the bar) | P(all 24 unseen cells) | P(RULE), twins at 0.90 with copied 0.03 |
|---|---|---|---|
| 0.80 | 0.56 | ≈ 0 | ≈ 0 |
| 0.85 | 0.93 | 0.19 | 0.19 |
| 0.90 | 0.999 | 0.98 | 0.98 |
| 0.95 | 1.00 | 1.00 | 0.997 |

| true accuracy per TRIGGER cell | P(cell ≤ 0.30) | P(all 21 TRIGGER accuracy cells) |
|---|---|---|
| 0.20 | 0.99 | 0.88 |
| 0.25 | 0.90 | 0.10 |

| group (cells) | P(all cells meet the floor) at a true 0.85 / 0.90 / 0.95 |
|---|---|
| magnitude add/transfer (6 × 67 rows) | 0.27 / 0.92 / 1.00 |
| magnitude scale (3 × 66 rows) | 0.55 / 0.96 / 1.00 |
| naming (6 × 100) | 0.66 / 0.995 / 1.00 |
| format (3 × 100) | 0.81 / 0.998 / 1.00 |

- RULE is a demanding intersection. A true rule-follower at 0.85 is called PARTIAL about 80% of the time; the wording then reports the numbers, and the per-stratum table carries the information.
- McNemar gain detection at worst-case Holm (simulated, 20,000 runs, independent errors): ON 0.90 vs OFF 0.60: 0.96 (status, m 24), 0.97 (groups, m 18, 100 rows), 0.82 (67 rows), 0.81 (66 rows); ON 0.85 vs OFF 0.70: 0.23 (m 24); ON 0.95 vs OFF 0.80: 0.48 (m 24).
- So "ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN" can mean a gain too small to detect, not an absent one. The wording says "without a *shown* gain".

## 5b. Instrument expectations (pre-declared; qualifiers, not claims)

- **E1:** C_void ON ≥ 0.95 in every seed. PREREG-2 had void steps at 1.00. A failure means the instrument or the adapter's in-distribution void behaviour differs from PREREG-2.
- **E2:** C_void OFF ≤ 0.30. PREREG-2 had OFF void at 0.04–0.13.

Each is reported as holds / does not hold, and a failure adds a qualifier sentence.

## 6. Secondary and descriptive analyses (no decision depends on them)

- **Per stratum and arm:** error classes; pairs and status-sensitive pairs.
- **Breakdowns:** approved / active / ok / dax / wug; Void / VOID / voided; unposted / not_posted / 'not posted'; letters vs words; number of accounts; amount band; key-order variant (top-level only vs all orders); stratum × kind.
- **In distribution vs out of range** (paired by base) per arm and single-stratum cell; **retained gain** per group and seed.
- **Supa vs OFF:** McNemar on every stratum (unadjusted).
- **Chains** (§4): CH_S and CH_X exact and regret at k = 1/2/4 for every arm, floors, ON − OFF bootstrap per seed, step accuracy by class. The chains show whether the one-step behaviour composes under self-rollout. They are not powered for a claim.

## 7. Procedure and integrity

1. **Pre-freeze records** (exist now; rewritten only before the freeze):
   - `docs/prereg3/SCORER_SELFTEST.json`;
   - `docs/prereg3/P3_DRYRUN.{json,md}`: statistics unit tests (exact McNemar and Holm, incl. the Holm boundary); a 41-case wording self-test through the real summary, McNemar, Holm and bootstrap code on synthetic per-row classes; 8 synthetic scenarios through the full pipeline, including input validation; tamper refusals; power;
   - `docs/prereg3/TINY_TEST.json`: tiny random Qwen3 runner tests, CPU only;
   - `docs/prereg3/WIDE_SCAN.json`: the wide freshness scan (§2.5);
   - `docs/prereg3/GEN_PLAN.json`;
   - `docs/prereg3/queue_hook_p3.sh.draft`.
2. **Freeze:** `python -B scripts/p3_run.py --freeze --approval "<who approved, when, DECISIONS entry>"` (CPU). It refuses if:
   - PREREG-3 is already frozen, or a freeze was interrupted (`.partial` files or a lone `PREREG_3.md`; the message names the recovery step);
   - **any PREREG-3 output exists** (`runs/bank_p3_oor*`, any `PREREG_3_RESULTS*` under runs/);
   - the bank does not verify, including a fresh name-skipping harvest, the multi-event rollouts and the census;
   - the scorer self-test fails;
   - the dry-run, tiny-test or wide-scan record was not produced by the current code and bank (sha checks);
   - an s0 file differs from PREREG-2's pins.

   Otherwise it hashes every input: the draft, all p3 scripts, imported confirm2/t2/l1 code, `rcwmt/*.py`, the bank and SOURCE, the pilot bank, the three adapters, the four anchor references, and the docs/prereg3 records. It then writes `PREREG_3.md` (this draft with the status line and §10 filled; the draft is kept) and `PREREG_3_FREEZE.json`, each first as `.partial` and then renamed into place, create-only. **Recovery from an interrupted freeze:** if `PREREG_3.md` exists and `PREREG_3_FREEZE.json.partial` records its sha256, rename the `.partial` into place and run `--verify`; otherwise move the `.partial` files (and a lone `PREREG_3.md`) aside and rerun `--freeze`. Either way, log it in DECISIONS.md.
3. **Verification:** `p3_run.py --verify`. The GPU job and the analysis call the same verifier, and neither runs unless it passes. It checks:
   - `PREREG_3.md`'s hash;
   - every required input against the freeze, with the `PREREG_3_AMENDMENT_*.json` chain applied (old → new hash);
   - the s0 pins;
   - the registered runtime;
   - the approval.
4. **GPU job:** `python -B scripts/p3_run.py`, one shared lease `rcwm-t1:prereg3`, under every lease, alternation, thermal and RAM rule (`rcwmt.runtime.gpu_lease`).
   - **Before the lease** (exit 3, not persisted): the freeze must verify ("not frozen" is never persisted, so it cannot block a later freeze); an installed `scripts/queue_hooks/60_prereg3.sh` must be the frozen hook draft byte for byte; and **no other RCWM job may hold or wait for the GPU lease** (one-waiter rule, R14: a live `rineya-gpu-0.waiting.rcwm.json` of another process, or a lock whose label starts with `rcwm-t1:` held by a live process; dead or reused pids do not count; the peer's BWM jobs are waited for in the lease as usual).
   - **After each model load, before generating:** the fingerprint, and torch / transformers / peft equal to the registered runtime (a drift stops the job, persisted).
   - **Stages** (create-only; all sides of a stage in one process, written together):
     - A0 anchor (4 sides);
     - S1, the 1,500 one-step rows × 4 sides;
     - S2, the 80 chains × 4 sides;
     - unload, VRAM check, pause-flag and resource wait;
     - S3, Supa on rows and chains.
   - **Turn cap** (D-22):
     - 4.5 h from lock acquisition;
     - no stage starts within 0.5 h of the cap;
     - a stage starts only if 1.25 × its estimate fits, plus a Supa reserve while S3 is outstanding;
     - if nothing fits, the model is not loaded.
   - **Completion:** a stage is done only if its files exist and provenance records them with matching sha256. `DONE.json` lists every output hash. Provenance and the job log record only the *planned* generation count for S2 and S3 (the actual count would reveal how many chains broke before the analysis).
   - **A one-runner lock** (`runs/bank_p3_oor/.run.lock`): a live lock refuses; a stale one is reclaimed.
   - **Exit codes:**

     | code | meaning |
     |---|---|
     | 0 | done |
     | 3 | deterministic stop: never retried; persisted as `STOPPED.json` when it concerns the run (not for "not frozen", "hook not the frozen draft", "another RCWM job holds or waits for the lease", "run lock held") |
     | 5 | deferred by the cap; resumes |
     | 1 | crash; the hook retries once |
   - **Projection** (`--plan`; `docs/prereg3/GEN_PLAN.json`): at most 12,428 generations, which is one turn either way:

     | stage | generations |
     |---|---|
     | A0 | 3,328 |
     | S1 | 6,000 |
     | S2 | 1,280 |
     | S3 | 1,820 |

     - **1.35 GPU-h** at the registered planning rate (2.7 generations/s);
     - **0.98 GPU-h** at PREREG-2's observed ledger chain_one rate on its sealed bank (3.8 /s).
5. **Queue.** The hook draft `docs/prereg3/queue_hook_p3.sh.draft` is installed byte for byte as `scripts/queue_hooks/60_prereg3.sh` only after the freeze. `master_chain.sh` expands its hook glob **once, when execution enters the hook loop**. The chain now running (master_chain.sh pid 28516 under confirm_chain.sh pid 40892, launched 2026-10-08 11:03) entered that loop at 18:38:31, when only `50_l1_screen.sh` existed; it will never source a hook installed later. After L1 it runs the remaining pilot GPU steps and exits, and nothing relaunches it. So hook 60 runs only in a **fresh chain**: `scripts/launch_master.cmd`, started hidden via WMI, after `runs/bank_pilot/master_log.txt` shows "master chain finished" **and** no `master_chain.sh` or `confirm_chain.sh` process is alive. In the fresh chain every earlier step whose output exists is skipped (if L1 has not finished, hook 50 runs it first). Never start a second chain while one is alive. Relaunching before "master chain finished" means stopping the live chain at a safe point first (as R14 did). The relaunch is done by the RCWM-T1 operator (the session running RCWM-T1) on the user's go-ahead and logged in DECISIONS.md. The hook:
   - is idempotent;
   - adds no second GPU waiter (and p3_run.py refuses to be one);
   - makes one retry after a crash;
   - never retries after exit 3 or `STOPPED.json`;
   - runs the analysis once, after `DONE.json`.
6. **No peeking.** Between the first bank output and the registered analysis, no one reads `runs/bank_p3_oor/onestep/*`, `chains/*` or `supa/*`, or computes anything from them.
   - Only `provenance.json`, `anchor/*` and the job log may be read, as instrument checks.
   - No analysis code changes after the first output, except a logged non-statistical crash fix (an amendment).
7. **Analysis:** `python -B scripts/p3_analysis.py`, once, CPU, 2 threads, numpy only. It runs in this order:
   - verify the freeze;
   - **the scorer self-test, before any output is read**;
   - load and validate every input. It dies on any failure:
     - DONE/provenance hashes;
     - every stage recorded, with all sides in one process;
     - the anchor passed, and every resumed turn re-anchored;
     - per-turn code hashes against the freeze and amendments;
     - runtime versions and peft;
     - adapter hashes at load;
     - fingerprints;
     - ids in bank order;
     - every chain state is the parse of its text.

   It writes `runs/bank_p3_oor/analysis/PREREG_3_RESULTS.{json,md}` and `…_PER_ROW.npz` as one create-only set.
8. **Report** exactly as computed: a REPORT section and a RESULTS file rendered from the JSON.

**Deviation policy** (as PREREG-2):
- **Analysis crash:** before the results set is written, an identical rerun is allowed. A non-statistical crash fix is allowed only as `PREREG_3_AMENDMENT_*.json` (file, old and new sha, diff, reason), logged in DECISIONS.md.
- **GPU crash or reset:**
  - a stage that is not done is moved to `runs/bank_p3_oor/_orphans/` (never deleted) and regenerated in full, all sides together;
  - no stage is stitched across processes;
  - a recorded output whose hash changes stops the job.
- **Anchor failure (first or re-check), runtime drift:** the job stops, and the failure is persisted (`anchor.json` pass false or `RECHECK_FAILED_*.json`, plus `STOPPED.json`). The cause is investigated and logged:
  - if it is a software or environment change, restore the registered environment, move the stop file aside with a DECISIONS entry, and rerun;
  - any other remedy is an amendment recorded before any affected bank output exists.
- **Supa stage fails twice:** an amendment may let the analysis run with Supa "not evaluable". No verdict depends on Supa.
- **Operational amendments** (batching, pauses, lease handling) are recorded before any affected output exists. The verifier applies the chain.

## 8. Wording (rendered mechanically by `p3_analysis.wording()`; every branch exercised by the 41-case self-test)

| outcome | headline | sentence (a–b are ranges over seeds and strata; values rendered from the results) |
|---|---|---|
| RULE | "STATUS: RULE (learned by the adapter)" / "(present in the base at the bar, strengthened by the adapter)" / "(already present without the adapter)" / "(attribution mixed)" | "In all three T2 seeds, Titan-Sparse with the adapter left unseen-status events unchanged on a–b of rows in every seed and stratum (each >= 0.80 with Wilson lower bound >= 0.70): five words that mean 'not applied' (pending, draft, cancelled, reversed, held), neutral and nonce words without that meaning (approved, active, ok, dax, wug), near-miss spellings of 'void' (Void, VOID, voided) and look-alikes of 'posted' (unposted, not_posted, 'not posted'), as the frame's rule states. It applied the same events when posted (twin accuracy …; twins copied …)." Then by attribution: learned: "Without the adapter the same rows scored c–d; ON > OFF in all 24 stratum x seed McNemar tests (Holm)."; strengthened: "Without the adapter Titan-Sparse already met the bar on every unseen status (c–d; its posted twins copied x), and ON > OFF was still Holm-significant in all 24 tests: the rule is present in the base and strengthened by the adapter."; already present: "… already met the bar on every unseen status (c–d; its posted twins copied x): the adapter preserved a rule the base already follows; P3 does not show that the adapter learned it."; mixed: "Without the adapter the same rows scored c–d; ON > OFF is not Holm-significant in k of 24 tests (cells with OFF and ON accuracy), and the base does not meet the bar on every unseen status with uncopied twins." |
| TRIGGER | "STATUS: TRIGGER" | "In all three T2 seeds the adapted model's no-op did not extend beyond 'void': accuracy on the seven unseen-status strata that contain no form of 'void' (…) was a–b (each <= 0.30), against C_void … (each >= 0.80); x of the y errors on those rows applied the event as if posted. Near-miss spellings of 'void' left unchanged: s0 … (unchanged / applied / mixed), …. The adapted model treats {'void' and its near-miss spellings (Void, VOID, voided) / the exact token 'void' (its near-miss spellings … were applied) / 'void' (its near-miss spellings … handled inconsistently)}, not 'any status other than posted', as the no-op cue." |
| PARTIAL | "STATUS: PARTIAL" | "Neither RULE nor TRIGGER: unseen-status accuracy ranged a–b across seeds (OFF c–d); cells below the bar (accuracy 0.80, Wilson LB 0.70) per seed: …" plus each applicable pattern: twins: "Every unseen-status cell met the bar; RULE was withheld only by the posted-twin guard (accuracy >= 0.80 and copied <= 0.10) in <seeds>: <seed>: {posted events left unchanged / arithmetic errors on posted events} (twin accuracy, copied, wrong value)."; semantic: "In every seed the five words that mean 'not applied' met the bar, but the neutral and nonce words (approved, active, ok, dax, wug) did not in <seeds (acc)>: consistent with a semantic 'not applied' cue, not the stated rule."; posted_substring: "Every unseen status word met the bar, but the look-alikes of 'posted' (unposted, not_posted, 'not posted') did not in <seeds (acc)>: consistent with a 'contains posted' cue."; trigger_not_interpretable: "Every unseen status without a form of 'void' was at or below 0.30 in every seed, but TRIGGER was withheld because {C_void was below 0.80 in … / fewer than half of the errors applied the event in …}: the low accuracy is not interpretable as a void trigger."; no_adapter_gain: "No unseen-status cell shows a Holm-significant adapter gain over OFF." |
| sub-variant qualifier | — | "Qualifier: sub-variants at or below 0.30 inside a stratum above 0.30 (adapted seeds): <stratum 'word' seed acc (n)>; …." |
| novelty qualifier | — | "Qualifier (novelty fallback): the adapted model also copied novel posted inputs: copied rate over the magnitude, naming and format rows s0 …, s1 …, s2 … (> 0.20 in <seeds>); its no-op on unseen statuses may partly be a fallback to copying on unfamiliar input rather than the stated rule." |
| E1/E2 failed | — | "Instrument expectation not met: E1 C_void ON >= 0.95 in every seed (…)." |
| always | — | "Base semantics (descriptive): unseen-status no-op accuracy OFF a–b, Supa 1.7 c–d; status-sensitive pairs (unseen row unchanged while its posted twin was changed) OFF …, Supa 1.7 …, adapted seeds …. Raw no-op accuracy cannot separate status semantics from copy bias; the paired rate can." |
| group lines | — | "<MAGNITUDE (add / transfer) / MAGNITUDE (scale: multi-digit multiplication) / NAMING / FORMAT>: <verdict> (cells: OFF, s0, s1, s2; the same bases in distribution, T_posted|<kind class>: …; retained gain [(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: s0 x [lo, hi]; …)." plus, mechanically: shared-limit sentence, OFF-meets sentence, HARMED shortcut sentence, or MIXED cell list (§5) |
| implication, RULE | — | "For PREREG-2: within the tested status vocabulary (16 non-posted strings in eight strata) the adapter's status handling follows the stated rule, not the token 'void'[ (present in the base already, strengthened by the adapter) / (but the base already did) / (attribution mixed)]; PREREG-2's limit 'fresh at payload level, not out-of-distribution' can be narrowed for the status rule only, as far as the magnitude / naming / format verdicts allow." With the novelty qualifier instead: "For PREREG-2: the adapted model meets the RULE bar, but it also copies novel posted inputs (novelty qualifier), so P3 cannot exclude a novelty fallback; PREREG-2's limit … stays as registered." |
| implication, TRIGGER | — | "For PREREG-2: the void gain is a token cue, not the stated rule; PREREG-2's limit 'fresh at payload level, not out-of-distribution' binds, and rule data must be surface-diversified before reuse." |
| implication, PARTIAL | — | twins: "For PREREG-2: status handling met the rule on every unseen status, but RULE was withheld by the posted-twin guard in <seeds>; PREREG-2's limit … stays as registered."; semantic: "For PREREG-2: the adapter's no-op generalises to status words that mean 'not applied' but not to neutral or nonce words, i.e. along word meaning rather than the stated rule; PREREG-2's limit … stays as registered, and rule data would need non-posted statuses without a not-applied meaning."; posted_substring: "… look-alikes of 'posted' are applied, consistent with a 'contains posted' cue rather than the stated rule; … rule data would need look-alike statuses."; no adapter gain: "For PREREG-2: no unseen-status cell shows an adapter gain over OFF; whatever no-op rate there is on unseen statuses is the base's, and PREREG-2's limit … stays as registered."; otherwise: "For PREREG-2: the adapter's status handling generalises only partly beyond 'void' (per-stratum table); PREREG-2's limit … stays as registered." |

**Limits that always accompany the verdicts** (rendered verbatim from `LIMITS`):
- Ledger only: Titan's one-step frame (1,500 rows: 100 bases × 15 strata) plus 80 four-step chain_one chains; T2 adapters s0 (PREREG-2's registered artifact), s1 and s2, unmerged on frozen Titan-Sparse.
- "Rule" is tested on 16 non-posted strings: five words that mean "not applied", five neutral or nonce words, three near-miss spellings of "void" and three look-alikes of "posted". It is not a claim about every possible status string.
- Case variants of "posted" ("Posted", "POSTED") and synonyms of it (booked, settled, ...) are not tested: the frame's rule text does not settle them.
- Out-of-range means outside T2's training rows (statuses posted/void, amounts −9..12, |before| ≤ 98, accounts A–D, sorted keys); Titan's own 384 training rows share these ranges; Supa's pretraining data are unknown. T_posted and C_void are fresh in-distribution controls, and CH_S events are in range.
- One factor at a time: every one-step stratum is the same 100 bases with one factor changed (status string, balances, amount, account names, number of accounts, key order), so each out-of-range effect is a paired, kind-matched contrast with the posted twins; magnitude is judged separately for add/transfer and for scale (multi-digit multiplication); the CH_X chains mix factors and are descriptive.
- No training and no self-distillation or untargeted-LoRA control: a verdict describes what the adapted model does, not why; PREREG-2's other limits (horizon ≤ 4, one frame, no non-LM comparator at matched labels) still apply.
- Thresholds are absolute (the bar: accuracy 0.80 with Wilson lower bound 0.70; twins copied ≤ 0.10; trigger 0.30) and were fixed before any output. Under independence a true accuracy of 0.85 passes one cell with probability about 0.93 but all 24 unseen-status cells with about 0.19; 0.90 passes all 24 with about 0.98 (power table in the dry run; the strata share bases, so the joint probabilities are lower bounds).
- Chains are descriptive: K = 4 candidates per parent (PREREG-2: 8), so normalised regret is not comparable to PREREG-2's 0.544 → 0.056; phi clips balances at ±300, so regret is partly blind to large-magnitude errors in CH_X (exact@k is the primary chain measure there); 10 parents per stratum, so bootstrap intervals are indicative.

## 9. Freeze condition

There is no seed gate: no training is involved, and seeds 1 and 2 already exist. PREREG-3 freezes only when all of the following hold:
1. the user gives an explicit go-ahead, quoted in `--approval` and logged in DECISIONS.md;
2. every §7.1 pre-freeze record was produced by the current code and bank;
3. no PREREG-3 output exists.

Nothing here selects among seeds: all three are required for RULE and TRIGGER.

## 10. Freeze record

(Filled in by `scripts/p3_run.py --freeze`.)

## Appendix A. Choices made in this draft, with reasons

- **One-step frame as the primary instrument.** T2 trained in exactly this frame. A one-step row isolates the status decision from compounding error, and every gold follows from one clause of the frame's text. Chains are the bridge to PREREG-2's decision claim, and are descriptive.
- **One factor at a time on shared bases.** All 15 strata are the same 100 (balance, event) bases with one factor changed, so strata differ only in the variable of interest, every row has a posted twin, and every out-of-range effect can be read against the same rows in distribution (in-vs-out McNemar, retained gain). The price is that strata are positively correlated, which makes the independence power figures conservative.
- **The status words.** pending and draft come from the FDS generator vocabulary (draft appears twice in an evaluation bank, never trained). cancelled, held and reversed complete the "not applied" group: cancelled is close to a synonym of void, held means not executed, and a reversed transaction's natural net effect is zero, which agrees with the gold. None of the five therefore sets the rule against world-knowledge semantics; U_neutral does (approved, active, ok, and the nonce words dax and wug). Synonyms and case variants of "posted" are excluded for lack of a gold.
- **Near-misses and look-alikes as RULE conditions.** They are unseen statuses, so the roadmap's "every unseen-status stratum" covers them; they also separate the rule from an exact-token trigger (N_voidcase) and from a "contains posted" cue (N_postedlike). A high N_voidcase alone does not discriminate (a case-insensitive void cue predicts the same no-op); a low one points to an exact-token trigger, which is how the TRIGGER wording uses it.
- **Kind quota 45/33/22.** This removes a chance kind mix from the comparisons. It was fixed on input composition only.
- **Magnitude by kind class; scale cell pooled.** See §5. The add/transfer verdict asks whether the trained operations extend to large numbers; the scale verdict asks about multi-digit multiplication, which may be a capacity limit of the base.
- **Absolute floors plus McNemar vs OFF** (spec item 3). A floor without a contrast cannot attribute a gain to the adapter. A contrast without a floor would call a 0.30 → 0.50 move "generalisation".
- **Supa.** It shows what the model before Titan's delta does with these words; because raw no-op accuracy mixes status semantics with copy bias (Supa's ledger one-step exact is 0.07–0.11), it is read through the paired status-sensitivity rate. The OFF arm is the adapter's own control.
- **No inverted-rule-text stratum.** It is not Titan's documented frame, and the task allows a prompt variant only if it is.
- **Freeze tool inside `p3_run.py`.** The runner and the analysis share one verifier, as in PREREG-2.

## Appendix B. Open questions for the user (defaults in force unless changed before the freeze)

- **Q1. RULE requires 24 unseen-status cells and 3 twin guards.** Under independence, a true 0.85 everywhere gives RULE about 19% of the time; 0.90 about 98%.
  - Default: keep (all eight unseen-status strata are needed to separate the rule from the semantic, substring and exact-token cues).
  - Alternative: pool the three seeds per stratum (n = 300).
- **Q2. Case variants and synonyms of "posted"** are excluded for lack of a defensible gold.
  - Default: exclude.
  - Alternative: add a descriptive "applied rate" stratum with no gold (+100 rows, about 3 GPU-minutes).
- **Q3. s1/s2 anchor references** are their own T2 eval files (`runs/t2/t2_sN/gen_chain_one.jsonl`). They are hashed at the freeze, never opened for planning. If those evals ran under a different environment, A0 fails and the job stops before any bank output.
  - Default: as written.
- **Q4. How PREREG-3 gets launched.** The running chain will not source hook 60 (§7.5). Both routes need the live chain to be gone first:
  - Default: install hook 60 after the freeze; relaunch a fresh chain with `launch_master.cmd` once `master_log.txt` shows "master chain finished" and no chain process is alive. After L1 the current chain still runs the remaining pilot GPU steps (lik, pvr, steer, the arms, regen), so this may be many GPU-hours away.
  - Alternative: the operator stops the current chain at a safe point after L1 (as R14 did), logs it in DECISIONS.md, and relaunches; or launches `p3_run.py` and then `p3_analysis.py` directly when no chain is alive.
- **Q5. The neutral words** (approved, active, ok, dax, wug).
  - Default: as written. "approved" is the strongest test (semantics suggest applying); if it is judged too close to "posted", the sub-variant breakdown shows it separately.
- **Q6. Additions to the roadmap's rule** (twin copy cap 0.10; TRIGGER requires C_void ≥ 0.80 and applied errors; novelty threshold 0.20; scale cell pooled).
  - Default: keep (reasons in §5).

## Appendix C. Changes from draft revision 1 (pre-freeze review, 2026-10-09)

- **Validity (must-fix):** added U_neutral (approved / active / ok / dax / wug) as a RULE condition; corrected Appendix A on cancelled / held / reversed. Also from validity and statistics: N_postedlike and N_voidcase are RULE conditions; TRIGGER now requires C_void ≥ 0.80 and applied-as-posted errors; attribution has a stated precedence, a "strengthened" label, and a copy check on OFF; a novelty-fallback qualifier; base behaviour is read through status-sensitive pairs; the "every event" wording is gone; N_voidcase is labelled by behaviour.
- **Statistics (must-fix):** the PARTIAL wording branches on its cause (twins: copying vs arithmetic, read from T_posted's error classes; semantic; substring; trigger not interpretable; no adapter gain), each with its own implication. Also: twin copy cap; magnitude split by kind class with the scale cell pooled; retained-gain and in-vs-out contrasts on shared bases (all GEN strata now derived from the status bases, F_keyorder split into top-level-only and all-orders variants); V2–V4 power rows; power labelled "under independence"; McNemar simulations and the twins-at-0.90 row in the frozen power table; unit tests of the exact McNemar and Holm; sub-variant qualifier; chain caveats (K = 4, phi clip, 10 parents).
- **Engineering (must-fix):** the queue section, the hook header and Q4 now state that the running chain fixed its hook list at 18:38:31 and that hook 60 runs only in a fresh chain. Also: one-waiter guard, runtime check after each load, installed-hook binding, "not frozen" never persisted, interruption-safe freeze with recovery, identical_texts in the anchor, planned-only generation counts for S2/S3, status census corrected and extended to multi-event prompts (binding), wide freshness scan.
- **Bank:** rebuilt as revision 2 (§2.1); the revision-1 bank is kept unused.

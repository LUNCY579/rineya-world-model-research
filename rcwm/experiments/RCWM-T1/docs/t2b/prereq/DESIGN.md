# T2b prerequisite representation screen (RCWM roadmap item 2; joint roadmap 5a)

**Status:** kit built and tested on CPU only (2026-10-09). No GPU job has run; no `runs/t2bp/` output exists.
Nothing here is a preregistration: this is a seed-0 screen on a dev bank. The decision rule below was written
before any output existed (`SCREEN_RULE.md`, hashes in `SCREEN_RULE.json`).

## 1. Question and inputs

- T2's prerequisite fix learned a COPY shortcut (false copy 0.039 -> 0.624; applicable state-changing 0.587 -> 0.353).
- Titan's own prerequisite "rule" is integer addition: right on 383/415 applicable steps where `state + add` equals the
  bitwise answer, and on 2/333 where it does not (R11).
- Deferral cannot fix it (R15). Computed predicates are not learned from bare-answer rows (D-32 (i)).
- The screen asks whether a de-shortcut training set, and/or a different target representation, lets the T2 recipe
  learn the precondition rule and the bitwise update, including out of range.
- **Decision point 2 (roadmap):** if every arm scores < 0.20 on int-add-wrong, stop P training and move P to a
  code-side bit interface, reported as an interface, not as skill.
- **Decision point 1:** if PREREG-3 says TRIGGER, use the surface-diversity option (section 4.4).

## 2. Arms: the same 1,500 rows, rendered four ways

Every arm's system prompt keeps T2's rule sentences byte for byte. Only two things vary:
- the format sentence ("Return only ...");
- for B-bits, the state sentence adds the bit-string convention.

The D-int prompt is T2's prerequisite frame (`env.multi_step_messages` with one event). It matched T2's 500 stored
prompts byte for byte.

| Arm | Prompt state / masks | Exact target text (row example: before 9, inapplicable, pre_off bit 3 set) | Parser |
|---|---|---|---|
| D-int (data control) | integers | `9` | lenient, exactly `env.parse_state(P, ., lenient=True)` with range [0, 2^w) |
| S-trace (joint format) | integers | `{"cmp":{"on_miss":[],"off_hit":[3],"viol":1,"ok":false},"calc":9,"after":9}` | strict |
| B-bits | w-digit bit strings, MSB first | `{"applicable":false,"after":"00001001"}` | strict |
| F-filler (control, OFF by default) | integers | `{"fill":" . . . (20 units) .","after":9}` | strict |

**S-trace** (joint trace format, binding amendments 1–4 of BWM D-52).
- Keys appear in this literal order: cmp -> calc -> after.
- cmp lists the predicate's inputs and its result:
  - `on_miss`: ascending bit positions (bit 0 = least significant) of pre_on bits that are clear in the state;
  - `off_hit`: the positions of pre_off bits that are set;
  - `viol = len(on_miss) + len(off_hit)`, which equals `popcount((pre_on & ~before) | (pre_off & before))` because pre_on and pre_off are disjoint;
  - `ok = viol == 0`.
- `calc = (before & ~delete) | add` if ok, else before (the chosen branch's result).
- `after` = the integer state, P's existing format.
- Serialization uses an explicit key order and never sorts keys. The parser rejects:
  - any other key order;
  - extra or duplicate keys;
  - a non-int or out-of-range `after`.

  A malformed trace counts as a wrong `after`. `cmp_ok` (exact, including key order and types) and `calc_ok` are
  scored separately. The decision parser reads only `after`.

**B-bits.** The target text is `'{"applicable":' + ('true'|'false') + ',"after":"' + format(after, '0{w}b') + '"}'`.
- The flag comes first.
- The parser rejects:
  - another key order;
  - a non-bool flag;
  - a bit string of the wrong length or containing non-binary digits.
- `flag_ok` is scored separately.

**F-filler.** The format is BWM's shared one: `{"fill":" . . .","after":N}`. The fill is the one-token unit `" ."`
repeated so that the whole filler target has EXACTLY the S-trace target's token count under Titan's tokenizer. This
held on all 1,500 training rows and 950 dev rows, under both the HF tokenizer and the `tokenizers` library.

**Training.** All arms train on the same canonical rows in the same file order, so the vendored trainer's seeded row
order is identical across arms.

## 3. Training data (`data/t2bp_train_{dint,strace,bbits,ffill}.jsonl`, seed 20261111)

### 3.1 Row types and the heuristic algebra

Notation: b = before, a = add, d = delete. The env generator draws a single-bit add, and d is disjoint from a.

| Type | Definition | copy | apply-regardless | int b+a | OR b\|a | Count |
|---|---|---|---|---|---|---|
| A1 | applicable; a not in b; d clears nothing | 0 | 1 | 1 | 1 | 530 |
| A2 | applicable; a not in b; d clears a set bit | 0 | 1 | 0 | 0 | 40 |
| A3 | applicable; a already in b; d clears nothing (gold = b) | 1 | 1 | 0 | 1 | 15 |
| A4 | applicable; a already in b; d clears a set bit (int-add-wrong, strict) | 0 | 1 | 0 | 0 | 65 |
| Za | applicable; add = 0 (delete-only); d clears a set bit | 0 | 1 | 0 | 0 | 100 |
| Ib | inapplicable; a not in b | 1 | 0 | 0 | 0 | 525 |
| Ia | inapplicable; a in b; d clears a set bit | 1 | 0 | 0 | 1 | 20 |
| Zi | inapplicable; add = 0; d clears a set bit | 1 | 0 | 1 | 1 | 205 |

**Why add = 0 rows exist.** Integer b+a is right only on A1 and on add = 0 rows.
- So holding b+a at 0.50 together with apply-regardless at 0.50 forces
  #Zi − #Za = #A2 + #A3 + #A4 (within the 0.02 tolerance); the derivation is in the header comment of `t2bp_common.py`.
- Without add = 0 rows, every delete or carry row would push int b+a below 0.48.
- The price is a cue: among add = 0 rows, 67% are copies. The answer rule "add==0 -> copy, else apply-regardless"
  scores 0.57 on the training rows, and apply-regardless scores 0.544 on the add ≠ 0 rows (copy 0.469).
- The dev bank has no add = 0 rows, so this cue cannot inflate any dev score. It can only bias the model mildly toward
  applying on add ≠ 0 rows; the inapplicable strata measure that.

**"(before&~del)|add ignoring preconditions"** is the same predictor as apply-regardless under env semantics. The
audit checks that they are identical on every row and gates both.

**Two descriptive integer heuristics are not gated:**
- (b&~d)+a scores 0.447. It cannot reach 0.50 while carry rows exist (it is wrong on every carry row and right on no
  inapplicable row).
- b&~d scores 0.409.

### 3.2 Audit (`scripts/t2bp_audit.py`; refuses otherwise)

| Check | Value | Gate |
|---|---|---|
| copy-input | 0.5100 | 0.48–0.52 |
| apply-regardless (= ignoring preconditions) | 0.5000 | 0.48–0.52 |
| integer before+add | 0.4900 | 0.48–0.52 |
| OR before\|add | 0.5133 | 0.48–0.52 |
| minimal pairs | 260 | ≥ 250 |
| int-add-wrong, strict (A4) | 65 | ≥ 60 |
| applicable-delete (A2+A4+Za) | 205 | ≥ 200 |
| int-add-wrong, R11 definition (applicable, gold ≠ b+a) | 220 | ≥ 200 |
| inapplicable pre_on-missing / pre_off-present | 376 / 374 | ≥ 300 each; none both |

**Minimal pairs (260).** The two members share the same before and the same event, except for exactly one added bit in
pre_on (a clear bit) or in pre_off (a set bit). Applicability flips and the golds differ.

| Applicable / inapplicable | Pairs |
|---|---|
| A1/Ib | 160 |
| A2/Ib | 25 |
| A4/Ia | 20 |
| Za/Zi | 55 |

The flip kinds alternate between pre_on and pre_off.

**Cue ceilings (descriptive; majority lookup on the training rows, predicting applicability):**

| Feature | Accuracy |
|---|---|
| popcount(pre_on\|pre_off) | 0.709 |
| pre_off == 0 | 0.668 |
| pre_on == 0 | 0.632 |
| others | ≤ 0.578 |

Part of this is the rule itself: a row with no precondition bits is always applicable. The rest is the recorded
marginals plus the add-one-bit minimal pairs.

### 3.3 Freshness

The exclusion union holds 310,258 payloads and 302,283 operative contents from 17 sources:
- the historical records tree, using `l1_make_ood.harvest_historical` via `p3_make_bank.load_all_sources`. The 4
  `HSKILL_CONFIRM_V2*` files are skipped by name and never opened; `rcwmt.bank.load_exclusions()` is not called;
- the pilot, confirm, confirm2 and s4000 banks;
- the 4 T2 data files;
- SELECT-VIEWS-001;
- the L1 OOD bank;
- 5 FDS/composition T banks;
- the PREREG-3 bank.

Results:
- Training rows, dev one-step rows and chain transitions have 0 payload overlap and 0 operative-content overlap with
  every source and with each other.
- Gold = `env.step`, checked against the FDS oracle (`env.cross_check`) on every row and every chain transition.
- No model, model output or GPU was used; torch was not imported (peak RSS 364 MiB).

### 3.4 Surface diversity (decision point 1); OFF by default

`t2bp_make_data.py --surface-diversity on` writes `data/t2bp_train_sd_<arm>.jsonl` plus
`data/t2bp_train_sd.SOURCE.json`.
- These hold the SAME canonical rows, read from the canonical files and never re-drawn.
- Each row gets a seeded rule-text paraphrase (`PARAPHRASES[0..3]`) and a field-name variant (`FIELD_VARIANTS[0..3]`).
  Variant 0 is canonical, and the two members of a minimal pair share a variant.
- Targets are unchanged.
- Variant 4 of both is reserved for held-out evaluation.

`t2bp_run.py --data-variant sd` trains on these files, with outputs under `runs/t2bp/sd_<tag>/`. Neither step needs a
code change. The tiny test renders all 16 combinations and checks that they pass the same audit.

## 4. Dev screen bank (`data/t2bp_dev.jsonl`; no model call)

| Stratum | n | Definition | Heuristic floors (copy / apply-regardless / b+a / OR / (b&~d)+a / b&~d) |
|---|---|---|---|
| D_app_add | 100 | A1 | 0 / 1 / 1 / 1 / 1 / 0 |
| D_app_unchanged | 100 | A3 (applicable, unchanged: carry) | 1 / 1 / 0 / 1 / 0 / 1 |
| D_app_delete | 100 | A2 | 0 / 1 / 0 / 0 / 1 / 0 |
| **D_int_add_wrong** | 150 | A4: bitwise ≠ b+a, ≠ (b&~d)+a, ≠ b | 0 / 1 / 0 / 0 / 0 / 1 |
| D_inapp_pre_on | 100 | only pre_on violated; applying would change the state | 1 / 0 / 0 / .09 / 0 / .75 |
| D_inapp_pre_off | 100 | only pre_off violated; applying would change the state | 1 / 0 / 0 / .11 / 0 / .72 |
| **D_ood10** | 150 | 10-bit masks; before ≥ 256; a bit ≥ 8 operative; 25 each of A1, A2, A3, A4, inapp-pre_on, inapp-pre_off | .5 / .667 / .167 / .40 / .333 / .54 |
| **D_ood12** | 150 | the same with 12 bits | .5 / .667 / .167 / .367 / .333 / .58 |

Notes on the strata:
- **OOD prompts.** These differ from the arm's 8-bit prompt only in the width word ("a 10-bit"). For B-bits, the bit
  strings are also 10 or 12 digits long.
- **Chains.** 30 prerequisite parents, each with 8 candidate plans of 4 events, drawn exactly as `rcwmt.bank.build`
  draws them (6-action library, at least 4 distinct finals per parent, all transitions fresh): 240 chains and 960
  steps. The copy floor at k=4 is 0.129.
- **Union cells reported in the analysis:**
  - applicable state-changing (the T2 definition);
  - app-del (the L1/T2 definition);
  - int-add-wrong (the R11 definition);
  - inapplicable;
  - out-of-range.
- **Structural overlap with the training rows (descriptive).** A bit-permutation-invariant signature overlaps on
  0.34–0.96 of each in-distribution stratum and 0 of the OOD strata. A dev bank built from the same generator is
  in-distribution by construction.

## 5. Runner (`scripts/t2bp_run.py`)

**Recipe.** This is T2 unchanged:
- the vendored trainer is called exactly as `run_t2.py` calls it;
- LoRA r16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down;
- frozen FP32 Titan-Sparse;
- WSD schedule, lr 1e-4, batch 16 (4 × 4 microbatches), 190 steps, seed 0;
- the trainer hashes are pinned.

**Evaluation.** The adapter is loaded unmerged with PEFT, as `run_t2.py` loads it. OFF (`disable_adapter()`) and ON run
in one process, with greedy FP32 decoding, `generation.generate_texts` and a 160-token cap.
- **E1:** the 950 one-step rows in the arm's own frame.
- **E2:** the 240 chains as a self-rollout in the arm's frame. The arm's after-only parser feeds each step's state into
  the next. For D-int this is `generation.chain(..., 'chain_m1')`, which the tests confirm.

**Anchor A0.** Once per turn, before the first evaluation stage:
- with the adapter disabled, run pilot TEST prerequisite chain_m1 (224 chains);
- the result must reproduce `runs/t2/t2_s0/gen_chain_m1_base.jsonl` (sha 7ebc8288…, only hashed here) on at least
  99% of chains;
- otherwise the job stops, and the stop is persisted.

**One lease, `rcwm-t1:t2b-prereq`.** The D-22 turn rules apply:
- 4.5 h cap from acquisition;
- an arm's training starts only if 1.25 × (its train + eval + load [+ anchor]) fits;
- each eval stage starts only if 1.25 × its estimate fits;
- no stage starts within 0.5 h of the cap;
- otherwise the arm is deferred to the next turn (exit 5). The next call resumes from the recorded stages and
  re-anchors.

**Checks before the lease:**
- data hashes equal the SOURCE records;
- every training prompt equals `tok.apply_chat_template(messages(row))`;
- the one-waiter rule (R14);
- a run lock.

**Outputs and exit codes:**
- Outputs are create-only, with stage records hashed in `provenance.json`.
- A partial stage is moved to `_orphans/`, never deleted, and redone.
- Exit codes: 0 done, 3 deterministic stop, 5 deferred, 1 crash.

**GPU estimate** (`GEN_PLAN.json`; no model loaded). Training time is T2's measured 190-step time scaled by tokens per
row. The generation time model is fitted to PREREG-2's measured P chain_m1 and ledger chain_one rates; planning
applies a factor of 3.8/2.7.

| Arm | Train | E1 | E2 | Pair (planning) | Pair (observed rates) |
|---|---|---|---|---|---|
| D-int | 38.6 min | 6.9 min | 6.9 min | 0.91 h | 0.76 h |
| S-trace | 64.5 min | 14.7 min | 14.9 min | 1.60 h | 1.32 h |
| B-bits | 62.7 min | 11.1 min | 11.0 min | 1.45 h | 1.21 h |
| F-filler | 50.1 min | 14.7 min | 14.9 min | 1.36 h | 1.11 h |

- **Screen (3 arms):** one turn, 4.01 h at the planning rate (3.33 h at the observed rates), plus the 3.2 min anchor.
  The B-bits pair passes the 1.25× rule with only 0.12 h to spare; if it does not fit, it defers to a second turn.
- **With F-filler:** a second turn of 1.42 h (total 5.43 h). It stays OFF unless the budget allows.
- **Seeds 1–2 of the winner:** about 0.9–1.6 h per seed, one pair per turn.
- **Generations:** 3,820 per arm (1,900 one-step + 1,920 chain steps, OFF+ON), plus 896 for the anchor in each turn.

## 6. Analysis (`scripts/t2bp_analysis.py`)

The analysis runs on CPU only and never imports torch. The scorer self-test runs before any output is read:
- golds are 950/950 in each of the 4 formats;
- the 6 heuristics are scored at their exact floors in every format;
- 15 malformed answers are rejected;
- the D-int parser equals `env.parse_state` on a 21-string battery.

The run is validated before scoring: DONE.json, provenance hashes, ids in order, and chain states equal to the parser
applied to the texts. The verdict follows the pre-stated rule (`SCREEN_RULE.md`). Everything else is descriptive:
- OFF baselines;
- per-stratum and OOD-subtype accuracies with Wilson CIs;
- heuristic agreement per stratum;
- cmp_ok, calc_ok and flag_ok;
- k=1/2/4 exact;
- step accuracy from the model's own state;
- G_chain regret, using `decisions.g_only` with 256 goals, seed 1001, and floors;
- paired ON−OFF and arm−D-int contrasts.

A run that lacks one of the three screen arms (for example a seed-1/2 run) gets NO_VERDICT and a descriptive report.

## 7. Open questions for the operator

1. **The add = 0 balancing rows** (section 3.1) are the only way to meet the binding 0.50 ± 0.02 audit for integer
   b+a while also including delete and carry rows. They add an "add 0 -> mostly copy" cue to the training data, and a
   slight bias toward applying (0.544) on add ≠ 0 rows. The alternative is to relax the int b+a gate, or to drop most
   delete and carry rows. That alternative would leave 60 or fewer of them and defeat the point.
2. **IAW = A4 only (strict).** The stratum is also solved by "apply delete, ignore add" (b&~d floor 1.0), because the
   add bit is already set. A2 is excluded because Titan's integer-with-delete reading (b&~d)+a solves it. The R11-wide
   union (A2+A3+A4) is reported. Should decision point 2 use the union instead?
3. **Representation choices.** S-trace lists bit positions, while BWM's P instance lists `viol` and `ok` only. The
   B-bits order is MSB first. OOD prompts change only the width word. Reviewers may prefer integer masks in cmp.
4. **ID non-inferiority margin of 0.10 against D-int and the composite R = mean(K4, IAW, OOR) with equal weights.**
   Both were chosen here, not given in the task. The task said to "rank by k4 exact and int-add-wrong"; D-52 (5) makes
   the out-of-range strata primary, so OOR is included.
5. **F-filler** fits only in a second turn (+1.42 h planning). Run it only if S-trace wins (R5 flag)?
6. **Queue placement.** The hook draft `queue_hook_t2bp.sh.draft` is NOT installed, and it must follow PREREG-3 under
   the one-waiter rule. Installing it needs a fresh chain, which is an operator decision.

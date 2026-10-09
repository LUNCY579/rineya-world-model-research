# T2b prerequisite representation screen (RCWM roadmap item 2; joint roadmap 5a), revision 2

**Status:** kit built and tested on CPU only (2026-10-09). No GPU job has run; no `runs/t2bp/` output exists.
Nothing here is a preregistration: this is a seed-0 screen on a dev bank. Revision 2 applies the review of revision 1
(section 8). The decision rule (revision 2) was written before any output existed and is frozen in `SCREEN_RULE.md`;
`SCREEN_RULE.json` records the sha256 of every frozen input, and the runner and the analysis refuse when one differs.
Revision 1 of the rule is kept as `SCREEN_RULE_rev1.md` / `SCREEN_RULE_rev1.json`.

## 1. Question and inputs

- T2's prerequisite fix learned a COPY shortcut (false copy 0.039 -> 0.624; applicable state-changing 0.587 -> 0.353).
- Titan's own prerequisite "rule" is integer addition: right on 383/415 applicable steps where `state + add` equals the
  bitwise answer, and on 2/333 where it does not (R11).
- Deferral cannot fix it (R15). Computed predicates are not learned from bare-answer rows (D-32 (i)).
- The screen asks whether a de-shortcut training set, and/or a different target representation, lets the T2 recipe
  learn the precondition rule and the bitwise update, including out of range.
- **Decision point 2 (roadmap):** if every arm scores < 0.20 on int-add-wrong, stop P training and move P to a
  code-side bit interface, reported as an interface, not as skill. Revision 2 also requires IAW >= 0.20 of the arm
  that is named.
- **Decision point 1:** if PREREG-3 says TRIGGER, use the surface-diversity option (section 3.6).

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
| F-filler (control, conditional) | integers | `{"fill":" . . . (20 units) .","after":9}` | strict |

**S-trace** (joint trace format, binding amendments 1–4 of BWM D-52).
- Keys appear in this literal order: cmp -> calc -> after.
- cmp lists the predicate's inputs and its result: `on_miss` (ascending bit positions, bit 0 = least significant, of
  pre_on bits that are clear in the state), `off_hit` (positions of pre_off bits that are set),
  `viol = len(on_miss) + len(off_hit)` (= `popcount((pre_on & ~before) | (pre_off & before))`; pre_on and pre_off are
  disjoint), `ok = viol == 0`.
- `calc = (before & ~delete) | add` if ok, else before (the chosen branch's result). `after` = the integer state.
- Serialization uses an explicit key order and never sorts keys. The parser rejects any other key order, extra or
  duplicate keys, and a non-int or out-of-range `after`. A malformed trace counts as a wrong `after`. `cmp_ok` and
  `calc_ok` are scored separately; the decision parser reads only `after`.

**B-bits.** Target `'{"applicable":' + ('true'|'false') + ',"after":"' + format(after, '0{w}b') + '"}'`; the flag comes
first. The parser rejects another key order, a non-bool flag, and a bit string of the wrong length or with non-binary
digits. `flag_ok` is scored separately.

**F-filler.** BWM's shared format `{"fill":" . . .","after":N}`: the one-token unit `" ."` repeated so that the whole
filler target has EXACTLY the S-trace target's token count under Titan's tokenizer. Checked on all 1,500 training rows
and 1,050 dev rows, under both the HF tokenizer and the `tokenizers` library.

**Format vs value (new in revision 2).** `t2bp_common.format_ok` says whether an answer is in the arm's format whatever
its value (D-int: the lenient parser finds an integer; S-trace/F-filler: the literal keys and an integer `after`;
B-bits: the literal keys, a boolean flag and a binary string). An out-of-range state such as Titan's integer reading
200 + 64 = 264 is a wrong answer in the right format. The instrument-validity gate (WF) counts format, not value.

**Training.** All arms train on the same canonical rows in the same file order, so the vendored trainer's seeded row
order is identical across arms.

## 3. Training data (`data/t2bp_train_{dint,strace,bbits,ffill}.jsonl`, seed 20261111)

### 3.1 Row types and the heuristic algebra

Notation: b = before, a = add, d = delete. The env generator draws a single-bit add, and d is disjoint from a.

| Type | Definition | copy | apply-regardless | int b+a | OR b\|a | Rev. 1 | **Rev. 2** |
|---|---|---|---|---|---|---|---|
| A1 | applicable; a not in b; d clears nothing | 0 | 1 | 1 | 1 | 530 | **490** |
| A2 | applicable; a not in b; d clears a set bit | 0 | 1 | 0 | 0 | 40 | **40** |
| A3 | applicable; a already in b; d clears nothing (gold = b) | 1 | 1 | 0 | 1 | 15 | **15** |
| A4 | applicable; a already in b; d clears a set bit (int-add-wrong, strict) | 0 | 1 | 0 | 0 | 65 | **150** |
| Za | applicable; add = 0 (delete-only); d clears a set bit | 0 | 1 | 0 | 0 | 100 | **55** |
| Ib | inapplicable; a not in b | 1 | 0 | 0 | 0 | 525 | **480** |
| Ia | inapplicable; a in b; d clears a set bit | 1 | 0 | 0 | 1 | 20 | **35** |
| Zi | inapplicable; add = 0; d clears a set bit | 1 | 0 | 1 | 1 | 205 | **235** |

**Why add = 0 rows exist.** Integer b+a is right only on A1 and on Zi rows. Holding b+a and apply-regardless both at
0.50 forces #Zi − #Za = #A2 + #A3 + #A4 (within the 0.02 tolerance); OR − b+a = #A3 + #Ia caps A3 + Ia at about 60
(50 here). The derivation is in the header comment of `t2bp_common.py`.

**Revision 2 dose.** A4, the int-add-wrong skill, rose from 65 to 150 rows (Ia 20 -> 35 so the "add bit already set ->
apply" cue stays near its revision-1 level). The precondition-aware integer composites are now separated from the
bitwise rule on more rows (section 3.3). The price is a stronger add = 0 cue (section 3.4).

### 3.2 Audit (`scripts/t2bp_audit.py`; refuses otherwise)

| Check | Value | Gate |
|---|---|---|
| copy-input | 0.5100 | 0.48–0.52 |
| apply-regardless (= "(before&~del)\|add ignoring preconditions", identical on every row) | 0.5000 | 0.48–0.52 |
| integer before+add | 0.4833 | 0.48–0.52 |
| OR before\|add | 0.5167 | 0.48–0.52 |
| minimal pairs | 275 | ≥ 250 |
| int-add-wrong, strict (A4) | 150 | ≥ 150 |
| applicable-delete (A2+A4+Za) | 245 | ≥ 200 |
| int-add-wrong, R11 definition (applicable, gold ≠ b+a) | 260 | ≥ 200 |
| inapplicable pre_on-missing / pre_off-present | 377 / 373 | ≥ 300 each; none both |

Descriptive integer heuristics (not gated): (b&~d)+a 0.390; b&~d 0.4067. (b&~d)+a cannot reach 0.50 while carry rows
exist, so it is not a shortcut on these rows.

**Minimal pairs (275).** The two members share before and event except for exactly one added bit in pre_on (a clear
bit) or pre_off (a set bit); applicability flips and the golds differ. A1/Ib 160, A2/Ib 25, A4/Ia 35, Za/Zi 55; flip
kinds alternate.

### 3.3 Shortcut policies on the training rows (descriptive; `AUDIT.json`)

"pa_" = precondition-aware: copy when inapplicable, else the named arithmetic. Titan's own reading (R11) is pa_int_add.

| Policy | Rev. 1 | Rev. 2 | Training rows that separate it from the rule (rev. 2) |
|---|---|---|---|
| pa_int_add (preconditions + b+a) | 0.853 | 0.827 | 260 (A2, A3, A4, Za) |
| pa_int_add_del (preconditions + (b&~d)+a) | 0.947 | 0.890 | 165 (A3, A4; was 80) |
| pa_or_add (preconditions + b\|a) | 0.863 | 0.837 | 245 (A2, A4, Za) |
| pa_delete_only | – | 0.647 | |
| pre_on_only (ignores pre_off) | – | 0.751 | |
| pre_off_only (ignores pre_on) | – | 0.749 | |
| precount_cue (popcount(pre_on\|pre_off) ≥ 2 -> copy, else apply) | – | 0.659 | |
| add0_cue (add = 0 -> copy, else apply) | 0.57 | 0.620 | |
| addset_apply (apply whenever the add bit is already set, else the rule) | – | 0.977 | 35 (Ia) |

The last line is the residual weakness of the composition: only the 35 Ia rows punish "add bit set -> apply". The dev
bank's new D_inapp_addset stratum and the false-apply gate FA catch it at screening time (section 4).

### 3.4 Cues and artefacts (descriptive)

- Cue ceilings (majority lookup predicting applicability): popcount(pre_on|pre_off) 0.705, pre_off == 0 0.652,
  pre_on == 0 0.642, add == 0 0.620, add bit already set 0.587, delete hits a set bit 0.577, popcount(before) 0.52.
  Part of this is the rule itself (no precondition bits -> applicable).
- add = 0 rows are 290 of 1,500 (19.3%); 81% of them are copies. The answer rule "add==0 -> copy, else
  apply-regardless" scores 0.620 (rev. 1: 0.57); apply-regardless scores 0.574 on add ≠ 0 rows (copy 0.438). The dev
  bank and the chains have no add = 0 action, so the cue cannot inflate a dev score; it can bias toward applying on
  add ≠ 0 rows, which FA measures.
- Among rows whose add bit is already set, 82.5% are applicable (rev. 1: 80%).
- **Minimal-pair twin artefacts.** Adding one precondition bit makes 45 inapplicable twins carry precondition counts
  that no other training row has (pre_off popcount 2: 39 rows; pre_on popcount 3: 6 rows), a perfect inapplicability
  cue on those rows. The dev bank has no such rows (its marginals are the generator's).

### 3.5 Freshness

The exclusion union holds 310,258 payloads and 302,283 operative contents from 17 sources: the historical records tree
(`l1_make_ood.harvest_historical` via `p3_make_bank.load_all_sources`; the 4 `HSKILL_CONFIRM_V2*` files skipped by
name, never opened; `rcwmt.bank.load_exclusions()` not called), the pilot, confirm, confirm2 and s4000 banks, the 4 T2
data files, SELECT-VIEWS-001, the L1 OOD bank, 5 FDS/composition T banks, and the PREREG-3 bank.
- Training rows (1,500), dev one-step rows (1,050) and chain transitions (960): 0 payload and 0 operative-content
  overlap with every source and with each other. `t2bp_make_data.py --verify` re-checks this with a FRESH harvest
  (3,510 items, 0 overlaps).
- Gold = `env.step`, checked against the FDS oracle (`env.cross_check`) on every row and chain transition.
- No model, model output or GPU; torch not imported; peak RSS 364 MiB; a rebuild from the final code is byte-identical.

### 3.6 Surface diversity (decision point 1); OFF by default

`t2bp_make_data.py --surface-diversity on` writes `data/t2bp_train_sd_<arm>.jsonl` plus
`data/t2bp_train_sd.SOURCE.json`: the SAME canonical rows (read from the canonical files, never re-drawn), each with a
seeded rule-text paraphrase (`PARAPHRASES[0..3]`) and field-name variant (`FIELD_VARIANTS[0..3]`); variant 0 is
canonical and pair members share a variant; targets are unchanged. `t2bp_run.py --data-variant sd` trains on them
(outputs under `runs/t2bp/sd_<tag>/`) and adds stage **E3**: the dev one-step rows in the held-out surface (paraphrase 4
and field names 4, never used in training), OFF and ON, reported descriptively by the analysis. No code change is
needed; the tiny test runs an sd job with E3 end to end.

## 4. Dev screen bank (`data/t2bp_dev.jsonl`; no model call)

Revision 2 appends D_inapp_addset; the 950 revision-1 rows and the 240 chains are unchanged (checked).

| Stratum | n | Definition | Floors: copy / apply-regardless / b+a / OR / (b&~d)+a / b&~d |
|---|---|---|---|
| D_app_add | 100 | A1 | 0 / 1 / 1 / 1 / 1 / 0 |
| D_app_unchanged | 100 | A3 (applicable, unchanged: carry) | 1 / 1 / 0 / 1 / 0 / 1 |
| D_app_delete | 100 | A2 | 0 / 1 / 0 / 0 / 1 / 0 |
| **D_int_add_wrong** | 150 | A4: bitwise ≠ b+a, ≠ (b&~d)+a, ≠ b | 0 / **1** / 0 / 0 / 0 / **1** |
| D_inapp_pre_on | 100 | only pre_on violated; applying would change the state | 1 / 0 / 0 / .09 / 0 / .75 |
| D_inapp_pre_off | 100 | only pre_off violated; applying would change the state | 1 / 0 / 0 / .11 / 0 / .72 |
| **D_inapp_addset** (new) | 100 | Ia: inapplicable, add bit already set, delete clears a set bit (50 pre_on, 50 pre_off) — the inapplicable twin of D_int_add_wrong | 1 / 0 / 0 / 1 / 0 / 0 |
| **D_ood10** | 150 | 10-bit masks; before ≥ 256; a bit ≥ 8 operative; 25 each of A1, A2, A3, A4, inapp-pre_on, inapp-pre_off | .5 / .667 / .167 / .40 / .333 / .54 |
| **D_ood12** | 150 | the same with 12 bits | .5 / .667 / .167 / .367 / .333 / .58 |

**IAW is not a rule test on its own:** apply-regardless and delete-only both score 1.0 on D_int_add_wrong (the add bit
is already set). That is why the rule pairs it with FA (inapplicable strata, including the twin D_inapp_addset) and ID.

Shortcut-policy floors (`t2bp_dev.SOURCE.json`):

| Policy | IAW | D_inapp_pre_on | D_inapp_pre_off | D_inapp_addset | OOR (10/12) | K4 exact |
|---|---|---|---|---|---|---|
| copy | 0 | 1 | 1 | 1 | .5 / .5 | 0.129 |
| apply-regardless (ignore preconditions) | 1 | 0 | 0 | 0 | .667 / .667 | 0.442 |
| pa_int_add (R11) | 0 | 1 | 1 | 1 | .5 / .5 | 0.229 |
| pa_int_add_del | 0 | 1 | 1 | 1 | .667 / .667 | 0.392 |
| pa_or_add | 0 | 1 | 1 | 1 | .667 / .667 | 0.454 |
| pa_delete_only | 1 | 1 | 1 | 1 | .667 / .667 | 0.221 |
| pre_on_only | 1 | 1 | 0 | .5 | .833 / .833 | 0.804 |
| pre_off_only | 1 | 0 | 1 | .5 | .833 / .833 | 0.633 |
| precount_cue | .88 | .60 | .27 | .38 | .767 / .747 | 0.583 |
| addset_apply | 1 | .91 | .89 | 0 | .933 / .967 | 0.908 |

- **OOD prompts** differ from the arm's 8-bit prompt only in the width word ("a 10-bit"); for B-bits the bit strings
  are 10 or 12 digits long.
- **Chains.** 30 prerequisite parents × 8 candidate plans × 4 events, drawn as `rcwmt.bank.build` draws them (6-action
  library, ≥ 4 distinct finals per parent, all transitions fresh): 240 chains, 960 steps; copy floor at k=4 0.129. No
  chain step has add = 0.
- **Structural overlap with the training rows (descriptive):** a bit-permutation-invariant signature overlaps on
  0.30–0.96 of each 8-bit stratum (D_inapp_addset 0.41) and 0 of the OOD strata.

## 5. Runner (`scripts/t2bp_run.py`)

**Recipe.** T2 unchanged: the vendored trainer called exactly as `run_t2.py` calls it (hashes pinned); LoRA r16 /
alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down; frozen FP32 Titan-Sparse; WSD, lr 1e-4, batch 16 (4 × 4),
190 steps, seed 0. No merge (D-28): the adapter is loaded unmerged with PEFT.

**Evaluation.** OFF (`disable_adapter()`) and ON in one process, greedy FP32, `generation.generate_texts`, 160-token
cap. E1: the 1,050 one-step rows in the arm's own frame. E2: the 240 chains as a self-rollout in the arm's frame (for
D-int identical to `generation.chain(..., 'chain_m1')`, tested). E3: surface-diversity runs only (section 3.6).

**Anchor A0.** Once per turn before the first evaluation stage: adapter disabled, pilot TEST prerequisite chain_m1
(224 chains) must reproduce `runs/t2/t2_s0/gen_chain_m1_base.jsonl` (sha 7ebc8288…, hashed, never read for any choice)
on ≥ 99% of chains, else a persisted stop.

**One lease, `rcwm-t1:t2b-prereq`;** D-22 turn rules: 4.5 h cap from acquisition; an arm's training starts only if
1.25 × (its train + eval + load [+ anchor]) fits; each eval stage only if 1.25 × its estimate fits; nothing within
0.5 h of the cap; otherwise deferral to the next turn (exit 5), resumed from the recorded stages, re-anchored.

**Checks before the lease (revision 2 additions in bold):** data hashes equal the SOURCE records; **every frozen input
equals `SCREEN_RULE.json` (refuse, not persisted)**; every training prompt equals `tok.apply_chat_template`; the
one-waiter rule (R14); a run lock. **A resumed run refuses if any frozen code file differs from the hashes recorded at
its first turn. A completed run refuses a call with other arms (e.g. `--with-filler` after DONE) instead of exiting 0.**
Outputs are create-only with stage records hashed in `provenance.json`; a partial stage is moved to `_orphans/` and
redone. Exit codes: 0 done, 3 deterministic stop, 5 deferred, 1 crash.

**GPU estimate** (`GEN_PLAN.json`; no model loaded). Training: T2's measured 190-step time × tokens per row. Generation:
fitted to PREREG-2's P chain_m1 and ledger chain_one rates. **Planning rate (conservative):** × 3.8/2.7, and the
adapter-OFF side of S-trace, B-bits and F-filler at the 160-token cap (base Titan has never seen those formats).
**Observed rate (optimistic):** measured rates, OFF at the gold length.

| Arm | Train | E1 (planning) | E2 (planning) | Pair, planning | Pair, observed |
|---|---|---|---|---|---|
| D-int | 38.6 min | 7.6 min | 6.9 min | 0.92 h | 0.77 h |
| S-trace | 64.5 min | 36.2 min | 33.1 min | 2.26 h | 1.34 h |
| B-bits | 62.7 min | 34.2 min | 31.1 min | 2.17 h | 1.22 h |
| F-filler | 50.1 min | 36.2 min | 33.1 min | 2.02 h | 1.13 h |

- **Screen (3 arms), per turn:** planning: turn 1 = D-int + S-trace (3.24 h), turn 2 = B-bits (2.22 h), 5.46 h total;
  observed: one turn of 3.37 h. So the screen takes 1–2 turns; the runner decides from its own measured rates (the
  observed/planned ratio of the stages already run corrects later estimates).
- **R5 filler control (only if S-trace wins):** its own run, one turn: 2.08 h planning / 1.17 h observed.
- **Seeds 1–2 of the winner:** 0.92–2.26 h per seed (planning), one per turn.
- **Surface-diversity screen (if PREREG-3 says TRIGGER):** 3 turns (6.81 h) planning, 1 turn (3.80 h) observed.
- Anchor 3.2 min per turn; generations per screen arm 4,020 (E1 2,100 + E2 1,920, OFF + ON).

## 6. Analysis (`scripts/t2bp_analysis.py`) and the rule

CPU only, never imports torch. Before any output is read: `SCREEN_RULE.json` must verify (every frozen input, the
rule text and `SCREEN_RULE.md`; else exit 3) and the scorer self-test must pass: golds 1,050/1,050 in each of the 4
formats; the 6 heuristics and 9 shortcut policies scored at their exact floors in every format; 15 malformed answers
rejected and classified (12 format failures, 3 value failures); `format_ok` true on every gold; the D-int parser equals
`env.parse_state` on a 21-string battery. Then the run is validated (DONE.json, provenance hashes, ids in order, chain
states equal to the parser applied to the texts) and the rule is applied.

**Rule (revision 2), in short** (verbatim in `SCREEN_RULE.md`):
- metrics on adapter ON: IAW (D_int_add_wrong), OOR (mean of D_ood10, D_ood12), K4 (240 chains), FC (false copy on the
  350 8-bit state-changing rows, T2's definition), **FA** (false apply: the worst of the three inapplicable strata),
  **WF** (format rate on the 750 8-bit rows), ID (mean of the 7 8-bit strata);
- **R0** WF < 0.90 in any screen arm, or a turn without a passed anchor -> NO_VERDICT_INSTRUMENT;
- **R1** IAW < 0.20 in every arm -> STOP_P_TRAINING (quoting the dose: 150 A4 rows of 1,500);
- **R2** eligible iff **IAW ≥ 0.20**, FC ≤ 0.25, **FA ≤ 0.25**, ID ≥ ID(D-int) − 0.10;
- **R3** none eligible -> NO_ELIGIBLE_ARM; **R4** rank by R = (K4 + IAW + OOR)/3 with the tie-breaks; **close call**
  wording if another eligible arm is within 0.05;
- **R5** F-filler is a control, never eligible, run on its own only after an S-trace win (`--tag s0_ffill --arms
  ffill`, then `--filler-run`); flag TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE if R(F) ≥ R(S) − 0.05; a malformed
  filler voids the control (DECODE_COMPUTE_CONTROL_VOID); until then "decode-compute control pending".

**Dry run (synthetic outcomes, `DRYRUN.json`): 47/47.** 16 end-to-end scenarios, 21 rule units (+1 anchor unit), 3
close-call units, 4 R5 addendum units, 3 refusals. The three review cases now fail the screen as they should:
- pre_on-only arm (IAW 1.0, K4 0.804, R 0.879) is excluded by FA 1.0; the 80%-rule arm (R 0.857) wins;
- ignore-preconditions arm (FA 1.0) vs a T2-like control (FC 0.569) -> NO_ELIGIBLE_ARM;
- precondition-count cue arm (FA 0.73) -> NO_ELIGIBLE_ARM;
- also: add-bit-set -> apply (FA 1.0 on D_inapp_addset), delete-only (FC 0.286, ID 0.714), Titan's R11 reading
  everywhere (IAW 0 -> STOP), malformed answers in one arm (WF 0.823) or everywhere -> NO_VERDICT_INSTRUMENT.

Everything else is descriptive: OFF baselines; per-stratum and subtype accuracies with Wilson CIs; out-of-range false
copy; union cells; heuristic and shortcut-policy agreement per stratum; cmp_ok, calc_ok, flag_ok; k=1/2/4 exact; step
accuracy from the model's own state; G_chain regret (`decisions.g_only`, 256 goals, seed 1001) and floors; paired
ON−OFF and arm−D-int contrasts; E3 held-out surface. A run without all three screen arms (seeds 1–2, the filler
control) gets NO_VERDICT and a descriptive report.

## 7. Limits (state them with every result)

- **Resolution.** The SE of a difference in R is about 0.02–0.03 (K4 n = 240, IAW n = 150, OOR n = 300). The 0.02
  tie rule and the tie-breaks decide near-ties mechanically; a win by < 0.05 is worded as a close call: a screen choice,
  not a finding. Only the top two arms enter the tie-breaks.
- **Controls are partial.** F-filler differs from S-trace in its system sentence and gets token-mean loss over a
  different target. There is no length control for B-bits, which also changes the input representation (bit strings).
- **calc adds nothing for one-step P.** calc == after on every one-step row, so calc_ok repeats after-accuracy, and
  S-trace offers no intermediate step for the bitwise update that IAW measures; its added content is the predicate
  inputs (cmp).
- **Training artefacts absent from the dev bank:** add = 0 actions (19.3% of training rows) and the 45 twin rows with
  out-of-marginal precondition counts (section 3.4).
- **Dose.** 150 A4 rows of 1,500; a STOP verdict is a statement about this dose and recipe, not about every recipe.
- Dev bank, seed 0, one recipe: no confirmatory claim (roadmap item 2, "Confirm" needs a sealed bank and a
  preregistration).

## 8. Review of revision 1: what changed

| Review item | Action |
|---|---|
| MUST-FIX: eligibility checks only the copy shortcut; mirror shortcuts can win | **Applied.** FA (false apply, worst inapplicable stratum, ≤ 0.25) in R2; new D_inapp_addset stratum (the twin of IAW); FA's max-over-strata is the per-stratum floor for pre_on and pre_off; the reviewer's three policies are dry-run scenarios. The optional minimal-pair dev stratum was **not** added: FA's per-stratum maximum already forces each precondition kind, and new paired draws would add rows without changing what the gate can see. |
| MUST-FIX: the named winner can have IAW < 0.20 | **Applied.** IAW ≥ 0.20 is part of eligibility; the reviewer's case is a rule unit (-> NO_ELIGIBLE_ARM). |
| SHOULD: the frozen rule is not enforced | **Applied.** `--freeze-rule` writes `SCREEN_RULE.json` with every frozen input's sha256; the runner refuses before the lease and on a resume under changed code; the analysis refuses (exit 3) before reading outputs. |
| SHOULD: FC pools OOD rows | **Applied.** FC on the 350 8-bit rows (T2's definition); OOD false copy descriptive. |
| SHOULD: Titan's precondition-aware integer rule is barely separated | **Applied.** A4 65 -> 150 within the same gates; composites and separating-row counts in `AUDIT.json` and section 3.3; the STOP wording quotes the dose. |
| SHOULD: R5 unreachable; "filler only if S-trace wins" incompatible | **Applied.** The filler control runs as its own run only after an S-trace win, with a pre-stated addendum (`--filler-run`); `--with-filler` after DONE is an explicit stop. |
| SHOULD: instrument failure gives a programme verdict | **Applied.** R0: WF (format, not value) ≥ 0.90 in every screen arm and a passed anchor in every evaluating turn, else NO_VERDICT_INSTRUMENT. |
| SHOULD: held-out surface 4 never evaluated | **Applied.** Stage E3 in `--data-variant sd` runs, descriptive. |
| NOTE: GPU plan optimistic for OFF in new frames | **Applied.** OFF planned at the 160-token cap for non-native frames; the screen is "1–2 turns". |
| NOTE: resolution and partial controls | **Applied.** Section 7 and the close-call wording in R4. |
| NOTE: artefacts, docstring, floors, overwritten docs | **Applied.** Twin and add = 0 artefacts recorded; `--verify` now re-harvests and re-checks freshness; IAW floors of apply-regardless and delete-only and the K4 floor of ignoring preconditions stated (section 4); kit records are never overwritten once any `runs/t2bp/` output exists (timestamped copies instead). |
| NOTE: CUDA_VISIBLE_DEVICES disclosure | Kept in the operator report; nothing in the kit changes. |

## 9. Open questions for the operator

1. **add = 0 balancing rows.** Revision 2's larger A4 dose strengthens the "add 0 -> copy" cue (0.57 -> 0.62). The
   alternative is to relax the int b+a gate. Accept?
2. **IAW = A4 only (strict).** It is solved by "apply delete, ignore add" too; FA and ID now guard against that. The
   R11-wide union (A2+A3+A4) is reported. Keep the strict stratum for decision point 2?
3. **Representation choices.** S-trace lists bit positions (BWM's P instance lists viol and ok only); B-bits is MSB
   first; OOD prompts change only the width word.
4. **Thresholds chosen here, not given:** ID margin 0.10, FA ≤ 0.25 (mirrors FC), WF ≥ 0.90, close call < 0.05, and the
   equal-weight R = mean(K4, IAW, OOR) (the task said rank by K4 and IAW; D-52 (5) makes OOR primary).
5. **Queue placement.** The hook draft `queue_hook_t2bp.sh.draft` is NOT installed; it must follow PREREG-3 under the
   one-waiter rule, which needs a fresh chain (operator decision).

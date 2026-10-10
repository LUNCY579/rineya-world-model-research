# T2b prerequisite representation screen (RCWM roadmap item 2; joint roadmap 5a), revision 3

**Status:** kit rebuilt and tested on CPU only (2026-10-09, after the 09:42 restart). No GPU job has run; no `runs/t2bp/`
output exists. Nothing here is a preregistration: this is a seed-0 screen on a dev bank.

Revision 3 applies the operator's decisions after PREREG-3 returned STATUS: TRIGGER (DECISIONS R-PREREG-3, D-35, R16,
D-36). The decision rule (revision 3) was written before any output existed. It is recorded in `SCREEN_RULE.md`, and
`SCREEN_RULE.json` holds the sha256 of 26 frozen inputs; the runner and the analysis refuse when one differs.
Revisions 1 and 2 of the rule are kept (`SCREEN_RULE_rev1.*`, `SCREEN_RULE_rev2.*`), and this document's revision 2 is
kept as `DESIGN_rev2.md`. The queue hook is still a draft (`queue_hook_t2bp.sh.draft`): not installed, not frozen.

## 0. Operator decisions applied in revision 3 (binding), each with its reason

| Decision | What the kit does now | Reason |
|---|---|---|
| **C0** interruption | A previous attempt was cut off by a usage limit. Its state was inspected before any change: all six scripts already carried the revision-3 code; the data had been re-rendered and the sd build written; DESIGN.md, the rule, the dry-run, scorer, tiny-test records and the hook draft were still revision 2. The work was finished, the data rebuilt from the final code (row files byte-identical; only the SOURCE records changed), and every check rerun. Every text file the kit writes now has LF line endings. The kit writes through `newline='\n'`; this fixed six records that had been written CRLF. | No half-applied revision; DECISIONS I-30 (CRLF from Python text-mode writes on Windows). |
| **C1** primary build = surface diversity (`sd`) | `t2bp_run.py` trains only `data/t2bp_train_sd_<arm>.jsonl`. Tags are `sd_*`, and a non-sd configuration is refused before the lease. The analysis scores only sd runs. The sd files hold the 1,500 canonical rows (read from the canonical D-int file after its hash check, never re-drawn) and their unchanged targets, each with a seeded paraphrase, field-name variant and key orders. The canonical-only variant is not run. | Decision point 1 fired (PREREG-3 TRIGGER): small LoRAs learn the surface form of the rule cases they see. |
| **C2** input key-order augmentation | Every JSON object of a training prompt is serialized in a key order drawn per row (seeded; pair members share it) from a training subset. Held out per object type, fixed by `ORDER_SEED` and recorded: the reversed canonical order plus 2 random permutations (section 3.6). The prompt's only list (`events`) has one element, so there is no list order to augment. Targets keep the literal cmp -> calc -> after order. Tests and the audit prove that no training prompt uses a held-out order, paraphrase or field variant: each surface is read back from the prompt text, and 8 text fragments are scanned. | R16: both LoRA world models are locked to JSON key order (BWM reversed order 0.458 vs 1.000; T2 0.37–0.47 vs 0.87–0.91). |
| **C3** held-out-surface evaluation | New stage **E3**: the 1,050 one-step rows, with the same rows and golds, in ONE combined held-out surface: paraphrase 4 with its state sentence, field names 4, the reversed top-level order, and one of the 3 held-out event orders, rotating by the row's index within its stratum. Adapter ON, paired by row with E1. The per-factor held-out renderings are **omitted**: they would take the screen from 2 to 3 turns at the planning rate (GEN_PLAN.json). | D-35: every claim gets a surface-form audit next to the out-of-range audit. |
| **C4 / 5a (c)** amendment 4b filler | `FILL_K = {"action": 26}`: one constant fill per (family, action kind). It is at least the longest trace over train + dev (all surfaces share targets) and is recomputed by the builder, which refuses on a mismatch. Per-row length matching is removed. A test shows the fill length carries zero information about the answer, and the residual is reported descriptively (section 3.7). | D-36: a filler matched to the trace per row leaks the answer through its length, giving it a hidden scratch channel. Here, the revision-2 length alone predicts applicability at 1.0 on the training rows. |
| **C5** turn cap | Everything fits the 4.5 h cap with the existing deferral/resume. At the planning rate the screen takes 2 turns, at the observed rate 1. `--plan`, `GEN_PLAN.json` and the hook draft (variant sd, tags `sd_s0` / `sd_s0_ffill`) are updated. | Operator constraint. |
| **C6** new rule revision | `t2bp_analysis.py --freeze-rule` wrote revision 3, with a "what changed and why" preface. It renamed revision 2 to `SCREEN_RULE_rev2.*`; nothing was deleted. No `runs/` output exists. | Pre-stated rule, revisions kept. |
| **5a (a)** ranking | R(A) = (K4 + IAW + OOR + HS) / 4. HS is the equal-weight mean accuracy of the seven 8-bit strata in E3. The rest is unchanged: eligibility (IAW ≥ 0.20, FC ≤ 0.25, FA ≤ 0.25, ID non-inferiority), R0 on the in-surface format rate, tie-breaks and close-call wording. | A surface-locked arm must not win on in-surface scores. |
| **5a (b)** open questions | All accepted as they were. The add = 0 -> copy cue at 0.62 stays descriptive, guarded by FC and FA. Strict A4 IAW is kept for decision point 2, with the R11 union reported. The three representation choices stay. The width word appears in held-out form in E3, because each paraphrase's state sentence words the width its own way (held out: "a flag mask 12 bits wide"). The thresholds stay: ID margin 0.10, FA ≤ 0.25, WF ≥ 0.90, close call < 0.05. | Operator answers. |

## 1. Question

- T2's prerequisite fix learned a COPY shortcut: false copy rose from 0.039 to 0.624, and applicable state-changing accuracy fell from 0.587 to 0.353.
- Titan's own prerequisite "rule" is integer addition (R11).
- The screen asks whether a de-shortcut training set and/or a different target representation lets the T2 recipe learn the precondition rule and the bitwise update. It must do so out of range AND, since revision 3, on a surface no training prompt used.
- **Decision point 2:** if every arm scores < 0.20 on int-add-wrong, stop P training and move P to a code-side bit interface.

## 2. Arms: the same 1,500 rows, rendered four ways (formats unchanged since revision 1)

| Arm | Prompt state / masks | Exact target (row: before 9, inapplicable, pre_off bit 3 set) | Parser |
|---|---|---|---|
| D-int (data control) | integers | `9` | lenient, exactly `env.parse_state` with range [0, 2^w) |
| S-trace (joint format, D-34) | integers | `{"cmp":{"on_miss":[],"off_hit":[3],"viol":1,"ok":false},"calc":9,"after":9}` | strict, literal key order |
| B-bits | w-digit bit strings, MSB first | `{"applicable":false,"after":"00001001"}` | strict |
| F-filler (control, conditional) | integers | `{"fill":" . … (26 units) .","after":9}`, the same fill on every row | strict |

- **Canonical surface:** paraphrase 0, field names 0, canonical key orders. In it, the D-int prompt is T2's prerequisite frame byte for byte (500/500 stored T2 prompts reproduced).
- **What varies by arm:** only the format sentence. B-bits also adds the bit-string convention to the state sentence.
- **`format_ok` vs value:** an out-of-range state is a wrong answer in the right format. WF counts format.

## 3. Training data

### 3.1–3.5 Rows, audit, shortcuts, cues, freshness (unchanged from revision 2; details in `DESIGN_rev2.md` §3)

- **Rows:** seed 20261111; 1,500 rows. Types: A1 490, A2 40, A3 15, A4 150, Za 55, Ib 480, Ia 35, Zi 235. There are 275 minimal pairs.
- **Audit gates** (each 0.48–0.52):

  | Heuristic | Score |
  |---|---|
  | copy | 0.5100 |
  | apply-regardless (= ignoring preconditions) | 0.5000 |
  | before+add | 0.4833 |
  | OR | 0.5167 |

- **Quotas:**

  | Quota | Count |
  |---|---|
  | int-add-wrong, strict (A4) | 150 |
  | applicable-delete | 245 |
  | int-add-wrong, R11 definition | 260 |
  | pre_on-missing / pre_off-present | 377 / 373 |

- **Shortcut policies (descriptive):**

  | Policy | Score |
  |---|---|
  | pa_int_add | 0.827 |
  | pa_int_add_del | 0.890 |
  | pa_or_add | 0.837 |
  | addset_apply | 0.977 |
  | add0_cue | 0.620 |
  | precount_cue | 0.659 |

- **Minimal-pair twins:** 45 twin rows carry precondition counts that no other row has.
- **Freshness:** `--verify` re-ran a fresh name-skipping harvest on 2026-10-09 at 12:13 (17 sources; the `HSKILL_CONFIRM_V2*` files were skipped by name and never opened; `load_exclusions()` was not called). All 3,510 items (training rows, dev rows and chain transitions) were checked against every source and against each other: 0 overlaps. Peak RSS 371 MiB.

### 3.6 The surface-diversity build (PRIMARY; C1, C2)

`data/t2bp_train_sd_<arm>.jsonl` + `data/t2bp_train_sd.SOURCE.json`. Same rows, same order, same targets as the canonical files (checked by the audit, the test and `--verify`). Each row (pair members share one surface) gets:

| Factor | Training subset | Held out (E3 only) |
|---|---|---|
| Rule-text paraphrase + its state sentence (which carries the width word) | `PARAPHRASES[0..3]` / `STATE_SENTENCES[0..3]` (0 = T2's) | index 4: "Apply the actions in order. Each action fires only when …" / "The state is a flag mask {w} bits wide" |
| Field names (before, events, add, delete, pre_on, pre_off) | `FIELD_VARIANTS[0..3]` (0 = T2's) | index 4: mask, moves, plus, minus, must_on, must_off |
| Top-level object {before, events} (2 keys) | the canonical order only (no other order exists) | the reversed order (events, before) |
| Event object (7 keys: add, cost, delete, kind, name, pre_off, pre_on) | 5,037 orders. The canonical order is drawn with p = 0.25; otherwise the order is uniform over the rest. | the reversed order; (add, delete, kind, pre_off, cost, pre_on, name); (cost, pre_off, delete, pre_on, name, kind, add). These are regenerated from `ORDER_SEED` = 202611116 and checked. |
| List order | the only list (events) has one element | none |

**Diversity actually used** (`AUDIT.json`):
- All 16 paraphrase × field-name combinations occur.
- Rows per paraphrase: 403 / 369 / 379 / 349.
- Rows per field-name variant: 396 / 354 / 321 / 429.
- The canonical event order is on 26.1% of rows; 829 distinct event orders occur.
- The top-level order is always canonical.
- Only **22** training rows are in the exact canonical surface that E1/E2 evaluate. Each of its factors is common (paraphrase 0 on 403 rows, names 0 on 396, the canonical event order on 392), but the combination is rare.

**Proof that no held-out surface is in training:**
- **Text read-back:** every one of the 6,000 sd training prompts (4 arms × 1,500) has its paraphrase, field variant, top and event orders and width read back from the prompt TEXT (`recover_surface`, independent of the recorded surface). The read-back equals the recorded surface and holds no held-out factor: 0 hits.
- **Text fragments:** none of the 8 held-out fragments occurs in any prompt: paraphrase 4's opening words, " bits wide", and the 6 held-out names as JSON keys.
- **Refusal tests:** the audit refuses a row rendered with any single held-out factor (6 cases), and a prompt in a held-out order whose record claims a training order.

### 3.7 Amendment 4b filler (C4)

`FILL_K = {"action": 26}` is BWM's formula: the longest trace head over train + dev + dev chain steps (3,510 items; 29 tokens) minus the `{"fill":""` base. It is recomputed by the builder under the `tokenizers` library and by the test under the HF tokenizer; both agree.

**Zero-information test** (`AUDIT.json` → `filler_4b`; tiny test §4): all 3,000 training filler targets (canonical + sd) and all 1,050 dev golds share one fill string. On the training rows, predicting each property from the fill length scores exactly the majority share, with mutual information 0 bits:

| Property | Accuracy from fill length (= majority share) |
|---|---|
| applicable | 0.500 |
| changed | 0.510 |
| after | 0.0147 |
| type | 0.327 |

**What revision 2's row-matched length would have revealed** (descriptive):

| Rows | Applicability (majority share) | Change (majority share) |
|---|---|---|
| Training | 1.00 (0.50) | 0.99 (0.51) |
| Dev 8-bit | 1.00 (0.60) | 0.87 (0.53) |
| Dev out-of-range | 0.997 (0.667) | 0.83 (0.50) |

**Residual covariate** (filler tokens − trace tokens; descriptive):
- Training: 2–7, mean 5.0, positive on 100% of rows. The mean is 5.49 on applicable rows and 4.51 on inapplicable rows. This is a property of the row's trace, not of the filler target, which is identical across rows.
- Dev: 0–7, mean 4.8.
- The analysis reports the filler arm's accuracy by residual value and stratum.

The superseded row-matched filler file and the revision-2 SOURCE record are kept as `data/t2bp_train_ffill.rev2_rowmatched.jsonl` and `data/t2bp_train.rev2.SOURCE.json`.

## 4. Dev screen bank (`data/t2bp_dev.jsonl`; unchanged, sha 46421b99…)

The bank has 1,050 one-step rows in 9 strata:

| Stratum | Rows |
|---|---|
| D_app_add | 100 |
| D_app_unchanged | 100 |
| D_app_delete | 100 |
| D_int_add_wrong | 150 |
| D_inapp_pre_on | 100 |
| D_inapp_pre_off | 100 |
| D_inapp_addset | 100 |
| D_ood10 | 150 |
| D_ood12 | 150 |

It also holds 240 k=4 chains (960 steps; the copy floor at k=4 is 0.129). Floors, subtypes and shortcut floors are in `DESIGN_rev2.md` §4 and `t2bp_dev.SOURCE.json`.

**E3 rendering.**
- **Surface:** `heldout_surface_of(row)` = paraphrase 4, names 4, reversed top order, and the event order `HELDOUT_ORDERS["event"][j % 3]`, where j is the row's index within its stratum (its id suffix; checked on all 1,050 rows). The 3 orders rotate evenly within every stratum (e.g. 34/33/33).
- **Read-back:** every E3 prompt reads back as that surface and encodes the row's own state and event.
- **OOR rows:** they differ from 8-bit rows only in the width word, in every surface: 200 system texts checked (4 arms × 5 paraphrases × 5 field variants × 10/12 bits).
- **Prompt length:** E3 prompts are about 6 tokens shorter than in-surface prompts. For D-int the mean is 145 vs 151 tokens.

## 5. Runner (`scripts/t2bp_run.py`)

**Recipe:** the T2 recipe, unchanged.
- LoRA r16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up and down.
- Frozen FP32 Titan-Sparse; WSD schedule, lr 1e-4, batch 16, 190 steps, seed 0.
- The vendored trainer is pinned. No merge (D-28).

**Stages per arm:**

| Stage | What it runs | Sides |
|---|---|---|
| T | training | – |
| E1 | 1,050 one-step rows, canonical surface | OFF + ON |
| E2 | 240 chains, self-rollout, canonical surface | OFF + ON |
| E3 | 1,050 one-step rows, combined held-out surface | **ON only** |

The anchor A0 runs once per turn (224 pilot chains, ≥ 99% identical to T2's chain_m1 OFF file).

**Checks before the lease:**
- the frozen inputs equal `SCREEN_RULE.json`;
- the sd files and the canonical row source match their SOURCE records;
- every training prompt equals the chat template of its messages (the row's own surface);
- the one-waiter rule and the run lock.

A resume refuses under changed code, and a completed run refuses other arms. Exit codes: 0 / 3 / 5 / 1.

All writes are create-only and LF. They were checked against the current rule record for tags `sd_s0`, `sd_s0_ffill` and `sd_s1` (CPU, without the lease).

**GPU plan** (`GEN_PLAN.json`; no model loaded; 4.5 h cap; a stage starts only if 1.25 × its estimate fits, and never within 0.5 h of the cap):

| Arm | Train | E1 | E2 | E3 (ON) | Pair, planning | Pair, observed |
|---|---|---|---|---|---|---|
| D-int | 39.9 min | 7.6 min | 6.9 min | 3.8 min | 1.00 h | 0.83 h |
| S-trace | 65.7 min | 36.2 min | 33.1 min | 8.2 min | 2.42 h | 1.45 h |
| B-bits | 64.1 min | 34.2 min | 31.1 min | 6.1 min | 2.29 h | 1.31 h |
| F-filler | 52.5 min | 37.0 min | 33.8 min | 8.9 min | 2.24 h | 1.31 h |

**Turns:**
- **Screen, planning rate** (OFF of non-native formats at the 160-token cap): 2 turns.
  - Turn 1: D-int + S-trace, 3.48 h.
  - Turn 2: B-bits, 2.35 h.
  - Total 5.83 h.
- **Screen, observed rate:** one turn of 3.64 h. The runner corrects its estimates from measured times and defers or resumes on its own.
- **Anchor:** 3.2 min per turn.
- **Generations:** 17,002 for the two-turn screen.
- **R5 filler control** (only if S-trace wins): its own run, one turn, 2.29 h planning / 1.34 h observed.
- **Seeds 1–2 of the winner:** 1.00–2.42 h per seed, one per turn.
- **Margin:** every arm's whole pair × 1.25 fits an empty turn (S-trace needs 3.09 h). A 3.0 h cap would be too small (tested).
- **Per-factor held-out renderings** (3 × 750 8-bit rows, ON): they would make the planning screen 3 turns (6.52 h), so they are omitted (C3).

## 6. Analysis and the rule (revision 3, verbatim in `SCREEN_RULE.md`)

Before any output is read:
- `SCREEN_RULE.json` must verify: 26 inputs, the rule text and `SCREEN_RULE.md`.
- The scorer self-test must pass:
  - golds score 1,050/1,050 in each of the 4 formats;
  - every heuristic and shortcut policy scores at its exact floor;
  - all 16 malformed strings are rejected;
  - one constant fill on every dev gold, and no filler shorter than its trace.

**Metrics** (adapter ON; all in-surface except HS):

| Metric | Definition |
|---|---|
| IAW | D_int_add_wrong |
| OOR | mean of D_ood10 and D_ood12 |
| K4 | 240 chains |
| **HS** | the seven 8-bit strata in E3, equal weight |
| FC | 350 rows |
| FA | the worst of 3 inapplicable strata |
| WF | in-surface format rate |
| ID | the seven 8-bit strata |

**Rule:**
- **R0:** WF < 0.90 or an unanchored turn → NO_VERDICT_INSTRUMENT. Malformed held-out answers count as wrong answers; they are not an instrument failure.
- **R1:** STOP_P_TRAINING, quoting the dose (150 A4 rows of 1,500).
- **R2:** eligibility.
- **R3:** NO_ELIGIBLE_ARM.
- **R4:** rank by **R = (K4 + IAW + OOR + HS)/4**, with tie-breaks at 0.02 (IAW, then OOR, then the order D-int, B-bits, S-trace). A margin < 0.05 is worded as a close call.
- **R5:** the F-filler control (4b fill, all stages including E3; its R includes its HS) flags TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE if R(F) ≥ R(S) − 0.05.
- **No verdict:** a run that is not sd, or lacks E3, gets no verdict.
- **Descriptive only:** held-out OOR strata; FC, FA, WF and ID in the held-out surface; paired held-out − in-surface contrasts per stratum; accuracy by held-out event order; the filler residual.

**Dry run: 61/61** (`DRYRUN.json`).
- **Counts:**

  | Unit type | Count |
  |---|---|
  | end-to-end scenarios | 21 |
  | rule units (incl. the anchor unit) | 28 |
  | close-call units | 3 |
  | R5 addendum units | 4 |
  | refusals | 5 |

- **New for revision 3:**
  - **Surface-locked S-trace** (the rule in-surface, malformed or copying on the held-out surface): it has the largest R under revision 2's formula (1.00 vs B-bits 0.889), but B-bits wins under revision 3 (0.917 vs 0.750 / 0.893).
  - **HS breaks an in-surface tie:** D-int loses the rule on the held-out surface, so S-trace wins.
  - **Malformed held-out answers everywhere:** not an instrument failure.
  - **Perfect HS:** it does not rescue an FA-excluded arm.
  - **HS = 0 everywhere:** still a verdict.
  - **The filler's R includes its HS.**
  - **A screen run without E3:** NO_VERDICT.

## 7. Limits (state them with every result)

- **One held-out surface.** E3 holds out ONE paraphrase, ONE name set and 1 + 3 key orders, combined. A drop on E3 cannot be attributed to a factor, because the per-factor renderings were omitted for budget. A pass shows transfer to this surface, not to every surface.
- **The top-level key order is never varied in training.** The object has 2 keys, so the only other order is the held-out reversed one. E3's top-level order is therefore a pure extrapolation; this follows from C2's definition.
- **In-surface means "made of training factors".** E1/E2 use the canonical surface, which is exactly matched by only 22 training rows; its factors are each common.
- **Resolution.** The SE of a difference in R is about 0.02–0.03 (HS adds n = 750, paired). A win by < 0.05 is a close call.
- **Controls are partial.**
  - F-filler differs from S-trace in its system sentence and its loss target. With the 4b fill it is also longer than the trace on almost every row (residual mean 5 tokens), so it has slightly more decode compute.
  - There is no length control for B-bits.
- **calc adds nothing for one-step P.** calc == after on every one-step row.
- **Training artefacts absent from the dev bank:** add = 0 rows (19.3%) and 45 twin rows.
- **Dose.** 150 A4 rows of 1,500.
- **Scope.** Dev bank, seed 0, one recipe: no confirmatory claim. A confirmation needs a sealed bank and a preregistration with held-out-surface primaries (D-35).

## 8. Records and tests (revision 3)

| Record | Content |
|---|---|
| `AUDIT.json` (sd) / `AUDIT_canonical.json` | every gate ok; 4b filler; surface proof and diversity |
| `SCORER_SELFTEST.json` | ok |
| `DRYRUN.json` | 61/61 |
| `TINY_TEST.json` | 79/79 checks, 125.6 s, peak RSS 1,155 MB, 2 threads, CPU only, tiny random Qwen3 |
| `GEN_PLAN.json` | `--plan`, torch not imported |
| `SCREEN_RULE.md/.json` | revision 3, 26 inputs, pre-freeze checks 6/6 |

The tiny test adds these checks for revision 3:
- OOR prompts differ only in the width word in every surface (the held-out one included);
- no CRLF in any kit file;
- the audit refuses held-out factors;
- the 6,000-prompt proof;
- held-out orders regenerate from the seed;
- E3 rotation and read-back;
- 4b under the HF tokenizer and the zero-information test;
- a tiny sd run with E3 ON only, where held-out prompts appear only in E3;
- an oracle run on the full dev bank with HS 1.0;
- the R5 addendum with the 4b filler and its residual covariate;
- C1 refusals in the runner and the analysis;
- a changed sd file refuses before the lease;
- crash/resume and deferral byte-identical with E3;
- the plan's turn counts and per-arm fit.

`--verify`: all checks ok (fresh harvest). Analysis CLI without a run: exit 3, nothing written.

**Line endings.** `SCREEN_RULE_rev1.json` (CRLF, written by revision 1) is kept byte for byte, because the revision-2 record pins its sha256. Every other kit file is LF.

## 9. Open questions for the operator

1. **Queue placement.** Installing the hook draft needs a fresh chain at a safe point (one-waiter rule); this is an operator decision logged in DECISIONS.md.
2. **Per-factor held-out renderings.** If the screen's first turn shows the observed rate (1 turn, 3.64 h), the three per-factor renderings would also fit one turn (4.10 h observed). Adding them would be a new kit revision before any run, not a mid-run change.

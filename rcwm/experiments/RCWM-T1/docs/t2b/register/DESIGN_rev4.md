# T2b-register screen, revision 4: intermediates in the answer, with an answer-free filler control

RCWM roadmap item 3 (`docs/roadmap/RCWM_ROADMAP_2026-10-08.md`); joint roadmap item 5b with BWM-001
(`JOINT_ROADMAP_2026-10-08.md`, BWM D-52); DECISIONS D-34, D-35, R16, D-36. Revision 4 applies the operator-selected
fixes from the independent verifier.

**Status:** exploratory screen.
- This is a CPU-only kit. It is not frozen, no GPU run has happened and no model output exists.
- The queue hook is a draft and is NOT installed.
- The GPU job refuses to run until `python -B scripts/t2br_run.py --freeze --approval "..."` has recorded the
  operator's go-ahead. That is a screen lock, not a preregistration.

**Revisions.** Each was written before any `runs/t2b_register` output existed. Superseded ones are kept verbatim,
renamed, never deleted.

| revision | when (2026-10-09) | design | go rule |
|---|---|---|---|
| 1 | about 03:04, first build | `DESIGN_rev1.md` | `GO_RULE_rev1.md` |
| 2 | about 07:54, after the adversarial review | `DESIGN_rev2.md` | `GO_RULE_rev2.md` |
| 3 | about 12:24, operator decisions after PREREG-3 (D-36) | `DESIGN_rev3.md` (the former DESIGN.md, renamed byte for byte, sha256 `36070a82…`) | `GO_RULE_rev3.md` (the former GO_RULE.md, renamed byte for byte, sha256 `b342898d…`) |
| 4 | about 17:45, operator-selected verifier fixes | `DESIGN.md` (this document) | `GO_RULE.md` (with the "what changed and why" preface) |

## 0. Changes

### 0.1 Revision 4: operator-selected fixes from the independent verifier (applied exactly)

**R4-1. Identification gap: new stratum `cond_sametens` and a third G1 clause.**

The gap:
- "Add iff tens(other) > tens(threshold) or other == threshold + 1" is exact on every primary conditional row:
  - far pairs and out-of-range-threshold pairs are separated by the tens digits;
  - every cond_oor_val pair is (x = t: false, x = t + 1: true).
- So is its mirror for the env's strict `>`: "add iff tens(other) >= tens(threshold) and other != threshold".
- Both therefore held every revision-3 G1 part, on both surfaces. No primary stratum separated them from the true rule.

How the equality terms follow from the code (`rcwmt/env.py`: `reg[other] > threshold`, strict):
- The boundary pair's first true value is t + 1. So the shortcut ORs in `x == t + 1`, and the mirror ANDs in `x != t`
  (the last false value).
- With a non-strict `>=` they would be `x == t` and `x != t - 1`.

What the training rows contradict (`AUDIT.json` `audit.digit_shortcuts`; reported, not gated):
- The 750 rows are fixed and never re-drawn.
- They contradict the shortcut on only 25 of 300 conditional rows: twins at +2..+9 inside one decade (accuracy 0.917).
- They contradict the mirror on 59 rows (accuracy 0.803).

The stratum (`t2br_make_data.build_sametens`):
- **Size and balance.** 100 rows: 50 branch-balanced minimal pairs, one ok-true and one ok-false member each.
- **Tens digits equal.** tens(other) == tens(threshold) in both members, so the branch is decided by the units digit.
- **Distance.** |other - threshold| lies in 2..9. The realized range is 2..7: both members lie in the decade of the
  number they share, so d_true + |d_false| <= 9. Counts: 34 / 26 / 24 / 10 / 4 / 2 rows at distance 2 / 3 / 4 / 5 /
  6 / 7.
- **Out of range.** |value| is 12..40, outside the trained -8..11, so the stratum is out of range as asked. Signs: 62
  positive, 38 negative (random draw).
- **In range otherwise.** Registers and thresholds are in range.
- **No wraps.** Neither branch wraps: the updated register lies in |value|..96-|value|. A row's exact 'after' is right
  iff its branch is right, given the in-branch arithmetic. So the stratum isolates the comparison from the modular
  reduction, which mix_oor, wrap_oor and cond_oor_* already test.
- **Pair types.** 25 pairs move the threshold and 25 move the other register (cond_mid's split).
- **Seed and registry.** It is drawn AFTER the chains, with its own seed `SEED_SAMETENS = 20261114`, against the full
  registry (PREREG-3's exclusion union + training + dev + chains). So the rows are fresh and disjoint from every
  training payload and operative content.
- **Existing rows untouched.** Every earlier dev row and the chain bank are unchanged. The new rows are appended as
  dv1100..dv1199.

The rule change:
- G1 adds "trace - plain >= +0.25 on cond_sametens alone, both in-surface and on the held-out surface". It is decided
  exactly on counts, like the other clauses (`t2br_analysis.COMPONENTS`: now 16 components).
- cond_sametens is NOT pooled into the primary 'conditional'.
- It is rendered in E1 (now 1,200 rows) and in the held-out-surface bank (now 500 rows, so E1H has 500).

Dry-run evidence (`DRYRUN.md`): see the table in 0.1.1 below.

**R4-2. Amendment 4b: the wording now states exactly what the code does, and the code was fixed.**
- **Revision 3's code** compared each row's filler target with THAT row's trace target: the smallest n with
  filler(n, after_i) >= trace_i on every row i. That is a per-row maximum: swap 5, add 14, mix 14, conditional 26.
  The filler target of a row with a short 'after' could be shorter than the kind's longest trace.
- **Revision 4** implements the constant definition: the smallest run of ' .' units such that the filler target is at
  least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev + the chains' gold steps,
  whatever its 'after'.
  - `t2br_common.min_constant_fill` compares the kind's shortest filler target with its longest trace target.
  - Revision 3's derivation is kept as `min_per_row_fill_rev3`, for the record.
  - The new constants are swap 7, add 17, mix 17, conditional 29.
- **Unchanged:** every filler target of a kind is identical up to 'after'.
- **Data:** the files change only in their filler fields (see section 13).

**R4-3. Notes** (section 14):
- the held-out state key orders are the three odd permutations;
- the generation bound now counts re-anchors;
- per-factor renderings are omitted as a stated choice.

#### 0.1.1 Dry-run evidence for R4-1

| trace side | conditional | cond_oor_val | cond_sametens | verdict |
|---|---|---|---|---|
| `tens_or_eq_plus1` (the verifier's shortcut) | 1 | 1 | 1/2 | NO-GO (failed: G1 [cond_sametens, cond_sametens HS]); every revision-3 G1 part holds |
| `tens_ge_not_eq` (the mirror) | 1 | 1 | 1/2 | the same |
| `go` (gold) | 1 | 1 | 1 | GO |
| `true_rule_digitwise` (tens, then units on a tie) | 1 | 1 | 1 | GO |
| `coarse_comparison` (tens only) | 253/300 | 53/100 | 1/2 | NO-GO (failed: G1 [cond_oor_val, cond_sametens, cond_oor_val HS, cond_sametens HS]) |

### 0.2 Operator decisions of revision 3 (binding; unchanged except where revision 4 says so)

- **C1. The primary build is the surface-diversity variant 'sd'; the canonical-only variant is not run.**
  - Reason: decision point 1 has fired. PREREG-3 returned STATUS: TRIGGER (REPORT section 3c): T2's void no-op is a
    token cue, and it fails on reordered JSON keys (0.37-0.47 against 0.87-0.91 in format), on word account names and
    on magnitudes.
  - Implementation:
    - The SAME 750 canonical rows (never re-drawn) get one seeded surface per row: a rule-text paraphrase, a field-name
      variant per field, and key orders (C2).
    - The build asserts the revision-2 row digests.
    - `--verify` reconstructs the four revision-2 files and the eight revision-3 files byte for byte from the current
      rows.
    - `t2br_run.py` refuses any variant other than `sd` (exit 3, not persisted).
    - The canonical rendering `data/t2br_train_<arm>.jsonl` is kept as the rows' source of truth. It is never
      trained.
- **C2. Input key-order augmentation from a TRAIN subset of permutations; held-out orders never in training.**
  - Reason: R16, agreed with BWM (their D-52 / D-54). Both LoRA world models are locked to key order but invariant to
    names.
  - Every JSON object of the prompt is serialized in a key order drawn per row from the TRAIN subset of its type:
    - the top level `{before, event}`;
    - the state `{X, Y, Z}`;
    - the event `{kind, other, register, threshold, tick, value}`.
  - Held out (`t2br_common.HELDOUT_ORDERS`, regenerated from seed 20261113 and checked at import): per type, the fully
    reversed canonical order plus 2 fixed permutations.

    | object | held-out orders | training orders |
    |---|---|---|
    | top | `event/before` | 1: the canonical order alone |
    | state | `Z/Y/X`, `Y/X/Z`, `X/Z/Y`: the three ODD permutations (section 14) | 3: the cyclic rotations `X/Y/Z`, `Y/Z/X`, `Z/X/Y` |
    | event | `value/tick/threshold/register/other/kind`, `other/kind/tick/threshold/register/value`, `threshold/kind/tick/other/value/register` | 717 |

  - Paraphrase 4 and field-name variant 4 of every field are held out too: `dest`, `peer`, `offset`, `pivot`, `opcode`,
    `memory`, `operation`.
  - The register prompt has no lists. Targets keep the literal `cmp -> calc -> after` order.
  - Proven on the files by `--verify` (decode of every training prompt + raw-string check) and by the tiny test (with a
    negative control).
- **C3. One held-out-surface stage, E1H.**
  - It renders the PRIMARY strata (mix_oor, cond_oor_val, cond_oor_thr, cond_far) and, since revision 4, cond_sametens:
    500 rows.
  - The rendering is paraphrase 4 + names 4 + a held-out key order per object type. Orders rotate by row (state and
    event j mod 3; the top level always uses its one held-out order).
  - Same rows and golds as in-surface, paired by row (`data/t2br_dev_hs.jsonl`). Its first 400 rows are revision 3's
    bank byte for byte.
  - All four sides are evaluated: plain and trace gate G1 / G2; OFF and filler give the surface audit.
  - Per-factor renderings are omitted by choice (section 14).
- **C4. Amendment 4b filler (proposed to BWM, adopted; BWM D-55).** The definition is made exact in revision 4 (R4-2).
  - The fill is a run of `" ."` of a constant length per action kind (`t2br_common.FILL_UNITS`): swap 7, add 17, mix
    17, conditional 29.
  - Each constant is the smallest run such that the filler target is at least as long as the LONGEST trace target of
    the kind over train + dev + the chains' gold steps, whatever its 'after'.
  - Targets do not depend on the prompt surface, so this covers every surface variant. The build re-derives the
    constants and fails on any other value.
  - Reason for 4b: revision 2's per-row matched fill leaked the answer through its length (wrap 0.957 / 0.970; branch
    0.57 / 0.85 / 0.72).
  - Test: every filler target of a kind carries the identical fill, so the fill length's best cut equals the majority
    rate in every cell. Zero information.
- **C5. Fits the 4.5 h turn cap with the existing deferral / resume** (section 9; updated for revision 4).
- **C6. Every rule change is a new revision with a "what changed and why" preface (`GO_RULE.md`).**
- **Kit 5b decisions:**
  - **(a)** G1 and G2 must hold both in-surface and on the held-out surface.
  - **(b)** G1 also requires trace - plain >= +0.25 on cond_oor_val alone. The tens-digit comparison scores 0.843 on
    the pooled conditional but about 0.5 on the boundary pairs. Revision 4 adds cond_sametens alone (R4-1).
  - **(c)** G3 uses the 4b filler.
  - **(d)** Accepted: the 5/6 rule-variant identification bar, the distance bands, 190 training steps, point-estimate
    margins with SE flags.
  - **(e)** The deviations from the joint template are listed in section 12.

## 1. Question

T2 trained 175 mix rows and got nothing: mix one-step accuracy was 0.022 before and 0.022 after. The conditional branch
choice is at chance (0.51) before and after T2, in all 3 seeds, while the in-branch arithmetic is right; OFF outputs the
unreduced sum. The joint finding (with BWM's RW2 `|2x| <= L` legality shortcut) is that computed-predicate rules are not
learned from bare-answer rows. This screen asks:

- Does supervising the intermediates in the answer (compare, then compute, then answer) teach these rules where bare
  answers did not, out of range AND on a held-out prompt surface, and does the comparison survive rows on which the
  tens digits do not decide (cond_sametens)?
- Is a gain supervision, or extra decode positions? The 4b filler arm is the control. It is answer-free and at least
  as long as the longest trace of its kind on every row.

## 2. Environment semantics (`rcwmt/env.py`, frozen; quoted, not changed)

State `{"X","Y","Z"}`: integers modulo 97. One instruction per step (`env.step`):

| kind | update |
|---|---|
| swap | `reg[r], reg[o] = reg[o], reg[r]` |
| add | `reg[r] = (reg[r] + v) % 97` |
| mix | `reg[r] = (reg[r] + 2*reg[o] + v) % 97` |
| conditional | `reg[r] = (reg[r] + (v if reg[o] > threshold else -v)) % 97` |

- The conditional predicate is STRICT: `reg[other] == threshold` subtracts.
- Python's `%` is non-negative: an underflow wraps up.
- "wrap": the raw value of the updated register lies outside [0, 96].
- distance d = reg[other] - threshold (ok iff d > 0).

## 3. Joint trace format and arms

- One JSON object with keys in LITERAL order cmp, calc, after, serialized with an explicit order (never `sort_keys`).
- cmp: conditional `{"x": reg[other], "thr": threshold, "ok": reg[other] > threshold}`; null for swap / add / mix.
- calc: `{"raw": unreduced value, "mod": raw % 97}`; null for swap.
- after: `env.dumps(state)`.

```
trace  {"cmp":{"x":35,"thr":34,"ok":true},"calc":{"raw":64,"mod":64},"after":{"X":35,"Y":64,"Z":49}}
plain  {"X":35,"Y":64,"Z":49}
filler {"fill":" . . . . . . . . . . . . . . . . . . . . . . . . . . . . .","after":{"X":35,"Y":64,"Z":49}}   (conditional: 29 units, every row)
```

- **Arms:** plain / trace / filler, on the same rows in the same order with the same (sd) prompt.
- **Parsing:** `parse_trace` / `parse_filler` reject any other key order, extra, missing or duplicate keys. The emitted
  fill's length is not scored (4b). The decision parser reads ONLY after (`rcwmt.env.parse_state(lenient=True)`); a
  malformed answer is wrong.

## 4. Training rows (unchanged rows; surface-diversity rendering)

| block | rows | design |
|---|---:|---|
| mix | 300 | 150 wrap / 150 no-wrap |
| wrap | 150 | add whose raw sum leaves [0, 96]: 75 over / 75 under |
| conditional, core | 150 | other = threshold - 1 / threshold / threshold + 1, 50 each |
| conditional, twins | 150 | one minimal-pair twin per core row, on the other side of the threshold at \|d\| 2..15; per core offset 25 move the threshold and 25 move the other register |

- Values -8..11; thresholds and registers 0..96. Swap and add/no-wrap are absent on purpose (the untouched controls).
- Mean target tokens: plain 15.55, trace 38.06, filler 42.35 (revision 3: 39.35).
- Files (`data/`): `t2br_train_sd_{plain,trace,filler}.jsonl` (the screen's training data) and
  `t2br_train_{plain,trace,filler}.jsonl` (the canonical rendering, never trained), with SOURCE records.
- Surface use, the C2 proof and the in-surface recombination property are unchanged from revision 3 (`DESIGN_rev3.md`
  section 4; `AUDIT.json` `surface.train_sd`).

**Audit** (`AUDIT.json`; exact counts; all pass; unchanged rows, so unchanged values):

| heuristic | accuracy | bar |
|---|---|---|
| copy | 0.000 | <= 0.55 |
| unreduced | all 0.467, mix 0.500, wrap 0.000 | <= 0.55 |
| always add / subtract, value > 0 | 0.500 | <= 0.55 |
| wrong register / other > 48 / threshold < 48 | 0.527 / 0.527 / 0.540 | <= 0.55 |
| rule variants `>=`, eq_plus1, units_eq_plus1, units_digit_gt, tens_digit_gt | 0.833 / 0.667 / 0.680 / 0.717 / 0.757 | <= 5/6 |
| digit shortcut tens_or_eq_plus1 (revision 4) | 0.917 (25 contradicting rows) | reported, NOT gated |
| digit shortcut tens_ge_not_eq (revision 4) | 0.803 (59 contradicting rows) | reported, NOT gated |

**Amendment 4b on the data** (`AUDIT.json` `fill_constant`; train + dev + chain gold steps):

| kind | rows | longest trace (tokens) | constant fill (units) | shortest filler (tokens) | revision 3's per-row value | residual (filler - trace tokens) |
|---|---:|---:|---:|---:|---:|---|
| swap | 475 | 26 | 7 | 26 | 5 | 2 on every row |
| add | 724 | 35 | 17 | 35 | 14 | 3..6, mean 4.32 |
| mix | 951 | 35 | 17 | 35 | 14 | 3..6, mean 3.46 |
| conditional | 1,336 | 47 | 29 | 47 | 26 | 3..8, mean 4.38 |

- Every cell has one fill string, so the fill length's best cut equals the majority rate: zero information.
- The residual is answer-dependent only through the trace's length. It is descriptive; the filler arm never emits it.
  Its best cut on the label: training mix 0.957, training conditional 0.527, mix_oor 0.97, cond_oor_val 0.57,
  cond_oor_thr 0.85, cond_far 0.72, cond_sametens 0.56. The analysis reports the filler side's accuracy split at each
  cell's median residual.

## 5. Dev screen bank (`data/t2br_dev.jsonl`), held-out-surface bank (`data/t2br_dev_hs.jsonl`), chains

The first 1,100 rows are unchanged from revision 2. cond_sametens (revision 4) is appended. The in-surface prompts are
Titan's verbatim frame.

| stratum | n | range | role | definition |
|---|---:|---|---|---|
| mix | 100 | in | non-inferiority (mix_in) | 50 wrap / 50 no-wrap |
| wrap | 80 | in | reported | add, raw outside [0, 96] |
| cond_xm1 / cond_x0 / cond_xp1 | 60 each | in | non-inferiority | 60 triplets: threshold = x + 1 / x / x - 1 |
| cond_mid | 100 | in | non-inferiority | 50 pairs at \|d\| 2..15 (25 move the threshold, 25 the other register) |
| swap | 80 | in | control | swap of unequal registers |
| add_nowrap | 80 | in | control | add, no wrap |
| mix_oor | 100 | OUT | PRIMARY (mix) | \|value\| 12..40; 50 wrap / 50 no-wrap |
| wrap_oor | 80 | OUT | reported | add, \|value\| 12..40, wraps |
| cond_oor_val | 100 | OUT | PRIMARY (conditional; G1 alone) | 50 boundary pairs: other = threshold (false) / other = threshold + 1 (true); \|value\| 12..40 |
| cond_oor_thr | 100 | OUT | PRIMARY (conditional) | 50 pairs: threshold -40..-1 (true) / 97..136 (false) |
| cond_far | 100 | OUT | PRIMARY (conditional) | 50 pairs at \|d\| 16..96, each moving the other register |
| **cond_sametens** | 100 | OUT | **IDENTIFICATION (G1 alone; not pooled)** | 50 minimal pairs, tens(other) == tens(threshold), \|d\| 2..9 (2..7 realized), \|value\| 12..40, no wrap on either branch; 25 move the threshold, 25 the other register |

**Held-out-surface bank** (500 rows, C3):
- It contains the rows of mix_oor, cond_oor_val, cond_oor_thr, cond_far and cond_sametens in dev order, with the same
  ids and golds.
- Row j is rendered in `t2br_common.heldout_surface(j)`. The orders rotate 167 / 167 / 166 over the 3 held-out state
  and event orders.
- Mean prompt tokens: 182.3 against 143.3 in-surface.
- Every prompt decodes to its row and uses ONLY held-out factors (`--verify`, the scorer self-test, the tiny test).

**Shortcut floors** (`AUDIT.json` `dev_heuristics`; branch shortcuts with exact arithmetic):

| shortcut | pooled conditional | cond_oor_val | cond_sametens |
|---|---|---|---|
| eq_plus1 | 0.667 | 1.0 | 0.5 |
| units-digit `>` | 0.677 | 0.97 | 1.0 |
| `>=` | 0.833 | 0.5 | 1.0 |
| tens-digit `>` | 0.843 | 0.53 | 0.5 |
| tens_or_eq_plus1 (revision 4) | 1.0 | 1.0 | 0.5 |
| tens_ge_not_eq (revision 4) | 1.0 | 1.0 | 0.5 |

Every single shortcut above fails at least one G1 part against a chance-level plain arm. Section 14 lists the digit
combinations that still pass.

**Chains** are unchanged byte for byte: 384 = CH_IN 32 + CH_X 16 parents, x 8 candidates, x 4 steps, in the chain_one
frame.

**Freshness:**
- Zero payload and zero operative-content overlap with PREREG-3's exclusion union. The harvest skips files by name: the
  `HSKILL_CONFIRM_V2*` files are skipped by name and never opened, and `rcwmt.bank.load_exclusions()` is not called.
- Units are mutually disjoint, cond_sametens included.

## 6. Arms, recipe, runner (`scripts/t2br_run.py`)

**Recipe:**
- the T2 recipe per arm: vendored BWM trainer; Titan-Sparse FP32; LoRA r16 / alpha 32 / dropout 0.05 on q, k, v, o,
  gate, up, down;
- 190 steps x batch 16; lr 1e-4 with WSD; max_len 512; micro 4; `train_generic`'s row order; seed 0;
- trained on `data/t2br_train_sd_<arm>.jsonl`.

**Runner:**
- one lease, `rcwm-t1:t2b-register`;
- D-22 stage gate inside the 4.5 h cap, re-checked after every resource / thermal wait;
- one-waiter rule, run lock, runtime pin, Sparse fingerprint, hook check;
- create-only outputs under `runs/t2b_register/s0_sd/`;
- adapters loaded unmerged (D-28).

Stages:
1. T_plain, T_trace, T_filler.
2. A0: the anchor. A resumed turn re-anchors.
3. E1: 1,200 in-surface rows x OFF / plain / trace / filler.
4. E1H: 500 held-out-surface rows x the same 4 sides. It runs before E2.
5. E2: 384 chains x 4 sides.

Exit codes: 0 done; 3 deterministic stop; 5 resumable deferral; 1 crash. Every text file the runner writes is LF-only.

## 7. Pre-stated go rule, revision 4 (verbatim; `scripts/t2br_common.GO_RULE_TEXT`, applied by `scripts/t2br_analysis.verdict`)

What changed from revision 3 and why: `GO_RULE.md` (preface) and section 0.1.

> PRE-STATED SCREEN GO RULE, REVISION 4 (T2b-register; supersedes revisions 1-3, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on cond_sametens. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. IDENTIFICATION stratum (G1 only; not pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it). FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone AND on cond_sametens alone, each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

**Components** (16, decided exactly, in `t2br_analysis.COMPONENTS`):

| condition | parts | surface |
|---|---|---|
| G1 | conditional; cond_oor_val; cond_sametens | in-surface and held-out (HS) |
| G2 | mix | in-surface and HS |
| G3 | conditional, mix | in-surface |
| G4 | swap, add_nowrap | in-surface |
| G5 | conditional_in, mix_in | in-surface |
| DP3 | conditional, mix | in-surface |

A NO-GO names the failed parts, for example `NO-GO (failed: G1 [cond_sametens, cond_sametens HS])`.

## 8. Reported, not gating

- Per stratum and side, on both renderings: accuracy (Wilson CI), parse rate and cluster-bootstrap contrasts.
- **Surface audit:** held-out minus in-surface, per side and held-out-surface stratum, paired by row, with a CI and
  the discordant row counts.
- Every component with its exact value, its bootstrap SE and a flag when it lies within 1 SE of its bar.
- Trace fields (format_ok, cmp_ok, ok_ok, calc_ok, after_follows_calc, calc_follows_cmp) on both renderings.
- Filler residual covariate, branch choice read from after, unreduced-sum answers.
- Controls: trace - OFF (gating), trace - plain, plain - OFF and filler - OFF.
- Chains: regret and exact state at k = 1, 2, 4, the copy floor and the per-step product law (canonical frame only).

## 9. GPU plan (`docs/t2b/register/GEN_PLAN.json`; `t2br_run.py --plan`, CPU, no torch)

| stage | planning (2.7 gens/s) | observed (3.8 gens/s) | generations |
|---|---:|---:|---:|
| T_plain | 37.7 min | 37.7 min | 534,460 training tokens |
| T_trace | 42.2 min | 42.2 min | 602,919 tokens |
| T_filler | 43.1 min | 43.1 min | 615,944 tokens |
| A0 | 7.1 min | 5.6 min | 832 |
| E1 | 52.9 min | 37.6 min | 4,800 |
| E1H | 23.3 min | 16.5 min | 2,000 |
| E2 | 67.7 min | 48.1 min | 6,144 |

- Training is T2's measured 1,858.8 s scaled by processed tokens.
- Trace and filler decode 2.45 / 2.69 x longer than plain in E1, and 2.65 / 2.89 x in E1H.
- **Generation bound.** 13,776 generations with the anchor counted ONCE. This excludes a resumed turn's re-anchor (832
  generations each).
  - At the planning rate (one re-anchor): 14,608.
  - At the observed rate (no re-anchor): 13,776.
  - Hook bound (3 calls, at most 2 re-anchors): 15,440.
  - Revision 3's "at most 12,976" also excluded its re-anchor; with it, it was 13,808.

**Turns** (cap 4.5 h; no stage starts within 0.5 h of the cap; 1.25 x its estimate must fit):

| rate | turn 1 | turn 2 | total |
|---|---|---|---|
| planning | T_plain, T_trace, T_filler, A0, E1, E1H: 3.44 h | A0 re-anchor + E2: 1.25 h | 4.69 GPU-h |
| observed | all stages: 3.85 h | - | 3.85 GPU-h |

- At the planning rate every gating output (E1, E1H) is produced in turn 1.
- The split is automatic: the runner exits 5 when the gate refuses E2. The hook's next call resumes after a re-anchor.
- The hook allows up to 3 calls per chain pass.

## 10. Tests and records (`docs/t2b/register/`; all rerun for revision 4)

- **AUDIT.json** (`t2br_make_data.py --verify`, full historical harvest): 30/30 checks pass. New in revision 4:
  - the revision-3 reconstruction (eight files byte for byte);
  - the append check (dv0000..dv1099 in revision-3 order, then cond_sametens; the HS bank's first 400 rows are
    revision 3's);
  - the cond_sametens stratum and pair checks;
  - the 4b constant definition (shortest filler >= longest trace).
- **SCORER_SELFTEST.json** (`t2br_analysis.py --selftest`): pass on 1,200 dev rows and 500 held-out-surface rows;
  22,800 malformed answers rejected; 3,600 fills of other lengths accepted; exact identities incl. cond_sametens
  (always-subtract = 1/2).
- **DRYRUN.json / DRYRUN.md** (`t2br_analysis.py --dryrun`): 19 synthetic runs (13 scenarios + 6
  engineered exact-boundary runs) and 20 property checks, all pass. They include:
  - the verifier's shortcut and its mirror;
  - the digit-wise true rule;
  - boundary runs on cond_sametens in-surface and HS;
  - exhaustive at-bar checks of the new clause on both surfaces;
  - the digit-shortcut enumeration (section 14).
- **TINY_TEST.json** (`t2br_test_tiny.py`, tiny random Qwen3 and the real vendored trainer): 29/29 checks pass.
  New in revision 4:
  - with AutoTokenizer, the shortest filler target of every kind >= its longest trace target, and one unit less fails;
  - the 500-row held-out bank;
  - the oracle surface-locked verdict names cond_sametens HS;
  - the freeze includes the revision-3 rule and design.
- **GEN_PLAN.json**, **queue_hook_t2br.sh.draft** (NOT installed), **GO_RULE.md**, **GO_RULE_rev1/2/3.md**,
  **DESIGN_rev1/2/3.md**: all are frozen inputs (`t2br_run.py --freeze`).

## 11. Decision points

- **DP1:** fired. The sd build is the screen.
- **DP3:** trace - filler < 0.10 on both in-surface primaries means STOP, with the section 7 wording.
- **GO:** seeds 1-2 (`--seed 1/2`). The roadmap then requires:
  - the same rule in all 3 seeds;
  - a sealed R bank with held-out-surface primaries and a same-tens identification stratum (D-35);
  - k4 regret <= -0.10.
- **Freeze and install** are operator steps, logged in DECISIONS.md: freeze; install the hook as
  `scripts/queue_hooks/80_t2br.sh`; launch a fresh chain at a safe point.

## 12. Deviations from the joint template (reported to BWM)

1. `cmp.ok` uses the env's STRICT `>`.
2. calc is the object `{"raw", "mod"}`.
3. cmp is `null` for add, mix and swap; calc is `null` for swap.
4. Duplicate keys are rejected.
5. Further reported items:
   - the reworded DP3 label;
   - G3 at +0.10 per in-surface primary;
   - the validity gates;
   - G1's cond_oor_val clause;
   - amendment 4b as implemented here: a per-kind constant over train + dev + the chains' gold steps, the shortest
     filler target against the longest trace target (revision 4; revision 3's code used a per-row maximum);
   - G1's cond_sametens clause (revision 4).

## 13. Provenance and interruption recovery

**Revision 3's interruption recovery (C0)** is unchanged: `DESIGN_rev3.md` section 13.

**Revision 4 data rebuild.**
- The twelve revision-3 data files were moved (renamed, not deleted) to `data/t2br_superseded/rev3/`, with their
  sha256 in `data/t2br_superseded/rev3_SHA256SUMS.txt`. Pre-edit copies of every kit file are in the session
  scratchpad.
- The rows were not re-drawn:
  - `--verify` reconstructs the four revision-2 files (dev `2c0250a2…`, plain `49067515…`, trace `1bad8116…`, filler
    `3102361d…`) and the eight revision-3 files (dev `503d7686…`, held-out `1ced9b71…`, train plain `058acce8…`, trace
    `81f4af02…`, filler `726134cd…`, sd plain `4c679f35…`, sd trace `941c1408…`, sd filler `e99f4f5f…`) byte for byte
    from the current rows.
  - The chain bank is byte-identical (`fe15c2b7…`).
- The current builder rebuilds every data file byte for byte from a fresh name-skipping harvest.
- What changed in the files:
  - **Dev bank:** 100 appended cond_sametens rows. On every row, the filler target, fill_units, fill_residual and the
    filler token count changed (4b constants). The other fields are byte-identical.
  - **Held-out-surface bank:** 100 appended rows. The first 400 rows are byte-identical.
  - **Training files:** the filler targets changed in the filler files; fill_units and fill_residual changed in all six
    files. Prompts, plain / trace targets, ids and golds are byte-identical.

## 14. Notes and limitations

- **Held-out state key orders are the three odd permutations.**
  - Held out: `Z/Y/X` (the reversal), `Y/X/Z` and `X/Z/Y`. These are exactly the three odd permutations of X/Y/Z
    (single transpositions).
  - Training: exactly the three cyclic rotations `X/Y/Z`, `Y/Z/X`, `Z/X/Y`.
  - Every training order has the cyclic orientation X -> Y -> Z -> X, and every held-out order has the reverse
    orientation. No held-out order is a rotation of a training order.
  - This is a harder-than-random hold-out: of the 6 possible draws of 2 extra orders from the 4 non-canonical,
    non-reversed ones, only 1 gives all-odd. It came out of the seeded draw (seed 20261113); it was not chosen.
  - A loss on the held-out surface may therefore partly reflect this orientation flip, on top of the paraphrase and
    names.
- **Generation bound.** Revision 3's "at most 12,976 generations" counted A0 once and excluded a resumed turn's
  re-anchor; with it, it was 13,808. Revision 4: 13,776 with A0 once; 14,608 with the projected re-anchor at the
  planning rate; 15,440 at the hook bound (section 9).
- **Per-factor held-out renderings are omitted BY CHOICE** (a GPU-budget choice, not a capacity limit).
  - Three more renderings (paraphrase only, names only, key order only) of the 500 rows x 4 sides cost about 70 min at
    the planning rate (6,000 generations).
  - At the planning rate they would fit in turn 2 (1.25 h used of 4.5 h). At the observed rate they would need a
    second turn and a re-anchor.
  - The operator chose not to spend GPU time on descriptive renderings. Consequence: a held-out-surface loss cannot be
    attributed to paraphrase, names or key order.
- **Digit shortcuts that still pass G1** (`DRYRUN.json` `shortcut_enumeration`; reported, not fixed).
  - The 6 classes (12 predicates) listed in `GO_RULE.md` pass against a chance-level plain arm. They are
    non-lexicographic digit-wise comparisons such as `tens(x)>tens(t) or units(x)>units(t)` (259/300 pooled, 1 on
    cond_oor_val and cond_sametens).
  - They err only where the tens and units comparisons disagree across decades: the ok-false member of a far or
    high-threshold pair whose units digit is larger, or the ok-true member whose units digit is smaller.
  - Against the dry run's lower plain, 12 classes (33 predicates) pass, including `units(x)>units(t)` alone.
  - A "cross-digit" identification stratum would catch all of them: branch-balanced rows at |d| >= 2 in different
    decades whose units comparison disagrees with the tens comparison. Checked on every such (other, threshold) in
    0..96 (3,978 rows): each of the 33 still-passing predicates scores 1/2 or 0 there, and the true rule scores 1. It is
    NOT added: the operator selected only cond_sametens.
  - The per-row outputs are kept, so whether a trace arm's errors on cond_far / cond_oor_thr concentrate on such rows
    can be checked post hoc (exploratory, not gating).
- **cond_sametens.** Distances are skewed small (2..7, mostly 2..4) by the one-decade minimal-pair construction. Value
  signs are 62 / 38. Decades 4 and 9 are not drawn. Neither branch wraps, so the stratum does not test the modular
  reduction.
- **Top level.** The 2-key top-level object trains in the canonical order only. Its held-out order is a factor never
  varied in training.
- **One combined rendering.** The held-out surface is ONE combined rendering.
- **Frame.** Chains, A0 and the controls run in Titan's frame only.
- **Residual.** The filler has 2..8 more tokens than the trace on every row: never fewer decode positions.

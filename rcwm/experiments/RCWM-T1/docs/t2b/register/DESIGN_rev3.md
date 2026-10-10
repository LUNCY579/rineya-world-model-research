# T2b-register screen, revision 3: intermediates in the answer, with an answer-free filler control

RCWM roadmap item 3 (`docs/roadmap/RCWM_ROADMAP_2026-10-08.md`); joint roadmap item 5b with BWM-001
(`JOINT_ROADMAP_2026-10-08.md`, BWM D-52); DECISIONS D-34, D-35, R16, D-36.

**Status:** exploratory screen. CPU-only kit; not frozen; no GPU run; no model output exists; the queue hook is a draft
and is NOT installed. The GPU job refuses to run until `python -B scripts/t2br_run.py --freeze --approval "..."` has
recorded the operator's go-ahead (a screen lock, not a preregistration).

**Revisions** (each written before any `runs/t2b_register` output existed; the superseded ones are kept verbatim):

| revision | when (2026-10-09) | design | go rule |
|---|---|---|---|
| 1 | about 03:04, first build | `DESIGN_rev1.md` | `GO_RULE_rev1.md` |
| 2 | about 07:54, after the adversarial review | `DESIGN_rev2.md` | `GO_RULE_rev2.md` |
| 3 | this document: operator decisions after PREREG-3 (D-36) | `DESIGN.md` | `GO_RULE.md` (with the "what changed and why" preface) |

## 0. Operator decisions applied in revision 3 (binding; each with its reason)

- **C1. The primary build is the surface-diversity variant 'sd'; the canonical-only variant is not run.**
  - Reason: decision point 1 has fired. PREREG-3 returned STATUS: TRIGGER (REPORT section 3c): T2's void no-op is a
    token cue, and it fails on reordered JSON keys (0.37-0.47 against 0.87-0.91 in format), on word account names and
    on magnitudes.
  - Implementation:
    - The SAME 750 canonical rows (never re-drawn: the build asserts the revision-2 row digests, and `--verify`
      reconstructs the four revision-2 files byte for byte from the current rows) get one seeded surface per row: a
      rule-text paraphrase, a field-name variant per field, and key orders (C2).
    - `t2br_run.py` refuses any variant other than `sd` (exit 3, not persisted).
    - The canonical rendering `data/t2br_train_<arm>.jsonl` is kept as the rows' source of truth. It is never
      trained.
- **C2. Input key-order augmentation from a TRAIN subset of permutations; held-out orders never in training.**
  - Reason: R16, agreed with BWM (their D-52 / D-54). Both LoRA world models are locked to key order but invariant to
    names, so key order must be augmented in the input and tested on orders never seen in training.
  - Every JSON object of the prompt is serialized in a key order drawn per row from the TRAIN subset of its type:
    - the top level `{before, event}`;
    - the state `{X, Y, Z}`;
    - the event `{kind, other, register, threshold, tick, value}`.
  - Held out (`t2br_common.HELDOUT_ORDERS`, regenerated from seed 20261113 and checked at import): per type, the
    fully reversed canonical order plus 2 fixed permutations:

    | object | held-out orders | training orders |
    |---|---|---|
    | top | `event/before` | 1: the 2-key object has only its reversed order to hold out, so it trains in the canonical order alone |
    | state | `Z/Y/X`, `Y/X/Z`, `X/Z/Y` | 3 |
    | event | `value/tick/threshold/register/other/kind`, `other/kind/tick/threshold/register/value`, `threshold/kind/tick/other/value/register` | 717 |

  - Paraphrase 4 and field-name variant 4 of every field are held out too: `dest`, `peer`, `offset`, `pivot`,
    `opcode`, `memory`, `operation`.
  - The register prompt contains no lists, so there is no list-order factor.
  - Targets keep the literal `cmp -> calc -> after` order. The parser rejects any other order.
  - Proven on the files:
    - `--verify` decodes every training prompt with `t2br_common.read_prompt` and also runs a raw-string check (see
      section 4).
    - The tiny test repeats the decode and adds a negative control: the detector does flag a held-out rendering of the
      same rows.
- **C3. One held-out-surface stage, E1H.**
  - It evaluates the PRIMARY strata in a combined held-out rendering: the 400 rows of mix_oor, cond_oor_val,
    cond_oor_thr and cond_far, rendered with paraphrase 4 + names 4 + a held-out key order per object type. The orders
    rotate by row: the state and event orders cycle j mod 3, and the top level always uses its one held-out order.
  - Same rows and golds as the in-surface strata, paired by row (`data/t2br_dev_hs.jsonl`).
  - No in-distribution stratum enters a held-out-surface condition, so none is rendered.
  - All four sides are evaluated: plain and trace gate G1 / G2; OFF and filler give the surface audit of every side.
  - Per-factor held-out renderings (paraphrase only, names only, key order only) are NOT run. Each would cost about
    one more E1H (18 min at the planning rate). The plan already needs a second turn at the planning rate (section
    9), so they do not fit the GPU budget.
- **C4. Amendment 4b filler (proposed to BWM, adopted; BWM D-55).**
  - The fill is a run of `" ."` of a CONSTANT length per action kind (`t2br_common.FILL_UNITS`): swap 5, add 14, mix
    14, conditional 26 units. Each is the shortest run whose filler target is at least as long, in Titan tokens, as
    every trace target of that kind over train + dev (one-step rows and the chains' gold steps).
  - Targets do not depend on the prompt surface, so this covers every surface variant. The build re-derives the
    constants from the data and fails on any other value.
  - Per-row length matching (amendment 4) is removed.
  - Reason: the per-row matched fill leaked the answer through its length. Fill length alone predicted wrap at 0.957
    (training mix) and 0.970 (dev mix_oor), and the branch at 0.57 / 0.85 / 0.72 on the out-of-range conditional
    strata. That is a hidden scratch channel.
  - Test (AUDIT and the analysis pre-check): every filler target of a kind carries the identical fill, so the fill
    length's best single cut equals the majority rate in every cell. Zero information.
  - The residual (filler tokens minus that row's trace tokens, always at least 0) is reported as a descriptive
    covariate (section 4).
- **C5. Fits the 4.5 h turn cap with the existing deferral / resume.**
  - The plan needs 2 turns at the registered planning rate (3.24 h + 1.21 h) and 1 turn at the observed rate (3.68 h).
  - `--plan`, `GEN_PLAN.json` and the hook draft are updated. The hook draft is fixed to variant sd and allows up to 3
    calls per chain pass.
  - Nothing is installed or frozen.
- **C6. The rule is a new revision (3) with a "what changed and why" preface (`GO_RULE.md`).**
  - Revisions 1 and 2 are kept verbatim, renamed, never deleted.
  - Revision 3 was stated before any `runs/` output existed (none exists).
- **C0. Interruption recovery.** See section 13.
- **Kit 5b decisions:**
  - **(a)** G1 and G2 must hold BOTH in-surface and in the combined held-out-surface rendering. Each is decided
    exactly on counts.
  - **(b)** G1 additionally requires trace - plain >= +0.25 on cond_oor_val ALONE, on both surfaces.
    - Reason: a coarse tens-digit comparison scores 0.843 on the pooled primary conditional, because every far pair
      and every out-of-range-threshold pair is separated by the tens digits. It would pass the pooled clause without
      computing the boundary.
    - The dry run's `coarse_comparison` scenario passes the pooled clause (+0.397 in-surface) and fails this one
      (+0.12).
  - **(c)** G3 (trace - filler >= +0.10 on each in-surface primary) uses the 4b filler. The fill_exact diagnostics of
    revision 2 are dropped in favour of the constant-length check.
  - **(d)** Accepted as revised:
    - the 5/6 rule-variant identification bar (0.55 is infeasible, section 4);
    - the distance bands: twins |d| 2..15, cond_far 16..96;
    - 190 training steps (T2 recipe parity; 4.05 epochs over 750 rows, where T2 had 2.03);
    - point-estimate margins with SE flags.
  - **(e)** The documented deviations from the joint template are kept and listed in section 12 as reported to BWM.

## 1. Question

T2 trained 175 mix rows and got nothing: mix one-step accuracy was 0.022 before and 0.022 after. The conditional branch
choice is at chance (0.51) before and after T2, in all 3 seeds, while the in-branch arithmetic is right; OFF outputs the
unreduced sum. The joint finding (with BWM's RW2 `|2x| <= L` legality shortcut) is that computed-predicate rules are not
learned from bare-answer rows. This screen asks:

- Does supervising the intermediates in the answer (compare, then compute, then answer) teach these rules where bare
  answers did not, out of range AND on a held-out prompt surface?
- Is a gain supervision, or extra decode positions? The 4b filler arm is the control. It is answer-free and at least
  as long as the trace on every row.

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

- One JSON object with keys in LITERAL order cmp, calc, after. It is serialized with an explicit order, never
  `sort_keys`.
- cmp:
  - conditional: `{"x": reg[other], "thr": threshold, "ok": reg[other] > threshold}`;
  - null for swap / add / mix.
- calc: `{"raw": unreduced value, "mod": raw % 97}`; null for swap.
- after: `env.dumps(state)`.

```
trace  {"cmp":{"x":35,"thr":34,"ok":true},"calc":{"raw":64,"mod":64},"after":{"X":35,"Y":64,"Z":49}}
plain  {"X":35,"Y":64,"Z":49}
filler {"fill":" . . . . . . . . . . . . . . . . . . . . . . . . . .","after":{"X":35,"Y":64,"Z":49}}   (conditional: 26 units, every row)
```

- **Arms:** plain / trace / filler. All three use the same rows, in the same order, with the same (sd) prompt.
- **Parsing:**
  - `parse_trace` rejects any other key order, extra keys, missing keys and duplicate keys.
  - `parse_filler` does the same for `["fill","after"]`. The emitted fill's length is not scored (4b): a filler answer
    with a fill of any length parses, and its after is read.
  - The decision parser reads ONLY after, through `rcwmt.env.parse_state(lenient=True)`. A malformed answer is wrong.

## 4. Training rows (unchanged rows; surface-diversity rendering)

| block | rows | design |
|---|---:|---|
| mix | 300 | 150 wrap / 150 no-wrap |
| wrap | 150 | add whose raw sum leaves [0, 96]: 75 over / 75 under |
| conditional, core | 150 | other = threshold - 1 / threshold / threshold + 1, 50 each |
| conditional, twins | 150 | one minimal-pair twin per core row, on the other side of the threshold at \|d\| 2..15; per core offset 25 move the threshold and 25 move the other register |

- Values are -8..11; thresholds and registers are 0..96 (the recorded CANDIDATE_VIEWS marginals).
- Swap and add/no-wrap are absent on purpose: they are the untouched-skill controls.
- Mean target tokens: plain 15.55, trace 38.06, filler 39.35 (4b: the filler is at least as long as the trace).

**Files** (`data/`):
- `t2br_train_sd_{plain,trace,filler}.jsonl`: the screen's training data. Same targets, ids and golds as the
  canonical rendering, and identical prompts across the arms.
- `t2br_train_{plain,trace,filler}.jsonl`: the canonical rendering, the rows' source of truth (never trained).
- SOURCE records for each.

**Surface use** in the 750 training prompts (`AUDIT.json` `surface.train_sd`):

| factor | use |
|---|---|
| paraphrases 0 / 1 / 2 / 3 | 172 / 181 / 196 / 201 rows |
| field-name variants 0-3 | 165-217 rows each, every variant of every field |
| state orders | `X/Y/Z` 270, `Y/Z/X` 246, `Z/X/Y` 234 |
| event orders | 464 distinct, of the 717 training orders |
| top level | canonical on every row |

- Mean prompt tokens: 159.3 (sd) against 142.3 (canonical).
- **C2 proof on the files** (`--verify` and the tiny test):
  - every sd prompt decodes to its row and recorded surface;
  - 0 rows use a held-out paraphrase, field name or key order;
  - 0 prompts contain a held-out name as a JSON key or a fragment of the held-out paraphrase;
  - every training paraphrase, name variant and state order is used.
- **Property of the in-surface evaluation frame.** Titan's verbatim frame (paraphrase 0, names 0, canonical orders)
  uses only training-subset factor values. With independent per-row draws, however, the exact combination appears on
  0 training rows: paraphrase 0 with every name at 0 occurs 0 times, and the canonical event order 3 times. So the
  in-surface strata also test recombination of trained factor values. The held-out surface tests factor values never
  trained.

**Audit** (`AUDIT.json`; exact counts; all pass).

| heuristic | accuracy | bar |
|---|---|---|
| copy | 0.000 (all, mix, wrap, cond) | <= 0.55 |
| unreduced | all 0.467, mix 0.500, wrap 0.000 | <= 0.55 |
| always add / always subtract | 0.500 / 0.500 | <= 0.55 |
| wrong register | 0.527 | <= 0.55 |
| other > 48 | 0.527 | <= 0.55 |
| threshold < 48 | 0.540 | <= 0.55 |
| value > 0 | 0.500 | <= 0.55 |
| rule variant `>=` | 0.833 | contradicted by >= 1/6 of the conditional rows (accuracy <= 5/6) |
| rule variant eq_plus1 | 0.667 | as above |
| rule variant units_eq_plus1 | 0.680 | as above |
| rule variant units_digit_gt | 0.717 | as above |
| rule variant tens_digit_gt | 0.757 | as above |

- The rule-variant bar is 5/6, not 0.55, because 0.55 is infeasible:
  - acc(eq_plus1) = P(ok false) + P(other = threshold + 1) >= 0.5 + 50/N, so 0.55 needs N >= 1,000 conditional rows;
  - acc(`>=`) = 1 - P(other = threshold).
- The best single-feature stump reaches 0.547, inside the permutation null's 95th percentile (reported).

**Amendment 4b on the data** (`AUDIT.json` `fill_constant`).

| kind | longest trace (Titan tokens) | constant fill (units) | residual (filler - trace tokens) |
|---|---:|---:|---|
| swap | 26 | 5 | 0 on every row |
| add | 35 | 14 | mean 1.32, max 3 (> 0 on 95% of rows) |
| mix | 35 | 14 | mean 0.46, max 3 (> 0 on 43%) |
| conditional | 47 | 26 | mean 1.37, max 5 (> 0 on 96%) |

- Every cell has one fill string, so the fill length's best cut equals the majority rate: zero information.
- What the RESIDUAL would tell about the label is descriptive only. It is answer-dependent through the trace's length;
  the filler arm never emits it.

  | cell | label | residual best cut |
  |---|---|---|
  | training mix | wrap | 0.957 |
  | training conditional | branch | 0.527 |
  | mix_oor | wrap | 0.97 |
  | cond_oor_val | branch | 0.57 |
  | cond_oor_thr | branch | 0.85 |
  | cond_far | branch | 0.72 |

- The analysis reports the filler side's accuracy split at each primary cell's median residual.

## 5. Dev screen bank (`data/t2br_dev.jsonl`), held-out-surface bank (`data/t2br_dev_hs.jsonl`), chains

The rows are unchanged from revision 2. The in-surface prompts are Titan's verbatim frame.

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
| cond_oor_val | 100 | OUT | PRIMARY (conditional; G1's second clause alone) | 50 boundary pairs: other = threshold (false) / threshold - 1 (true); \|value\| 12..40 |
| cond_oor_thr | 100 | OUT | PRIMARY (conditional) | 50 pairs: threshold -40..-1 (true) / 97..136 (false) |
| cond_far | 100 | OUT | PRIMARY (conditional) | 50 pairs at \|d\| 16..96, each moving the other register |

**Held-out-surface bank** (400 rows, C3):
- It contains the primary rows in dev order, with the same ids and golds.
- Row j is rendered in `t2br_common.heldout_surface(j)`. The orders rotate 133 / 133 / 134 over the 3 held-out state
  and event orders.
- Mean prompt tokens: 182.2 against 143.2 in-surface.
- Every prompt decodes to its row and uses ONLY held-out factors (`--verify`, the scorer self-test, the tiny test).

**Shortcut floors on the pooled primary conditional:**

| shortcut | score |
|---|---|
| eq_plus1 | 0.667 |
| threshold < 48 | 0.667 |
| wrong register | 0.670 |
| units-digit `>` | 0.677 |
| `>=` | 0.833 |
| tens-digit `>` | 0.843 (on cond_oor_val alone it is near 0.5: G1's new clause) |

**Chains** are unchanged: 384 = CH_IN 32 + CH_X 16 parents, x 8 candidates, x 4 steps, in the chain_one frame.

**Freshness:**
- Zero payload and zero operative-content overlap with PREREG-3's exclusion union (name-skipping harvest: the
  `HSKILL_CONFIRM_V2*` files are skipped by name and never opened; `rcwmt.bank.load_exclusions()` is not called).
- Units are mutually disjoint.
- Surface renderings do not change payloads.

## 6. Arms, recipe, runner (`scripts/t2br_run.py`)

**Recipe:**
- the T2 recipe per arm: vendored BWM trainer; Titan-Sparse FP32; LoRA r16 / alpha 32 / dropout 0.05 on q, k, v, o,
  gate, up, down;
- 190 steps x batch 16; lr 1e-4 with WSD; max_len 512; micro 4;
- `train_generic`'s row order; seed 0;
- trained on `data/t2br_train_sd_<arm>.jsonl`.

**Runner:**
- one lease, `rcwm-t1:t2b-register`;
- D-22 stage gate inside the 4.5 h cap, re-checked after every resource / thermal wait; a larger `--cap-h` is refused;
- one-waiter rule, run lock, runtime pin, Sparse fingerprint, hook check;
- create-only outputs under `runs/t2b_register/s0_sd/`;
- adapters loaded unmerged (D-28).

Stages:
1. T_plain, T_trace, T_filler.
2. A0: the anchor. A resumed turn re-anchors.
3. E1: 1,100 in-surface rows x OFF / plain / trace / filler.
4. **E1H: 400 held-out-surface rows x the same 4 sides.** New in revision 3. It runs before E2, so the gating
   outputs come first.
5. E2: 384 chains x 4 sides.

Exit codes: 0 done; 3 deterministic stop; 5 resumable deferral; 1 crash. Every text file the runner writes is LF-only.

## 7. Pre-stated go rule, revision 3 (verbatim; `scripts/t2br_common.GO_RULE_TEXT`, applied by `scripts/t2br_analysis.verdict`)

What changed from revision 2 and why: `GO_RULE.md` (preface) and section 0.

> PRE-STATED SCREEN GO RULE, REVISION 3 (T2b-register; supersedes revisions 1 and 2, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. FILLER (amendment 4b): the fill is a run of ' .' of a CONSTANT length per action kind (swap 5, add 14, mix 14, conditional 26 units: the shortest run whose filler target is at least as long as every trace target of that kind in train + dev), so every filler target of a kind is identical up to 'after' and its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional AND >= +0.25 on cond_oor_val alone, each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and primary stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings are not run (GPU budget). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

**Components** (14, decided exactly, in `t2br_analysis.COMPONENTS`):

| condition | parts | surface |
|---|---|---|
| G1 | conditional; cond_oor_val | in-surface and held-out (HS) |
| G2 | mix | in-surface and HS |
| G3 | conditional, mix | in-surface |
| G4 | swap, add_nowrap | in-surface |
| G5 | conditional_in, mix_in | in-surface |
| DP3 | conditional, mix | in-surface |

A NO-GO names the failed parts, for example `NO-GO (failed: G1 [conditional HS, cond_oor_val HS]; G2 [mix HS])`.

## 8. Reported, not gating

- Per stratum and side, on both renderings: accuracy (Wilson CI), parse rate and cluster-bootstrap contrasts.
- **Surface audit:** held-out minus in-surface per side and primary stratum, paired by row, with a CI and the
  discordant row counts.
- Every component with its exact value, its bootstrap SE and a flag when it lies within 1 SE of its bar. The G4 / G5
  margins are about 1 SE at n = 80-100; the operator accepted point estimates with flags.
- Trace fields (format_ok, cmp_ok, ok_ok, calc_ok, after_follows_calc, calc_follows_cmp), on both renderings.
- **Filler residual covariate:** the filler side's accuracy at or below / above the median residual, on both
  renderings.
- Branch choice read from after.
- Unreduced-sum answers.
- Controls: trace - OFF (gating), trace - plain, plain - OFF and filler - OFF. "add => wrap" is 1.0 on the training
  adds by design.
- Chains: regret and exact state at k = 1, 2, 4, the copy floor and the per-step product law. Chains run in the
  canonical frame only.

## 9. GPU plan (`docs/t2b/register/GEN_PLAN.json`; `t2br_run.py --plan`, CPU, no torch)

| stage | planning (2.7 gens/s) | observed (3.8 gens/s) | generations |
|---|---:|---:|---:|
| T_plain | 37.7 min | 37.7 min | 534,460 training tokens |
| T_trace | 42.2 min | 42.2 min | 602,919 tokens |
| T_filler | 42.5 min | 42.5 min | 606,824 tokens |
| A0 | 7.1 min | 5.6 min | 832 |
| E1 | 46.9 min | 33.3 min | 4,400 |
| E1H | 18.0 min | 12.8 min | 1,600 |
| E2 | 65.4 min | 46.5 min | 6,144 |

- Training is T2's measured 1,858.8 s scaled by processed tokens. The sd prompts are longer than the canonical ones,
  and the 4b filler is longer than the trace.
- Trace and filler decode 2.42 / 2.48 x longer than plain in E1, and 2.62 / 2.68 x in E1H.
- At most 12,976 generations in total.

**Turns** (cap 4.5 h; no stage starts within 0.5 h of the cap; 1.25 x its estimate must fit):

| rate | turn 1 | turn 2 | total |
|---|---|---|---|
| planning | T_plain, T_trace, T_filler, A0, E1, E1H: 3.24 h | A0 re-anchor + E2: 1.21 h | 4.45 GPU-h |
| observed | all stages: 3.68 h | - | 3.68 GPU-h |

- At the planning rate every gating output (E1, E1H) is produced in turn 1. E2 (chains, reported only) follows in
  turn 2 after the re-anchor.
- The split is automatic: the runner exits 5 when the gate refuses E2. The hook's next call waits for the next RCWM
  turn and resumes.
- The hook allows up to 3 calls per chain pass, enough for a slower-than-planned run that needs a third turn.

## 10. Tests and records (`docs/t2b/register/`)

- **AUDIT.json** (`t2br_make_data.py --verify`):
  - golds, targets, token counts;
  - strata, pairs and twins;
  - rows identical to revision 2; the four revision-2 files reconstructed byte for byte;
  - 4b constants and the information test;
  - the C2 / C3 surface proof;
  - the rebuild check with a fresh name-skipping harvest;
  - freshness, audit and dev floors.
- **SCORER_SELFTEST.json** (`t2br_analysis.py --selftest`):
  - golds are 1.0 on every side, in-surface and on the held-out-surface index;
  - the held-out bank decodes;
  - the 4b constant-fill check;
  - malformed answers are rejected, and fills of other lengths are accepted (length not scored);
  - floors and exact identities.
- **DRYRUN.json / DRYRUN.md** (`t2br_analysis.py --dryrun`):
  - 10 synthetic scenarios, including the new `surface_locked` (a trace arm right only in Titan's frame) and
    `coarse_comparison` (tens digits);
  - 4 engineered exact-boundary runs, including `boundary_hs_cond_oor_val`;
  - exhaustive at-bar checks on both surfaces, including the new G1 cond_oor_val clause.
- **TINY_TEST.json** (`t2br_test_tiny.py`): tiny random Qwen3 and the real vendored trainer:
  - formats and 4b;
  - the C1 / C2 / C3 surface proof;
  - a full run with E1H;
  - crash and resume; deferral, including an E1H deferral and resume;
  - stops;
  - oracle runs (GO, DP3, surface-locked NO-GO);
  - the freeze tool;
  - the plan.
- **GEN_PLAN.json**, **queue_hook_t2br.sh.draft** (NOT installed), **GO_RULE.md** and **GO_RULE_rev1/2.md**,
  **DESIGN_rev1/2.md**: all are frozen inputs (`t2br_run.py --freeze`).

## 11. Decision points

- **DP1:** fired. The sd build is the screen.
- **DP3:** trace - filler < 0.10 on both in-surface primaries means STOP, with the section 7 wording.
- **GO:** seeds 1-2 (`--seed 1/2`). The roadmap then requires the same rule in all 3 seeds, a sealed R bank with
  held-out-surface primaries (D-35), and k4 regret <= -0.10.
- **Freeze and install** are operator steps, logged in DECISIONS.md:
  - freeze;
  - install the hook as `scripts/queue_hooks/80_t2br.sh`;
  - launch a fresh chain at a safe point.

## 12. Deviations from the joint template (reported to BWM)

1. `cmp.ok` uses the env's STRICT `>`. The joint template's `v >= t` is replaced by the register semantics, as the
   template instructs.
2. calc is the object `{"raw", "mod"}`, not a scalar. raw exposes the unreduced sum that OFF stops at.
3. cmp is `null` for add, mix and swap (no branch predicate), as BWM's `finish`. calc is `null` for swap.
4. Duplicate keys are rejected. This is stricter than BWM's `traces.parse_trace`, which keeps the last duplicate.
5. Further reported items:
   - the reworded DP3 label;
   - G3 at +0.10 per in-surface primary;
   - the validity gates (format_ok >= 0.95 on each primary);
   - amendment 4b as implemented here: constant per action kind over train + dev, chains' gold steps included;
   - G1's cond_oor_val clause.

## 13. Interruption recovery (C0)

A first attempt at this revision was cut off by a usage limit, and the machine was rebooted at 09:42. The state found
on resumption (12:05) was consistent but incomplete:
- `t2br_common.py` (08:14) and `t2br_make_data.py` (08:22) were revised and matched the data rebuilt at 08:24. A
  `--no-historical` AUDIT passed 25/25.
- `t2br_run.py` (08:27) had the E1H stage.
- `t2br_analysis.py` and `t2br_test_tiny.py` were still revision 2: they referenced the removed per-row filler and
  would have failed.
- `DESIGN.md` had been renamed to `DESIGN_rev1.md` with no new DESIGN.md. The revision-2 rule was saved as
  `_rev1_rule.txt`.
- GEN_PLAN, DRYRUN, SCORER_SELFTEST, TINY_TEST and the hook draft were stale revision-2 records. Several of them had
  CRLF line endings.

How it was completed:
- **Naming.** Revisions are now numbered 1-3, consistently with kit 5a:
  - the first build's design was restored from the session scratchpad as `DESIGN_rev1.md`;
  - the post-review design was renamed `DESIGN_rev2.md`;
  - `_rev1_rule.txt` was renamed `GO_RULE_rev2.md`.
  - Nothing was deleted. Pre-edit copies of every kit file are in the session scratchpad.
- **Row provenance.** It was re-proven independently: the revision-2 dev bank and training files are reconstructed
  byte for byte from the current rows (dev `2c0250a2…`, plain `49067515…`, trace `1bad8116…`, filler `3102361d…`).
- **Code.** The analysis and the tiny test were rewritten for revision 3. All writers now emit LF only.
- **Data and records.** The data were rebuilt after moving the interrupted build's files to the scratchpad. The data
  files are byte-identical; only the SOURCE records changed. Every check and record was then rerun.

## 14. Limitations

- The 2-key top-level object cannot be both augmented in training and held out; it trains in the canonical order
  only. Its held-out order (`event` before `before`) is therefore a factor never varied in training.
- The held-out surface is ONE combined rendering. A held-out loss cannot be attributed to paraphrase, names or key
  order (per-factor renderings are not run).
- Chains, A0 and the controls run in Titan's frame only.
- The filler's residual is at least 0 on every row, so the filler never has fewer decode positions than the trace. On
  most rows it has 1-5 more.

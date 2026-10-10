# T2b-register screen, revision 5: intermediates in the answer, with an answer-free filler control

RCWM roadmap item 3 (`docs/roadmap/RCWM_ROADMAP_2026-10-08.md`); joint roadmap item 5b with BWM-001
(`JOINT_ROADMAP_2026-10-08.md`, BWM D-52); DECISIONS D-34, D-35, R16, D-36. Revision 5 applies the operator's
decisions on the independent checker's two remaining identification gaps.

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
| 3 | about 12:24, operator decisions after PREREG-3 (D-36) | `DESIGN_rev3.md` | `GO_RULE_rev3.md` |
| 4 | about 17:45, operator-selected verifier fixes | `DESIGN_rev4.md` (the former DESIGN.md, renamed byte for byte, sha256 `b796a256…`) | `GO_RULE_rev4.md` (the former GO_RULE.md, renamed byte for byte, sha256 `e602f63d…`) |
| 5 | about 22:50, operator decisions on the checker's two gaps | `DESIGN.md` (this document) | `GO_RULE.md` (with the "what changed and why" preface) |

## 0. Changes

### 0.1 Revision 5: the operator's decisions on the checker's two remaining identification gaps

**R5-1. Decision (1): an absolute, arithmetic-independent predicate clause on cond_sametens.**

The gap:
- Revision 4's cond_sametens clause is relative (trace - plain >= +0.25), and cond_sametens has an out-of-range |value|
  (12..40).
- Against a plain arm that is weak on that arithmetic, a branch shortcut at 1/2 clears it. Against a plain arm at 0 it
  needs only 1/4.
- The dry run shows it (`weak_plain_tens_or_eq_plus1`, `weak_plain_tens_ge_not_eq`). With a plain arm that answers
  every out-of-range row wrong, the verifier's shortcut and its mirror hold every relative G1 part and every revision-4
  component. Revision 4's rule would say GO.

The fix (as decided):
- **The metric.** The trace side's PREDICATE ACCURACY (`t2br_common.pred_ok`): the strict joint parser accepts the
  answer, cmp is an object with exactly the keys x, thr, ok in that literal order, and ok is a JSON boolean equal to
  the env's predicate (other > threshold).
  - x and thr are not scored. A malformed trace or a wrong key order counts as wrong.
  - Interpretation: "wrong key order" covers the top level and the cmp object, because the joint format fixes both. This
    can only lower the trace side's score.
- **The clause.** Predicate accuracy >= 0.80 (`t2br_common.PRED_MIN` = 4/5) on cond_sametens, in-surface AND on the
  held-out surface. It is decided exactly on counts (>= 80 of 100) and has no plain-arm term.
- **Kept.** The trace - plain >= +0.25 clause on cond_sametens is unchanged.
- **Code.**
  - `t2br_analysis`: the pseudo-metrics `cond_sametens_pred` / `cond_oor_edge_pred` of the trace side, the absolute
    components (side b None), and the bootstrap CI of the trace side's own metric for the SE flag.
  - The scorer self-test adds three negatives that must fail the predicate but keep the 'after' right: cmp keys
    reordered, a cmp extra key, a cmp without ok.
  - Synthetic trace sides now state, in cmp.ok, the predicate their answer followed.

**R5-2. Decision (2): rows at other = threshold - 1, threshold and threshold + 1, via the new stratum
`cond_oor_edge`.**

The gap:
- 'true rule OR other == threshold - 1' scored 1 on every revision-4 G1 part: no G1 stratum had a row at
  other = threshold - 1.
- cond_oor_val has only (threshold: ok false; threshold + 1: ok true). cond_sametens has |d| >= 2, cond_far |d| >= 16,
  and cond_oor_thr has out-of-range thresholds.

The stratum (`t2br_make_data.build_edge`; 80 rows appended as dv1200..dv1279):
- **Quadruples.** 20 branch-balanced minimal quadruples. The four members have other - threshold = -1, 0 (ok false)
  and +1, +2 (ok true), 20 rows at each distance.
- **Pair types.** 10 quadruples share the row and move the threshold (threshold = x + 1, x, x - 1, x - 2). 10 share the
  event and move the other register (other = threshold - 1 .. threshold + 2).
- **Out of range.** |value| is 12..40 (realized -38..30; signs 44 + / 36 -).
- **In range otherwise.** Registers and thresholds are in range (thresholds realized 3..80).
- **No wraps.** Neither branch wraps, so a row's 'after' is right iff its branch is right.
- **Decades.** The decade of the boundary is drawn, not chosen: 8 members in 7 quadruples lie across a decade.
- **Seed and registry.** It is drawn after cond_sametens with its own seed (`SEED_EDGE = 20261115`), against the full
  registry (exclusion union + training + dev + chains + cond_sametens). So it is fresh and disjoint from every training
  payload and operative content.
- **Not pooled** into 'conditional'. It is rendered in E1 and in the held-out-surface bank (rows 500..579).

Why it identifies the off-by-one family:
- An off-by-one variant differs from the comparison only at d in a non-empty set S of {-1, 0, +1}.
- Every quadruple has a member at each of these distances, so the variant is contradicted on |S| / 4 >= 1/4 of
  cond_oor_edge, and its predicate accuracy there is at most 3/4.
- With cond_oor_val (d = 0, +1), G1 sees rows at d = -1, 0 and +1.

**R5-2b. My design choice: cond_oor_edge carries BOTH of cond_sametens' clauses.** That is trace - plain >= +0.25 and
predicate accuracy >= 0.80, in-surface and held out. The operator can reverse it.
- **A relative clause alone cannot meet decision (2).** Against a plain arm exactly at chance, a single-offset flip
  scores 3/4 on cond_oor_edge: exactly on the +0.25 bar, so the clause holds. Against a weak plain arm it holds easily.
- **The predicate clause fails every variant.** Since 3/4 < 4/5, it fails each of them, whatever the plain arm does.
- **Why quadruples.** They keep the stratum branch-balanced with every boundary offset present in every cluster. A
  pair design (t - 1, t + 1) would leave x == t to cond_oor_val's relative clause alone.

**R5-3. Decision (3): the enumeration, again** (`DRYRUN.json` `shortcut_enumeration`; reported).
- **The family.** 3,483 predicates (972 distinct on the dev conditional rows):
  - 122 digit bases (tens / units comparisons, AND / OR, lexicographic), plus the off-by-one bases x > t + c for
    c = -3..+3;
  - times 27 modifiers: at each of x == t-1, x == t, x == t+1 independently, unchanged, forced true or forced false.
- **How G1 is decided.** By `verdict()`, against three plain baselines: chance (1/2), the dry run's coin-flip plain,
  and a weak plain (0 on every relative G1 part).
- **Off-by-one family.** No member passes under any baseline.
- **Digit shortcuts and one off-by-two form still pass.** Against the chance plain, 26 predicates in 9 classes:
  - eight digit-wise classes (section 14);
  - `x > t or x == t-2` (cond_sametens has 18 rows at d = -2: 41/50).

  Against the coin-flip plain, 68 in 20 classes; against the weak plain, 110 in 41 classes.
- **Target: not met for the digit-wise classes and the off-by-two form.** A cross-digit stratum would catch every
  surviving digit-wise class: 40 of the 41 representatives score below 0.80 on the 4,824 cross-digit rows of the
  in-range domain. Not implemented, not selected (section 14).

**R5-4. Consequences.**
- **Components.** 22 (G1: 12). E1 has 1,280 rows and E1H 580.
- **GPU plan.** 4.82 GPU-h in two turns at the planning rate; 3.94 h in one turn at the observed rate (section 9).
  Stage names are unchanged, so the hook draft changes only in its comments.
- **Data.** Every training file and the chain bank are byte-identical to revision 4. The dev bank and the held-out
  bank are strict appends (section 13).

#### 0.1.1 Dry-run evidence for revision 5 (`DRYRUN.md`; full pipeline, exact)

| scenario | trace on G1 parts | verdict |
|---|---|---|
| `tens_or_eq_plus1`, plain at chance | conditional 1, cond_oor_val 1, cond_oor_edge 63/80 (predicate 63/80), cond_sametens 1/2 (predicate 1/2) | NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens, cond_sametens predicate, + HS]) |
| `tens_ge_not_eq`, plain at chance | the same except cond_oor_edge exactly 4/5 | NO-GO (failed: G1 [cond_sametens, cond_sametens predicate, + HS]) |
| `weak_plain_tens_or_eq_plus1` (plain 0 out of range) | as above; every relative part holds | NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens predicate, + HS]); revision 4's rule: GO |
| `weak_plain_tens_ge_not_eq` | as above | NO-GO (failed: G1 [cond_sametens predicate, + HS]); revision 4's rule: GO |
| `weak_plain_go` (gold trace, weak plain) | all 1 | GO |
| 7 off-by-one variants x {chance, weak} plain | cond_oor_edge (predicate) 3/4, 1/2 or 1/4 | NO-GO through G1 in all 14 (the cond_oor_edge predicate clause fails on both surfaces in every case; the table is in `GO_RULE.md`) |
| `go`, `true_rule_digitwise` | all 1 | GO |
| `coarse_comparison` (tens only), plain at chance | conditional 253/300; cond_oor_val 53/100; cond_oor_edge 11/20; cond_sametens 1/2 | NO-GO: every single-stratum G1 part fails, the pooled conditional holds |

### 0.2 Revision 4 (unchanged unless revision 5 says so; `DESIGN_rev4.md` section 0.1)

- **R4-1.** The identification stratum cond_sametens (100 rows, seed 20261114) and G1's trace - plain clause on it.
- **R4-2.** Amendment 4b implemented as the per-kind constant: swap 7, add 17, mix 17, conditional 29. The kind's
  shortest filler target is at least as long as its longest trace target.
- **R4-3.** Notes: the held-out state orders are the three odd permutations; the generation bound counts re-anchors;
  the per-factor renderings are omitted by choice.

### 0.3 Operator decisions of revision 3 (binding; unchanged except where revisions 4-5 say so)

- **C1. The primary build is the surface-diversity variant 'sd'; the canonical-only variant is not run.**
  - Reason: decision point 1 has fired. PREREG-3 returned STATUS: TRIGGER (REPORT section 3c): T2's void no-op is a
    token cue, and it fails on reordered JSON keys (0.37-0.47 against 0.87-0.91 in format), on word account names and
    on magnitudes.
  - Implementation:
    - The SAME 750 canonical rows (never re-drawn) get one seeded surface per row: a rule-text paraphrase, a field-name
      variant per field, and key orders (C2).
    - The build asserts the revision-2 row digests.
    - `--verify` reconstructs the four revision-2 files, the eight revision-3 files and the eight revision-4 files byte
      for byte from the current rows.
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
  - It renders the PRIMARY strata (mix_oor, cond_oor_val, cond_oor_thr, cond_far) and the identification strata
    cond_sametens (revision 4) and cond_oor_edge (revision 5): 580 rows.
  - The rendering is paraphrase 4 + names 4 + a held-out key order per object type. Orders rotate by row (state and
    event j mod 3; the top level always uses its one held-out order).
  - Same rows and golds as in-surface, paired by row (`data/t2br_dev_hs.jsonl`). Its first 400 rows are revision 3's
    bank and its first 500 revision 4's, byte for byte.
  - All four sides are evaluated: plain and trace gate G1 / G2; OFF and filler give the surface audit.
  - Per-factor renderings are omitted by choice (section 14).
- **C4. Amendment 4b filler (proposed to BWM, adopted; BWM D-55).** Its definition was made exact in revision 4.
  - The fill is a run of `" ."` of a constant length per action kind (`t2br_common.FILL_UNITS`): swap 7, add 17, mix
    17, conditional 29.
  - Each constant is the smallest run such that the filler target is at least as long as the LONGEST trace target of
    the kind over train + dev + the chains' gold steps, whatever its 'after'.
  - Targets do not depend on the prompt surface, so this covers every surface variant. The build re-derives the
    constants and fails on any other value. The cond_oor_edge rows leave them unchanged.
  - Reason for 4b: revision 2's per-row matched fill leaked the answer through its length (wrap 0.957 / 0.970; branch
    0.57 / 0.85 / 0.72).
  - Test: every filler target of a kind carries the identical fill, so the fill length's best cut equals the majority
    rate in every cell. Zero information.
- **C5. Fits the 4.5 h turn cap with the existing deferral / resume** (section 9; updated for revision 5).
- **C6. Every rule change is a new revision with a "what changed and why" preface (`GO_RULE.md`).**
- **Kit 5b decisions:**
  - **(a)** G1 and G2 must hold both in-surface and on the held-out surface.
  - **(b)** G1 also requires trace - plain >= +0.25 on cond_oor_val alone. The tens-digit comparison scores 0.843 on
    the pooled conditional but about 0.5 on the boundary pairs.
    - Revision 4 adds cond_sametens alone.
    - Revision 5 adds cond_oor_edge alone and the predicate clauses.
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
  answers did not, out of range AND on a held-out prompt surface?
- Does the comparison survive rows on which the tens digits do not decide (cond_sametens) and rows at the boundary
  itself (cond_oor_edge)? It is judged also by the predicate the trace states, which does not depend on the plain arm
  or on the arithmetic.
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
- distance d = reg[other] - threshold (ok iff d > 0). In this document x = reg[other] and t = threshold.

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
- **Predicate accuracy (revision 5, `t2br_common.pred_ok`):** per conditional row, true iff:
  - `parse_trace` accepts the answer;
  - the cmp keys are exactly `["x", "thr", "ok"]` in that order;
  - ok is a JSON boolean equal to the gold predicate.

  x and thr are not scored. It is a separate outcome from 'after', decided only on the identification strata.

## 4. Training rows (unchanged rows and files; surface-diversity rendering)

| block | rows | design |
|---|---:|---|
| mix | 300 | 150 wrap / 150 no-wrap |
| wrap | 150 | add whose raw sum leaves [0, 96]: 75 over / 75 under |
| conditional, core | 150 | other = threshold - 1 / threshold / threshold + 1, 50 each |
| conditional, twins | 150 | one minimal-pair twin per core row, on the other side of the threshold at \|d\| 2..15; per core offset 25 move the threshold and 25 move the other register |

- Values -8..11; thresholds and registers 0..96. Swap and add/no-wrap are absent on purpose (the untouched controls).
- Mean target tokens: plain 15.55, trace 38.06, filler 42.35.
- Files (`data/`): `t2br_train_sd_{plain,trace,filler}.jsonl` (the screen's training data) and
  `t2br_train_{plain,trace,filler}.jsonl` (the canonical rendering, never trained), with SOURCE records. All six are
  byte-identical to revision 4.
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
| digit shortcut tens_or_eq_plus1 / tens_ge_not_eq (revision 4) | 0.917 / 0.803 (25 / 59 contradicting rows) | reported, NOT gated |
| off-by-one family (revision 5): single flips / double flips / the triple flip | 0.833 / 0.667 / 0.500 (50 / 100 / 150 contradicting rows: the core rows) | reported, NOT gated |

**Amendment 4b on the data** (`AUDIT.json` `fill_constant`; train + dev + chain gold steps):

| kind | rows | longest trace (tokens) | constant fill (units) | shortest filler (tokens) | revision 3's per-row value | residual (filler - trace tokens) |
|---|---:|---:|---:|---:|---:|---|
| swap | 475 | 26 | 7 | 26 | 5 | 2 on every row |
| add | 724 | 35 | 17 | 35 | 14 | 3..6, mean 4.32 |
| mix | 951 | 35 | 17 | 35 | 14 | 3..6, mean 3.46 |
| conditional | 1,416 | 47 | 29 | 47 | 26 | 3..8, mean 4.38 |

- Every cell has one fill string, so the fill length's best cut equals the majority rate: zero information.
- The residual is answer-dependent only through the trace's length. It is descriptive; the filler arm never emits it.
  Its best cut on the label: training mix 0.957, training conditional 0.527, mix_oor 0.97, cond_oor_val 0.57,
  cond_oor_thr 0.85, cond_far 0.72, cond_sametens 0.56, cond_oor_edge 0.54 (residual 4..8). The analysis reports the
  filler side's accuracy split at each cell's median residual.

## 5. Dev screen bank (`data/t2br_dev.jsonl`), held-out-surface bank (`data/t2br_dev_hs.jsonl`), chains

The first 1,100 rows are unchanged from revision 2, and the first 1,200 are byte for byte revision 4's. cond_sametens
(revision 4) and cond_oor_edge (revision 5) are appended. The in-surface prompts are Titan's verbatim frame.

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
| cond_sametens | 100 | OUT | IDENTIFICATION (G1 alone, both clauses; not pooled) | 50 minimal pairs, tens(other) == tens(threshold), \|d\| 2..9 (2..7 realized), \|value\| 12..40, no wrap on either branch; 25 move the threshold, 25 the other register |
| **cond_oor_edge** | 80 | OUT | **IDENTIFICATION (G1 alone, both clauses; not pooled)** | 20 minimal quadruples at d = -1, 0 (false) / +1, +2 (true), \|value\| 12..40, no wrap on either branch; 10 move the threshold, 10 the other register |

**Held-out-surface bank** (580 rows, C3):
- It contains the rows of mix_oor, cond_oor_val, cond_oor_thr, cond_far, cond_sametens and cond_oor_edge in dev order,
  with the same ids and golds.
- Row j is rendered in `t2br_common.heldout_surface(j)`. The orders rotate 194 / 193 / 193 over the 3 held-out state
  and event orders.
- Mean prompt tokens: 182.3 against 143.3 in-surface on the same rows.
- Every prompt decodes to its row and uses ONLY held-out factors (`--verify`, the scorer self-test, the tiny test).

**Shortcut floors** (`AUDIT.json` `dev_heuristics`; branch shortcuts with exact arithmetic; the trace's predicate
accuracy on cond_oor_edge / cond_sametens equals the floor for a trace that states the shortcut):

| shortcut | pooled conditional | cond_oor_val | cond_oor_edge | cond_sametens |
|---|---|---|---|---|
| always subtract | 0.5 | 0.5 | 0.5 | 0.5 |
| eq_plus1 | 0.667 | 1.0 | 0.75 | 0.5 |
| units-digit `>` | 0.677 | 0.97 | 0.9 | 1.0 |
| tens-digit `>` | 0.843 | 0.53 | 0.55 | 0.5 |
| tens_or_eq_plus1 (revision 4) | 1.0 | 1.0 | 0.7875 | 0.5 |
| tens_ge_not_eq (revision 4) | 1.0 | 1.0 | 0.8 | 0.5 |
| obo_m1 `x > t or x == t-1` | 1.0 | 1.0 | 0.75 | 1.0 |
| obo_0 `x >= t` / obo_p1 `x > t+1` | 0.833 | 0.5 | 0.75 | 1.0 |
| obo_m1_0 / obo_m1_p1 | 0.833 | 0.5 | 0.5 | 1.0 |
| obo_0_p1 | 0.667 | 0.0 | 0.5 | 1.0 |
| obo_m1_0_p1 | 0.667 | 0.0 | 0.25 | 1.0 |

Every single shortcut above fails at least one G1 part whatever the plain arm does. Section 14 lists the digit
combinations that still pass.

**Chains** are unchanged byte for byte: 384 = CH_IN 32 + CH_X 16 parents, x 8 candidates, x 4 steps, in the chain_one
frame.

**Freshness:**
- Zero payload and zero operative-content overlap with PREREG-3's exclusion union (308,726 payloads, 301,783 operative
  contents, 17 sources). The harvest skips files by name: the `HSKILL_CONFIRM_V2*` files are skipped by name and never
  opened, and `rcwmt.bank.load_exclusions()` is not called.
- Units are mutually disjoint, both identification strata included.

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

Stages (names unchanged):
1. T_plain, T_trace, T_filler.
2. A0: the anchor. A resumed turn re-anchors.
3. E1: 1,280 in-surface rows x OFF / plain / trace / filler.
4. E1H: 580 held-out-surface rows x the same 4 sides. It runs before E2.
5. E2: 384 chains x 4 sides.

Exit codes: 0 done; 3 deterministic stop; 5 resumable deferral; 1 crash. Every text file the runner writes is LF-only.

## 7. Pre-stated go rule, revision 5 (verbatim; `scripts/t2br_common.GO_RULE_TEXT`, applied by `scripts/t2br_analysis.verdict`)

What changed from revision 4 and why: `GO_RULE.md` (preface) and section 0.1.

> PRE-STATED SCREEN GO RULE, REVISION 5 (T2b-register; supersedes revisions 1-4, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on the two identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; neither is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows. PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone AND on cond_sametens alone (a balanced stratum gives a branch-blind or tens-digit shortcut about 0.50, and an off-by-one variant scores at most 0.75 on cond_oor_edge; this clause has no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let a shortcut through), each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

**Components** (22, decided exactly, in `t2br_analysis.COMPONENTS`):

| condition | parts | kind | surface |
|---|---|---|---|
| G1 | conditional; cond_oor_val; cond_oor_edge; cond_sametens | trace - plain >= +0.25 | in-surface and held-out (HS) |
| G1 | cond_oor_edge predicate; cond_sametens predicate | trace predicate accuracy >= 0.80 (absolute) | in-surface and HS |
| G2 | mix | trace - plain >= +0.30 | in-surface and HS |
| G3 | conditional, mix | trace - filler >= +0.10 | in-surface |
| G4 | swap, add_nowrap | trace - OFF >= -0.05 | in-surface |
| G5 | conditional_in, mix_in | trace - plain >= -0.05 | in-surface |
| DP3 | conditional, mix | trace - filler < +0.10 (both) | in-surface |

A NO-GO names the failed parts in the components' order, for example
`NO-GO (failed: G1 [cond_sametens predicate, cond_sametens predicate HS])`.

## 8. Reported, not gating

- Per stratum and side, on both renderings: accuracy (Wilson CI), parse rate and cluster-bootstrap contrasts.
- **Surface audit:** held-out minus in-surface, per side and held-out-surface stratum, paired by row, with a CI and
  the discordant row counts.
- Every component with its exact value, its bootstrap SE and a flag when it lies within 1 SE of its bar. An absolute
  component uses the bootstrap CI of the trace side's own predicate accuracy.
- Trace fields: format_ok, pred_ok, cmp_ok, ok_ok, calc_ok, after_follows_calc, calc_follows_cmp, on both renderings
  (pred_ok on every conditional stratum, gating only on the two identification strata).
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
| E1 | 56.7 min | 40.3 min | 5,120 |
| E1H | 27.1 min | 19.3 min | 2,320 |
| E2 | 68.1 min | 48.4 min | 6,144 |

- Training is T2's measured 1,858.8 s scaled by processed tokens.
- Trace and filler decode 2.47 / 2.71 x longer than plain in E1, and 2.66 / 2.91 x in E1H.
- **Generation bound.** 14,416 generations with the anchor counted ONCE. This excludes a resumed turn's re-anchor (832
  generations each).
  - At the planning rate (one re-anchor): 15,248.
  - At the observed rate (no re-anchor): 14,416.
  - Hook bound (3 calls, at most 2 re-anchors): 16,080.

**Turns** (cap 4.5 h; no stage starts within 0.5 h of the cap; 1.25 x its estimate must fit; two turns allowed):

| rate | turn 1 | turn 2 | total |
|---|---|---|---|
| planning | T_plain, T_trace, T_filler, A0, E1, E1H: 3.57 h | A0 re-anchor + E2: 1.25 h | 4.82 GPU-h |
| observed | all stages: 3.94 h | - | 3.94 GPU-h |

- At the planning rate every gating output (E1, E1H) is produced in turn 1. E1H starts at about 3.1 h and needs
  1.25 x 27.1 min, so it ends by about 3.7 h (inside the cap).
- The split is automatic: the runner exits 5 when the gate refuses E2. The hook's next call resumes after a re-anchor.
- The hook allows up to 3 calls per chain pass.

## 10. Tests and records (`docs/t2b/register/`; all rerun for revision 5)

- **AUDIT.json** (`t2br_make_data.py --verify`, full historical harvest): 31/31 checks pass. New in revision 5:
  - the revision-4 reconstruction (eight files byte for byte);
  - the append check (dv0000..dv1199 in revision-4 order, then cond_oor_edge; the HS bank's first 500 rows are
    revision 4's);
  - the cond_oor_edge stratum and quadruple checks.
- **SCORER_SELFTEST.json** (`t2br_analysis.py --selftest`): pass on 1,280 dev rows and 580 held-out-surface rows.
  - 26,680 malformed answers are rejected, incl. three new predicate negatives that keep the 'after' right.
  - 3,840 fills of other lengths are accepted.
  - Gold traces have pred_ok 1 on every conditional row and on both surfaces.
  - The exact identities include cond_oor_edge (always-subtract = 1/2).
- **DRYRUN.json / DRYRUN.md** (`t2br_analysis.py --dryrun`): 42 synthetic runs (30 scenarios + 12 engineered
  exact-boundary runs) and 23 property checks, all pass. They include:
  - the weak-plain scenarios of decision (1);
  - the 7 off-by-one variants against a chance plain and against a weak plain (expected verdicts derived from the
    design);
  - engineered boundary runs on cond_oor_edge and on both predicate clauses, in-surface and HS;
  - exhaustive at-bar checks of the new relative clause and every count 0..N of both predicate clauses on both
    surfaces;
  - the enumeration (section 14).
- **TINY_TEST.json** (`t2br_test_tiny.py`, tiny random Qwen3 and the real vendored trainer; 2 threads): 30/30 checks
  pass, peak working set 1,102 MB. New in revision 5:
  - a fourth full-bank oracle run (plain weak on every out-of-range row, trace following the tens-digit shortcut).
    The analysis on the run dir says NO-GO on exactly the four predicate parts, reading cmp.ok from the run's own
    trace texts;
  - the 580-row held-out bank;
  - the surface-locked verdict naming both predicate clauses HS;
  - the freeze including the revision-4 rule and design.
- **GEN_PLAN.json**, **queue_hook_t2br.sh.draft** (NOT installed), **GO_RULE.md**, **GO_RULE_rev1/2/3/4.md**,
  **DESIGN_rev1/2/3/4.md**: all are frozen inputs (`t2br_run.py --freeze`).

## 11. Decision points

- **DP1:** fired. The sd build is the screen.
- **DP3:** trace - filler < 0.10 on both in-surface primaries means STOP, with the section 7 wording.
- **GO:** seeds 1-2 (`--seed 1/2`). The roadmap then requires:
  - the same rule in all 3 seeds;
  - a sealed R bank with held-out-surface primaries and the identification strata (D-35);
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
   - amendment 4b as implemented here (revision 4);
   - G1's cond_sametens clause (revision 4);
   - G1's cond_oor_edge clauses and the absolute predicate-accuracy clauses, which also read the cmp key order
     (revision 5).

## 13. Provenance and interruption recovery

**Revision 3's interruption recovery (C0)** is unchanged: `DESIGN_rev3.md` section 13.

**Revision 5 interruption recovery.** A first attempt at revision 5 was cut off by a usage limit and the machine was
rebooted at 18:38.
- Before any edit, every kit file was hashed against the revision-4 report. All 30 files the report lists matched
  their reported sha256, and no kit file was newer than the report's last record (18:05). So the cut-off attempt had
  changed nothing.
- No `runs/t2b*register*` output exists.
- Pre-edit copies of every kit file are in the session scratchpad (`rev5_preedit/`, with SHA256SUMS).

**Revision 5 data rebuild.**
- The twelve revision-4 data files were moved (renamed, not deleted) to `data/t2br_superseded/rev4/`, with their
  sha256 in `data/t2br_superseded/rev4_SHA256SUMS.txt`.
- The builder was rerun from a fresh name-skipping harvest. It asserts the revision-2 row digests and reconstructs the
  revision-3 and revision-4 files byte for byte from the rows (otherwise it writes nothing).
- What changed in the files:
  - **Training files and the chain bank:** byte-identical to revision 4.
  - **Dev bank:** 80 appended cond_oor_edge rows. The first 1,200 lines are revision 4's file byte for byte.
  - **Held-out-surface bank:** 80 appended rows. The first 500 lines are revision 4's file byte for byte.
  - **SOURCE records:** rewritten. They record revision 5, the cond_oor_edge description and seed, the rule text, the
    revision-4 reconstruction and the dev floors including the off-by-one family.
- `--verify` rebuilds every data file in memory with the current builder and requires byte equality (31/31).

## 14. Notes and limitations

- **What G1 still lets through (reported, not fixed; `DRYRUN.json` `shortcut_enumeration`).**
  - **Off-by-one family.** No member passes, against any plain arm.
  - **Against a plain arm exactly at chance, 26 predicates in 9 classes pass.** Eight classes are non-lexicographic
    digit-wise comparisons:
    - `tens(x)>tens(t) or units(x)>units(t)` (259/300 pooled), and the same with `and x!=t-1`;
    - `tens(x)>tens(t) or units(x)>=units(t)` with `and x!=t`, and with `and x!=t-1, and x!=t`;
    - `tens(x)>=tens(t) and units(x)>units(t)`, and the same with `or x==t+1`;
    - `tens(x)>=tens(t) and units(x)>=units(t)` with `and x!=t`, and with `and x!=t, or x==t+1`.

    Each is right on every cond_sametens row and on 19/20 to 1 of cond_oor_edge. They err only where the tens and units
    comparisons disagree across a decade. The `and x!=t-1` variants also repair cond_oor_edge's across-decade x = t-1
    members (x = 10k - 1, t = 10k).
  - **The ninth class is off by two.** `x > t or x == t-2` is outside the off-by-one family. cond_sametens has 18 of its
    100 rows at d = -2, so its predicate accuracy is 41/50, above 0.80; cond_oor_edge has no d = -2 row.
  - **Other baselines.** Against the dry run's coin-flip plain (5/12 pooled, 8/25 on cond_oor_val), 68 predicates in 20
    classes pass. Against a plain arm at 0 on every relative part, 110 in 41 classes pass. All are digit-wise comparisons
    or the off-by-two form.

    A weak plain arm makes the pooled conditional and cond_oor_val clauses permissive (they are relative). The
    absolute clauses cover only the two identification strata, as decided.
- **What would catch them (not implemented; not selected): a cross-digit identification stratum.** Rows in different
  decades, |d| >= 2, whose units comparison disagrees with the tens comparison (sign of the units difference != sign of
  the tens difference), gated by both clauses.
  - On all 4,824 such rows of the in-range domain (2,412 per branch), the true rule scores 1.
  - 40 of the 41 surviving class representatives score below 0.80 there (36 at most 1/2).
  - The off-by-two form survives (267/268). It would need rows at d = -2 in a predicate-gated stratum, for example a
    fifth member per cond_oor_edge cluster.
  - A boundary stratum cannot catch the digit-wise classes by itself. A quadruple has at most one across-decade member
    on the ok-false side (d = -1) and at most two on the ok-true side (d = +1, +2), never both.
    - `tens(x)>tens(t) or units(x)>units(t)` errs only on the first kind. It falls below 0.80 only if more than 4/5 of
      all quadruples carry such a member.
    - `tens(x)>=tens(t) and units(x)>units(t)` errs only on the second kind. It falls below 0.80 only if more than 2/5
      of all quadruples carry two such members.
    - No composition satisfies both.
- **cond_oor_edge.**
  - 8 of its 80 members lie across a decade (drawn, not chosen).
  - Value signs are 44 / 36.
  - The threshold decades 0..8 are drawn unevenly (11 / 5 / 14 / 18 / 4 / 12 / 1 / 14 / 1).
  - Neither branch wraps, so the stratum does not test the modular reduction.
  - tens_ge_not_eq scores exactly 4/5 there, on the predicate bar; it is caught by cond_sametens.
- **Predicate clause.**
  - It reads the trace's stated cmp.ok, not the branch its 'after' applied. A trace whose cmp.ok is right but whose
    'after' follows the other branch passes the predicate clause and fails the relative clause on that row.
  - "Wrong key order" includes the cmp object's own order (x, thr, ok).
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
- **Generation bound.** Revision 5: 14,416 with A0 once; 15,248 with the projected re-anchor at the planning rate;
  16,080 at the hook bound (section 9).
- **Per-factor held-out renderings are omitted BY CHOICE** (a GPU-budget choice, not a capacity limit).
  - Three more renderings (paraphrase only, names only, key order only) of the 580 rows x 4 sides cost about 81 min at
    the planning rate (6,960 generations).
  - At the planning rate they would fit in turn 2 (1.25 h used of 4.5 h). At the observed rate they would need a
    second turn and a re-anchor.
  - The operator chose not to spend GPU time on descriptive renderings. Consequence: a held-out-surface loss cannot be
    attributed to paraphrase, names or key order.
- **cond_sametens.** Distances are skewed small (2..7, mostly 2..4) by the one-decade minimal-pair construction. Value
  signs are 62 / 38. Decades 4 and 9 are not drawn. Neither branch wraps, so the stratum does not test the modular
  reduction.
- **Top level.** The 2-key top-level object trains in the canonical order only. Its held-out order is a factor never
  varied in training.
- **One combined rendering.** The held-out surface is ONE combined rendering.
- **Frame.** Chains, A0 and the controls run in Titan's frame only.
- **Residual.** The filler has 2..8 more tokens than the trace on every row: never fewer decode positions.

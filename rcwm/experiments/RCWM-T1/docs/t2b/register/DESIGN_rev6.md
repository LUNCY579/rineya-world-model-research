# T2b-register screen, revision 6: intermediates in the answer, with an answer-free filler control

RCWM roadmap item 3 (`docs/roadmap/RCWM_ROADMAP_2026-10-08.md`); joint roadmap item 5b with BWM-001
(`JOINT_ROADMAP_2026-10-08.md`, BWM D-52); DECISIONS D-34, D-35, R16, D-36, D-42. Revision 6 applies the operator's
decisions on the revision-5 check: an explicit extended shortcut family H, one adversarial identification stratum
cond_adv, a report-only shortcut-agreement analysis and a scoped rule text.

**Status:** exploratory screen.
- This is a CPU-only kit. It is not frozen, no GPU run has happened and no model output exists.
- The queue hook is a draft and is NOT installed.
- The GPU job refuses to run until `python -B scripts/t2br_run.py --freeze --approval "..."` has recorded the
  operator's go-ahead. That is a screen lock, not a preregistration.

**Revisions.** Each was written before any `runs/t2b_register` output existed. Superseded ones are kept verbatim,
renamed, never deleted.

| revision | when | design | go rule |
|---|---|---|---|
| 1 | 2026-10-09 about 03:04, first build | `DESIGN_rev1.md` | `GO_RULE_rev1.md` |
| 2 | 2026-10-09 about 07:54, after the adversarial review | `DESIGN_rev2.md` | `GO_RULE_rev2.md` |
| 3 | 2026-10-09 about 12:24, operator decisions after PREREG-3 (D-36) | `DESIGN_rev3.md` | `GO_RULE_rev3.md` |
| 4 | 2026-10-09 about 17:45, operator-selected verifier fixes | `DESIGN_rev4.md` | `GO_RULE_rev4.md` |
| 5 | 2026-10-09 about 22:50, operator decisions on the checker's two gaps | `DESIGN_rev5.md` (the former DESIGN.md, renamed byte for byte, sha256 `76470c0e…`) | `GO_RULE_rev5.md` (the former GO_RULE.md, renamed byte for byte, sha256 `af28cb65…`) |
| 6 | 2026-10-10 about 03:55, operator decisions D-42 (1)-(5) | `DESIGN.md` (this document) | `GO_RULE.md` (with the "what changed and why" preface) |

## 0. Changes

### 0.1 Revision 6: the extended shortcut family H and cond_adv (DECISIONS D-42)

**The checker's finding on revision 5.**
- The predicate-gated strata compare only numbers in one decade with the same number of digits.
- So heuristics that are right within a decade passed G1: lexicographic `str(x) > str(t)`, digit-sum comparison,
  units-only comparison (with sign- or length-aware variants), and the isolated exceptions at d = -4..-2, +3, +4.
- The rule text also overstated the predicate clause.

**R6-1. Decision (1): the extended shortcut family H** (`scripts/t2br_shortcuts.py`; section 15.1).
- H = revision 5's enumerated family (3,483 predicates, the branch-blind constants included) + 36 new members:
  lexicographic (with / without sign handling), digit-sum, units-only and tens-only (each plain, sign-aware,
  length-aware, sign+length-aware), magnitude-only, and the isolated exceptions at |d| = 2..4 of both signs.
- 3,519 predicates in all; 30 of them equal the comparison on the domain (other 0..96, threshold -40..136).

**R6-2. Decision (2): the adversarial identification stratum cond_adv** (section 15.2).
- **As decided:**
  - 100 rows, 50 / 50, seed 20261116, appended as dv1280..dv1379;
  - fresh against the name-skipping harvest registry;
  - drawn by seeded rejection sampling over valid conditional rows;
  - rendered in-surface and on the held-out surface (rows 580..679);
  - G1's absolute clause (trace predicate accuracy >= 0.80, exact counts, both surfaces) with no relative clause;
  - every existing G1 clause kept.
- **Infeasible as decided: "every member of H wrong on >= 40 of the 100 rows".** No 100-row branch-balanced stratum can
  do it:
  - five members of H have pairwise-disjoint error sets (one is wrong on <= 20 rows of any 100);
  - three revision-5 G1 survivors are pairwise disjoint (one is wrong on <= 33);
  - the exceptions at -4, -3, -2 need 63 > 50 ok-false rows to be caught together (section 15.3).
- **Implemented instead (my design choice, reversible):** the maxima of what G1 needs.
  - Every revision-5 G1 survivor (weak plain) that is not a distance exception is wrong on >= 33 rows.
  - The surviving exceptions at -3, -2, +3, +4 are wrong on 21 each (79/100).
  - The fewest rows on any one cell (5) and the most distinct cells (67).
- **The draw.** 15 design types from 13 signature classes of H: sign-crossing, cross-length (1 vs 2, 2 vs 1, 1 vs 3,
  2 vs 3 digits) and cross-decade cells, and cells at d = -3, -2, +3, +4.

**R6-3. Decision (3): the report-only shortcut-agreement analysis** (`t2br_analysis.shortcut_agreement`; section 15.5).
- It compares the trace's stated cmp.ok against the true predicate and against every member of H, over all
  conditional rows, in each rendering.
- It flags `SHORTCUT_CONSISTENT` if a member agrees at least as well as the true predicate.
- It never changes the verdict. It is quoted with the verdict (with any GO).

**R6-4. Decision (4): scoped rule text and always-quoted limits.**
- The predicate-clause sentence now says that G1 excludes the members of H documented by the enumeration, and that it
  cannot exclude every conceivable heuristic.
- `t2br_common.ALWAYS_QUOTED_LIMITS` (the rule's LIMITS sentence) is rendered beside every verdict:
  - the H scope;
  - the d = -4 survivor;
  - per-factor renderings omitted;
  - seed 0.

**R6-5. Decision (5): the enumeration over H** (`DRYRUN.json` `shortcut_enumeration`; section 15.4).
- Against the three plain baselines, exactly one member of H that differs from the comparison passes G1, on both
  surfaces: `exception d=-4: x>t or x==t-4`.
- cond_adv catches every other revision-5 survivor: 40 (chance plain), 83 (coin-flip plain) and 125 (weak plain)
  predicates.

**R6-6. Consequences.**
- **Components.** 24 (G1: 14). E1 has 1,380 rows and E1H 680.
- **GPU plan.** 4.99 GPU-h in two turns at the planning rate; 4.06 h in one turn at the observed rate (section 9).
- **Data.** The training files and the chain bank are byte-identical to revision 5. The dev bank and the held-out
  bank are strict appends (section 13).
- **Hook.** The hook draft changes only in its comments. Its stale hook-number comment is fixed: 70 is PREREG-LB, and
  the prerequisite screen is 75.

#### 0.1.1 Dry-run and tiny-test evidence for revision 6 (`DRYRUN.md`, `TINY_TEST.json`; full pipeline, exact)

| scenario | trace on cond_adv (predicate) | verdict |
|---|---|---|
| `weak_plain_lex_signed`, `weak_plain_magnitude`, `weak_plain_digitsum_sl` | 67/100 | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) |
| `weak_plain_units` | 1/100 | the same |
| `weak_plain_exc_m3`, `_exc_m2`, `_exc_p3`, `_exc_p4` (the isolated exceptions) | 79/100 | the same |
| `weak_plain_exc_m4` (the exception at -4) | 1 | GO, quoted with `SHORTCUT_CONSISTENT` (the documented limit) |
| `lex_signed`, plain at chance | 67/100 | NO-GO (failed: G1 [cond_adv predicate, cond_adv predicate HS]) |
| `surface_locked` | 1 in-surface, coin on HS | NO-GO adds `cond_adv predicate HS` |
| engineered `boundary_in_pred_cond_adv` / `_hs_` | 79/100 on one surface | NO-GO (failed: G1 [cond_adv predicate]) / [cond_adv predicate HS] |
| every other scenario of revision 5 | the shortcuts 99/100 or 1 | unchanged verdicts |

- The tiny test runs the exception at -4 and the lexicographic comparison end to end through the runner and the
  analysis: GO with the flag, and NO-GO on exactly the cond_adv clause.
- The report quotes the flag and every limit.

### 0.2 Revision 5 (unchanged unless revision 6 says so): the checker's two remaining identification gaps

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

**R5-3. Decision (3): the enumeration, again** (`DRYRUN.json` `shortcut_enumeration`; reported). Revision 5's results;
superseded by section 15.4 (revision 6 catches every class listed here).
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

**R5-4. Consequences** (revision 5's numbers; revision 6: section 0.1).
- **Components.** 22 (G1: 12). E1 has 1,280 rows and E1H 580.
- **GPU plan.** 4.82 GPU-h in two turns at the planning rate; 3.94 h in one turn at the observed rate (section 9).
  Stage names are unchanged, so the hook draft changes only in its comments.
- **Data.** Every training file and the chain bank are byte-identical to revision 4. The dev bank and the held-out
  bank are strict appends (section 13).

#### 0.2.1 Dry-run evidence for revision 5 (as recorded then; revision 6's dry run reproduces every verdict, section 0.1.1)

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

### 0.3 Revision 4 (unchanged unless revisions 5-6 say so; `DESIGN_rev4.md` section 0.1)

- **R4-1.** The identification stratum cond_sametens (100 rows, seed 20261114) and G1's trace - plain clause on it.
- **R4-2.** Amendment 4b implemented as the per-kind constant: swap 7, add 17, mix 17, conditional 29. The kind's
  shortest filler target is at least as long as its longest trace target.
- **R4-3.** Notes: the held-out state orders are the three odd permutations; the generation bound counts re-anchors;
  the per-factor renderings are omitted by choice.

### 0.4 Operator decisions of revision 3 (binding; unchanged except where revisions 4-6 say so)

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
    cond_sametens (revision 4), cond_oor_edge (revision 5) and cond_adv (revision 6): 680 rows.
  - The rendering is paraphrase 4 + names 4 + a held-out key order per object type. Orders rotate by row (state and
    event j mod 3; the top level always uses its one held-out order).
  - Same rows and golds as in-surface, paired by row (`data/t2br_dev_hs.jsonl`). Its first 400 rows are revision 3's
    bank, its first 500 revision 4's and its first 580 revision 5's, byte for byte.
  - All four sides are evaluated: plain and trace gate G1 / G2; OFF and filler give the surface audit.
  - Per-factor renderings are omitted by choice (section 14).
- **C4. Amendment 4b filler (proposed to BWM, adopted; BWM D-55).** Its definition was made exact in revision 4.
  - The fill is a run of `" ."` of a constant length per action kind (`t2br_common.FILL_UNITS`): swap 7, add 17, mix
    17, conditional 29.
  - Each constant is the smallest run such that the filler target is at least as long as the LONGEST trace target of
    the kind over train + dev + the chains' gold steps, whatever its 'after'.
  - Targets do not depend on the prompt surface, so this covers every surface variant. The build re-derives the
    constants and fails on any other value. The cond_oor_edge and cond_adv rows leave them unchanged.
  - Reason for 4b: revision 2's per-row matched fill leaked the answer through its length (wrap 0.957 / 0.970; branch
    0.57 / 0.85 / 0.72).
  - Test: every filler target of a kind carries the identical fill, so the fill length's best cut equals the majority
    rate in every cell. Zero information.
- **C5. Fits the 4.5 h turn cap with the existing deferral / resume** (section 9; updated for revision 6).
- **C6. Every rule change is a new revision with a "what changed and why" preface (`GO_RULE.md`).**
- **Kit 5b decisions:**
  - **(a)** G1 and G2 must hold both in-surface and on the held-out surface.
  - **(b)** G1 also requires trace - plain >= +0.25 on cond_oor_val alone. The tens-digit comparison scores 0.843 on
    the pooled conditional but about 0.5 on the boundary pairs.
    - Revision 4 adds cond_sametens alone.
    - Revision 5 adds cond_oor_edge alone and the predicate clauses.
    - Revision 6 adds cond_adv's predicate clause (no relative clause).
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
- Does it survive the adversarial rows of cond_adv (revision 6)? On them every member of the extended shortcut family H
  that passed revision 5's G1 is wrong on at least 21 rows, and on at least 33 unless it is an isolated distance
  exception. G1 identifies the comparison only against H (section 15).
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
- **Stated predicate (revision 6, `t2br_common.stated_ok`):** the boolean the trace states in cmp.ok, read exactly as
  pred_ok reads it (None if unreadable). It feeds the report-only shortcut-agreement analysis (section 15.5).
  pred_ok(text, row) == (stated_ok(text) == gold ok) on every gold and negative text (scorer self-test, 15,480 texts).

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
| conditional | 1,516 | 47 | 29 | 47 | 26 | 3..8, mean 4.42 |

- Every cell has one fill string, so the fill length's best cut equals the majority rate: zero information.
- The residual is answer-dependent only through the trace's length. It is descriptive; the filler arm never emits it.
  Its best cut on the label: training mix 0.957, training conditional 0.527, mix_oor 0.97, cond_oor_val 0.57,
  cond_oor_thr 0.85, cond_far 0.72, cond_sametens 0.56, cond_oor_edge 0.54 (residual 4..8), cond_adv 0.74 (residual 3..8). The analysis reports the
  filler side's accuracy split at each cell's median residual.

## 5. Dev screen bank (`data/t2br_dev.jsonl`), held-out-surface bank (`data/t2br_dev_hs.jsonl`), chains

The first 1,100 rows are unchanged from revision 2, the first 1,200 are byte for byte revision 4's and the first 1,280
revision 5's. cond_sametens (revision 4), cond_oor_edge (revision 5) and cond_adv (revision 6) are appended. The
in-surface prompts are Titan's verbatim frame.

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
| cond_oor_edge | 80 | OUT | IDENTIFICATION (G1 alone, both clauses; not pooled) | 20 minimal quadruples at d = -1, 0 (false) / +1, +2 (true), \|value\| 12..40, no wrap on either branch; 10 move the threshold, 10 the other register |
| **cond_adv** | 100 | OUT | **IDENTIFICATION (G1: the predicate clause alone; not pooled)** | 50 / 50 rows drawn by seeded rejection sampling to the quotas of 15 design types (13 signature classes of H): sign-crossing, cross-length and cross-decade cells, cells at d = -3, -2, +3, +4; \|value\| 12..40, no wrap on either branch; thresholds -40..136 (section 15.2) |

**Held-out-surface bank** (680 rows, C3):
- It contains the rows of mix_oor, cond_oor_val, cond_oor_thr, cond_far, cond_sametens, cond_oor_edge and cond_adv in
  dev order, with the same ids and golds.
- Row j is rendered in `t2br_common.heldout_surface(j)`. The orders rotate 227 / 227 / 226 over the 3 held-out state
  and event orders.
- Mean prompt tokens: 182.3 against 143.3 in-surface on the same rows.
- Every prompt decodes to its row and uses ONLY held-out factors (`--verify`, the scorer self-test, the tiny test).

**Shortcut floors** (`AUDIT.json` `dev_heuristics`; branch shortcuts with exact arithmetic; the trace's predicate
accuracy on cond_oor_edge / cond_sametens / cond_adv equals the floor for a trace that states the shortcut):

| shortcut | pooled conditional | cond_oor_val | cond_oor_edge | cond_sametens | cond_adv |
|---|---|---|---|---|---|
| always subtract | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| eq_plus1 | 0.667 | 1.0 | 0.75 | 0.5 | 0.5 |
| units-digit `>` | 0.677 | 0.97 | 0.9 | 1.0 | 0.01 |
| tens-digit `>` | 0.843 | 0.53 | 0.55 | 0.5 | 1.0 |
| tens_or_eq_plus1 (revision 4) | 1.0 | 1.0 | 0.7875 | 0.5 | 1.0 |
| tens_ge_not_eq (revision 4) | 1.0 | 1.0 | 0.8 | 0.5 | 0.99 |
| obo_m1 `x > t or x == t-1` | 1.0 | 1.0 | 0.75 | 1.0 | 1.0 |
| obo_0 `x >= t` / obo_p1 `x > t+1` | 0.833 | 0.5 | 0.75 | 1.0 | 1.0 |
| obo_m1_0 / obo_m1_p1 | 0.833 | 0.5 | 0.5 | 1.0 | 1.0 |
| obo_0_p1 | 0.667 | 0.0 | 0.5 | 1.0 | 1.0 |
| obo_m1_0_p1 | 0.667 | 0.0 | 0.25 | 1.0 | 1.0 |

Every single shortcut above fails at least one G1 part whatever the plain arm does. Section 15 enumerates the extended
family H: only the isolated exception at d = -4 still passes.

**Chains** are unchanged byte for byte: 384 = CH_IN 32 + CH_X 16 parents, x 8 candidates, x 4 steps, in the chain_one
frame.

**Freshness:**
- Zero payload and zero operative-content overlap with PREREG-3's exclusion union (308,726 payloads, 301,783 operative
  contents, 17 sources). The harvest skips files by name: the `HSKILL_CONFIRM_V2*` files are skipped by name and never
  opened, and `rcwmt.bank.load_exclusions()` is not called.
- Units are mutually disjoint, the three identification strata included (each cond_adv row is its own unit).

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
3. E1: 1,380 in-surface rows x OFF / plain / trace / filler.
4. E1H: 680 held-out-surface rows x the same 4 sides. It runs before E2.
5. E2: 384 chains x 4 sides.

Exit codes: 0 done; 3 deterministic stop; 5 resumable deferral; 1 crash. Every text file the runner writes is LF-only.

## 7. Pre-stated go rule, revision 6 (verbatim; `scripts/t2br_common.GO_RULE_TEXT`, applied by `scripts/t2br_analysis.verdict`)

What changed from revision 5 and why: `GO_RULE.md` (preface) and section 0.1.

> PRE-STATED SCREEN GO RULE, REVISION 6 (T2b-register; supersedes revisions 1-5, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on the three identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; none is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows; 'cond_adv' (revision 6) = 100 rows, 50 ok true and 50 ok false, drawn by seeded rejection sampling over valid conditional rows (other 0..96, threshold -40..136, |value| 12..40, neither branch wraps) to fixed quotas of 15 design types: 13 signature classes of the extended shortcut family H (sets of (other, threshold) cells on which exactly the same members of H are wrong), one of them split by digit lengths, covering sign-crossing cells, cross-length cells (1 vs 2, 2 vs 1, 1 vs 3 and 2 vs 3 digits), cross-decade cells whose units comparison disagrees with the tens comparison and cells at other - threshold = -3, -2, +3, +4; H (scripts/t2br_shortcuts.py, DESIGN.md section 15; 3,519 predicates) = the enumerated digit and off-by-one family with its modifiers at other = threshold - 1, threshold and threshold + 1 (3,483 predicates, the branch-blind constants included) + lexicographic string comparison with and without sign handling, digit-sum, units-only and tens-only comparisons with sign- and length-aware variants, magnitude-only |other| > |threshold| and the isolated single-distance exceptions at |other - threshold| = 2..4; on cond_adv every member of H that differs from the comparison somewhere on that domain and passes revision 5's G1 against a plain arm at 0 is wrong on at least 33 rows, except the isolated distance exceptions, of which those at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 (the operator's target, every member of H wrong on at least 40 rows, is infeasible on any 100-row branch-balanced stratum: five members of H have pairwise-disjoint error sets). PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone, on cond_sametens alone AND on cond_adv alone (cond_adv has no trace - plain clause), each both in-surface and on the held-out surface (the predicate clauses have no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let through a member of H that they catch; G1 excludes the members of H, documented by the enumeration over H against three plain baselines (DRYRUN.json shortcut_enumeration: every member of H that differs from the comparison, except the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part); it cannot exclude every conceivable heuristic); (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; the shortcut-agreement analysis (the trace side's cmp.ok against the env's predicate and against every member of H over all conditional rows of the dev bank, in each rendering: 'SHORTCUT_CONSISTENT' if a member of H that differs from the predicate on those rows agrees with the trace's cmp.ok at least as well as the predicate does; it never changes the verdict and is quoted with any GO); per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). LIMITS, quoted with every statement of the verdict: G1 identifies the comparison only against H (it excludes the members of H that the enumeration shows failing it, not every conceivable heuristic); the isolated exception 'other > threshold or other == threshold - 4' passes G1 (on any 100-row branch-balanced stratum one of the exceptions at -4, -3, -2 passes the 0.80 bar); per-factor held-out renderings are omitted; a GO is a seed-0 screen result. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

**Components** (24, decided exactly, in `t2br_analysis.COMPONENTS`):

| condition | parts | kind | surface |
|---|---|---|---|
| G1 | conditional; cond_oor_val; cond_oor_edge; cond_sametens | trace - plain >= +0.25 | in-surface and held-out (HS) |
| G1 | cond_oor_edge predicate; cond_sametens predicate; cond_adv predicate (revision 6) | trace predicate accuracy >= 0.80 (absolute) | in-surface and HS |
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
  (pred_ok on every conditional stratum, gating only on the three identification strata).
- **Shortcut agreement (revision 6; section 15.5):** the trace's stated cmp.ok against the true predicate and every
  member of H, over all conditional rows, each rendering; flag `SHORTCUT_CONSISTENT`. It is quoted with the verdict
  and never changes it.
- **Always-quoted limits (revision 6):** `t2br_common.ALWAYS_QUOTED_LIMITS`, rendered beside every verdict and in every
  verdict statement.
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
| E1 | 61.6 min | 43.7 min | 5,520 |
| E1H | 32.0 min | 22.7 min | 2,720 |
| E2 | 68.5 min | 48.7 min | 6,144 |

- Training is T2's measured 1,858.8 s scaled by processed tokens.
- Trace and filler decode 2.49 / 2.74 x longer than plain in E1, and 2.68 / 2.94 x in E1H.
- **Generation bound.** 15,216 generations with the anchor counted ONCE. This excludes a resumed turn's re-anchor (832
  generations each).
  - At the planning rate (one re-anchor): 16,048.
  - At the observed rate (no re-anchor): 15,216.
  - Hook bound (3 calls, at most 2 re-anchors): 16,880.

**Turns** (cap 4.5 h; no stage starts within 0.5 h of the cap; 1.25 x its estimate must fit; two turns allowed):

| rate | turn 1 | turn 2 | total |
|---|---|---|---|
| planning | T_plain, T_trace, T_filler, A0, E1, E1H: 3.73 h | A0 re-anchor + E2: 1.26 h | 4.99 GPU-h |
| observed | all stages: 4.06 h | - | 4.06 GPU-h |

- At the planning rate every gating output (E1, E1H) is produced in turn 1. E1H starts at about 3.2 h and needs
  1.25 x 32.0 min, so it ends by about 3.9 h (inside the cap).
- At the observed rate E2 starts at about 3.25 h and needs 1.25 x 48.7 min: it still fits the one turn (4.06 h).
- The split is automatic: the runner exits 5 when the gate refuses E2. The hook's next call resumes after a re-anchor.
- The hook allows up to 3 calls per chain pass.

## 10. Tests and records (`docs/t2b/register/`; all rerun for revision 6)

- **AUDIT.json** (`t2br_make_data.py --verify`, with a fresh historical name-skipping harvest): 33/33 checks pass.
  - The CURRENT builder rebuilds every data file byte for byte.
  - New in revision 6:
    - the revision-5 reconstruction (eight files byte for byte);
    - the append check (dv0000..dv1279 in revision-5 order, then cond_adv; the HS bank's first 580 rows are revision
      5's);
    - cond_adv's stratum definition (every row in its design type, |value| 12..40, no wrap);
    - one cluster per cell, 50 / 50;
    - the H target report (`cond_adv`: every count, the targets and the infeasibility certificates).
  - The record's code hashes cover `t2br_make_data.py`, `t2br_common.py` and `t2br_shortcuts.py`.
- **SCORER_SELFTEST.json** (`t2br_analysis.py --selftest`): pass on 1,380 dev rows and 680 held-out-surface rows.
  - 28,980 malformed answers are rejected.
  - 4,140 fills of other lengths are accepted.
  - Gold traces have pred_ok 1 on every conditional row and on both surfaces, cond_adv included.
  - The exact identities include cond_adv (always-subtract = 1/2, both indices).
  - New in revision 6:
    - `stated_ok` agrees with `pred_ok` on 15,480 gold and negative texts;
    - the shortcut-agreement analysis flags nothing on gold traces (860/860 and 580/580; the best member of H reaches
      846 and 568).
- **DRYRUN.json / DRYRUN.md** (`t2br_analysis.py --dryrun`): 54 synthetic runs (40 scenarios + 14 engineered
  exact-boundary runs) and 27 property checks, all pass. New in revision 6:
  - nine members of H as full scenarios against the weak plain arm, plus lex signed against the chance plain;
  - engineered boundary runs of the cond_adv predicate clause (in-surface and HS);
  - exhaustive at-bar checks of every count 0..100 of the cond_adv clause on both surfaces;
  - the enumeration over H (3,519 predicates, three plain baselines, both surfaces) with the survivor and why;
  - the shortcut-agreement flag of every scenario.
- **TINY_TEST.json** (`t2br_test_tiny.py`, tiny random Qwen3 and the real vendored trainer; 2 threads): 31/31 checks pass (168 s), peak working set 1,227 MB.
  New in revision 6:
  - two more full-bank oracle runs: a trace arm following the isolated exception at -4 (GO and flagged) and one
    following the lexicographic comparison (NO-GO on exactly the cond_adv clause);
  - the shortcut-agreement flags and the quoted statements and limits, read from the run directories;
  - the 680-row held-out bank;
  - the surface-locked verdict naming cond_adv predicate HS;
  - the freeze including the revision-5 rule and design and `t2br_shortcuts.py`;
  - the plan check: a 3 h cap now needs three turns (training; A0 + E1 + E1H; re-anchor + E2). The check accepts more
    than one turn, each later turn starting with the anchor.
- **GEN_PLAN.json** (`t2br_run.py --plan`): section 9.
- **Frozen inputs** (`t2br_run.py --freeze`): GEN_PLAN.json, queue_hook_t2br.sh.draft (NOT installed), GO_RULE.md,
  GO_RULE_rev1..5.md, DESIGN_rev1..5.md, and every record above. `scripts/t2br_shortcuts.py` is a frozen script.
- **Pre-freeze checks.** Revision 6 tightens one: the scorer self-test must also match the current analysis, common and
  shortcut code and the held-out-surface bank.

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
     (revision 5);
   - the extended shortcut family H, G1's cond_adv predicate clause and the report-only shortcut-agreement analysis
     (revision 6).

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

**Revision 6 interruption recovery.** A first attempt at revision 6 was cut off by a usage limit after a few steps.
- **Check before any edit (2026-10-10 about 03:02).** Every kit file was hashed against revision 5's records:
  - all five scripts matched the code hashes in TINY_TEST.json, SCORER_SELFTEST.json and AUDIT.json;
  - all twelve data files matched AUDIT.json and their SOURCE records;
  - no kit file was newer than revision 5's last write (DESIGN.md, 2026-10-09 22:57).
  - So the cut-off attempt had changed nothing.
- **Outputs.** No `runs/t2b*register*` output exists.
- **Backups.** Pre-edit copies of every kit file are in the session scratchpad (`rev6_preedit/`, with SHA256SUMS).

**Revision 6 data rebuild.**
- **Superseded files.** The twelve revision-5 data files were moved (renamed, not deleted) to
  `data/t2br_superseded/rev5/`, with their sha256 in `data/t2br_superseded/rev5_SHA256SUMS.txt`. They equal the
  pre-edit backups.
- **The rebuild.** The builder was rerun from a fresh name-skipping harvest (308,726 payloads, 301,783 operative
  contents, 17 sources; the HSKILL_CONFIRM_V2* files skipped by name, never opened).
  - It asserts the revision-2 row digests.
  - It reconstructs the revision-3, revision-4 and revision-5 files byte for byte from the rows.
  - It checks cond_adv's targets. Otherwise it writes nothing.
- **What changed in the files:**
  - **Training files and the chain bank:** byte-identical to revision 5.
  - **Dev bank:** 100 appended cond_adv rows. The first 1,280 lines are revision 5's file byte for byte.
  - **Held-out-surface bank:** 100 appended rows. The first 580 lines are revision 5's file byte for byte.
  - **SOURCE records:** rewritten. They record revision 6, cond_adv's description, seed, design types and target
    report, the rule text, the revision-5 reconstruction and the dev floors.
- **Verify.** `--verify` rebuilds every data file in memory with the current builder and requires byte equality
  (33/33).

## 14. Notes and limitations

- **What G1 excludes, and what it cannot** (revision 6; section 15; the always-quoted limits).
  - **Scope.** G1 identifies the comparison only against the extended shortcut family H. Every member of H that differs
    from the comparison fails G1 under the three plain baselines, except one.
  - **The one survivor.** The isolated exception `x>t or x==t-4` passes every G1 part. No 100-row branch-balanced
    stratum can catch the exceptions at -4, -3 and -2 together.
  - **Outside H.** A heuristic outside H is not excluded by construction. Examples: a different modular slip, a
    comparison that is wrong only on cells no stratum contains, or a mixture of the true rule and a shortcut on some
    rows.
  - **Report-only flag.** The shortcut-agreement analysis reports how well the trained trace's stated predicate matches
    every member of H. It is a description, not a gate.
- **cond_adv** (section 15).
  - **Concentration.** Its rows are concentrated by design. 33 sign-crossing rows have t < 0 (out of range, like
    cond_oor_thr), and 25 lie on five cells at d = +3 / +4 with x <= |t|. At most 5 rows share one cell. The bootstrap
    clusters by cell (67 clusters).
  - **Counts fixed by design.** Members of H in its target are wrong on exactly 33 rows (lex signed, magnitude,
    digitsum sign+length-aware, units sign+length-aware) or on exactly 21 (the exceptions at -3, -2, +3, +4). A
    trace that follows such a member on only part of the rows where it differs from the comparison can pass the 0.80
    bar: to fail, it must follow a 33-row member on at least 21 of those rows, and a 21-row exception on all 21.
  - **No wraps.** Neither branch wraps, so cond_adv does not test the modular reduction.
  - **No relative clause.** cond_adv has no trace - plain clause, as decided. It gates the stated predicate only.
- **Revision 5's enumeration and its cross-digit option** (`DESIGN_rev5.md` section 14) are superseded. Every class
  they listed is caught by cond_adv; the cross-digit option was not implemented.
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
- **Generation bound.** Revision 6: 15,216 with A0 once; 16,048 with the projected re-anchor at the planning rate;
  16,880 at the hook bound (section 9).
- **Per-factor held-out renderings are omitted BY CHOICE** (a GPU-budget choice, not a capacity limit).
  - Three more renderings (paraphrase only, names only, key order only) of the 680 rows x 4 sides cost about 96 min at
    the planning rate (8,160 generations).
  - At the planning rate they would fit in turn 2 (1.26 h used of 4.5 h). At the observed rate they would need a
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

## 15. The extended shortcut family H and the adversarial stratum cond_adv (revision 6)

### 15.1 H (`scripts/t2br_shortcuts.py`; decision (1))

Every member is a predicate 'add iff P(x, t)' of x = reg[other] and t = threshold. It is evaluated on the domain of this
kit's register rows: x in 0..96, t in -40..136 (17,169 cells).

| part | members | definition |
|---|---:|---|
| enumerated (revisions 4-5, unchanged) | 3,483 | 122 digit bases (14 atoms: TRUE, FALSE, tens(x) OP tens(t), units(x) OP units(t) with floor semantics; 72 AND / OR pairs; 36 digit-wise lexicographic forms) + 'x > t + c', c = -3..+3; each with the 27 modifiers at x == t-1, t, t+1 (unchanged / forced true / forced false) |
| lexicographic | 4 | `str(x) OP str(t)` with sign handling (sign first; on the domain = Python's string order, '-' sorting before digits) and `str(|x|) OP str(|t|)` without it; OP in {>, >=} |
| units-only, tens-only, digit-sum | 24 | f(\|x\|) OP f(\|t\|) for f = units digit, tens digit ((\|n\| // 10) % 10), digit sum; each plain, sign-aware (a non-negative number is above a negative one), length-aware (more digits is larger) and sign+length-aware (sign, then length, then f); OP in {>, >=} |
| magnitude-only | 2 | \|x\| > \|t\|, \|x\| >= \|t\| |
| isolated single-distance exceptions | 6 | 'x > t' with exactly one distance d = x - t flipped, d in -4..-2 and +2..+4 (`x>t or x==t-4`, ..., `x>t and x!=t+4`) |

- **Branch-blind constants.** TRUE and FALSE are atoms of the enumerated family.
- **Equivalents of the comparison.** 30 members equal 'x > t' on the whole domain (e.g. `x>t`, the digit-wise
  lexicographic form). They are not shortcuts and are excluded from every target.
- **Distinct functions.** The other 3,489 predicates form 1,003 distinct functions on the domain.

### 15.2 cond_adv: the design (decision (2))

**Fixed as decided.**
- 100 rows, 50 ok-true / 50 ok-false, seed 20261116.
- |value| 12..40 (out of range) and neither branch wraps, so a row's 'after' is right iff its branch is.
- Registers in range; thresholds -40..136 (the kit's domain: in-range plus cond_oor_thr's out-of-range bands).
- Gated by the predicate clause alone (in-surface and HS); not pooled; no relative clause.

**The draw: seeded rejection sampling** (`t2br_make_data.build_adv`).
- A proposal draws its cell uniformly (other 0..96, threshold -40..136), then a value, an event and the other two
  registers.
- It is accepted iff:
  - its cell is in a design type whose quota is not filled;
  - the cell is one of the type's least-used cells, so every cell of a type is used once before any is used twice;
  - no branch wraps;
  - the row is fresh against the full registry (each row its own unit).
- 484,756 proposals were drawn.
- A cluster (for the bootstrap) is one (x, t) cell: 67 clusters.

**The design types** (`t2br_shortcuts.ADV_TYPES`).
- A type is a SIGNATURE CLASS: all domain cells on which exactly the same members of H are wrong. So every member's
  count on cond_adv is fixed by the quotas.
- The class `len_lead` is split by digit lengths, so that 1 vs 2, 1 vs 3 and 2 vs 3 digits are all drawn.
- "Wrong (extension)" lists the extension members wrong on the type; the enumerated family's counts are in `AUDIT.json`
  `cond_adv.wrong_counts`.

| type | ok | rows | cells (class) | cells used | the cells | wrong there (extension) |
|---|---|---:|---:|---:|---|---|
| sc_far | true | 8 | 94 | 8 | sign-crossing, x and \|t\| in one decade, units(x) < units(\|t\|), units(x) + units(\|t\|) <= 9, d >= 5 | lex unsigned; units and digit-sum plain / length-aware; tens plain / length-aware `>`; magnitude |
| sc_d4 | true | 10 | 2 | 2 | (0, -4), (1, -3) | as sc_far, + exception +4 |
| sc_d3 | true | 10 | 2 | 2 | (0, -3), (1, -2) | as sc_far, + exception +3 |
| sc_d4_eq | true | 5 | 1 | 1 | (2, -2) | the `>` forms of sc_far, + exception +4 |
| len21_d4 | true | 6 | 4 | 4 | (10, 6) .. (13, 9) | lex (all 4); units and digit-sum plain / sign-aware; exception +4 |
| len21_d3 | true | 6 | 3 | 3 | (10, 7) .. (12, 9) | the same, exception +3 |
| dec_d3 | true | 5 | 24 | 5 | x = 10k + 0..2, t = x - 3, k = 2..9 | units and digit-sum (all 16 variants); exception +3 |
| dec0_m2 | false | 1 | 79 | 1 | t = x + 2 inside one decade | tens `>=` (4 variants); exception -2 |
| len_lead_12 / _13 / _23 | false | 3 / 2 / 3 | 84 / 44 / 23 | 8 | x shorter than t (1 vs 2 / 1 vs 3 / 2 vs 3 digits), str(x) > str(t), units(x) > units(t), digitsum(x) < digitsum(t) | lex (all 4); units plain / sign-aware |
| len12_m3 | false | 3 | 3 | 3 | (7, 10), (8, 11), (9, 12) | lex; units and digit-sum plain / sign-aware; exception -3 |
| len12_m2 | false | 10 | 2 | 2 | (8, 10), (9, 11) | the same, exception -2 |
| dec_m3 | false | 18 | 24 | 18 | x = 10k + 7..9, t = x + 3, k = 1..8 | units and digit-sum (all 16 variants); exception -3 |
| dec_m2 | false | 10 | 16 | 10 | x = 10k + 8..9, t = x + 2, k = 1..8 | the same, exception -2 |

**Realized** (`data/t2br_dev.SOURCE.json` `describe.cond_adv`, `cond_adv_targets`):
- 67 distinct cells, at most 5 rows on one cell;
- 33 sign-crossing rows, 33 cross-length rows (5 with a three-digit threshold), 33 cross-decade rows;
- 21 rows at each of d = -3, -2, +3, +4;
- value signs 53 / 47; thresholds -39..135; other 0..89;
- trace targets 40..47 tokens (the 4b constants are unchanged; the longest conditional trace stays 47).

### 15.3 Targets: decided, infeasible, implemented

**The operator's target: every member of H wrong on >= 40 of the 100 rows. It is infeasible on ANY 100-row
branch-balanced stratum.** These bounds are checked on the domain by `adv_target_report` and recorded in
`AUDIT.json` / the dev SOURCE.
1. **Five members of H have pairwise-disjoint error sets.** On any 100 rows one of them is wrong on <= 20 rows, fewer
   than the 21 needed to fail the 0.80 bar there:
   - `tens(x)>tens(t) | or x==t+1`: same-decade ok-true cells;
   - `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)!=units(t)) | and x!=t-1`: same-decade ok-false cells;
   - `magnitude: |x|>|t|`: sign-crossing cells with x <= |t|;
   - `lex signed: str(x)>str(t)`: cross-length cells;
   - `digitsum sign+length-aware: >`: same-length cross-decade cells.
2. **Three of them pass revision 5's G1 against a weak plain arm** (magnitude, lex signed, digitsum
   sign+length-aware). On any 100 rows one of them is wrong on <= 33.
3. **The isolated exceptions at d = -4, -3, -2.**
   - Each is wrong only on ok-false rows at its own distance. Failing the bar needs >= 21 such rows each, and
     3 x 21 = 63 > 50.
   - No other G1 part catches them: cond_sametens has 12 / 12 / 18 rows at d = -4 / -3 / -2 (accuracy >= 0.82), and
     cond_oor_edge has none.

**Implemented (my design choice; reversible): the maxima of what G1 needs, in this order** (an integer program over
the 156 signature classes):
1. Every member of H that passes revision 5's G1 against the weak plain arm (126 predicates) and is not a function of d
   alone (119) is wrong on >= 33 rows. The minimum is exactly 33.
2. The distance exceptions among those survivors at d = -3, -2, +3, +4 (6 predicates, aliases included) are wrong on
   exactly 21 rows. The exception at -4 (1 predicate) is not caught; by bound 3 one of -4, -3, -2 must stay, and -4 is
   the farthest from the boundary.
3. The fewest rows on any one cell (5), then the most distinct cells (67).
   - Bound: target 1 needs 33 sign-crossing ok-true rows with x <= |t| for magnitude, and target 2 needs 42 ok-true
     rows at d = +3 / +4.
   - So at least 25 rows lie on the five sign-crossing cells at d = +3 / +4.

**How far the operator's 40 is met.**
- 2,444 of the 3,489 predicates that differ from the comparison are wrong on >= 40 rows (660 of 1,003 distinct
  functions); 2,512 are wrong on >= 21.
- Members that stay low on cond_adv (minimum 0, for example `tens sign-aware: >`) fail another G1 part under every
  plain baseline.

### 15.4 The enumeration over H (decision (5); `DRYRUN.json` `shortcut_enumeration`)

**Method.**
- Each member answers every conditional with its branch and exact arithmetic, and states itself in cmp.ok.
- `verdict()` decides the 14 G1 components exactly, in-surface and HS (same rows).
- Three plain baselines: chance 1/2; the dry run's coin-flip plain; a weak plain at 0.

| plain baseline | passed revision 5's G1 (not the comparison) | of them in the extension | pass revision 6's G1 | caught by cond_adv | off-by-one family passing |
|---|---:|---:|---:|---:|---:|
| chance (1/2) | 41 | 15 | 1 | 40 | 0 |
| coin-flip dry-run plain | 84 | 16 | 1 | 83 | 0 |
| weak plain (0) | 126 | 16 | 1 | 125 | 0 |

**Target "no member of H passes G1 on either surface": not met, by one predicate.**
- **The survivor.** `exception d=-4: x>t or x==t-4` passes under all three baselines, on both surfaces.
- **Its exact values.**
  - Pooled conditional, cond_oor_val and cond_oor_edge: 1 (relative and predicate).
  - cond_sametens 22/25: it has 12 rows at d = -4.
  - cond_adv: 1 (no row at d = -4).
- **Why it cannot be caught as well.** Bound 3 of 15.3: catching it would mean dropping -3 or -2.

**Everything else in H fails G1 under every baseline.** That covers the off-by-one family, every digit-wise shortcut
of revisions 4-5, lexicographic, digit-sum, units-, tens- and magnitude-only (every variant), and the exceptions at -3,
-2, +2 (caught by cond_oor_edge), +3 and +4.

**Full pipeline** (dry run; tiny test for two of them).
- Against the weak plain arm, each of lex signed, magnitude, digit sum sign+length-aware, units and the exceptions at
  -3, -2, +3, +4 is NO-GO through exactly `G1 [cond_adv predicate, cond_adv predicate HS]`.
- The exception at -4 is GO, quoted with `SHORTCUT_CONSISTENT`.

### 15.5 The report-only shortcut-agreement analysis (decision (3))

**How it is computed** (`t2br_analysis.shortcut_agreement`).
- **What is read.** For each seed it reads the trace side's stated cmp.ok on every conditional row:
  - all 860 in-surface rows and the 580 conditional held-out-surface rows;
  - read exactly as the predicate clause reads it (`t2br_common.stated_ok`; a row with no readable ok agrees with
    nothing).
- **What is counted.** The rows where it equals the env's predicate, and the rows where it equals each member of H.
- **Which members.**
  - Members equal to the comparison on the domain are not candidates.
  - Members equal to the predicate on every row of a surface cannot be told apart there. They are counted, not
    compared.
- **The flag.** `SHORTCUT_CONSISTENT` on a surface if some candidate agrees at least as often as the predicate;
  overall if on either surface.

**How it is reported.**
- It is report-only: it never changes the verdict.
- It is quoted with every verdict statement (`statements` in the results JSON; the report's "Quoted with the verdict"
  line), so it is quoted with any GO.
- The always-quoted limits follow it.

**Checks.**
- **Gold traces.** Agreement 860/860 and 580/580; the best member (the exception at -4) reaches 846 and 568. Not
  flagged (scorer self-test).
- **Dry run.**
  - Not flagged: `go`, `weak_plain_go`, `true_rule_digitwise`.
  - Flagged: every scenario whose trace states a member of H, including the GO of the exception at -4.
- **Tiny test.** The same, read from the run directories' own trace texts.

**Limits of the analysis.**
- It describes the stated predicate only. A model whose cmp.ok is noisy can be flagged by chance, because some of the
  3,489 members may match its errors.
- A flag is a reason to look, not a verdict.

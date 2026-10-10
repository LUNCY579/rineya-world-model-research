# T2b-register screen go rule, REVISION 5 (current; pre-stated)

Applied by `scripts/t2br_analysis.verdict` (exact rationals). The same text is `scripts/t2br_common.GO_RULE_TEXT`, is
written into `data/t2br_dev.SOURCE.json`, and is a frozen input (`scripts/t2br_run.py --freeze`). Stated 2026-10-09
(about 22:50) before any `runs/t2b_register` output existed (none exists) and before any model output of this screen
was produced.

Earlier revisions are kept verbatim:
- `GO_RULE_rev1.md`: the first build, about 03:04.
- `GO_RULE_rev2.md`: after the review, about 07:54.
- `GO_RULE_rev3.md`: the operator decisions after PREREG-3, about 12:24.
- `GO_RULE_rev4.md`: the operator-selected verifier fixes, about 17:45. This is the former `GO_RULE.md`, renamed byte
  for byte (sha256 `e602f63dcae339f750d7d0e74210d2ce21fd7a92725e16d1abc0d4ded3f0c454`).

sha256 of the rule text (one line, UTF-8, no trailing newline): `241befabca5352b07b7d20c97c017654ad57ef3c40d476ab6d4897dcfe6f9abf`

## What changed from revision 4, and why

The independent checker found two remaining identification gaps. The operator decided (1) and (2) and asked for (3).
They are applied as decided. One design choice was mine and is marked as such (2b); the operator can reverse it.

### 1. An absolute, arithmetic-independent predicate clause on cond_sametens

**The gap.** Revision 4's cond_sametens clause is relative: trace - plain >= +0.25. cond_sametens has an out-of-range
|value| (12..40). If the plain arm is weak on that arithmetic, a branch shortcut at 1/2 clears the clause: against a
plain arm at 0 it needs only 1/4.

**Shown in the dry run (`weak_plain_tens_or_eq_plus1`).** The plain arm answers every out-of-range row wrong. The
verifier's shortcut 'add iff tens(other) > tens(threshold) or other == threshold + 1' then holds every relative G1 part
and every revision-4 component, so revision 4's rule would say GO. The mirror ('tens(other) >= tens(threshold) and
other != threshold') behaves the same.

**The fix (as decided).**
- **Predicate accuracy, defined.** The share of a stratum's rows whose trace answer:
  - the strict joint parser accepts (keys cmp, calc, after in that order; no extra, missing or duplicate key);
  - has a cmp object with exactly the keys x, thr, ok, in that literal order;
  - has an ok that is a JSON boolean equal to the env's predicate (other > threshold) for that row.

  x and thr are not scored. A malformed trace or a wrong key order counts as wrong.
- **The clause.** The trace side's predicate accuracy on cond_sametens must be >= 0.80, in-surface AND on the
  held-out-surface rendering. It is decided exactly on counts (>= 80 of 100 rows) and has no plain-arm term.
- **Kept.** The trace - plain >= +0.25 clause on cond_sametens is unchanged.
- **Interpretation, stated.** "Wrong key order" is read as covering both the top level and the cmp object, because the
  joint format fixes both. It can only lower the trace side's score.

A balanced 50/50 stratum gives any branch-blind or tens-digit shortcut about 0.50, so the 0.80 bar separates them. The
shortcut and its mirror score exactly 1/2 on cond_sametens, in-surface and held out.

### 2. Rows at other = threshold - 1, threshold and threshold + 1: the new stratum cond_oor_edge

**The gap.** 'true rule OR other == threshold - 1' scored 1 on every revision-4 G1 part:
- no G1 stratum had a row at other = threshold - 1;
- cond_oor_val has only the pair (threshold: ok false; threshold + 1: ok true);
- cond_sametens has |other - threshold| >= 2, cond_far >= 16, and cond_oor_thr has out-of-range thresholds.

**The fix: a new small boundary stratum, `cond_oor_edge` (80 rows, appended as dv1200..dv1279).**
- **Quadruples.** 20 branch-balanced minimal quadruples at the boundary of the env's strict `>`: other - threshold =
  -1 and 0 (ok false), +1 and +2 (ok true).
- **Pair types.** 10 quadruples share the row and move the threshold (threshold = x + 1, x, x - 1, x - 2). 10 share the
  event and move the other register (other = threshold - 1 .. threshold + 2).
- **Out of range.** |value| is 12..40, so the stratum is out of range like cond_oor_val.
- **In range otherwise.** Registers and thresholds are in range (thresholds realized 3..80).
- **No wraps.** Neither branch wraps, so a row's 'after' is right iff its branch is right (given the in-branch
  arithmetic).
- **Decades.** The decade of the boundary is drawn, not chosen: 8 of the 80 members lie across a decade boundary, in 7
  quadruples.
- **Freshness.** It is drawn after cond_sametens, with its own seed (20261115), against the full registry: PREREG-3's
  exclusion union (the name-skipping harvest), training, dev, chains and cond_sametens. So it is fresh and disjoint from
  every training payload.
- **Not pooled.** It is not pooled into 'conditional', so the primary is unchanged.
- **Byte-identical elsewhere.** Every other data file is byte-identical to revision 4: the six training files and the
  chain bank. The dev bank's first 1,200 rows and the held-out-surface bank's first 500 rows are revision 4's, byte for
  byte. `--verify` reconstructs all eight revision-4 data files from the current rows. The 4b fill constants are
  unchanged (the new traces are 40..46 tokens; the longest conditional trace is still 47).

**Why every off-by-one variant fails.**
- **Every variant is contradicted on >= 1/4 of cond_oor_edge.** An off-by-one variant differs from the comparison only
  at other - threshold in a non-empty set S of {-1, 0, +1}. Every quadruple has one member at each of these distances.
- **So its predicate accuracy there is at most 3/4 < 0.80.**

**2b. My design choice: cond_oor_edge carries cond_sametens' two clauses.** That is trace - plain >= +0.25 AND
predicate accuracy >= 0.80, both surfaces. The reason: a relative clause alone cannot meet decision (2).
- Against a plain arm exactly at chance (1/2), a single-offset flip scores 3/4 on cond_oor_edge. That is exactly on the
  +0.25 bar, so the relative clause holds.
- Against a weak plain arm, it holds by a wide margin.
- Only the predicate clause fails it, and it does so whatever the plain arm does.
- Without 2b, 'true rule OR x == t-1' would pass G1 again. A pair design (t-1 false, t+1 true) under a relative clause
  alone would fail it against a chance plain, but not against a weak plain: the gap of decision (1).

**Shown in the dry run** (full pipeline; the expected verdicts are derived from the bank's design, not from a run):

| variant | form | contradicted share: conditional / cond_oor_val / cond_oor_edge / cond_sametens | verdict, plain exactly at chance | verdict, plain weak out of range |
|---|---|---|---|---|
| obo_m1 | `x > t or x == t-1` | 0 / 0 / 1/4 / 0 | NO-GO (failed: G1 [cond_oor_edge predicate, cond_oor_edge predicate HS]) | the same |
| obo_0 | `x >= t` | 1/6 / 1/2 / 1/4 / 0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, + HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, + HS]) |
| obo_p1 | `x > t+1` | 1/6 / 1/2 / 1/4 / 0 | the same as obo_0 | the same as obo_0 |
| obo_m1_0 | `x >= t-1` | 1/6 / 1/2 / 1/2 / 0 | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, + HS]) | NO-GO (failed: G1 [cond_oor_edge predicate, + HS]) |
| obo_m1_p1 | `(x > t or x == t-1) and x != t+1` | 1/6 / 1/2 / 1/2 / 0 | the same as obo_m1_0 | the same |
| obo_0_p1 | `x == t or x > t+1` | 1/3 / 1 / 1/2 / 0 | NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, + HS]) | NO-GO (failed: G1 [cond_oor_val, cond_oor_edge predicate, + HS]) |
| obo_m1_0_p1 | `x in {t-1, t} or x > t+1` | 1/3 / 1 / 3/4 / 0 | as obo_0_p1, and G5 [conditional_in] | as obo_0_p1 (weak), and G5 [conditional_in] |

"+ HS" means the same parts on the held-out surface. Each variant is contradicted on at least 1/6 of a G1-gated
stratum (1/4 to 3/4 of cond_oor_edge; 1/2 or 1 of cond_oor_val for the six that touch it). The training rows contradict every single flip on
exactly 50 of their 300 conditional rows (the core rows at -1 / 0 / +1); this is reported, not gated.

### 3. The enumeration, again (reported)

**The family.** 3,483 predicates (972 distinct on the dev conditional rows):
- 122 tens / units digit bases, plus the off-by-one family bases 'x > t + c' for c = -3..+3;
- each combined with every modifier at x == t-1, x == t and x == t+1 (each point unchanged, forced true or forced
  false: 27 modifiers).

**How G1 is decided.** By `verdict()` on exact counts, against three plain baselines:
- chance (exactly 1/2);
- the dry run's coin-flip plain;
- a weak plain (0 on every relative G1 part).

**Results.**
- **The off-by-one family.** No member passes under any baseline: target met.
- **Digit shortcuts, chance plain.** 26 predicates in 9 classes still pass. Target not met. Eight classes are
  non-lexicographic digit-wise comparisons:
  - `tens(x)>tens(t) or units(x)>units(t)`, and the same with `and x!=t-1`;
  - `tens(x)>tens(t) or units(x)>=units(t)` with `and x!=t`, and with `and x!=t-1, and x!=t`;
  - `tens(x)>=tens(t) and units(x)>units(t)`, and the same with `or x==t+1`;
  - `tens(x)>=tens(t) and units(x)>=units(t)` with `and x!=t`, and with `and x!=t, or x==t+1`.

  They are right on every cond_sametens row and on 19/20 to 1 of cond_oor_edge. They err only where the tens and units
  comparisons disagree across a decade.
- **The ninth class is off by two.** `x > t or x == t-2` is outside the off-by-one family. cond_sametens has 18 of its
  100 rows at d = -2, so it scores 41/50 there, above 0.80.
- **The other baselines.** 68 predicates in 20 classes pass against the dry run's coin-flip plain, and 110 in 41
  classes against the weak plain. All of them are digit-wise or the off-by-two form.

**What would catch them (not implemented; not selected).** A cross-digit identification stratum: rows in different
decades, |d| >= 2, whose units comparison disagrees with the tens comparison.
- Over all 4,824 such rows of the in-range domain, the true rule scores 1.
- 40 of the 41 surviving class representatives score below 0.80 there (36 at most 1/2, 4 at 315/536).
- Only the off-by-two form survives (267/268). It would need rows at other = threshold - 2 in a predicate-gated
  stratum.

See `DESIGN.md` section 14 and `DRYRUN.json` (`shortcut_enumeration.cross_digit_option`).

### 4. Consequences

- **Components.** The rule now has 22 components: G1 has 12 (conditional; cond_oor_val; cond_oor_edge and its
  predicate clause; cond_sametens and its predicate clause; each in-surface and HS).
- **Stages.** E1 has 1,280 rows and E1H 580 (the primary strata + cond_sametens + cond_oor_edge).
- **GPU plan.** 4.82 GPU-h in two turns at the planning rate (turn 1: training + A0 + E1 + E1H, 3.57 h; turn 2:
  re-anchor + E2, 1.25 h). 3.94 h in one turn at the observed rate. Stage names are unchanged.

### Unchanged

- Training: sd only, held-out factors, the 750 rows and every training file (byte for byte).
- Outcome, the two renderings, the primaries and the pooled 'conditional'.
- cond_sametens and its trace - plain clause.
- Validity, DP3 and its labels, G2 to G5 and their bars, the 4b filler constants.
- Exact rational decisions; SE flags.
- The seed-0 screen.

## The rule (verbatim)

> PRE-STATED SCREEN GO RULE, REVISION 5 (T2b-register; supersedes revisions 1-4, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on the two identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; neither is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows. PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone AND on cond_sametens alone (a balanced stratum gives a branch-blind or tens-digit shortcut about 0.50, and an off-by-one variant scores at most 0.75 on cond_oor_edge; this clause has no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let a shortcut through), each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

# T2b-register screen go rule, REVISION 6 (current; pre-stated)

Applied by `scripts/t2br_analysis.verdict` (exact rationals). The same text is `scripts/t2br_common.GO_RULE_TEXT`, is
written into `data/t2br_dev.SOURCE.json`, and is a frozen input (`scripts/t2br_run.py --freeze`). Stated 2026-10-10
(about 03:55) before any `runs/t2b_register` output existed (none exists) and before any model output of this screen
was produced.

Earlier revisions are kept verbatim:
- `GO_RULE_rev1.md`: the first build, about 03:04 (2026-10-09).
- `GO_RULE_rev2.md`: after the review, about 07:54.
- `GO_RULE_rev3.md`: the operator decisions after PREREG-3, about 12:24.
- `GO_RULE_rev4.md`: the operator-selected verifier fixes, about 17:45.
- `GO_RULE_rev5.md`: the operator decisions on the checker's two identification gaps, about 22:50. This is the former
  `GO_RULE.md`, renamed byte for byte (sha256 `af28cb65d62f5610fd6e87454a4c51510e67db964533a5257743cdabef0ec893`; rule
  text `241befab…`).

sha256 of the rule text (one line, UTF-8, no trailing newline): `61888db4b1ec7bf6c18d29c30a7413f29b7e4d5f70efcee7fdf737c7a36a5c84`

## What changed from revision 5, and why

**The checker's finding.** The predicate-gated strata (cond_sametens, cond_oor_edge) only compare numbers in the same
decade and with the same number of digits. Heuristics that are right within a decade therefore passed G1:
- lexicographic string comparison `str(x) > str(t)`;
- digit-sum comparison;
- units-only comparison, with sign- or length-aware variants;
- isolated single-distance exceptions at d = -4..-2, +3, +4.

The revision-5 rule text also overstated the predicate clause ("cannot let a shortcut through").

The operator's decisions (DECISIONS D-42, items (1) to (5)) are applied as decided. Where a decided target is
mathematically infeasible, the infeasibility is proved, the closest feasible version is implemented and the remainder
is reported (sections 2 and 5). The design choices in (2) are mine; each is marked, and the operator can reverse it.

### 1. The extended shortcut family H (decision 1)

`scripts/t2br_shortcuts.py` (new; numpy only). Every member is a predicate 'add iff P(x, t)' of x = reg[other] and
t = threshold. It is evaluated on the domain of this kit's rows: x in 0..96, t in -40..136. H has 3,519 predicates:

- **The enumerated family, unchanged: 3,483 predicates.**
  - Bases: the 122 digit bases (tens / units atoms with floor semantics, AND / OR, the digit-wise lexicographic form)
    and 'x > t + c' for c = -3..+3.
  - Each base with the 27 modifiers at x == t-1, t, t+1.
  - It includes the branch-blind constants TRUE / FALSE.
- **The extension of revision 6: 36 predicates.**
  - Lexicographic string comparison: with sign handling (= Python's `str(x) > str(t)` on the domain) and without it
    (`str(|x|)`); ops `>` and `>=` (4).
  - Digit features: units-only, tens-only and digit-sum comparisons of the magnitudes. Each comes plain, sign-aware,
    length-aware and sign+length-aware; ops `>` and `>=` (24).
  - Magnitude-only `|x| > |t|` and `|x| >= |t|` (2).
  - The isolated single-distance exceptions at d = x - t in -4..-2 and +2..+4, both signs: 'x > t' with the one
    distance flipped (6).
- **True-rule equivalents.** 30 members equal the comparison on the whole domain (for example `x>t`, the digit-wise
  lexicographic form). They are not shortcuts.

### 2. One adversarial identification stratum, cond_adv (decision 2)

**What was decided.** n = 100, branch-balanced 50/50, fresh against the name-skipping harvest registry, its own seed,
drawn by seeded rejection sampling over valid conditional rows, so that EVERY member of H is wrong on at least 40 of
the 100 rows. It is rendered in-surface and in the combined held-out surface. G1 gains the absolute clause "trace
predicate accuracy on cond_adv >= 0.80, both surfaces, decided exactly on counts", with no relative clause.

**What is implemented, as decided:**
- **Size and balance.** 100 rows, 50 ok-true / 50 ok-false, appended as dv1280..dv1379 (held-out-surface rows
  580..679).
- **Seed.** 20261116, unused by this kit (the kit's seeds are 20261109..20261115) and unused anywhere in RCWM-T1.
- **Freshness.** Fresh against the full registry: PREREG-3's name-skipping exclusion union, training, dev, chains,
  cond_sametens and cond_oor_edge. Each row is its own unit.
- **Rows.** Env-exact. |value| 12..40 and neither branch wraps, so a row's 'after' is right iff its branch is right.
- **G1 clause.** The absolute clause on both surfaces, with no trace - plain clause. Every existing G1 clause is kept.

**The 40-row target is infeasible, on ANY 100-row branch-balanced stratum.** Three bounds, each checked on the domain by
`t2br_shortcuts.adv_target_report`:
- **(a) Five members of H have pairwise-disjoint error sets.** They are:
  - `tens(x)>tens(t) | or x==t+1`: wrong only on same-decade ok-true cells;
  - `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)!=units(t)) | and x!=t-1`: wrong only on same-decade ok-false
    cells;
  - `magnitude: |x|>|t|`: wrong only on sign-crossing cells with x <= |t|;
  - `lex signed: str(x)>str(t)`: wrong only on cross-length cells;
  - `digitsum sign+length-aware: >`: wrong only on same-length, cross-decade cells.

  So on any 100 rows one of them is wrong on at most 20. That is below even the 21 needed to fail the 0.80 bar.
- **(b) Three of those members pass revision 5's G1 against a weak plain arm** (magnitude, lex signed, digitsum
  sign+length-aware). So one of them is wrong on at most 33 rows of any 100.
- **(c) The isolated exceptions at d = -4, -3 and -2** are wrong only on ok-false rows at their own distance.
  - Catching one at the 0.80 bar needs >= 21 such rows, and 3 x 21 = 63 > 50 ok-false rows.
  - No other G1 part catches them: cond_sametens has 12 / 12 / 18 rows at d = -4 / -3 / -2.
  - So one of them passes G1 on any such stratum.

**What is implemented instead (my design choice; the operator can reverse it).** The stratum targets what G1 needs:
every member of H that passes revision 5's G1 must now fail it. The targets are met at their maxima, in this order:
1. **The non-exception members.** Every member of H that passes revision 5's G1 against the weak plain arm (126
   predicates) and is not an isolated distance exception (119 predicates) is wrong on >= 33 rows. That is predicate
   accuracy <= 0.67, the maximum by (b).
2. **The distance exceptions.** Those at d = -3, -2, +3 and +4 (and the enumerated family's aliases of -2) are wrong on
   exactly 21 rows (79/100 < 0.80). The one at d = -4 is not caught; by (c) one of -4, -3, -2 must stay uncaught, and
   -4 is the farthest from the boundary.
3. **Cell concentration.** The fewest rows on any one (x, t) cell: 5. At least 25 ok-true rows must be sign-crossing
   cells at d = +3 / +4 with x <= |t|, and only five such cells exist. Then the most distinct cells: 67.

**The design.**
- **Quotas.** An integer program over the 156 signature classes of the domain found the quotas. A signature class is
  the set of cells on which exactly the same members of H are wrong.
- **Rows.** 13 classes are used. One of them is split by digit lengths, so that 1 vs 2, 1 vs 3 and 2 vs 3 digits are
  all drawn: 15 design types (`t2br_shortcuts.ADV_TYPES`, DESIGN.md section 15).
- **Coverage.**
  - Sign-crossing cells: 33 rows.
  - Cross-length cells: 33 rows (2 vs 1, 1 vs 2, 1 vs 3 and 2 vs 3 digits; 5 rows have a three-digit threshold).
  - Cross-decade cells whose units and tens comparisons disagree: 33 rows.
  - Rows at d = -3 / -2 / +3 / +4: 21 each.
- **The draw.** Seeded rejection sampling: 484,756 uniform proposals over the domain. A proposal is accepted iff its
  cell is in a type with quota left and is one of that type's least-used cells, no branch wraps and the row is fresh.
- **Outcome.** Every member's count is fixed by the quotas. The build and `--verify` both check the targets.

**How far the operator's 40 is met.** 2,444 of the 3,489 predicates of H that differ from the comparison are wrong on
>= 40 rows (660 of 1,003 distinct functions), and 2,512 on >= 21. Some members are wrong on 0 rows of cond_adv, for
example `tens sign-aware: >`. Those members already fail another G1 part under every plain baseline.

### 3. Report-only shortcut-agreement analysis (decision 3)

`t2br_analysis.shortcut_agreement`, run on every seed's outputs.
- **What is compared.** The trained trace side's stated cmp.ok, read exactly as the predicate clause reads it
  (`t2br_common.stated_ok`), against the env's predicate and against every member of H. This covers all 860
  conditional rows in-surface and all 580 conditional rows of the held-out-surface bank.
- **Which members.** Members equal to the comparison on the domain are not candidates. Members equal to it on a
  surface's rows are counted, not compared.
- **The flag.** `SHORTCUT_CONSISTENT` if any candidate agrees at least as often as the true predicate, on either
  surface.
- **Report-only.** It never changes the verdict. It is quoted with the verdict, with any GO, in the results JSON
  (`statements`, `shortcut_agreement_flags`) and in the report.
- **Checks.**
  - On gold traces it flags nothing (scorer self-test).
  - In the dry run it flags every scenario whose trace states a member of H. That includes the GO of the d = -4
    exception.

### 4. Scoped rule text and the always-quoted limits (decision 4)

The predicate-clause sentence now says what it excludes: "G1 excludes the members of H, documented by the enumeration
over H against three plain baselines (... all of H except the isolated exception 'other > threshold or other ==
threshold - 4' ...); it cannot exclude every conceivable heuristic".

The rule has a LIMITS sentence. `t2br_common.ALWAYS_QUOTED_LIMITS` renders it beside the verdict in every report and
every verdict statement:
1. G1 identifies the comparison only against H; it cannot exclude every conceivable heuristic.
2. The isolated exception at other = threshold - 4 passes every G1 part.
3. Per-factor held-out renderings are omitted.
4. A GO is a seed-0 screen result.

### 5. The enumeration over H, again (decision 5)

**Method.** `DRYRUN.json` `shortcut_enumeration`. Every member of H answers every conditional with its branch and
exact arithmetic and states itself in cmp.ok. `verdict()` then decides all 14 G1 components exactly, both surfaces,
against three plain baselines:
- chance (exactly 1/2);
- the dry run's coin-flip plain;
- a weak plain (0 on every relative G1 part).

**Results (predicates of H that differ from the comparison):**

| plain baseline | passed revision 5's G1 | of them in the extension | pass revision 6's G1 | caught by cond_adv |
|---|---:|---:|---:|---:|
| chance (1/2) | 41 | 15 | 1 | 40 |
| coin-flip dry-run plain | 84 | 16 | 1 | 83 |
| weak plain (0) | 126 | 16 | 1 | 125 |

**Target ("no member of H passes G1 on either surface"): not met, by one predicate.**
- **The survivor.** `exception d=-4: x>t or x==t-4` passes under all three baselines, on both surfaces.
- **Why.**
  - It is wrong only at d = -4.
  - cond_adv has no row at d = -4, cond_oor_edge has none, and cond_sametens has 12 of 100 (predicate accuracy
    88/100).
  - No relative clause sees it.
  - By (c) above, no 100-row branch-balanced stratum can also catch it while catching -3 and -2.
- **Everything else in H fails G1** under every baseline: the off-by-one family, every digit shortcut, lexicographic,
  digit-sum, units / tens / magnitude-only, and the exceptions at -3, -2, +2, +3, +4.

**Full-pipeline scenarios, against the weak plain arm.** Every member of H that passed revision 5's G1 and was run as a
scenario is NO-GO through exactly `G1 [cond_adv predicate, cond_adv predicate HS]`:
- lex signed; magnitude; digit sum sign+length-aware; units;
- the exceptions at -3, -2, +3, +4.

The exception at -4 is GO, with the report-only flag `SHORTCUT_CONSISTENT`. The tiny test reproduces the GO, the flag
and the lex NO-GO end to end through the runner and the analysis.

### 6. Consequences

- **Components.** The rule now has 24 components; G1 has 14 (cond_adv's predicate clause, in-surface and HS).
- **Stages.** E1 has 1,380 rows and E1H 680. Stage names are unchanged, so the hook draft changes only in its comments.
  The hook draft's stale hook-number comment is fixed: 70 is PREREG-LB, and the prerequisite screen is 75.
- **GPU plan.**
  - At the planning rate: 4.99 GPU-h in two turns (turn 1: training + A0 + E1 + E1H, 3.73 h; turn 2: re-anchor + E2,
    1.26 h).
  - At the observed rate: 4.06 h in one turn.
  - Every gating stage is in turn 1 at both rates.
- **Data.** Every training file and the chain bank are byte-identical to revision 5. The dev bank's first 1,280 lines
  and the held-out-surface bank's first 580 lines are revision 5's files byte for byte; `--verify` reconstructs all
  eight revision-5 data files.

### Unchanged

- Training, the 750 rows and every training file (byte for byte).
- Outcome, the two renderings, the primaries and the pooled 'conditional'.
- cond_sametens, cond_oor_edge and all their clauses.
- Validity, DP3 and its labels, G2 to G5 and their bars, the 4b filler constants (the longest conditional trace is still
  47 tokens).
- Exact rational decisions; SE flags.
- The seed-0 screen.

## The rule (verbatim)

> PRE-STATED SCREEN GO RULE, REVISION 6 (T2b-register; supersedes revisions 1-5, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on the three identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; none is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows; 'cond_adv' (revision 6) = 100 rows, 50 ok true and 50 ok false, drawn by seeded rejection sampling over valid conditional rows (other 0..96, threshold -40..136, |value| 12..40, neither branch wraps) to fixed quotas of 15 design types: 13 signature classes of the extended shortcut family H (sets of (other, threshold) cells on which exactly the same members of H are wrong), one of them split by digit lengths, covering sign-crossing cells, cross-length cells (1 vs 2, 2 vs 1, 1 vs 3 and 2 vs 3 digits), cross-decade cells whose units comparison disagrees with the tens comparison and cells at other - threshold = -3, -2, +3, +4; H (scripts/t2br_shortcuts.py, DESIGN.md section 15; 3,519 predicates) = the enumerated digit and off-by-one family with its modifiers at other = threshold - 1, threshold and threshold + 1 (3,483 predicates, the branch-blind constants included) + lexicographic string comparison with and without sign handling, digit-sum, units-only and tens-only comparisons with sign- and length-aware variants, magnitude-only |other| > |threshold| and the isolated single-distance exceptions at |other - threshold| = 2..4; on cond_adv every member of H that differs from the comparison somewhere on that domain and passes revision 5's G1 against a plain arm at 0 is wrong on at least 33 rows, except the isolated distance exceptions, of which those at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 (the operator's target, every member of H wrong on at least 40 rows, is infeasible on any 100-row branch-balanced stratum: five members of H have pairwise-disjoint error sets). PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone, on cond_sametens alone AND on cond_adv alone (cond_adv has no trace - plain clause), each both in-surface and on the held-out surface (the predicate clauses have no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let through a member of H that they catch; G1 excludes the members of H, documented by the enumeration over H against three plain baselines (DRYRUN.json shortcut_enumeration: every member of H that differs from the comparison, except the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part); it cannot exclude every conceivable heuristic); (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; the shortcut-agreement analysis (the trace side's cmp.ok against the env's predicate and against every member of H over all conditional rows of the dev bank, in each rendering: 'SHORTCUT_CONSISTENT' if a member of H that differs from the predicate on those rows agrees with the trace's cmp.ok at least as well as the predicate does; it never changes the verdict and is quoted with any GO); per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). LIMITS, quoted with every statement of the verdict: G1 identifies the comparison only against H (it excludes the members of H that the enumeration shows failing it, not every conceivable heuristic); the isolated exception 'other > threshold or other == threshold - 4' passes G1 (on any 100-row branch-balanced stratum one of the exceptions at -4, -3, -2 passes the 0.80 bar); per-factor held-out renderings are omitted; a GO is a seed-0 screen result. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

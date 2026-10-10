# T2b-register screen go rule, REVISION 4 (current; pre-stated)

Applied by `scripts/t2br_analysis.verdict` (exact rationals). The same text is `scripts/t2br_common.GO_RULE_TEXT`, is
written into `data/t2br_dev.SOURCE.json`, and is a frozen input (`scripts/t2br_run.py --freeze`). Stated 2026-10-09
(about 17:45) before any `runs/t2b_register` output existed (none exists) and before any model output of this screen
was produced.

Earlier revisions are kept verbatim:
- `GO_RULE_rev1.md`: the first build, about 03:04.
- `GO_RULE_rev2.md`: after the review, about 07:54.
- `GO_RULE_rev3.md`: the operator decisions after PREREG-3, about 12:24. This is the former `GO_RULE.md`, renamed byte
  for byte (sha256 `b342898d5c32cf25a9996779f9b34b484587d3bd543c45cedf55e1ee8489cdfa`).

sha256 of the rule text (one line, UTF-8, no trailing newline): `8f9dc0cb3af79dae2fde091e1878bcf29a6b2a2048783a52379f8da100095d54`

## What changed from revision 3, and why

The operator selected these fixes from the independent verifier's report. They are applied exactly.

### 1. A new identification stratum, `cond_sametens`, and a third G1 clause

**The gap.** Take the shortcut "add iff tens(other) > tens(threshold) or other == threshold + 1". It is exact on every
primary conditional row of the bank:
- Every far pair (|d| 16..96) is separated by the tens digits.
- So is every out-of-range-threshold pair.
- Every cond_oor_val boundary pair is (other = threshold: false; other = threshold + 1: true).

So the shortcut held every revision-3 G1 part, on both surfaces. The same holds for its mirror, "add iff
tens(other) >= tens(threshold) and other != threshold".

Both equality terms follow from the env's STRICT `>` (`rcwmt/env.py`: `reg[other] > threshold`):
- `other == threshold + 1` is the first true value.
- `other != threshold` excludes the last false value.

For a non-strict env the terms would be `other == threshold` and `other != threshold - 1`. No primary stratum
separated either shortcut from the true rule. The 750 training rows, which are fixed and never re-drawn, contradict
the shortcut on only 25 of their 300 conditional rows: the twins at +2..+9 inside one decade. They contradict the
mirror on 59.

**The fix.** `cond_sametens` has 100 rows: 50 branch-balanced minimal pairs.
- **Units digit decides.** In every row tens(other) == tens(threshold), so the branch is decided by the units digit.
- **Distance.** |other - threshold| lies in 2..9. The realized range is 2..7, because both members of a pair lie in
  one decade, so d_true + |d_false| <= 9.
- **Out of range.** |value| is 12..40, outside the trained -8..11. So the stratum is out of range, as asked.
- **In range otherwise.** Registers and thresholds are in range.
- **No wraps.** Neither branch wraps, so a row is right iff its branch is right (given the in-branch arithmetic).
- **Pair split.** 25 pairs move the threshold and 25 move the other register.
- **Freshness.** It is drawn after the chains, with its own seed (20261114), against the full registry: the exclusion
  union, training, dev and chains. So it is fresh, disjoint from every training payload, and every earlier row is
  unchanged.

G1 gains the clause "trace - plain >= +0.25 on cond_sametens alone, both in-surface and on the held-out surface". It is
decided exactly on counts, like the other clauses. cond_sametens is NOT pooled into 'conditional'. E1 now has 1,200
rows, and E1H has 500 (the primary strata + cond_sametens).

**Shown in the dry run (`DRYRUN.md`):**

| scenario | trace on G1 parts | verdict |
|---|---|---|
| `tens_or_eq_plus1` | conditional 1, cond_oor_val 1, cond_sametens exactly 1/2 | NO-GO (failed: G1 [cond_sametens, cond_sametens HS]); every revision-3 G1 part holds |
| `tens_ge_not_eq` (mirror) | the same | the same |
| `go` (true rule) | all 1 | GO |
| `true_rule_digitwise` (tens first, units on a tie) | all 1 | GO |

**Other digit shortcuts (enumeration, reported).** The dry run enumerates 1,098 predicates (380 distinct on the dev
conditional rows). They are tens / units comparisons, combined with AND / OR and with x == t+1, x == t, x != t+1,
x != t. G1 is decided by `verdict()` itself.
- **Chance plain (exactly 1/2 per part).** 73 predicates passed revision 3's G1. 55 of them are now caught by
  cond_sametens.
- **What still passes (chance plain).** 18 predicates pass revision 4's G1. 6 of them equal the true rule on the whole
  domain. The other 12 form 6 classes on the dev rows:
  - `tens(x)>tens(t) or units(x)>units(t)`
  - `tens(x)>tens(t) or units(x)>=units(t)`, and x != t
  - `tens(x)>=tens(t) and units(x)>units(t)`
  - the same, or x == t+1
  - `tens(x)>=tens(t) and units(x)>=units(t)`, and x != t
  - the same, (or x == t+1) and x != t

  Each is a non-lexicographic digit-wise comparison:
  - It is right on every cond_sametens and cond_oor_val row (cond_oor_val 97/100 to 1).
  - It errs only on far / out-of-range-threshold rows where the tens and units comparisons disagree.
  - It scores 61/75 to 259/300 on the pooled conditional, which clears 3/4.
- **Dry run's lower plain.** Against the dry run's lower plain (coin-flip branch, unreduced out of range), 33
  predicates in 12 classes pass, including `units(x)>units(t)` alone.

These are reported, not fixed: the operator selected only the cond_sametens fix. A cross-digit stratum would catch all
33: different decades, |d| >= 2, and a units comparison that disagrees with the tens comparison. Each scores 1/2 or 0
on it. See `DESIGN.md` section 14.

### 2. Amendment 4b now says exactly what the code does, and the code was fixed to the constant definition

**What revision 3's code did.** Revision 3's text said "the shortest run whose filler target is at least as long as
every trace target of that kind". Its code (`min_constant_fill`) compared each row's filler target with THAT row's
trace target. That is a per-row maximum (swap 5, add 14, mix 14, conditional 26), so the filler target of a row with a
short 'after' could be shorter than the kind's longest trace target.

**What revision 4 does.** The code implements the stated constant: the smallest run of ' .' units such that the filler
target is at least as long, in Titan tokens, as the LONGEST trace target of that kind over train + dev and the chains'
gold steps, whatever its 'after'. That is, the kind's shortest filler target is compared with its longest trace target.
- **New constants:** swap 7, add 17, mix 17, conditional 29.
- **Shortest filler = longest trace** in every kind: 26 / 35 / 35 / 47 tokens.
- **One fewer unit fails**, as the tiny test checks with AutoTokenizer.
- **Unchanged:** every filler target of a kind is identical up to 'after'.
- **Residual:** now 2..8 tokens, at least 2 on every row (revision 3: 0..5).

The data files change only in their filler fields: the filler targets, fill_units, fill_residual and the filler token
counts. `--verify` reconstructs all eight revision-3 data files byte for byte from the current rows, using revision 3's
constants.

### 3. Wording

- "Per-factor held-out renderings are not run (GPU budget)" now reads "omitted by choice (a GPU-budget choice, not a
  capacity limit)". At the planning rate they would fit in the second turn.
- The surface audit now covers every held-out-surface stratum.
- The notes on the held-out state orders and on the generation bound are in `DESIGN.md`. They are not rule text.

### Unchanged

- Training: sd only, held-out factors.
- Outcome, the two renderings, the primaries and the pooled 'conditional'.
- Validity.
- DP3 and its labels.
- G1's bars, the conditional and cond_oor_val clauses.
- G2 to G5 and their bars.
- Exact rational decisions; SE flags.
- The seed-0 screen.

## The rule (verbatim)

> PRE-STATED SCREEN GO RULE, REVISION 4 (T2b-register; supersedes revisions 1-3, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on cond_sametens. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. IDENTIFICATION stratum (G1 only; not pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it). FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone AND on cond_sametens alone, each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

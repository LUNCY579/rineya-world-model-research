# T2b-register screen results (s0; training variant sd)

**Overall:** SCREEN (seed 0): NO-GO (failed: G2 [mix, mix HS]; G3 [mix]; G4 [swap])

**Limits (quoted with every statement of the verdict):**

- G1 excludes the members of the enumerated extended shortcut family H (3,519 predicates of other and threshold; scripts/t2br_shortcuts.py, DESIGN.md section 15) that differ from the comparison, as the enumeration over H against three plain baselines documents (DRYRUN.json shortcut_enumeration), EXCEPT the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part (on any 100-row branch-balanced stratum one of the exceptions at other - threshold = -4, -3, -2 passes the 0.80 predicate bar; cond_adv catches -3 and -2).
- Heuristics OUTSIDE H are NOT excluded, and natural ones pass every G1 part (the revision-6 independent check, confirmed with verdict() on exact counts, both surfaces; DESIGN.md section 14): isolated single-distance exceptions at |other - threshold| >= 5 (H has them only at 2..4; e.g. 'other > threshold except at other - threshold = +5') and flipped distance bands (e.g. 5..9); three-digit-threshold slips (the hundreds digit dropped, only the first two digits read, a threshold above 96 reduced modulo 97, string comparison only against three-digit thresholds, always adding when the threshold is above 96); a register of 0 never counted above a negative threshold; the sign of a threshold at or below -10 dropped; a circular comparison ((other - threshold) mod 97 in 1..48); string comparison only for one-digit thresholds or one-digit registers; guessing on the far rows (|other - threshold| > 15) by string comparison or a prior (other > 48, threshold < 48), and, against a plain arm well below chance on the pooled conditional (the dry run's coin-flip plain at 5/12, or a weak plain arm), also by always adding, always subtracting or the units digit; and any rule that departs from the comparison only on cells that no G1 stratum contains (e.g. in-range cross-decade cells at |other - threshold| 5..15, decided by the units digit or with the branch flipped). The report-only shortcut-agreement flag compares the members of H only, so its absence does not rule these out.
- cond_adv catches its targets only at narrow margins: the exceptions at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 of its 100 rows (79/100, one row below the 0.80 bar), and the signed string comparison, the magnitude comparison and the sign+length-aware digit-sum and units comparisons on exactly 33 (67/100). A trace that follows a caught member on only part of its rows can pass: following it everywhere except on any 1 of those 21 rows, or any 13 of those 33, passes G1; and a trace that departs from the comparison on at most 1/5 of each predicate-gated stratum's rows passes the predicate clauses.
- Per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice: a held-out-surface loss cannot be attributed to one factor.
- A GO is a seed-0 screen result: the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

Pre-stated rule, revision 7 (verbatim):

> PRE-STATED SCREEN GO RULE, REVISION 7 (T2b-register; supersedes revisions 1-6, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on the three identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; none is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows; 'cond_adv' (revision 6) = 100 rows, 50 ok true and 50 ok false, drawn by seeded rejection sampling over valid conditional rows (other 0..96, threshold -40..136, |value| 12..40, neither branch wraps) to fixed quotas of 15 design types: 13 signature classes of the extended shortcut family H (sets of (other, threshold) cells on which exactly the same members of H are wrong), one of them split by digit lengths, covering sign-crossing cells, cross-length cells (1 vs 2, 2 vs 1, 1 vs 3 and 2 vs 3 digits), cross-decade cells whose units comparison disagrees with the tens comparison and cells at other - threshold = -3, -2, +3, +4; H (scripts/t2br_shortcuts.py, DESIGN.md section 15; 3,519 predicates) = the enumerated digit and off-by-one family with its modifiers at other = threshold - 1, threshold and threshold + 1 (3,483 predicates, the branch-blind constants included) + lexicographic string comparison with and without sign handling, digit-sum, units-only and tens-only comparisons with sign- and length-aware variants, magnitude-only |other| > |threshold| and the isolated single-distance exceptions at |other - threshold| = 2..4; on cond_adv every member of H that differs from the comparison somewhere on that domain and passes revision 5's G1 against a plain arm at 0 is wrong on at least 33 rows, except the isolated distance exceptions, of which those at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 (the operator's target, every member of H wrong on at least 40 rows, is infeasible on any 100-row branch-balanced stratum: five members of H have pairwise-disjoint error sets). PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone, on cond_sametens alone AND on cond_adv alone (cond_adv has no trace - plain clause), each both in-surface and on the held-out surface (the predicate clauses have no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let through a member of H that they catch; G1 excludes the members of the enumerated family H that differ from the comparison, except the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part, as documented by the enumeration over H against three plain baselines (DRYRUN.json shortcut_enumeration); it does NOT exclude heuristics outside H, and natural ones pass every G1 part (LIMITS)); (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; the shortcut-agreement analysis (the trace side's cmp.ok against the env's predicate and against every member of H over all conditional rows of the dev bank, in each rendering: 'SHORTCUT_CONSISTENT' if a member of H that differs from the predicate on those rows agrees with the trace's cmp.ok at least as well as the predicate does; it never changes the verdict; it is quoted with every statement of the verdict, so with any GO, and whenever it is set the OVERALL headline carries it beside the verdict, with a pointer to the LIMITS); per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity limit). LIMITS, quoted with every statement of the verdict (t2br_common.ALWAYS_QUOTED_LIMITS; DESIGN.md section 14): (1) G1 excludes the members of the enumerated family H that differ from the comparison EXCEPT the isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part (on any 100-row branch-balanced stratum one of the exceptions at other - threshold = -4, -3, -2 passes the 0.80 bar); (2) heuristics OUTSIDE H are NOT excluded, and natural ones pass every G1 part (the revision-6 independent check, confirmed by verdict() on exact counts): isolated single-distance exceptions at |other - threshold| >= 5 (H has them only at 2..4) and flipped distance bands (e.g. 5..9); three-digit-threshold slips (the hundreds digit dropped, only the first two digits read, a threshold above 96 reduced modulo 97, string comparison only against three-digit thresholds, always adding when the threshold is above 96); a register of 0 never counted above a negative threshold; the sign of a threshold at or below -10 dropped; a circular comparison ((other - threshold) mod 97 in 1..48); string comparison only for one-digit thresholds or one-digit registers; guessing on the far rows (|other - threshold| > 15) by string comparison or a prior (other > 48, threshold < 48), and, against a plain arm well below chance on the pooled conditional (the dry run's coin-flip plain at 5/12, or a weak plain arm), also by always adding, always subtracting or the units digit; any rule that departs from the comparison only on cells that no G1 stratum contains (e.g. in-range cross-decade cells at |other - threshold| 5..15, decided by the units digit or with the branch flipped); the report-only shortcut-agreement flag compares the members of H only, so its absence does not rule these out; (3) cond_adv catches its targets only at narrow margins: the exceptions at other - threshold = -3, -2, +3, +4 are wrong on exactly 21 of its 100 rows (79/100, one row below the bar) and the signed string, magnitude, and sign+length-aware digit-sum and units comparisons on exactly 33 (67/100), so a trace that follows a caught member on only part of its rows can pass (following it everywhere except on any 1 of those 21 rows, or any 13 of those 33, passes G1), and a trace that departs from the comparison on at most 1/5 of each predicate-gated stratum's rows passes the predicate clauses; (4) per-factor held-out renderings are omitted; (5) a GO is a seed-0 screen result. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

## Seed s0: NO-GO (failed: G2 [mix, mix HS]; G3 [mix]; G4 [swap])

**Quoted with the verdict** (every verdict): shortcut agreement (report-only; never changes the verdict): no member of H agrees with the trace's cmp.ok as well as the true predicate - in-surface: true predicate 858/860, best member of H 'exception d=-4: x>t or x==t-4' 844/860; held-out surface: true predicate 560/580, best member of H 'exception d=-4: x>t or x==t-4' 548/580.

- in: rows 860, stated ok 860, true predicate 858, top members of H: `exception d=-4: x>t or x==t-4` 844; `exception d=+4: x>t and x!=t+4` 823; `exception d=-3: x>t or x==t-3` 822; `exception d=+2: x>t and x!=t+2` 819; `x>t+2 | and x!=t, or x==t+1` 819 (0 members equal the predicate on these rows; flag: none)
- hs: rows 580, stated ok 580, true predicate 560, top members of H: `exception d=-4: x>t or x==t-4` 548; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>=units(t)) | or x==t-1, and x!=t` 542; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>=units(t)) | or x==t-1, and x!=t, or x==t+1` 542; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>units(t)) | or x==t-1` 542; `tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>units(t)) | or x==t-1, and x!=t` 542 (0 members equal the predicate on these rows; flag: none)

| condition | holds | failed parts |
|---|---|---|
| G1 trace-plain >= +0.25 on conditional (OOR pooled), on cond_oor_val, cond_oor_edge and cond_sametens alone, and trace predicate accuracy >= 0.80 on cond_oor_edge, cond_sametens and cond_adv, in-surface and held-out surface | yes | - |
| G2 trace-plain >= +0.30 on mix (OOR), in-surface and held-out surface | NO | mix, mix HS |
| G3 filler below trace: trace-filler >= +0.10 on each in-surface primary (4b filler) | NO | mix |
| G4 controls: trace ON-OFF >= -0.05 on swap and add_nowrap | NO | swap |
| G5 in-distribution non-inferiority: trace-plain >= -0.05 on conditional_in and mix_in | yes | - |
| V filler side valid: filler in-surface format_ok >= 0.95 on each primary | yes | - |

Validity (in-surface): trace format_ok {'primary_conditional': 1.0, 'primary_mix': 1.0}, filler format_ok {'primary_conditional': 1.0, 'primary_mix': 1.0} (bar 0.95 on each primary).

Trace predicate accuracy (G1's absolute clause, bar 0.80; strict parser, cmp keys x, thr, ok in order, boolean ok equal to the env's predicate): cond_oor_edge 1.000 in-surface / 0.975 held-out; cond_sametens 1.000 in-surface / 0.980 held-out; cond_adv 1.000 in-surface / 0.990 held-out.

| component | surface | contrast | outcome | value (exact) | bar | holds | SE | within 1 SE |
|---|---|---|---|---|---|---|---|---|
| G1 | in | trace-plain | primary_conditional | +0.5133 (77/150) | >= +0.25 | yes | 0.020409 |  |
| G1 | in | trace-plain | cond_oor_val | +0.4600 (23/50) | >= +0.25 | yes | 0.04847 |  |
| G1 | in | trace-plain | cond_oor_edge | +0.4875 (39/80) | >= +0.25 | yes | 0.057399 |  |
| G1 | in | trace | cond_oor_edge_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | in | trace-plain | cond_sametens | +0.4300 (43/100) | >= +0.25 | yes | 0.045919 |  |
| G1 | in | trace | cond_sametens_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | in | trace | cond_adv_pred | +1.0000 (1) | >= +0.80 | yes | 0.0 |  |
| G1 | hs | trace-plain | primary_conditional | +0.4200 (21/50) | >= +0.25 | yes | 0.027194 |  |
| G1 | hs | trace-plain | cond_oor_val | +0.3500 (7/20) | >= +0.25 | yes | 0.045919 |  |
| G1 | hs | trace-plain | cond_oor_edge | +0.4750 (19/40) | >= +0.25 | yes | 0.063777 |  |
| G1 | hs | trace | cond_oor_edge_pred | +0.9750 (39/40) | >= +0.80 | yes | 0.015944 |  |
| G1 | hs | trace-plain | cond_sametens | +0.4400 (11/25) | >= +0.25 | yes | 0.051021 |  |
| G1 | hs | trace | cond_sametens_pred | +0.9800 (49/50) | >= +0.80 | yes | 0.012755 |  |
| G1 | hs | trace | cond_adv_pred | +0.9900 (99/100) | >= +0.80 | yes | 0.008597 |  |
| G2 | in | trace-plain | primary_mix | +0.0100 (1/100) | >= +0.30 | NO | 0.020409 |  |
| G2 | hs | trace-plain | primary_mix | +0.0100 (1/100) | >= +0.30 | NO | 0.02046 |  |
| G3 | in | trace-filler | primary_conditional | +0.4900 (49/100) | >= +0.10 | yes | 0.022092 |  |
| G3 | in | trace-filler | primary_mix | +0.0100 (1/100) | >= +0.10 | NO | 0.020409 |  |
| G4 | in | trace-off | swap | -1.0000 (-1) | >= -0.05 | NO | 0.0 |  |
| G4 | in | trace-off | add_nowrap | +0.0250 (1/40) | >= -0.05 | yes | 0.035077 |  |
| G5 | in | trace-plain | conditional_in | +0.4925 (921/1870) | >= -0.05 | yes | 0.017832 |  |
| G5 | in | trace-plain | mix_in | -0.0300 (-3/100) | >= -0.05 | yes | 0.02296 | FLAG |
| DP3 | in | trace-filler | primary_conditional | +0.4900 (49/100) | < +0.10 | NO | 0.022092 |  |
| DP3 | in | trace-filler | primary_mix | +0.0100 (1/100) | < +0.10 | yes | 0.020409 |  |

In-surface (Titan's verbatim frame):

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.330 | 0.387 | 0.900 | 0.410 | +0.513 [+0.473, +0.553] | +0.490 [+0.447, +0.533] |
| primary_mix | 0.000 | 0.020 | 0.030 | 0.020 | +0.010 [-0.030, +0.050] | +0.010 [-0.030, +0.050] |
| conditional_in | 0.410 | 0.488 | 0.981 | 0.475 | +0.492 [+0.457, +0.527] | +0.506 [+0.465, +0.546] |
| mix_in | 0.000 | 0.040 | 0.010 | 0.020 | -0.030 [-0.080, +0.010] | -0.010 [-0.050, +0.020] |
| mix | 0.000 | 0.040 | 0.010 | 0.020 | -0.030 [-0.080, +0.010] | -0.010 [-0.050, +0.020] |
| wrap | 0.000 | 0.450 | 1.000 | 0.438 | +0.550 [+0.438, +0.662] | +0.562 [+0.463, +0.675] |
| cond_xm1 | 0.450 | 0.750 | 1.000 | 0.600 | +0.250 [+0.133, +0.367] | +0.400 [+0.283, +0.517] |
| cond_x0 | 0.433 | 0.767 | 1.000 | 0.600 | +0.233 [+0.133, +0.350] | +0.400 [+0.283, +0.517] |
| cond_xp1 | 0.350 | 0.233 | 0.983 | 0.350 | +0.750 [+0.633, +0.850] | +0.633 [+0.517, +0.750] |
| cond_mid | 0.440 | 0.440 | 0.960 | 0.450 | +0.520 [+0.460, +0.580] | +0.510 [+0.440, +0.580] |
| swap | 1.000 | 0.025 | 0.000 | 0.000 | -0.025 [-0.062, +0.000] | +0.000 [+0.000, +0.000] |
| add_nowrap | 0.900 | 0.438 | 0.925 | 0.225 | +0.487 [+0.375, +0.600] | +0.700 [+0.588, +0.812] |
| mix_oor | 0.000 | 0.020 | 0.030 | 0.020 | +0.010 [-0.030, +0.050] | +0.010 [-0.030, +0.050] |
| wrap_oor | 0.000 | 0.037 | 0.688 | 0.037 | +0.650 [+0.537, +0.750] | +0.650 [+0.537, +0.762] |
| cond_oor_val | 0.180 | 0.260 | 0.720 | 0.310 | +0.460 [+0.360, +0.550] | +0.410 [+0.330, +0.490] |
| cond_oor_thr | 0.490 | 0.440 | 0.990 | 0.490 | +0.550 [+0.490, +0.610] | +0.500 [+0.430, +0.570] |
| cond_far | 0.320 | 0.460 | 0.990 | 0.430 | +0.530 [+0.490, +0.570] | +0.560 [+0.510, +0.610] |
| cond_sametens | 0.190 | 0.320 | 0.750 | 0.300 | +0.430 [+0.340, +0.520] | +0.450 [+0.360, +0.530] |
| cond_oor_edge | 0.237 | 0.362 | 0.850 | 0.375 | +0.487 [+0.375, +0.600] | +0.475 [+0.338, +0.613] |
| cond_adv | 0.200 | 0.360 | 0.750 | 0.380 | +0.390 [+0.277, +0.491] | +0.370 [+0.269, +0.466] |

Held-out surface (paraphrase 4 + field names 4 + held-out key orders), the primary rows and the identification strata:

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.143 | 0.237 | 0.657 | 0.223 | +0.420 [+0.367, +0.473] | +0.433 [+0.380, +0.487] |
| cond_oor_val | 0.030 | 0.060 | 0.410 | 0.050 | +0.350 [+0.260, +0.440] | +0.360 [+0.270, +0.460] |
| cond_oor_thr | 0.230 | 0.310 | 0.750 | 0.320 | +0.440 [+0.340, +0.530] | +0.430 [+0.340, +0.520] |
| cond_far | 0.170 | 0.340 | 0.810 | 0.300 | +0.470 [+0.390, +0.550] | +0.510 [+0.420, +0.600] |
| primary_mix | 0.000 | 0.020 | 0.030 | 0.040 | +0.010 [-0.030, +0.050] | -0.010 [-0.060, +0.040] |
| mix_oor | 0.000 | 0.020 | 0.030 | 0.040 | +0.010 [-0.030, +0.050] | -0.010 [-0.060, +0.040] |
| cond_sametens | 0.000 | 0.070 | 0.510 | 0.060 | +0.440 [+0.340, +0.540] | +0.450 [+0.350, +0.550] |
| cond_oor_edge | 0.025 | 0.050 | 0.525 | 0.100 | +0.475 [+0.350, +0.600] | +0.425 [+0.300, +0.550] |
| cond_adv | 0.010 | 0.050 | 0.460 | 0.050 | +0.410 [+0.319, +0.495] | +0.410 [+0.330, +0.491] |

Surface audit (held-out minus in-surface, paired by row; rows right only in-surface / only held-out):

| outcome | off | plain | trace | filler |
|---|---|---|---|---|
| primary_conditional | -0.187 [-0.243, -0.133] (78/22) | -0.150 [-0.197, -0.103] (77/32) | -0.243 [-0.297, -0.193] (79/6) | -0.187 [-0.237, -0.137] (80/24) |
| cond_oor_val | -0.150 [-0.220, -0.070] (17/2) | -0.200 [-0.280, -0.120] (24/4) | -0.310 [-0.420, -0.190] (36/5) | -0.260 [-0.340, -0.180] (28/2) |
| cond_oor_thr | -0.260 [-0.380, -0.140] (39/13) | -0.130 [-0.210, -0.050] (26/13) | -0.240 [-0.320, -0.160] (24/0) | -0.170 [-0.270, -0.070] (28/11) |
| cond_far | -0.150 [-0.240, -0.060] (22/7) | -0.120 [-0.200, -0.040] (27/15) | -0.180 [-0.250, -0.120] (19/1) | -0.130 [-0.210, -0.050] (24/11) |
| primary_mix | +0.000 [+0.000, +0.000] (0/0) | +0.000 [-0.040, +0.040] (2/2) | +0.000 [+0.000, +0.000] (0/0) | +0.020 [-0.030, +0.070] (2/4) |
| mix_oor | +0.000 [+0.000, +0.000] (0/0) | +0.000 [-0.040, +0.040] (2/2) | +0.000 [+0.000, +0.000] (0/0) | +0.020 [-0.030, +0.070] (2/4) |
| cond_sametens | -0.190 [-0.260, -0.120] (19/0) | -0.250 [-0.320, -0.180] (30/5) | -0.240 [-0.360, -0.120] (33/9) | -0.240 [-0.310, -0.170] (29/5) |
| cond_oor_edge | -0.212 [-0.312, -0.113] (18/1) | -0.312 [-0.412, -0.212] (29/4) | -0.325 [-0.450, -0.212] (27/1) | -0.275 [-0.375, -0.175] (27/5) |
| cond_adv | -0.190 [-0.280, -0.114] (19/0) | -0.310 [-0.443, -0.189] (34/3) | -0.290 [-0.391, -0.195] (33/4) | -0.330 [-0.444, -0.226] (34/1) |

Controls (a G4 loss shared by every trained arm is a property of the data: every training add wraps):

- swap: trace-off -1.000, plain-off -0.975, filler-off -1.000, trace-plain -0.025
- add_nowrap: trace-off +0.025, plain-off -0.463, filler-off -0.675, trace-plain +0.487

Trace fields (trace side, in-surface):

| stratum | format | predicate (pred_ok) | cmp_ok | ok_ok | calc_ok | after follows calc |
|---|---|---|---|---|---|---|
| mix | 1.0 | - | 1.0 | - | 0.01 | 1.0 |
| wrap | 1.0 | - | 1.0 | - | 1.0 | 1.0 |
| cond_xm1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_x0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_xp1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9833 | 1.0 |
| cond_mid | 1.0 | 0.99 | 0.99 | 0.99 | 0.96 | 1.0 |
| swap | 1.0 | - | 1.0 | - | 0.0 | 0.0 |
| add_nowrap | 1.0 | - | 1.0 | - | 0.925 | 1.0 |
| mix_oor | 1.0 | - | 1.0 | - | 0.03 | 1.0 |
| wrap_oor | 1.0 | - | 1.0 | - | 0.6875 | 1.0 |
| cond_oor_val | 1.0 | 1.0 | 1.0 | 1.0 | 0.72 | 1.0 |
| cond_oor_thr | 1.0 | 1.0 | 1.0 | 1.0 | 0.99 | 1.0 |
| cond_far | 1.0 | 0.99 | 0.99 | 0.99 | 0.99 | 1.0 |
| cond_sametens | 1.0 | 1.0 | 1.0 | 1.0 | 0.75 | 1.0 |
| cond_oor_edge | 1.0 | 1.0 | 1.0 | 1.0 | 0.85 | 1.0 |
| cond_adv | 1.0 | 1.0 | 1.0 | 1.0 | 0.75 | 1.0 |

Filler residual covariate (reported, not gating; amendment 4b: the fill is constant per action kind, so its length carries no answer information; residual = filler tokens - that row's trace tokens):

| cell | n | residual mean / median / range | filler acc, residual <= median (n) | > median (n) | held-out: <= median | > median |
|---|---|---|---|---|---|---|
| cond_oor_val | 100 | 4.19 / 4.0 / [3, 7] | 0.3614 (83) | 0.0588 (17) | 0.0602 | 0.0 |
| cond_oor_thr | 100 | 3.91 / 4.0 / [3, 7] | 0.5125 (80) | 0.4 (20) | 0.3375 | 0.25 |
| cond_far | 100 | 4.38 / 4.0 / [4, 7] | 0.4286 (70) | 0.4333 (30) | 0.3 | 0.3 |
| mix_oor | 100 | 3.61 / 4.0 / [3, 6] | 0.0208 (96) | 0.0 (4) | 0.0417 | 0.0 |
| cond_sametens | 100 | 4.48 / 4.0 / [4, 6] | 0.3289 (76) | 0.2083 (24) | 0.0658 | 0.0417 |
| cond_oor_edge | 80 | 4.4875 / 4.0 / [4, 8] | 0.4194 (62) | 0.2222 (18) | 0.0806 | 0.1667 |
| cond_adv | 100 | 4.95 / 5.0 / [3, 8] | 0.4559 (68) | 0.2188 (32) | 0.0294 | 0.0938 |
| primary_conditional | 300 | 4.16 / 4.0 / [3, 7] | 0.4335 (233) | 0.3284 (67) | 0.2275 | 0.209 |
| primary_mix | 100 | 3.61 / 4.0 / [3, 6] | 0.0208 (96) | 0.0 (4) | 0.0417 | 0.0 |

Branch choice read from 'after' (correct / wrong / neither), conditional strata (in-surface):

- off: cond_xm1 0.45/0.35/0.2; cond_x0 0.4333/0.3667/0.2; cond_xp1 0.35/0.45/0.2; cond_mid 0.44/0.46/0.1; cond_oor_val 0.19/0.17/0.64; cond_oor_thr 0.51/0.44/0.05; cond_far 0.32/0.47/0.21; cond_sametens 0.19/0.2/0.61; cond_oor_edge 0.2375/0.2625/0.5; cond_adv 0.2/0.2/0.6
- plain: cond_xm1 0.75/0.1833/0.0667; cond_x0 0.7667/0.2/0.0333; cond_xp1 0.2333/0.7167/0.05; cond_mid 0.44/0.45/0.11; cond_oor_val 0.26/0.27/0.47; cond_oor_thr 0.44/0.41/0.15; cond_far 0.46/0.47/0.07; cond_sametens 0.32/0.27/0.41; cond_oor_edge 0.3625/0.375/0.2625; cond_adv 0.36/0.27/0.37
- trace: cond_xm1 1.0/0.0/0.0; cond_x0 1.0/0.0/0.0; cond_xp1 0.9833/0.0/0.0167; cond_mid 0.96/0.01/0.03; cond_oor_val 0.72/0.0/0.28; cond_oor_thr 0.99/0.0/0.01; cond_far 0.99/0.01/0.0; cond_sametens 0.75/0.0/0.25; cond_oor_edge 0.85/0.0/0.15; cond_adv 0.75/0.0/0.25
- filler: cond_xm1 0.6/0.3167/0.0833; cond_x0 0.6/0.3167/0.0833; cond_xp1 0.35/0.55/0.1; cond_mid 0.45/0.42/0.13; cond_oor_val 0.31/0.3/0.39; cond_oor_thr 0.49/0.36/0.15; cond_far 0.43/0.49/0.08; cond_sametens 0.3/0.31/0.39; cond_oor_edge 0.375/0.35/0.275; cond_adv 0.38/0.25/0.37

Chains (k=4; normalised regret / exact; mean [95% CI]):

| stratum | side | regret k4 | exact k4 | product-law predicted k4 | observed |
|---|---|---|---|---|---|
| CH_IN | off | 0.340 [+0.286, +0.394] | 0.129 | 0.1062 | 0.1289 |
| CH_IN | plain | 0.665 [+0.609, +0.726] | 0.004 | 0.0047 | 0.0039 |
| CH_IN | trace | 0.633 [+0.592, +0.675] | 0.031 | 0.044 | 0.0312 |
| CH_IN | filler | 0.756 [+0.705, +0.808] | 0.000 | 0.0009 | 0.0 |
| CH_X | off | 0.578 [+0.503, +0.654] | 0.016 | 0.0121 | 0.0156 |
| CH_X | plain | 0.726 [+0.645, +0.804] | 0.000 | 0.0003 | 0.0 |
| CH_X | trace | 0.658 [+0.597, +0.721] | 0.008 | 0.0046 | 0.0078 |
| CH_X | filler | 0.808 [+0.718, +0.897] | 0.000 | 0.0 | 0.0 |
| ALL | off | 0.419 [+0.366, +0.472] | 0.091 | 0.0675 | 0.0911 |
| ALL | plain | 0.684 [+0.637, +0.732] | 0.003 | 0.0025 | 0.0026 |
| ALL | trace | 0.642 [+0.607, +0.678] | 0.023 | 0.0262 | 0.0234 |
| ALL | filler | 0.773 [+0.727, +0.819] | 0.000 | 0.0003 | 0.0 |


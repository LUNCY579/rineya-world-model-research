<!-- synthetic dry run: NOT model output -->
# T2b-register screen results (s0; training variant sd)

**Overall:** SCREEN (seed 0): GO [SYNTHETIC DRY RUN]

Pre-stated rule, revision 3 (verbatim):

> PRE-STATED SCREEN GO RULE, REVISION 3 (T2b-register; supersedes revisions 1 and 2, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. FILLER (amendment 4b): the fill is a run of ' .' of a CONSTANT length per action kind (swap 5, add 14, mix 14, conditional 26 units: the shortest run whose filler target is at least as long as every trace target of that kind in train + dev), so every filler target of a kind is identical up to 'after' and its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional AND >= +0.25 on cond_oor_val alone, each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and primary stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings are not run (GPU budget). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

## Seed s0: GO

| condition | holds | failed parts |
|---|---|---|
| G1 trace-plain >= +0.25 on conditional (OOR pooled) and on cond_oor_val alone, in-surface and held-out surface | yes | - |
| G2 trace-plain >= +0.30 on mix (OOR), in-surface and held-out surface | yes | - |
| G3 filler below trace: trace-filler >= +0.10 on each in-surface primary (4b filler) | yes | - |
| G4 controls: trace ON-OFF >= -0.05 on swap and add_nowrap | yes | - |
| G5 in-distribution non-inferiority: trace-plain >= -0.05 on conditional_in and mix_in | yes | - |
| V filler side valid: filler in-surface format_ok >= 0.95 on each primary | yes | - |

Validity (in-surface): trace format_ok {'primary_conditional': 1.0, 'primary_mix': 1.0}, filler format_ok {'primary_conditional': 1.0, 'primary_mix': 1.0} (bar 0.95 on each primary).

| component | surface | contrast | outcome | value (exact) | bar | holds | SE | within 1 SE |
|---|---|---|---|---|---|---|---|---|
| G1 | in | trace-plain | primary_conditional | +0.5833 (7/12) | >= +0.25 | yes | 0.02722 |  |
| G1 | in | trace-plain | cond_oor_val | +0.6300 (63/100) | >= +0.25 | yes | 0.040817 |  |
| G1 | hs | trace-plain | primary_conditional | +0.5833 (7/12) | >= +0.25 | yes | 0.02722 |  |
| G1 | hs | trace-plain | cond_oor_val | +0.6300 (63/100) | >= +0.25 | yes | 0.040817 |  |
| G2 | in | trace-plain | primary_mix | +0.5000 (1/2) | >= +0.30 | yes | 0.051021 |  |
| G2 | hs | trace-plain | primary_mix | +0.5000 (1/2) | >= +0.30 | yes | 0.051021 |  |
| G3 | in | trace-filler | primary_conditional | +0.5833 (7/12) | >= +0.10 | yes | 0.02722 |  |
| G3 | in | trace-filler | primary_mix | +0.5000 (1/2) | >= +0.10 | yes | 0.051021 |  |
| G4 | in | trace-off | swap | +0.0000 (0) | >= -0.05 | yes | 0.0 |  |
| G4 | in | trace-off | add_nowrap | +0.0000 (0) | >= -0.05 | yes | 0.0 |  |
| G5 | in | trace-plain | conditional_in | +0.5045 (111/220) | >= -0.05 | yes | 0.031608 |  |
| G5 | in | trace-plain | mix_in | +0.0000 (0) | >= -0.05 | yes | 0.0 |  |
| DP3 | in | trace-filler | primary_conditional | +0.5833 (7/12) | < +0.10 | NO | 0.02722 |  |
| DP3 | in | trace-filler | primary_mix | +0.5000 (1/2) | < +0.10 | NO | 0.051021 |  |

In-surface (Titan's verbatim frame):

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.417 | 0.417 | 1.000 | 0.417 | +0.583 [+0.530, +0.637] | +0.583 [+0.530, +0.637] |
| primary_mix | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| conditional_in | 0.485 | 0.495 | 1.000 | 0.495 | +0.504 [+0.441, +0.565] | +0.504 [+0.441, +0.565] |
| mix_in | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mix | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| wrap | 0.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| cond_xm1 | 0.517 | 0.517 | 1.000 | 0.517 | +0.483 [+0.350, +0.617] | +0.483 [+0.350, +0.617] |
| cond_x0 | 0.483 | 0.517 | 1.000 | 0.517 | +0.483 [+0.350, +0.617] | +0.483 [+0.350, +0.617] |
| cond_xp1 | 0.483 | 0.483 | 1.000 | 0.483 | +0.517 [+0.383, +0.650] | +0.517 [+0.383, +0.650] |
| cond_mid | 0.470 | 0.480 | 1.000 | 0.480 | +0.520 [+0.410, +0.630] | +0.520 [+0.410, +0.630] |
| swap | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| add_nowrap | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mix_oor | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| wrap_oor | 0.000 | 0.000 | 1.000 | 0.000 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] |
| cond_oor_val | 0.370 | 0.370 | 1.000 | 0.370 | +0.630 [+0.550, +0.710] | +0.630 [+0.550, +0.710] |
| cond_oor_thr | 0.420 | 0.420 | 1.000 | 0.420 | +0.580 [+0.490, +0.670] | +0.580 [+0.490, +0.670] |
| cond_far | 0.460 | 0.460 | 1.000 | 0.460 | +0.540 [+0.450, +0.640] | +0.540 [+0.450, +0.640] |

Held-out surface (paraphrase 4 + field names 4 + held-out key orders), the primary rows:

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.417 | 0.417 | 1.000 | 0.417 | +0.583 [+0.530, +0.637] | +0.583 [+0.530, +0.637] |
| cond_oor_val | 0.370 | 0.370 | 1.000 | 0.370 | +0.630 [+0.550, +0.710] | +0.630 [+0.550, +0.710] |
| cond_oor_thr | 0.420 | 0.420 | 1.000 | 0.420 | +0.580 [+0.490, +0.670] | +0.580 [+0.490, +0.670] |
| cond_far | 0.460 | 0.460 | 1.000 | 0.460 | +0.540 [+0.450, +0.640] | +0.540 [+0.450, +0.640] |
| primary_mix | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| mix_oor | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |

Surface audit (held-out minus in-surface, paired by row; rows right only in-surface / only held-out):

| outcome | off | plain | trace | filler |
|---|---|---|---|---|
| primary_conditional | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_oor_val | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_oor_thr | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| cond_far | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| primary_mix | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |
| mix_oor | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) | +0.000 [+0.000, +0.000] (0/0) |

Controls (a G4 loss shared by every trained arm is a property of the data: every training add wraps):

- swap: trace-off +0.000, plain-off +0.000, filler-off +0.000, trace-plain +0.000
- add_nowrap: trace-off +0.000, plain-off +0.000, filler-off +0.000, trace-plain +0.000

Trace fields (trace side, in-surface):

| stratum | format | cmp_ok | ok_ok | calc_ok | after follows calc |
|---|---|---|---|---|---|
| mix | 1.0 | 1.0 | - | 1.0 | 1.0 |
| wrap | 1.0 | 1.0 | - | 1.0 | 1.0 |
| cond_xm1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_x0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_xp1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_mid | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| swap | 1.0 | 1.0 | - | 1.0 | 1.0 |
| add_nowrap | 1.0 | 1.0 | - | 1.0 | 1.0 |
| mix_oor | 1.0 | 1.0 | - | 1.0 | 1.0 |
| wrap_oor | 1.0 | 1.0 | - | 1.0 | 1.0 |
| cond_oor_val | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_oor_thr | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_far | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Filler residual covariate (reported, not gating; amendment 4b: the fill is constant per action kind, so its length carries no answer information; residual = filler tokens - that row's trace tokens):

| cell | n | residual mean / median / range | filler acc, residual <= median (n) | > median (n) | held-out: <= median | > median |
|---|---|---|---|---|---|---|
| cond_oor_val | 100 | 1.19 / 1.0 / [0, 4] | 0.3855 (83) | 0.2941 (17) | 0.3855 | 0.2941 |
| cond_oor_thr | 100 | 0.91 / 1.0 / [0, 4] | 0.4625 (80) | 0.25 (20) | 0.4625 | 0.25 |
| cond_far | 100 | 1.38 / 1.0 / [1, 4] | 0.5143 (70) | 0.3333 (30) | 0.5143 | 0.3333 |
| mix_oor | 100 | 0.61 / 1.0 / [0, 3] | 0.4792 (96) | 1.0 (4) | 0.4792 | 1.0 |
| primary_conditional | 300 | 1.16 / 1.0 / [0, 4] | 0.4506 (233) | 0.2985 (67) | 0.4506 | 0.2985 |
| primary_mix | 100 | 0.61 / 1.0 / [0, 3] | 0.4792 (96) | 1.0 (4) | 0.4792 | 1.0 |

Branch choice read from 'after' (correct / wrong / neither), conditional strata (in-surface):

- off: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_mid 0.48/0.52/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0; cond_far 0.49/0.51/0.0
- plain: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_mid 0.48/0.52/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0; cond_far 0.49/0.51/0.0
- trace: cond_xm1 1.0/0.0/0.0; cond_x0 1.0/0.0/0.0; cond_xp1 1.0/0.0/0.0; cond_mid 1.0/0.0/0.0; cond_oor_val 1.0/0.0/0.0; cond_oor_thr 1.0/0.0/0.0; cond_far 1.0/0.0/0.0
- filler: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_mid 0.48/0.52/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0; cond_far 0.49/0.51/0.0

Chains (k=4; normalised regret / exact; mean [95% CI]):

| stratum | side | regret k4 | exact k4 | product-law predicted k4 | observed |
|---|---|---|---|---|---|
| CH_IN | off | 0.296 [+0.254, +0.339] | 0.234 | 0.2287 | 0.2344 |
| CH_IN | plain | 0.044 [+0.027, +0.063] | 0.660 | 0.6501 | 0.6602 |
| CH_IN | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| CH_IN | filler | 0.044 [+0.027, +0.063] | 0.660 | 0.6501 | 0.6602 |
| CH_X | off | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| CH_X | plain | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| CH_X | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| CH_X | filler | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| ALL | off | 0.379 [+0.328, +0.433] | 0.190 | 0.1758 | 0.1901 |
| ALL | plain | 0.213 [+0.143, +0.290] | 0.474 | 0.4404 | 0.474 |
| ALL | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| ALL | filler | 0.213 [+0.143, +0.290] | 0.474 | 0.4404 | 0.474 |


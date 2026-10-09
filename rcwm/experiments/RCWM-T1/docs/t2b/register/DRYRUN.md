<!-- synthetic dry run: NOT model output -->
# T2b-register screen results (s0)

**Overall:** SCREEN (seed 0): GO [SYNTHETIC DRY RUN]

Pre-stated rule (verbatim):

> PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank; 95% cluster-bootstrap CIs reported, not gating). Outcome = exact 'after' state (decision parser reads only 'after'; a malformed trace/filler answer is wrong). PRIMARY outcomes are the out-of-range strata (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr pooled (branch-balanced by construction); 'mix' = mix_oor. GO iff ALL of: (G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below trace: trace - filler > 0 on both primaries; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean of cond_xm1/cond_x0/cond_xp1) and on mix_in. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH primaries, STOP and report 'extra decode compute, not supervision' (sub-reason 'neither format helped' when filler - plain < 0.10 on both primaries as well). Otherwise NO-GO, naming every failed condition. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

## Seed s0: GO

| condition | holds |
|---|---|
| G1 trace-plain >= +0.25 on conditional (OOR) | yes |
| G2 trace-plain >= +0.30 on mix (OOR) | yes |
| G3 filler below trace on both primaries | yes |
| G4 controls: trace ON-OFF >= -0.05 on swap and add_nowrap | yes |
| G5 in-distribution non-inferiority: trace-plain >= -0.05 on conditional_in and mix_in | yes |

| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |
|---|---|---|---|---|---|---|
| primary_conditional | 0.395 | 0.395 | 1.000 | 0.395 | +0.605 [+0.545, +0.665] | +0.605 [+0.545, +0.665] |
| primary_mix | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| conditional_in | 0.492 | 0.500 | 1.000 | 0.500 | +0.500 [+0.421, +0.579] | +0.500 [+0.421, +0.579] |
| mix_in | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mix | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| wrap | 0.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| cond_xm1 | 0.517 | 0.517 | 1.000 | 0.517 | +0.483 [+0.350, +0.617] | +0.483 [+0.350, +0.617] |
| cond_x0 | 0.483 | 0.517 | 1.000 | 0.517 | +0.483 [+0.350, +0.617] | +0.483 [+0.350, +0.617] |
| cond_xp1 | 0.483 | 0.483 | 1.000 | 0.483 | +0.517 [+0.383, +0.650] | +0.517 [+0.383, +0.650] |
| swap | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| add_nowrap | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| mix_oor | 0.500 | 0.500 | 1.000 | 0.500 | +0.500 [+0.400, +0.600] | +0.500 [+0.400, +0.600] |
| wrap_oor | 0.000 | 0.000 | 1.000 | 0.000 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] |
| cond_oor_val | 0.370 | 0.370 | 1.000 | 0.370 | +0.630 [+0.550, +0.710] | +0.630 [+0.550, +0.710] |
| cond_oor_thr | 0.420 | 0.420 | 1.000 | 0.420 | +0.580 [+0.490, +0.670] | +0.580 [+0.490, +0.670] |

Trace fields (trace side):

| stratum | format | cmp_ok | ok_ok | calc_ok | after follows calc |
|---|---|---|---|---|---|
| mix | 1.0 | 1.0 | - | 1.0 | 1.0 |
| wrap | 1.0 | 1.0 | - | 1.0 | 1.0 |
| cond_xm1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_x0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_xp1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| swap | 1.0 | 1.0 | - | 1.0 | 1.0 |
| add_nowrap | 1.0 | 1.0 | - | 1.0 | 1.0 |
| mix_oor | 1.0 | 1.0 | - | 1.0 | 1.0 |
| wrap_oor | 1.0 | 1.0 | - | 1.0 | 1.0 |
| cond_oor_val | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cond_oor_thr | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Branch choice read from 'after' (correct / wrong / neither), conditional strata:

- off: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0
- plain: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0
- trace: cond_xm1 1.0/0.0/0.0; cond_x0 1.0/0.0/0.0; cond_xp1 1.0/0.0/0.0; cond_oor_val 1.0/0.0/0.0; cond_oor_thr 1.0/0.0/0.0
- filler: cond_xm1 0.5167/0.4833/0.0; cond_x0 0.5167/0.4833/0.0; cond_xp1 0.4833/0.5167/0.0; cond_oor_val 0.5/0.5/0.0; cond_oor_thr 0.44/0.56/0.0

Chains (k=4; normalised regret / exact; mean [95% CI]):

| stratum | side | regret k4 | exact k4 | product-law predicted k4 | observed |
|---|---|---|---|---|---|
| CH_IN | off | 0.296 [+0.254, +0.339] | 0.234 | 0.2287 | 0.2344 |
| CH_IN | plain | 0.049 [+0.030, +0.069] | 0.664 | 0.6562 | 0.6641 |
| CH_IN | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| CH_IN | filler | 0.049 [+0.030, +0.069] | 0.664 | 0.6562 | 0.6641 |
| CH_X | off | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| CH_X | plain | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| CH_X | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| CH_X | filler | 0.547 [+0.459, +0.635] | 0.102 | 0.0819 | 0.1016 |
| ALL | off | 0.379 [+0.328, +0.433] | 0.190 | 0.1758 | 0.1901 |
| ALL | plain | 0.215 [+0.146, +0.292] | 0.477 | 0.4447 | 0.4766 |
| ALL | trace | 0.000 [+0.000, +0.000] | 1.000 | 1.0 | 1.0 |
| ALL | filler | 0.215 [+0.146, +0.292] | 0.477 | 0.4447 | 0.4766 |


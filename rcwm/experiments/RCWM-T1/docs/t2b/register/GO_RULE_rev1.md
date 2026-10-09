# T2b-register screen go rule, REVISION 1 (SUPERSEDED; kept verbatim, never applied)

Stated 2026-10-09 at about 03:04 with the first build of the kit (design: `DESIGN_rev1.md`, section 7), before any
T2b-register output existed. Superseded by revision 2 (about 07:54, after the adversarial review) and then by revision 3
(`GO_RULE.md`, operator decisions D-36), each before any output existed. No run ever used it.

sha256 of the rule text below (one line, UTF-8, no trailing newline): `c7be67e0039c0687550e952ff7be98f4d108f31817835f8926e99859d377f640`
(pinned in `scripts/t2br_common.GO_RULE_REV1_SHA256`).

> PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank; 95% cluster-bootstrap CIs reported, not gating). Outcome = exact 'after' state (decision parser reads only 'after'; a malformed trace/filler answer is wrong). PRIMARY outcomes are the out-of-range strata (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr pooled (branch-balanced by construction); 'mix' = mix_oor. GO iff ALL of: (G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below trace: trace - filler > 0 on both primaries; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean of cond_xm1/cond_x0/cond_xp1) and on mix_in. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH primaries, STOP and report 'extra decode compute, not supervision' (sub-reason 'neither format helped' when filler - plain < 0.10 on both primaries as well). Otherwise NO-GO, naming every failed condition. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

# T2b-register screen go rule, REVISION 3 (current; pre-stated)

Applied by `scripts/t2br_analysis.verdict` (exact rationals). The same text is `scripts/t2br_common.GO_RULE_TEXT`, is
written into `data/t2br_dev.SOURCE.json`, and is a frozen input (`scripts/t2br_run.py --freeze`). Stated 2026-10-09
before any `runs/t2b_register` output existed (none exists) and before any model output of this screen was produced.
Earlier revisions are kept verbatim: `GO_RULE_rev1.md` (first build, about 03:04) and `GO_RULE_rev2.md` (after review,
about 07:54).

sha256 of the rule text (one line, UTF-8, no trailing newline): `aff13a4e9937af49daec6df178d555ff1ecb116e3da490b94f6c47bbceabe9b7`

## What changed from revision 2, and why

1. **Training is the surface-diversity build 'sd' only (C1).** PREREG-3 returned STATUS: TRIGGER (REPORT section 3c,
   DECISIONS R-PREREG-3, D-35): the T2 ledger LoRA took the literal token 'void' as its no-op cue, and it failed on
   reordered JSON keys (0.37-0.47 against 0.87-0.91 in the trained format), word account names and magnitudes. BWM's
   Blocksworld delta world model is likewise locked to key order (reversed order 0.458 against 1.000; R16). Decision
   point 1 has therefore fired. The canonical-only build is not run.
2. **Input key-order augmentation from a TRAIN subset of permutations (C2).** Every JSON object of the training prompt
   (top level, state, event) is serialized in a key order drawn per row from the permutations that are NOT held out.
   Held out per object type: the fully reversed canonical order plus 2 fixed permutations drawn with seed 20261113. The
   2-key top-level object has only its reversed order to hold out, so it trains in the canonical order alone. Rule-text
   paraphrase 4 and field-name variant 4 are held out as well. Targets keep the literal cmp -> calc -> after order.
   Reason: R16's consequence, agreed with BWM (their D-52 / D-54): test on orders never seen in training.
3. **G1 and G2 must hold in-surface AND on the combined held-out surface (C3, kit decision (a)).** A new stage (E1H)
   evaluates the 400 primary rows in paraphrase 4 + field names 4 + a held-out key order per object type, rotating over
   the held-out orders by row. Same rows and golds as in-surface, paired by row. Reason: D-35. Every claim gets a
   surface-form audit alongside the out-of-range audit, and a gain that holds only in the trained surface is
   trigger-level, not rule-level.
4. **G1 also requires trace - plain >= +0.25 on cond_oor_val alone (kit decision (b)).** Reason: a coarse tens-digit
   comparison scores 0.843 on the pooled primary conditional, about +0.34 over chance, because every far pair and
   every out-of-range-threshold pair is separated by the tens digits. It could pass the pooled clause without
   computing the boundary. The boundary pairs (other = threshold vs threshold - 1) defeat it. The dry run's
   coarse-comparison scenario passes the pooled clause (+0.40) and fails this one (+0.12).
5. **Amendment 4b filler (C4, kit decision (c)).** The fill is a run of ' .' of a CONSTANT length per action kind (swap
   5, add 14, mix 14, conditional 26 units): the shortest run whose filler target is at least as long as every trace
   target of that kind over train + dev. So every filler target of a kind is identical up to 'after', and its length
   carries no information about the answer. Reason: revision 2's per-row token-matched fill leaked the answer. Its
   length alone predicted wrap at 0.957 (training mix) and 0.970 (dev mix_oor), and the branch at 0.57 / 0.85 / 0.72
   on the out-of-range conditional strata. That gave the filler arm a hidden scratch channel. The fill_exact
   diagnostics are dropped in favour of the constant-length check. The residual (filler tokens minus the row's trace
   tokens, at least 0) is reported as a descriptive covariate. Decision point 3's label is reworded, because the
   filler is now answer-free.
6. **Wording only.**
   - Validity is stated on the in-surface primaries.
   - A NO-GO names every failed condition and its failed parts.
   - The held-out-surface results of the OFF and filler sides, the surface audit and the residual covariate are
     reported, not gating.
   - Per-factor held-out renderings are not run (GPU budget: the plan already needs a second turn at the planning
     rate).

**Unchanged:**
- G3 at +0.10 on each in-surface primary;
- G4;
- G5;
- exact rational decisions;
- the SE flags;
- the seed-0 screen.

**Also kept, as accepted by the operator (kit decision (d)):**
- the 5/6 rule-variant identification bar (0.55 is infeasible);
- the distance bands (twins |d| 2..15, cond_far 16..96);
- 190 training steps (T2 recipe parity; 4.05 epochs over 750 rows);
- point-estimate margins with SE flags.

## The rule (verbatim)

> PRE-STATED SCREEN GO RULE, REVISION 3 (T2b-register; supersedes revisions 1 and 2, stated before any output exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by row, on the primary strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. FILLER (amendment 4b): the fill is a run of ' .' of a CONSTANT length per action kind (swap 5, add 14, mix 14, conditional 26 units: the shortest run whose filler target is at least as long as every trace target of that kind in train + dev), so every filler target of a kind is identical up to 'after' and its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional AND >= +0.25 on cond_oor_val alone, each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and primary stratum); the filler residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings are not run (GPU budget). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

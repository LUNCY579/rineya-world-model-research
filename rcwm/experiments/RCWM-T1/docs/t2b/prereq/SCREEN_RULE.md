# Pre-stated screen rule (revision 4): T2b prerequisite representation screen (seed 0, surface-diversity build)

## What changed and why (revision 4, 2026-10-09)

Revision 4 was written before any `runs/t2bp/` output existed; no model was loaded and no model output was read to
choose it. Revisions 1-3 are kept (`SCREEN_RULE_rev1.*`, `SCREEN_RULE_rev2.*`, `SCREEN_RULE_rev3.*`). It makes exactly
two changes to revision 3, both selected by the operator from the independent verifier's review:

- **A corrected sentence (C4, amendment 4b).** Revision 3 said that revision 2's row-matched length "alone predicts
  applicability perfectly" on the training rows. That figure was computed from the token count of the full trace
  text, which contains the answer. Revision 2 actually used a fill of (trace-head tokens minus 3) units, decoded
  before `after`. Computed from that fill length, the length predicts applicability with accuracy 0.749 on the
  training rows (majority share 0.500), 0.837 on the 8-bit dev rows (0.600) and 0.813 on the out-of-range dev rows
  (0.667). That is well above the majority share, so amendment 4b stands, but the prediction is not perfect. The
  corrected sentence is in the C4 item below and in item (3) of the rule text.
- **Held-out-surface chains, report-only (stage E4).** The 240 dev k=4 chains are also self-rolled out in the combined
  held-out surface, adapter ON only. It is E3's surface, with the held-out event key order rotating by the chain's index
  in the dev bank (E3 rotates by the row's index within its stratum). The analysis reports K4_HS and the gap K4 - K4_HS
  per arm. E4 is descriptive: it enters neither R nor eligibility, and HS stays defined on E3's seven 8-bit one-step
  strata. The screen still fits the 4.5 h turn cap (`GEN_PLAN.json`).
- **Unchanged.** Everything else: R = (K4 + IAW + OOR + HS) / 4 with the in-surface K4, eligibility, R0-R5, the
  tie-breaks, the close-call wording, the rows, the dev bank, the arms' answer formats and parsers, and the recipe.

### Revision 3's changes (carried over; only the C4 sentence is corrected)

Revision 3 was written before any `runs/t2bp/` output existed; no model was loaded and no model output was read to
choose it.

- **Why.** PREREG-3 returned STATUS: TRIGGER (DECISIONS R-PREREG-3, D-35, R16). The T2 ledger LoRA took the literal
  token 'void' as its no-op cue instead of the stated rule, and it failed on reordered JSON keys (0.37-0.47 vs 0.87-0.91
  in format). BWM's delta world model is likewise key-order-locked (reversed key order 0.458 vs 1.000). A small LoRA can
  learn a surface cue in place of a rule, so a screen that only tests the training surface can pick a cue-learner.
- **C1 (decision point 1 fired).** The screen trains the surface-diversity build: the same 1,500 canonical rows (never
  re-drawn), each with a seeded rule-text paraphrase (with its own state sentence, which carries the width word), a
  field-name variant and input key orders. The canonical-only build is not run.
- **C2 (input key-order augmentation).** Every JSON object of a training prompt is serialized in a key order drawn from
  a training subset of permutations. Held out per object type: the reversed canonical order plus 2 fixed random
  permutations (the 2-key top-level object has only the reversed order to hold out). Targets keep the literal
  cmp -> calc -> after order.
- **C3 / 5a (a) (held-out surface in the ranking).** A new stage E3 evaluates the 1,050 one-step rows in a combined
  held-out surface (paraphrase 4 + field names 4 + a held-out key order, rotating by row), adapter ON, paired by row with
  the in-surface stage. HS = the mean accuracy of the seven 8-bit strata there. R = (K4 + IAW + OOR + HS) / 4.
  Eligibility (IAW >= 0.20, FC <= 0.25, FA <= 0.25, ID non-inferiority), R0 (in-surface format rate), the tie-breaks and
  the close-call wording are unchanged. Per-factor held-out renderings are not run (they do not fit the planning budget).
- **C4 / 5a (c) (amendment 4b).** The F-filler's fill length is a constant per action kind (26 units for the single
  prerequisite kind), at least the longest trace over the training and dev rows. Revision 2 matched the length to the
  trace row by row, and on these training rows that fill length alone predicts applicability with accuracy 0.749,
  against a majority share of 0.500.
- **5a (b) (revision 2's open questions, answered by the operator; nothing changes).** Accepted as they were: the
  add=0 -> copy cue at 0.62 (descriptive; FC and FA guard it); strict A4 IAW for decision point 2 (the R11 union is
  reported); S-trace's cmp lists bit positions; B-bits writes the most significant bit first; out-of-range prompts change
  only the width word, and since each paraphrase's state sentence words the width its own way, the held-out surface
  words it in held-out form too; ID margin 0.10, FA <= 0.25, WF >= 0.90, close call < 0.05.
- **Unchanged.** The rows, the dev bank, the arms' answer formats and parsers, the recipe, R0-R3, R5's thresholds.

## The rule (verbatim)

Copied verbatim from the docstring of `scripts/t2bp_analysis.py`, which implements it (`apply_rule`, `apply_filler`). Hashes of every frozen input: `SCREEN_RULE.json`.

```
PRE-STATED SCREEN RULE, revision 4 (T2b prerequisite representation screen, seed 0, surface-diversity build).
What changed and why. Revisions 1 (2026-10-09 03:28), 2 (07:34, after review) and 3 (12:24) were written before any
runs/t2bp output existed and are kept (docs/t2b/prereq/SCREEN_RULE_rev1.*, SCREEN_RULE_rev2.*, SCREEN_RULE_rev3.*).
PREREG-3 had returned STATUS: TRIGGER (DECISIONS R-PREREG-3, D-35, R16): the T2 ledger LoRA took the literal token
'void' as its no-op cue instead of the stated rule and failed on reordered JSON keys, and BWM's delta world model is
likewise locked to key order. A LoRA can learn a surface cue in place of a rule. Revision 3 applies the operator's
decisions (DECISIONS D-36: C1-C6 and 5a (a)-(c)), again before any runs/t2bp output existed and without any model
output being read:
  (1) decision point 1 has fired: the screen trains the surface-diversity build (the same canonical rows, each with a
      seeded rule-text paraphrase, a field-name variant and input key orders drawn from training subsets only); the
      canonical-only build is not run;
  (2) the ranking adds HS, accuracy on a combined held-out surface that no training prompt uses (R = the mean of K4,
      IAW, OOR and HS), so an arm that learned the rule only on its training surfaces cannot win on in-surface scores;
  (3) the F-filler control uses a constant fill length per action kind (amendment 4b), because a fill length matched
      to the trace row by row reveals the answer (on these training rows revision 2's fill length alone, trace-head
      tokens - 3, predicts applicability with accuracy 0.749 against a majority share of 0.500), which would give the
      control a hidden scratch channel.
Revision 4 (fixes the operator selected from the independent verifier's review, 2026-10-09, again before any
runs/t2bp output existed and without any model output being read) changes two things only: (a) it corrects the figure
in (3), which revision 3 gave as "the trace length alone predicts applicability perfectly", computed from the full
trace text (which contains the answer) instead of revision 2's fill length (on the dev rows that fill length predicts
applicability with accuracy 0.837 on the 8-bit rows and 0.813 out of range, majority shares 0.600 and 0.667); (b) it
adds stage E4, the 240 dev chains in the combined held-out surface (adapter ON), reported as K4_HS and K4 - K4_HS per
arm; E4 is report-only: it enters neither R nor eligibility, and HS stays defined on E3's one-step strata.
Eligibility, R0 (on the in-surface format rate), R1-R3, the tie-breaks, the close-call wording and R5's thresholds are
unchanged. The rule text and the sha256 of every frozen input are recorded in docs/t2b/prereq/SCREEN_RULE.json; the
runner and this analysis refuse when a frozen input differs. Verbatim in docs/t2b/prereq/SCREEN_RULE.md.

  Inputs. Adapter ON, seed 0, each screen arm A in {D-int, S-trace, B-bits} trained on its surface-diversity file
  data/t2bp_train_sd_<arm>.jsonl (run runs/t2bp/sd_s0), on the dev screen bank data/t2bp_dev.jsonl (1,050 one-step rows
  in 9 strata; 240 k=4 chains). The one-step rows and the chains are evaluated in the in-surface rendering (the
  canonical surface: stages E1, E2); the one-step rows also in the COMBINED held-out-surface rendering (stage E3:
  paraphrase 4 with its state sentence, field names 4, the reversed top-level key order and one of the 3 held-out event
  key orders, rotating by the row's index within its stratum; no training prompt uses any of these; the same rows and
  golds as E1, paired by row). The chains are also rolled out in that combined held-out surface (stage E4, adapter ON;
  the held-out event key order rotating by the chain's index in the dev bank; report-only, see the last paragraph).
  All three arms must be complete with every stage (DONE.json; every output hash-verified; the frozen inputs
  unchanged; the scorer self-test passes before any output is read); a run that is not the surface-diversity build, or
  lacks stage E3, gets no verdict. Answers are scored by the arm's after-only parser (t2bp_common.parse: D-int lenient
  exactly as rcwmt.env.parse_state; the others strict); an unparseable or malformed answer is wrong.
  Metrics (point estimates, adapter ON; "8-bit rows" = the 750 one-step rows outside D_ood10 and D_ood12; every metric
  except HS is in-surface):
    IAW(A)  accuracy on D_int_add_wrong (150 rows: applicable, the add bit is already set and the delete clears a set
            bit, so the bitwise answer differs from before+add, from (before&~delete)+add and from before);
    OOR(A)  mean of the accuracies on D_ood10 and D_ood12 (10- and 12-bit masks; out of range; primary per D-52 (5));
    K4(A)   k=4 exact final-state accuracy over the 240 dev chains (self-rollout in A's own frame; a broken chain is
            wrong);
    HS(A)   the mean of the accuracies of the seven 8-bit strata (the ID strata below) in the combined held-out surface
            (stage E3), equal weight;
    FC(A)   false copy (T2's definition): the share of answers equal to the input state among the 350 8-bit rows whose
            gold differs from the input (D_app_add, D_app_delete, D_int_add_wrong);
    FA(A)   false apply (the mirror of FC): the LARGEST, over the inapplicable strata D_inapp_pre_on (100),
            D_inapp_pre_off (100) and D_inapp_addset (100; the add bit is already set and the delete clears a set bit:
            the inapplicable twin of D_int_add_wrong), of the share of answers that are not the input state (on every
            such row the gold is the input and applying the action would change it; a malformed answer counts);
    WF(A)   the share of answers in the arm's format on the 750 8-bit rows (t2bp_common.format_ok: D-int, the lenient
            parser finds an integer; S-trace, the literal keys and an integer after; B-bits, the literal keys, a
            boolean flag and a binary string; the value is not checked here, so an out-of-range state such as
            200 + 64 = 264 is a wrong answer, not a format failure);
    ID(A)   the mean of the accuracies of the seven 8-bit strata (D_app_add, D_app_unchanged, D_app_delete,
            D_int_add_wrong, D_inapp_pre_on, D_inapp_pre_off, D_inapp_addset), equal weight.
  R0 (instrument validity). If WF(A) < 0.90 for any screen arm (in-surface; a malformed held-out-surface answer is a
     wrong answer that lowers HS, not an instrument failure), or a turn that ran an evaluation stage has no passed
     anchor record: verdict NO_VERDICT_INSTRUMENT. This is an instrument or format failure, not a result: no programme
     decision follows; it is investigated and logged in DECISIONS.md.
  R1 (decision point 2). If IAW(A) < 0.20 for every screen arm: verdict STOP_P_TRAINING. Stop training the
     prerequisite family and move P to a code-side bit interface (code converts the integer state and the masks and
     applies the bitwise rule), reported as an interface, not as model skill. Every statement of it quotes the dose:
     150 of the 1,500 training rows were int-add-wrong (A4) rows. No arm is named.
  R2. Otherwise an arm is ELIGIBLE iff IAW(A) >= 0.20 (decision point 2 holds for the named arm itself), FC(A) <= 0.25,
     FA(A) <= 0.25 and ID(A) >= ID(D-int) - 0.10 (non-inferiority against the data control; D-int meets this last
     condition by definition).
  R3. If no arm is eligible: verdict NO_ELIGIBLE_ARM (in every arm, int-add-wrong below 0.20, the copy shortcut, a
     false-apply shortcut or an in-distribution collapse); no arm is named for seeds 1-2.
  R4. Otherwise rank the eligible arms by R(A) = (K4(A) + IAW(A) + OOR(A) + HS(A)) / 4. The winner is the arm with the
     largest R; if the top two differ by less than 0.02 in R, the one with the larger IAW wins; if those differ by less
     than 0.02, the one with the larger OOR; if still tied, the first in the order D-int, B-bits, S-trace (fewest
     decode tokens). Verdict WINNER=<arm>: that arm (its rendering, the same rows and surfaces, the T2 recipe) runs
     seeds 1 and 2. If the winner is D-int, the result is worded "the representation does not beat the
     rebalanced-data control". If another eligible arm's R is within 0.05 of the winner's, the result is worded as a
     close call: a screen choice, not a finding.
  R5 (decode-compute control). F-filler (the amendment-4b constant fill) is a control and is never eligible. It runs
     only if S-trace wins, as its own run (tag sd_s0_ffill: seed 0, the same rows and surfaces, the T2 recipe, every
     stage including E3; analysed by --filler-run), unless it was part of the sd_s0 run. If R(F-filler) >=
     R(S-trace) - 0.05 (R as in R4), the S-trace result carries the flag TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE
     (the winner is unchanged; the flag is quoted with every statement of the result). If WF(F-filler) < 0.90 or a
     filler turn lacks a passed anchor, the control is void and the result carries DECODE_COMPUTE_CONTROL_VOID. Until
     the control is analysed, an S-trace win is stated with "decode-compute control pending".
  Everything else (OFF baselines, the other strata and subtypes, out-of-range false copy, the held-out-surface
  out-of-range strata, FC/FA/WF and ID in the held-out surface, the paired held-out minus in-surface contrasts,
  accuracy by held-out key order, the held-out-surface chains of stage E4 (K4_HS, the k=4 exact final-state accuracy
  of the 240 chains in the combined held-out surface, and the gap K4 - K4_HS, per arm), the filler's residual length
  (filler minus trace tokens) as a covariate, union cells, shortcut agreement, cmp_ok / calc_ok / flag_ok, decision
  regret, paired contrasts) is descriptive and cannot change the verdict. This is a screen on a dev bank: no
  confirmatory claim follows from it; a confirmatory claim needs a sealed bank and a preregistration (roadmap item 2,
  "Confirm"), with held-out-surface primaries (D-35).
```

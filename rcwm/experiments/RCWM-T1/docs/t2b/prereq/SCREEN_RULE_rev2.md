# Pre-stated screen rule (revision 2): T2b prerequisite representation screen (seed 0)

Copied verbatim from the docstring of `scripts/t2bp_analysis.py`, which implements it (`apply_rule`, `apply_filler`). Revision 2 was written on 2026-10-09 after review, before any `runs/t2bp/` output existed. Revision 1 is kept as `SCREEN_RULE_rev1.md` / `SCREEN_RULE_rev1.json`. Hashes of every frozen input: `SCREEN_RULE.json`.

```
PRE-STATED SCREEN RULE, revision 2 (T2b prerequisite representation screen, seed 0). Revision 1 was written 2026-10-09
before any runs/t2bp output existed. Revision 2 replaces it after review (a mirror false-apply gate, IAW >= 0.20 for the
named arm, an instrument-validity gate, false copy on the 8-bit rows, the filler control as its own run), also before
any runs/t2bp output existed and without any model output being read. Its text and the sha256 of every frozen input
are recorded in docs/t2b/prereq/SCREEN_RULE.json; the runner and this analysis refuse when a frozen input differs.
Verbatim in docs/t2b/prereq/SCREEN_RULE.md.

  Inputs. Adapter ON, seed 0, each screen arm A in {D-int, S-trace, B-bits}, on the dev screen bank data/t2bp_dev.jsonl
  (1,050 one-step rows in 9 strata; 240 k=4 chains). All three arms must be complete (DONE.json; every output
  hash-verified; the frozen inputs unchanged; the scorer self-test passes before any output is read); otherwise no
  verdict is issued. Answers are scored by the arm's after-only parser (t2bp_common.parse: D-int lenient exactly as
  rcwmt.env.parse_state; the others strict); an unparseable or malformed answer is wrong.
  Metrics (point estimates, adapter ON; "8-bit rows" = the 750 one-step rows outside D_ood10 and D_ood12):
    IAW(A)  accuracy on D_int_add_wrong (150 rows: applicable, the add bit is already set and the delete clears a set
            bit, so the bitwise answer differs from before+add, from (before&~delete)+add and from before);
    OOR(A)  mean of the accuracies on D_ood10 and D_ood12 (10- and 12-bit masks; out of range; primary per D-52 (5));
    K4(A)   k=4 exact final-state accuracy over the 240 dev chains (self-rollout in A's own frame; a broken chain is
            wrong);
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
  R0 (instrument validity). If WF(A) < 0.90 for any screen arm, or a turn that ran an evaluation stage has no passed
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
  R4. Otherwise rank the eligible arms by R(A) = (K4(A) + IAW(A) + OOR(A)) / 3. The winner is the arm with the largest
     R; if the top two differ by less than 0.02 in R, the one with the larger IAW wins; if those differ by less than
     0.02, the one with the larger OOR; if still tied, the first in the order D-int, B-bits, S-trace (fewest decode
     tokens). Verdict WINNER=<arm>: that arm (its rendering, the same rows, the T2 recipe) runs seeds 1 and 2. If the
     winner is D-int, the result is worded "the representation does not beat the rebalanced-data control". If another
     eligible arm's R is within 0.05 of the winner's (about two standard errors at these n), the result is worded as a
     close call: a screen choice, not a finding.
  R5 (decode-compute control). F-filler is a control and is never eligible. It runs only if S-trace wins, as its own
     run (tag s0_ffill: seed 0, the same rows, the T2 recipe; analysed by --filler-run), unless it was part of the s0
     run. If R(F-filler) >= R(S-trace) - 0.05, the S-trace result carries the flag
     TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE (the winner is unchanged; the flag is quoted with every statement of
     the result). If WF(F-filler) < 0.90 or a filler turn lacks a passed anchor, the control is void and the result
     carries DECODE_COMPUTE_CONTROL_VOID. Until the control is analysed, an S-trace win is stated with "decode-compute
     control pending".
  Everything else (OFF baselines, the other strata and subtypes, out-of-range false copy, union cells, shortcut
  agreement, cmp_ok / calc_ok / flag_ok, decision regret, paired contrasts, the held-out-surface stage E3 of a
  surface-diversity run) is descriptive and cannot change the verdict. This is a screen on a dev bank: no confirmatory
  claim follows from it; a confirmatory claim needs a sealed bank and a preregistration (roadmap item 2, "Confirm").
```

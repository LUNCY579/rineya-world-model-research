# Pre-stated screen rule: T2b prerequisite representation screen (seed 0)

Copied verbatim from the docstring of `scripts/t2bp_analysis.py`, which implements it (`apply_rule`). Written 2026-10-09, before any `runs/t2bp/` output existed. Hashes: `SCREEN_RULE.json`.

```
PRE-STATED SCREEN RULE (T2b prerequisite representation screen, seed 0). Written 2026-10-09, before any runs/t2bp
output existed; its sha256 (and this file's) are recorded in docs/t2b/prereq/SCREEN_RULE.json. Verbatim in
docs/t2b/prereq/SCREEN_RULE.md.

  Inputs. Adapter ON, seed 0, each screen arm A in {D-int, S-trace, B-bits}, on the dev screen bank data/t2bp_dev.jsonl.
  All three arms must be complete (DONE.json; every output hash-verified; the scorer self-test passes before any output
  is read); otherwise no verdict is issued. Answers are scored by the arm's after-only parser (t2bp_common.parse: D-int
  lenient exactly as rcwmt.env.parse_state; the others strict); an unparseable or malformed answer is wrong.
  Metrics (point estimates, adapter ON):
    IAW(A)  accuracy on stratum D_int_add_wrong (applicable; the add bit is already set and the delete clears a set bit,
            so the bitwise answer differs from before+add, from (before&~delete)+add and from before);
    OOR(A)  mean of the accuracies on D_ood10 and D_ood12 (10- and 12-bit masks; out of range; primary per D-52 (5));
    K4(A)   k=4 exact final-state accuracy over the 240 dev chains (self-rollout in A's own frame; a broken chain is
            wrong);
    FC(A)   false-copy rate: the share of answers equal to the input state among all dev one-step rows whose gold
            differs from the input state;
    ID(A)   the mean of the accuracies of the six in-distribution strata (D_app_add, D_app_unchanged, D_app_delete,
            D_int_add_wrong, D_inapp_pre_on, D_inapp_pre_off), equal weight.
  R1 (decision point 2). If IAW(A) < 0.20 for every A in {D-int, S-trace, B-bits}: verdict STOP_P_TRAINING. Stop
     training the prerequisite family and move P to a code-side bit interface (code converts the integer state and the
     masks and applies the bitwise rule), reported as an interface, not as model skill. No arm is named.
  R2. Otherwise an arm is ELIGIBLE iff FC(A) <= 0.25 and ID(A) >= ID(D-int) - 0.10 (non-inferiority control against
     the data control; D-int meets the second condition by definition).
  R3. If no arm is eligible: verdict NO_ELIGIBLE_ARM (the copy shortcut, or an in-distribution collapse, persists in
     every arm); no arm is named for seeds 1-2.
  R4. Otherwise rank the eligible arms by R(A) = (K4(A) + IAW(A) + OOR(A)) / 3. The winner is the arm with the
     largest R; if the top two differ by less than 0.02 in R, the one with the larger IAW wins; if those differ by less
     than 0.02, the one with the larger OOR; if still tied, the first in the order D-int, B-bits, S-trace (fewest
     decode tokens). Verdict WINNER=<arm>: that arm (its rendering, the same rows, the T2 recipe) runs seeds 1 and 2.
     If the winner is D-int, the result is worded "the representation does not beat the rebalanced-data control".
  R5. F-filler (only if it was run) is a control and is never eligible. If S-trace is the winner and
     R(F-filler) >= R(S-trace) - 0.05, the verdict carries the flag TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE (the
     winner is unchanged; the flag is quoted with every statement of the result).
  Everything else (OFF baselines, the other strata, union cells, cmp_ok / calc_ok / flag_ok, decision regret, paired
  contrasts) is descriptive and cannot change the verdict. This is a screen on a dev bank: no confirmatory claim follows
  from it; a confirmatory claim needs a sealed bank and a preregistration (roadmap item 2, "Confirm").
```

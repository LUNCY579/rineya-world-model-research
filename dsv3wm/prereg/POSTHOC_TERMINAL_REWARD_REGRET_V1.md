# Posthoc source-episode terminal reward and bound-normalized regret

This is an explicitly posthoc diagnostic of already published native development
records, not preregistered confirmation or a replacement for original P2.
The original planner selected for nonterminating goal-state achievement. Its
recorded success, null failure cost/regret and all choices remain unchanged.
The new diagnostic uses the different, preexisting `rw2.execute` episode reward.
The analyst already knows all60 recorded episodes; this document does not imply
that the following normalization was chosen before their outcomes were observed.

## Fixed reward, cap and normalization

The frozen `bwm/code/bwmn/rw2.py` declares
`R = 1[terminated at goal] - 0.01 * executed steps - 0.1 * illegal executed steps`.
The six-action cap supplies a universal return lower bound `L = -0.66`.
This bound is conservative and need not be attainable for every initial state.
It is neither an estimated worst policy nor a candidate-mean normalization.

For each fixed task, use its already independently verified optimal nonfinish
distance `d`, and `R* = 1 - 0.01 * (d + 1)`. The additional step is legal final
`finish`. All12 fixed DEV tasks have d3/4/5, fitting the six-action terminal
budget. The normalized regret is `(R* - R) / (R* - L)`; a zero value means the
source-defined optimum and one means the universal lower bound. Preserve raw
actual/optimal returns beside this normalization, with the contract difference
prominent. Do not tune penalties, bounds or task selection from an arm contrast.

## Evidence and strict execution reconstruction

Consume the existing choices and recorded independent verification labels from
12 episodes each for B0/A0/A1/A2/A5. Bind actual task, output, trace and source
identities, including the unchanged native configuration and scoring provenance.
The analysis must call neither a model nor an exact transition/solver. Never
append a finish, repair a choice, derive a replacement action from verification,
or substitute a zero for original failed move cost/regret.

Recorded execution arrays must match choice length, strict state/flag types,
parent identity and task/cohort order. Execution is a prefix through the first
finish. Count finish itself as an executed action; later actions are ignored.
Require a legal nonmutating finish, no termination before a nonfinish action,
unchanged state on illegal executed steps and an absorbing ignored suffix.
Validate aggregate verifier fields against the recorded arrays before computing
reward. An ignored suffix may be source-episode equivalent, but H3's complete
plan contract rejects actions after final finish; never label this diagnostic
as H3 complete-plan success.

Null/malformed choices or missing/inconsistent execution labels are unevaluable.
Use strict refusal or explicit null reward/regret plus coverage; never reinterpret
an absent plan as a recorded empty execution. An explicitly recorded empty list
with complete empty arrays may have source-defined return zero. Current60 records
have complete nonempty choices, including A5's wrong-goal finish. Using its actual
three executed actions is distinct from imputing its original null move cost.

## Descriptive reporting

Report every actual episode's executed length, illegal count, termination,
terminal goal success, raw return, optimal return and normalized regret; preserve
the original P2 success and original nullable cost/regret fields separately.
Summarize all12 parents and unchanged12-task ID/OOD partitions with evaluated
coverage. Paired arm differences must use the same parent units; use only
explicitly descriptive single-seed uncertainty. Do not treat60 arm-times-case
observations as60 independent environments or claim seed/general-domain efficacy.

All original P2 outcomes, confirmation gates, retention limits and historical
sources remain intact. This calculation supplies a narrow all-case secondary
diagnostic under the terminal source reward. It cannot promote a mechanism,
qualify H3, establish confirmation or excuse a state/legality regression.

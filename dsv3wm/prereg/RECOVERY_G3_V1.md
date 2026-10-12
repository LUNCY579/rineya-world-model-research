# G3 recovery pilot v1: useful registered-F warm start

Drafted before new recovery native outcomes, 2026-10-11 America/Chicago.
This is a new development protocol. Its actual source/config/data/model freeze
must be recorded before its qualification or training runs. It does not amend
the outcomes or byte-frozen inputs of the earlier pilot.

## Reason for this recovery

Preserve every earlier cold-start 4/12-step training result and every original,
cached and chat-format generation diagnostic, including malformed responses,
failures and resource costs. The previous format failure motivates restoring a
useful inherited interface; it does not establish an MTP or state-horizon effect.
Do not combine earlier selected examples or outcomes with this new cohort to
manufacture a positive comparison. The existing report remains the record of
those experiments.

The new shared initialization is the registered original-format one-step F LoRA
on the same frozen Titan-Sparse backbone. Training adds updates to that unmerged
adapter. **B0 is this registered F with no new adaptation.** It is distinct from
Titan-Sparse with its adapter off, and from A0, which receives 32 new updates.
Warm starting is a changed experimental question and must appear in every report.

## Fixed artifacts and numerical contract

- Parent checkpoint SHA-256:
  `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`.
- Shared initial registered-F adapter SHA-256:
  `c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466`.
- FP32 end to end; TF32 and autocast disabled; no dense adapter merge,
  quantization, attention conversion or parent update.
- Preserve original rank 16, alpha and all original projection targets
  (`q/k/v/o/gate/up/down`) and dropout 0.05. Load each arm from the same adapter
  bytes; reset optimizer state rather than inheriting an unknown optimizer.
- One training seed, recorded explicitly as seed 0; batch size 1, CPU at most
  four threads. Seed auxiliary initialization separately or reset the training
  RNG afterward, so merely constructing A1's modules does not consume A0's
  initial training random stream. Preserve deterministic row ordering and all
  recorded dropout/randomness settings.
- Reuse the validated chunked loss, source-hashed native loader and exclusive
  CPU/GPU coordination mechanism. Do not start while another programme holds
  the shared lane; honor its priority waiters at work boundaries. No paid cloud
  compute is part of this protocol.

Before execution, pin source and sidecar hashes, adapter configuration, actual
trainable-parameter counts, optimizer configuration, dataset manifests, task
hashes and the analysis instrument. A hash or interface mismatch stops the run;
it is not repaired by silently selecting another registered artifact.

## Data and exposure interpretation

Generate and freeze 128 fresh composed training trajectories and 24 development
trajectories, with one independent parent per row, and no cross-split parent
overlap. Training limits are 24/32/40. Development has exactly 12 ID trajectories
at limits 32/48/64 and 12 OOD trajectories at limits 80/96; include the declared
high-magnitude subset with `abs(x) >= 65` in OOD. Freeze the generator, fixed
seeds, balances/strata, action lengths and semantic deduplication signatures.
All generated and rejected draws remain traceable; do not redraw after seeing
model accuracy.

The initial-F audit found that its original training covers all 1,455 one-step
keys at ID limits 32/48/64. Therefore **ID means newly composed trajectories,
not unseen one-step states or actions.** Initial-adapter exposure to composed
trajectories and any other predecessor corpus is not fully known. OOD labels
refer to limits/ranges outside that audited initial-adapter training surface;
they do not prove that the parent model has never encountered such arithmetic.
Training at new limits 24/40 also changes rule coverage. Report those changes
instead of attributing every gain to the prediction objective.

Construct a fixed 12-task planning cohort with 3–5 exact optimal moves,
balanced six ID/six OOD. Its parents are a subset of the 24 dynamics-development
parents and remain disjoint from training. The dependence between planning and
dynamics outcomes is explicit; these are paired development screens, not an
independent confirmation family. If the fixed generator cannot meet those
counts, preserve the failed construction and amend before model outcomes;
do not substitute easier tasks after evaluation. The oracle may establish task
difficulty during construction. Model inputs contain initial state, rules and
goal; they never contain the optimal plan or oracle successor states.

Only the new training/development cohorts and constructed permitted retention
fixtures may be accessed. Do not open, re-evaluate or tune against historical
sealed banks or the existing confirmation split. No G5 confirmation is authorized
by this pilot, and no frozen confirmation result enters hyperparameter selection.

## Prompt, output and causal semantics

Horizon 1 retains the legacy registered-F prompt and reply shape. For horizons
2/4, minimally edit the legacy chat instruction to accept the ordered action
prefix and requested resulting state; serialize the same initial state and rule
specification. Freeze all three actual prompt templates before qualification.
Keep goal information out of dynamics inputs.

The reply schema is exactly `{"x": integer, "legal": boolean, "term": boolean}`.
`legal` labels the **last action** at the requested horizon; it is not cumulative
plan legality. `term` is the absorbing termination flag. Do not add a `failed`
target to the legacy or direct multi-step reply. Duplicate JSON keys, extra
keys, floats in place of integer state values, numeric booleans, prose/fences
and incomplete arrays/objects are malformed. The duplicate-key tightening is
an explicit parser countermeasure shared by every arm, including B0.

For the one-step planner wrapper only, derive `failed = not predicted_legal`
from that step's own parsed output. A free-running failure diagnostic aggregates
illegal flags of **attempted predicted steps**, preserving prior predicted
failures; skipped steps after predicted termination must not fabricate a newly
executed illegal action. Direct horizon-2/4 outputs do not establish cumulative
failure correctness and that metric is not a primary direct-horizon outcome.

Gold intermediate/final states appear only in training loss targets and offline
evaluation labels. Every direct horizon conditions on `s0` and the action prefix
through its horizon. Every free-running transition consumes the previous
predicted state. Teacher forcing within the current response-token prefix is
legitimate training supervision; it must not become a gold-state reset during
evaluation. Sequential MTP may consume the preceding supervised response token
at an auxiliary depth, never its own target token or an oracle future state in
the conditioning prompt.

P1 for this pilot is four-step **free-running exact final-state accuracy**,
without oracle normalization, state repair or teacher-state resets. Report
direct one/two/four-step predictions, intermediate accuracy, parsing and
legality/termination separately. Malformed predictions count as failures and
stop that free-running chain; there is no fallback to the exact transition rule.

## B0 qualification before new training

Run B0 on the frozen 12 ID one-step development inputs, using the exact legacy
prompt, common strict parser, eval mode (dropout off) and fixed greedy decoding.
Require:

1. Valid reply syntax on at least 90%: at least **11/12** ID cases.
2. Exact one-step state accuracy on at least 80%: at least **10/12** ID cases.

These operational thresholds qualify a useful starting interface, not a
statistically established accuracy bound. If either fails, preserve outputs and
stop new training under this protocol. Diagnose via a new amendment rather than
changing this cohort, output schema or thresholds after seeing results.

If B0 passes ID, also evaluate and preserve its fixed OOD, four-step and planning
outcomes. OOD weakness alone does not block the recovery pilot; learning its
effect is part of the question. Qualification cases are development data, so
their later results remain exploratory and are never called independent
confirmation.

## Fixed native comparison arms

| Arm | Additional training | Objective | Interpretation |
|---|---|---|---|
| B0 | None | Registered one-step F unchanged | Shared warm-start reference |
| A0 | 32 optimizer steps | Legacy one-step next-state CE | New-adaptation baseline |
| A1 | 32 optimizer steps | A0 plus sequential MTP, two total future tokens, auxiliary rank 16 and weight 0.1 | Auxiliary token-training adaptation |
| A2 | 32 optimizer steps | State horizons 1/2/4, loss weights 1/0.5/0.5 | Explicit action-consequence supervision |
| A5 | 96 optimizer steps | Legacy one-step CE | Matches A2's nominal three decoder forwards per 32 updates |

All arms use learning rate `2e-5`, batch 1 and the same original dropout 0.05.
Preserve the warm-start weights in each new run's provenance, and save the
resulting adapter separately. The parent remains frozen. Use the same first
32 trajectory draws across A0/A1/A2/A5, with A5's additional draw schedule fixed
before outcomes. Freeze optimizer hyperparameters and scheduler behavior rather
than accepting unrecorded library defaults.

A1's auxiliary module adds approximately 51,200 parameters; verify its actual
count. The inherited rank-16 all-projection LoRA has approximately 10 million
trainable parameters; report its exact count. A1's additional head computation
and supervision are real differences. A5's 96 steps equal A2's nominal decoder
forward count, **not** its FLOPs, token count, recomputation, weighted label
allocation or number of optimizer updates. Report the mismatches and do not
claim perfect compute matching or objective superiority from this control alone.

Record optimizer steps, input/supervised/auxiliary tokens, parent draws,
supervised transitions, actual decoder forwards and vocabulary projections,
auxiliary/trainable parameters, wall time, active time if measurable, peak
memory and any duty/wait pauses. Preserve errors, interrupted runs and partial
logs. Use bounded retries for a resource interruption; resume the identical
configuration and cohort instead of replacing an unfavorable run.

## Evaluation and retention

Every arm, including B0, uses the same frozen plain legacy-compatible inference
templates, greedy max-32 output-token cap and qualified existing FP32 KV decoder.
Eval mode disables dropout. Use the same causal/parser/cache correctness tests;
record any EOS behavior and actual token counts. No chat-format substitution is
allowed within a comparative arm after grammar outcomes are known.

Planning uses the fixed learned-model beam interface, with at most **30 learned
transition queries per task** and a common **120,000 processed decoder-token
position cap per task**. Beam width is1, maximum depth6, static action order is
add1/sub1/double/halve/finish. Thirty queries allow six full five-action levels,
so goals requiring five moves are not structurally excluded by the budget.
Freeze
termination behavior and tie-breaking before evaluation. Count prompt prefill
and decoded positions, including an unused final decoder forward at the cap.
Exact scoring runs after the selected plan is fixed. It supplies no search,
selection, repair or fallback feedback. Report actual calls/tokens as well as
caps: equal caps do not establish equal realized compute.

Execute six fixed permitted old-format behavioral fixtures by greedy generation
for B0 and each adapted arm. Preserve prompt, raw reply, parsing, exact outcome
and paired changes; teacher-forced CE alone is not retention evidence. Freeze
fixture identities before the first new outcome. They must be constructed or
permitted development cases, never copied from a sealed evaluation bank.

If runtime permits, enable additional fresh arithmetic and ledger behavior
fixtures (12 each), with their prompts/scoring/hash lists fixed before adapted
outcomes. Otherwise record them as deferred with the resource reason. Report
every regression and mismatch; six or twelve fixtures cannot certify broad
retention or compatibility of stacked skills. No retention-pass claim or model
promotion follows merely from a loss-only diagnostic.

## Operational continuation rule

Evaluate the full fixed 24-parent dynamics cohort and all 12 planning tasks;
do not select the best domain, case subset, seed or generated candidate using
correctness labels. For A1 and A2 separately, a candidate may proceed to a
larger **development** study only if all conditions hold:

1. Its paired four-step free-running net gain versus A0 is at least **2/24**
   (candidate-only successes minus A0-only successes).
2. Its ID four-step free-running accuracy loses no more than **2/12** paired net successes
   versus A0; report individual gains and losses as well as the net difference.
3. Its fixed top-choice planning success count is no worse than A0 on the same
   12 tasks, at the common caps, without exact selection feedback.
4. Native runtime/memory is feasible in the authorized coordinated environment,
   parser and causal checks pass, and material retention regressions are
   explicitly investigated rather than concealed.

These small-sample thresholds are engineering continuation criteria. They are
not a significance threshold, confidence statement, G4 benefit claim or G5
confirmation. Report paired effect sizes and all outcomes. Any McNemar/Holm
analysis here is descriptive; neither positive nor nonsignificant p-values
replace these fixed operational rules. A5 and B0 remain required comparators
even if their outcomes weaken the interpretation of A1/A2.

If neither candidate passes, preserve the negative result and recommend against
larger native training under this recipe. An inconclusive result remains
inconclusive. Any further recipe, cohort or budget change needs a new dated
development protocol before its outcomes. Never keep extending training or
reshuffling draws until the threshold happens to be met.

## Conditional H3 teacher and student work

After useful B0 qualification, search on **TRAIN-only** states/goals using
learned predicted transitions. Independently execute candidate plans only after
learned ranking finishes; retain exact legal goal-reaching plans and labelled
hard negatives with verifier/source/data/parent provenance. Model-predicted
success alone is not a verified teacher label. No dev or frozen confirmation
plan is permitted in the distilled training allowlist.

Native A3/A4 32-step pilots remain conditional: require at least **16 distinct
verified training parents**, meaningful plan diversity, and working basic reply
grammar before launching. Freeze their shared accepted corpus and goal/plan
prompt, optimizer, label weighting and costs before training. A3 extends A2 with
this corpus; A4 extends A0 with the same corpus. Keep adapters unmerged and
preserve the corresponding pre-distillation parent artifacts.

Evaluate the direct student using the exact training planning prompt, fixed
candidate ordering and seeded sampling. Finish all candidate generation before
independent exact scoring. Top-1 stays the fixed first candidate; pass@k is
offline proposal coverage, not verifier-assisted deployment success. Report
diversity, valid full plans and proposal/query/token costs separately from
learned-model beam search. Teacher search and independent verification costs
belong in the training budget. No native student result is inferred from the
implemented wrapper or mock tests.

## Reporting and boundaries

List every arm as qualified, trained, evaluated, failed, interrupted or deferred
according to its actual artifacts. Report the prior format failures and this
recovery together while preserving their separate protocols and denominators.
Include all ID/OOD, teacher-forced/free-running, planning, retention and resource
outcomes, including negative findings. No change to the dense core, public model
upload, merge, routing/memory follow-up or integration is authorized by success
at this gate. A larger multi-seed, fresh sealed confirmation study requires its
own preregistration and independent claim/retention/compute criteria.

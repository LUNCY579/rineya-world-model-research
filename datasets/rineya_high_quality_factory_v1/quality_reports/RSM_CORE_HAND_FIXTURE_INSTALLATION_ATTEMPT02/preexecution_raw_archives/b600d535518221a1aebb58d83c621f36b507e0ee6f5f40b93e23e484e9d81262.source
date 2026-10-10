# RSM prospective option resumption contract v1

Frozen prospective public design, 2026-10-10 UTC. Actual author: `/root/research_architect`. New task kind `option_resume`, integer version `1`. This does not amend legacy `kind=skills`, its release, checker or targets. No interpreter, sample, gold, scorer or release is produced here. This contract specifies a finite supplied procedure, not discovery of an option policy or planning among alternative implementations.

## Typed input

The task has exactly `kind,version,initial_state,options,entry,mode,history,checkpoint,event_budget`. `kind` is `option_resume`; `version` is integer 1. Integers exclude booleans; no floats, coercion, overflow or implicit truncation. Objects are compared independently of field order; list order is significant. All option/parameter names are nonempty strings. All integer arithmetic is exact.

- `initial_state` has exactly integer registers `x,y`.
- `options` is a nonempty map from distinct names to objects with exactly `params,initiation,success,body`. `params` is an ordered list of distinct names. `body` is an instruction list. Every option's call graph, including calls in both conditional branches and unused options, is acyclic.
- An expression has exactly one form: `{const:integer}`, `{reg:"x"|"y"}`, or `{param:existing_parameter_name}`. Registers read the current global state; parameters read the current option's immutable bindings. A predicate is exactly `{all:[atoms]}`; an atom is exactly `{left:expression,cmp:"lt"|"eq"|"ge",right:expression}`. Evaluate all atoms against the same state/bindings; empty `all` is true. Predicates have no effects.
- Instructions have exactly these forms: `{op:"add",register:"x"|"y",value:expression}`, `{op:"scale",register:"x"|"y",value:expression}`, `{op:"swap"}`, `{op:"call",option:existing_option,args:{parameter:expression,...}}`, or `{op:"if",test:predicate,then:[instructions],else:[instructions]}`. `add` increments and `scale` multiplies the selected register; `swap` exchanges both registers simultaneously. Calls share global registers and return before the next caller instruction. Branches execute exactly one selected list.
- `entry` is exactly `{option:existing_option,args:{parameter:expression,...}}`. Every call/entry `args` map has exactly the called option's parameter keys. Evaluate **all** arguments simultaneously in the caller's pre-call state/bindings, then copy those integers into the callee. No later register change can rebind an argument. Root entry expressions may use constants/registers but cannot use parameters. All parameter references anywhere in an option refer to that option's declared parameters, including expressions in nested branches and call arguments.
- `mode` is exactly `fresh`, `resume` or `restart`. `history` is the ordered list of already committed semantic events from the original execution starting at `initial_state` and `entry`. `checkpoint` is null or the exact snapshot below. `event_budget` is a nonnegative integer: the maximum number of **new semantic events**, including entry, branch and return events, to execute. It is not a primitive-effect count.

Initiation is checked once, on each new option entry after argument binding and before body execution. A false initiation causes `initiation_failed` and no callee body effects. A successful entry does **not** imply a successful option. Success is checked against the current state and retained bindings only at normal body exhaustion. A false success causes `success_failed`. It does not roll back effects. Success is not an early-stop predicate, and initiation is not rechecked while resuming an already entered frame. Any runtime failure halts the whole run; there is no hidden retry, rollback, exception handler or alternate policy.

## Frames and event boundaries

A snapshot has exactly `state,frames,next_instance_id,root_pending,phase`. State is the two-register map. `next_instance_id` is a nonnegative integer. `root_pending` is boolean. `phase` is `running`, `completed`, `initiation_failed` or `success_failed`. `frames` is the ordered stack, bottom first. Every frame has exactly `frame_kind,instance_id,option,bindings,body_path,next_index`:

- `frame_kind` is `option` or `branch`; `instance_id` identifies its owning option invocation; `bindings` is the exact immutable parameter-to-integer map for that invocation.
- An option frame has `body_path=[]` and indexes its option's body. A branch frame indexes a selected branch list in that same option. Its `body_path` alternates an integer instruction index and `then`/`else`, for example `[2,"then",0,"else"]`; the empty path selects the body. A path must actually resolve through `if` instructions to a list. `next_index` is the next unexecuted instruction in that list, in range 0..list length.
- A caller or branch cursor advances **before** its call/selected-branch frame is pushed. Completed effects therefore cannot remain pending behind a child. Branch frames copy their owning invocation's bindings; they do not introduce new variables or invocation IDs.

The initial machine has `state=initial_state`, no frames, `next_instance_id=0`, `root_pending=true`, `phase=running`. Remove exhausted branch frames without generating an event, repeatedly, before every event and snapshot. Do not remove an exhausted option frame: its next event must test success and return. This normalization is part of the canonical checkpoint; accepting an unnormalized supplied stack is prohibited.

Each event is one atomic transition. The following event objects have exactly the listed fields:

| Event | Fields and transition |
|---|---|
| `enter` | `{event:"enter",instance_id,option,bindings,call_site,initiation_ok}`. Allocate and increment the invocation counter even if initiation fails. Root `call_site` is null; a nested site's exact shape is `{caller_instance,body_path,index}`. Bind arguments in the pre-call state. On true initiation push the new option frame with cursor 0; on false halt. Root entry clears root_pending. A nested call advances its caller cursor before this event's push/failure. |
| `branch` | `{event:"branch",instance_id,option,body_path,index,condition}`. Evaluate the current `if` predicate once, advance its containing cursor and push only the chosen branch frame. Record the actual boolean; never reevaluate a suspended chosen branch. |
| `effect` | `add`/`scale`: `{event:"effect",instance_id,option,body_path,index,op,register,value,state}`; `swap`: `{event:"effect",instance_id,option,body_path,index,op:"swap",state}`. `value` is the evaluated integer operand and `state` is the complete **after** state. Advance the cursor exactly once. The body locator is the instruction before advancement. |
| `return` | `{event:"return",instance_id,option,success_ok}`. Check the exhausted option's success predicate, pop its frame on either outcome, then normalize completed branches. False success halts with any unfinished caller frames retained. True root return, leaving no frames, completes the run; a successful child returns to its already advanced caller. |

Event IDs/locators are derived from actual execution, not supplied naming patterns. Running with root_pending generates root enter next; otherwise use the top frame's current instruction or exhausted-option return. Every valid program is finite because the instruction tree and call DAG are finite. Two interruptions cannot move a completed instruction back into the pending stack.

## History validation, modes and malformed policy

Validate in this order; there is no best-effort repair:

1. Validate the exact task envelope and program, expression types/references, call acyclicity, mode/version and budget. Missing envelope fields, unknown fields, bad integer types, missing option/parameter names or recursive calls produce `invalid_task`.
2. Starting from the initial machine, independently generate the exact next event for each history position and compare the supplied event with it using typed object equality and ordered lists. A malformed/non-list history, incorrect event field/value/order, or event after terminal phase produces `invalid_history`. A history is a committed **prefix**, not an unordered bag or an instruction-count claim.
3. For fresh mode require an empty history and null checkpoint. For resume/restart require a correctly typed checkpoint exactly equal to the normalized machine after that history, including every state value, frame, binding, cursor, invocation counter, root_pending and phase. Missing, malformed or unequal checkpoints, or forbidden fresh history/checkpoint combinations, produce `invalid_checkpoint`. Validation uses the original initial state/entry even for restart. This makes the history and checkpoint independently auditable instead of trusting a producer-supplied stack.

After validation:

- `fresh` continues the initial machine.
- `resume` continues the exact checkpoint machine. It executes only pending work and retains previously bound arguments/selected branches. A terminal checkpoint resumes with no new event and its existing terminal phase.
- `restart` deliberately starts a **new** machine from the checkpoint's current state, root entry and declared options, with frames empty, next_instance_id 0 and root_pending true. Root arguments/guards are evaluated again against that state. It does not pretend completed effects were undone. Invocation numbers restart inside this new run; history and new_events are separate logs. A valid failed/completed checkpoint may be restarted. Resume and restart are different visible requests, not an invisible corrective policy.

Execute until a terminal phase or event_budget new events, whichever occurs first. A terminal transition at the last allowed event is terminal, not paused. With zero budget a running machine is paused before its next event. Failed initiation/success retains the state containing all prior effects; unfinished caller frames remain in the terminal snapshot and cannot be resumed past failure under this version.

## Unique typed output

Return exactly `{status,error,checkpoint,new_events,previous_effects,new_effects,next_event}`. Status is `completed`, `paused`, `initiation_failed`, `success_failed` or `invalid`.

- On invalid input return exactly status invalid, error the first validation-stage code above, checkpoint null, new_events empty, previous_effects null, new_effects 0 and next_event null. This policy intentionally includes visible malformed histories/checkpoints in the valid supervision domain.
- Otherwise error is null. checkpoint is the exact normalized final snapshot. Its phase is running when output status is paused; otherwise phase equals the terminal status. new_events contains only actually executed continuation/new-run events, in order. previous_effects counts effect events in the validated history; new_effects counts effect events in new_events. For resume, history followed by new_events is a single valid original trace. For restart it is not: the two logs intentionally describe different runs.
- At pause, next_event is the uniquely determined next full event computed from the snapshot **without committing it**. Otherwise next_event is null. Predicting its after-state does not advance the checkpoint, count or cursor.

The supplied policy and event timing are deterministic, so there is exactly one typed result. There is no optimization, unspecified action selection or legal-plan tie to break. JSON field-order variants of the same object are equivalent; all specified list orders remain significant. Do not select a convenient teacher trace among alternative policies: introducing policy choice or a set of legal plans requires a separate public contract accepting all legal ties.

## Public matched-control meaning

All facts needed for execution, boundaries, arguments, guards, history, mode and budget are student-visible task fields. Context letters, difficulty, mechanism labels and expected outcomes must stay outside the prompt. The full contract is included before `Task JSON:\n` and the exact task JSON; no expected status or target is encoded in metadata as an input hint.

- Resume versus uninterrupted execution: use the same program/root request and a true committed prefix. Complete resumption must reproduce the uninterrupted terminal state/status and exact remaining event suffix. Prefix effects do not occur again.
- Resume versus restart: hold program, validated history and checkpoint fixed; change only mode. Require an actually committed non-idempotent effect, new future work and a measured difference in continuation trace and terminal state or status. Merely calling an operation non-idempotent is insufficient.
- Bound argument control: an argument evaluated from a register on entry remains fixed after the register changes; an actual later effect/guard/success depends on that saved value. A positive argument counterfactual changes one root argument while both runs initiate, and changes an actual outcome or trace. Do not credit unused formal parameters.
- Guard controls: change one root argument for a matched initiation pass/fail case, with no callee effects on failure. Separately change one literal in the root success predicate so the same body/entry/history produces the same effects but opposite success at return; this is a success-contract control, not a rollback claim.
- Multiple checkpoints: pausing at a second actual event prefix and resuming its independently validated snapshot must match the same complete suffix. Demonstrate retained nested callers/selected branch where actually present; do not infer a second interruption from a long trace alone.
- Tampered checkpoint: change one actually consequential state, cursor or saved binding while retaining the original history. Exact validation must reject it before executing any new event. A correct-looking final state cannot rescue a false history/stack. Actual validity/status must be derived, not inferred from a context label.

Later independent oracles must test every event boundary, zero/full budgets, failed entry/exit, empty branches/bodies, retained caller cursors, simultaneous binding, typed malformed cases and resume-prefix/suffix composition. These are prospective properties, not tests executed here.

This is bounded explicit interpreter execution. It does not establish learned hierarchy, discovered options, chosen termination policies, general recovery, student-state aggregation, sealed transfer or model efficacy. Public-design authorship and any later adjudication overlap, root guidance and common-model correlation must be disclosed in later receipts.

# Recovery H3 v1: verified planning distillation — prospective draft

**DRAFT ONLY. This is not an execution freeze or a confirmed result.** Freeze
the final corpus/task/config/source/adapter contracts before teacher collection
or student training. Native work remains subject to the shared resource lane.
No model was loaded and no sealed label was accessed to write this document.

## Question and baseline interpretation

Can independently verified plans generated through learned dynamics improve
amortized plan proposals while preserving the useful registered-F interface?
The qualified B0 adapter is already 24/24 on the fixed recovery four-step
state benchmark. Keep that ceiling and the fixed cohort. A planning gain is
possible even when no state-accuracy gain is possible; recovering an A0/A2
adaptation loss is not evidence of outperforming B0's state dynamics.

Use the existing recovery A2 and A0 **final 32-step** artifacts as two fixed
student starting points. Never select a checkpoint, parent seed or candidate
using development success. A3 continues A2; A4 continues A0. The same verified
planning corpus, row order, planner prompt, extra optimization and evaluation
tasks are used for both. Report changes relative to each arm's own warm parent,
not just the final A3-versus-A4 difference.

H1/H2 remain on hold for larger dynamics studies when their fixed recovery gate
fails. H3 is an independent teacher/student feasibility question using useful
B0 and the explicitly pinned final A0/A2 prototypes. Using A2 as A3's starting
artifact does not promote A2, override a failed H2 gate, or establish a benefit
of multi-horizon dynamics. Any H3 success remains a planning-adaptation pilot.

## Existing code that can and cannot be used unchanged

Permitted source review:

- `bwm/code/bwmn/rw2.py`: `proposer_prompt`, `target_actions`,
  `proposer_sft`, `parse_plan`, and the registered chat serializer.
- `bwm/code/bwmn/planner.py`: the action proposer contract and common-random-number
  candidate order (candidate 0 greedy; remaining candidates seeded).
- `bwm/code/scripts/train_proposer.py` and `self_distill.py`: training/data-flow
  examples only. Do not execute their historical pool builders or import frozen
  forward tables, evaluation banks or solver plans for this new study.
- New `dsv3wm` planning/proposer, distillation, recovery trainer and learned W1
  interfaces: reusable components, with the limitations below.

The present generic `distill.py` requires an all-legal source trajectory. Every
recovery training trajectory deliberately has two legal and two illegal steps,
so that filter cannot supply this H3 corpus. Teacher **planning tasks** must
instead reference the approved TRAIN parent/initial-state/rule allowlist; their
desired goals are independent of the source trajectory's action labels.

The generic `planning_prompt` and direct proposer use bare JSON arrays. They
are implemented research interfaces, but are different from the registered BWM
RegisterWorld action proposer, whose target is `{"actions":[...]}`. The old
cold-format failure makes replacing that registered interface again a poor
default. The current warm recovery trainer also explicitly blocks A3/A4; the
generic trainer creates a fresh different adapter and is not a valid warm-H3
runner. Keep those files/records intact and add explicit new wrappers.

## Registered interface and terminal-plan contract

The primary H3 wrapper must call these existing serializers without modifying
their bytes:

`rw2.proposer_prompt(x, goal, limit, budget=6, outcome="success")`

`rw2.target_actions(plan)`

Use action-only targets; no oracle intermediate states appear in a student
prompt or target. Training and inference use exactly the same legacy chat
prefix, system instruction, JSON object keys/order/separators and plan target.
Add a byte-equality regression test across representative permitted inputs.
Hash the reused `rw2.py` source and rendered prompt/target fixtures. Do not edit
registered BWM files or automatically import its old proposer data/weights.

The instruction says reach the goal **then finish**, within six actions. Primary
student success therefore requires a syntactically complete action object,
at most six actions, legal independent execution, a final `finish`, actual
termination at the goal, and no actions after that termination. This is stricter
than the existing generic `verify_plan` goal-only predicate. Keep the old
recovery planning result's goal-only metric unchanged; report its terminal
counterpart separately if needed.

A new syntax wrapper may reject duplicate JSON keys/nonfinite constants before
calling `rw2.parse_plan(text, "actions")`. This is an explicit common validation
tightening, not transition simulation. Unknown actions, wrong types, extra keys,
prose/fences, empty plans and overlength plans count as malformed. Action syntax
alone does not determine legality. Neither student generation nor syntax parsing
may call an exact transition or solver.

## Prospective teacher pool and learned-search budget

Before **any teacher outcomes**, freeze 32 TRAIN-only planning tasks from 32
distinct parents in the approved recovery 128-parent TRAIN bank. Include the
existing 16 training planning parents and choose the additional 16 using a fixed
input-only rule/seed. Desired goals use the declared 2–4 exact-move training-task
collection surface, with scoring metadata kept separate. Coverage difficulty
selection during task construction is allowed and recorded; no optimal plan
is emitted as a teacher target.

Freeze task IDs, parent IDs, goal hashes, originating TRAIN-row hashes, task
order, rule-limit coverage and generator/source hashes. Exclude every DEV parent
and all public/frozen confirmation parents according to the existing registry.
Retain the whole prospective task pool and its failures. Do not add easier
parents or new goal choices after seeing search success.

Teacher W1 uses the existing legacy learned forward interface:

- Beam width 1, depth at most 6, fixed action order.
- At most 30 learned transition queries per task.
- Prompt length at most 512 and at most 32 generated tokens per transition.
- At most 120,000 processed decoder-token positions per task, including padding
  envelope and unused capped forward accounting.
- Greedy dynamics generation, eval mode, the qualified FP32 standard cache and
  the coordinated CPU lane with at most four threads.

These are the current W1/depth6/query30 limits. They are explicitly different
from the earlier query20 pilot wording and must have their own prospective
teacher freeze. Record actual costs, not only caps.

For a shared corpus, use qualified immutable B0 as the default learned teacher F.
Do not let A3 and A4 each receive a selectively better corpus from its own F.
If an alternative teacher F is chosen, pin it before teacher outcomes and use
it for both students. No exact fallback is present in search, ranking, heuristic
distance or invalid-action handling.

## Delayed independent verification and corpus admission

Complete learned search/ranking for **all 32 tasks** and freeze those candidates
and raw causal traces before exact verification sees any candidate. Verify only
already generated learned nodes in their declared rank order. Do not insert
oracle paths, restart search after an exact failure, rerank a DEV plan using
exact success, or label a learned prediction as ground truth.

To meet the legacy terminal contract, a learned-goal candidate that does not
already end in `finish` may receive a deterministic syntactic final `finish`
**before independent verification**, provided its total length is at most six.
Record the uncompleted learned plan, completion rule and completed candidate hash.
Do not complete a candidate because an oracle reveals success; the trigger is
the already stored learned-goal prediction. Reject interior `finish` or
overlength candidates. This completion does not repair a wrong state prediction;
the completed plan still has to pass independent exact execution.

Run a source-hashed exact verifier after candidates are fixed. Check all actions
legal, every action executed, actual final termination, and final state equal to
the desired goal. Keep the first exact-positive candidate in the pre-existing
learned rank per parent, at most one whole-plan target per parent. Acceptance is
TRAIN-only label curation; it is not a test-time oracle planning result.

Positive admission additionally requires the candidate's already recorded
**learned predicted-goal success**. A true-positive discovered only by exact
scoring of a learned-nongoal candidate is diagnostic and does not enter the
primary student corpus. Likewise, extra exact-positive candidates after the
first retained one are diagnostic, not additional weighted parent labels.

Admission gate before native student training:

1. At least **16 distinct approved TRAIN parents** yield independently verified
   positive plans.
2. Accepted parents cover at least two training limits and accepted targets
   contain at least four distinct action-sequence fingerprints. Report full
   action/length/goal-frequency distributions and teacher coverage over all 32.
3. Every target passes fresh exact re-verification and canonical parent/TRAIN-row
   allowlist checks; every prompt/target is byte-equal to the registered serializers.
4. Teacher traces contain no gold-state resets, exact selection feedback,
   unregistered retries or new candidate construction after exact scoring.

If more than 16 parents qualify, retain all qualifying parents, one plan each,
in the frozen input-task order. Do not keep only the shortest/easiest 16 after
looking at outcomes. Rejected candidates remain **hard negatives**, including
wrong-state/false-goal predictions and malformed traces. They have no successful
plan-CE target and are not mixed into positive SFT. A separate negative-learning
objective is deferred; simply relabelling a failed plan `outcome=success` is
forbidden.

For every positive/negative, record task/parent/source-row SHA, original learned
candidate and rank, optional pre-verification completion, learned predicted
trajectory, exact verifier result/hash, raw text, costs and split provenance.
An exact-successful final plan does not imply the learned intermediate states
were correct; preserve that disagreement independently.

## Small warm student pilot and controls

After corpus admission, freeze final A0/A2 adapter/config/run hashes and the
positive/negative corpus hashes. Use FP32/TF32-off, unmerged rank16/alpha32/all
original projection targets and dropout 0.05. No dense parent update or new
vocabulary head is introduced. Fresh AdamW uses learning rate `2e-5`,
betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`, clip 1 and no scheduler.
Batch 1, 32 new optimizer steps, one declared training seed; do not extend the
run until a success threshold happens to be met.

At each step, use one shared recovery TRAIN trajectory and one verified plan
parent. Fix the trajectory draw order and a shuffled cyclic positive-parent
order before training; use the identical plan order for A3/A4. Each loss is a
mean over its valid response tokens:

- **A3:** warm A2, `L_h1 + 0.5 L_h2 + 0.5 L_h4 + L_verified_plan`.
- **A4:** warm A0, `L_h1 + L_verified_plan`.

The coefficient 1 on plan loss is this draft's fixed pilot proposal; changing it
requires a pre-execution config revision. Both arms receive identical successful
plan supervision. A3 has four decoder forwards per update and A4 two; they are
not equal-compute final arms. Count all labels, target/input tokens, updates,
forwards, vocabulary projections, active/runtime/memory costs and teacher search
plus verification costs.

Before a mechanism claim, add matched continuation controls:

- C3 starts from the same A2 artifact and replaces plan supervision with an
  extra one-step loss for the same 32 updates/four nominal forwards.
- C4 starts from the same A0 artifact and similarly uses two nominal one-step
  forwards per update.

These match nominal forward/update counts, not the different prompt lengths,
plan labels, head cost or data allocation. Report the mismatch. If resources
allow only A3/A4 and their warm-parent readouts, call it an implementation/pilot
comparison with continued-training and extra-supervision confounds. It does not
isolate a benefit of verified planning from receiving more labels or compute.

Whole-plan rows are primary. Suffix/first-action augmentation is deferred because
it increases labels per parent and may introduce new state inputs. Any later
suffix curriculum must use independently verified prefix states, remain tied
to the original TRAIN parent group and have its own matched supervision control.

## Fixed direct-student and with-search readouts

Use the existing fixed 12 DEV planning tasks, six ID/six OOD, whose parents are
shared with the 24-parent dynamics DEV cohort. This dependence is explicit;
neither endpoint is an independent confirmation bank. Freeze the new legacy
plan-contract scoring metadata before adapted outcomes and preserve every old
goal-only planning readout.

Read out warm A0/A2, A4/A3 and any continuation controls in this same interface.
For direct proposals use K=4, candidate 0 **greedy**, and candidates 1–3 sampled
with fixed per-parent CRN seeds, temperature 1 and top-p 0.95. Keep the BWM seed
and candidate-order convention. Maximum 80 output tokens per candidate, six
actions including `finish`, prompt at most 512, one requested outcome `success`.
Do not replace candidate 0 with the first parseable/exact-positive candidate.

Finish all candidate generation before exact scoring. Report:

- Fixed greedy top-1 full-plan validity, termination and goal success.
- Offline pass@4 coverage, explicitly not verifier-guided deployment selection.
- First-action legality and exact optimality, scored only after the action is
  fixed: for nonterminal goals, a legal first move reduces exact distance by one.
- Unique syntactically valid/unique exact-valid plans, length/action diversity,
  malformed count, proposal calls, tokens and processed positions.
- Dynamics h1/h2/h4 free/direct accuracy and the six existing permitted greedy
  behavioral retention fixtures; report every paired loss.

For a matched **student plus learned-search** readout, fix the forward adapter
to immutable B0 for all students. Four complete student candidates are rolled
under B0 in their original order, then selected by a frozen learned goal/length
ranking and deterministic tie-break. Exact execution occurs only afterward.
Label this concrete variant **learned candidate reranking**, distinct from W1
beam search and from a pure proposer.

Use a global cap of 30 candidate/transition queries per task and 120,000 processed
positions: four proposer queries plus at most 26 forward queries. All generated
tokens, prefill, padded work and capped forwards are charged. A six-action plan
needs at most six forward queries, so K=4 complete evaluation fits the cap.
Malformed candidates spend proposal compute but are never oracle-repaired.
Compare each student to its own warm parent's same reranking system and, as a
separate search reference, the frozen W1/depth6/query30 B0 planner. Equal caps
still require reporting actual compute and different search policies.

An optional student-seeded W1 beam variant needs a separately frozen wrapper and
explicit global proposal+search accounting. It may not quietly give students a
30-query W1 budget **in addition to** uncounted proposal calls. An evaluation
that also changes the forward adapter to A3/A4 measures joint proposer/dynamics
changes and must be labelled secondary, with the fixed-B0 contrast retained.

## Honest continuation gates

Teacher failure preserves a valid negative result; no student run starts without
the corpus admission gate. A trained student proceeds only to a larger
development study if it has at least 9/12 syntactically valid greedy proposals,
at least +2/12 paired net greedy full-plan successes versus its own warm parent,
no lost success in the matched fixed-B0 reranking readout, feasible coordinated
runtime, and no unexplained old-format behavior regression. Its dynamics ID
h4 net loss must not exceed 2/12; report total and OOD state losses as well.

These thresholds are operational pilot gates. Twelve cases/one seed do not
establish efficacy, calibration, broad retention or seed robustness. Report all
paired gains/losses and descriptive uncertainty, including negative results and
any grammar or complete-plan ceiling in warm parents. Comparing A3 with A4 alone
does not isolate multi-horizon transfer or verification. A larger seeded study
must freeze matched continuation controls, practical thresholds, claim family
and a fresh sealed bank before confirmation; no such access is part of this draft.

## If W1 fails to supply the corpus

If the fixed 32-task W1 pass yields fewer than 16 diverse verified parents,
preserve its coverage, search failures, negatives and costs; mark H3 native
training deferred under v1. Do not silently train on fewer parents, select new
easy goals or splice exact solver plans into its targets.

Permissible separately preregistered next experiments are:

1. Increase learned beam width/queries on the same approved TRAIN tasks, with
   explicit teacher compute limits and no exact feedback to search.
2. Generate additional candidates from a separately pinned existing registered
   action proposer using the exact legacy interface, rank using learned B0 and
   only then verify independently. This imports a new learned teacher skill and
   must disclose its training exposure/artifact hash and cost.
3. Prospectively collect a new TRAIN-only teacher pool before its outcomes, with
   a fixed difficulty curriculum and unchanged DEV cohort. The original failed
   pool remains in the report.

An oracle-plan teacher may be studied only as an explicitly named separate
oracle-supervised arm/protocol; it is not an automatic recovery fallback for H3.
No alternative changes old registered records, this fixed DEV cohort, or the
ban on historical sealed labels and plans.

## Minimal executable additions before an execution freeze

1. New legacy-H3 serializer/parser/student-generation wrapper with byte-equality
   tests, greedy/CRN order, strict syntax and terminal-plan contract. Reuse
   existing BWM serializers/sampler; do not rewrite the registered interface.
2. Prospective 32-task TRAIN allowlist and TRAIN-only teacher collector, with
   all-task delayed verification, raw causal/candidate provenance and negatives.
   Reuse learned W1 and the exact verifier as separate stages; generalize split
   checking only in a newly qualified source version.
3. New H3 row admission/re-verification function that checks legacy object
   prompt/target bytes, final termination, goal, all actions legal and parent/
   task/origin-row allowlists. The generic bare-array gate cannot be reused as
   though its contract were identical.
4. New warm-H3 trainer accepting explicit A0/A2 starting-adapter hashes, positive
   corpus allowlist and mixed objectives. Never use the generic fresh-adapter
   trainer or bypass the current recovery A3/A4 rejection implicitly.
5. Direct and fixed-B0 reranking readouts with first-action/offline pass@4 metrics,
   global cost accounting and delayed exact scoring. Include oracle-raises tests
   for generation/search and a top1-false/pass@4-true no-fallback fixture.
6. Source/config/data/corpus/adapter/native-runtime freeze and fail-closed checks.
   Record implementation, CPU tests, teacher-qualified, trained, evaluated,
   failed or deferred status from actual evidence. This draft itself supplies
   none of those native outcomes.

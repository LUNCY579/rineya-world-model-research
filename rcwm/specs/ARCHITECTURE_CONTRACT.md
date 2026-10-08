# Architecture contract / RCWM-001

Status: proposed production design. The CPU package implements only the explicitly identified subset.
Authoritative scientific definitions: report chapters “Formal model” and “Architecture.”

## Boundary A: observe

`HistoryEncoder.encode(available_history, environment_id) -> ContextBatch`

ContextBatch carries vectors `[B,Lc,d]`, validity mask `[B,Lc]`, availability timestamps,
checkpoint/tokenizer/capture hashes, and a feature-layout version. No future observations,
realized future policy actions or outcome labels enter this interface. A missing source hash blocks
native caching. Source tensors are read-only. The cache key includes mode, padding, precision,
position policy, prompt template and selected capture layers.

## Boundary B: propose and validate

`Proposer.propose(context, legal_registry, budget) -> CandidateSet`

A CandidateSet includes exact executable operators, arguments, executor versions, dependencies,
preconditions and a candidate-set hash. Record the set BEFORE scoring. Candidate generation must
not inspect real branch outcomes. Alternative evaluators reuse the same set. Proposed plans are
not silently replaced by realized adaptive action suffixes.

## Boundary C: predict

`Predictor.predict(context, candidate_set, mask_mode) -> PredictedConsequences`

The default predicts step consequences causally. Separate methods may score full precommitted
plans. State/feature distributions, feature schema, horizon, policy/plan identity and model version
are explicit. Goal weights enter only a readout for preference-only goals. Operative task instructions
that change execution belong in the dynamics input and executor arguments.

Suggested initial sidecar: width 256, 3 layers, 4 heads, latent width 64, horizon 4. These are development
screen values, not optimality claims. The supplied toy model is smaller (40,325 parameters).

## Boundary D: score and admit

`Readout.score(prediction, goal) -> utility, uncertainty, support`

A learned latent needs a fitted readout; do not dot a hand-designed six-element reward vector into
an arbitrary latent vector. Distinguish expected utility from utility of the expected latent. Hard
permissions and constraints are enforced independently. Unsupported or uncertain candidates may
trigger a registered fallback. Calibrate thresholds on development data only.

## Boundary E: execute and record

`Executor.step(action, snapshot) -> actual_observation, status, costs, done_flags`

Execute the admitted first action, not an entire speculative plan by default. Append actual evidence,
then replan. Tool outputs are untrusted data. Record timeouts and failures, not only successful traces.
Observed and predicted consequences never share one field. A label source must be auditable.

## Native typed-slot route

Preserve the original language route and output head. Use an explicit mode switch and separate
projections/heads. Infer dimensions from the pinned native configuration. Any new token additions,
embedding resize, tied-weight change, precision change or bidirectional mask is a separate qualified
intervention. Register trainable modules before optimizer creation and verify no other gradient or
weight changes. Do not merge the adapter into the anchor during initial qualification.

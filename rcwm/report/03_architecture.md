# Architecture and implementation boundaries

## One conceptual model, two implementation routes

The intended architecture is a Transformer that relates history, actions and future consequences in a shared computation. The safest first experiment uses a **sidecar** so representation learning can be debugged without changing Rineya. That is a staging choice, not abandonment of the user's unified-model idea. A later **native typed-slot adapter** implements the same contracts inside the actual Rineya backbone. Compare both rather than assuming that sharing weights is always more efficient.

For the sidecar, a frozen capture adapter extracts a bounded set of Rineya context vectors. A small typed Transformer receives those vectors and candidate actions. It outputs consequence features, next-state summaries and optional uncertainty estimates. The goal readout scores predicted consequences. A controller chooses a feasible action, executes it in the real environment and re-encodes the actual new observation.

For the native route, introduce typed embeddings or projected continuous slots at Rineya's verified hidden width. Train limited adapters and small output heads while retaining the ordinary language route. Use the supported embedding-input interface or a tested wrapper; do not silently resize the language vocabulary and retie its output weights without checking exact language-logit preservation. Stock Qwen3 documentation is an interface reference, not proof that a custom Rineya loader implements it identically. [S42]

## Proposed component contracts

| Component | Input | Output / invariant |
|:--|:--|:--|
| History encoder | Available text, evidence and events | Time-stamped context slots; no future facts |
| Action encoder | Operator, arguments, budgets, version | Typed action vectors; executable semantics |
| Consequence predictor | History and permitted action prefix/plan | Consequence distributions or embeddings |
| Outcome decoder | Consequence representation | Independent observable targets for grounding |
| Goal readout | Goal weights or learned task embedding | Utility; no alteration of physical facts |
| Candidate proposer | Available state and legal operator set | Bounded candidate plans with validity checks |
| Controller | Predictions, support, constraints | One admitted action or a safe fallback |
| Recorder | Actual action and actual result | Immutable training/evaluation evidence |

The predictor must not receive its targets in its forward signature. In particular, a tensor named `future_states` should not be accepted by the prospective evaluator. Retrospective representation training may encode outcomes to create targets, but target construction lives in a separate module with a separately audited visibility contract.

## Typed token construction

For event $i$, form a projected vector

$$
x_i=P_{\mathrm{type}(i)}v_i+e_{\mathrm{type}(i)}+e_{\mathrm{time}(i)}+e_{\mathrm{subtask}(i)}.
$$

Text uses the ordinary tokenizer and embedding path. Continuous state summaries and consequence slots use learned projections. Operator embeddings distinguish operations; argument encodings distinguish `CALCULATE(2+2)` from `CALCULATE(large_expression)`. Duration and resource encodings prevent the model from treating an expensive verification process as a free semantic label. Type IDs distinguish a proposed outcome from an observed outcome even when their dimensions match.

A candidate serial sequence can be serialized as history slots, an action slot, an outcome-query slot, another action slot and another query. Query slots contain learned placeholders, not ground-truth outcomes. In the first causal version, the query for step $t$ can access only the initial observed history and candidate actions through $t$. The model learns to integrate the consequences of earlier planned actions implicitly; it does not receive teacher-forced future observations at evaluation.

Outcome vectors do not have to be literal vocabulary tokens. Separate projections and heads can preserve the useful Transformer interface without allocating thousands of arbitrary reward strings. Learned discrete codes are a later representation choice, not a prerequisite for the architecture.

## Pilot tensor specification

The following are proposed screen values, not measured optima. Start with a 128–256-dimensional sidecar, 2–4 layers, 4 attention heads, 32–64 learned consequence dimensions and a small set of grounded features. For native integration, infer width and layer layout from the checkpoint. Do not hard-code stock-model dimensions into persisted captures.

A training batch contains context `[B,Lc,dc]`, proposed actions `[B,H]` plus argument features, target consequences `[B,H,F]`, target-validity masks `[B,H,F]`, step-validity masks `[B,H]`, optional next-observation targets and a split/parent index. Goal weights may have shape `[B,F]` for a measured-feature baseline; learned-latent readouts must fit weights in that latent space using allowed calibration data. A hand-written six-dimensional goal vector cannot simply be dotted into an arbitrary learned 64-dimensional latent.

The predictor should expose both `predict_step` and `predict_plan`. If these share weights, specify their masks. Output a mean and per-target uncertainty only where the training likelihood makes this meaningful. A vector of independent standard deviations is not a full covariance model. Store units and feature-normalization metadata with every model artifact.

## Backbone reuse and target stability

Extract representations at explicit action boundaries. Compare last-token, pooled evidence and multiple selected-layer summaries on development data. Normalization and pooling must be consistent across capture modes. Do not assume a raw subtraction of hidden states from different prompts measures a meaningful semantic transition.

A frozen encoder gives stable targets and cheap caching. A learned encoder can improve task relevance, but then the predictor chases moving targets. Use a separately versioned target encoder, stop-gradient targets and a controlled exponential moving average only after the frozen-target baseline works. Recompute or invalidate caches whenever encoder weights, tokenizer, prompt template, mask, precision or capture location changes. A stale representation cache is a scientific data error, not merely an engineering inconvenience.

The CPU scaffold intentionally uses a tiny synthetic state input rather than pretending to load Rineya. Its learned latent bottleneck is grounded through feature prediction. It demonstrates the interface and tests selected invariants; the fully self-supervised latent-consequence arm and native checkpoint integration remain planned work.

# Executable actions and consequence semantics

## Give each token an operational definition

`RETRIEVE` must name a retrieval corpus, query generator, top-$k$, evidence budget and tool version. `VERIFY` must identify what is checked and the verifier's authority. `RECUR` must specify a qualified recurrent module, depth, gating value and termination rule. `ANSWER` must identify the format and final evaluator. Two runs with the same operator name but different prompts or tool versions are different action mechanisms.

An action record contains its ID, parent events, arguments, preconditions, anticipated resource needs and permitted side effects. The executor returns a status from a fixed set such as success, invalid precondition, timeout, tool error or explicit refusal. A failure is not silently dropped from training. Dropping unsuccessful calls biases the predictor toward imaginary reliability.

For the first real Rineya pilot, use only a few controllable operations: retrieve a document span from a fixed corpus, execute a bounded arithmetic expression, check an answer against a symbolic constraint, and submit a final answer. Broader cognitive labels such as “decompose” or “think harder” can follow once their execution contracts are stable. Learned action representations are useful precisely when there is actual structure in the available operations. [S10]

## Grounded feature dictionary

Use a versioned dictionary with units, observability and label source. A initial example includes terminal task success, a newly verified constraint, contradiction detected, useful evidence obtained, elapsed milliseconds, generated tokens, tool failure and irreversible constraint violation. Some are binary, some counts, some nonnegative costs. Predict their distributions or normalized continuous values as appropriate.

Do not equate an observed reduction in the model's entropy with a correct conclusion. Call it “entropy change,” not “information gain,” unless the experiment provides a well-defined information target. Similarly, “progress” requires an external measure such as satisfied prerequisites or a reduction in known distance to a simulator goal. A fluent narrative that says progress occurred is not a measurement.

Earlier brainstorming placed retention beside compute and correctness in a reward vector. For ordinary inference actions, that is misleading: the frozen model does not change its weights when it retrieves a document. Retention belongs in **cross-update evaluation**, unless the selected action is itself a training update with measured downstream effects. Keep the feature inventory honest about the timescale of each quantity.

## Avoid double-counted reward

A step that verifies a prerequisite can deserve a shaping signal, but the final success reward should not be paid repeatedly after success. The plan objective must specify whether intermediate signals are explanatory features, auxiliary supervision or actual additive rewards. Those roles differ.

For example, store elapsed time as a positive measured quantity and place a negative coefficient in the utility readout. Do not encode negative time and also multiply by a negative penalty. Count candidate-generation and scoring costs in system reporting even though they do not appear in the environment transition. An improved controller that spends ten times as much on evaluation has not demonstrated compute efficiency.

## A constructed example

A task requires a key, an opened door and a collected object. A shortcut collects the object immediately but causes damage. Consequence features are `[object_gained, damage, action_cost]`. A safe three-step route yields a cumulative vector `[1,0,3]`; a one-step shortcut yields `[1,1,1]`.

With weights $w_{\mathrm{safe}}=[10,-20,-1]$, the route scores $7$ and the shortcut scores $-11$. With $w_{\mathrm{fast}}=[10,-1,-2]$, the route scores $4$ and the shortcut scores $7$. These are constructed arithmetic values, not model predictions or empirical results. The physical futures did not change; the preference did.

This example is intentionally simple. If success also requires no damage, then utility is a conjunction rather than this linear readout. Add the relevant joint feature, predict a joint distribution, or use a nonlinear scorer with an appropriate outcome distribution. Do not use linear scoring outside the feature family and blame the world model for an incorrectly specified objective.

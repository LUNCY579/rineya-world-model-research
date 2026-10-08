# Planning, uncertainty and real-environment feedback

## Separate proposing from evaluating

Generate a bounded set of legal action plans using fixed heuristics or the frozen Rineya proposer. Validate syntax, preconditions, dependency constraints and permissions before scoring. Cache the candidate set and its hash. Every evaluator in a comparison receives exactly that set.

For an admitted candidate $A$, score an expected task utility and a development-calibrated model-risk penalty:

$$
J(A,g)=\mathbb E[U_g(\widehat\Phi(A))]-\beta\,\widehat{\mathcal U}(A).
$$

This is a heuristic conservative score unless a formal bound is separately established. Enforce hard constraints through a feasible-set filter, not through arbitrarily large learned penalties. A permission violation must not become acceptable because the model predicts enough reward elsewhere.

Execute only the first admitted action, observe the actual result, update the history and replan. An open-loop plan is a proposal, not a commitment to ignore new evidence. A safe fallback can be a verified tool, an inexpensive direct answer on a qualified class, a request for missing information, or abstention under the task's rules.

## What uncertainty means

Separate stochastic outcome variability from lack of model knowledge. In a deterministic toy environment, residual uncertainty primarily reflects model error or missing state; in a stochastic environment, repeated execution can vary even under a perfect model. Ensembles are a practical baseline, not an oracle for ignorance. Test whether uncertainty predicts error under held-out actions, longer horizons and shifted contexts. [S24, S25]

Report calibration by operator and difficulty as well as overall. Use Brier score or log loss for probabilities; reliability plots, empirical coverage and interval sharpness for continuous outcomes. Define the abstention threshold on development data and freeze it. Measure coverage and real task quality together: rejecting everything is not a useful uncertainty system.

Candidate search itself can exploit the evaluator. Sweep plan count and horizon, and compare predicted gains with actual utility. If predicted reward increases while real utility falls, the controller is exploiting model error. Keep failed cases as a dedicated stress set rather than retraining on sealed evaluation failures. Proxy-reward overoptimization and offline distribution shift make this an essential test. [S23, S26]

## Direct selection, distillation and RL

The first controller can choose the highest-scoring permitted action without online policy-gradient training. That does not make it fundamentally unrelated to RL: it is a learned decision system trained from interaction data. The question is which learning components are needed, not which label is more impressive.

A later distilled policy can imitate the choices of a qualified, more expensive planner, reducing inference cost. Verify that the teacher's preference remains valid under real outcomes. Imagined RL is later still: mix short simulated trajectories with real anchors, use conservative support checks and validate every promoted policy in the actual environment. The Dreamer and MBPO inspirations support this staged idea, not a reason to start with unrestricted self-generated reward loops. [S01, S22]

## Evaluation of control versus representation

Keep three levels of results separate: prediction fidelity, decision quality within a fixed candidate set, and end-to-end agent performance including proposal failures. A strong scorer cannot select an action that was never proposed. Report the candidate oracle gap to show whether further improvement should target the proposer or evaluator.

A model can predict many irrelevant details accurately yet rank actions poorly. Conversely, it can rank familiar candidates well without generalizing dynamics. Both are legitimate limited results. The “world model” label should be scoped to the evaluated environment, observable variables and intervention coverage. Recent language-world-model work similarly motivates evaluating fidelity and actual agent utility separately. [S38]

# Long context, delayed credit and parallel task structure

## Long-context evaluation grid

Propose context bands of 512, 2,048, 4,096 and 8,192 input tokens, subject to measured hardware limits; action horizons of 1, 2, 4 and 8; and dependency distances varied independently. Hold total relevant information fixed while adding distractors in one test. In another, increase the number of genuinely relevant prerequisites. A context-size configuration is not a claim that the model uses that context effectively.

Use exact synthetic references for early tests: a constraint introduced near the beginning changes which final action is legal, two distant facts must be combined, or an intermediate tool response invalidates an earlier assumption. Report position sensitivity and truncation behavior. LongBench and RULER offer later external context-evaluation references, but do not replace interactive tests. [S29, S30]

For long raw prompts, compare full-history capture, bounded evidence retrieval and hierarchical event summaries. Count encoding and retrieval costs. A compressed history can make the sidecar cheap while discarding the only fact that distinguishes two actions. Test that possibility using adversarially placed prerequisites rather than trusting a summarizer's own explanation.

## Delayed credit without hindsight leakage

A useful early action may have no immediate success reward. Supervise the eventual consequences while preserving when evidence becomes observable. A retrospective return-decomposition model can propose credit assignments; a prospective decision model must predict those outcomes from information available before acting. RUDDER is relevant to delayed credit, but an attention link from an early action to a final reward is not automatically causal attribution. [S19]

A useful ablation compares terminal-only grounding, measured prerequisite features and learned future-consequence embeddings. Match the final objective and label costs. Intermediate supervision may improve training, but then the gain is not purely architectural; the study should identify the contribution of extra feedback.

## Represent parallel tasks as events and dependencies

A parallel task is not simply a serial action list with subtask tags. Define a directed acyclic graph of prerequisite relationships. Each event has a start time, completion time, dependencies, resource claims and output-availability time. Two actions may be independent in logic but compete for the same GPU, file lock or API quota. A plan is feasible only when both dependency and resource constraints hold.

Begin with synchronous rounds: independent actions launch together, then their outcomes become available together. Later add asynchronous completion, failures and cancellation. A query at time $t$ may attend only to outcomes available by $t$, not to a sibling task that finishes later. Proposed action metadata can be visible according to the selected planning mode; actual results cannot arrive early because their token was serialized earlier.

Test concurrency benefits against a serial controller at equal resources and against an oracle scheduler. Metrics include task success, makespan, resource cost, constraint violations, wasted work and cross-subtask error propagation. A shorter action sequence is not necessarily a shorter wall-clock run.

## Cross-task consequence interactions

When two tasks must both succeed, marginal success probabilities are insufficient unless independence is justified. A shared failure source can make the joint success probability very different from the product of marginals. Include joint outcome labels, shared scenario variables or sampled joint trajectories for this stage.

Test whether the model learns reusable prerequisites rather than task IDs. Hold out new compositions of familiar operations, permutations of independent subtasks and changes in resource contention. Preserve counterfactual families in a single split. A system that succeeds only on the serialized order seen in training has not learned the claimed parallel structure.

## What attention does not buy for free

Attention provides a mechanism to connect positions and train on many positions in parallel. It does not enumerate an exponential plan space, make all future observations available, infer causal structure from any dataset, or execute tools simultaneously. The plan must still have a proposer, a feasibility check and a measured execution budget. The project's hypothesis is that shared contextual computation improves prediction and ranking—not that attention abolishes planning. [S40]

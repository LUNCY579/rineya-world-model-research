# Learning the consequence space and testing clustering

## Preserve the discovery hypothesis

The first measured-feature model is a debugging instrument. The actual research hypothesis is that a representation learned from transitions contains useful distinctions we did not fully prescribe. A credible program therefore needs three parallel representation tracks: measured features, learned continuous features, and learned discrete or clustered features. Each must be compared using the same action data and the same downstream reward-label budget.

For learned continuous features, encode observed transition or trajectory outcomes with a target encoder and predict that representation from the preceding history plus action. Add a small decoder for independently checkable properties. This makes it possible to detect a representation that predicts itself perfectly while carrying no useful information. Use a held-out linear probe and a nonlinear probe separately: a high-capacity probe can hide a poorly organized representation.

Predictive contrastive learning, variance/covariance regularization and vector quantization supply alternative mechanisms. They should be introduced one at a time. None guarantees that the resulting latent dimensions correspond to useful rewards. [S14–S16]

## Clustering protocol

Begin with train-only standardization and a continuous baseline. Fit $k$-means to TRAIN embeddings for a small predeclared set such as $K\in\{16,64,128\}$. Freeze centers before development or test assignment. Compare hard assignments, soft distances and the unclustered vector under equal downstream model capacity. A random projection and random-label cluster control help identify whether a gain comes from regularization or parameter count rather than semantic grouping.

For a vector-quantized arm, log code usage, entropy, empty/dead codes, assignment turnover, quantization distortion and downstream regret. Resetting dead codes is a training intervention that must be logged and must not use evaluation examples. A codebook can stabilize a vocabulary of outcomes, but can also merge rare catastrophic failures with frequent harmless events. Evaluate failure distinctions explicitly rather than relying on average reconstruction error.

Inspect clusters only after the quantitative screen. Draw a blinded sample per cluster, have an evaluator describe observable commonalities, then test those descriptions on fresh examples. Do not let a compelling hand-assigned name turn into an unsupported assertion that the model has discovered a cognitive primitive.

## Sparse grounding experiment

Use nested label budgets, for example 1%, 5%, 10%, 25% and 100% of the same TRAIN parent set. All representation arms receive the same unlabeled transitions; all receive identical labeled parents at each budget. Count verifier calls, human comparisons, teacher generation, rejected traces and environment execution separately. “Unlabeled” transitions may still be expensive to collect.

Fit the utility readout using only the permitted calibration labels. Separate two transfer tests. **In-span revaluation** changes weights on already represented features. **New-outcome transfer** introduces a property not explicitly included in prior grounding. The latter can benefit from learned latent structure, but requires evidence that the representation retained that property. Neither test may reuse variants of held-out parents in pretraining or clustering.

The principal label-efficiency endpoint is the number of grounding labels required to reach a preregistered action-regret or success target. Report uncertainty on that curve and the total data-production cost. A claim that clustering uses “10% of labels” is not justified by selecting one favorable random label subset.

## Representation failure diagnostics

Track feature variance, covariance spectrum, nearest-neighbor stability, decoder quality, cross-goal sufficiency, action sensitivity and nuisance sensitivity. Compare prediction under real actions against shuffled action labels. If shuffling barely matters, the model may predict outcome frequencies from the state while ignoring the proposed intervention.

Do not force all successful outcomes to be neighbors. Different successful routes can have different costs, risks and prerequisite effects. A contrastive objective that collapses them may improve a binary success probe while harming goal transfer. Likewise, “different action” is not automatically a negative pair: two tools may cause the same observable effect.

# Information flow, masking and counterfactual validity

## Three masks, three meanings

**Prospective causal transition mask.** A query for the consequence of action $a_t$ may see the observed prefix and candidate actions through $t$. It may not see later actions, outcomes or rewards. Changing $a_{t+1}$ must not change the predicted consequence at $t$. This is the default dynamics interface.

**Prospective whole-plan mask.** A plan-value or plan-outcome query may see a fully specified open-loop candidate plan, including later proposed actions. All unobserved future outcomes remain absent or masked. This can score the total plan, but it should not claim that later choices physically cause earlier consequences. A suffix-aware auxiliary scorer and a causal dynamics head may share some parameters while maintaining separate visibility tests.

**Retrospective mask.** A representation learner may encode a completed trajectory to learn outcome structure or redistribute credit. Its output is a training target or retrospective diagnostic, not a prospectively available state. Confusing this mode with the deployed evaluator is a particularly easy way to manufacture excellent validation scores. Masked trajectory research motivates these interfaces; deployment validity is our responsibility. [S08, S09, S18]

## The subtle future-action leak

Even when rewards and observations are masked, a recorded future action can reveal a hidden observation. A logged policy may choose `GO_LEFT` only after seeing a secret clue. Showing that future action to an earlier predictor leaks the clue. An action column is not safe merely because it is called an action.

For the first plan-conditioned dataset, choose candidate plans **before** executing their branches, using only the shared starting observation and allowed information. Record `plan_origin=open_loop_precommitted` or a policy-tree contract with explicit observable branch conditions. For retrospective trajectories selected by an adaptive policy, use causal prefixes or include an honest policy-conditioned target; do not treat the realized suffix as a freely chosen plan.

## Predictive association versus causal effect

Logged trajectories can be confounded by environment difficulty, hidden state and behavior-policy competence. To learn an action effect, prefer exact simulator snapshots and branching interventions: restore the same starting state, execute different permitted actions, and compare actual consequences. Keep all branches in the same data split. In stochastic tasks, log random seeds and use repeated draws; shared random numbers can reduce variance but do not turn stochastic outcomes into deterministic facts.

A language-model hidden-state intervention is not automatically an intervention on an external task variable. First test whether its predicted changes correspond to observable environment changes. Only then study stronger internal causal hypotheses. Attention maps, cluster assignments and decoded labels provide hypotheses for interventions, not proof of causal structure.

## Tests that must fail before the fix and pass after it

Perturb a hidden future reward and confirm the prospective input bytes and outputs are unchanged. Perturb future action suffixes and confirm causal earlier outputs are unchanged. Introduce a future timestamp in a supposedly known observation and reject the record. Replace a legal precommitted plan with a behavior-selected suffix and require a different mode or an error. Change padding, batch order and unrelated examples and verify the expected numerical tolerance.

Mask Boolean conventions deserve explicit tests: the `torch.nn.Transformer` API marks blocked positions with `True`; other attention APIs can use the opposite convention. A shape-correct mask can therefore be completely wrong. Pin the installed API and test information flow behavior, not just matrix dimensions. [S41]

## Parallel output is not joint uncertainty

A Transformer can produce reward or feature predictions at every query position in one pass. Independent heads at those positions generally estimate marginals. They do not guarantee coherent joint samples across steps, tasks or reward dimensions. If a task requires “both subtasks succeed” or a correlated failure mode, test the joint event directly. Later options include a shared latent scenario variable, an autoregressive consequence decoder or a distributional trajectory model. Count their sampling cost in comparisons.

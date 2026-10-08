# Data program and environment ladder

## Start where errors can be localized

The first environment should be simple enough that an oracle can exhaustively evaluate candidate plans. The supplied KeyDoor simulator is a mechanics example, not the final benchmark. Build a broader symbolic suite with prerequisite chains, bounded program execution, document-evidence retrieval and resource-constrained parallel subtasks. Each environment should expose immutable snapshots, legal-action validation, independent outcome checking and a controllable observation function.

Use separate axes for context length, decision horizon, branching factor and partial observability. A 16,000-token prompt with one action is not a long-horizon task. A 20-action sequence with a tiny state is not a long-context reading task. A fair evaluation varies one axis at a time, then tests their combinations.

Only after the exact simulator gates should the program move to TextWorld or ALFWorld-style interactions, and later to a sandboxed web benchmark. These environments offer richer interactive structure but introduce additional language and state-aliasing failure modes. External benchmark acquisition, licenses and versions remain a preflight task. No real purchases or account changes belong in the training environment. [S31–S33]

## Collection policy and coverage

Collect a mixture of legal random exploration, heuristic actions, the frozen Rineya proposal policy and deliberately selected alternatives. Log which policy selected each action and its available information. Retain failures, invalid attempts, no-ops and timeouts with explicit status labels. An all-successful dataset teaches the model little about the actions it most needs to reject.

For a practical pilot, propose 2,000 independent training parents, 300 development parents and 500 sealed evaluation parents, with several matched branches per parent. These are **starting budget proposals**, not a power calculation or a guarantee of sufficient coverage. Use the development pilot to estimate variance and effect sizes; freeze the final sample size before opening the sealed set. If the required sample size exceeds the compute budget, narrow the claim or improve measurement rather than reporting an underpowered absence of harm.

Balance training by informative transition types, not only by trajectory count. A million repeated no-ops can dominate reconstruction loss. Preserve natural prevalence for a separate calibration set so a balanced training sampler does not produce misleading real-use probabilities. Report support for each operator, argument type, context band, goal family and failure mode.

## Data contract

Every trajectory record needs a schema version, parent ID, counterfactual family ID, episode ID, split, environment/version hash, initial observation, source history and observation timestamps. Events carry action/operator IDs, arguments, executor hash, preconditions, start and end times, dependencies, actual outputs, statuses, feature values, feature-validity masks, reward-source IDs, termination and truncation. Optional hidden-state references carry model/tokenizer/capture hashes and byte offsets.

Keep **observed facts**, **proposed actions**, **predicted futures** and **retrospective annotations** in distinct fields. Do not overload an `outcome` field with whichever version is currently convenient. A typed schema reduces the chance that a training target later becomes an inference feature.

Prediction logs need candidate-set identity, all candidate scores, uncertainty/support estimates, chosen action, fallback reason, actual outcome and resource totals. Storing only the selected action hides whether gains come from the proposer or scorer. Store all candidates before execution so later analysis cannot silently change the comparison set.

## Split integrity

Partition by a stable problem-family key before generating branches, paraphrases, goal variants or labels. All descendants remain together. When tasks share a generator template or source document, use a sufficiently broad grouping key to prevent template leakage. A random row split of trajectories from the same parent is not a generalization test.

Normalizers, codebooks, representation pretraining, reward readouts and hyperparameter selection must respect the split boundary. Separate a **new-goal calibration set** from the held-out evaluation parents. New goal labels may be permitted during adaptation, but labels from evaluation scenarios are not. Record this distinction in the experiment contract rather than calling every goal adaptation “zero-shot.”

Preserve a frozen ledger of hashes for raw data, parent partitions and generated variants. Deduplicate before split finalization and recheck after generation. Where a benchmark has its own train/test partition, retain it and add internal development splits only within training data.

## Label provenance and privacy

Each feature label records how it was obtained: simulator state, executable checker, deterministic tool, human comparison or learned proxy. Report accuracy against independent checks for any learned labeler. Use separate model identities for generator and evaluator when appropriate, and disclose shared training or correlated errors. Never label the reward model's own prediction as ground truth.

Do not include private email, real credentials, personal identifiers or proprietary documents in synthetic training data by default. Tool outputs are untrusted data, not instructions to alter the experiment. Redact secrets before logging; retain enough provenance to reproduce a sanitized example. Store only the hidden representations needed for the experiment, with explicit access and retention policies.

# Training recipe and staged objectives

## Stage A: measured consequences as an instrument

Train a small model to predict known consequence features from initial state/history plus a precommitted action prefix. Use binary cross-entropy for Bernoulli events, cross-entropy for categorical state variables and a suitable regression likelihood for measured continuous quantities. The CPU scaffold uses mean-squared error for simple synthetic numerical features; production binary probabilities require the corresponding calibrated heads.

Normalize continuous targets using TRAIN statistics. Preserve physical units in the artifact. Apply feature-validity masks so an unavailable label is not treated as zero. Start with a fixed-length short horizon and a single frozen encoder. Require the model to overfit a tiny diagnostic set before larger experiments; failure there usually indicates a contract, optimization or observability problem.

## Stage B: learned consequence representations

Let $z_t=E_{\bar\eta}(\tau_t)$ be a target outcome representation and $\hat z_t=F_\theta(H_t,a_t)$ its prediction. A basic predictive loss is

$$
\mathcal L_{\mathrm{pred}}=\frac{1}{|M|}\sum_{t\in M}\|\hat z_t-\operatorname{sg}(z_t)\|_2^2.
$$

The stop-gradient operator prevents the predictor from trivially moving both sides through that term, but does not alone prevent the whole representation from collapsing. Combine it with a grounded decoder, contrastive prediction or an explicitly tested variance/covariance objective. Start with one anti-collapse mechanism, not all of them. CPC and VICReg provide relevant objective families; our exact transition augmentations and feature targets remain hypotheses. [S14, S16]

An auxiliary observable loss can decode state predicates, action status and elapsed resources. Evaluate both the auxiliary prediction and downstream utility. A representation might predict all easy surface properties while dropping a rare critical prerequisite. Stratified diagnostics and held-out conjunctions are needed to expose that failure.

## Stage C: sparse utility grounding

Freeze or slowly adapt the consequence encoder and fit $u_\omega(z,g)$ using verified returns or preference pairs. Use a linear readout first because it exposes what is actually reusable. Compare a small nonlinear readout only after the linear result is known. For pairwise labels, train a calibrated preference model rather than pretending that every comparison reveals an absolute numeric reward.

When fitting goal weights, constrain them to the declared feature family or record the extrapolation. New-language goal prompts must be translated to a versioned task representation and tested against explicit objective definitions. Do not claim immediate generalization from a natural-language goal embedding without evaluating that mapping.

## Stage D: future features and consistency

Add direct predictions of discounted future consequences at horizons 1, 2, 4 and 8. For a policy-conditioned successor head, enforce the correct continuation-policy identity. A finite-horizon consistency target has the form

$$
\psi_K^\pi(b_t,a_t)\approx\phi_t+\gamma\,\mathbb E_{a'\sim\pi}[\psi_{K-1}^\pi(b_{t+1},a')].
$$

Bootstrap only where continuation semantics are valid. True terminal states have no future contribution. A time-limit truncation is not automatically a terminal state; use an explicit modeling convention. The environment API distinguishes these cases, but the learner must still handle them correctly. [S43]

Compare direct successor prediction with recursive one-step rollouts. The former can reduce accumulated prediction steps but hides intermediate structure; the latter supports detailed replanning but can accumulate error. A hybrid is a research option, not the default until these failure modes are measured. [S04, S22]

## Objective budget and optimization

A later combined objective may contain predictive, observable, reward, variance, contrastive, codebook and retention terms. Record each coefficient, its units, its gradient contribution and its activation stage. Do not present a single long loss equation as an implementation recipe. Begin with observable prediction, then add one mechanism at a time.

For the sidecar, propose AdamW with a small development-only sweep over learning rates such as $10^{-4}$, $3\times10^{-4}$ and $10^{-3}$, fixed token/transition exposure, early stopping on an independent development criterion, gradient clipping and finite-loss checks. These are engineering starting points, not paper-derived optimal settings. Log samples seen, optimizer steps and total modeled tokens. Match baselines on training examples and total tuning budget, not only epochs.

For native adapter training, verify forward/gradient parity on a small case, then train only registered parameter sets. LoRA and quantized-base adaptation are reasonable mechanisms, but their efficiency does not certify retention or numerical fidelity. Quantization must be evaluated as a separate variable. [S27, S28]

## Representation drift and replay

When the representation encoder changes, old goal weights and codebook centers can become invalid. Either keep the encoder frozen, version and refit dependent readouts using allowed data, or train an explicit alignment map and evaluate its error. Store representation version in every prediction record. A resumed training run that silently combines old caches with new encoder weights is invalid.

Retain a small, fixed rehearsal set for language behavior and previous task capabilities only when native weights are adapted. Do not use sealed retention tests as rehearsal. Record update order and test acquisition/retention separately so a model cannot pass preservation merely by learning nothing new.

# Formal model: actions, consequences and goals

## Define the state that the model actually has

Let $o_t$ be an observation and $H_t=(o_0,a_0,o_1,\ldots,o_t)$ the available interaction history. The model receives a belief representation $b_t=E_\eta(H_t,e)$, where $e$ contains environment rules and available tools. This is not necessarily the true Markov state. In partially observed tasks, two identical current observations can imply different consequences because their histories differ.

An action is not merely the string `VERIFY`. Define

$$
a_t=(\text{operator},\text{arguments},\text{budget},\text{executor version}).
$$

The transition record includes the resulting observation, executed status, duration, resource use and independent checks. Let $\phi_t=\Phi(\tau_t)$ encode the consequences of transition $\tau_t=(H_t,a_t,o_{t+1})$. A useful starting decomposition is

$$
\phi_t=[\phi_t^{\mathrm{measured}};z_t^{\mathrm{learned}}].
$$

Measured components anchor the model to checkable events. Learned components preserve the user's proposed latent semantic space. Neither is assumed to be intrinsically good or bad. A failed action, information acquisition, elapsed computation and final task success are different events and should remain distinguishable.

## Three related prediction targets

A one-step model predicts $p_\theta(\phi_t,b_{t+1},d_t\mid H_t,a_t,e)$, where $d_t$ describes termination. A plan-conditioned model predicts consequences for a **specified** candidate action plan $A$:

$$
p_\theta(\phi_{t:t+K-1},b_{t+1:t+K},d_{t:t+K-1}\mid H_t,A,e).
$$

A successor-feature model predicts a discounted accumulation under an explicitly named continuation policy $\pi$:

$$
\psi_K^\pi(b_t,a_t)=\mathbb{E}_{\pi}\left[\sum_{k=0}^{K-1}\gamma^k\phi_{t+k}\mid b_t,a_t\right].
$$

These targets are not interchangeable. The same first action can be excellent under a competent continuation policy and poor under one that repeatedly makes mistakes. Every successor target must therefore carry a policy identity or the actual open-loop plan. A target mixed across behavior policies without that information is a mixture, not an optimal-action value.

For variable-duration actions, either use explicit per-decision discounting or use elapsed-time discounting with cumulative duration. State which convention is being estimated. A long tool call must not quietly receive the same latency cost as a cheap token action. The reference scaffold implements finite-horizon discounted features, not an infinite-horizon policy-learning algorithm.

## Utility is a readout, not the world itself

A tractable first reward family is

$$
r_g(\tau_t)=w_g^\top\phi_t,\qquad Q_g^\pi(b,a)=w_g^\top\psi^\pi(b,a).
$$

This factorization is grounded in successor-feature research. Its advantage is limited but useful: a new weighting of already represented consequences can change values without relearning dynamics. It does not mean that every possible new objective lies in the learned feature span, or that a policy optimized for an old goal remains optimal for a new one. Generalized policy improvement uses a set of supported policies; planning can search other supported candidates. Neither guarantees recovery of unobserved skills. [S02, S03]

A nonlinear utility $u_g(\phi)$ is legitimate, but it changes the requirements. In general,

$$
\mathbb{E}[u_g(\phi)]\ne u_g(\mathbb{E}[\phi]).
$$

A mean embedding that combines “safe success” and “dangerous failure” is not an actual intermediate outcome. Risk constraints, threshold rewards and conjunctions may require joint samples, richer statistics or a distribution over successor features. DiSPO is a particularly relevant precedent for distributions of achievable long-run features; its dataset-supported reachability is an important limitation to retain. [S04]

## Preference-only goals versus goals that alter behavior

When a goal simply says “value speed more” and the proposed plan is fixed, the predicted physical consequences should not change. Only utility should change. This is a valuable invariance test.

When the goal changes an action's actual execution—for example, a `WRITE_REPORT` operator whose prompt is goal-dependent—the consequences can change. Supply the operative goal to the executor and model. Do not enforce goal invariance where it contradicts the environment. Distinguish **preference context**, **task specification** and **control instruction** in the schema.

## What an embedding can and cannot mean

An embedding is a learned coordinate system. Its dimensions do not automatically correspond to correctness, information or reasoning phases. For an invertible matrix $A$, setting $\phi'=A\phi$ and $w'=A^{-\top}w$ preserves the scalar reward. Consequently, many geometries can explain the same observations. Nearest-neighbor plots and isolated cosine values are diagnostics, not proof of a unique semantic decomposition.

Meaningful tests concern preserved information and behavior: can a small readout recover held-out observable properties, can new goals be fitted with fewer labels, do controlled action changes cause the predicted consequence changes, and do those predictions improve action selection? A cluster name is assigned after a blinded audit; it is not a training truth because the cluster looks intuitively coherent.

There is also no preference-identification miracle in unsupervised data. The same transition dataset is compatible with $w$ and $-w$. Unsupervised learning can discover structure; something must ground which outcomes are desirable. A verifier, task success flag, preference comparison, explicit goal or chosen intrinsic objective supplies that grounding. DIAYN and preference-learning work illuminate different sides of this distinction. [S17, S20]

# Relationship to prior art and the defensible contribution

## Four families to distinguish

**Sequence-modeling policies.** Decision Transformer predicts behavior from history and desired return. It is a useful baseline for goal-conditioned action generation, but a desired return is not an observed future outcome. Generalized Decision Transformer broadens the conditioning statistics; it also highlights why hindsight targets and deployment inputs must be separated. [S06, S18]

**Trajectory world models.** Trajectory Transformer gives the closest early precedent for representing states, actions and rewards as token sequences. UniMASK and Masked Trajectory Models broaden the tasks expressible by masking. They motivate the interface, not a conclusion that a causal language model should simply be switched to bidirectional attention without requalification. [S07–S09]

**Task-relevant latent models.** Dreamer and MuZero show different ways of making predictive models useful for decisions. DeepMDP and bisimulation-oriented representation work motivate preserving decision-relevant transition structure rather than reconstructing all raw observations. These approaches also caution against an excessively compressed representation that removes distinctions required by later goals. [S01, S11–S13]

**Reusable future-consequence representations.** Successor features separate consequences from a reward readout. DiSPO models richer future-feature distributions. SF-GEN already brings successor features to controlled language generation, so “successor features for an LLM” cannot be our novelty claim. [S02–S05]

## The research space that remains worth testing

The proposed contribution is a controlled combination: pretrained compact-language representations; executable, parameterized reasoning or tool actions; learned consequence geometry; goal revaluation; long-distance and cross-subtask dependencies; and preservation-aware integration. A combination is not scientifically important merely because its parts can be connected. It becomes interesting when it provides a reproducible benefit that a matched alternative cannot explain.

Candidate publishable findings include a lower reward-label requirement for new goals; improved choice among plans requiring held-out skill compositions; improved accuracy at equal total reasoning cost; or a learned representation that preserves useful consequence distinctions across domains while scalar evaluators fail. An equally useful negative finding would show that clustering adds interpretability but no control value, or that long-context improvements arise entirely from additional context rather than consequence factorization.

The comparison should not pit a full trajectory Transformer against a scalar MLP with only one local state. A scalar model can use attention, long context, rich hidden features and multiple outputs. Match those capabilities before attributing a gain to the representation. This is an experimental design requirement, not a theorem that the richer model must win.

## Closest precedents: adoption decisions

| Reference | Adopt first | Defer or reject as an initial dependency |
|:--|:--|:--|
| Successor features | Explicit reward factorization and policy identity | Claims of arbitrary zero-shot optimality |
| DiSPO | Direct future-feature prediction comparison | Full diffusion outcome generator before a baseline works |
| SF-GEN | Frozen-LM feature/value comparator | Claims that this is the first language application |
| Trajectory and masked models | Typed slots and audited masks | Unrestricted future visibility during planning |
| Dreamer / MBPO | Re-anchoring and model-error gates | Long imagined-RL rollouts at project start |
| CPC / VICReg / VQ | A staged representation ablation | All objectives and codebooks enabled at once |

The current literature register also includes text-world-model and visual predictive-model work through 2026. Those sources refine the roadmap but do not establish a Rineya result. Earlier conversation percentages and venue attributions that were not reverified are not used as evidence in this report. [S36–S39]

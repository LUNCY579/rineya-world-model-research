# Delivery roadmap and ownership

## A gated sequence, not a calendar promise

The following effort blocks are planning estimates. One person can hold several roles, but scientific decisions, data validation and implementation should still receive separate reviews. Native hardware availability, existing repository interfaces and the pilot's statistical requirements determine actual duration.

**Block 1: approximately 3–5 focused work sessions.** Research owner freezes the hypothesis, subject identity, environment, primary metric and scope. Engineering owner qualifies schemas, split logic, masks and snapshots. Reviewer checks that future information cannot leak. Deliverable: G0 report and a tiny end-to-end synthetic run.

**Block 2: approximately 1–2 working weeks.** Implement the measured-feature and scalar baselines, collect the first bounded parent set, establish candidate-oracle performance and measure laptop capture. Deliverable: G1 report, failure taxonomy, measured resource envelope and finalized statistical contract. No self-supervised codebook is required yet.

**Block 3: approximately 1–2 working weeks.** Add the learned continuous outcome encoder/predictor and sparse goal readout. Run B2/B3/B4 with matched data and budgets. Deliverable: G2 result and a decision to promote, repair or prefer the simpler evaluator.

**Block 4: approximately 1–2 working weeks.** Test nested label budgets, new goals and controlled clustering. Add direct successor targets only if needed by the temporal failure analysis. Deliverable: G3 transfer and discovery report with negative controls.

**Block 5: approximately 1–2 working weeks.** Expand context and dependency horizon, then introduce synchronous DAG tasks. Deliverable: G4 report including causal visibility, joint outcomes, scheduling and model-exploitation stress tests.

**Block 6: approximately 1–3 working weeks, conditional.** Port the successful interface to native typed-slot adapters, run acquisition/retention and hardware gates, then evaluate a distilled policy or short imagined-RL extension. Deliverable: G5/G6 qualification or an explicit reason to retain the sidecar. Visual modeling remains a later independent campaign.

These blocks are not a commitment to continuous background work. The priority is to complete a decisive comparison before expanding the architecture. The roadmap and experiment registry include explicit dependencies to prevent every idea from becoming simultaneous active work.

## First implementation handoff

The first agent or engineer should read `START_HERE.md`, the architecture and data contracts, then run the reference tests. Next, inspect the current repository in read-only mode, identify the actual checkpoint loader and capture interface, and prepare an integration diff plan. Stop for clarification when the subject identity, sealed-data policy, allowed paths or native budget is unresolved.

The first substantive pull request should contain contracts, a simulator adapter, split checks, mask tests and an end-to-end measured-feature baseline—not native training, recurrence, codebooks and imagined RL all at once. A separate pull request should add frozen Rineya capture after numerical qualification. Each should include its exact tests, resource observations and limitations.

## Decision log at every gate

Record what was expected, what was observed, which comparison isolates the mechanism, what is uncertain and what next action follows. Maintain rejected and deferred alternatives. Do not rewrite the original hypothesis after seeing evaluation results without labeling the new experiment. Useful stopping decisions are part of the research output.

# Risk register and diagnostic responses

| Risk | Detect it | Response |
|:--|:--|:--|
| Future-information leakage | Causal perturbation and provenance tests | Block training/evaluation; rebuild inputs |
| Representation collapse | Variance, rank and decoder tests | Fix target/regularization before scaling |
| Hidden-state aliasing | Same observation, different history branches | Improve belief/history representation |
| Proxy exploitation | Predicted gain rises, real gain falls with search | Reduce search, add real anchors, revisit support |
| Goal entanglement | Preference-only goal changes physical prediction | Separate task instruction from utility weights |
| Rare failure merged away | Cluster audit by failure type | Increase resolution or retain explicit hazard features |
| Invalid action semantics | Same token, inconsistent executor behavior | Version and parameterize actions |
| Policy-mixed successor target | Different continuations share one target | Condition on policy/plan or label the mixture |
| Frozen-weight but worse system | End-to-end retention degrades | Restrict routing or revert controller |
| Apparent efficiency from hidden cost | Full cost ledger changes ranking | Recompare under total budget |
| Goal transfer outside feature span | New-property readout fails | Add grounded features or collect targeted data |
| Native numerical mismatch | Prefix/bulk or resume parity failure | Stop native extraction; repair mechanics |

The highest-risk scientific shortcut is to call any useful latent reward predictor a novel world model. Keep the claim tied to the tested outcome variables and interventions. The highest-risk engineering shortcut is to accept a perfect validation loss without proving that the evaluator saw only deployment-available information.

# Future extensions and research questions

## Distributional consequences and reachable plans

After the deterministic linear-utility baseline succeeds, add stochastic outcomes and nonlinear preferences. Compare direct successor distributions with recursive latent rollouts and shared-scenario sampling. Evaluate joint failure, tail risk and reachability. Do not select an attractive predicted consequence unless a supported action or policy can plausibly realize it. DiSPO supplies an especially relevant conceptual reference for coupling future features to reachable behavior. [S04]

## Learned action vocabularies

Manual operators provide a stable initial intervention vocabulary. A later model can discover reusable macro-actions from trajectories, with each code decoded into an executable, bounded option. Test whether learned options transfer across tasks and retain meaningful preconditions. Interpretability and compression are secondary to reliable execution. An action code that cannot be executed consistently is not a useful control primitive.

## Internal compute as an environment

Once native recurrence is qualified, model the consequences of additional computation, checking, retrieval or stopping. Treat hidden-state prediction as an internal computation model, and evaluate whether it improves final correctness per actual latency. Compare recurrence against ordinary extra generation and tool use. This is where TAPS/CDE can become a downstream application rather than a duplicated controller project.

## Visual version

A later visual Rineya variant can attach a frozen vision encoder and learn action-conditioned future visual features. Do not start by generating pixels or reproducing internet-scale video pretraining. V-JEPA 2, DINO-WM and V-JEPA 2.1 offer different predictive-representation inspirations; the relevant experiment is whether a compact action-conditioned model preserves task-critical visual variables at the available budget. [S36, S37, S39]

Separate visual perception failures from dynamics errors by comparing ground-truth state input, frozen visual features and learned visual adaptation. Test unseen appearances with fixed dynamics and unseen dynamics with similar appearances. A text-world success is not evidence of physical-world competence.

## Cumulative skill acquisition

Investigate whether a consequence vocabulary supports adding a new goal readout without changing old behavior, and whether adding genuinely new transition knowledge can be localized in an adapter. Evaluate composition of old and new operators in one plan. This links the project to Rineya's broader aim of accumulating capabilities, but the world model does not automatically solve catastrophic forgetting.

A particularly informative future study changes only the goal, then only the dynamics, then only the observation rendering. These shifts test three different forms of reuse. Calling all three “generalization” conceals which part of the architecture actually transfers.

# How to use the delivered resources

Read the executive decision and formal model first. Use the annotated bibliography to understand the closest precedents and the limits of each analogy. The `specs` directory is the implementation reference; `configs`, `schemas` and `templates` turn the report into explicit contracts. The `roadmap` directory records dependencies and future gates. The `reference` package runs without Rineya weights and demonstrates selected mechanics. Its own README states what it does not implement.

Validation outputs are included separately from proposed experiment results. The CPU demonstration is not a benchmark result and does not justify a claim that learned consequence semantics, clustering, long-context planning or skill retention have been solved. It is a reproducible starting point for detecting mistakes before native experiments become expensive.

The program's strongest claim, should later evidence support it, is precise: **under a frozen data, compute and preservation budget, a learned consequence representation improved held-out goal-conditioned action selection or reduced the required grounding labels compared with a matched scalar evaluator.** That is worth investigating. It is also specific enough to fail, be refined and ultimately produce useful knowledge.

# Glossary and notation

| Term | Meaning in this bundle |
|:--|:--|
| $H_t$, history | Observations and actions available at decision time |
| $b_t$, belief | Encoded history; not necessarily true environment state |
| $a_t$, action | Executable operator plus arguments, budget and version |
| $\phi_t$, consequence | Measured or learned description of a transition |
| $z_t$, latent | Learned consequence coordinates without assumed named axes |
| $g$, goal | Task/preference specification; roles recorded separately |
| $w_g$, utility weights | Readout for an explicitly represented feature family |
| $\psi^\pi$, successor features | Expected discounted consequences under policy $\pi$ |
| $A$, candidate plan | Precommitted actions or an explicit conditional policy tree |
| Counterfactual family | Branches, goals and variants from the same underlying parent |
| Grounding | Independent task feedback connecting representations to utility |
| In-span revaluation | Changing utility over already represented consequences |
| Native qualification | Tests using the exact selected Rineya artifact and runtime |
| Promotion | Meeting predeclared scientific, preservation and cost gates |

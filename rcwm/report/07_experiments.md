# Experimental program and promotion gates

## Hypotheses that can be falsified

**H1, useful representation:** learned consequence features improve independently measured candidate selection over a matched scalar Transformer. Falsify the claimed representation advantage when the effect disappears after matching context, parameters, labels or tuning budget.

**H2, transferable utility:** a frozen consequence representation supports new in-span goals with fewer grounding labels than fitting a new scalar evaluator. Falsify the transfer claim when it depends on evaluation-parent leakage, a task-specific encoder update or reward properties not actually represented.

**H3, temporal composition:** trajectory modeling helps when useful outcomes depend on multiple separated actions. Falsify the long-range mechanism when gains remain unchanged under removal of dependencies, or are explained by a larger context window alone.

**H4, discovered structure:** clustering or learned discrete codes improve label efficiency, compositional transfer or interpretability without harming rare outcomes. Falsify the benefit when continuous or random-projection controls perform equally well, code usage collapses, or clusters reflect surface task IDs rather than consequences.

**H5, deployable integration:** the native typed-slot route or a sidecar improves actual task performance within the target cost and preservation envelope. Falsify deployment readiness when the gain vanishes after counting planning overhead, fails under the real executor, or harms protected behavior beyond the preregistered margin.

## Minimal model matrix

| ID | Model / representation | Question isolated |
|:--|:--|:--|
| B0 | Legal random or simple heuristic choice | Environment difficulty and trivial strategies |
| B1 | Frozen Rineya direct action/answer | Existing deployed baseline |
| B2 | Scalar trajectory Transformer | Strong non-factorized evaluator |
| B3 | Measured consequence features + goal readout | Benefit of explicit factorization |
| B4 | Learned continuous consequences + readout | Added value of learned geometry |
| B5 | B4 with train-only clustering or VQ | Added value of discrete structure |
| B6 | Policy-conditioned successor features | Direct future accumulation versus rollout |
| B7 | Native Rineya typed-slot adapter | Backbone-sharing benefit and retention cost |
| O1 | Exact simulator evaluator on same candidates | Candidate-set ceiling; not deployable baseline |

Do not run all combinations immediately. B0, B2, B3 and O1 validate the environment and contracts. B1 enters when native capture and action execution are qualified. B4 is the central scientific arm. B5, B6 and B7 are conditional extensions. A scalar baseline must have a comparable representation-learning budget; otherwise an apparent factorization gain may come from additional pretraining.

## Staged gates

**G0: provenance and mechanics.** Pass schema, split, snapshot, mask, finite-loss and deterministic replay checks. Pin source identity and demonstrate capture agreement within a declared tolerance. Required output: a qualification report with no native-quality claim.

**G1: environment and instrument.** The oracle solves the intended class; a tiny diagnostic set can be fitted; a measured-feature model learns nontrivial action dependence. Required output: failures stratified by observation ambiguity, label validity and optimization. A broken instrument blocks later architecture comparisons.

**G2: learned-representation utility.** Compare B2/B3/B4 on paired held-out parents. Primary endpoint: actual utility or task success under a fixed candidate set. Secondary endpoints: consequence fidelity, label efficiency and runtime. Required output: effect estimate, uncertainty interval and resource accounting.

**G3: reuse and discovery.** Run nested grounding budgets and held-out goals. Add clustering only after the continuous result. Required output: learning curves with identical labeled subsets and a rare-failure audit. A pretty latent plot is insufficient.

**G4: composition and horizon.** Increase dependency distance, horizon and DAG complexity separately. Required output: free-running prediction and actual-control results, including model-exploitation and support stress tests.

**G5: native adaptation and preservation.** Add typed slots/adapters behind a mode switch. Required output: checkpoint invariance where promised, acquired world-model skill, old-behavior noninferiority, measured laptop memory and system latency.

**G6: optional policy improvement.** Distill a qualified planner or add short imagined RL. Required output: real-environment improvement beyond the non-RL controller at matched data and compute. Simulation-only gains are not promotion evidence.

## Sample size and statistical unit

The primary independent unit is the **parent problem or environment family**, not every correlated branch, goal relabel or token. Aggregate paired outcomes within parents, then use a parent-level paired bootstrap or an appropriate clustered model. Preserve gained/lost cases for binary results. Training-seed variability is a separate uncertainty source; report seed-level results and a hierarchical summary where justified. The statistical-precedence literature motivates avoiding confident aggregate claims from fragile run summaries. [S34]

Use a pilot to estimate variance and choose a final sample size before sealed evaluation. As an illustrative planning calculation, if paired parent-level utility differences have standard deviation $\sigma$, a rough normal approximation for detecting a mean difference $\delta$ at 5% two-sided significance and 80% power is

$$
n\approx\left(\frac{(1.96+0.84)\sigma}{\delta}\right)^2.
$$

This is a derived approximation, not a substitute for clustered, binary or heavy-tailed simulation-based power analysis. With $\sigma=0.25$ and $\delta=0.03$, it suggests roughly 545 independent parents. More correlated branches do not create 545 independent parents.

Predeclare one primary comparison and a small set of secondary questions. Correct or clearly label exploratory multiplicity. Freeze model selection on development data. Sealed evaluation is not a hyperparameter search tool.

## Practical effect, preservation and cost

A suggested contract requires a positive primary paired confidence interval, a preregistered minimum meaningful effect, and no unacceptable cost or retention violation. The actual effect threshold must be chosen using the task scale and pilot variance. Configurations deliberately leave final promotion thresholds unset until that decision; a null threshold means **blocked**, not “no threshold required.”

For preservation, select a noninferiority margin on protected capabilities. Failure to reject a difference is not evidence of no harm. Require the lower bound on new-minus-base performance to exceed the allowed loss margin, while separately showing acquisition. A frozen base weight hash is necessary for a strict frozen route but does not certify behavior with the new controller. [I02]

Report peak allocated and reserved GPU memory, process-tree RAM, disk bytes, model and adapter parameters, encoding time, candidate generation, scoring, tool execution, total latency and failure retries. Compare quality at equal wall-clock or token budgets as well as raw accuracy. Never hide a larger candidate search behind “same model size.”

## Decision outcomes

Promote a mechanism when it clears the declared accuracy, uncertainty, preservation and cost gates. Refine it when the failure is a diagnosed contract or coverage issue. Pivot to the simpler baseline when the extra mechanism provides no useful benefit at the agreed budget. Preserve a dormant extension when prerequisites are unmet; do not describe it as empirically disproved when it was never qualified.

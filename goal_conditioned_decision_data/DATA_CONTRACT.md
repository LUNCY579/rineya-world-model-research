# GCDL-001 canonical data contract v1.0

This is an experimental synthetic decision-data contract. Oracle outputs, uncertainty labels, candidate rankings and finite-search benefits are mathematical targets; none implies that a model has learned general decision intelligence. The executable primary oracle is rechecked by a separately implemented checker before acceptance.

Every JSONL line is one canonical record. JSON uses sorted keys, finite integers and exact rational objects (`{"numerator": n, "denominator": d}` with positive denominator), never fabricated floating-point confidence. Top-level keys are `schema_version`, `id`, `family`, `difficulty`, `split`, `provenance`, `input`, `targets`, `metadata`; the acceptance pipeline appends `verification` and `quality_gate`.

`schema_version` is `gcdl-1.0`. `id` is a deterministic SHA-256-derived identifier computed by `generators.build.record_id`; audit statuses and runtime timing do not affect identity. `provenance` records the seed, generator/environment/config versions, configuration hash, record index, reserved template, parent world and protected lineage. All goals, counterfactuals and perturbations from one parent bundle remain in one split. Canonical IDs are immutable after acceptance; a changed task or target produces a new ID.

## Inference input

`input` contains only information available to the eventual decision model:

- `environment`: `ledger`, `blocksworld`, or `register`.
- `initial_state`, `rules`, `goal`, `constraints`, and an explicit `utility_definition`.
- `candidates`: a list of `{id, plan}`; each plan is an ordered list of executable actions. Empty plans are allowed. Candidate identifiers are opaque seed/parent-specific random 96-bit strings; semantic generator roles exist only in metadata. Actions and governing rules remain visible because the task requires understanding them.
- Optional instruction paraphrases, irrelevant context, and `search_request` declaring the immediate/expanded candidate budgets.
- For partial information, a recursively observed `initial_state` with unknown leaves represented by null, complete `possible_worlds`, `information` equal to `missing` or `ambiguous`, and `probability_assumption="uniform_over_possible_worlds"`. Supplied worlds form an explicit finite universe consistent with every observed leaf. No undisclosed true world is exposed.

Future states, rankings, rewards, success probabilities, search improvements and optimal choices occur exclusively under targets. Optional natural-language explanation is not ground truth and is currently omitted.

## Authoritative outcome and utility semantics

`targets.oracle` is exactly the primary oracle's returned value. For each candidate and each world, it stores legal execution, the first illegal step, attempted transitions, intermediate successful states, terminal state, successful-action cost, exact goal atoms satisfied, goal completion, and initial/intermediate constraint violations. Illegal plans stop at the first illegal action; failed actions consume no task cost. Step indices begin at one; initial constraint checks have index zero.

A candidate's per-world utility is lexicographic `(feasible, goal_atoms_satisfied, -cost)`. Feasible means every action is legal and no initial or intermediate state violates a constraint. Numerical goals are exact equalities; there is no undocumented distance reward. Aggregate utility is the component-wise uniform expectation over the explicitly declared finite worlds. Comparison uses exact fractions in lexicographic order. Feasibility has priority over goal atoms, which have priority over task cost; this fixed ordering is stated in input.

`success_probability` is the exact fraction of declared worlds where the plan is legal, all constraints hold and every goal atom is satisfied. `feasible_probability` is the exact fraction with legal execution and no constraint violation, regardless of goal completion. In a fully observed deterministic task these are binary facts. Finite-world probabilities are conditional on the explicitly stated uniform assumption; they are not calibrated estimates for an unknown real distribution.

Ledger states hold account balances and signed ledger entries. Settle applies the entry; void explicitly either reverses or retains it under the supplied rule. Transfer moves a positive amount between accounts. Blocksworld states are acyclic support forests; moves require clear source/destination, obey height limits, and reject same-destination moves. Register rules declare integer bounds, overflow behavior, enabled operations and task costs. Full legality and transition definitions live in the executable environment/oracle code.

## Decision labels and abstention

`targets.decision.candidates` projects each ID, exact utility, success probability and feasible probability. `ranking` orders distinct utility levels from rank one and sorts tied IDs deterministically. `best_candidate_ids` contains every top-ranked ID unless all candidates have zero feasibility probability; in that case it is empty and abstention is required. Ranking infeasible candidates does not recommend one of them.

`uncertainty.label` uses this priority: `no_feasible_candidate`, `impossible_in_candidate_set`, `tied_best`, `partially_observed`, then `determined`. `uncertainty.abstention` is respectively `no_feasible_candidate`, `no_candidate_can_succeed`, or `not_required`. Ties do not fabricate confidence. A finite-world task can be both uncertain and impossible; the more actionable impossibility label takes priority while `oracle.information` preserves partial-observation status.

`optimality` is `{"scope":"supplied_candidates","proof":"exhaustive_candidate_execution"}`. This proves comparison only within the supplied alternatives. It is not global unbounded planning optimality.

## Measured computation and distillation

`targets.search` stores matched cheap and expanded runs over the **same candidate universe**, with prefix candidate budgets. The immediate run evaluates one supplied plan; the expanded run evaluates all supplied plans. Each is an actual oracle execution. Costs count candidate executions and attempted transition nodes over all worlds, separately. `realized_utility_delta` is exact expanded-minus-cheap utility, and additional costs are explicit. These realized gains are oracle-only targets, never decision-time estimates. Timing is nondeterministic operational evidence stored separately by the pipeline, not canonical labels.

Candidate order is preserved in these search examples because prefix budgets are order-sensitive. The controlled option-position experiment rotates only ordinary decision examples, where reordering leaves candidate availability and utility unchanged.

The omitted-proposer case deliberately omits the settle action from supplied candidates. `targets.bounded_proof` exhaustively executes all sequences of a declared five-action alphabet through horizon two, including the empty plan: 31 plans. It records a successful witness if one exists and the best utility. Proof scope is exactly that finite alphabet/horizon; arbitrary transfer amounts, longer plans and unbounded optimality are excluded.

## Family and split design

A uses identical states/rules/candidates under multiple goals. B contrasts an immediate credit with a later reversal and includes a one-cost close call. C proposes genuine random initial/goal Blocksworld forests and executable multi-step candidate paths. D records entity and status renaming, option/field order, paraphrase/distractor controls, governing-rule reversal, signed numerical reversal, and explicit OOD stress evidence. E covers finite-world missing/ambiguous outcomes, exact probabilities, ties, impossible candidate sets, goal/constraint conflict and all-infeasible abstention. F measures matched candidate deliberation. G supplies direct success, search repair/failure and bounded omitted-solution evidence. H supplies simultaneous goals, familiar individual constraint primitives, a withheld conjunction, and aligned ledger/register probes. H is an evaluation design, not a demonstrated learned-transfer result.

`configs/curriculum.json` reserves semantic structures before sampling seeds:

| Split | Ledger accounts/entries | Block count | Entry magnitude | Base constraint structure |
|---|---:|---:|---:|---|
| train | 2/1 | 3 | 2–19 | none; individual primitive support |
| dev | 3/2 | 4 | 20–39 | minimum balance |
| sealed | 3/3 | 5 | 40–59 | cost cap |
| ood | 4/4 | 6 | 1000–1200 | minimum balance + cost cap + forbidden status |

The simultaneous three-constraint conjunction is excluded from training support. Goal-conflict controls and surface variants remain within their parent bundle. A protected signature derives actual environment, state sizes and rule/constraint keys; adding a split prefix to a template name is insufficient protection. Seeds generate diversity within reserved templates, never assign train/test by themselves. Semantic fingerprints remove entity/status spelling, option IDs/order and prose, and detect normalized task duplicates. This normalization is bounded and environment-specific, not a universal semantic equivalence solver.

The immutable sealed pool is frozen by the pipeline before development optimization. Generator code does not read accepted or sealed datasets. Access isolation between local agents is procedural because all agents share a host; this limitation must remain disclosed. Development experiments use only explicit development seeds and compare option-position counts plus unchanged exact oracle labels. They cannot demonstrate model performance or eliminate every shortcut.

## Training views and acceptance

`decision_labels.views.training_views(record)` yields dictionaries of pointwise, pairwise and listwise examples, each with separate `input` and `target`. Pointwise inputs contain one candidate; pairwise inputs contain two; listwise inputs retain the supplied set. Pairwise preference is +1/0/-1 for left better/tied/worse under exact lexicographic utility. Metadata, oracle trajectories and verification evidence are never copied into input. Explicitly rejected or quarantined records cannot become views. The export loader additionally requires `quality_gate.status="accepted"`; unreviewed candidate views are permitted solely for audit/schema checks.

Every accepted record must pass schema, exact primary/independent outcome agreement, independently reconstructed decision/search/proof labels, deterministic reproduction, split/duplicate integrity and leakage checks. Any discrepancy is quarantined. Accepted evidence and shard hashes are immutable. The pilot remains experimental until all release gates pass; synthetic coverage does not establish real-world decision generalization.

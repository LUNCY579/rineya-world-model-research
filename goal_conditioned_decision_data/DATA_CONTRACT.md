# GCDL-001 canonical data contract v1.0

This is an experimental synthetic decision-data contract. Oracle outputs, uncertainty labels, candidate rankings and finite-search benefits are mathematical targets; none implies that a model has learned general decision intelligence. The executable primary oracle is rechecked by a separately implemented checker before acceptance.

Every JSONL line is one canonical record. JSON uses sorted keys, finite integers and exact rational objects (`{"numerator": n, "denominator": d}` with positive denominator), never fabricated floating-point confidence. Top-level keys are `schema_version`, `id`, `family`, `difficulty`, `split`, `provenance`, `input`, `targets`, `metadata`; the acceptance pipeline appends `verification` and `quality_gate`.

`schema_version` is `gcdl-1.0`. `id` is a deterministic SHA-256-derived identifier computed by `generators.build.record_id`; audit statuses and runtime timing do not affect identity. `provenance` records the seed, generator/environment/config versions, configuration hash, record index, reserved template, parent world and protected lineage. All goals, counterfactuals and perturbations from one parent bundle remain in one split. Canonical IDs are immutable after acceptance; a changed task or target produces a new ID.

## Versioned generation profiles

`configs/generation_profiles.json` allowlists exact archived identities. `generators.profiles.resolve_record_profile` resolves the tuple `(config_version, config_hash, generator_version, environment_version)` to a trusted curriculum and checks both its canonical JSON hash and file-byte SHA-256. Unknown/mixed tuples, altered archive/registry bytes and symlinks fail closed; record text cannot supply a file path or import. Schema/template/reservation validation uses this resolved profile rather than the currently active default.

| Profile | Generator | Curriculum | Schema / environment |
|---|---|---|---|
| v1 | `generator-1.0` | `curriculum-1.0`, exact byte copy in `configs/curriculum-v1.json` | `gcdl-1.0` / `environments-1.0` |
| v2 | `generator-2.0` | `curriculum-2.0`, `configs/curriculum-v2.json` | `gcdl-1.0` / `environments-1.0` |

V1 generator/curriculum, transition/oracle semantics, rational targets, record-ID algorithm, templates and original normalizer remain unchanged. All 1,936 nonsealed accepted v1 rows passed archived-profile validation with exact original IDs and protected fingerprints; archived canonical/shard/manifest bytes remain immutable. Profile-derived size bounds run before factorial normalization. V2 first authenticates and reverses its explicit namespace maps, including invalid references, then invokes the unchanged normalizer. Fingerprints have no generation-version prefix, so equivalent cross-version tasks cannot evade duplicate checks by changing versions. See [compatibility evidence](reports/PROFILE_COMPATIBILITY_DESIGN.md) and [the independent review](reports/INDEPENDENT_V2_REVIEW.md).

Jobs store their selected profile at enqueue time. Legacy jobs/manifests resolve as v1 and cannot be reinterpreted when the active default changes. New `gcdl-manifest-2` batch receipts require explicit `generation_profile` and `generation_profile_evidence`, including canonical and file-byte curriculum commitments; removing those commitments is not a valid downgrade. An existing immutable batch cannot replay under a different profile.

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

A uses identical states/rules/candidates under multiple goals. B contrasts an immediate credit with a later reversal and includes a one-cost close call. V1 C proposes genuine random initial/goal Blocksworld forests and executable multi-step candidate paths. V2 C supplies the same six ordered candidates under three full, distinct, non-initial terminal goals with disjoint nonempty exact best sets; every goal has a feasible successful best candidate. Goal/index/targets do not enter the shared bundle identity, which hashes environment, initial state, rules, constraints, ordered candidates and utility definition. D records entity and status renaming, option/field order, paraphrase/distractor controls, governing-rule reversal, signed numerical reversal, and explicit OOD stress evidence. E covers finite-world missing/ambiguous outcomes, exact probabilities, ties, impossible candidate sets, goal/constraint conflict and all-infeasible abstention. F measures matched candidate deliberation. G supplies direct success, search repair/failure and bounded omitted-solution evidence. H supplies simultaneous goals, familiar individual constraint primitives, a withheld conjunction, and aligned ledger/register probes. H is an evaluation design, not a demonstrated learned-transfer result.

The unchanged v1 `configs/curriculum.json`/archive and the v2 archived curriculum reserve the same semantic structures before sampling seeds; v2 has distinct protected template/lineage identities:

| Split | Ledger accounts/entries | Block count | Entry magnitude | Base constraint structure |
|---|---:|---:|---:|---|
| train | 2/1 | 3 | 2–19 | none; individual primitive support |
| dev | 3/2 | 4 | 20–39 | minimum balance |
| sealed | 3/3 | 5 | 40–59 | cost cap |
| ood | 4/4 | 6 | 1000–1200 | minimum balance + cost cap + forbidden status |

The simultaneous three-constraint conjunction is excluded from training support. Goal-conflict controls and surface variants remain within their parent bundle. A protected signature derives actual environment, state sizes and rule/constraint keys; adding a split prefix to a template name is insufficient protection. Seeds generate diversity within reserved templates, never assign train/test by themselves. Semantic fingerprints remove entity/status spelling, option IDs/order and prose, and detect normalized task duplicates. This normalization is bounded and environment-specific, not a universal semantic equivalence solver.

V2 declares 384 globally distinct eight-lowercase-letter strings: 16 per namespace/split for accounts, entries, blocks, registers, statuses and table. Reserved TRAIN/DEV/OOD/SEALED pools are disjoint complete strings with the same grammar. V2 vocabulary metadata authenticates total injective reference maps, declared split/stratum/presentation, deterministic parent-based assignments, pool membership and exact source-task reconstruction; structural keys and opcodes are retained. Explicit DEV `familiar_control` examples use TRAIN strings, remain paired with `reserved_dev` controls, and do not count as heldout vocabulary. Exact-string reservation does not establish unseen tokenizer pieces, natural-language meaning or learned lexical transfer. Immutable v1 keeps its original shared vocabulary.

The immutable sealed pool is frozen by the pipeline before development optimization. Generator code does not read accepted or sealed datasets. Access isolation between local agents is procedural because all agents share a host; this limitation must remain disclosed. Development experiments use only explicit development seeds. Surface/vocabulary controls require unchanged exact oracle labels; goal-only triplets measure exact goal-conditioned labels and fixed-selector counts over the same candidates. They cannot demonstrate model performance or eliminate every shortcut.

## Training views and acceptance

`decision_labels.views.training_views(record)` yields dictionaries of pointwise, pairwise and listwise examples, each with separate `input` and `target`. Pointwise inputs contain one candidate; pairwise inputs contain two; listwise inputs retain the supplied set. Pairwise preference is +1/0/-1 for left better/tied/worse under exact lexicographic utility. Metadata, oracle trajectories and verification evidence are never copied into input. Explicitly rejected or quarantined records cannot become views. The export loader additionally requires `quality_gate.status="accepted"`; unreviewed candidate views are permitted solely for audit/schema checks.

Every accepted record must pass schema, exact primary/independent outcome agreement, independently reconstructed decision/search/proof labels, deterministic reproduction, split/duplicate integrity and leakage checks. Any discrepancy is quarantined. Accepted evidence and shard hashes are immutable. The pilot remains experimental until all release gates pass; synthetic coverage does not establish real-world decision generalization.

V2 admission is atomic over the entire generated batch. After authoritative per-record audits, `validate_v2_group(..., require_full_curriculum=True)` requires the archived A–H minimums, one resolved v2 profile, complete three-member C bundles with indices 0/1/2, identical non-goal inputs/ordered candidates and protected custody, disjoint nonempty best sets and a feasible successful best candidate for each goal. Matched vocabulary pairs must retain both strata, the same restored task and unchanged decision labels. Any record or group failure quarantines every sibling; no incomplete triplet/pair can enter the accepted shard. Legacy v1 retains its established partial accepted-subset policy.

The reviewed 42-file source/configuration digest is `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`; both complete suites passed 189 tests. Active-v2 approval additionally binds the exact hashes of [V2_EXPERIMENT_FREEZE.json](manifests/V2_EXPERIMENT_FREEZE.json) and [V2_CONTROLLED_EXPERIMENT.json](reports/V2_CONTROLLED_EXPERIMENT.json) named in [INDEPENDENT_V2_REVIEW.md](reports/INDEPENDENT_V2_REVIEW.md). The approval guard requires matching current source/configuration, review and evidence commitments; receipt-only rehashing/removal cannot enlarge that frozen approval. Existing sealed bytes remain the original freeze.

At this documentation update, all 2,152 canonical accepted records are v1, the 160 controlled v2 experiment tasks exist only as in-memory measurement cases, and zero v2 canonical or rendered datasets are published. The active v2 selector applies to new jobs; production publication remains paused until the normal review/freeze gates and the minimum 10 GiB free-disk resource gate pass. A successful controlled experiment does not relax these gates.

## Versioned rendered training exports

`decision_labels.rendering.render_training_views` derives five accepted-only views: pointwise, pairwise, listwise, computation and distillation. Only TRAIN, DEV and OOD may be rendered. The model sees `input_text`; `target` is supervision. All other top-level fields are provenance, identity, split or verification metadata and must be excluded from model input. Fixed framing carries the requested view; JSON map order, whitespace and ordinary candidate order vary deterministically without using oracle labels. Two executable oracles re-evaluate each parsed prompt and check its original candidate-subset outcomes. Pairwise orientation, preference sign and projected utilities follow the presented candidate order. Search records retain their candidate-prefix order in every rendered view.

Computation targets contain the decision and exact matched prefix-search outcomes/costs. Distillation targets preserve decision plus available search and/or bounded-proof evidence. Realized benefits and proof witnesses remain exclusively in targets. Finite exhaustive proof scope is not global unbounded optimality. A positive accepted batch with no eligible auxiliary records receives an explicit zero-row view manifest, carrying source provenance without invented supervision.

The programme authenticates source shards against immutable canonical batch receipts, including exact bytes, row count, accepted status, code hashes, seed, semantic fingerprint and protected world/template indices. Mixed acceptance/quarantine batches export only their accepted shard. Each dataset manifest records source commitments, selected IDs, executable hashes, rendering version and every bounded derived shard hash. Immutable per-batch receipts define a completed resumable export. Training consumers must select a committed dataset manifest/receipt and pass only its `input_text` and `target` fields; they must not discover incomplete artifacts by indiscriminate file globbing.

V1 keeps `gcdl-rendering-1.0` identities and exact existing presentation commitments. V2 uses `gcdl-rendering-2.0`, with source profile custody recorded by the rendered export receipt. Each C goal triplet uses its authenticated goal-independent bundle identity for both candidate permutation and seeded nested map order. At the same view, candidate subset and variant, the three parsed/model-visible surfaces differ only in the goal; goal index, goals, record IDs and targets do not seed surface order. Distinct canonical/view IDs still identify each goal member. Search-prefix order is preserved for both versions.

Three variants of one source world are three rendered rows, not three independent tasks. Changing an approved executable snapshot creates a new derived export specification. Preserve source IDs and split lineage for deduplication and evaluation; do not mix variants across protected splits. The [original controlled comparison](reports/CONTROLLED_SURFACE_COMPARISON.md) documents the unchanged v1 shared-vocabulary and goal-blind C limitations. The frozen v2 experiment measures complete-string custody and the one-third bound for fixed single-choice goal-blind selectors within balanced triplets; it does not measure a trained model, all-candidate selectors, goal-aware terminal shortcuts, universal semantic equivalence or real-world generalization.

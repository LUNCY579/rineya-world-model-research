# Generator v2 implementation boundary

This proposal implements the preregistered construction in `NEXT_GENERATOR_VERSION_DESIGN.md`. It is an in-memory, CPU-only generator change. It does not publish accepted data, train a model, use a paid service, or read sealed gold. The production resource pause remains in force. The official 420000–420031 and 421000–421031 pilot seeds were not used during implementation or fixture debugging; the root research runner must freeze its complete specification and source snapshot before measuring them.

## Version and configuration custody

The new module is `generators/version2.py`; its identities are generator `generator-2.0`, curriculum `curriculum-2.0`, wire schema `gcdl-1.0` and environment `environments-1.0`. `configs/curriculum-v2.json` retains v1 environment sizes, magnitudes, constraint reservations and exact utility semantics. Its template/lineage identities have a v2 suffix. Its candidate policy is `parent_hash`, so the unchanged v1 builder's gold-dependent `balanced` rotation is never called by the adapter.

The v2 configuration was declared before generating any v2 task:

| Identity | SHA-256 |
|---|---|
| Canonical curriculum JSON | `1fa733236552daae035fc844be2b09381005d6343aa732483e19185f61ec92d6` |
| Configuration file bytes | `90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd` |
| Canonical lexical-pool JSON | `5913a57e252ea84e249fdede5fd6ba3dc89b3dd1b2a025c2c26edf6322bf8024` |

`curriculum(config=None)` reads this profile or takes a deep copy of an explicitly supplied exact v2 profile. A changed hash, identity or budget fails; it does not interpret an operational budget object as a new curriculum. The independently owned profile resolver allowlists the exact archived configuration/version tuple for schema validation and generation replay. It retains v1 as the default operator profile. V1 source bytes and IDs are not rewritten.

## Public APIs

| API | Return and boundary |
|---|---|
| `generate_group(seed, config=None, split="train")` | Complete v2 A-H record list; inadequate C raises `GenerationError` with explicit diagnostics |
| `generate_goal_triplet(seed, split="dev", config=None)` | `(records, diagnostics)`; three records for a valid parent, or an empty list plus the bounded construction exclusion |
| `generate_vocabulary_pair(seed, split="dev", config=None)` | `(records, diagnostics)`; DEV familiar-control and reserved-DEV ledger records |
| `bundle_identity(task)` | Hash of exactly environment, initial_state, rules, constraints, ordered candidates and utility_definition |
| `rename_task(task, maps)` | Deep-copied task with explicit namespace references renamed; no structural/opcode replacement |
| `restore_source_names(record, config=None)` | Validated inverse transformation returning the exact original task dictionary |
| `validate_vocabulary(record, config=None)` | Enforces vocabulary custody, total mappings, pool membership and exact source round trip |

These functions return candidate records, not accepted records. Acceptance, release review and any later publication remain separate. Helpers do not read accepted shards, sealed shards, model results or external services.

## C: three full goals over one fixed candidate bundle

The adapter retains the v1 core's deterministic initial Blocksworld forest and unchanged rule/cost schedule for that seed and split. DEV has four blocks and horizon four. It proposes at most 24 legal paths of the declared horizon, using a separate seeded RNG and an explicitly enumerated legal action order. No-op actions are excluded from proposals. Legality is checked by the existing transition implementation, and complete candidate outcomes are independently rechecked by both oracles.

The first three proposed paths with pairwise distinct, non-initial terminal forests are retained. Endpoints are selected by state equality only, never by a favorable candidate label. They become P0, P1 and P2. The six candidates are P0, P1, P2, P0 without its last move, P0's exact action-order reverse, and the empty plan. Reverse legality is simulated. Each endpoint path has the same length and move-cost schedule. Candidate IDs are opaque 96-bit hashes derived from the versioned world/rules/plan bundle and role before any goal or gold exists. Candidate order is permuted once from that parent key and reused for all three goals.

Each goal specifies the complete `on` forest of one retained endpoint. Initial state, rules, constraints, instruction, utility definition, candidate IDs, plans and candidate order remain byte-equivalent within the task dictionaries; only `goal` changes. Records have distinct record indices and new immutable IDs, while parent custody is shared. The pre-vocabulary underlying bundle determines `parent_world_id`. After the shared vocabulary bijection, `metadata.goal_bundle_id=bundle_identity(input)` supplies the exact model-visible bundle identity for the separately versioned renderer. It excludes goal, instruction, targets and record index, and is identical across the triplet.

All six candidates are evaluated under all three goals with both oracles. A feasible full-goal-reaching path exists for each goal. Best sets must be nonempty and pairwise disjoint; disagreement or overlap is a `FAIL` with diagnostic evidence, not an exclusion silently removed from the denominator. A shorter prefix may win a goal and multiple candidate IDs may tie for a goal. Neither is removed. A hand-constructed fixture specifically proves retention of a cheaper prefix as one goal's best candidate and the complete path/reverse tie for another.

If 24 proposals yield fewer than three distinct non-initial endpoints, the goal helper returns `EXCLUDED`, reason `insufficient_distinct_noninitial_endpoints`, actual proposal count and the declared cap. It does not sample a replacement seed, increase the cap or publish two members of a triplet. The full A-H generator raises on that inadequate parent instead of returning a group missing C. Other oracle or consistency failures raise `GenerationError`; diagnostics preserve the disagreement rather than convert it to a favorable exclusion.

## Vocabularies and exact dynamics

The declared pools contain 384 globally distinct strings: 16 tokens for each of accounts, entries, blocks, registers, ledger status values and the table sentinel in each of TRAIN, DEV, OOD and future sealed structural reservations. All tokens have the same eight-lowercase-letter grammar and distribution; none contains a visible split prefix. The pools were generated once from the declared fixed pool recipe, without sampling pilot outcomes. Declaring a future sealed pool is not reading sealed gold.

Every v2 core task receives its split's reserved names, including partial worlds, goals, constraints, candidate action arguments and the explicit observation-path grammar. Invalid action references such as `absent` receive reserved tokens distinct from active entities, so they remain invalid. Table support is a separate namespace and cannot collide with block names. Register add/multiply/set/copy/swap argument positions are explicit. Status **values** in `rules.statuses`, entry statuses, status goals and forbidden-status constraints change consistently; semantic role keys `pending`, `settled`, `void` and every opcode stay unchanged. Numbers, costs, rule meanings, candidate IDs, prose and structural keys do not change.

Pool assignment uses parent/namespace SHA-256 order, independent of gold. The same map is reused across A's multi-goal tasks and C's triplet. Existing D entity/status perturbations retain distinct surfaces: declared `entity_rename` presentation salts change only account/entry assignment, and `status_vocabulary` salts change only status assignment. A deterministic pool rotation resolves the rare case where a salted used prefix would equal the base prefix. Other namespaces remain shared. D's explicit candidate-order control reverses the same parent permutation; it is not collapsed into the base order. Search-bearing tasks retain their original candidate order because prefix evaluation is order-sensitive.

Metadata records maps for all six namespaces, stratum, lexical split, presentation and a canonical source-task hash. `restore_source_names` verifies injectivity, exact allowed pool, split/stratum custody, a complete inverse round trip and the expected deterministic parent/presentation map. Removing an unknown-reference mapping or adding an unused/unregistered mapping fails. Profile validation calls this check before accepting a v2 record.

V1's bounded normalizer deliberately retains literal invalid-reference markers such as `absent`. Therefore raw v1 normalization of two renamed v2 inputs can differ even when execution is unchanged. The v2-aware auditor restores declared source names, including unknown references, and then invokes the unchanged v1 normalizer. This preserves historical v1 fingerprints and cross-version duplicate meaning. It does not add a version prefix, enlarge factorial limits or treat arbitrary changes as semantic equivalence.

The ledger vocabulary helper uses one DEV structural parent and returns a `familiar_control` using TRAIN pools and a `reserved_dev` member using DEV pools. Both remain DEV. Familiar controls are explicitly separated in metadata and never count as held-out vocabulary. Only complete reserved strings are held out; no tokenizer-piece or natural-language-generalization claim follows.

Exact utilities, success/feasibility probabilities, rankings and uncertainty are recomputed with two-oracle agreement and must preserve source candidate projections by ID. Matched search labels are rerun and must equal their original exact labels. Bounded-proof actions are renamed explicitly and their finite proof is rerun, retaining its scope and absence of an unbounded claim. Future states and witnesses remain under targets or nonmodel construction evidence; they are never added to inference input.

## Compatibility and fixture verification

The implementation uses fixture seeds 700–713 and one manually declared legal forest for debugging; it does not use the official pilot ranges. The focused tests cover deterministic regeneration, full-goal/candidate invariants, exact label agreement, valid prefix/tie retention, explicit 24-attempt exclusion, failure evidence, reserved pools across all core families, separate familiar controls, namespace/observation references, D surface distinction and malicious mapping edits.

The v1 builder file remains `78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575`; the active v1 curriculum file remains `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780`. Tests regenerate v1 before and after a v2 call and require identical results. Existing accepted-byte preservation and archive-profile revalidation are checked separately by the compatibility auditor. This source is a proposal requiring fresh independent review and a frozen bounded pilot. Fixture correctness is not a measured pilot result or a model-performance claim.

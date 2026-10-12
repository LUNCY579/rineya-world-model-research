# GCDL-001 rendered training surfaces, version 1.0

Canonical accepted records remain the gold source. The new renderer fixes two observed export weaknesses: sorted JSON removed generator D's field-order variation from model-visible serialization, and ordinary decision examples retained candidate-position patterns caused by the scheduler's stride of six. It also preserves the measured search and bounded distillation supervision omitted by the original decision-only export views. No generator, transition, utility, oracle, canonical ID, accepted shard or sealed pool changes are part of this layer. No training or model-performance experiment is launched.

## Model boundary and reversible transformations

Every output row has exactly one model feature, `input_text`, and a separate `target`. `record_id`, new stable `view_id`, `view`, `split`, full original `provenance`, `surface` and `verification` are nonmodel fields. Consumers must use the manifest's explicit `model_fields=["input_text"]` allowlist rather than concatenate whole rows. Parent world, lineage and split provenance are retained for sampling and leakage checks.

`input_text` contains a fixed neutral instruction, one marked JSON task and a fixed view request. Three bounded variants render every nested JSON object with ascending keys, descending keys or a deterministic seeded permutation; they use compact, indented or spaced JSON and three semantically equivalent instruction frames. Seeded ordering guarantees a third distinct order for objects with at least three keys. For two-key objects there are only two permutations. All values, status vocabularies, entity names, operation names, goal meanings and noncandidate array order remain unchanged. The source's own instruction/context fields remain inference input.

For ordinary listwise, pairwise and proof-only distillation examples, candidate order is sorted by SHA-256 of `[provenance.seed, provenance.record_index, variant.name, "candidate_order", candidate.id]`. Neither gold labels, best candidate, family, split nor `record_id` enters this ordering recipe; source IDs hash targets and therefore are deliberately excluded. Every format variant, including ascending-key format, uses this recipe. Source target utilities and probabilities remain exact by candidate ID. Listwise decision projections follow the rendered candidate order while ranking groups and best-ID sets stay unchanged. Pairwise left/right IDs and utility fields follow the rendered input, and preference changes sign when necessary. Pointwise has one candidate.

Any source with `targets.search` preserves its source candidate order in every view. Those targets measure the first-candidate versus full-candidate policies; changing that order would change the task. This restriction applies to computation and search-bearing distillation as well as decision views of those records. The renderer never repairs an order-sensitive search example by changing its gold.

`parse_input_text` accepts only complete registered framing, rejects duplicate object keys, nonfinite JSON constants and extra text, and handles marker-looking text inside JSON strings. Parsing recovers the rendered task exactly. Restoring candidates to source subset order must then recover the exact canonical input, including all other list order and leaf values. Each parsed task is executed by the separately implemented primary and independent oracles; both must agree with the original candidate outcomes by ID. Exact decision/search/proof labels are independently rechecked. Thus the permitted relation to canonical input is explicitly candidate permutation, not unrestricted semantic equivalence.

## Views and auxiliary supervision

| View | Source selection | Separate target |
|---|---|---|
| pointwise | Every accepted source; one candidate per row | Original exact utility, success and feasibility projection |
| pairwise | Every accepted source; every unordered candidate pair | Oriented IDs, exact utilities and +1/0/-1 preference |
| listwise | Every accepted source | Full decision projections, ranking, best IDs, uncertainty and supplied-candidate optimality scope |
| computation | Accepted source with `targets.search` | Exact `decision` and complete matched `search` |
| distillation | Accepted source with `search` and/or `bounded_proof` | Exact `decision` and every available search/proof target |

Computation framing declares the fixed first-candidate/full-candidate policies outside the task JSON. Present `search_request` input stays unchanged. Computation export rejects another policy budget rather than silently inventing a request or relabel its measurements. Distillation framing asks generically to compare alternatives and consider a plan. A proof's action alphabet, horizon, solution-existence flag and witness stay exclusively in the target. Its finite scope is oracle-validation supervision, not decision-time knowledge of an unknown proof or an unbounded planning claim. `global_unbounded_optimality_claim=false` and `scope=finite_declared_action_alphabet_and_horizon` remain intact. The renderer adds no entity/status aliases; existing executable source variants already have their own checked gold.

## Publication and API

`render_training_views(record, view="listwise", variant_budget=3)` yields deterministic accepted-only rows. All five views reject missing acceptance status, rejected/quarantined data and sealed records. Every source is rerun through `audit_record`; the stored verification evidence hash must match the fresh result. Previously accepted gold is preserved and reverified under the current unchanged semantic code.

`export_rendered_dataset(source_root, split, view, max_records, variant_budget, output_dir=None, source_paths=None)` returns its manifest dictionary. `max_records` bounds accepted source rows examined, including rows without the selected auxiliary target. `variant_budget` is 1..3. Optional `source_paths` selects newly accepted batch shards; paths must be named `split-seed.jsonl`, directly beneath `data/accepted`. Each shard must match its existing `manifests/batch-split-seed.json`: exact path, SHA-256, byte count, full row count, successful acceptance audit, historical source-code hashes and protected index. `PASS` requires no reported quarantine; `PARTIAL_QUARANTINE` requires an explicit positive integer quarantine count. Present audit accepted/quarantined counts must match the accepted shard and batch receipt. Mixed batches can export their accepted shard, while quarantined shards and rows remain excluded. Canonical IDs, actual semantic fingerprint, parent/template/split and seed must match that index; each record's historical verification code hashes must match its batch receipt. Every selected accepted row still receives a full fresh audit and evidence-hash check. Unmanaged or mutated sources cannot export by merely setting an accepted flag.

The output base must be beneath `source_root/data/rendered`. A deterministic dataset-ID directory holds immutable JSONL shards and `manifest.json`. The manifest records selection IDs and source-record hashes, full parent/split provenance, source shard and acceptance-receipt hashes, historical source-code hashes, current rendering code hashes/digest, explicit versions, output shard hashes/bytes/counts, model/target field allowlists and `training_launched=false`. `manifest_path` and all shard paths are relative to `source_root`. Byte-identical replay returns the same manifest; conflicts fail. Changed code produces a separately identified derived dataset. A manifest is the completion marker; files from a failed, incomplete publication have no authoritative manifest.

Before publication, source and receipt hashes and current code hashes are rechecked. Every shard is strictly below 8 MiB. The shared programme must remain within 512 MiB and retain at least 10 GiB free, with unpublished buffer bytes reserved before growth and each write. Destination directory, shard, manifest and temporary-file symlink escapes are rejected; canonical source directories and receipt paths cannot redirect. No canonical datasets or manifests are overwritten.

Example bounded CLI invocation, from the proposal programme:

```sh
python3 scripts/export_training.py --split dev --view listwise --max-records 27 --variant-budget 3 --source-shard data/accepted/dev-1000003.jsonl
```

`--view computation` or `--view distillation` selects the eligible rows among the same bounded source selection. The CLI rejects `--split sealed`, requires both limits, publishes data and prints a concise manifest pointer. It does not train.

An accepted subset may contain no computation or distillation targets, especially after partial quarantine. Such a selection produces a deterministic immutable zero-eligible manifest with `rendered_rows=0`, `eligible_source_records=0` and `shards=[]`, retaining its real source hashes, canonical IDs and provenance. This records skipped auxiliary supervision without inventing targets or pausing otherwise valid production. A selection with no accepted source records still fails. Full accepted-shard status checks also reject quarantined rows outside a bounded selection's examined prefix.

## Controlled observed DEV/OOD comparison

This comparison used only the existing accepted DEV/OOD shards in the proposal worktree. It did not read sealed inputs or targets. DEV seeds were exactly 1000003, 1000009, 1000015 and 1000021; OOD seeds were 1000004, 1000010, 1000016 and 1000022. These are the actual scheduler stride, rather than consecutive development seeds. Every source record generated three listwise rendered rows and passed canonical-subset equivalence, two-oracle outcome checks and unchanged exact label checks.

| Observed measure | Canonical/old export | Rendered output |
|---|---:|---:|
| DEV source records / derived rows | 108 | 324 |
| DEV noncanonical top-level JSON key orders | 0 | 216 of 324 |
| Distinct top-level key orders for each DEV task | 1 | 3 |
| DEV B close-call unique best at zero-based slots 0 / 1 | 0 / 4 | 6 / 6 |
| OOD source records / derived rows | 116 | 348 |
| OOD noncanonical top-level JSON key orders | 0 | 232 of 348 |
| OOD B close-call unique best at zero-based slots 0 / 1 | 4 / 0 | 4 / 8 |

The DEV `BEFORE 0 -> AFTER 216` result is an actual change in keys within model-visible prompt JSON, not a comparison of metadata tags. Key orders were read from `parse_input_text(row["input_text"])`, and nested rule keys were independently tested for ascending/reverse order. DEV close-call gold IDs were `gcdl_c910b54b9db788a8e58373710f6da1fe`, `gcdl_954e31081c9adcb26d42d8d5bc4951f2`, `gcdl_65b8a5bf0b047b50c2ff58023341f6de` and `gcdl_cec191e3727bd12e814734416f1ee190`. The best-position counts use original best-ID membership only for measurement, after permutations have already been determined without targets.

All eight accepted shard hashes were unchanged by the comparison. DEV shard SHA-256 values, in seed order, were `91e716b03ee868b0f687c0f03274342cadffd231392ee37bec7feb198854cade`, `f8e0b6b6b1af410ddaec03b463682c194dfe584af3094ee301eacba48e87469c`, `97a45436ccc10688c0aec3b301a68eff09bb4a1ac3b4533ea2b4af14009823c7` and `3ee0d598daa18ecd495445112666a79bc2539491fb62a9bce65d1a819e0797ee`. The new tests also verify target-only auxiliary supervision, pairwise orientation, immutable replay/hash conflicts, accepted source receipts, limits and source/destination isolation.

The small observed DEV count happens to balance the two best slots; the algorithm does not use gold to force balance, and OOD remains 4/8 in this sample. These results show preservation of executable semantics and removal of the observed deterministic slot assignment, with real surface variation. They establish neither universal shortcut resistance nor a model's generalization or performance.

# GCDL-001 continuation coverage audit

Audit snapshot: 2026-10-12 00:34:41 UTC (local evening of 11 October 2026). Scope is a read-only code and nonsealed-corpus audit. Only this Markdown report was written. No production source, configuration, or accepted shard was modified; no sealed gold shard or sealed target was opened; no model training, model inference API, or paid provider was used. Aggregate sealed counts below come from `state/metrics.json`. The daemon advanced during the audit: earlier exploratory reads saw 1,800 accepted rows, while the final snapshot sees 1,976. Final tables use the final snapshot. This is a coverage diagnosis, not approval to promote an upgrade or a fresh release review.

## Current accepted corpus and exports

`state/metrics.json` reports 1,976 accepted, zero quarantined, oracle agreement 1.0, schema pass rate 1.0, zero queue, 168.364 active generation/audit seconds, and 28,260,770 programme bytes. Those agreement/pass fields are producer metrics, not a new exhaustive independent re-execution by this audit. Split counts are TRAIN 1,200, DEV 270, OOD 290, and SEALED 216 (aggregate count only). `reports/QUALITY_REPORT.json` still describes 920 accepted and should not be treated as the live corpus size.

| Split | Accepted records | Immutable shards | Pointwise rows | Pairwise rows | Listwise rows |
|---|---:|---:|---:|---:|---:|
| train | 1,200 | 40 | 4,760 | 7,680 | 1,200 |
| dev | 270 | 10 | 1,010 | 1,470 | 270 |
| ood | 290 | 10 | 1,100 | 1,630 | 290 |

Across the allowed 1,760 rows, environments are ledger 1,630, Blocksworld 70, and register 60. A-H are present in every allowed split. This diversity is heavily weighted toward additive ledger templates. All allowed rows read here have `quality_gate.status=accepted`.

The actual `scripts/export.py --split <train|dev|ood> --view listwise` interface was run for all three splits. Row counts match the accepted shards; every recursively nested inference dictionary has sorted keys in the emitted bytes. No accepted target trajectory or metadata is copied into a view input by the inspected `training_views` implementation.

## Concrete failures and shortcuts

### 1. The field-order perturbation does not reach model input

`generators/build.py:406-412` reverses the Python input dictionary and combines that with candidate reversal, one paraphrase, and one irrelevant context sentence. Canonical persistence already sorts dictionary keys; `scripts/export.py:15` sorts them again. TRAIN 40, DEV 10, and OOD 10 records have the `field_order` tag, yet all have the same sorted input key order: `candidates, constraints, context, environment, goal, initial_state, instruction, rules, utility_definition`.

Removing instruction/context and restoring candidate order makes the tagged inference task identical to its recorded base. This was checked in the earlier complete 54-record nonsealed snapshot; the final export-wide recursive sorting assertion establishes the same serialization failure for the now 60 tagged nonsealed records. Candidate order, prose, and context do survive, so the whole D perturbation is not empty. The specific key-order robustness instrument is empty. A test comparing Python insertion order before persistence would miss the failure; validation must inspect the bytes ultimately fed to a model.

### 2. Renaming exists, but a vocabulary holdout does not

`generators/build.py:393-405` uses the same account/entry aliases `cedar, river, tally` and the same status surfaces `awaiting, recorded, withdrawn` in TRAIN, DEV, and OOD. Every split has exactly the same ledger status surface vocabulary: `awaiting, pending, recorded, settled, void, withdrawn`. The number of DEV or OOD status surfaces absent from TRAIN is **0/6**. The three alternative status surfaces are not a held-out lexical test.

Entity lexicons repeat all renamed active ledger entities. Additional `account2/account3` and `entry1/entry2/entry3` strings enter larger templates, but accepted DEV/OOD settle/void actions reference only `e`, `tally`, or intentionally `absent`; they never operate on the added entries. Larger ledgers therefore do not currently demand multi-entry interaction. Structural, signed-number, and magnitude shifts are real; unseen renamed vocabulary and interactions among the added ledger entries are not established by these exported tasks.

### 3. The production seed stride aliases with the option rotation rule

`generators/build.py:618-625` places a unique best candidate at `(seed + record_index) % candidate_count`. `configs/production.json` assigns DEV seeds `1000003 + 6k` and OOD seeds `1000004 + 6k`. For the two-option B close-call fixture at index 4, this mathematically fixes every DEV winner at slot 1 and every OOD winner at slot 0. Current accepted DEV is **10/10 at slot 1**; OOD is **10/10 at slot 0**. For the four-option A fixture at index 0, DEV occupies only slots 1 and 3 (5/5), while OOD occupies only slots 0 and 2 (5/5).

The recorded DEV improvement used 32 consecutive seeds, which balance modulo 2/4. It does not reproduce the production stride of six. Aggregate winner counts across unrelated fixtures can look balanced while each fixture retains a strong position correlation. This is a concrete defect in the measured position-coverage guarantee for production DEV/OOD, not a claim that every exported task is solved by position.

### 4. A goal-blind candidate-relationship heuristic is strong on C

An input-only heuristic selects a candidate whose plan has both (a) another supplied plan equal to its one-step-shorter prefix and (b) another supplied plan equal to its exact reverse. If several qualify, it chooses the lowest opaque candidate ID. It reads no goal, initial state, rules, oracle, metadata, or candidate role. The generator constructs the goal from that path and supplies its prefix/reverse as distractors (`build.py:455-482`).

| Split | C records | Candidate identified | Chosen ID belongs to exact best set | Entire best set predicted exactly |
|---|---:|---:|---:|---:|
| train | 40 | 40 | 38/40 | 22/40 |
| dev | 10 | 10 | 10/10 | 8/10 |
| ood | 10 | 10 | 9/10 | 8/10 |

These are mechanical-baseline counts, not trained-model results. Best-set membership and exact tied-set recovery are deliberately reported separately. More unrelated candidate paths and counterfactual goals on the same candidate set would help distinguish actual goal-conditioned planning from generator-role recognition.

Similarly, all 60 allowed H simultaneous-goal fixtures have `settle` as sole best. All register fixtures expose six enabled operations but supplied candidates use only `add` and `noop`; no production register candidate executes multiply/set/copy/swap. The OOD constraint-key conjunction is withheld, but current winner behavior does not force a novel interaction among its primitives.

### 5. Search and finite-proof supervision is absent from current exports

The allowed corpus contains **300** records with `targets.search` and **60** with `targets.bounded_proof`. `decision_labels/views.py:15-36` projects only `targets.decision`: pointwise utility/probabilities, pairwise preference/utilities, and listwise ranking/uncertainty. No current view emits measured cheap/expanded execution counts, realized utility delta, finite alphabet/horizon, omitted-solution witness, or bounded-proof existence.

These targets exist in accepted records and are audited by the production pipeline. This is an export coverage limitation: the current trainer-facing views support outcome scoring/ranking, not explicit supervision of computation value or proof/witness generation. Adding those views would require supplying declared budgets/action alphabets at inference time while retaining all oracle results exclusively in targets.

## Small next improvement and explicit DEV validation

The smallest measurable improvement is to repair model-visible presentation coverage, with production position aliasing treated as a separate required diagnostic. Keep existing canonical accepted records and IDs immutable. A versioned opt-in export renderer can emit an inference-only JSON **string** whose internal field order is deterministically permuted; outer canonical JSON sorting cannot erase order inside a string. The downstream training contract must choose that rendered text as its actual model payload. Adding shuffled dictionaries beside canonical structured inputs is insufficient. Keep semantic keys unchanged; render no metadata, target labels, or candidate roles.

For renaming, declare a versioned split-reserved lexicon before generation/rendering and bijectively rename every active account, entry, block, register, and status surface consistently. Reserve different TRAIN/DEV/OOD strings and several deterministic parent-specific permutations. Keep semantic rule keys and action opcodes visible and unchanged. Existing accepted bytes need no rewrite. A new canonical task requires a new immutable ID; an exported presentation derivative should name its immutable source record and renderer version.

**Before DEV:** actual accepted DEV has 10 tagged rows but only one sorted field order; 0/6 DEV status surfaces are absent from TRAIN. Production-shaped B winner positions are 0/10 at slot 0 and 10/10 at slot 1. The previous consecutive-seed experiment misses the schedule effect.

**After DEV, pending implementation and a fresh independent reviewer:** run paired seeds `410000..410031` and a separate production-shaped set `1000003 + 6k` for `k=0..31`; do not consult sealed gold. Require:

1. At least four distinct rendered field orders, each surviving the exact exporter-to-model-payload path. Parsing each rendered input must equal the corresponding structured inference task after only the declared bijection. A negative control using sorted serialization must fail the order-coverage requirement.
2. Every renamed DEV active entity/status token absent from the TRAIN reserved renamed lexicon; all three renamed status surfaces unseen in TRAIN. Report ordinary structural vocabulary separately, so rule keys such as `pending` are not falsely counted as held-out surface strings.
3. Zero changed exact candidate utilities, success/feasible probabilities, ranking, best-set, and abstention under semantics-preserving rendering/renaming; both executable oracles must agree. Compare renamed state strings through declared bijections, not raw equality of renamed trajectories.
4. Zero accepted-shard byte/hash changes and no sealed reads. Preserve protected parent custody and semantic fingerprints under namespace normalization. Export rejection/quarantine gates remain enforced.
5. For any position fix, validate the actual stride-six fixture and each A/B/C/D/H stratum independently. The 32-pair two-option close-call after-count should be 16/16 and the four-option A after-count 8/8/8/8. Preserve prefix-budget search order because changing it changes the computation policy. A fresh reviewer decides promotion.

No after measurement or promotion is claimed here. The renderer/lexicon proposal addresses a falsifiable surface-coverage gap; it does not demonstrate model learning, general decision intelligence, or real-world causal transfer. A separate later improvement should diversify C candidate relationships and export search/proof supervision rather than combine those broader changes with this presentation repair.

## Snapshot commitments and reproducibility

SHA-256 of sorted lines `relative_shard_path:sha256(shard_bytes)` for the allowed shard collections:

- TRAIN `cfb6a1424723475f3a2a74682abc7c05f131711e64096f912fbc7ee6fa05cbdc`
- DEV `2551431fa0b1b79fcb210b28f0e8378d0c05169a08f99e1b74f6e5dc089d6ef6`
- OOD `ba204829df84d75a6a89b6cbcf10ac310e14607d223218000b554ae14eec5f74`

Actual listwise export stdout SHA-256, including terminal newlines:

- TRAIN `9ba34495172363de2050bbfbf1109e52a672c7f54e1a0af07829ea8b2804d555`
- DEV `ef201ccda4e2586ebc0a008d1ddca515130351649b0dfad4c825ddc983371524`
- OOD `1d16ccefaba5d0d36e587e2e058f3be7a7697b210f2bbfa00cb6e26eb54c5400`

Source/configuration hashes at audit:

- `DATA_CONTRACT.md`: `e92b9e720aa70cd46589eb1c2c8c0ee7d7523f36e338d3f1581ab20fba7f5762`
- `generators/build.py`: `78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575`
- `decision_labels/views.py`: `3a6851fc7ba836bd66ae6c8b4f1188821c28190c61e5681ac8e75d3bad02372e`
- `scripts/export.py`: `94e319ac4c24e6cdae8e46a0f32edefcf65fc3631f6d6b6e71d5dd63fc84cccb`
- `configs/curriculum.json`: `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780`

Reproduce counts and sorting using the three allowed `scripts/export.py --split SPLIT --view listwise` calls. Parse each emitted line with Python `json.loads`, which preserves dictionary order, and recursively assert `list(obj) == sorted(obj)` for dictionaries under `input`. For the C baseline, inspect only candidate plans for the prefix/reverse relationships above, choose the lowest qualifying opaque ID, and compare it with the best-set target afterward. CPU work was bounded to nonsealed accepted-shard reads and nonsealed listwise exports; no factorial semantic-fingerprint enumeration over the whole corpus was launched.

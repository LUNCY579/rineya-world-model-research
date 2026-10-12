# Controlled rendered-surface comparison

Measured at 2026-10-12T00:44:36.209401+00:00. Proposal programme: `/Users/chengli/.codex/gcdl001-host/improvement-worktree/goal_conditioned_decision_data`. Immutable source programme: `/Users/chengli/.codex/gcdl001-host/worktree/goal_conditioned_decision_data`. This is an independent read-only analysis of the proposal renderer, not approval to promote its code. Only this report was written. No canonical source record, source configuration, source manifest, or sealed gold was modified or read beyond the allowed nonsealed sources. No model training, model inference, or paid provider was used.

The earlier [coverage audit](/Users/chengli/.codex/gcdl001-host/worktree/goal_conditioned_decision_data/reports/COVERAGE_AUDIT_20261011_CONTINUATION.md) found that canonical export erased field-order variation, repeated renamed vocabulary across splits, and aliased production stride-six seeds with winner rotation. This comparison tests the proposed renderer against those concrete failures. Its source digest was stable throughout measurement: `3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12`. The final frozen renderer and all executable/test/config source files were rechecked in a 17.144-second full rerun; the complete source map was identical before and after. The final renderer SHA-256 is `b9ac033f423deb4221d9c85199a967317c4d9092fb049c4cf9f4f0a7b341f700`. These measurements bind to that exact final snapshot and still do not constitute promotion approval.

## Sources and hypothesis

At measurement, production metrics reported 2152 accepted records: TRAIN 1320, DEV 297, OOD 319, and SEALED 216 (sealed aggregate count only). This agent read every current DEV/OOD shard, with seeds `1000003 + 6k` and `1000004 + 6k`, respectively, for `k=0..10`. New daemon shards can appear afterward and are outside this fixed snapshot.

Hypothesis: reversible nested JSON key ordering plus candidate ordering computed without gold labels exposes actual model-visible order variations and reduces the deterministic position correlation of ordinary decision fixtures, while preserving exact per-candidate outcomes and decision labels. The candidate shuffle uses seed, record index, surface name, and opaque candidate ID; it does not use best labels, family, split, or target-derived record ID. Search-prefix records preserve their original candidate order because that order defines computation policy.

Each source shard was authenticated against its immutable batch manifest: path, SHA-256, byte/record count, accepted/PASS status, protected parent/split identity, and original verification code-hash map were checked using the proposal source helpers at the tested digest. The final rerun also validated each source seed and recomputed its protected semantic fingerprint, and checked the final complete-source accepted-flag authentication. All source and manifest bytes remained unchanged.

| Split | Canonical records | Accepted shards | Rendered listwise rows | Sources with three distinct surviving top-key orders | Parse / oracle / label mismatches |
|---|---:|---:|---:|---:|---:|
| dev | 297 | 11 | 891 | 297/297 | 0 / 0 / 0 |
| ood | 319 | 11 | 957 | 319/319 | 0 / 0 / 0 |

## Before and after actual model payloads

For each source, the canonical baseline was serialized with sorted keys. The renderer emitted three registered `input_text` variants: `canonical_compact`, `reverse_indented`, and `seeded_spaced`. Each complete rendered row was serialized again with `canonical(item) + "\n"`, exactly the publisher's JSONL encoding, decoded with `json.loads`, and its retained `input_text` passed to `parse_input_text`. Thus measurement crossed the outer canonical serialization boundary that erased the old dictionary perturbation. No rendered dataset was published by this agent.

Every canonical source has one sorted dictionary order for its own schema; different optional fields create four source schemas overall, not within-task order diversity. **Every one of the 616 tested records has three distinct decoded top-level orders after rendering.** DEV has 303 distinct top-level key sequences across the three renderings and heterogeneous schemas; OOD has 323. All 11 DEV and 11 OOD `field_order`-tagged sources therefore expose real within-source order changes. Reverse ordering also reaches nested dictionaries because the renderer applies its policy recursively. All arrays other than the permitted candidate list remain exact.

Ordinary four-option A index-0 and two-option B close-call fixtures were counted separately. Slot vectors are in zero-based slot order. Before has 11 source decisions per cell; after pools three presentations of those same 11 decisions, so each after vector totals 33.

| Split | Fixture | Before slot counts | After pooled slot counts | Maximum slot fraction |
|---|---|---|---|---|
| dev | A_goal0 | 0 / 5 / 0 / 6 | 9 / 6 / 10 / 8 | 54.5% → 30.3% |
| dev | B_close_call | 0 / 11 | 18 / 15 | 100.0% → 54.5% |
| ood | A_goal0 | 6 / 0 / 5 / 0 | 8 / 8 / 4 / 13 | 54.5% → 39.4% |
| ood | B_close_call | 11 / 0 | 13 / 20 | 100.0% → 60.6% |

The strong fixed-slot close-call correlation is reduced in both real stride-six strata. The hash shuffle does **not** guarantee exact balance: after is 18/15 rather than 16.5/16.5 in DEV and 13/20 in OOD. A formerly unreachable slot now appears in every A stratum. This is a mechanical position diagnostic; it is not a model-performance improvement.

Variant-specific counts matter because a consumer selecting only one surface may retain more imbalance:

| Split | Fixture | Surface | After slot counts |
|---|---|---|---|
| dev | A_goal0 | canonical_compact | 3 / 2 / 2 / 4 |
| dev | A_goal0 | reverse_indented | 4 / 2 / 3 / 2 |
| dev | A_goal0 | seeded_spaced | 2 / 2 / 5 / 2 |
| dev | B_close_call | canonical_compact | 6 / 5 |
| dev | B_close_call | reverse_indented | 9 / 2 |
| dev | B_close_call | seeded_spaced | 3 / 8 |
| ood | A_goal0 | canonical_compact | 3 / 2 / 1 / 5 |
| ood | A_goal0 | reverse_indented | 1 / 2 / 2 / 6 |
| ood | A_goal0 | seeded_spaced | 4 / 4 / 1 / 2 |
| ood | B_close_call | canonical_compact | 6 / 5 |
| ood | B_close_call | reverse_indented | 5 / 6 |
| ood | B_close_call | seeded_spaced | 2 / 9 |

In particular, DEV reverse-indented close-call is 9/2 and OOD seeded-spaced close-call is 2/9. Pooling improves overall exposure but does not establish balance for an individual surface. The 33 rows per cell share 11 source parents and cannot be treated as 33 independent scientific observations. No significance test or model score is claimed.

## Exact semantics, target preservation, and custody

For all 1,848 serialized/reparsed listwise rows, this audit independently checked:

- Parsed task equals the original inference input after restoring candidate order by opaque ID. Candidate IDs, plans, values, goals, rules, constraints, prose, unknown leaves, and possible-world arrays are unchanged.
- The primary and separately implemented independent oracles agree. Their complete per-candidate outcomes equal accepted gold after only reordering the outer candidate list by ID.
- Decision targets equal original accepted decision targets after normalizing only the candidate projection array. Exact utilities, success/feasibility fractions, ranking, best-set, uncertainty, abstention, and optimality labels remain unchanged.
- The original in-memory canonical row and every source shard SHA-256 remain unchanged after all renderings.

All four mismatch/mutation counts are zero. DEV has 726 gold-independent shuffled rows and 165 prefix-order-preserved rows; OOD has 792 and 165, respectively. This comparison covers listwise semantic equivalence and presentation diagnostics. Pointwise/pairwise orientation, computation/distillation labels, exporter publication gates, resource ceilings, and final source containment are separate reviewer responsibilities.

## Explicit unresolved gaps

**No vocabulary holdout was added.** The decoded DEV and OOD status-surface sets before and after are exactly `awaiting, pending, recorded, settled, void, withdrawn`, the same set already observed in TRAIN by the earlier coverage audit. The renderer performs no renaming, and source equivalence permits candidate permutation only. It consequently cannot create unseen active entity/status vocabulary. Framing and whitespace changes are presentation variants, not new ledger terminology.

**Canonical generator C was not changed.** `generators/build.py` has the same SHA-256 as the earlier audit. The goal is still constructed from one candidate's path, and the supplied prefix/reverse still identify that candidate. The following heuristic reads only the candidate plans: qualify a plan when another supplied plan equals its one-step-shorter prefix and another equals its exact reverse; choose the lowest opaque ID among qualifiers. It never reads goal, state, rules, labels, or metadata. Because it uses ID set and plan relationships rather than option position, shuffling cannot defeat it.

| Split | Before chosen ID belongs to best set | Before exact tied best-set recovery | After best-set membership | After exact best-set recovery | Chosen ID changed by renderer |
|---|---:|---:|---:|---:|---:|
| dev | 11/11 | 9/11 | 33/33 | 27/33 | 0 |
| ood | 10/11 | 9/11 | 30/33 | 27/33 | 0 |

The same selected ID is retained for every C rendering. DEV remains 100% best-membership correct; OOD remains 10/11. Exact tied-set recovery is weaker and separately reported. This unresolved shortcut prevents attributing strong C selection results to goal-conditioned multi-step reasoning.

The proposal renderer adds computation/distillation view support, whereas the old exports omitted such supervision. This comparison did not measure those views and does not extend its listwise equivalence claim to them.

## Proposed next controlled generator-v2 hypothesis

A narrow future experiment can add **balanced reverse-goal pairs** without changing the transition world or candidate plans within a pair. For each parent C problem, use (a) its generated terminal-forest goal and (b) its valid initial-forest goal; retain the exact same initial state, rules, constraints, candidates, IDs within the pair, and cost semantics. The terminal goal should favor the generated path, while the initial goal favors the empty plan. Give the changed canonical task a new immutable record ID, keep both records in the same protected parent/split bundle, and recompute all labels with both oracles.

Before pre-registration, check that paired exact best sets are disjoint and that both tasks have a feasible candidate. If a proposed pair fails those conditions, record the exclusion and reason rather than silently claim balance. Use a fresh declared DEV seed range, for example `420000..420031`, and a paired parent-level analysis; never consult or rewrite sealed gold. The canonical-v1 corpus remains immutable and independently distinguishable from v2.

Measurable hypothesis: on 32 verified disjoint pairs (64 tasks with equal goal weighting), this goal-blind candidate-relationship heuristic makes the same choice in both tasks and can have at most 50% best-membership correctness. Require zero primary/independent disagreements, paired candidate/world equality apart from goal, full custody, and explicit before/after heuristic counts. A goal-aware oracle remains exact; no trained-model improvement is predicted by the audit.

This initial/terminal contrast would remove one established construction shortcut but introduce an obvious `goal == initial_state` cue. A later extension should balance goals drawn from multiple valid candidate endpoints and diversify distractor relationships. That broader design is not part of the current renderer proposal. No generator-v2 implementation or promotion is claimed by this report.

## Source commitments and reproduction

Aggregate digest is SHA-256 of canonical JSON of the complete final Python/configuration file-to-SHA map below (including the frozen reviewer tests). It was identical before/after the final full comparison, and the rendered JSONL stream bytes were identical to the earlier measurement:

- `__init__.py`: `64c91b2f24f24214f818c1716c5e181f86625b648914e1eadf6880e82114346e`
- `auditors/integrity.py`: `1ce9a68a04d2eb71c3d3bcb8188dd3d392af69cda105b0caa408be9c6c53a62d`
- `auditors/release.py`: `c49f36f518385825fa92fc0a9f097fd83fd93d18b7f2c8d87ec3b6895eedeaea`
- `configs/curriculum.json`: `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780`
- `configs/production.json`: `ce4826f6342363100ff88c539beab79f47f3cc15b4be48c922527481ff303afd`
- `decision_labels/__init__.py`: `c46518068c5e4407e3f5c1a43eed2df9a775977a055773c9263b21bdddf1e637`
- `decision_labels/rendering.py`: `b9ac033f423deb4221d9c85199a967317c4d9092fb049c4cf9f4f0a7b341f700`
- `decision_labels/views.py`: `3a6851fc7ba836bd66ae6c8b4f1188821c28190c61e5681ac8e75d3bad02372e`
- `environments/__init__.py`: `0097b9cfbd6e2cf98c050afad5e9f8f7058739f2095792b336339750af581bc4`
- `environments/core.py`: `b4e383c455ba899737a8c034a239a23ac01fbc13279791b3e0bc66acb6ec014d`
- `generators/__init__.py`: `d45c107da508d34b0d040fcfe634a6f16fe2db2218244a5dfee49fa008742e11`
- `generators/build.py`: `78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575`
- `generators/improvement.py`: `4c4ad9b85050fa1b21a31bf71b752e217ce2661fafcf0bdd934463c01571ed0a`
- `oracles/__init__.py`: `d2ff07ef6ae530121d8484a744a81e3c9c12755e01a350f533e047240cc06397`
- `oracles/independent.py`: `0453985a3347ea9e503c44dff4eef9ad7f7de1a9efa88537c05d569c96a032ae`
- `oracles/primary.py`: `0797623f90d1d53de1034c0c8fb07fbfc56207924b829f9945729dcd52639d19`
- `pipelines/common.py`: `26eafc2cccefb424c8aff89b21cbb6cf81e73a67474c97b9749e02b942358b82`
- `pipelines/rendered.py`: `b5ef28f13dad6dc0e35b37161937b2995df44f1c77c9d07b06d7dbea08c5b2d5`
- `pipelines/worker.py`: `4b83c9cd71f592115eb1fa7e01dcdbbcb687d7cbcb0d166a0bed11b7e6394d1f`
- `scripts/backup.py`: `f0812c0fd9dda0c44250a16ac5c42d2fa908691813fb46065f0528ac4e47eb61`
- `scripts/control.py`: `9990ad7a7538dc9ce8fdd10d3db26f9a592f56f04d74ef6138f7f5054f8ea909`
- `scripts/export.py`: `94e319ac4c24e6cdae8e46a0f32edefcf65fc3631f6d6b6e71d5dd63fc84cccb`
- `scripts/export_training.py`: `0aefb65fd8cccc4d3e8086a437db9e8a51c6b65abb96f9803fc9b9a9588fc33f`
- `scripts/run_cycle.py`: `39e2f9e5096b87b7c856ea555bb5f07b31f352b1abddba1891ae60f37da9073c`
- `tests/test_adversarial_review.py`: `9dca1face6426a079b5c0b001b0e63eb68e2cfb3463e069784be6d22fc16ad59`
- `tests/test_generation.py`: `0dbc041d7cbf52d9dd6bc83a7e51efe4e4fc19e6bbbcc00e964affdc436a7341`
- `tests/test_operations.py`: `0015dbd1e6240045e5dfde95140b142c7a92cfdde835e647d17f990e77f4968b`
- `tests/test_oracles.py`: `e1995c834809e7a0ec45531465cba382408c4e9dfe0d309f72d9c469db340ff1`
- `tests/test_rendered_pipeline.py`: `ccf6293e1aedc15896e40aa14c690710001e8ef13e33027009a07ee17fa4847d`
- `tests/test_rendering.py`: `fc97a6b684221b6da7f2b96035bf89aa04d2cbd37146c45dc3545e4f2fc1d9fe`
- `tests/test_surface_review.py`: `79bf0c7bfb76324dd538af54f44818bae314847f4c566c059c6011a82600dc53`

Per-split source collection digest is SHA-256 of canonical JSON of the appendix's ordered source receipt list, including shard bytes, manifest bytes, and source code-map digest:

- dev: `25ac1328752393a479ad36c195d213542a15ef302c083900a70d7b3c6ef5220c` (11 shards; source and manifest commitments in the machine-readable appendix).
- ood: `1419b420597b33e195f77b76b55c3f3748b614867452cb48c37d8768c4688ad9` (11 shards; source and manifest commitments in the machine-readable appendix).

In-memory rendered JSONL stream digests, using the exact outer canonical encoding:

- dev: `d7b560ed0223b51dd5edc416ac6cc44cd3a19d53aadb89e44b3868aa7e408561` (891 canonical JSONL rows, in sorted source-shard order and source record/variant order).
- ood: `c5bd2943c6317ac20f37ab074d013633fde78f2696d54122625092c5d34f3a17` (957 canonical JSONL rows, in sorted source-shard order and source record/variant order).

To reproduce, import the proposal package from its own parent directory; read the explicitly committed production source shards; authenticate their batch receipts; call `render_training_views(record, "listwise", 3)`; serialize each returned row as `canonical(item) + "\n"`; decode and pass its `input_text` to `parse_input_text`. Compare source and parsed tasks/oracle outputs/decision projections by opaque candidate ID, preserving all other content. Run the goal-blind relationship heuristic before and after, and stratify winner slots by B close-call and A record-index zero. Do not include sealed paths in globbing or generation.

The comparison supports real field-order exposure and a reduction of one deterministic production position correlation. It does not approve the full exporter, solve vocabulary/C shortcuts, demonstrate model learning, or establish general decision intelligence.

## Machine-readable source snapshot

```json
{
  "utc": "2026-10-12T00:44:36.209401+00:00",
  "proposal_source_digest": "3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12",
  "source_snapshot_stable": true,
  "sources": {
    "dev": [
      {
        "code_map_sha256": "86b788cc171a844a49d6c33ae06ea59806f9e3a32ff27244b4a5f4c7444b0a7c",
        "path": "data/accepted/dev-1000003.jsonl",
        "records": 27,
        "sha256": "91e716b03ee868b0f687c0f03274342cadffd231392ee37bec7feb198854cade",
        "source_batch_manifest": "manifests/batch-dev-1000003.json",
        "source_batch_manifest_sha256": "bc13a93f71ff8c969aa997f7f351023fa9f0d0cf26538368498a9c6415e2b0ff"
      },
      {
        "code_map_sha256": "86b788cc171a844a49d6c33ae06ea59806f9e3a32ff27244b4a5f4c7444b0a7c",
        "path": "data/accepted/dev-1000009.jsonl",
        "records": 27,
        "sha256": "f8e0b6b6b1af410ddaec03b463682c194dfe584af3094ee301eacba48e87469c",
        "source_batch_manifest": "manifests/batch-dev-1000009.json",
        "source_batch_manifest_sha256": "7876fa0cb6dd09f9514f9f0708dedf2e89e5147bbccb6c612e8aa703501b0cef"
      },
      {
        "code_map_sha256": "6ab9696269e42a9aeaee1198d24a6cb49e6b3e39b3102e0b4056488fc47b265a",
        "path": "data/accepted/dev-1000015.jsonl",
        "records": 27,
        "sha256": "97a45436ccc10688c0aec3b301a68eff09bb4a1ac3b4533ea2b4af14009823c7",
        "source_batch_manifest": "manifests/batch-dev-1000015.json",
        "source_batch_manifest_sha256": "9fcd24f4b710e1f24319d93c0997a9cfce039afc7fe3fcce6913f41be99185ba"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000021.jsonl",
        "records": 27,
        "sha256": "3ee0d598daa18ecd495445112666a79bc2539491fb62a9bce65d1a819e0797ee",
        "source_batch_manifest": "manifests/batch-dev-1000021.json",
        "source_batch_manifest_sha256": "29d6396379f391f2e9fe8e21413dbe5b08b7e49536ef973d0305655355829f3d"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000027.jsonl",
        "records": 27,
        "sha256": "5b36adb1e5666ce8a1e89914d947992e217e7be3154feab69572d6e0f04d944f",
        "source_batch_manifest": "manifests/batch-dev-1000027.json",
        "source_batch_manifest_sha256": "3a24bd0850f72b7404bbe592b22d2da205813fc505622e58e1fdf0cea2c61705"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000033.jsonl",
        "records": 27,
        "sha256": "349e02263c662e9f4162a922f80c91ce4a6a588c9c47072d388fc38e575e28ed",
        "source_batch_manifest": "manifests/batch-dev-1000033.json",
        "source_batch_manifest_sha256": "b7a6eaf1b348c14f84ef89ce38a3aecc70cce0d31282d34b3e4b1bba9e74352d"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000039.jsonl",
        "records": 27,
        "sha256": "f49add536c96ba7c892737d733fa6106fa615aa638a96bda8c7011949ee0989c",
        "source_batch_manifest": "manifests/batch-dev-1000039.json",
        "source_batch_manifest_sha256": "f0343b1d352af192289fac0fda95f9a58ea784031df71773d86ea70aba25acb7"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000045.jsonl",
        "records": 27,
        "sha256": "8a9308f809980ad38ea1c7c04f6fba85d0d134e25af757825650b4827a555a98",
        "source_batch_manifest": "manifests/batch-dev-1000045.json",
        "source_batch_manifest_sha256": "139846d97396bdd1e895f6a151c14e6d97462d36f6e9ad8fe01213dfd8b4dfb2"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000051.jsonl",
        "records": 27,
        "sha256": "77f3e25d252a0f9bf259c4761a940421158c8a38f69d03f345025b15984e0632",
        "source_batch_manifest": "manifests/batch-dev-1000051.json",
        "source_batch_manifest_sha256": "9ec4ea420d7b2bf3e88c016a6aa17ad4fe3a6b49c37699ca28611847e017fab6"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000057.jsonl",
        "records": 27,
        "sha256": "1aed03281b8cd34e9a05229d8a0271ae862c35dbb990a28cfd3db9e98f9dcff9",
        "source_batch_manifest": "manifests/batch-dev-1000057.json",
        "source_batch_manifest_sha256": "2f027de83db92120c88687c876ed892237cfa45c50bd4a78f168ab3508048274"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/dev-1000063.jsonl",
        "records": 27,
        "sha256": "8a18f390e331137f41721c83d3bdafe7bbdcb91b74b7901a094930229796ad27",
        "source_batch_manifest": "manifests/batch-dev-1000063.json",
        "source_batch_manifest_sha256": "1e6ec428e3d2df6ffd7a498a9dad32454f17429ad20bbdffbbd0c1aa5748c5d5"
      }
    ],
    "ood": [
      {
        "code_map_sha256": "86b788cc171a844a49d6c33ae06ea59806f9e3a32ff27244b4a5f4c7444b0a7c",
        "path": "data/accepted/ood-1000004.jsonl",
        "records": 29,
        "sha256": "37585a5b67757305b8aa0e6721705ef168793d0c59a31329348254659e416b4c",
        "source_batch_manifest": "manifests/batch-ood-1000004.json",
        "source_batch_manifest_sha256": "b87257feb0136fda1bd51b672255cd2f3cc8b72a143dea2f37640a0369438418"
      },
      {
        "code_map_sha256": "86b788cc171a844a49d6c33ae06ea59806f9e3a32ff27244b4a5f4c7444b0a7c",
        "path": "data/accepted/ood-1000010.jsonl",
        "records": 29,
        "sha256": "491f790f3df928edec75d45018eca3381d2810cde35b979f1511c6777ed37b31",
        "source_batch_manifest": "manifests/batch-ood-1000010.json",
        "source_batch_manifest_sha256": "b58f5ac4dbfa231351e776fb8c2b7f09bc818f05b2f6a146d7c72b80bc6f24dc"
      },
      {
        "code_map_sha256": "6ab9696269e42a9aeaee1198d24a6cb49e6b3e39b3102e0b4056488fc47b265a",
        "path": "data/accepted/ood-1000016.jsonl",
        "records": 29,
        "sha256": "bbee4a943abb8f19cc39815de18d52147f0b654fb7e9a8f793690296667ab4b3",
        "source_batch_manifest": "manifests/batch-ood-1000016.json",
        "source_batch_manifest_sha256": "8cf8b9b360649c89a4d2b6966b5139e5362c9903726e5578c9414ea8f5c0ecad"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000022.jsonl",
        "records": 29,
        "sha256": "88097296d693bf5652da4faea932185b5ffe21e51c01b245d81a3a915ddf8582",
        "source_batch_manifest": "manifests/batch-ood-1000022.json",
        "source_batch_manifest_sha256": "9344f6b52d7236be420dc379a8e64149a153b276534d27bd300770f9daae620e"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000028.jsonl",
        "records": 29,
        "sha256": "6756f095bb8f8b02f23a08dd7628cd3dbf1b6357225a3d50617b6593a517b010",
        "source_batch_manifest": "manifests/batch-ood-1000028.json",
        "source_batch_manifest_sha256": "dda92ccd1e4ce7d6aa4de1f784077c5da918ab864a5eaef02c89640ad19a1b16"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000034.jsonl",
        "records": 29,
        "sha256": "f67496d0b72a388a3e0b974f951223e8912ff738227ed99d7f93dfda9aefe183",
        "source_batch_manifest": "manifests/batch-ood-1000034.json",
        "source_batch_manifest_sha256": "60cfce2c41e28ce6a44410c71a3b65f9aa41329f85bc4ed2ecfba5e9a7ab4ea4"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000040.jsonl",
        "records": 29,
        "sha256": "ba2fbef28c67738da96294f96c6249d7ed46ace6e179e82c7ec9aa5862a75673",
        "source_batch_manifest": "manifests/batch-ood-1000040.json",
        "source_batch_manifest_sha256": "838a4490a825bafdfcc2d315e5069294e220815600a78a9640c20e8856b341e9"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000046.jsonl",
        "records": 29,
        "sha256": "65766331f1570bcb5a7066768d95f021b92df60f2c408d657635409f36c445d6",
        "source_batch_manifest": "manifests/batch-ood-1000046.json",
        "source_batch_manifest_sha256": "926b94da7730a4897b9d3f50e87dd0b7abbf80918aceaedc49f697fe9ee90b6b"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000052.jsonl",
        "records": 29,
        "sha256": "0b5b3b10c4aec5f5b3f5ba2eb2d97f34b6a11590b99c33ae22b576f90e64b636",
        "source_batch_manifest": "manifests/batch-ood-1000052.json",
        "source_batch_manifest_sha256": "4a3004bf41d8e186a0fb3826e690b68555569bfa4f959e72f527adf8e9aad6f0"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000058.jsonl",
        "records": 29,
        "sha256": "76fb73ffd1778569b8c4a72b1b74f4c442d67bcf497963f74e03864b1470cc28",
        "source_batch_manifest": "manifests/batch-ood-1000058.json",
        "source_batch_manifest_sha256": "a8c1a66dc6f8b2b231e96b30f4d7ba76e1c86d55bcc6d4d22bf6426a9d33451a"
      },
      {
        "code_map_sha256": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
        "path": "data/accepted/ood-1000064.jsonl",
        "records": 29,
        "sha256": "068bae9885a03cf9b775b91b56f1e9f1e359a0b7366f091397a6f478ef7a5f0f",
        "source_batch_manifest": "manifests/batch-ood-1000064.json",
        "source_batch_manifest_sha256": "e4eeb50c96b0c1cdef44f0a12984701fa1e56f9411a84b1415a72f1075daa966"
      }
    ]
  }
}
```


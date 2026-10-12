# Central Rineya datasets

Use **research/rineya-high-quality-dataset-v1** for every future dataset or evidence backup. Open [the dataset index](datasets/CENTRAL_DATASET_INDEX.json) to choose a collection.

- [High-quality factory](datasets/rineya_high_quality_factory_v1/) contains the ten tracks. Its [active release catalog](datasets/rineya_high_quality_factory_v1/manifests/ACTIVE_RELEASE_CATALOG.json) selects approved releases; historical and pending records retain their states.
- [Goal-conditioned decision data](goal_conditioned_decision_data/) is a separate collection with [historical snapshots](datasets/_history/gcdl/).
- [RAKDC backup controls](datasets/rakdc_backups/) retain exact original snapshots and private GitHub Release asset recovery pointers. The original tags and assets are preserved.

The [source-head inventory](datasets/_consolidation/20261012/SOURCE_HEADS.json) and [object coverage map](datasets/_consolidation/20261012/SOURCE_OBJECT_COVERAGE.json.gz) locate the consolidated work. All captured dataset commits remain ancestors of the central branch. Existing commit/tree/blob recovery pins remain historical authority; branch removal does not authorize relaxed recovery checks. GCDL and RAKDC readers must complete their own reviewed migration before their old references are retired.

Append future releases and modest lossless evidence shards to this branch with a freshly verified expected parent and no force push. Verify private identity, complete remote bytes, hashes, modes and inventory before any local cleanup. Do not create per-run dataset branches, change main, silently rewrite accepted releases, merge the collections into an undifferentiated corpus, or promote pending records through consolidation.

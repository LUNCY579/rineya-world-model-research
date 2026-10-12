# Archived generation-profile compatibility

Implementation scope: unapproved proposal worktree only. The generator-v1 module, environment/oracle semantics, original active curriculum bytes, accepted records/manifests, protected-index authorities, and sealed gold remain untouched. No sealed labels were opened, no large data was published, and no model training/inference service was used. Production remains paused below the 10 GiB free-space floor. This report records a concrete compatibility implementation for fresh independent review; it does not approve the full v2 generator or promote it.

## Exact profile contract

`generators.profiles.resolve_profile(profile_id="v1", *, root=None)` resolves an operator-selected allowlisted profile and returns a fresh dictionary containing `id`, `schema_version`, `generator_version`, `environment_version`, `config_version`, `config_hash`, `config`, `config_path`, and `config_file_sha256`. The legacy API default remains v1. The worker may select v2 explicitly through its separate reviewed job configuration; it must pass the selected `config` to the appropriate static generator adapter for generation and deterministic rerun.

`resolve_record_profile(provenance, *, root=None)` matches exactly `(config_version, config_hash, generator_version, environment_version)`. It has no current-default fallback and derives no executable or filesystem path from record text. `root` is an operator/test override for a complete authenticated checkout, never an inference or provenance field. Returned configuration dictionaries are independent copies and there is no mutable resolution cache.

The registry file is pinned by its exact raw SHA-256 in trusted Python source. On every call, the resolver authenticates the registry bytes, checks its closed entry shape and exactly the known v1/v2 identities, and requires their flat known archive filenames. It then verifies the selected archive's raw file SHA-256 and canonical JSON digest/version. Both profiles use `gcdl-1.0` and `environments-1.0`. Unknown identities, tampered registry/archive bytes, duplicate JSON keys, nonfinite constants, path/version/schema changes, missing files, archive symlinks, and a symlinked config directory fail closed. The 1 MiB read bound applies to configuration files. Damage to an unselected v2 archive does not reinterpret or prevent resolution of an intact v1 archive.

Changing the registry or either allowed configuration requires a coordinated reviewed source/config change, including its pin. This deliberate strictness is not an automatic registry update or discovery mechanism. It prevents a modified config and a self-updated registry from silently authorizing themselves without a source change.

## V1 preservation and v2 normalization boundary

`configs/curriculum-v1.json` is an exact byte copy of the unchanged `configs/curriculum.json`. `auditors.integrity.validate_schema` now reconstructs each record's reserved template/sizes/lineage from its exact resolved archived profile rather than the active default. The stable ID formula, wire schema, utility definition, label checks, and v1 template requirements remain unchanged. V2 uses its separately declared template/lineage names with the same environment semantics and additionally validates declared namespace custody through `version2.validate_vocabulary`.

The auditor's fingerprint dispatcher also resolves the recorded profile. V1 checks its archived exact world-size reservation, then calls the original `build.semantic_fingerprint` unchanged. No v1 unknown-reference marker, normalized payload, version prefix, record identity, or protected-index value is rewritten.

V1 normalization preserves literal invalid references, such as `{'unknown_entry': 'absent'}`. V2 legitimately renames those references along with active namespaces, so raw v1 normalization would differ between familiar and reserved vocabulary presentations. Only the v2 branch first calls `version2.restore_source_names(record, config=resolved_config)`, then applies the unchanged v1 normalizer to a copied record with that restored input. The restorer verifies total/injective maps, permitted pools/strata, deterministic parent/presentation assignment, and the exact restored-source task hash. This allows semantically matching v1/v2 tasks to retain the same fingerprint rather than hiding cross-version duplicates behind a version prefix. Mathematical transition/utility code remains independent of the vocabulary mapping metadata. Exact profile-derived world-size checks now run before any factorial normalization for both profiles and again after v2 restoration. The reviewer's oversized-input spies confirm the original normalizer is never called for unreserved block/account/register counts; valid v1 fingerprints remain exact.

## Verified compatibility evidence

At 2026-10-12 01:14:40 UTC, after the final size/C/group gates were added, the full allowed live v1 corpus was read through explicit train/dev/ood shard globs: TRAIN 1,320 records in 44 shards, DEV 297 in 11, and OOD 319 in 11; total **1,936 records across 66 immutable shards**. Every row passed the archived-profile schema validator. Its recomputed ID matched the stored ID; its new auditor fingerprint matched both the unchanged v1 normalizer and its original immutable batch `protected_index` value. All mismatch counts were **zero**, and every source shard's bytes stayed identical before/after the audit. No sealed shard was opened.

The final audit elapsed time was 54.986 seconds under concurrent test load; profile/archive/integrity bytes stayed stable throughout. An earlier compatibility pass took 24.635 seconds. The SHA-256 of the ordered JSON list of `[relative_shard_path, sha256(shard_bytes)]` commitments was `108aef10776bce898f821ab58095e5e2895a20c02dda1baaf4cc52bebbe6b7c7`. This is compatibility evidence, not a new trained-model result or an authorization to reread sealed gold.

The focused `tests/test_profiles.py` suite contains **18 passing tests**. It verifies v1 byte-copy/default identity; exact four-field provenance selection; raw-versus-canonical hash distinction; fresh returned config copies; archive/registry tampering, symlink and path/version/schema rejection; preservation under an unrelated active default; unselected archive isolation; wrong templates/identities; and v2 unknown-reference restoration/forgery refusal. In the v2 positive fixture, both familiar/reserved surfaces restore the exact original v1 fingerprint, while raw v1 normalization of their different invalid-reference spellings differs. Removing the `absent` mapping remains rejected even after recomputing the record ID.

A combined profile/generation/adversarial run passed **54 tests** after the coordinator corrected legacy direct-database job recovery. That run included the initial 11 profile tests; the two subsequently added v2-dispatch tests and five final size/C/group tests passed in the final 18-test focused run. The coordinator corrected the legacy-recovery fixtures using canonical temporary paths and template guards. The final full integrated suite passed 189 tests in 75.309 seconds against the frozen 42-file source digest `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`; there are no pending worker-fixture failures. The official DEV measurement and independent promotion review remain separate from this mechanical suite receipt. The full integrated generator/worker/rendering suite and its final source freeze are owned by the coordinator and fresh reviewer. These component checks are not their approval receipt.

## V2 atomic record/group checks

The v2-only C record gate requires Blocksworld, full goal keys matching the initial forest, a goal different from the initial state, exactly six candidates, an integer `goal_index` in 0..2 (booleans excluded), and exact `goal_bundle_id` reconstructed independently from only environment, initial state, rules, constraints, ordered candidates, and utility definition. Goals, indices, record IDs, targets and audit metadata do not enter that identity. C metadata cannot be relabeled as another family to bypass this record gate. V1 metadata remains interpreted exactly as before.

`auditors.integrity.validate_v2_group(records, require_full_curriculum=False)` returns None on success. Every member must resolve to v2 and pass schema/vocabulary checks; mixed profiles and duplicate IDs reject. Every C bundle must contain exactly indices 0/1/2, one protected parent/split custody tuple, identical complete non-goal model inputs including ordered candidates and prose, three distinct goals, disjoint nonempty exact best sets, and a feasible successful best candidate for every goal. Matched vocabulary controls require both DEV family-D strata, consistent parent custody, exact restored-source equality and unchanged decision labels. With `require_full_curriculum=True`, the pinned A-H minima also apply, so missing C cannot escape by a family change.

The worker must run authoritative `audit_record` for every row before relying on best-set labels here. The helper supplies cross-record contrast/custody checks, not a replacement mathematical oracle. A row or group failure requires the coordinator's worker to quarantine the whole v2 batch; legacy v1 accepted-subset behavior stays separate. Standalone controlled triplets/vocabulary pairs use the default helper, while full generated batches must pass True. No incomplete pair/triplet may be promoted from its surviving members.

The final focused tests challenge missing/duplicate indices, wrong metadata, non-initial/full-goal/six-candidate requirements, family tampering, changed instruction after authentic namespace round-trip, overlapping/empty best sets, absent successful candidates, missing vocabulary partners, mixed profiles, A-H minima and the pre-normalization size bound. Synthetic helper tests use a known debug seed outside official pilot ranges and publish nothing.

## Boundaries and limits

This component implements profile resolution and validation dispatch only. It does not implement worker scheduling, authorize publishing below the resource floor, select promotion, or measure the pre-registered DEV goal/vocabulary pilot. Small synthetic test fixtures use seeds outside the declared DEV measurement ranges and are never accepted/published. The builder owns the v2 configuration/generator/restoration implementation; the coordinator owns explicit job profile integration. Unknown future environments/wire schemas need a separate review rather than a relaxed fallback.

An archival registry is trusted through reviewed source hashes, not a cryptographic signature from an external authority. All agents share a host/account, so procedural sealed isolation and correlated engineering limitations remain. Exact string holdouts and normalized duplicate detection remain bounded synthetic instruments, not proof of model generalization or arbitrary semantic equivalence.

## Component commitments

Registry raw SHA-256 pin: `f54978838729e85eb849724d2b2c512da65b9969b8bd4e9876aab7271cf9388a`.

- v1: config version `curriculum-1.0`, generator `generator-1.0`, canonical config hash `d98651723bc9469105627dc55d78f0217eea67b98428020a5b1552452b4ca2f2`, archive raw SHA-256 `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780`.
- v2: config version `curriculum-2.0`, generator `generator-2.0`, canonical config hash `1fa733236552daae035fc844be2b09381005d6343aa732483e19185f61ec92d6`, archive raw SHA-256 `90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd`.

Owned implementation/config/test bytes at this component freeze:

- `generators/profiles.py`: `fc83161135243a328804511d8950fc1e0327d1657676590f071ab496a0fbbe5b`
- `configs/curriculum-v1.json`: `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780`
- `configs/generation_profiles.json`: `f54978838729e85eb849724d2b2c512da65b9969b8bd4e9876aab7271cf9388a`
- `auditors/integrity.py`: `b6fb75a24bb27ac5f0000dd36d8e89968ad39975e21e103bda8658fe830424f9`
- `tests/test_profiles.py`: `d1758c6150b3fb2f2dd75b91571fb17a8c3a3c31d94e3a197fb83a8720a9538a`

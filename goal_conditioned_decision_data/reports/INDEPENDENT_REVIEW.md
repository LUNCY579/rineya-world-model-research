# Independent GCDL archival review

FINAL_VERDICT: PASS

Reviewed on 2026-10-11 by the independent archival reviewer. This is a manual, scoped approval of the exact source below, following source inspection and adversarial tests. An executable gate did not author this verdict.

The approved source digest is `3cc00cd2c73b37829c8488babe654296aba4df28fada67c459a3b8a2f7c58835` over the complete 54-file map at the end of this report. The reviewed proposal is `/Users/chengli/.codex/gcdl001-host/archive-improvement-worktree/goal_conditioned_decision_data`. Independent full validation passed **329 tests in 96.289 seconds**; the complete source maps before and after that run were exactly equal. The root agent independently reported **329 tests in 96.158 seconds** on the same digest. A further read-only source check matched the same map.

## Scope and independence

Approval covers the GCDL-only archive provider, custody index and bounded reads, archive-aware worker/rendering/release consumers, runtime snapshots, backup deletion guards, composed archive cycle, and replacement/cleanup of the owned Git store. It is limited to the explicit user-authorized GCDL paths and the qualified `LUNCY579/rineya-world-model-research` repository. It permits the reviewed workflow to remove eligible local payload copies only after the workflow establishes and rechecks exact remote custody. It does not confer approval on later source/configuration changes or bypass the current review, STOP, pause, profile, provenance, publication, or resource guards.

I did not implement the archival workflow. I authored `tests/test_archive_review.py` (62 tests) and this report, and adapted only storage retrieval and the named temporary-root fixture in `tests/test_v2_review.py` at the parent agent's request. Historical assertions and captured expected hashes in those tests remain unchanged. Other builders implemented the archive provider, integration, snapshot/backup/orchestration and compaction.

This review performed no production eviction, GitHub publication, production state mutation, model training, paid API operation, sealed production label read, or new official v2 experiment. Disposable fixtures use small manually specified payloads and an in-memory remote, including opaque sealed fixture bytes. Real Git compaction tests use disposable repositories. Operational completion of a live archival cohort must be demonstrated by that cohort's custody, backup, eviction and completed cleanup receipts; this source approval alone does not establish completion.

## Evidence and historical continuity

The full 329-test suite includes the original 189 regressions and 140 added archival tests: 62 independent review, 21 integration, 19 provider, 7 snapshot, 8 cycle, 11 actual Git compaction and 12 compaction refusal tests. The final run had zero failures/errors, with all source bytes held fixed.

The read-only historical v1 tests still verify the fixed 66 nonsealed accepted shards containing 1,936 rows. Their byte collection commitment remains `45cb9526e674ec8d2a59c9fa40e219890f40dc3961b48198fd3c9c9f8d294d02`; the captured nonsealed batch-receipt collection commitment remains `5cb4454f6d927e13b35914814981199771bf1c3c70f98ed61ce85f4bc65fbb07`. Record IDs, protected indices, mathematical outcomes, semantic fingerprints, exact v1 generator/config/output commitments and pointwise/pairwise/listwise rendering goldens remain checked. Retrieval now uses bounded `iter_paths/read_bytes` so those same commitments remain enforceable after cold eviction; it explicitly refuses historical sealed gold reads.

The earlier bounded production inventory established 2,152 v1 accepted records, including 216 sealed records, with the 149 existing shard/batch/sealed-freeze byte hashes unchanged and 66 pending plus 66 superseded rendered receipts. That inventory was metadata/hash only for sealed artifacts. The new integration tests preserve authoritative accepted counts, queued/completed profile and seed lineage, seen IDs/fingerprints, protected split reservations, batch/source receipts and recovery frontier across missing local payloads. They refuse lost batch custody, profile reinterpretation, missing unarchived bytes and replay conflicts. Metadata-only sealed cold recovery performs no blob read.

The v1 generator remains byte-identical at `78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575`; the v1 curriculum raw bytes remain `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780` (canonical config digest `d98651723bc9469105627dc55d78f0217eea67b98428020a5b1552452b4ca2f2`). Generator, environment, oracle, profile/curriculum and official experiment specification bytes were compared with the prior scientific freeze. Rendering changes concern archived storage/receipt access and retain the exact historical output goldens.

The prior v2 scientific freeze remains bound to its original 42-file source digest `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`. I rehashed the existing freeze/result bytes and did not rerun or tune the official DEV experiment. This archival approval preserves its required evidence exactly; it makes no new model-quality or generalization claim. The following canonical map is part of this fixed approval statement:

```json
{"manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

## Destructive-path findings

No unresolved blocker was found in the final source. The tests separately specify expected raw SHA-256, Git blob SHA-1, lengths, paths, object modes, commit trees and outcomes; success is not inferred from the implementation's own validator.

Remote custody binds each payload and catalogue to both the committed source tree and the durable snapshot tree, the full qualified snapshot reference, and exact raw bytes. Verification rejects absent/truncated/wrong trees, wrong object mode/OID/size/path, same-length byte corruption, moved references, corruption after prior verification and offline proof. The snapshot publisher now uses GitHub's create-reference POST, validates the exact returned reference/type/commit and confirms the server reference afterward. A concurrent actor creating an ancestor reference is refused rather than fast-forwarded. GitHub documents distinct create and update endpoints, so ordinary branch push semantics cannot supply the create-only guarantee. References themselves remain mutable by authorized remote writers; the reviewed workflow repeatedly checks their exact commit identity. [GitHub reference API](https://docs.github.com/en/rest/git/refs#create-a-reference)

Cold reads authenticate raw bytes and enforce caller/object limits before exposing data, retain only a bounded RAM cache, and do not restore payload files. Default sealed reads refuse before network access; explicit release inspection is a separate caller path. The provider uses raw Git blob media, which GitHub documents as supported by the blob API. [GitHub blob API](https://docs.github.com/en/rest/git/blobs?apiVersion=2022-11-28#get-a-blob)

Locale eviction rechecks the entire cohort remotely before deletion. Present files are bound to byte/stat identities and ordered parent-directory ancestry. Atomic, directory-FD-anchored holding moves permit deletion only of the verified moved inode; symlink, hardlink, inode, ancestor or final-check-to-move substitutions refuse. Interrupted evacuation has a durable custody journal; restoration never overwrites a new destination. Offline retry retains proof and local bytes. Partial progress has immutable custody for already absent files rather than silently treating them as verified anew.

The composed cycle shares the live worker lock, confirms payload/catalogue and custody/index backups before eviction, and freshly rechecks source approval, STOP and the exact original pause at destructive boundaries. A hidden holding copy of the last payload is recovered before any Git staging/backup, even when disk free exceeds the floor. Malformed/orphan journals refuse before authentication or backup. Intentional active-branch removals require absent-byte custody and a durable snapshot; unrelated or staged control deletions refuse before commit/authentication.

Owned Git compaction verifies published history, retained code/control/runtime bytes, runtime and worker-lock inodes, and cold blob OID absence with lazy fetching disabled. A shallow partial clone and sparse layout remove local dataset objects without changing the canonical worktree/state location. Dirty, foreign, untracked/ignored, symlink/hardlink, extra-worktree, wrong-branch/remote and unmirrored-reference cases refuse. Private transaction/pointer identities prevent fixed-name collisions. Pre-install failures roll back; after verified replacement is installed, incomplete old-store removal persists an explicit pending receipt and retries against the healthy replacement. Completion is bound to the exact current archive-index digest.

An operator pause changed during remote proof, payload movement or Git cleanup is retained and cannot be overwritten by subsequent status writes. Pending cleanup remains explicitly incomplete and can retry above 10 GiB. Only a pause created by this maintenance operation, or its own cleanup marker after verified completion, may clear; original disk/migration pauses remain. STOP and unrelated pause cases retain their no-op behavior. The automatic fresh cohort stays bounded to 12 files/128 MiB with a 280-second composed budget; the provider's separately bounded manual cohort is at most 128 files/128 MiB and each blob is at most 96 MiB. Metadata cleanup deferral requires an explicit manual CLI invocation.

## Operational limits

The retained controls include code/configuration, authoritative batch and sealed-freeze metadata, queue/runtime state, current review/validation, archive index/custody, STOP/PAUSED and the live lock. Eligible accepted/sealed/quarantined/demo/backup/rendered payloads and historical reports may become cold; active control reports and source files are excluded. Cleanup is confined to the owned GCDL host repository transaction and GCDL payload paths, not unrelated user or agent files.

The workflow trusts the local OS/UID and the authorized GitHub owner. A hostile process with the same UID or an authorized remote writer can alter state beyond application-level immutability; the reviewed ordinary worker/archive paths serialize under the shared lock and reject observed byte/reference/pause changes. Network failure, missing proof and unfinished cleanup fail closed. Storage approval does not assert that enough disk space will remain for publication; the original 10 GiB guards remain binding.

## Complete approved source map

Paths are relative to `goal_conditioned_decision_data`; values are SHA-256 of the exact bytes. The digest above is SHA-256 of this mapping in canonical sorted-key compact JSON.

```json
{
  "__init__.py": "64c91b2f24f24214f818c1716c5e181f86625b648914e1eadf6880e82114346e",
  "auditors/integrity.py": "b6fb75a24bb27ac5f0000dd36d8e89968ad39975e21e103bda8658fe830424f9",
  "auditors/release.py": "8457e67f8222bdce042a5e701b592e9b9d9e0e2b75d9cf9a372b83e31da780a5",
  "configs/curriculum-v1.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
  "configs/curriculum-v2.json": "90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd",
  "configs/curriculum.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
  "configs/generation_profiles.json": "f54978838729e85eb849724d2b2c512da65b9969b8bd4e9876aab7271cf9388a",
  "configs/production.json": "9fe42c98d1cc4a701337dff72f634c8123a056badc8fb9dcf1ad04df7011628b",
  "configs/v2_experiment.json": "19d88bd2da381806c481b88ade73dedf411ebcc07d154054b94d8733ed4b5b3a",
  "decision_labels/__init__.py": "c46518068c5e4407e3f5c1a43eed2df9a775977a055773c9263b21bdddf1e637",
  "decision_labels/rendering.py": "2b69c373c8f4077addf1aa44052eae6c6be0f5bec313d43c3b6e63de831ee9df",
  "decision_labels/views.py": "3a6851fc7ba836bd66ae6c8b4f1188821c28190c61e5681ac8e75d3bad02372e",
  "environments/__init__.py": "0097b9cfbd6e2cf98c050afad5e9f8f7058739f2095792b336339750af581bc4",
  "environments/core.py": "b4e383c455ba899737a8c034a239a23ac01fbc13279791b3e0bc66acb6ec014d",
  "generators/__init__.py": "d45c107da508d34b0d040fcfe634a6f16fe2db2218244a5dfee49fa008742e11",
  "generators/build.py": "78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575",
  "generators/improvement.py": "4c4ad9b85050fa1b21a31bf71b752e217ce2661fafcf0bdd934463c01571ed0a",
  "generators/profiles.py": "fc83161135243a328804511d8950fc1e0327d1657676590f071ab496a0fbbe5b",
  "generators/v2_experiment.py": "8771bc1cacaac80e431444fdeed22933f5bc4205891c4270976e4b1c1b1e6218",
  "generators/version2.py": "5c6cb47f947eaf4a414d8baf215d600cc3c62ceed9fa55416e0eadf38e0963d4",
  "oracles/__init__.py": "d2ff07ef6ae530121d8484a744a81e3c9c12755e01a350f533e047240cc06397",
  "oracles/independent.py": "0453985a3347ea9e503c44dff4eef9ad7f7de1a9efa88537c05d569c96a032ae",
  "oracles/primary.py": "0797623f90d1d53de1034c0c8fb07fbfc56207924b829f9945729dcd52639d19",
  "pipelines/common.py": "26eafc2cccefb424c8aff89b21cbb6cf81e73a67474c97b9749e02b942358b82",
  "pipelines/remote_archive.py": "886c0ec958d49198fcc798ae5b748a6334795f6cc8e5c4c8b4d934cae006410e",
  "pipelines/rendered.py": "e03e35731794c0c2a7e766f1acfbab8643a426b8ac15455243b50d01acb556c3",
  "pipelines/worker.py": "756c21b6ebf459af2fca99002c25cda4e931547dc20698d54b2074f7f951fae0",
  "scripts/archive.py": "6aa490a52721e09cd1c4faf9144f7fbd21fb88aa966d356af07d71f99ea56abd",
  "scripts/archive_cycle.py": "7ddb94cfd2af9f7891a301d37da8b614d32df97fed0646dd7bd9ac175bb1ef01",
  "scripts/backup.py": "6797b3d6892a39db0c28effe35601d1020807e10bdf7cefb7d9d5aeb24944816",
  "scripts/compact_git.py": "95b231416f5e299d7d4cfd7b2af654abbbe490d5558ca9ffe5fd0ba1f529c90b",
  "scripts/control.py": "9990ad7a7538dc9ce8fdd10d3db26f9a592f56f04d74ef6138f7f5054f8ea909",
  "scripts/export.py": "9137f4303865a2cb93957a2a7a28b1232c4fa93aa342369b2507b495b49b51d5",
  "scripts/export_training.py": "0aefb65fd8cccc4d3e8086a437db9e8a51c6b65abb96f9803fc9b9a9588fc33f",
  "scripts/run_cycle.py": "39e2f9e5096b87b7c856ea555bb5f07b31f352b1abddba1891ae60f37da9073c",
  "scripts/snapshot_state.py": "918ea3f7e2c35c86d0413c4826af1a4b01215ec5efc26dfefac490fcf346d140",
  "tests/test_adversarial_review.py": "286dc2fe63b81c0bb4f76b59cea84a7c7cdc8ba57a5ef0501c9a2f48fcd3bf8e",
  "tests/test_archive_cycle.py": "aedcbb0e31da9e86087601c9bdc2d9a5e96d8b60e5564cd6383f76d45feac379",
  "tests/test_archive_integration.py": "b1c7130e61bf0766b20ab87006e387dcbecfa21a323a17969976b87e18155673",
  "tests/test_archive_operations.py": "a5f039a5d27a2b6c6629d91d7e00761146fd76bb986d9752a88f900357c6fc66",
  "tests/test_archive_review.py": "7428ee10cf4934159736908793e915bc45df74d47f278a19705ebf6e6e2fa0c7",
  "tests/test_compact_git.py": "a5952e844f732c13c30891f45a9815b4c9a13952579ccfbcbf2a2c39395dc7e3",
  "tests/test_compact_guards.py": "f31918cb969864b7ab5339ed75c3b4a168276fba2957f8cd0c8e548e9b41aec0",
  "tests/test_experiment_custody.py": "c037c686fb54d9aefd73e594ec4588d1e96d6728396761fd067f5bb982b39141",
  "tests/test_generation.py": "0dbc041d7cbf52d9dd6bc83a7e51efe4e4fc19e6bbbcc00e964affdc436a7341",
  "tests/test_operations.py": "42a45f67c00147f160209861ffd1f65c357853e07066b45240fd81a127eadc42",
  "tests/test_oracles.py": "e1995c834809e7a0ec45531465cba382408c4e9dfe0d309f72d9c469db340ff1",
  "tests/test_profiles.py": "d1758c6150b3fb2f2dd75b91571fb17a8c3a3c31d94e3a197fb83a8720a9538a",
  "tests/test_remote_archive.py": "e5af92bfce0d7a85d763e0899dc2fc336780bb132322868a4589e6f96066c4ca",
  "tests/test_rendered_pipeline.py": "4cb9642415f5f495fcfe33629330fd50e67e1ea8e12f24eb4a39c5c983954b42",
  "tests/test_rendering.py": "31a0335a99e7f59b30fa1fc854b473cfccddf02c0747a4ee603b6016931fb5a4",
  "tests/test_surface_review.py": "c6dbc104a66122ad4a0c7255afb57c423fb7add32a988d9763cde8db05f3fe03",
  "tests/test_v2_review.py": "adb97f1e6ab879609a13492f346fdbb6afe7e4d11767b3e8f7ffeeb75268e1b0",
  "tests/test_version2.py": "5c58b03e8518932ca3536a212ff0deb4ff18c0ca100bb9f14b54745e934698be"
}
```


# Independent CODE10 Git-store custody review

FINAL_VERDICT: PASS

This is a fresh, scoped approval of the exact 68 Python/configuration files below and the separately hashed archive-policy change. Both complete 476-test runs passed with identical source maps before and after execution. I independently reviewed the selector, catalogue history, default full-byte consumers, object-inventory races, pending cleanup, ownership, alias and pause boundaries; challenged two initially incomplete deletion sets; and verified the fixed evidence and original production commitments.

Candidate: /Users/chengli/.codex/gcdl001-host/store-proof-proposal-worktree/goal_conditioned_decision_data.

Approved canonical source-map SHA256: 866bb13ee7f3df468a1718c81a15ee3d06788b9fd40d3e41eba4a920846c69ab.

ARCHIVE_POLICY.md: 7,259 bytes; SHA256 52cd074feb13d9b86a01be8164e51f3beed963824c92d28bae5f74633d60b36e. Markdown policy is outside the 68-file Python/configuration map and must be copied and byte-verified separately during promotion.

Approval is based on the exact implementation and evidence, not inferred from a preceding PASS. No production eviction, store replacement, code promotion, scientific pilot or training was performed by this reviewer. Actual CODE10 deployment and GitHub compaction timing are not claimed here. The reviewed resource, STOP, source/evidence approval, pause, backup and custody gates remain required.

## Problem and exact scope

The live CODE9 system had already evicted the local raw payload files but still retained archived blobs in the owned Git object database. A fresh complete raw reproof of all 366 historical indexed payloads was reported to exceed the existing 180-second compactor bound before store mutation. Repeating raw downloads for payload blobs absent from the owned store did not correspond to another removal of those local bytes.

CODE10 permits only the owned Git-store compactor to select raw blobs that are actually present in the store it will replace. Every immutable catalogue reference, complete raw catalogue and every payload entry in both snapshot and original source trees is still freshly proved. Full raw byte verification remains mandatory for every selected present archived blob. Already absent historical payload bytes are not removed again.

The exact diff from live CODE9 consists of pipelines/remote_archive.py, scripts/compact_git.py, the new tests/test_store_proof_review.py, and the additional paragraph in ARCHIVE_POLICY.md. No other existing Python/configuration file changes or disappears. The Python/configuration source count grows from 67 to 68 because of the one new test file. Scientific generators, oracles, environments, profiles, configurations, label renderers, H/v2 evidence and producer behavior remain byte-identical to CODE9.

The candidate source inventory is reports/CODE10_CANDIDATE_SOURCE.json, 7,063 bytes, SHA256 e112df6cd26697bfff7d2016c4478617018a8bce091ce2316595f1c9a0880017. The complete approved map is included below rather than inferred from this diff.

A separate implementation author owns the two Python changes and policy text. A separate independent tester authored the new adversarial tests. I authored no implementation or test change; I reviewed the actual source and fixtures, identified missing deletion-set cases before freeze, and ran a separate full suite. Only this final review artifact was written by this reviewer.

## Proof selector and unchanged full-byte consumers

The private _remote_proof helper accepts a keyword-only required_blob_ids selector. A supplied selector must be an exact set/frozenset of lowercase 40-character hexadecimal OIDs, with every OID covered by that validated catalogue. Unsupported container/value types, malformed IDs and uncovered IDs are refused before remote requests. The compactor supplies the exact per-catalogue intersection with the captured present archived-object set.

The default None mode retains the original complete raw proof. Public verify_archive and evict_verified do not expose or supply a selective mode: they continue to verify the complete returned bytes of every payload in the cohort before granting custody or unlinking local files. Their default behavior is explicitly tested after prior successful verification and with either old or new remote bytes corrupted. The selector does not create partial archive custody, change receipt/index schemas, alter the eviction allowlist or authorize a metadata-only unlink.

In either mode, _remote_proof confirms the qualified snapshot ref, fetches/checks complete canonical catalogue bytes, proves every snapshot tree entry, proves every original source tree entry, and confirms the snapshot ref again. Tree checks retain path, blob type, allowed regular-file mode, exact OID and exact integer byte size. Raw selected paths additionally require exact length, SHA256 and Git-blob SHA1. A selected OID used by multiple catalogue paths still receives the path-specific byte/commitment checks; there is no first-path-only exemption.

An empty selector therefore authenticates the full catalogue and all source/snapshot metadata while fetching no payload raw bodies. It can be used by the compactor only when the captured owned-store inventory contains none of that catalogue's archived objects. Missing/moved refs, catalogue corruption and changed unselected tree entries still refuse the operation. It does not rehash all historical raw payloads at a fresh replacement when their local blobs are already absent; explicit full archive verification, eviction and audits remain the paths for complete raw rechecking.

The return shape stays the existing descriptor. Its selective use is confined to the reviewed compactor call site and bounded by the exact approved source. It is not a new externally configurable deletion authority.

## Complete historical deletion set and object races

The deletion cold-ID universe is the union of latest index-file OIDs and every OID in all validated immutable catalogue histories. It is not restricted to the current path index. This matters when an eligible historical report path has been reused: the latest index can name a newer version while an older catalogued blob still occupies the local Git store.

The universe is used consistently for current-HEAD alias refusal, present-object selection, repeated object-inventory comparisons, replacement/installed-store absence checks and removal counters. An archived blob required by active HEAD under any code/configuration/control path refuses before replacement, including older archived versions absent from the latest path index. No reference or byte alias is silently discarded.

Object enumeration uses the actual owned common Git store with lazy fetching/network materialization disabled. The original branch, exact origin mirror, clean owned primary/worktree layout, confirmed backup, remote owner, allowed worktree set and indexed custody gates remain intact. The selected set is captured from the actual existing objects, not from a declared count or guessed file size.

The compactor re-enumerates archived objects after remote proof/authentication and again immediately before the first store rename. Any change to the captured present cold set refuses before mutation. This prevents a historical archived OID materialized during proof/preparation from being deleted without its required raw reproof. The clone and installed store must contain none of the complete historical cold universe. The existing source/runtime bytes, live worktree/state/lock inodes, pause bytes, branch identity, rollback checks and durable committed-cleanup behavior remain enforced.

Two initially incomplete cases were raised during independent design review. First, selection from only current index.files would skip older catalogue versions of a reused path. The final implementation includes the validated historical union and tests both successful removal/counting and refusal on old-version corruption or active-HEAD alias. Second, active-store inventory alone cannot prove a retired store left after replacement; the pending path uses the full proof described next. Both corrections landed before source freeze without weakening existing assertions.

## Pending retired-store cleanup

The read-only _pending_cleanup_record helper first validates the existing journal's repository/programme, original HEAD, archive-index digest, exact managed transaction name, transaction identity, allowed children and any still-present retired/worktree identities. It makes no deletion while deciding the proof mode.

Any validated compacted_cleanup_pending record retains full raw proof for all catalogue payloads. It never treats the empty healthy replacement inventory as evidence that old retired blobs are absent. This remains true when partial cleanup has removed retired .git/HEAD or other metadata. Corrupt/missing/offline raw proof preserves remaining retired bytes and the healthy active store.

After proof, the validated pending record must still equal the initially captured record before cleanup. A newly introduced or changed pending journal cannot turn a selective proof into permission to delete a retired store. Object inventory is also checked again. The original durable receipt, no-rollback-after-partial-cleanup rule, identity guards and per-removal pause/deadline checks remain in place.

This is deliberately a conservative fallback. Pending cleanup does not receive the fresh-swap download reduction. On a large historical archive, its complete proof may still exceed the existing 180-second bound and leave cleanup safely incomplete. This review does not guarantee fast resumed cleanup or authorize a weaker proof, longer timeout, inferred empty retired store or journal rewrite. A future bounded retired-OID proof design would require separate review.

## Independent adversarial and complete regressions

| Validation | Result | Duration |
| --- | --- | --- |
| Root complete frozen suite | 476 passed; 0 failures, 0 errors, 0 skipped | 366.959 seconds |
| Independent complete frozen suite | 476 passed; 0 failures, 0 errors, 0 skipped | 372.079 seconds |
| New independently authored focused suite | 27 passed | 9.171 seconds |
| Original focused archive/store tests | 42 passed | 12.941 seconds |

The new test file is 32,979 bytes, SHA256 68c5266672ca94f528b90fbec5ad4541a78ea7555ad037571b4f6805d354a44c. It uses independently computed canonical JSON/SHA256/Git-blob expectations, opaque hand-specified bytes, an in-memory remote and disposable actual local Git repositories/bare snapshots. It uses no production data, credentials, paid API, real GitHub transport, model or scientific experiment.

The 27 cases cover default/public full raw proof, selected same-length/short/overlong/hash corruption, exact per-catalogue selection, malformed/unknown selectors, empty selector, moved refs, raw catalogue corruption, missing/type/mode/OID/size changes on unselected source and snapshot trees, actual shared-store object removal, historical reused-path versions, old-version HEAD alias, original store preservation on transport/corruption, late cold-object appearance, pending retired-only corruption, partial retired metadata loss, malformed/replaced pending transactions and a late valid journal introduced after selective proof.

The complete 476 count is the unchanged previous 449 cases plus these 27. Every original assertion executes; no case is removed, skipped, replaced or weakened. The final source map was compared before and after both complete runs, after evidence inspection, and before this report was frozen. The suite also retains the previous standalone import regression, archive eviction races/recovery, sparse Git cleanup/rollback, immutable snapshot creation, profile/batch replay, metadata-only sealed recovery, deadline/source/pause refusal, render-control qualification, goal-only rendered comparisons and historical scientific/custody checks.

The root result reports/CODE10_ROOT_SUITE.json is 640 bytes, SHA256 60ec6e8432b2327f27cd47cadac12d2722c37ac6041ad77a90b2c831d99802a0. The root harness SHA256 is 769ac42899c5cb559a9a68c7a1293c8ad65c4f30763dd0f8c2f7dc2d8ba4f7ff and log SHA256 2affdd011d0747c839a1a7d7bab59be208588a970201de9ba41b741cf42c29ac. Their recorded counts, zero failure/error/skip fields and before/after map match the final outcome.

The independent runner was separate, passed through stdin, and left no runner file in the proposal. Both complete runs used the declared public RAM read-session accommodation only for these two unchanged production fixture methods:

- IndependentV1Compatibility.test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts
- IndependentV1Compatibility.test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints

A replacement session opens after 60 seconds only between completed historical reads. Each run used five sessions for the all-corpus method and one for the three-batch method. Every original row, byte hash, ID, candidate, label, fingerprint and receipt assertion executes. No other test or temporary root is wrapped. No source edit, disk cache, production global deadline change or scientific expectation adjustment is made by this harness.

## Fresh production preservation and observed inventory

I reread the archived CODE8_IMMUTABLE_BEFORE control through bounded authenticated read-through into RAM. Its raw SHA256 remains 2f2360e1b340d2b9deaf2a454ece5ebe4eed1c8080e52ba7d558f67296d975c0 and its 173-entry canonical map digest bc386b68aed8b32b82870b244d88a894cf8f8ae271c4ccfbb57bf1865a0fbad2.

Every entry matched current production: 86 canonical shard commitments through public metadata, 86 exact retained raw batch-control hashes, and the SEALED_FREEZE control hash. All eight sealed shards were absent locally, and a guard explicitly forbade data/sealed/ local-byte-reader calls; network subprocesses were forbidden during this commitment comparison. No production sealed payload was opened, fetched or decoded. This proves stored commitment preservation, not a new sealed-target evaluation.

Current metadata-only totals are 2,904 accepted records in 98 canonical groups: v1 2,152/74 groups and v2 752/24 groups, with zero quarantine. Accepted split counts are train 1,832, dev 413, ood 443, sealed 216. The 32 completed render groups contain 3,984 derived rows: train listwise/computation/distillation 1,872/300/360; dev 510/90/108; ood 546/90/108. These surfaces repeat canonical worlds and are not extra independent worlds. No current row payload was read for this counting step.

The current archive index is 366 paths across 13 catalogues, digest 6a8f9d086d7fdc94c6163b63b72030d765d09b655e1c0e61fa2e21d1583fba2c. A separate read-only, no-lazy-fetch Git metadata inventory found 70 present archived OIDs; their unique raw lengths sum to 15,747,931 bytes. None of those present OIDs belongs to a sealed path. All current catalogue histories have 366 entries and 363 distinct archived OIDs, with no historical-only present OID in this particular live inventory. The defensive historical-union tests remain necessary even though the current live sample has no reused-path version difference.

These are metadata observations, not a download or execution benchmark of CODE10. A raw OID can require multiple path/catalogue checks, and the complete catalogue/tree/ref proof still runs. Actual compaction may fail its existing bound and preserve the store; a successful production cleanup is not asserted before deployment.

All scientific files match CODE9; this review's historical compatibility method again checked all 1,936 original nonsealed v1 rows against their captured commitments. It does not newly audit every current nonsealed row or infer a generalization result from the increased operational corpus counts.

## Unchanged gates and scientific scope

CODE10 introduces no new archive authorization, deletion path outside the owned Git store, filesystem allowlist, sealed reservation, data generator, label logic, training operation or external publication target. Canonical/raw eviction retains complete fresh raw proof. Only fresh owned-store replacement may avoid downloading already absent archived objects.

The programme ceiling stays 512 MiB, accepted/rendered publication floor 10 GiB, processing/compaction bound 180 seconds, rendering allowance 144 seconds and watchdog 340 seconds. The render-control limits remain 16 KiB per capsule, 8 MiB aggregate, 32 MiB total controls, two 8 MiB RAM proof-cache units, four new receipts per activation and the original metadata reserve. The ordinary owner/lock/OS trust boundary is unchanged; this review does not claim resistance to arbitrary same-UID code/memory tampering outside that protocol.

The H prototype remains inactive and DEV-only. Its once-only 800-case measurement belongs to the original 61-file source freeze 4a4c0c9cbe233dba9a9857e5d70963dfd04e4b436df9448be14493afb896cd3e. H module SHA256 remains 18751b15853905d79ed644c6ced6ce4e21a46300a189defc54465b564d3621f0 and spec SHA256 dc2a17a16860c5087ca3a8a583bd7fd3a3c6abfb7c8b08ebc697252102ce7ea9. This later operations map does not retrospectively become its freeze.

The fixed H result retains zero mismatches in 800 DEV cases (96 goal, 128 constraint, 544 atom ablation, 32 baseline), 96 three-step goal winners, all 544 changed nonempty exact ablation best sets, 32 complete-conjunction Q/Q-reverse ties and baseline success 32/32. Its recorded simple blind-positive-add membership is 7/96, highest/longest shortcut 32/96, and constraint-blind/terminal-only membership 64/128. Full baseline generation of 928 rows with 896 unused remains separate from the 800 evaluated-case denominator.

Developmental index overlap, seed-driven identity/order, literal a/b/c vocabulary, register/reject semantics, formula-aware shortcuts and supplied-candidate optimality remain the stated limits. No learned held-out generalization, transfer, global-plan optimum, production H profile, model training or sealed scientific result is claimed.

The v2 confirmation retains its original 42-file source freeze fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467, exact evidence and prior accounting/interpretation. Neither H nor v2 was rerun here. Their custody and replay regressions remain part of the full suite.

## Exact scientific evidence commitments

All five required evidence files were reread through the bounded interface and matched exact raw SHA256. Byte lengths are H freeze 25,929; H STARTED 251; v2 freeze 6,375; H result 805,131; v2 result 112,806. The sole literal required_evidence map authorized by this report is:

```json
{"manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json":"3b22f95b23189bc6fa92439559c67259d6d6eef916a8a9546c28ed12aa98101c","manifests/H_COMPOSITION_EXPERIMENT_STARTED.json":"45722e9caa95880ed6cb26b0849a75c1c6da0d758a25e42e951275d4d38940ab","manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json":"4e4ec400a1aba413ce0112e7b43cd8bfd220a3fd5bbbbede70b08f1a031f27ca","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

Missing or changed evidence, reviewed source bytes or active review invalidates this approval under the existing source/evidence guard. The separately pinned policy bytes must also match the approved promotion. Historical catalogue/index/custody records remain separately bound; they cannot broaden this statement or turn selective store proof into partial file-eviction custody.

## Complete approved source map

Python/configuration count: 68.

Canonical map SHA256: 866bb13ee7f3df468a1718c81a15ee3d06788b9fd40d3e41eba4a920846c69ab.

This complete map, the fixed review bytes, the five-path evidence map and the separately hashed policy define the approval scope. Operational result artifacts are not silently added to the source-map digest.

```json
{
  "__init__.py": "64c91b2f24f24214f818c1716c5e181f86625b648914e1eadf6880e82114346e",
  "auditors/integrity.py": "b6fb75a24bb27ac5f0000dd36d8e89968ad39975e21e103bda8658fe830424f9",
  "auditors/release.py": "5c02cc34b68aacc9f87337b3194eb3d651890a4aa31218fc4a43fa54ed940be9",
  "configs/curriculum-v1.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
  "configs/curriculum-v2.json": "90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd",
  "configs/curriculum.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
  "configs/generation_profiles.json": "f54978838729e85eb849724d2b2c512da65b9969b8bd4e9876aab7271cf9388a",
  "configs/h_composition_dev_experiment.json": "dc2a17a16860c5087ca3a8a583bd7fd3a3c6abfb7c8b08ebc697252102ce7ea9",
  "configs/production.json": "9fe42c98d1cc4a701337dff72f634c8123a056badc8fb9dcf1ad04df7011628b",
  "configs/v2_experiment.json": "19d88bd2da381806c481b88ade73dedf411ebcc07d154054b94d8733ed4b5b3a",
  "decision_labels/__init__.py": "c46518068c5e4407e3f5c1a43eed2df9a775977a055773c9263b21bdddf1e637",
  "decision_labels/rendering.py": "3a7298783781664bdde2f8ed2c8dda7ce197c8d5748ff93633d65e4dc72cdbec",
  "decision_labels/views.py": "3a6851fc7ba836bd66ae6c8b4f1188821c28190c61e5681ac8e75d3bad02372e",
  "environments/__init__.py": "0097b9cfbd6e2cf98c050afad5e9f8f7058739f2095792b336339750af581bc4",
  "environments/core.py": "b4e383c455ba899737a8c034a239a23ac01fbc13279791b3e0bc66acb6ec014d",
  "generators/__init__.py": "d45c107da508d34b0d040fcfe634a6f16fe2db2218244a5dfee49fa008742e11",
  "generators/build.py": "78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575",
  "generators/h_composition_experiment.py": "18751b15853905d79ed644c6ced6ce4e21a46300a189defc54465b564d3621f0",
  "generators/improvement.py": "4c4ad9b85050fa1b21a31bf71b752e217ce2661fafcf0bdd934463c01571ed0a",
  "generators/profiles.py": "fc83161135243a328804511d8950fc1e0327d1657676590f071ab496a0fbbe5b",
  "generators/v2_experiment.py": "8771bc1cacaac80e431444fdeed22933f5bc4205891c4270976e4b1c1b1e6218",
  "generators/version2.py": "5c6cb47f947eaf4a414d8baf215d600cc3c62ceed9fa55416e0eadf38e0963d4",
  "oracles/__init__.py": "d2ff07ef6ae530121d8484a744a81e3c9c12755e01a350f533e047240cc06397",
  "oracles/independent.py": "0453985a3347ea9e503c44dff4eef9ad7f7de1a9efa88537c05d569c96a032ae",
  "oracles/primary.py": "0797623f90d1d53de1034c0c8fb07fbfc56207924b829f9945729dcd52639d19",
  "pipelines/common.py": "26eafc2cccefb424c8aff89b21cbb6cf81e73a67474c97b9749e02b942358b82",
  "pipelines/remote_archive.py": "ff31136f45a22a90d2ceb138f0e9d63d7fc94bfec137d4244d592ad27c4ba7c2",
  "pipelines/render_controls.py": "069d1228ab0213e669b27ebc5130e529594c1d1a08b36f43c9f387e79298b001",
  "pipelines/rendered.py": "ae1e322fa40f7917ebb0f15eb01e2096ba5777c4174dfbb1c3b7f29ae5239014",
  "pipelines/worker.py": "143056de86e910a62148ff534dac4f6545f83c4edb147d35f26c78149510764f",
  "scripts/archive.py": "6aa490a52721e09cd1c4faf9144f7fbd21fb88aa966d356af07d71f99ea56abd",
  "scripts/archive_cycle.py": "155cd58e8316cf8f47c68cde92c1ba4f18ab057c8ae6cc05bcb8c1143771e330",
  "scripts/backup.py": "ab3155c896b10475bf4ebc58be7fc8fc00f70796d397f11b0111707ca14931c8",
  "scripts/compact_git.py": "c7d4710a4029e60fb8436bb1bfd3582ddf7c9e328980244e48300df47785f092",
  "scripts/control.py": "9990ad7a7538dc9ce8fdd10d3db26f9a592f56f04d74ef6138f7f5054f8ea909",
  "scripts/export.py": "9137f4303865a2cb93957a2a7a28b1232c4fa93aa342369b2507b495b49b51d5",
  "scripts/export_training.py": "0aefb65fd8cccc4d3e8086a437db9e8a51c6b65abb96f9803fc9b9a9588fc33f",
  "scripts/render_controls.py": "6b916baeef73cbca1e1718d15f46cd9e8578763d7c7bce5559763b65ddd9149d",
  "scripts/run_cycle.py": "39e2f9e5096b87b7c856ea555bb5f07b31f352b1abddba1891ae60f37da9073c",
  "scripts/snapshot_state.py": "918ea3f7e2c35c86d0413c4826af1a4b01215ec5efc26dfefac490fcf346d140",
  "tests/test_adversarial_review.py": "286dc2fe63b81c0bb4f76b59cea84a7c7cdc8ba57a5ef0501c9a2f48fcd3bf8e",
  "tests/test_archive_cycle.py": "aedcbb0e31da9e86087601c9bdc2d9a5e96d8b60e5564cd6383f76d45feac379",
  "tests/test_archive_integration.py": "b1c7130e61bf0766b20ab87006e387dcbecfa21a323a17969976b87e18155673",
  "tests/test_archive_operations.py": "a5f039a5d27a2b6c6629d91d7e00761146fd76bb986d9752a88f900357c6fc66",
  "tests/test_archive_read_session.py": "477d2b1ccf515594a4380c540b69db730ee98ff5df0ce0a4c6a8261feab2bdc2",
  "tests/test_archive_review.py": "7428ee10cf4934159736908793e915bc45df74d47f278a19705ebf6e6e2fa0c7",
  "tests/test_archive_standalone_review.py": "6215e0fe206726782a9eef98be01d5966ae4675127c9ef365b5ae28afbbb1266",
  "tests/test_compact_git.py": "a5952e844f732c13c30891f45a9815b4c9a13952579ccfbcbf2a2c39395dc7e3",
  "tests/test_compact_guards.py": "f31918cb969864b7ab5339ed75c3b4a168276fba2957f8cd0c8e548e9b41aec0",
  "tests/test_cycle_budget.py": "1a5c39854b52e5976caf1d654d4ac2df869a2d9d84ce8f544ba7c23b643917fe",
  "tests/test_experiment_custody.py": "c037c686fb54d9aefd73e594ec4588d1e96d6728396761fd067f5bb982b39141",
  "tests/test_generation.py": "0dbc041d7cbf52d9dd6bc83a7e51efe4e4fc19e6bbbcc00e964affdc436a7341",
  "tests/test_h_composition_experiment.py": "78c514c7d6b7e4eda8c677881d9e43b8d2cdccfa8a540939408e006c534eda34",
  "tests/test_h_composition_review.py": "e9d1f9de1e0d75b387cacca91511a04af15f370d6cbef842cdff97cac51bca7f",
  "tests/test_operations.py": "42a45f67c00147f160209861ffd1f65c357853e07066b45240fd81a127eadc42",
  "tests/test_oracles.py": "e1995c834809e7a0ec45531465cba382408c4e9dfe0d309f72d9c469db340ff1",
  "tests/test_profiles.py": "d1758c6150b3fb2f2dd75b91571fb17a8c3a3c31d94e3a197fb83a8720a9538a",
  "tests/test_remote_archive.py": "e5af92bfce0d7a85d763e0899dc2fc336780bb132322868a4589e6f96066c4ca",
  "tests/test_render_controls.py": "22c5c4b2b9ec2e1bacb0ccf6ded2dc5c1215b140dd495d2ca91fc4f1e6d755f2",
  "tests/test_rendered_control_lifecycle.py": "aea9b37613da4d95e22aed5d274213904e489c231b06b5ef0d1e402dc7bfca21",
  "tests/test_rendered_metadata_review.py": "2092d0b6505c6af62e52aca7080a9f9b7b9f6761273b35afa55938c92a862437",
  "tests/test_rendered_pipeline.py": "4cb9642415f5f495fcfe33629330fd50e67e1ea8e12f24eb4a39c5c983954b42",
  "tests/test_rendering.py": "31a0335a99e7f59b30fa1fc854b473cfccddf02c0747a4ee603b6016931fb5a4",
  "tests/test_runtime_session_review.py": "3a0ff8514b8c77052b53d3be1b772d9eac0f89dffdaa407074bccdf49522f65b",
  "tests/test_store_proof_review.py": "68c5266672ca94f528b90fbec5ad4541a78ea7555ad037571b4f6805d354a44c",
  "tests/test_surface_review.py": "c6dbc104a66122ad4a0c7255afb57c423fb7add32a988d9763cde8db05f3fe03",
  "tests/test_v2_review.py": "adb97f1e6ab879609a13492f346fdbb6afe7e4d11767b3e8f7ffeeb75268e1b0",
  "tests/test_version2.py": "5c58b03e8518932ca3536a212ff0deb4ff18c0ca100bb9f14b54745e934698be"
}
```


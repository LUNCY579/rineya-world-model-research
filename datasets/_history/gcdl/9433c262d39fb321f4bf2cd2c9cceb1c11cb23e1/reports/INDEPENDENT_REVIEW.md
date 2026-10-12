# Independent CODE8 review

FINAL_VERDICT: PASS

This is a fresh, scoped approval of the exact 66 Python/configuration files listed below, including the operational render-control projection publisher, runtime validation, and archive lifecycle. Both complete 447-test runs passed with the exact source map unchanged. I independently inspected the custody, derivation, publication, cache, pause, source-approval, deadline, sealed-admission, and storage boundaries. Approval is based on these checks and the fixed evidence bytes, not on the presence of a previous PASS marker.

Candidate reviewed: /Users/chengli/.codex/gcdl001-host/runtime-proposal-worktree/goal_conditioned_decision_data. Production was read only during this review. The approval does not assert that CODE8 has already been deployed, that the 24 actual production control groups have been migrated, or that production throughput has been measured after deployment. Those actions remain subject to the reviewed locking, source/evidence approval, publication, byte-preservation and resource gates.

## Exact validation and independence

| Validation | Result | Duration |
| --- | --- | --- |
| Root complete frozen suite | 447 passed; 0 failures, 0 errors, 0 skipped | 345.981 seconds |
| Independent complete frozen suite | 447 passed; 0 failures, 0 errors, 0 skipped | 343.526 seconds |
| Independent wrapper elapsed time | Exact before/after map equal | 343.533 seconds |
| Independent focused projection review | 18 passed | 3.070 seconds |

The full count consists of the unchanged prior 399 tests, 14 author projection tests, 18 independent projection tests, and 16 independently authored lifecycle tests. No original assertion or test was removed, skipped, relaxed or replaced. The independent test file is tests/test_rendered_metadata_review.py, 24,344 bytes, SHA256 2092d0b6505c6af62e52aca7080a9f9b7b9f6761273b35afa55938c92a862437. I did not implement the projection module, CLI, rendered integration, or archive publisher.

The root result is reports/CODE8_ROOT_SUITE.json, 639 bytes, SHA256 ebcca2a8efe069ed8226efdea8221acc1206a178f6842cdfdcfb4efcbca355a8. Its recorded before/after digests, count and zero failure/error/skip fields match the root's final report. The declared root harness is reports/CODE8_ROOT_HARNESS.txt, SHA256 fc040d58620c16c2a073ce1b95cf2b8b88254fb71163766f9e559028e3908738. The candidate source inventory is reports/CODE8_CANDIDATE_SOURCE.json, SHA256 a7002f6b540a57f522dc872469be2001d404895c39f37cebece104dabaf53a73.

The independent run used its own runner, separate from the root runner, and the same narrowly declared RAM-only retrieval accommodation. Only these two original historical production compatibility methods ran in public read_session(PRODUCTION) contexts:

- IndependentV1Compatibility.test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts
- IndependentV1Compatibility.test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints

Their historical byte helper opened a replacement session after 60 seconds only between completed reads. The all-corpus method used five sessions; the three-batch method used one. All original row, ID, candidate, label, fingerprint, byte-hash and receipt assertions executed. No other tests, temporary roots or fixtures were wrapped. This accommodation neither creates a disk cache nor changes a transport limit, production global deadline, test denominator, source file or scientific expectation. The full independent run had no failed or excluded case.

The new independent tests use small owned temporary GCDL roots, existing hand-specified nonsealed fixture records, real tiny exporter outputs, independently computed JSON/SHA256/Git-blob expectations, and an in-memory snapshot server. The fixture transport forbids real subprocess/network calls. None accesses production sealed payloads or generates official H/v2 pilot cases.

## Production commitments and historical compatibility

After the full suites, I independently compared all 173 entries in canonical reports/CODE8_IMMUTABLE_BEFORE.json with current canonical commitments. All 86 canonical shard commitments, all 86 exact raw batch-control hashes, and the SEALED_FREEZE control hash matched. The inventory file SHA256 is 2f2360e1b340d2b9deaf2a454ece5ebe4eed1c8080e52ba7d558f67296d975c0. The canonical 173-entry commitment-map digest remains bc386b68aed8b32b82870b244d88a894cf8f8ae271c4ccfbb57bf1865a0fbad2.

The comparison used public payload_info for archived shard metadata and ordinary hashing for retained batch/freeze controls. All eight sealed shard paths were absent locally, and a guard explicitly forbade any data/sealed/ call to the local payload reader. Network subprocess calls were forbidden during this check. No sealed content was opened, decoded or fetched. This establishes preservation of stored byte commitments; it is not a new evaluation of sealed labels.

The canonical remote archive index still had 278 paths and digest 3d08d15daa62dfafd5a73fc9d602aebbb4460dd5cf91a718b62b024bbb65adbb. The existing actual corpus is 2,528 accepted canonical records in 86 groups: 2,152 v1 records and 376 v2 records, with no new canonical publication from this review. The 2,976 rendered rows are derived surfaces rather than additional independent worlds. The original 1,936 nonsealed v1 corpus assertions and three predeclared historical batch comparisons passed in the full independent suite, retaining their captured commitments and expectations.

The independently inspected retained production render metadata comprised 24 completed receipts, 72 distinct cold dataset manifests, and 2,716,134 manifest bytes, spread across two original payload snapshot/catalogue references. It contained train/dev/ood exports only, with three views per group. This explains why repeated fresh raw header retrieval could exhaust the bounded recovery phase even though no scientific generation was failing.

CODE8 adds operational metadata qualification; it does not alter the scientific v1/v2/H generators, configurations or oracles. The source map below pins those files alongside the operations changes. Existing scientific hashes and conclusions retain their original freeze identities.

## Projection authority and raw derivation

A prepared capsule is untrusted staging. A self-consistent hash, invented verified marker, or successful storage of its bytes on a remote branch cannot substitute for raw derivation. prepare and derive call the existing full historical receipt validator with use_controls=False. This includes the original source/specification, source batch, protected-index, code, split, eligibility, shard commitment and policy predicates.

activate requires an exact qualified predecessor, retains every previously qualified capsule without change, and accepts at most four genuinely new receipt groups. It independently rederives each new capsule from the original full raw manifests and compares the exact resulting object with staged bytes. Restarted CLI publication therefore cannot turn fabricated disk staging into authority. The same-process public read session may reuse only already byte-verified RAM reads. No persistent prompt/label cache or unauthenticated private stage token is introduced.

The strict capsule carries only the fields needed to reproduce receipt-validation inputs: receipt path/raw SHA; parsed batch digest; specification, source-code, rendering-code and protected-index digests; minimal dataset identity, policy and scalar counts; source lineage commitments; ordered shard path/row/byte commitments; and original manifest/shard archive descriptors. It does not persist prompts, targets, per-record labels, selected-record arrays, full provenance, full protected indices or full code maps. The latter dictionaries are reconstructed in RAM from retained batch controls when the unchanged validator needs them.

The closed schema rejects unknown gold/verified fields, booleans masquerading as counts, negative counts, sealed/unknown split admission, path traversal and malformed dataset/shard identities. The dataset ID grammar matches the actual existing exporter: gcdl_rendered_ followed by 32 lowercase hexadecimal digits. Capsule and bundle argument paths are admitted before reading their bodies, including before any opaque sealed fixture can be opened.

try_projection returns validator inputs only for an attested covered receipt. Unattested or future exports continue through the original full raw validation path. Relevant receipt/batch/specification/source/protected-index bytes and fields are checked on each lookup, and current archive descriptors are compared exactly, including receipt_id and the original catalogue snapshot commitment. Appending an unrelated archived cohort does not invalidate old controls; changing any covered lineage/custody descriptor refuses the projection. These controls do not confer deletion custody.

## Remote proof, sessions and publication

Qualified control bytes are bound to an exact snapshot commit in the existing qualified archive reference namespace. Activation verifies the actual reference identity, tree entry type/mode/OID/size, raw length, SHA256 and Git-blob SHA1, and confirms the reference again after raw proof. Missing, moved, wrongly typed, altered or truncated remote objects refuse authority installation. Original payload catalogue references are also confirmed before and after the control proof, deduplicated across the aggregate. This preserves the original payload snapshot commitments while allowing recovery to avoid fetching every historical raw manifest header.

Runtime proof is cached only within the approved public read session for its provider and exact authority digest. Local aggregate bytes are reread and rehashed on every lookup, fresh local receipt/source/custody bindings are compared, and the current root/deadline is checked. An explicit foreign-root provider is refused before requests even outside a read session. Local byte corruption in an already warm session, moved original payload refs and exhausted lifetimes fail closed. This is bounded reuse of established proof, not a promise to detect a remote operator mutation on every individual lookup after a session proof; a new session establishes proof again.

In the independent 72-cold-manifest lifecycle fixture, 24 groups were qualified in six waves, after which raw header retrieval was unavailable. Repeated database recovery and inventory retained all 24 completed groups and exactly 2,976 rows. One latest control blob and one tree were fetched per session, plus six reference calls: pre/post checks for the two original payload catalogue snapshots and the latest control snapshot. No rendered shard or raw dataset-manifest retrieval was required on that attested recovery path. This is a measured fixture request-count result, not an actual production throughput result.

The composed archive publisher and direct CLI publication both bind STOP, the exact initial pause bytes, current independent source/evidence approval and the shared deadline. The activation publication_guard runs after fresh remote proof before immutable authority publication, and again before the pointer switch; the idempotent activation path is also guarded. A late operator-pause change or late source-approval failure during proof preserves the operator state and refuses the active pointer. Source approval cannot be inferred merely because preparation or an earlier proof succeeded.

The root lifecycle includes guarded raw derivation, planning, backup, create-only qualified snapshot publication, activation and subsequent custody backup. Existing snapshot creation remains create-only; no reference-update or force-push path is introduced. Work proceeds under the ordinary worker lock. These protections assume the configured qualified GitHub owner and the ordinary same-user operating-system/process trust boundary; this review does not assert resistance to an arbitrary same-UID process replacing code or mutating memory outside the lock protocol.

## Crash behavior, archive selection and bounded storage

Only two aggregate paths are permitted: manifests/render_controls/bundle-slot-0.json and bundle-slot-1.json. Planning writes the inactive slot while the existing pointer and active slot remain unchanged. The snapshot commit, raw SHA/Git OID and embedded bundle identity authenticate the historical bytes at that fixed path. Activation publishes its immutable authority record first and changes the active pointer last after rechecking the predecessor and publication guard. A crash, truncated inactive slot or unqualified plan cannot invalidate the old active aggregate. Later reuse of an inactive local slot does not rewrite the immutable remote snapshot version.

This avoids quadratic accumulation of full aggregate copies. Prepared capsules and authority history remain linear and bounded. The final reviewed limits are: 16 KiB per capsule; 8 MiB per latest aggregate; 32 MiB cumulative control storage; two 8 MiB RAM proof-cache units; four new receipts per activation; 4,096 receipt entries maximum; and a 1 MiB free metadata reserve. The 512 MiB programme bound, 10 GiB publication floor, 180-second processing bound, 144-second render allowance and 340-second watchdog remain unchanged. Hitting a control/programme bound retains raw payloads and refuses publication; it does not grant an unlimited history/specification guarantee.

The read-only actual legacy capsule derivation reported before source freeze was 9,273 bytes and 6.77 seconds, with no output publication or sealed raw access. The reviewed control ceilings can accommodate this current cohort without relaxing the accepted/rendered publication floor. Actual migration remains a separate bounded operational step.

The archive lifecycle requires a valid attested render projection and receipt validation for matching completed export folders before raw eviction, including resumed partial cold-receipt branches. Completed render cohort selection retains its original bounded archival limits and whole-group membership. Future exports must be qualified before their raw headers are evicted; unprojected exports retain their raw validation fallback. The controls do not authorize deleting code, minimal controls, batch lineage, operator state or source/evidence approval records.

## Defects challenged before freeze

The independent positive fixture initially exposed the wrong dataset-ID grammar. It was corrected to preserve existing exporter identities. A separate negative fixture exposed explicit foreign-root provider acceptance outside a read session; the root check is now unconditional before requests. Direct CLI proof-race tests exposed the late pause/source-approval gap; the publication callback now guards both authority and pointer publication. Those assertions stayed strict and passed against the final candidate.

The author additionally closed pre-read path admission for opaque sealed/unmanaged arguments, and the independent lifecycle reviewer exposed a STOP transition during backup before snapshot publication; that stage now rechecks the publication gate. The two-slot design replaced unbounded quadratic local aggregate history before the candidate freeze. All fixes are within the exact frozen map below. No known unresolved approval blocker remains in the scoped operational changes.

## Existing scientific evidence remains historical

The H prototype is inactive and DEV-only. Its once-only 800-case measurement remains attached to the original 61-file source freeze 4a4c0c9cbe233dba9a9857e5d70963dfd04e4b436df9448be14493afb896cd3e, not retrospectively to this 66-file operations candidate. The H module SHA remains 18751b15853905d79ed644c6ced6ce4e21a46300a189defc54465b564d3621f0 and its specification SHA remains dc2a17a16860c5087ca3a8a583bd7fd3a3c6abfb7c8b08ebc697252102ce7ea9.

The preserved H result is PASS for 800 evaluated DEV cases in 5.722628 seconds with zero oracle/label/replay mismatches: 96 goal cases, 128 constraint cases, 544 goal-atom ablations, and 32 matched baseline cases. All 96 goal winners require three steps; all 544 ablations change a nonempty exact best set; the 32 complete-conjunction cases retain the Q/Q-reverse ties. The baseline succeeds 32/32; the recorded positive-add/lowest-ID blind shortcut succeeds 7/96, the highest/longest shortcut 32/96, and constraint-blind/terminal-only membership 64/128. Full baseline group generation produced 928 rows of which 896 were unused, disclosed separately from the 800 evaluated-case denominator.

These are deterministic mechanical supplied-candidate checks. Numeric dynamics depend on developmental indices; seed mainly controls identities/order, developmental index overlap is disclosed, and literal a/b/c vocabulary is not a vocabulary holdout. The prototype covers register/reject semantics and supplied-plan optimality, not global action-space optimality, independent held-out worlds, learned generalization, transfer, model training, or production-profile promotion. CODE8 introduces no new scientific conclusion, new H profile, canonical prototype publication or sealed evaluation.

The v2 experiment likewise retains its original 42-file scientific source freeze fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467 and the existing exact result/freezing commitments. Neither H nor v2 was rerun during this review. The original raw-source/result custody, accounting, ablation and replay regression tests remain in the complete suite.

All five required scientific evidence files were reread through the approved bounded read-through interface and matched their exact raw SHA256 values. Their byte lengths were: H freeze 25,929; H STARTED 251; v2 freeze 6,375; H result 805,131; v2 result 112,806. The sole literal required_evidence map authorized by this report is:

```json
{"manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json":"3b22f95b23189bc6fa92439559c67259d6d6eef916a8a9546c28ed12aa98101c","manifests/H_COMPOSITION_EXPERIMENT_STARTED.json":"45722e9caa95880ed6cb26b0849a75c1c6da0d758a25e42e951275d4d38940ab","manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json":"4e4ec400a1aba413ce0112e7b43cd8bfd220a3fd5bbbbede70b08f1a031f27ca","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

Any missing or changed required evidence, changed reviewed source byte, or changed report invalidates this approval under the existing source/evidence guard. Historical archive/index/custody receipts remain separately bound; none may silently broaden the scope of this statement.

## Exact approved source map

Source/configuration count: 66.

Canonical SHA256 of the complete map: 5c2088aad6384040aff24aae72671298d39850f46c0356584cb7a96ba7e391c2.

The independent before-suite, after-suite, post-evidence and final report checks used this exact map. Report prose and operational result artifacts are not silently added to the Python/configuration source map; the fixed active review and required evidence must separately be pinned by the production validation receipt.

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
  "pipelines/remote_archive.py": "3f4d71088cfc9968a6bf030921c1f8cd4d43b033bcaf4683afbdb16f7b4ed4a3",
  "pipelines/render_controls.py": "069d1228ab0213e669b27ebc5130e529594c1d1a08b36f43c9f387e79298b001",
  "pipelines/rendered.py": "ae1e322fa40f7917ebb0f15eb01e2096ba5777c4174dfbb1c3b7f29ae5239014",
  "pipelines/worker.py": "143056de86e910a62148ff534dac4f6545f83c4edb147d35f26c78149510764f",
  "scripts/archive.py": "6aa490a52721e09cd1c4faf9144f7fbd21fb88aa966d356af07d71f99ea56abd",
  "scripts/archive_cycle.py": "1f133c88ae20271c4b821ef816b4928286233871c3b43c103e8d4e56fe485ff6",
  "scripts/backup.py": "ab3155c896b10475bf4ebc58be7fc8fc00f70796d397f11b0111707ca14931c8",
  "scripts/compact_git.py": "95b231416f5e299d7d4cfd7b2af654abbbe490d5558ca9ffe5fd0ba1f529c90b",
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
  "tests/test_surface_review.py": "c6dbc104a66122ad4a0c7255afb57c423fb7add32a988d9763cde8db05f3fe03",
  "tests/test_v2_review.py": "adb97f1e6ab879609a13492f346fdbb6afe7e4d11767b3e8f7ffeeb75268e1b0",
  "tests/test_version2.py": "5c58b03e8518932ca3536a212ff0deb4ff18c0ca100bb9f14b54745e934698be"
}
```


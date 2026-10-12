# Independent CODE11 review

Reviewed 2026-10-12, final assessment at 12:57 UTC.

FINAL_VERDICT: FAIL

The frozen candidate does not satisfy its amended failure contract. All 503 regression tests pass, scientific and historical commitments are intact, and the matched operational comparison meets the timing threshold. Two additional paired fault reproductions nevertheless disprove the requirement to match the old durable queue output when an already-completed receipt was corrupt before recovery began. No promotion is approved by this report.

This is a rejection of the stated transaction acceptance gate, not a finding that the candidate published false labels, deleted unverified data, or changed accepted bytes. Both demonstrated prefix writes concern an independently authenticated valid receipt. The distinction does not make the unfulfilled gate pass.

## Exact scope and source

Candidate: /Users/chengli/.codex/gcdl001-host/recovery-proposal-worktree/goal_conditioned_decision_data.

Reviewed source count: 69 Python/config files. Source digest: 92bb2c86063943721a9bb7759a705b2796fb59f679a0087a4084fa04dda5357f.

The baseline contains 68 files, digest 866bb13ee7f3df468a1718c81a15ee3d06788b9fd40d3e41eba4a920846c69ab. Comparing the complete maps shows exactly one existing source change, pipelines/rendered.py, plus tests/test_recovery_once_review.py. The candidate implementation SHA-256 is 90107a2a69b5e808518e4ebe3202c122b5fbb9076a70cfd301f4df9cd20111bc. The new test file has 27 cases, 29,579 bytes, SHA-256 f4b4e05b848491fb8e348e6c8bc8529ff9f50071b6577c1e3dc1d3db853f4e5d.

I authored the independent tests and fault witnesses, not the implementation. A further fresh child reviewer read the recovery/transaction boundaries without running tests or changing files and identified the combined-fault gap. I then reproduced both witnesses using the actual validator, actual SQLite and actual worker.event.

ARCHIVE_POLICY.md remains 7,259 bytes, SHA-256 52cd074feb13d9b86a01be8164e51f3beed963824c92d28bae5f74633d60b36e. Remote/archive verification, selective shared-Git-store proof, pending cleanup full raw proof, sealed scope, source approval, qualified credentials, STOP/pause controls and time/resource ceilings are unchanged. The configured processing/publication/watchdog windows remain 180/144/340 seconds. No implementation, canonical source, runtime state, queue, production payload, GitHub ref or remote object was changed by this reviewer.

## Why approval is withheld

The original candidate, implementation SHA-256 d773b840bca15e8cdebdc2e11596fa751d5d0bc56a3030c6c8b3b369bc5dc5dd, deferred validation of an existing completed receipt until its discovered upsert position. A single late corrupt completed receipt could then allow worker.event to commit an earlier authenticated prefix where the old preflight performed no recovery writes. Approval was withheld. The first timing result was preserved, rather than overwritten or described as blanket equivalence.

The immutable amendment was written before the conservative v2 implementation. It requires: “Match old preexisting-done fault durable output.” It explicitly permits stricter rollback for a later concurrent fault in a previously completed receipt, while preserving the original newly discovered fault and upsert-error behavior. The frozen v2 captures done pointers before discovery and rolls back only when the discovered validator raises for a captured done pointer.

That works for the isolated cases in the 27 tests. It fails when an earlier failure prevents recovery from reaching a corrupt completed receipt:

| Paired witness | Old recovery | Frozen candidate after worker.event |
| --- | --- | --- |
| Discovery A valid/pending, B newly discovered/corrupt, C previously done/corrupt before entry | Preflight rejects C before any recovery upsert; A remains pending | Upserts A, rejects B without rollback, never checks C; A becomes done |
| Discovery A valid/pending, B valid/fresh but its SQLite upsert aborts, C previously done/corrupt before entry | Preflight rejects C before any recovery upsert; A remains pending | Upserts A, actual SQLite trigger aborts B outside the rollback handler, never checks C; A becomes done |

The first witness uses the real metadata validator and two initially identical SQLite databases. The second uses a real SQLite BEFORE INSERT trigger with RAISE(ABORT), rather than mocking the upsert or merely asserting a handler location. Both invoke the actual worker.event commit and observe durable rows through independent connections. Network/subprocess/raw data access is forbidden in the fixtures. Source remained 92bb2c86063943721a9bb7759a705b2796fb59f679a0087a4084fa04dda5357f throughout.

The exact preserved witnesses are:

- /tmp/gcdl-source11-independent-mixed-fault-review.json, 4,122 bytes, SHA-256 7600a5e732b11968f63576cf977c5b5b8e81855884036f751e796f0627b5b70b.
- /tmp/gcdl-source11-independent-upsert-mixed-fault-review.json, 4,297 bytes, SHA-256 62aa3e4764015d52553ef83e93cc4f3b735173ae6e661f5777a089eb50714543.

These witnesses were produced after the full candidate/test freeze and did not alter any source or test assertion. The existing test suite lacks their combinations; its passing result is not evidence that the combined-fault contract holds. Discovery failure introduces no recovery DML, and absent/null/malformed done pointers still fail before upserts, so those observations do not establish another durable-write witness. Error precedence may also differ when discovery expires before an old done preflight would detect corruption; no blanket diagnostic equivalence is claimed.

Before reconsideration, the owner must either implement the current durable-output contract, or retain these failures and explicitly amend the intended fault semantics before another candidate comparison and fresh approval. A once-only design with broader conservative rollback would need its new-fault/upsert behavior specified and tested. Restoring the original loop satisfies the old preflight semantics but forfeits this optimization. This report chooses neither implementation for the owner and authorizes no deadline/resource relaxation.

## Completed regression evidence

Root: 503 tests PASS in 374.087 seconds, zero failures/errors/skips, complete 69-file map identical before and after.

Independent: 503 tests PASS in 374.264 seconds, zero failures/errors/skips, complete 69-file map identical before and after. Original 476 assertions were retained and executed, with all 27 new tests. The two additional witnesses above are separate manual paired reproductions and do not inflate the 503-test count.

Independent result:

```json
{"version":"code11-independent-suite-1","tests":503,"failures":0,"errors":0,"skipped":0,"seconds":374.264,"source_count":69,"source_before":"92bb2c86063943721a9bb7759a705b2796fb59f679a0087a4084fa04dda5357f","source_after":"92bb2c86063943721a9bb7759a705b2796fb59f679a0087a4084fa04dda5357f","source_equal":true,"historical_read_session_rotations":{"test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts":5,"test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints":1}}
```

Both full runners use a declared RAM-only harness around exactly two original IndependentV1Compatibility methods: test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts and test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints. The harness opens the existing public read_session(PRODUCTION) and rotates it after 60 seconds only between completed historical reads. The original methods, assertions, byte hashes, IDs, candidates, goldens and semantic checks remain intact. It does not wrap the whole suite, change worker/provider deadlines, create a disk cache, skip cases, or restart the scientific experiment. Historical session counts were 5 and 1 respectively for both suites.

Retained root runner evidence: reports/CODE11_ROOT_HARNESS.txt SHA-256 a6c136b44057c5e22a8187a64e8e36f85533180f87b6492e568b6ff1a208ecbe; reports/CODE11_ROOT_SUITE.json SHA-256 340104ee85a7b7b98448096805fb9eb3dc445ac5eee8fee1845ef20e3785ef35; reports/CODE11_ROOT_SUITE.log SHA-256 2913690b7464d3a9b567075e87d7e8d5fb11950a690f157c4e27702d59c89c15.

The new tests use hand-specified, already-attested label-free metadata at the existing projection boundary, the actual receipt validator, real SQLite and actual worker.event. They cover null/malformed/orphan done pointers before writes; omitted done pointers; duplicate pointers/discovery; fresh mutations and invalid projected fields/paths; successful manual and old-loop equality; caller rollback and event commits; old setup commit preserving earlier caller work; existing/new interrupts and deadline failures; exact exception propagation; and upsert errors outside the validator handler. They do not generate/read accepted, rendered or sealed raw fixture bodies. Scientific oracle regressions remain in the original suite. Their limitation is now explicit: isolated-fault coverage missed combined failures.

## Operational measurements and custody

I read and hashed the retained profile, first comparison, amendment and revised comparison; I did not rerun their profiling workload. The frozen baseline binds the approved source68, archive index, 203 control dependencies, RAM-only logical database state and original173 commitments. The revised result binds both freeze and amendment and the exact candidate recover file. All hash/body associations checked successfully.

| Retained artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| manifests/SOURCE11_OPERATIONAL_COMPARISON_FREEZE.json | 34301 | 453735f3046c4569a4fa90638e3251861ec757926f97e9337da2d75aa12a2d59 |
| manifests/SOURCE11_OPERATIONAL_COMPARISON_AMENDMENT.json | 2806 | 31a8ee7bcabed4ee44c2e409c162acf6bd6dae15251b3e8bc40ac9684291042b |
| reports/SOURCE11_OPERATIONAL_BASELINE_PROFILE.json | 1456 | 1a5f4430a0f4ee5356f2b6f63931fb2a2a87acf43c403edf5614d64b6ea46e94 |
| reports/SOURCE11_OPERATIONAL_CONTROLLED_COMPARISON.json | 1725 | 9df68142861d9cfbb34192d8654ecf8ae3ef11f8329d250ec51aa914d6eafbf6 |
| reports/SOURCE11_OPERATIONAL_CONTROLLED_COMPARISON_V2.json | 1850 | 4f66faeb31c7f331038ff42b563c79c72a5c9cc14a161aa9060ed49de7b55b2c |
| reports/SOURCE11_INDEPENDENT_FAULT_REVIEW.md | 2735 | d02c73cfa0ab78fcdb1688805605a54332b5a5c1139673d6679db6f83502e838 |
| reports/SOURCE11_INDEPENDENT_FAULT_REVIEW_V2.md | 1944 | ff42aa1b6fd27428fb012995ecb89d1e941e7c2648abd760c1375330559f990e |

Matched nice10 baseline: 133.531 seconds. Initial candidate: 107.195 seconds; this successful-state timing result remains valid as a measurement but its blanket failure-equivalence claim was rejected. Conservative v2: 108.417 seconds, 25.114 seconds or about 18.81% less than baseline, under the preregistered 120-second threshold. Rendered recovery phase fell from 74.6007 to 48.4480 seconds. Each of 42 completed receipts is validated once during recovery, then independently again during inventory: total validator calls 126 to 84, load_index calls 1840 to 1336. Inclusive nested function timings overlap and must not be summed.

The revised profile declares zero bulk requests, no sealed target reads, RAM-only database, unchanged live queue, nice10 and peak RSS 37,306,368 bytes versus baseline 39,059,456. I independently confirmed the frozen controls and current durable queue commitments rather than treating a PASS label as their proof. This is one matched operational metadata comparison, with network/host timing variation; it is not a model benchmark or an estimate of generalization. It does not satisfy the failed fault gate. The additional required bounded production cycle after approval has not been accepted by this reviewer, and no approval is given for such a promotion here.

## Fresh preservation checks

Scientific source: the 19 generators/oracles/environments/config files match every baseline byte commitment. Their canonical map digest is faa465fddd10e8155223873b45cb45bf140b320ac8f26e6cffda264df394404e. All other existing executable/test files also match baseline except the stated recover implementation. Thus generator parameters, profile reservations, label semantics, H prototype activation state and approved experiment bodies are unchanged.

Fresh canonical metadata verification found:

- Archive index digest c3c0d2851b182a75bb70292f1cb27b87d94b513cd7248463b7b99e09b630a434, 455 paths.
- All 203 control-file byte hashes match the frozen dependency map, canonical digest 81425b04540a96c58b76090e0767e981298e6636d1658f06bf0e4ce3bf0ea25b.
- Original173 map digest bc386b68aed8b32b82870b244d88a894cf8f8ae271c4ccfbb57bf1865a0fbad2 unchanged: 86 canonical shard SHA-256 metadata commitments, 86 exact raw batch controls and one SEALED_FREEZE commitment. The immutable before inventory, retrieved through the authenticated public archive read path, remains SHA-256 2f2360e1b340d2b9deaf2a454ece5ebe4eed1c8080e52ba7d558f67296d975c0.
- Eight sealed shard paths were absent locally and checked exclusively with stored commitment metadata. During the 173/control comparison, sealed _local reads and network subprocesses were explicitly forbidden. No sealed gold was fetched, decoded or audited.
- Read-only SQLite query_only checks match digest(sorted(SELECT * rows)) for jobs 38f6edba223853793ad83370d7b0c9bab572a113a536c1f0f8aca7f9ff112724, rendered_jobs 5c3d3590b7a908f6694cf58e12675f41e450723269c1e8f53523143d90f39bde and seen ce72f551b51648786333914c44ffba0dc27e93c69f041386d896856ee2989508.
- 105 completed canonical groups, 3124 records: v1 74 groups/2152 records and v2 31 groups/972 records, zero quarantine and zero pending canonical jobs. 42 completed rendered controls sum to 5250 derived rows; rendered queue has 87 pending and 268 superseded rows. The original owned operational PAUSED marker, with prior_pause null, remains present.

Fresh scientific evidence reads checked the five exact byte hashes below through the bounded public archive session. No experiment was rerun. The H confirmation belongs to the earlier 61-file scientific source digest 4a4c0c9cbe233dba9a9857e5d70963dfd04e4b436df9448be14493afb896cd3e; V2 confirmation belongs to its original 42-file freeze fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467. Operational CODE11 does not retroactively claim either experiment ran under this source.

The H prototype remains inactive for production. Its frozen DEV case evidence establishes the previously reviewed supplied-candidate mechanical behavior, not learned generalization: numeric-index developmental overlap, literal vocabulary, register/reject domain, supplied plans and possible formula shortcuts remain material limits. No new canonical H publication, training, paid/model API, transfer or out-of-domain claim is introduced.

Required scientific evidence map retained exactly; this is a custody commitment, not a passing approval:

```json
{"manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json":"3b22f95b23189bc6fa92439559c67259d6d6eef916a8a9546c28ed12aa98101c","manifests/H_COMPOSITION_EXPERIMENT_STARTED.json":"45722e9caa95880ed6cb26b0849a75c1c6da0d758a25e42e951275d4d38940ab","manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json":"4e4ec400a1aba413ce0112e7b43cd8bfd220a3fd5bbbbede70b08f1a031f27ca","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

## Complete reviewed source map

The canonical JSON digest of the following 69-entry map is 92bb2c86063943721a9bb7759a705b2796fb59f679a0087a4084fa04dda5357f. The full suite checked it before and after execution. Later manual witnesses checked the same digest again.

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
  "pipelines/rendered.py": "90107a2a69b5e808518e4ebe3202c122b5fbb9076a70cfd301f4df9cd20111bc",
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
  "tests/test_recovery_once_review.py": "f4b4e05b848491fb8e348e6c8bc8529ff9f50071b6577c1e3dc1d3db853f4e5d",
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

The frozen source is fully reviewed and rejected for the concrete unmet fault gate above. Passing regressions, successful preservation and a faster valid-state measurement do not remove that finding.


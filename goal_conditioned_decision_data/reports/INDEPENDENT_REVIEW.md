# Independent recovery structural-facts review

Reviewed 2026-10-12. This review concerns the final SOURCE11 facts candidate and supersedes neither the retained rejection nor the scientific experiment freezes.

FINAL_VERDICT: PASS

I approve the exact 69-file software snapshot below for the next bounded operational validation. The implementation restores the original recovery and SQL failure order byte-for-byte and skips only repeated successful, deterministic structural checks of identical freshly reread local custody controls. Both complete 511-test suites pass, the amended matched comparison meets its threshold, and fresh scientific/data/control commitments are preserved.

This is a source/evidence approval before deployment. It does not claim that the promoted live worker has completed a new production cycle or that future inventories will always recover below 120 seconds. Promotion, proposal retirement and resume still require their actual source, pause, STOP, storage and custody gates. The transient orchestration envelope specified below is part of this scoped assessment.

## Exact reviewed source and change

Candidate: /Users/chengli/.codex/gcdl001-host/recovery-proposal-worktree/goal_conditioned_decision_data.

Reviewed source count: 69 Python/config files. Source digest: f9acce125818a8b169a90e0b7ac6ac0c560da5367d70de20960836c0910b672c.

Approved SOURCE10 baseline: 68 files, digest 866bb13ee7f3df468a1718c81a15ee3d06788b9fd40d3e41eba4a920846c69ab. Comparing complete maps shows exactly one existing source change, pipelines/remote_archive.py, and one new file, tests/test_local_validation_facts_review.py. The implementation SHA-256 is c4c52b65fb498a48b329aa806a03beba9f8e394faeb9a42543645f0751d2387b.

The entire pipelines/rendered.py, including recover(), is byte-identical to the approved baseline, SHA-256 ae1e322fa40f7917ebb0f15eb01e2096ba5777c4174dfbb1c3b7f29ae5239014. No change to preflight, discovery, fresh receipt checks, upsert order, setup/final commits, caller event commits, exception scope or rollback policy is introduced.

The 35-case independent test file is 50,066 bytes, SHA-256 7dc43be658edec82193f5f9441e26445b91b771a1d571b49276c05fea2ef2de5. I authored its initial 27 cases and prior fault fixtures; a second independent reviewer audited it and added eight targeted cases. Neither reviewer implemented the archive patch. The source author separately passed 85 original focused assertions. The second reviewer froze the test file after 35/35 focused PASS in 0.283 seconds. The 476 original full-suite assertions remain intact.

ARCHIVE_POLICY.md remains 7,259 bytes, SHA-256 52cd074feb13d9b86a01be8164e51f3beed963824c92d28bae5f74633d60b36e. Scientific code/configs, generator profiles/reservations, oracle semantics, source approval, sealed bounds, archive publication/deletion, shared Git-store proof, active resource ceilings and runtime deadline settings are unchanged.

## Why this version resolves the rejected designs

The first recover optimization passed its valid-state comparison but could leave a validated pending prefix caller-committable before rejecting a corrupt previously completed receipt. The conservative second version rolled back when that previously completed receipt itself failed, but earlier newly discovered corruption or an earlier upsert error could prevent that check and leave the same durable difference.

I reproduced both combined-fault cases with the actual metadata validator, real SQLite, actual worker.event and independent observing connections. The upsert witness used a real BEFORE INSERT RAISE(ABORT) trigger. The old preflight left A pending; the rejected candidate could commit A as done. No false accepted labels or unverified deletion was established, but the amended durable-output acceptance gate failed. Its complete 503-test successes did not override that finding.

The rejection report remains unchanged, SHA-256 f12a3897a2e6b22639b8849764d96be51b3412ecb8a006cee6ed7fbf795bd1e3. Both exact mixed-fault witnesses, the original 133.531-second baseline, initial 107.195-second comparison and conservative 108.417-second comparison are retained in the operational evidence map below. The rejected 27-test source was preserved as reports/SOURCE11_REJECTED_REVIEW_TESTS.txt, SHA-256 f4b4e05b848491fb8e348e6c8bc8529ff9f50071b6577c1e3dc1d3db853f4e5d.

Before removing that obsolete test source from my owned proposal, I compared all 29,579 bytes with the live retained TXT and fetched exact remote tree/blob bytes at owner repository commit 62dfbee89b1d9e3159f32e724b7982a17dd2b6b7. SHA-256, length and Git blob OID f5f44072cb17fb9a0cb4050d7b8c7740054dfd17 matched. No code/data evidence was deleted based solely on push/HEAD equality.

Immutable amendment 2, SHA-256 e38126079a4d655e3664e4402661283832a7f818271e03ce11431d3ea7126291, was recorded before the new implementation and rejects both recovery reorderings. It requires the original failure order, every original fresh control check, a bounded positive structural-fact memo, unchanged commitments and a fresh comparison/independent approval. The final candidate conforms to that design.

## Implementation and trust boundaries

The code adds two fields to the existing same-root read session, clears them at final session exit, introduces one private helper and replaces two local load_index() structural-validator calls. The helper always freshly parses the current bytes. A memo entry is only a successful boolean fact keyed by the exact raw byte string, actual structural-validator callable and actual parser callable. It holds no decoded JSON object, validator output, exception, remote authority or model-visible payload.

Every load_index() still performs the original root/deadline checks and _local reads of the active index, immutable index snapshot, every catalog and every custody receipt, in the original order. _local still hashes actual bytes and checks regular type, one hardlink, bounds, inode/file identity and ancestry. Index/snapshot equality, expected SHA-256/Git OIDs, catalog/receipt IDs, orphan/cross-bindings and every indexed-file metadata relation remain freshly checked. The memo is consulted only after those current body reads/hashes, and it skips only _catalog/_receipt structural work already passed by those identical bytes.

The retained raw-byte budget is at most 4 MiB, separately counted from the existing remote catalog cache; the entry bound is 128. These are retention bounds, not new validity limits on history. Oversized bodies, misses and calls outside a read session run the original full validator. LRU hits do not inflate accounting; eviction causes revalidation. Foreign-root reuse fails and nested sessions cannot extend the existing deadline. Normal and exceptional outer exit clear all retained facts and their byte counter.

The safe premise is the audited deterministic parser/helper semantics under the immutable reviewed code and unchanged callable identities. Changing the parser or validator callable creates a different key and is explicitly tested, including a same-name replacement returning malformed data. Arbitrary hot-patching of nested trusted Python helpers or process memory is outside that premise; source guard and reviewed runtime code remain unchanged. No new promise to defend against arbitrary Python code execution is inferred from the memo.

The memo is not deletion custody. Public verify_archive() and evict_verified() still perform their complete raw remote proofs by default, and owned Git-store compaction retains its already-reviewed selector rules and full pending cleanup proof. No qualified projection or local payload checkpoint was moved or skipped.

## Independent coverage and complete regressions

The 35 tests verify:

- Exact raw-byte/validator/parser key separation, whitespace misses, fresh parse on hits, malformed replacements and failures never memoized.
- Independent returned objects and deep caller mutation resistance.
- Actual 128-entry and 4-MiB limits, exact boundary/oversized behavior, LRU promotion/eviction and byte accounting.
- Outside-session fallback; normal/exceptional cleanup; same-root nesting, earlier/later deadline restoration, foreign-root refusal and expiry on warm hits.
- Changed catalog/receipt bytes rejected at original checkpoints, then accepted after restoration; snapshot and independent Git-OID failures; self-rehashed malformed controls and warm-body pointer aliases rejected.
- Fresh graph relations, orphan/index metadata failures, symlinks/hardlinks/directories, and in-flight inode/ancestry replacement.
- Warm structural facts cannot substitute for complete public remote verification or evacuation proofs; corrupt/offline remote bytes refuse with the archive index unchanged.
- Exact baseline rendered-file bytes, both combined-fault durable outcomes, prior caller work committed before old preflight, original newly discovered error prefix behavior and a reachable real SQLite upsert-trigger failure after valid preflight.

These tests use tiny own metadata fixtures and opaque RAM payloads, real local filesystem checks, real SQLite and actual caller commits. Actual remote transport, authentication, subprocesses and network connections are forbidden in these focused fixtures. No live or sealed gold is read, no model/training job is started, and no scientific experiment is run.

Root complete suite: 511 PASS in 366.945 seconds, zero failures/errors/skips, exact 69-file map before/after. Retained JSON SHA-256: 265be1ce694f4813023ce9872c24098fbf77aae52a0c5990265c83077d8874b1.

Independent complete suite: 511 PASS in 368.870 seconds, zero failures/errors/skips, exact 69-file map before/after. Retained JSON SHA-256: c8d1f43e8f496a8da198b6a6c3ddf7c0aaa1aab4dbd3cfaf047b2e746241946f.

Independent result:

```json
{"errors":0,"failures":0,"historical_read_session_rotations":{"test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts":5,"test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints":1},"seconds":368.87,"skipped":0,"source_after":"f9acce125818a8b169a90e0b7ac6ac0c560da5367d70de20960836c0910b672c","source_before":"f9acce125818a8b169a90e0b7ac6ac0c560da5367d70de20960836c0910b672c","source_count":69,"source_equal":true,"tests":511,"version":"code11-facts-independent-suite-1"}
```

Both complete runners preserve all original tests and assertions. A declared RAM-only wrapper is applied to exactly two unchanged IndependentV1Compatibility historical methods: test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts and test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints. It opens the existing public read_session(PRODUCTION) and rotates after 60 seconds only between completed reads. It does not wrap the whole suite, skip cases, change worker/provider limits, cache bytes on disk, or restart any scientific pilot. Both suites used five sessions for the 1,936-row method and one for the three-batch method. All old byte hashes, IDs, candidates, labels, rendered goldens and fingerprint assertions execute.

Retained harness/log commitments:

| Artifact | SHA-256 |
| --- | --- |
| reports/CODE11_FACTS_CANDIDATE_SOURCE.json | bc518ea53a4866bc391db1b0689140e5c06e16aae7d59c342e8635a043c7c2d9 |
| reports/CODE11_FACTS_ROOT_HARNESS.txt | 81fc424986f3f8a66370fc422ab2bb9cb1b7ac34b277c618e653a9e56614c56f |
| reports/CODE11_FACTS_ROOT_SUITE.log | a90e6812d309418e8c3e03975e969320a50e4cf526ef8758806884a4aea661ba |
| reports/CODE11_FACTS_INDEPENDENT_HARNESS.txt | fb75fe4e8c16831d1ab59b4df03ae5a2ef98904f956368d4d907d9088d916e5a |
| reports/CODE11_FACTS_INDEPENDENT_SUITE.log | 2ebe7f0b313b6d1ece6f60fbd821f9d655d6732267d3d278e634c48a89e6102d |

## Matched operational comparison and limits

I inspected the freeze, both amendments, all three retained comparisons and exact raw hashes. I did not execute the profiler or retune against its result. V3 result, 2,215 bytes, SHA-256 4335f1d90da1b1050289cf47f594f873e5f88f34d93e86f51e90d9465e5feaa9, binds amendment 2, original freeze, exact candidate module c4c52..., baseline source68 and unchanged archive index.

The V3 measurement injects only the candidate remote_archive module into the approved baseline runtime in a separate RAM process, leaving all other live functions, including original recover(), unchanged. It measures that exact module under the frozen operational workload; it is not a claim that the completed full69 test snapshot was already deployed during profiling.

Matched nice10 total: 133.531 seconds baseline to 100.146 seconds V3, 33.385 seconds or about 25.00% saved, under the preregistered 120-second threshold. Recovery phase: 74.6007 to 57.5756 seconds; inventory: 29.1786 to 19.8452 seconds. Original counts remain 126 receipt validations, 1840 load_index calls and 92996 _local reads. V3 freshly calls _json 90477 times and runs _catalog 26/_receipt 24 times. Repeated structural work, rather than control reads or receipt checks, is reduced. Nested function durations overlap and must not be added.

The result records zero accepted/sealed/rendered bulk requests, no sealed target reads, unchanged live queue/controls, a RAM-only database, no training and peak RSS 36,978,688 bytes. I independently verified its hash/body associations and preservation, rather than accepting its PASS label alone.

This is one matched metadata comparison with host/network timing variation. It supports this frozen workload, not an unbounded growth/performance claim. Cache eviction and oversized fallback can reduce future savings without changing validity. Actual post-promotion approval and a bounded completed production cycle remain to be established by the owner; no such live outcome is asserted here. Existing windows remain processing180/publication144/watchdog340 seconds, with storage/publication floors unchanged.

Operational evidence is separately pinned; it is not added to the runtime scientific required_evidence fetch list. Exact ten-path map:

```json
{"manifests/SOURCE11_OPERATIONAL_COMPARISON_AMENDMENT.json":"31a8ee7bcabed4ee44c2e409c162acf6bd6dae15251b3e8bc40ac9684291042b","manifests/SOURCE11_OPERATIONAL_COMPARISON_AMENDMENT_2.json":"e38126079a4d655e3664e4402661283832a7f818271e03ce11431d3ea7126291","manifests/SOURCE11_OPERATIONAL_COMPARISON_FREEZE.json":"453735f3046c4569a4fa90638e3251861ec757926f97e9337da2d75aa12a2d59","reports/SOURCE11_MIXED_FAULT_REVIEW.json":"7600a5e732b11968f63576cf977c5b5b8e81855884036f751e796f0627b5b70b","reports/SOURCE11_OPERATIONAL_BASELINE_PROFILE.json":"1a5f4430a0f4ee5356f2b6f63931fb2a2a87acf43c403edf5614d64b6ea46e94","reports/SOURCE11_OPERATIONAL_CONTROLLED_COMPARISON.json":"9df68142861d9cfbb34192d8654ecf8ae3ef11f8329d250ec51aa914d6eafbf6","reports/SOURCE11_OPERATIONAL_CONTROLLED_COMPARISON_V2.json":"4f66faeb31c7f331038ff42b563c79c72a5c9cc14a161aa9060ed49de7b55b2c","reports/SOURCE11_OPERATIONAL_CONTROLLED_COMPARISON_V3.json":"4335f1d90da1b1050289cf47f594f873e5f88f34d93e86f51e90d9465e5feaa9","reports/SOURCE11_REJECTED_CODE11_REVIEW.md":"f12a3897a2e6b22639b8849764d96be51b3412ecb8a006cee6ed7fbf795bd1e3","reports/SOURCE11_UPSERT_MIXED_FAULT_REVIEW.json":"62aa3e4764015d52553ef83e93cc4f3b735173ae6e661f5777a089eb50714543"}
```

The retained map file reports/CODE11_FACTS_OPERATIONAL_EVIDENCE.json remains 1,219 bytes, SHA-256 956b188843e1f82043c8fbbc6399f2f1a90d340432186d72528508698f8fbec6. Fresh byte hashing of all ten named live artifacts matches it, including rejected review and both fault witnesses.

## Fresh scientific/data/control preservation

Only remote_archive.py differs from the complete approved baseline source map. All 19 generators/oracles/environments/config files match every byte, canonical submap digest faa465fddd10e8155223873b45cb45bf140b320ac8f26e6cffda264df394404e. All original other code/test files are unchanged.

Fresh read-only checks establish:

- Archive index digest c3c0d2851b182a75bb70292f1cb27b87d94b513cd7248463b7b99e09b630a434, 455 paths.
- Exact frozen 203 control-file hashes, map digest 81425b04540a96c58b76090e0767e981298e6636d1658f06bf0e4ce3bf0ea25b.
- Original173 unchanged: 86 canonical shard SHA-256 metadata commitments, 86 raw batch-control byte hashes and one SEALED_FREEZE hash. Map digest bc386b68aed8b32b82870b244d88a894cf8f8ae271c4ccfbb57bf1865a0fbad2. Authenticated read-through of reports/CODE8_IMMUTABLE_BEFORE.json matched raw SHA-256 2f2360e1b340d2b9deaf2a454ece5ebe4eed1c8080e52ba7d558f67296d975c0.
- All eight old sealed shard paths are absent locally and checked only through stored metadata. During the commitment comparison, sealed _local reads and network subprocesses were explicitly forbidden. No sealed payload was fetched or decoded.
- Read-only query_only SQLite canonical sorted-row digests match frozen jobs 38f6edba223853793ad83370d7b0c9bab572a113a536c1f0f8aca7f9ff112724, rendered_jobs 5c3d3590b7a908f6694cf58e12675f41e450723269c1e8f53523143d90f39bde and seen ce72f551b51648786333914c44ffba0dc27e93c69f041386d896856ee2989508.
- 105 completed canonical groups/3124 records: v1 74/2152 and v2 31/972, zero quarantine and zero pending canonical jobs. 42 completed rendered controls sum to 5250 derived rows; current rendered queue is 42 done/87 pending/268 superseded. The original owned operational PAUSED marker, prior_pause null, is preserved.

All five scientific evidence bytes were freshly read through the bounded authenticated archive path and hashed. No official H or V2 trial was rerun. V2 confirmation retains its original 42-file source freeze fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467; H confirmation retains its earlier 61-file freeze 4a4c0c9cbe233dba9a9857e5d70963dfd04e4b436df9448be14493afb896cd3e. Neither is retroactively reattributed to this operational source.

The H-composition DEV prototype remains inactive as a production profile. Its mechanical supplied-candidate evidence does not establish learned generalization: developmental numeric-index overlap, literal vocabulary, register/reject scope, supplied plans and possible formula shortcuts remain limitations. No new H canonical publication, model training, paid API, transfer or out-of-domain claim is approved here.

Exact complete runtime required_evidence map, retained without expansion or reduction:

```json
{"manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json":"3b22f95b23189bc6fa92439559c67259d6d6eef916a8a9546c28ed12aa98101c","manifests/H_COMPOSITION_EXPERIMENT_STARTED.json":"45722e9caa95880ed6cb26b0849a75c1c6da0d758a25e42e951275d4d38940ab","manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json":"4e4ec400a1aba413ce0112e7b43cd8bfd220a3fd5bbbbede70b08f1a031f27ca","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

## Promotion/retirement orchestration envelope

I read the owner scripts and flagged their original late-inventory, pause/source and evidence pin gaps. The revised scripts require exact511/source-before-and-after equality, the fixed independent review hash, both exact suite JSON hashes and the fixed ten operational hashes. Promotion rechecks STOP, original pause bytes, candidate/source bytes and all203 controls before the first source overwrite, preserves the approved SOURCE10 receipt/review/quality/status, copies only the exact two source paths and approved report bytes, validates a fresh process, retains PAUSED and confirms backup.

Retirement captures the complete physical file/directory inventory, tracked/status/ignored sets and proposal/canonical ancestry identities. It proves the fixed owned detached base and approved backed-up HEAD through create-only qualified immutable snapshot refs, checks unchanged tracked content against the base tree, and fetches exact raw bytes for all changed/untracked sources/evidence and all unchanged eligible base payload copies. It binds SHA-256/Git OID/length/local identity, rechecks refs/HEAD/source approval/pause/STOP and complete inventories before removal, then removes only the fixed owned proposal. No other worktree, branch or ad hoc Git payload cleanup is approved.

Read-only inspected script hashes: /tmp/gcdl-promote-code11-facts.py 3554580f47bfdc1dac890b9affb2605172bc10809fb6a6a5997aefd92a30cfa9; /tmp/gcdl-retire-code11-facts.py 090bf6b899d25c5c5347e4f47b5ba0c8e4dd34efb6d380bf4c89e77af53a2835. These transient scripts are outside the69 software map. No script was executed by this reviewer.

For those exact inspected scripts, local Git subprocesses MUST inherit GIT_ALLOW_PROTOCOL='' and GIT_PROTOCOL_FROM_USER='0', as well as GIT_NO_LAZY_FETCH='1'; for example, supply that explicit environment when invoking each script. GIT_NO_LAZY_FETCH alone is insufficient on the host Git version. The already-reviewed _publish_snapshot explicitly qualifies HTTPS only for the intentional owner ref operations. An equivalent explicit offline guard added inside a script must be reread/hash-pinned before using that changed script. This review gives no permission to execute the scripts without that envelope or to delete any unverified local payload.

All reviewer processes and final artifact writes must finish before retirement starts. The owner must repeat live preservation and source approval after promotion and validate the actual bounded cycle before claiming production recovery. This source PASS supplies the independent approval prerequisite; it does not replace those operational receipts.

## Complete reviewed source map

Canonical JSON digest of this exact69-entry map: f9acce125818a8b169a90e0b7ac6ac0c560da5367d70de20960836c0910b672c.

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
  "pipelines/remote_archive.py": "c4c52b65fb498a48b329aa806a03beba9f8e394faeb9a42543645f0751d2387b",
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
  "tests/test_local_validation_facts_review.py": "7dc43be658edec82193f5f9441e26445b91b771a1d571b49276c05fea2ef2de5",
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

The approved change is the bounded same-session local structural-fact memo under the conditions above. Earlier rejected recovery reorderings remain rejected and preserved.


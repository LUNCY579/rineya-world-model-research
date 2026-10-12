# Independent CODE7 review: bounded runtime and inactive H confirmation

FINAL_VERDICT: PASS

Reviewed on 2026-10-12. This is a manual, scoped approval of the exact 61-file source/configuration map below, the bounded operational changes, and the fixed inactive H DEV confirmation evidence. It is not an automatically inferred verdict.

Approved source digest: `4a4c0c9cbe233dba9a9857e5d70963dfd04e4b436df9448be14493afb896cd3e`. Reviewed proposal: `/Users/chengli/.codex/gcdl001-host/h-proposal-worktree/goal_conditioned_decision_data`. The final independent run passed **399/399 tests in 331.773 seconds** (331.779 seconds including wrapper accounting), with exact before/after source-map equality. The root agent separately reported **399/399 in 854.131 seconds** using the unchanged standalone historical-read baseline, also with exact source-map equality.

## Scope, independence and test accounting

I implemented neither the new operational changes nor the H prototype. I authored only `tests/test_runtime_session_review.py` (eight independent tests) and this final report for this change. Other agents/root authored the 31 cold-read session tests, seven budget tests and 24 H tests. The final suite includes all 329 previously approved regressions plus those 70 added tests. No source/test bytes changed after the candidate freeze or H measurement.

The eight independent tests require tighter explicit caller deadlines, refusal before provider creation on an expired deadline, bounded yield before database/generation/backup when session entry is already late, source and custody failures retaining a pause and exact committed counts, explicit sealed-read permission for opaque local fixtures, fresh destructive remote proof despite a warmed read catalogue, and refusal when a rewritten receipt drops any H custody evidence from the fixed complete reviewer map. These use small owned fixtures, not production targets or real remote writes.

The first independent full run retained exact source bytes but passed 398/399 tests in 257.117 seconds: its long historical corpus method reached a reused provider's 180-second deadline during the final rehash. No approval was issued for that run. The passing retry used a declared temporary, RAM-only unittest runner harness around only two original methods in `IndependentV1Compatibility`: `test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts` and `test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints`. Each uses the public `read_session(PRODUCTION)`; between completed historical reads the harness exits/re-enters the session after 60 seconds. The corpus method used five sessions and the three-batch method one. All 399 original tests, original assertions, hashes, denominators and mathematical checks executed; every other-root fixture remained outside the harness. There was no source edit, disk cache, raised request/provider limit or skipped custody check. This scopes the test audit's long duration without changing production's global read/publication deadline behavior.

I did not run or rerun the scientific pilot, execute its baseline generator, read sealed production labels, mutate production/runtime, perform a GitHub write, publish canonical records, launch training, download a model or invoke a paid service. I inspected the completed receipts and separately checked their recorded arithmetic. I also validated the existing-result custody path with both oracle evaluators and baseline generation explicitly forbidden.

## Operational findings

No unresolved blocker was found in the frozen operational source. Six existing files change: archive reads, worker scheduling, rendered orchestration/export, release approval reads and backup budget propagation. Production configuration remains byte-identical to approved v6: active profile v2, single worker, 180-second cycle, 300-second schedule and the binding 10 GiB floor. The external watchdog is unchanged at 340 seconds. The registry remains v1/v2 only; H is not a production profile.

The new read session shares one owner-qualified process-local transport and bounded immutable metadata within a programme root. Tree and catalogue caches each have a four-MiB byte budget; the existing raw RAM cache remains 16 MiB. Local archive index/catalogue/receipt controls are freshly validated per read. Immutable catalogue reuse is bound to provider, repository, commit, path, raw SHA-256, Git blob OID and length. Uncached cold payload reads retain pre/post snapshot-reference checks and exact raw length/SHA-256/Git OID validation. Existing raw-cache hits remain tied to the exact validated index/descriptor tuple and are rehashed; they do not constitute fresh eviction proof. Mixed roots, changed controls, corrupt metadata/bytes, moved references, truncated trees, explicit alternate providers and nested shorter deadlines have refusal tests.

The measured fake API fixture reduces twelve cold reads from 132 calls/twelve authentications to 44 calls/one authentication, retaining 24 reference checks. This verifies reuse/accounting, not actual production throughput or a guarantee that remote service requests complete. Default providers remain independently bounded to 180 seconds; network or provider failures still fail closed. Credentials remain process-local, with no account switch or credential file cache.

The worker clock now starts before cold inventory recovery. Recovery exhaustion yields a checkpoint instead of resetting the generation clock. Generation/rendering work is anchored to the same cycle start; rendered work uses the 144-second publication boundary under the unchanged configuration. The 180-second processing deadline and at most 140-second finish reserve remain within the 340-second watchdog. The remaining finish budget caps both parent backup wait and child backup timeout, followed by final approval/deadline checks before a healthy heartbeat.

Time exhaustion is a typed bounded yield. Invalid source approval, custody failures, resource failures and other non-time errors retain the normal pause/error behavior. Partial immutable rendered views remain pending until a complete validated receipt commits, and deterministic retry preserves their bytes. An archival action exits its worker invocation before opening the production database or generating new records. STOP, original PAUSED causes, the shared worker lock, source approval, provenance/profile checks, dual oracles and publication floor remain binding. No original pause is implicitly cleared by these changes.

Destructive provider/cycle/backup/Git-compaction code retains the previously approved custody protocol. The new independent warm-catalogue test changes server catalogue bytes after a successful session read and requires evacuation to fetch fresh raw proof and refuse without local mutation. Source/read caching does not approve eviction or replace its whole-cohort reference/tree/raw-byte proof.

## Preserved canonical history

A read-only comparison independently matched all 173 commitments in `reports/CODE7_IMMUTABLE_BEFORE.json`: 86 canonical shard commitments, 86 exact raw batch-manifest hashes and the sealed-freeze hash. The commitment-map canonical digest is `bc386b68aed8b32b82870b244d88a894cf8f8ae271c4ccfbb57bf1865a0fbad2`; the inventory file SHA-256 is `4f4c6875fd97192baba5cc068ee552c2004f6c5c8ec300dacd4fa2e2c23aac20`. The eight sealed shards were absent locally and checked only through archive metadata, without a raw sealed payload read.

Authoritative batch metadata still counts 74 v1 groups/2,152 accepted records and twelve v2 groups/376 accepted records: 2,528 total, zero quarantined, with 1,576 TRAIN, 355 DEV, 381 OOD and 216 sealed records. Current canonical source remains the approved 54-file v6 digest `3cc00cd2c73b37829c8488babe654296aba4df28fada67c459a3b8a2f7c58835`; this is a pre-promotion check, not a claim of completed CODE7 deployment.

The original 66 nonsealed v1 shards/1,936 rows retain byte collection digest `45cb9526e674ec8d2a59c9fa40e219890f40dc3961b48198fd3c9c9f8d294d02`. Complete record/receipt before-and-after byte checks, source receipts, IDs, candidates, labels, fingerprints, v1 generator/config/output commitments and rendering goldens remain checked by the full suite. V1/v2 generators, environment/oracle semantics, profile archives and canonical labels are unchanged by this proposal. The earlier v2 experiment remains bound to its original 42-file source digest `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467` and unchanged freeze/result bytes.

## Frozen inactive H confirmation

The H freeze contains the exact approved 61-file source map, fixed specification and all 32 input-commitment parents. STARTED has the exact supported body, freeze hash, source digest and positive timestamp; result custody binds both file hashes and its own body integrity digest. All five required evidence files were independently rehashed; the older v2 result was retrieved with bounded archive read-through. The H result is 805,131 bytes, below the one-MiB ceiling, and records one completed attempt in 5.722628083 seconds under the 300-second bound. Interrupted/replaced/truncated/forged evidence and changed source/specification have refusal tests.

The fixed cohort uses DEV identities 430000 through 430031 without replacements. Its 800 evaluated cases are 96 goal tasks, 128 constraint tasks, 544 individual goal-atom ablations and 32 selected baseline H tasks. Baseline construction additionally generated 32 full v2 groups/928 rows, including 896 unused family rows whose oracle work is explicitly disclosed outside the evaluated-case count; the wall bound covers that work.

I independently verified the recorded exact rational utility rankings and hand-formula endpoints across all 224 prototype tasks, the complete best sets across all 544 ablations, all 96 three-step goal winners and all 32 conjunction ties. Per-task inputs match frozen commitments, sibling candidate IDs/order stay fixed, goal optima are disjoint, none/protected-c retains the negative winner-flip control, and every atom ablation has a changed nonempty exact best set. The 32 baseline receipts bind their singleton selected best. The saved result reports zero differential-oracle, label or reproduction mismatches and supplies all six operations. This inspection uses the recorded summaries and arithmetic, not new confirmation oracle calls.

Observed goal-blind selected-best membership is 7/96 for the valid positive singleton-add and lowest-ID controls, 12/96 for first option, 0/96 for shortest plan and 32/96 for longest plan. Every fixed choice has a one-third ceiling within each goal triplet. Constraint-blind and terminal-only constraint controls recover 64/128 choices; terminal-only complete-set recovery is also 64/128. Each conjunction preserves the complete two-way Q/Q-reverse tie. Baseline singleton-add recovery is 32/32. These are finite construction diagnostics, not learned-model performance.

Numeric worlds derive from parameter index; seed changes only opaque IDs/order, and developmental indices overlap confirmation dynamics. The cohort is not an unseen semantic holdout or 32 IID samples. It uses literal a/b/c register vocabulary, deterministic register dynamics and supplied-candidate optima. It does not demonstrate global action-space optimality, absence of formula-aware shortcuts, generic composition, ledger H coverage, vocabulary transfer or learned generalization. Cross-version baseline candidate/goals/plan construction differ; only the within-bundle goal/constraint contrasts isolate the declared axis. No H canonical profile, TRAIN/OOD reservation, sealed instrument, training export or model is promoted by this review.

## Fixed required evidence

The exact complete canonical map below is part of this manual reviewer statement. It is the only literal required-evidence map in this document; an altered receipt cannot substitute a weaker subset. The final validation receipt must retain all five paths/hashes and this exact report hash. Historical reviews must remain separate from the active fixed statement.

```json
{"manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json":"3b22f95b23189bc6fa92439559c67259d6d6eef916a8a9546c28ed12aa98101c","manifests/H_COMPOSITION_EXPERIMENT_STARTED.json":"45722e9caa95880ed6cb26b0849a75c1c6da0d758a25e42e951275d4d38940ab","manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json":"4e4ec400a1aba413ce0112e7b43cd8bfd220a3fd5bbbbede70b08f1a031f27ca","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

Approval applies only to the exact source and these fixed evidence bytes. Actual deployment, remote backup and resumed production remain separate operations requiring their own post-copy source/evidence and immutable-history checks. The local OS/UID and authorized remote owner remain the trust boundary; observed corruption, moved custody, missing proof, incomplete operations and source changes fail closed. Publication remains subject to existing resource and review gates.

## Complete approved source map

Paths are relative to `goal_conditioned_decision_data`. Values are SHA-256 of exact bytes; the approved digest is SHA-256 of this map in canonical sorted-key compact JSON.

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
  "pipelines/rendered.py": "0cea289cc6f676a649a304049b98043d62121fcdc67b1d3bdf49ced202a168a7",
  "pipelines/worker.py": "143056de86e910a62148ff534dac4f6545f83c4edb147d35f26c78149510764f",
  "scripts/archive.py": "6aa490a52721e09cd1c4faf9144f7fbd21fb88aa966d356af07d71f99ea56abd",
  "scripts/archive_cycle.py": "7ddb94cfd2af9f7891a301d37da8b614d32df97fed0646dd7bd9ac175bb1ef01",
  "scripts/backup.py": "ab3155c896b10475bf4ebc58be7fc8fc00f70796d397f11b0111707ca14931c8",
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
  "tests/test_rendered_pipeline.py": "4cb9642415f5f495fcfe33629330fd50e67e1ea8e12f24eb4a39c5c983954b42",
  "tests/test_rendering.py": "31a0335a99e7f59b30fa1fc854b473cfccddf02c0747a4ee603b6016931fb5a4",
  "tests/test_runtime_session_review.py": "3a0ff8514b8c77052b53d3be1b772d9eac0f89dffdaa407074bccdf49522f65b",
  "tests/test_surface_review.py": "c6dbc104a66122ad4a0c7255afb57c423fb7add32a988d9763cde8db05f3fe03",
  "tests/test_v2_review.py": "adb97f1e6ab879609a13492f346fdbb6afe7e4d11767b3e8f7ffeeb75268e1b0",
  "tests/test_version2.py": "5c58b03e8518932ca3536a212ff0deb4ff18c0ca100bb9f14b54745e934698be"
}
```


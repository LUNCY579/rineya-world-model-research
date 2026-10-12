# Independent CODE9 standalone-entry review

FINAL_VERDICT: PASS

This is a fresh, narrowly scoped approval of the exact 67 Python/configuration files below. CODE9 repairs the two lazy package-relative imports in the archive CLI and adds two standalone-context regression tests. Both complete 449-test runs passed with identical before/after source maps. I independently inspected the complete diff, challenged the entry-mode and custody boundaries, reproduced both original failures in an empty owned fixture, verified the fixed evidence, and checked preservation of the actual production commitments and counts.

Candidate: /Users/chengli/.codex/gcdl001-host/cli-proposal-worktree/goal_conditioned_decision_data.

Approved source-map SHA256: 00cc8253898f8f1c108632817663d5868257e297facf33c47bcc0aa4f99fdedd.

This approval authorizes only the reviewed source and its existing guarded operations. It does not state that CODE9 has already been promoted, that a destructive production command has been executed against it, or that a new scientific experiment has run. Production remained read only for this review. The worker's preserved migration pause and ordinary review, custody, STOP, resource and publication gates remain applicable.

## Problem, exact change and independence

The actual CODE8 standalone command python3 -B scripts/archive_cycle.py --manual reached _publish_render_controls with no package context and raised ImportError: attempted relative import with no known parent package. _require_render_controls had the same lazy-import defect. The preceding CODE8 suite covered packaged invocation and custody behavior but missed this standalone helper import mode. Root reported that the actual attempt stopped after remote catalogue verification and custody backup, before eviction, leaving all eleven selected payloads intact. The reviewed packaged -m invocation subsequently completed that already-authorized cleanup under unchanged CODE8 approval.

The complete CODE9 diff from the independently frozen CODE8 source map is exactly:

- scripts/archive_cycle.py: _publish_render_controls imports render_controls from goal_conditioned_decision_data.pipelines; _require_render_controls imports rendered and render_controls from the same absolute package. No other line or existing source file changes.
- tests/test_archive_standalone_review.py: one new file containing exactly two targeted tests.

The script already establishes ROOT.parent on sys.path and uses this absolute package convention at its top level. The two replacements therefore preserve the static module allowlist and helper behavior for both standalone and packaged invocation. They do not introduce dynamic imports, a configurable module path, a custody exception, a source-approval bypass, or a publication-limit change. Static inspection found no other package-relative import in the CLI scripts.

The changed script SHA256 is 155cd58e8316cf8f47c68cde92c1ba4f18ab057c8ae6cc05bcb8c1143771e330. The new test file is 6,229 bytes, SHA256 6215e0fe206726782a9eef98be01d5966ae4675127c9ef365b5ae28afbbb1266. All other 65 existing Python/configuration files match the previously reviewed CODE8 map exactly; none is removed. The frozen candidate receipt reports/CODE9_CANDIDATE_SOURCE.json is 6,727 bytes, SHA256 fac29c136b013585f45c877133df0dbd8d1a00cc7c7e23644f2ec6273c0ea57d.

I did not implement the import replacements or author the two new test cases. A separate independent reviewer authored the tests. I reviewed their fixture boundaries and assertions, reproduced the old errors separately, and ran the full final suite with my own runner. No source or test file was changed during final review.

## Standalone regression and custody checks

Both new tests load the actual script with runpy.run_path under a non-package run name. They assert an empty/None __package__ in the loaded script/function globals, reproducing the failing import context without executing a destructive production main routine.

The first invokes the real _publish_render_controls against an empty, resolved, owned temporary GCDL root. It uses the real empty pending inventory and expects None. Preparation, remote transport, archive subprocesses and eviction are forbidden; the programme remains empty.

The second supplies a matching completed rendered receipt using the existing gcdl_rendered_<32 lowercase hexadecimal digits> folder grammar. Its source batch points to an opaque, nonsealed accepted-shard fixture and binds the exact raw SHA256 and byte length. The real archive local IO and real rendered._batch admission execute before the mocked missing projection. The real helper must raise the exact existing qualified-controls RuntimeError. try_projection receives the exact parsed source batch; full receipt validation, remote/network/subprocess boundaries and eviction remain uncalled; every temporary programme file retains its exact bytes.

This second case prevents an import-only or nonrender no-op test from concealing loss of the guard needed before raw cohort eviction. The opaque source fixture is never parsed as a scientific record. The tests do not inspect production, fetch sealed payloads, produce official DEV cases or contact GitHub.

I also loaded the unchanged live CODE8 script under a standalone runpy context and invoked each failing helper with an empty owned temporary programme, forbidding network and archive actions. Both independently reproduced the same ImportError before any fixture mutation. The new complete suite passes both helpers under the repaired imports while retaining all previous guard tests. This is targeted coverage of the lazy import boundaries, not an actual destructive production run of the candidate.

## Complete frozen regression results

| Run | Result | Duration |
| --- | --- | --- |
| Root complete final suite | 449 passed; 0 failures, 0 errors, 0 skipped | 349.635 seconds |
| Independent complete final suite | 449 passed; 0 failures, 0 errors, 0 skipped | 351.690 seconds |
| Separately authored focused standalone cases | 2 passed | 0.034 seconds |

The 449 cases are the prior 447 cases plus the two standalone tests. Every original assertion executes. No case is removed, skipped, weakened or replaced. The exact 67-file map above was checked before and after both complete suites, and again after independent evidence and commitment inspection.

The root result is reports/CODE9_ROOT_SUITE.json, 639 bytes, SHA256 f34a614b653f0bea84e73bc4bf153fb95269d1ce63a854346ad1997008baaab1. Its count, zero failure/error/skip fields, identical source digests and historical-session counts match the final root outcome. The root runner is retained as reports/CODE9_ROOT_HARNESS.txt, SHA256 01603f9e66d50acf32b9e5f84ef9d7496a3eb1ba63cd6428b948636c3ce427e8. The root log SHA256 is 0369f1763ed9de66f310c876bbd3aaac2b264be03eff52d8a8b001cf2347ca94.

The independent runner was separate and used direct tests-directory discovery. An initial runner preflight using the package parent as unittest top_level_dir refused the namespace tests directory before executing any test. Corrected discovery loads all 449 cases and wraps the actual discovered historical module. No repository file, test assertion or scientific expectation was edited to resolve that runner issue.

Both final suites used the previously declared RAM-only public read-session accommodation for only two original production compatibility methods:

- IndependentV1Compatibility.test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts
- IndependentV1Compatibility.test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints

A session may be replaced after 60 seconds only between completed historical reads. In each final run the all-corpus method used five sessions and the three-batch method one. All original byte commitments, IDs, labels, candidate order, fingerprints and receipt assertions still execute, including every one of the 1,936 original nonsealed v1 rows. No other test or temporary fixture runs inside this accommodation. It writes no disk cache and changes no source byte, transport bound, production global deadline, test denominator or mathematical expectation.

The complete suite continues to exercise archive tamper/missing/offline/ref movement, custody before unlink, interrupted move restoration, path/inode/ancestry checks, sparse-Git cleanup and rollback, immutable snapshot creation, historical source/receipt lineage, sealed metadata-only recovery, atomic v2 triplets, goal-only rendered comparisons, proof/root/session/deadline refusal, late operator-pause/source changes, and the CODE8 metadata-control qualification lifecycle. PASS is a manually scoped conclusion from those results and this source/custody inspection, rather than an automatic inference from a test counter or older report.

## Production byte commitments and counts

I retrieved only the already archived reports/CODE8_IMMUTABLE_BEFORE.json control through bounded authenticated read-through into RAM, then compared its entire 173-entry map with current production. All 86 canonical shard SHA256 commitments, all 86 exact raw batch-control SHA256 values and the SEALED_FREEZE control SHA256 matched.

The inventory file SHA256 is 2f2360e1b340d2b9deaf2a454ece5ebe4eed1c8080e52ba7d558f67296d975c0. Its 173-entry canonical map digest remains bc386b68aed8b32b82870b244d88a894cf8f8ae271c4ccfbb57bf1865a0fbad2.

Shard comparison used public payload_info metadata. The eight sealed shard paths were absent locally; a guard explicitly refused every data/sealed/ local-byte-reader call. During the commitment comparison, remote subprocess calls were also forbidden. The batch and freeze controls were hashed as their exact retained raw bytes. No sealed payload was opened, decoded or fetched. This checks custody commitments, not sealed outcome correctness.

The current archive index has 289 paths and digest 0f24960361d854a7b02e1ba626269140e98f268ee2f89125fcdea5a77a46ce20. Root's C4 snapshot was 2baa84e8e2dbbbc44034b978b07366bb2d75c368, closing the eleven-payload cohort under unchanged CODE8 packaged execution. My metadata-only historical comparison after that cleanup still matched all 173 entries.

I separately summed the retained production batch and rendered-receipt metadata, without reading row payloads. The exact current totals remain:

| Inventory | Count |
| --- | --- |
| Canonical groups | 86 |
| v1 groups / accepted records | 74 / 2,152 |
| v2 groups / accepted records | 12 / 376 |
| All accepted canonical records | 2,528 |
| Quarantined records | 0 |
| Accepted train / dev / ood / sealed | 1,576 / 355 / 381 / 216 |
| Completed rendered groups / derived rows | 24 / 2,976 |

Derived rows retain the original split/view accounting: dev listwise/computation/distillation 336/60/72; ood 360/60/72; train 1,488/240/288. These surfaces repeat canonical worlds and are not counted as additional independent scientific examples. No canonical or rendered publication occurred in this review.

## Unchanged operations and bounded scope

The CODE8 projection authority, raw derivation, remote proofs, lifecycle and storage behavior are byte-identical in CODE9. Prepared self-hashed capsules remain untrusted. Every newly activated receipt is rederived with the original full raw validator and use_controls=False; previous qualified capsules must be inherited exactly. The qualified aggregate binds a fixed snapshot ref/commit, tree mode/OID/length, raw SHA256 and Git-blob SHA1, with pre/post control and original payload catalogue ref checks. Relevant current receipt/source/specification/protected-index/custody descriptors are compared on each projected lookup. Unrelated archived cohorts do not invalidate a valid historical projection; changed covered lineage refuses it. Unprojected exports retain their full raw fallback.

Only label-free scalar/lineage metadata is persisted. Source prompts, targets, per-record labels, selected-record arrays and full provenance/gold do not enter the projection capsule. Session reuse remains root/provider/authority/deadline bounded and rehashes local aggregate bytes; a new session establishes remote proof again. This does not assert detection of every arbitrary remote mutation on every lookup after a completed session proof.

The two alternating local aggregate slots preserve the active pointer until exact new derivation, predecessor identity, remote proof and final publication guards succeed. The ordinary lock protocol and qualified owner/same-user OS trust boundary remain unchanged. CODE9 does not claim protection from arbitrary same-UID code/memory replacement outside that protocol.

The reviewed ceilings remain 16 KiB per capsule, 8 MiB per aggregate, 32 MiB total controls, two 8 MiB RAM proof-cache units, four new receipts per activation, 4,096 covered receipt entries, 1 MiB metadata reserve, 512 MiB total programme storage, a 10 GiB accepted/rendered publication floor, 180-second processing budget, 144-second rendering allowance and 340-second watchdog. Hitting a gate refuses work and retains raw payloads; no unlimited history or specification-upgrade guarantee is added.

STOP, exact initial pause bytes, current independent source/evidence approval and the shared deadline still guard preparation, backup, create-only snapshot publication, activation and pointer installation. Completed rendering cohorts still require attested controls before raw eviction, including resumed partial cohorts. No admission rule, deletion allowlist, cleanup journal, backup qualification, source guard or sealed reservation changes in this patch.

Root reported the actual CODE8 24-group control migration complete, bounded cold recovery in 37.849 seconds, and a full 2,312-row nonsealed release check passing eleven gates in 146.801 seconds. Those are prior operational observations under CODE8, not new independent throughput measurements or experiments under CODE9. The present approval's new runtime claim is the repaired standalone import boundary plus the final complete regressions.

## Frozen scientific evidence remains historical

All scientific generators, profiles, configurations, environments and oracles retain the exact CODE8 bytes. The inactive H prototype remains DEV-only; no production H profile, training, canonical prototype publication or new sealed evaluation is introduced.

The once-only H measurement remains tied to its original 61-file source freeze 4a4c0c9cbe233dba9a9857e5d70963dfd04e4b436df9448be14493afb896cd3e. The H module SHA256 remains 18751b15853905d79ed644c6ced6ce4e21a46300a189defc54465b564d3621f0 and its spec SHA256 dc2a17a16860c5087ca3a8a583bd7fd3a3c6abfb7c8b08ebc697252102ce7ea9. This later operations source map does not retrospectively become its experimental freeze.

The preserved H result is PASS for 800 evaluated DEV cases in 5.722628 seconds with zero mismatches: 96 goal cases, 128 constraint cases, 544 atom ablations and 32 baseline cases. It retains all 96 three-step goal winners, all 544 changed nonempty exact ablation best sets, the 32 complete-conjunction Q/Q-reverse ties and baseline success 32/32. Recorded goal-blind positive-add/lowest-ID membership is 7/96, the highest/longest shortcut 32/96, and constraint-blind/terminal-only membership 64/128. Baseline generation of 928 v2 rows with 896 unused rows remains explicitly separate from the 800 evaluated-case denominator.

These are mechanical checks of supplied candidate plans. Dynamics depend on developmental index, with developmental overlap disclosed; seed mainly controls identities/order. Literal a/b/c vocabulary supplies no vocabulary holdout, register/reject cases supply no ledger transfer result, and supplied-plan optima supply no global action-space claim. Formula-aware shortcuts remain possible. There is no claim of held-out learned generalization, model improvement, training, transfer or prototype production readiness.

The v2 confirmation likewise retains the original 42-file freeze fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467 and its recorded results/limits. H and v2 were not rerun in this review; their raw custody, accounting and replay regressions remain in the complete suite.

I reread all five required evidence files through the bounded read-through interface and verified their exact raw SHA256. Byte lengths remain H freeze 25,929; H STARTED 251; v2 freeze 6,375; H result 805,131; v2 result 112,806. The sole literal required_evidence map authorized by this report is:

```json
{"manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json":"3b22f95b23189bc6fa92439559c67259d6d6eef916a8a9546c28ed12aa98101c","manifests/H_COMPOSITION_EXPERIMENT_STARTED.json":"45722e9caa95880ed6cb26b0849a75c1c6da0d758a25e42e951275d4d38940ab","manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json":"4e4ec400a1aba413ce0112e7b43cd8bfd220a3fd5bbbbede70b08f1a031f27ca","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

Any changed/missing evidence, changed reviewed source byte or changed active report invalidates this scoped approval under the existing review guard. Historical catalogue, index and custody evidence remains separately bound and cannot silently broaden the statement.

## Exact approved source map

Python/configuration count: 67.

Canonical SHA256 of the complete map: 00cc8253898f8f1c108632817663d5868257e297facf33c47bcc0aa4f99fdedd.

The complete map is listed here, not inferred from the two-line diff. Operational result artifacts and this report remain separately pinned by the validation receipt; they are not silently included in the Python/configuration source-map digest.

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
  "scripts/archive_cycle.py": "155cd58e8316cf8f47c68cde92c1ba4f18ab057c8ae6cc05bcb8c1143771e330",
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
  "tests/test_surface_review.py": "c6dbc104a66122ad4a0c7255afb57c423fb7add32a988d9763cde8db05f3fe03",
  "tests/test_v2_review.py": "adb97f1e6ab879609a13492f346fdbb6afe7e4d11767b3e8f7ffeeb75268e1b0",
  "tests/test_version2.py": "5c58b03e8518932ca3536a212ff0deb4ff18c0ca100bb9f14b54745e934698be"
}
```


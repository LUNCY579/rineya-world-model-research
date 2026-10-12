# Independent rendered-surface implementation review

FINAL_VERDICT: PASS

Reviewer: fresh independent agent `/root/surface_review`. Date: 2026-10-12 UTC. This reviewer authored only `tests/test_surface_review.py` and this report, and did not author the renderer, exporter, automatic publication integration, environments, oracles, generator, or production configuration. The implementation approval is for code authored by the other agents; the included reviewer-test hash identifies its evidence.

Approved source digest: `3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12`, computed as SHA-256 of canonical JSON of the complete 31-file Python/configuration hash map below. The map was identical before and after the final complete regression run and the independent stride-6 measurements. This report can serve as the new code/config review only when its immutable validation receipt binds this exact map, digest, and report hash.

## Scope and preserved core approval

This PASS approves the proposed rendering/export layer and its automatic queue, receipt, recovery, resource, and release-inventory integration, with the regression and mechanical evidence described here. It preserves the prior independently approved core without claiming a fresh release audit of every current canonical record. It does not certify the current dataset as research-ready, authorize bypassing a resource gate, establish model learning/generalization, or claim complete shortcut resistance.

The core baseline has source digest `51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7`. Its exact review and validation receipt are to be retained at `reports/review_history/INDEPENDENT_REVIEW_v3.md` and `manifests/validation_history/VALIDATED_CODE-3.json`, respectively; their exact SHA-256 values and baseline source map appear below. The existing approved receipt matched all 25 current production Python/configuration files, its stored report hash matched the original review, and that review contained both `FINAL_VERDICT: PASS` and its source digest. The archived `reports/review_history/QUALITY_REPORT_v3.json` is historical dataset evidence rather than a new release result.

Twenty-two of those 25 existing source/configuration files remain byte-identical in the proposal. Only these existing files change: `pipelines/worker.py` adds guarded derived-view publication/recovery and approval checks; `auditors/release.py` adds committed rendered-inventory integrity; `configs/production.json` adds the three rendered-view settings. The curriculum, generator, transition/scoring environments, both oracles, canonical decision labels, old export paths, backup/controller/watchdog code, and all four core test files are unchanged. The original production configuration values remain unchanged apart from adding the rendered settings.

All 25 production source/configuration files remained unchanged during this review. All 66 pre-existing nonsealed accepted shards remained byte-identical, with collection digest `45cb9526e674ec8d2a59c9fa40e219890f40dc3961b48198fd3c9c9f8d294d02`. The proposal's 24 inherited nonsealed shards also remained identical to their corresponding production files, with collection digest `eb437507ece91fcd0acc4b9613154be2ae8b03c9461d53c4140a89cfacdd7421`. The appendix includes the historical 66-shard receipt. Production sealed gold was not opened, and this reviewer changed no production source, configuration, review, or data.

## Validation and independent challenges

The original 58 core tests passed in 1.807 seconds. The final complete suite passed **114 tests in 25.016 seconds**: 58 preserved core tests, 16 builder tests, 9 integration tests, and 31 tests authored by this reviewer.

Reproduction from the proposal programme:

```sh
PYTHONPATH=/Users/chengli/.codex/gcdl001-host/improvement-worktree python3 -m unittest discover -s tests -v
```

The hand-calculated ledger fixture independently specifies settling a +3 credit from balance 10 as utility `(1,1,-1)`, reversing it as `(1,0,-3)`, and an empty plan as `(1,0,0)`. Additional fixtures specify zero-cost ties, initially infeasible plans, and three supplied possible worlds with exact success/goal fractions of `1/3`. These expectations are not copied from generated labels. Full source execution evidence is built with the existing oracle, then the independent expected projections challenge every pointwise, pairwise, and listwise rendered target. Pair reversal changes the sign of preference and preserves oriented exact utilities; tied comparisons remain zero. The no-feasible case retains its empty best set and abstention, and missing-world arrays and exact rational targets survive rendering.

The independent checks cover complete framing, Unicode and delimiter-like strings, actual nested key orders, duplicate keys/nonfinite/extra-text rejection, exact task reversibility, source immutability, and deterministic replay. Canary values in metadata, verification, quality gates, provenance, stable record IDs, and parent/template/lineage hashes do not enter `input_text`. The model boundary is the explicit `model_fields=["input_text"]` allowlist; provenance and supervision remain separate nonmodel/target fields.

Candidate order remains independent of gold and the target-derived canonical record ID when goals and resulting labels change while seed, record index, and candidate IDs remain fixed. Search-bearing records preserve their entire candidate-prefix policy order in every applicable view. Computation retains complete matched-search supervision. Distillation retains all available search and finite-bound proof supervision; witness, proof, search-benefit, and decision labels remain outside model text. Proof-only records are not relabeled as computation examples.

Source publication challenges cover accepted-only status, missing or corrupted evidence, exact shard/receipt commitments, protected ID/semantic fingerprint/parent/template/split/seed relations, historical source-code provenance, duplicate sources, mismatched splits, and an unselected quarantined tail. Mixed accepted/quarantine batches export their accepted shard and never open the quarantine shard. An accepted subset containing only an ordinary A task commits deterministic zero-eligible computation/distillation manifests with zero rows, zero eligible sources, and no shards, preserving real source commitments and inventing no target.

Path checks cover traversal, redirected accepted/manifest directories, destination-dataset symlinks, existing artifact and receipt leaf symlinks, and temporary-file symlinks. The queue now validates containment before decoding each source manifest, including metadata that claims to be sealed. Corrupt artifact/source commitments stop recovery. Source, receipt, or executable-code drift prevents completion manifests/receipts. Receipt publication reserves storage/free space too.

Interrupted exports and refused receipts remain pending; byte-identical partial artifacts replay without modification and a complete immutable receipt repairs database loss. A real two-job fixture obeys the configured one-group limit, and an expired deadline publishes nothing. Counts distinguish derived presentations from canonical source worlds. Worker publication rechecks the exact independent approval around rendering and retains its single-worker/watchdog bounds.

Review found and reproduced destination-directory symlink escape, unguarded receipt publication, a mixed-batch/zero-auxiliary integration failure, and manifest metadata decoded before containment. The final reviewed implementation fixes these cases; their regression checks pass.

## Actual stride-6 evidence and its limits

The independent controlled schedule uses **32 DEV groups at `1000003 + 6k` and 32 OOD groups at `1000004 + 6k`, for `k=0..31`**, exactly the production scheduler strata. These generated controlled fixtures are separate from the currently published sample.

| Close-call stratum | Before slots 0 / 1, 32 decisions | After slots 0 / 1, 96 presentations | Maximum fraction |
|---|---:|---:|---:|
| DEV | 0 / 32 | 47 / 49 | 100% -> 51.0417% |
| OOD | 32 / 0 | 42 / 54 | 100% -> 56.25% |

This confirms reduction of the actual stride-6 fixed-slot correlation. It does not guarantee balance for every source or individual surface, and the three presentations repeat each canonical decision.

The separate fresh coverage agent's `reports/CONTROLLED_SURFACE_COMPARISON.md`, SHA-256 `016bfc9dd8aef7d8925c15a239c57e9737028f96d0afefe7dad6cb77138587f6`, binds the same source digest. I inspected its methods, source commitments, and scoped findings. It reads the currently committed **616 DEV/OOD records from 22 accepted shards** and measures **1,848 listwise rows after actual outer JSONL serialization and reparsing**. It finds zero parse, dual-oracle, decision-label, or source-mutation mismatches and three surviving top-level key orders per source. In that smaller actual sample, DEV close-call pooled counts change from 0/11 to 18/15, and OOD from 11/0 to 13/20. Its original-source and rendered-stream hashes are retained in that report.

The two experiments have different samples and denominators and are not combined into a model-performance claim. Individual variants retain imbalance (for example DEV reverse-indented close-call 9/2 and OOD seeded-spaced 2/9). Existing vocabulary overlap remains unchanged; the generator-C goal-blind plan-relationship shortcut remains effective. No new vocabulary holdout, generator-C redesign, model evaluation, significance test, or general decision-intelligence result is approved by this report.

## Resource and operational limits

Every derived shard is strictly smaller than 8 MiB. Source selection is bounded at 12,000 records and variants at 1..3; the automatic group budget is 12 (validated maximum 16). The shared programme ceiling is 512 MiB, and every new artifact/receipt reserves enough space to retain 10 GiB free. The current 180-second worker cycle has a 144-second soft publication deadline and retains the existing external watchdog at cycle timeout plus 160 seconds, with its termination grace. This is bounded CPU-only data publication; no latency or model-training performance claim is made.

At this review's actual resource check, free space was **9676111872 bytes**, below the required **10,737,418,240 bytes**. The real guard returned `rendered export would leave fewer than 10 GiB free`. Consequently no real rendered dataset was published under that condition. Successful publication tests used bounded temporary fixture roots and explicitly simulated healthy free space; dedicated tests independently force and verify both floor and ceiling refusal. This PASS approves the guarded implementation and does not authorize real publication or resume while the floor is unmet. A later full core release audit remains the operator's responsibility.

No training, model inference, or paid API was used. Same-user filesystem custody and independent review do not constitute cryptographic isolation, and the existing correlated-oracle/specification and synthetic-task limitations remain.

## Exact signed reviewer statement and source receipt

Signed reviewer statement: `/root/surface_review` independently approves the implementation/regression scope above at the exact source digest below. This is an attributed review statement, not a cryptographic signature. The validation receipt must bind the immutable SHA-256 of this report and the following exact source map; any source/configuration change requires fresh scoped review.

```json
{
  "final_verdict": "PASS",
  "review_kind": "Independent implementation and regression review; mechanical experiment scope only",
  "reviewer": "/root/surface_review",
  "source_digest": "3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12",
  "source_hashes": {
    "__init__.py": "64c91b2f24f24214f818c1716c5e181f86625b648914e1eadf6880e82114346e",
    "auditors/integrity.py": "1ce9a68a04d2eb71c3d3bcb8188dd3d392af69cda105b0caa408be9c6c53a62d",
    "auditors/release.py": "c49f36f518385825fa92fc0a9f097fd83fd93d18b7f2c8d87ec3b6895eedeaea",
    "configs/curriculum.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
    "configs/production.json": "ce4826f6342363100ff88c539beab79f47f3cc15b4be48c922527481ff303afd",
    "decision_labels/__init__.py": "c46518068c5e4407e3f5c1a43eed2df9a775977a055773c9263b21bdddf1e637",
    "decision_labels/rendering.py": "b9ac033f423deb4221d9c85199a967317c4d9092fb049c4cf9f4f0a7b341f700",
    "decision_labels/views.py": "3a6851fc7ba836bd66ae6c8b4f1188821c28190c61e5681ac8e75d3bad02372e",
    "environments/__init__.py": "0097b9cfbd6e2cf98c050afad5e9f8f7058739f2095792b336339750af581bc4",
    "environments/core.py": "b4e383c455ba899737a8c034a239a23ac01fbc13279791b3e0bc66acb6ec014d",
    "generators/__init__.py": "d45c107da508d34b0d040fcfe634a6f16fe2db2218244a5dfee49fa008742e11",
    "generators/build.py": "78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575",
    "generators/improvement.py": "4c4ad9b85050fa1b21a31bf71b752e217ce2661fafcf0bdd934463c01571ed0a",
    "oracles/__init__.py": "d2ff07ef6ae530121d8484a744a81e3c9c12755e01a350f533e047240cc06397",
    "oracles/independent.py": "0453985a3347ea9e503c44dff4eef9ad7f7de1a9efa88537c05d569c96a032ae",
    "oracles/primary.py": "0797623f90d1d53de1034c0c8fb07fbfc56207924b829f9945729dcd52639d19",
    "pipelines/common.py": "26eafc2cccefb424c8aff89b21cbb6cf81e73a67474c97b9749e02b942358b82",
    "pipelines/rendered.py": "b5ef28f13dad6dc0e35b37161937b2995df44f1c77c9d07b06d7dbea08c5b2d5",
    "pipelines/worker.py": "4b83c9cd71f592115eb1fa7e01dcdbbcb687d7cbcb0d166a0bed11b7e6394d1f",
    "scripts/backup.py": "f0812c0fd9dda0c44250a16ac5c42d2fa908691813fb46065f0528ac4e47eb61",
    "scripts/control.py": "9990ad7a7538dc9ce8fdd10d3db26f9a592f56f04d74ef6138f7f5054f8ea909",
    "scripts/export.py": "94e319ac4c24e6cdae8e46a0f32edefcf65fc3631f6d6b6e71d5dd63fc84cccb",
    "scripts/export_training.py": "0aefb65fd8cccc4d3e8086a437db9e8a51c6b65abb96f9803fc9b9a9588fc33f",
    "scripts/run_cycle.py": "39e2f9e5096b87b7c856ea555bb5f07b31f352b1abddba1891ae60f37da9073c",
    "tests/test_adversarial_review.py": "9dca1face6426a079b5c0b001b0e63eb68e2cfb3463e069784be6d22fc16ad59",
    "tests/test_generation.py": "0dbc041d7cbf52d9dd6bc83a7e51efe4e4fc19e6bbbcc00e964affdc436a7341",
    "tests/test_operations.py": "0015dbd1e6240045e5dfde95140b142c7a92cfdde835e647d17f990e77f4968b",
    "tests/test_oracles.py": "e1995c834809e7a0ec45531465cba382408c4e9dfe0d309f72d9c469db340ff1",
    "tests/test_rendered_pipeline.py": "ccf6293e1aedc15896e40aa14c690710001e8ef13e33027009a07ee17fa4847d",
    "tests/test_rendering.py": "fc97a6b684221b6da7f2b96035bf89aa04d2cbd37146c45dc3545e4f2fc1d9fe",
    "tests/test_surface_review.py": "79bf0c7bfb76324dd538af54f44818bae314847f4c566c059c6011a82600dc53"
  },
  "test_runs": {
    "core": {
      "count": 58,
      "seconds": 1.807,
      "result": "PASS"
    },
    "complete": {
      "count": 114,
      "seconds": 25.016,
      "result": "PASS"
    },
    "independent_added_tests": 31
  },
  "preserved_core_baseline": {
    "source_digest": "51e28002089b43743c1f889425222590c581fea182a85e3c49a55e1ab54a4df7",
    "receipt": "manifests/validation_history/VALIDATED_CODE-3.json",
    "receipt_sha256": "1c358d932048f0fb6a0a585f009cfc31d4cebfd09cde32cf06a6b442137b187b",
    "independent_review": "reports/review_history/INDEPENDENT_REVIEW_v3.md",
    "independent_review_sha256": "0a38115eb3df3ea7de6ac1443cc4e4b3a0f5d58751ab588385fe642a74586075",
    "code_hashes": {
      "__init__.py": "64c91b2f24f24214f818c1716c5e181f86625b648914e1eadf6880e82114346e",
      "auditors/integrity.py": "1ce9a68a04d2eb71c3d3bcb8188dd3d392af69cda105b0caa408be9c6c53a62d",
      "auditors/release.py": "f120a65b840b8a5a0a5e707ec8fc6897aa95dd9f48e213afd090752d7d73e2be",
      "configs/curriculum.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
      "configs/production.json": "6f5a4440538223907a00a220a899dac013ac6cbddaebdae823f7b8e73ee898e5",
      "decision_labels/__init__.py": "c46518068c5e4407e3f5c1a43eed2df9a775977a055773c9263b21bdddf1e637",
      "decision_labels/views.py": "3a6851fc7ba836bd66ae6c8b4f1188821c28190c61e5681ac8e75d3bad02372e",
      "environments/__init__.py": "0097b9cfbd6e2cf98c050afad5e9f8f7058739f2095792b336339750af581bc4",
      "environments/core.py": "b4e383c455ba899737a8c034a239a23ac01fbc13279791b3e0bc66acb6ec014d",
      "generators/__init__.py": "d45c107da508d34b0d040fcfe634a6f16fe2db2218244a5dfee49fa008742e11",
      "generators/build.py": "78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575",
      "generators/improvement.py": "4c4ad9b85050fa1b21a31bf71b752e217ce2661fafcf0bdd934463c01571ed0a",
      "oracles/__init__.py": "d2ff07ef6ae530121d8484a744a81e3c9c12755e01a350f533e047240cc06397",
      "oracles/independent.py": "0453985a3347ea9e503c44dff4eef9ad7f7de1a9efa88537c05d569c96a032ae",
      "oracles/primary.py": "0797623f90d1d53de1034c0c8fb07fbfc56207924b829f9945729dcd52639d19",
      "pipelines/common.py": "26eafc2cccefb424c8aff89b21cbb6cf81e73a67474c97b9749e02b942358b82",
      "pipelines/worker.py": "4130b27b7332c6fba9d0c58decd4e950d240817bc044a9823f96d1f8e40ee076",
      "scripts/backup.py": "f0812c0fd9dda0c44250a16ac5c42d2fa908691813fb46065f0528ac4e47eb61",
      "scripts/control.py": "9990ad7a7538dc9ce8fdd10d3db26f9a592f56f04d74ef6138f7f5054f8ea909",
      "scripts/export.py": "94e319ac4c24e6cdae8e46a0f32edefcf65fc3631f6d6b6e71d5dd63fc84cccb",
      "scripts/run_cycle.py": "39e2f9e5096b87b7c856ea555bb5f07b31f352b1abddba1891ae60f37da9073c",
      "tests/test_adversarial_review.py": "9dca1face6426a079b5c0b001b0e63eb68e2cfb3463e069784be6d22fc16ad59",
      "tests/test_generation.py": "0dbc041d7cbf52d9dd6bc83a7e51efe4e4fc19e6bbbcc00e964affdc436a7341",
      "tests/test_operations.py": "0015dbd1e6240045e5dfde95140b142c7a92cfdde835e647d17f990e77f4968b",
      "tests/test_oracles.py": "e1995c834809e7a0ec45531465cba382408c4e9dfe0d309f72d9c469db340ff1"
    },
    "unchanged_proposal_core_files": 22,
    "intended_changed_existing_files": [
      "auditors/release.py",
      "configs/production.json",
      "pipelines/worker.py"
    ],
    "baseline_code_and_receipt_match": true,
    "production_code_unchanged_during_review": true,
    "historical_nonsealed_shards_unchanged": 66,
    "historical_nonsealed_shards_sha256": "45cb9526e674ec8d2a59c9fa40e219890f40dc3961b48198fd3c9c9f8d294d02",
    "historical_nonsealed_shards": {
      "data/accepted/dev-1000003.jsonl": "91e716b03ee868b0f687c0f03274342cadffd231392ee37bec7feb198854cade",
      "data/accepted/dev-1000009.jsonl": "f8e0b6b6b1af410ddaec03b463682c194dfe584af3094ee301eacba48e87469c",
      "data/accepted/dev-1000015.jsonl": "97a45436ccc10688c0aec3b301a68eff09bb4a1ac3b4533ea2b4af14009823c7",
      "data/accepted/dev-1000021.jsonl": "3ee0d598daa18ecd495445112666a79bc2539491fb62a9bce65d1a819e0797ee",
      "data/accepted/dev-1000027.jsonl": "5b36adb1e5666ce8a1e89914d947992e217e7be3154feab69572d6e0f04d944f",
      "data/accepted/dev-1000033.jsonl": "349e02263c662e9f4162a922f80c91ce4a6a588c9c47072d388fc38e575e28ed",
      "data/accepted/dev-1000039.jsonl": "f49add536c96ba7c892737d733fa6106fa615aa638a96bda8c7011949ee0989c",
      "data/accepted/dev-1000045.jsonl": "8a9308f809980ad38ea1c7c04f6fba85d0d134e25af757825650b4827a555a98",
      "data/accepted/dev-1000051.jsonl": "77f3e25d252a0f9bf259c4761a940421158c8a38f69d03f345025b15984e0632",
      "data/accepted/dev-1000057.jsonl": "1aed03281b8cd34e9a05229d8a0271ae862c35dbb990a28cfd3db9e98f9dcff9",
      "data/accepted/dev-1000063.jsonl": "8a18f390e331137f41721c83d3bdafe7bbdcb91b74b7901a094930229796ad27",
      "data/accepted/ood-1000004.jsonl": "37585a5b67757305b8aa0e6721705ef168793d0c59a31329348254659e416b4c",
      "data/accepted/ood-1000010.jsonl": "491f790f3df928edec75d45018eca3381d2810cde35b979f1511c6777ed37b31",
      "data/accepted/ood-1000016.jsonl": "bbee4a943abb8f19cc39815de18d52147f0b654fb7e9a8f793690296667ab4b3",
      "data/accepted/ood-1000022.jsonl": "88097296d693bf5652da4faea932185b5ffe21e51c01b245d81a3a915ddf8582",
      "data/accepted/ood-1000028.jsonl": "6756f095bb8f8b02f23a08dd7628cd3dbf1b6357225a3d50617b6593a517b010",
      "data/accepted/ood-1000034.jsonl": "f67496d0b72a388a3e0b974f951223e8912ff738227ed99d7f93dfda9aefe183",
      "data/accepted/ood-1000040.jsonl": "ba2fbef28c67738da96294f96c6249d7ed46ace6e179e82c7ec9aa5862a75673",
      "data/accepted/ood-1000046.jsonl": "65766331f1570bcb5a7066768d95f021b92df60f2c408d657635409f36c445d6",
      "data/accepted/ood-1000052.jsonl": "0b5b3b10c4aec5f5b3f5ba2eb2d97f34b6a11590b99c33ae22b576f90e64b636",
      "data/accepted/ood-1000058.jsonl": "76fb73ffd1778569b8c4a72b1b74f4c442d67bcf497963f74e03864b1470cc28",
      "data/accepted/ood-1000064.jsonl": "068bae9885a03cf9b775b91b56f1e9f1e359a0b7366f091397a6f478ef7a5f0f",
      "data/accepted/train-1000000.jsonl": "1f9e3da1550b1e81541f78510f947c025808238ec605b16d67840779e16d8959",
      "data/accepted/train-1000001.jsonl": "380c5e23a60b14bdd897396dccf70ea1e75f6ab4918db09f12f783c1c7fabfc2",
      "data/accepted/train-1000002.jsonl": "11288495b4ad5292901a99cca269d1520de131fb464fe94e3dd2910f9498e6e9",
      "data/accepted/train-1000005.jsonl": "2414a770fd6bfc16c5ef82f86360b4bebbce22788e8de858873285fe2863387b",
      "data/accepted/train-1000006.jsonl": "6b6d3a010789216b92f0f72193dbcdcc1b77d0d1728484523ad5af510e0c160e",
      "data/accepted/train-1000007.jsonl": "614a8df08be069f36f4daff7308781ce0ed1abac3528a86610f89e3dbdbd7d62",
      "data/accepted/train-1000008.jsonl": "9dd81f822e4c2294f4ea7d5a6e88d58a449d35d81557955947bef3aa1c310f60",
      "data/accepted/train-1000011.jsonl": "b92bc94ba2aaa2b662ac69c1b48d52591dcc2f1c5fed807ade850a278e6906be",
      "data/accepted/train-1000012.jsonl": "fb6616b3ae1ed4d6c361c31874c60913abdf0264ae888b4ceacdacc0e2011b08",
      "data/accepted/train-1000013.jsonl": "40786a9e460f2dab0dbebcfb54606939ae577b5cbe9479ac6b4693db48766af8",
      "data/accepted/train-1000014.jsonl": "6355cf809d59c99c8d72040a753c2d4ddc63c3a8a9dd3a63d3302eb1adb91c09",
      "data/accepted/train-1000017.jsonl": "acc9857deed6457245017f1cfa30c6b68de8793069e6dfb49bebaf968b341b8e",
      "data/accepted/train-1000018.jsonl": "8cfdcd02df2a7078eaa5e318d6cb00510b9cd5df92596f79230ad3a7053d6b9b",
      "data/accepted/train-1000019.jsonl": "2d29c4f6fea20ce1fc3d35303d2532f15d974c2894e9f595b5cb305b06ba73e6",
      "data/accepted/train-1000020.jsonl": "58be51700bc0b3ddc30dac4f123a2fe029bc620a04257ac0897ea3a81297d560",
      "data/accepted/train-1000023.jsonl": "3e9dc5ed1e3162d8032034181ccd910992c46515fe44349fd655e5f5c7a7192c",
      "data/accepted/train-1000024.jsonl": "1cbd4043dde5d0434c38ab4d6bb5df5000b2cbd12dd757c7f3313797446e0169",
      "data/accepted/train-1000025.jsonl": "4cf4094f6f261412cd629a71b952ac8a4e7c67471e5ab174403df75c625baf48",
      "data/accepted/train-1000026.jsonl": "cda8195767a01f995980907519b7463f0ad03e843e61e50c31f98e5ef1fdc771",
      "data/accepted/train-1000029.jsonl": "7d28af0e4b5bad33da9af482ab35eea0434641c2887a8b660f48529f7e59beab",
      "data/accepted/train-1000030.jsonl": "4ff7474929755ffc2d873f6d85269d087055e640118ad8fc8c6669f11fb80e53",
      "data/accepted/train-1000031.jsonl": "2fb0e4721390622dfbe1a97dae126cd2d411aa2bb0bcd329e0d62a664f095be2",
      "data/accepted/train-1000032.jsonl": "0d87b881f9712e7b3a74e27d6194b3ee7d6e9e3aedce72f2e23b0a6872af5c91",
      "data/accepted/train-1000035.jsonl": "28e800052b245858acdc978ad54b80c56f53013d9be2b4897dc6edac1c00e5cb",
      "data/accepted/train-1000036.jsonl": "0c84ed032eeeda7bd7c2ceb52c51e2623d5cc8bd633638bb4cda0a68b8f86653",
      "data/accepted/train-1000037.jsonl": "bfad9c054c787cbdc5594fc7481b794f79c4a8488dd9e312edaec59c75543b01",
      "data/accepted/train-1000038.jsonl": "04807131219a956990c8b9731c238a808325235f377874df1b5c6d70642d1e70",
      "data/accepted/train-1000041.jsonl": "d4720e28632c0220fac2bff92653ca04d368a4ad2e30ea6890a170392db653d8",
      "data/accepted/train-1000042.jsonl": "be22704a51ea5516c1804b6982f58baea2205baa07d5a98c0eb1ab38beda9714",
      "data/accepted/train-1000043.jsonl": "d869e293b567e17b350ef0e40f6498f9bde08a0510a675935bd02a68b0b37c67",
      "data/accepted/train-1000044.jsonl": "bc6362040edaf8c80c7c5b4709108da4bd160f44f06bf3c9bff147964b163734",
      "data/accepted/train-1000047.jsonl": "6ce39d195e25c8de63e65d7db58f83505a0f920120fef766238674ad59221704",
      "data/accepted/train-1000048.jsonl": "aad6106a8fe030a6e838441205f143fd011b48670670c48a392fa792688f0259",
      "data/accepted/train-1000049.jsonl": "7d103642bf4e12063367cb1f8fc313fdad413b35c3ea94e66a69db5fd9e09f84",
      "data/accepted/train-1000050.jsonl": "7d08171a8f63cc8ba3d82e6e3c4d279b6c7bf8772a02d78364f22fc1b3049a7e",
      "data/accepted/train-1000053.jsonl": "b6f13f141c6f4171537667de12a6d2071f6c5e8b3e48560f49c88d76e81c683a",
      "data/accepted/train-1000054.jsonl": "c274d805b7114b5dc8419732737dd2946e442bc276644a64aa00a917d94bf959",
      "data/accepted/train-1000055.jsonl": "7c7160725b847af635e3f460badda46c2ea5ed7282ba660b64f9c34b10665383",
      "data/accepted/train-1000056.jsonl": "92a9896e84fbd1764ccf53d8e76a4a714e045c333816b05230084cc4add97ace",
      "data/accepted/train-1000059.jsonl": "41ad295703a60f247e33ed5bc93278a860536fa335614ce93f17fc11db36b1d6",
      "data/accepted/train-1000060.jsonl": "fdd82bd182ded5c077a345c2a35d60891c97ee6840a4530b47f9305af7ad7766",
      "data/accepted/train-1000061.jsonl": "bfabcc0524d1993827478c1cab64b583733afd66b9012c4c44aed3a04d27c80d",
      "data/accepted/train-1000062.jsonl": "239174cb472c877d74b0521bf26c26b4efc9a0571ba7173cb5ff867c7241f580",
      "data/accepted/train-1000065.jsonl": "8999460c072af16ac2bda3b6e8375142b9cdd18f90d1fdc3491f3a249b7cc9fd"
    },
    "inherited_proposal_shards_unchanged": 24,
    "inherited_proposal_shards_sha256": "eb437507ece91fcd0acc4b9613154be2ae8b03c9461d53c4140a89cfacdd7421"
  },
  "source_snapshot_stable": true,
  "stride6_control": {
    "dev": {
      "after": {
        "0": 47,
        "1": 49
      },
      "after_max_fraction": 0.5104166666666666,
      "before": {
        "1": 32
      },
      "before_max_fraction": 1,
      "seeds": [
        1000003,
        1000009,
        1000015,
        1000021,
        1000027,
        1000033,
        1000039,
        1000045,
        1000051,
        1000057,
        1000063,
        1000069,
        1000075,
        1000081,
        1000087,
        1000093,
        1000099,
        1000105,
        1000111,
        1000117,
        1000123,
        1000129,
        1000135,
        1000141,
        1000147,
        1000153,
        1000159,
        1000165,
        1000171,
        1000177,
        1000183,
        1000189
      ]
    },
    "ood": {
      "after": {
        "0": 42,
        "1": 54
      },
      "after_max_fraction": 0.5625,
      "before": {
        "0": 32
      },
      "before_max_fraction": 1,
      "seeds": [
        1000004,
        1000010,
        1000016,
        1000022,
        1000028,
        1000034,
        1000040,
        1000046,
        1000052,
        1000058,
        1000064,
        1000070,
        1000076,
        1000082,
        1000088,
        1000094,
        1000100,
        1000106,
        1000112,
        1000118,
        1000124,
        1000130,
        1000136,
        1000142,
        1000148,
        1000154,
        1000160,
        1000166,
        1000172,
        1000178,
        1000184,
        1000190
      ]
    }
  },
  "observed_comparison": {
    "report": "reports/CONTROLLED_SURFACE_COMPARISON.md",
    "sha256": "016bfc9dd8aef7d8925c15a239c57e9737028f96d0afefe7dad6cb77138587f6",
    "source_digest": "3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12",
    "actual_nonsealed_sources": 616,
    "serialized_reparsed_listwise_presentations": 1848,
    "parse_oracle_label_mutation_errors": 0
  },
  "actual_resource_guard": {
    "free_bytes": 9676111872,
    "reason": "rendered export would leave fewer than 10 GiB free",
    "status": "REJECT"
  },
  "production_sealed_gold_opened": false,
  "training_or_models_launched": false,
  "paid_apis_used": false
}
```


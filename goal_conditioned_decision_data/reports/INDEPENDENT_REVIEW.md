# Independent generator-v2 and profile review

FINAL_VERDICT: PASS

Reviewer: fresh independent agent `/root/v2_review`, 12 October 2026 UTC. This is a manual, scoped implementation approval after source inspection, adversarial checks, complete regressions, and inspection of the frozen DEV evidence. Executable `PASS` fields alone were not treated as approval.

Approved source digest: `fdfd63b45b071928080ac14e730f4f17df107dc6ffa6fd1b950a7ecd5e34b467`, the SHA-256 of canonical JSON of the complete **42-file Python/configuration hash map** appended below. The frozen experiment, the root's complete regression run, and this reviewer's independent complete regression run bind that same map. This reviewer's map was identical before and after its final run.

The reviewer authored `tests/test_v2_review.py` with 39 independent methods and this report. At the parent's explicit request, the reviewer also corrected two existing fixtures in `tests/test_adversarial_review.py`: an opaque sealed-byte sentinel now occupies the required temporary sealed path and trusted v1 template, and a malformed-hash fixture now has valid custody metadata so it still tests hash refusal before inventory mutation. The opaque content and original safety assertions are preserved. The reviewer did not implement the profiles, generator, namespace transformer, oracles, worker, renderer, or experiment harness. No production file was written and no production sealed labels were opened.

## Approval boundary and compatibility

This PASS approves the reviewed profile resolver, v2 generator and namespace policy, explicit job/manifest profile custody, parent-atomic v2 acceptance, v2 goal-bundle rendering policy, bounded experiment harness, and evidence-bound promotion guard. It preserves v1 mathematical semantics and provenance; it does not approve a resource-gate exception, certify dataset readiness, or establish learned planning/generalization.

The approved production baseline remains digest `3249586daa51d27188636dfe7e342ad8661ea51fbf40afdc7ef3a2becb738b12`. Twenty-three of its 31 source/configuration files remain byte-identical in the proposal. The eight changed existing files are `auditors/integrity.py`, `auditors/release.py`, `configs/production.json`, `decision_labels/rendering.py`, `pipelines/rendered.py`, `pipelines/worker.py`, and the two fixture-bearing test files. The v1 generator, current v1 curriculum, both executable oracles, transitions, rational labels, record-ID algorithm, and original semantic normalizer remain byte-identical.

The v1 curriculum canonical digest remains `d98651723bc9469105627dc55d78f0217eea67b98428020a5b1552452b4ca2f2`; its current and archived file-byte digest remains `c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780`. The v1 generator remains `78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575`. Canonical and file-byte hashes are checked separately.

All **1,936 nonsealed accepted rows in 66 production shards** validated through the archived resolver without changing their canonical bytes or original record/batch source maps. Their collection digest remains `45cb9526e674ec8d2a59c9fa40e219890f40dc3961b48198fd3c9c9f8d294d02`; the nonsealed batch-receipt collection remains `5cb4454f6d927e13b35914814981199771bf1c3c70f98ed61ce85f4bc65fbb07`. The original production source map remains the approved 31-file baseline. Its existing validation receipt and surface review remain byte-identical, respectively SHA-256 `7b122cf562596144146c585be6b77348b715a31fdac3e856a3b3335fbabcf63c` and `0e1a6647ce4834b91852cae059a876a0126ec641739fc27d8dc49a6ce5f3ecac`.

Six predeclared v1 TRAIN/DEV/OOD output commitments, including the actual production seed strata, match exactly. Three production batches at TRAIN 1000000, DEV 1000003, and OOD 1000004 reproduce all **86 IDs, ordered candidate bytes, exact targets, provenance, and protected fingerprints**. Nine pointwise/pairwise/listwise fixtures, totaling 129 v1 presentations, match their complete previously captured output commitments and retain `gcdl-rendering-1.0` identities. This is bounded reproduction, not a claim that every archived job was regenerated.

## Independent validation

The final independent complete suite passed **189 tests in 136.504 seconds**, with zero errors/failures and an unchanged 42-file source map. The root's separate complete run passed the same 189 tests in 75.309 seconds. Reproduction command:

```sh
PYTHONPATH=/Users/chengli/.codex/gcdl001-host/improvement-worktree python3 -m unittest discover -s tests -v
```

Hand-specified ledger expectations exercise status goals, forbidden statuses, costs, void reversal, and missing worlds with exact fractions. A consistent permutation of status values preserves labels and fingerprint; changing only the semantic role declarations changes the actual dynamics and fingerprint. Explicit ledger/block/table/register mapping fixtures cover state and possible-world references, goals, constraints, every operation argument, and invalid references that must remain invalid. Structural keys and opcodes remain intact.

Independent four-block fixtures verify illegal reverse execution rather than assuming reversal validity. A cheaper supplied prefix remains the exact best candidate when it reaches another endpoint; a legal reverse tie remains in the complete best set. A manually defined clear-block executor separately checks generated utilities. Changing fabricated gold changes canonical record IDs while leaving candidate IDs, order, inference input, and parent identity unchanged.

The checks reject unknown or mixed profile tuples, canonical/file hash confusion, changed archive bytes even when parsed JSON is unchanged, registry tampering, archive/config-directory symlinks, forged namespace maps, missing/noninjective/cross-split mappings, false source commitments, and forged parents. Profile-specific size gates reject invalid ledger/register sizes and seven-block or wrong-reservation forests before calling the factorial normalizer.

Cross-record checks reject incomplete triplets/pairs, duplicate goals/indices, altered non-goal inputs, overlapping best sets, missing full-curriculum families, and mixed profiles. A single corrupted v2 goal quarantines the entire supplied v2 batch. Legacy queued jobs retain v1 when the active default changes; explicit v2 jobs use the same trusted adapter/configuration for both generation runs. An existing batch cannot be replayed under another profile before generation or candidate-cache publication. New manifest-v2 commitments cannot be removed to infer v1, including under a forged legacy header; protected template indices must match the resolved profile. Legacy v1 recovery remains metadata-only and accepts opaque temporary sealed bytes without parsing labels.

The independent scheduler check uses **32 DEV seeds `1000003 + 6k` and 32 OOD seeds `1000004 + 6k`, `k=0..31`**. It verifies exact legal utilities, goal-only triplet equality, disjoint best sets, fixed candidate IDs/order, and all three rendered variants. Rendered candidate order and seeded nested key order remain identical after removing the goal. These fixtures are separate from the official pilot and are not additional independent pilot observations.

Review found and challenged late partial-parent acceptance, changed-profile replay work before refusal, unbounded pre-validation normalization, profile-stripped recovery downgrade, and optional v2 promotion evidence. The final implementation closes these cases and the targeted regressions pass. Approval under active v2 requires both frozen evidence files, their exact hash map named in this report, and matching file hashes; receipt-only rehashing/removal cannot broaden the statement below.

## Frozen DEV experiment

The source, specification, profile, and lexicons were frozen before the one official measurement. Specification SHA-256 is `19d88bd2da381806c481b88ade73dedf411ebcc07d154054b94d8733ed4b5b3a`; lexicon canonical SHA-256 is `5913a57e252ea84e249fdede5fd6ba3dc89b3dd1b2a025c2c26edf6322bf8024`. Completed-result reuse returns the existing committed result rather than rerunning the measurements. This reviewer inspected the frozen spec, source map, result commitments, and all 64 parent receipts and recomputed membership, tie, singleton-recovery, and vocabulary-overlap accounting without rerunning or tuning the official seeds.

Required evidence, verbatim canonical JSON:

```json
{"manifests/V2_EXPERIMENT_FREEZE.json":"b79b718d82040eff44ea5694235ea1329c30b8a98d2a75c02e4066c58c7313f0","reports/V2_CONTROLLED_EXPERIMENT.json":"ba1fe158612645ee017feefa3403676bf4efadc05050834c6e4ed5e7de15daf3"}
```

The run used exactly C seeds 420000–420031 and V seeds 421000–421031, no replacement seeds, at most 24 endpoint proposals per C parent, one CPU worker, and **160 tasks in memory**. It completed in **36.249 seconds**, below the 300-second cap. All 32 C parents were valid, with zero exclusions; all 32 vocabulary pairs were valid. There were zero differential-oracle, stored-target, or deterministic-reproduction mismatches, 288 rendered goal-only checks, and zero canonical records published.

The parent-receipt check finds 128 distinct semantic task fingerprints: 96 C tasks and 32 lexical parents whose two members intentionally match. There are zero semantic duplicates across different parents.

| Fixed selector | V2 best-membership / 96 | Singleton exact-best-set recovery / 96 |
|---|---:|---:|
| Supplied prefix/reverse relationship, opaque-ID tie break | 32 | 29 |
| First option | 19 | 17 |
| Shortest plan, then opaque ID | 0 | 0 |
| Longest plan, then opaque ID | 25 | 23 |
| Lowest opaque ID | 16 | 15 |

The matched v1 relationship selector was best-membership correct for **26/32** and exactly recovered a singleton best set for **21/32**, with five tied tasks. V2 has three tied goal tasks. V1 C supplies five candidates; v2 supplies six, so the v1/v2 comparison includes a changed construction. The clean goal-only comparison is within each v2 triplet: each fixed selector chooses the same ID for three full, distinct, non-initial goals; their exact best sets are disjoint, so membership is structurally bounded by one-third. The observed primary selector meets that bound at 32/96. This says nothing about a selector returning all candidates or a goal-aware model.

No rendering or goal task is treated as a new independent parent for uncertainty. For the 32 baseline parents, the descriptive 95% Wilson interval for 26/32 is 64.69%–91.11%; for zero C exclusions among 32 attempts it is 0%–10.72%. These intervals describe this bounded sample. The one-third result is a structural statement, not an IID confidence interval over 96 tasks.

The 32 lexical pairs preserve exact decision labels and normalized fingerprints. Their reserved DEV members exercise 48 distinct complete namespace strings with zero overlap against the TRAIN pools. Familiar controls remain a separate DEV stratum and do not count as lexical holdout. All 384 declared strings across the six namespaces/four splits have the same eight-letter shape, with no global collisions or structural/opcode collisions. Complete-string holdout does not establish unseen tokenizer pieces or natural-language meanings.

## Limits and operational disposition

The proposal retains finite candidate universes, synthetic grammars, small forests, endpoint-derived goals, and shared specification ancestry between separately implemented oracles. Goal-aware terminal-structure shortcuts may remain. No model training, inference service, paid API, research publication, learned transfer, causal generalization, or unbounded optimality result is included in this approval. Same-user filesystem custody is not cryptographic isolation.

The real publication guard was tested read-only against production and returned `resource gate: publication would leave fewer than 10 GiB free`; the observed free space was 3,248,496,640 bytes against the 10,737,418,240-byte floor. Successful publication checks used bounded temporary fixtures and explicitly simulated healthy space. Production publishing must remain paused while the real floor fails. This PASS permits the exact reviewed implementation to proceed through the normal promotion workflow; it does not authorize bypassing the pause or changing archived bytes.

## Approved source hash map

```json
{
  "__init__.py": "64c91b2f24f24214f818c1716c5e181f86625b648914e1eadf6880e82114346e",
  "auditors/integrity.py": "b6fb75a24bb27ac5f0000dd36d8e89968ad39975e21e103bda8658fe830424f9",
  "auditors/release.py": "23f9e3c3b4055de7b82b3343b0f597e339759dbd5059a455d240113f14359d7f",
  "configs/curriculum-v1.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
  "configs/curriculum-v2.json": "90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd",
  "configs/curriculum.json": "c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780",
  "configs/generation_profiles.json": "f54978838729e85eb849724d2b2c512da65b9969b8bd4e9876aab7271cf9388a",
  "configs/production.json": "56dfce1076eb8eddf53b1be88a637ca98fc0e05a320f8ac8fcd5dc5179f5065d",
  "configs/v2_experiment.json": "19d88bd2da381806c481b88ade73dedf411ebcc07d154054b94d8733ed4b5b3a",
  "decision_labels/__init__.py": "c46518068c5e4407e3f5c1a43eed2df9a775977a055773c9263b21bdddf1e637",
  "decision_labels/rendering.py": "efd1dde1e9e028e7a2ad8bb25d5939bc66e00e3d7c9c02e115e8c8f90be04ed0",
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
  "pipelines/rendered.py": "53c0316f785e77d576a657fd439a6a2ac67025db47b2ce12f5e8085221c90827",
  "pipelines/worker.py": "6365316fc41fd9c08a094f0263d10e56e7318f7c7d9306c9f697dc69de624e9e",
  "scripts/backup.py": "f0812c0fd9dda0c44250a16ac5c42d2fa908691813fb46065f0528ac4e47eb61",
  "scripts/control.py": "9990ad7a7538dc9ce8fdd10d3db26f9a592f56f04d74ef6138f7f5054f8ea909",
  "scripts/export.py": "94e319ac4c24e6cdae8e46a0f32edefcf65fc3631f6d6b6e71d5dd63fc84cccb",
  "scripts/export_training.py": "0aefb65fd8cccc4d3e8086a437db9e8a51c6b65abb96f9803fc9b9a9588fc33f",
  "scripts/run_cycle.py": "39e2f9e5096b87b7c856ea555bb5f07b31f352b1abddba1891ae60f37da9073c",
  "tests/test_adversarial_review.py": "1ffc7e030588dea8fdbfc5fd6c86cff64b903e8794048757510d6610a5beea8a",
  "tests/test_experiment_custody.py": "52d99b0b1b119aa28d975aff81c9484e48ba01e57202e154b3142f0b65abc3e6",
  "tests/test_generation.py": "0dbc041d7cbf52d9dd6bc83a7e51efe4e4fc19e6bbbcc00e964affdc436a7341",
  "tests/test_operations.py": "1241c4c38ea060ccde90a69b091d5ee3e93de68dd2381aaeb1ff5fbe9e094c3a",
  "tests/test_oracles.py": "e1995c834809e7a0ec45531465cba382408c4e9dfe0d309f72d9c469db340ff1",
  "tests/test_profiles.py": "d1758c6150b3fb2f2dd75b91571fb17a8c3a3c31d94e3a197fb83a8720a9538a",
  "tests/test_rendered_pipeline.py": "ccf6293e1aedc15896e40aa14c690710001e8ef13e33027009a07ee17fa4847d",
  "tests/test_rendering.py": "fc97a6b684221b6da7f2b96035bf89aa04d2cbd37146c45dc3545e4f2fc1d9fe",
  "tests/test_surface_review.py": "79bf0c7bfb76324dd538af54f44818bae314847f4c566c059c6011a82600dc53",
  "tests/test_v2_review.py": "37420f7cb7856b266f73107d87cfd9f77d543803098c0ff655e7cb9695110971",
  "tests/test_version2.py": "5c58b03e8518932ca3536a212ff0deb4ff18c0ca100bb9f14b54745e934698be"
}
```

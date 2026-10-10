The NEW prospective archival stage runner is prepared and tested, but no production inventory or stage has run.

Expected parent: `4d0b5eff1af3184fedf7fa25ada729a6cc909268`. Actual author/preparer/test runner: `/root/backup_manager`; eventual stage runner: `/root`. Same-model independence, no model-diversity claim or data approval.

Before invoking this source, create and freeze a NEW full inventory and a NEW execution config. Include every prior 2,685 tracked path, all completed postcut artifacts, exact 222 immutable release bindings across capped files and oversized raw exclusions, all current raw packages, all prior compressed history, and only explicitly named genuinely unfinished new paths. The stage does not build or authorize an inventory.

The required execution config is a JSON object with these fixed fields:

- `schema`: `prospective-archival-stage-v3-execution-v1`
- `author_execution`: `/root/backup_manager`; `runner_execution`: `/root`; `grants_data_approval`: false
- `expected_parent`: the full parent above; `prior_spec_sha256`: `6b310f0656a468dd65920c64f15efe82e0b592d3d76250366f9555cef46aaad5`
- `prior_tracked_path_count`: 2685; `prior_release_file_count`: 210; `expected_release_file_count`: 222; `readonly_reference_count`: 65
- `spec_sha256`: actual NEW spec SHA; `checkpoint_id`: that actual spec ID
- `source_freeze_sha256`: `ef30d7677c722626d4bb48d03441f1328f720a692713a08cfbffa24ef36e9999`
- `custody_adapter_sha256`: `7978cab638a00c185a9d391259c6a5c77b8a5c97ea5fefdc79fa67a31e58e023`
- `exception_config_sha256`: `bd6639e10289bcbd7c27e3bf68241e83dbf7f0d9985400b7abf9df9ae9b9fa4b`
- `release_files`: sorted list of all 222 exact path/size/sha256 objects from the disjoint spec.files + spec.raw_size_exclusions union
- `raw_lossless_required_count`: actual NEW raw-exclusion count, equal to spec declaration. Current root reports31; determine/bind actual count at cut.
- `readonly_reference_files`: all65 absolute path/size/sha256 records. Their hashes/paths MUST equal frozen `quality_reports/RGM_RENDERER_ARCHIVAL_BOUNDARY_PREFLIGHT.json` SHA03ec34f4f778698c9dc52b3bd4624f324abc525aaca1e33417d4c6042d8168a1.
- `historical_raw_relocations`: the exact single published audit relocation object: old_raw_path=`audits/diversity/belief-gap-v4-belief-controls.json`, old_raw_bytes=123141333, old_raw_sha256=279beb55eff0c5b6ea60c04d614a67d00606c25c4d1dbdaf99b8ae4ef825fe25, current_raw_sha256=14909842a90afdb3de9d78b7e7ff843f9c0ecdb0aa8e5e4bff927850513323ee, historical_raw_paths=[`audits/diversity/history/belief-gap-v4-belief-controls-279beb55eff0c5b6ea60c04d614a67d00606c25c4d1dbdaf99b8ae4ef825fe25.json`,`handoff/report_history/belief-v4-publication-attempt01/279beb55eff0c5b6ea60c04d614a67d00606c25c4d1dbdaf99b8ae4ef825fe25-belief-gap-v4-belief-controls.json`]. Both archives must have exact old bytes and standard matching recovery packages; old gzip/manifest bytes stay exact in capped inventory. Current149 requires its distinct matching package. Other prior raw changes fail closed.

The future inventory builder must separately verify old279 gzip decode against archived279, retain exact old279/current149 manifest/path distinction and reject any other stale manifest. This stage's lossless checker consumes the exact NEW raw mappings; it does not reinterpret stale manifests.

Reviewable command template (placeholders are not executable inputs):

```sh
python3 -B /Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs/run_provisional_archival_checkpoint_v3.py \
  --spec ABSOLUTE_NEW_SPEC --expected-spec-sha256 ACTUAL_NEW_SPEC_SHA \
  --source-freeze /Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs/PROSPECTIVE_ARCHIVAL_V3_SOURCE_FREEZE.json \
  --expected-freeze-sha256 ef30d7677c722626d4bb48d03441f1328f720a692713a08cfbffa24ef36e9999 \
  --custody-baseline /Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json \
  --expected-custody-sha256 7978cab638a00c185a9d391259c6a5c77b8a5c97ea5fefdc79fa67a31e58e023 \
  --exception-config /Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs/PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json \
  --expected-exception-sha256 bd6639e10289bcbd7c27e3bf68241e83dbf7f0d9985400b7abf9df9ae9b9fa4b \
  --execution-config ABSOLUTE_NEW_FROZEN_EXECUTION_CONFIG \
  --expected-execution-config-sha256 ACTUAL_NEW_EXECUTION_CONFIG_SHA \
  --expected-parent 4d0b5eff1af3184fedf7fa25ada729a6cc909268 \
  --receipt ABSOLUTE_FRESH_STAGE_RECEIPT --local-commit-receipt ABSOLUTE_FRESH_LOCAL_COMMIT_RECEIPT \
  --ledger ABSOLUTE_FRESH_STAGE_LEDGER --message 'Actual archival checkpoint message' --runner /root --execute
```

Declare any NEW ignored inventory paths individually using `--allow-ignored-path` only after exact check-ignore proof. Already tracked ignored paths need no extra force declaration. No ignore/whitespace config change is implemented.

The real unrestricted Git diff-check MUST return exit2, exactly six warnings at only the original two fixture paths, exact bound line payloads, and empty stderr. Its full actual bytes/hash/exit remain in the receipt; the report explicitly says NONZERO_EXIT_2_NOT_AN_UNQUALIFIED_PASS. Every other changed path requires strict exit0 with no diagnostics. Missing/additional warning, source/target/index change, outside-prefix change, unrelated worktree spill, parent/private drift or insufficient 2x full inventory+512MiB reserve fails before commit/push.

A successful stage ends only PUSHED_REMOTE_BYTES_VERIFICATION_PENDING. Root must run unchanged bounded recover_remote_checkpoint.py with two workers for every actual unique remote blob. Only after its completed actual proof may unchanged verify_archival_remote_lossless.py fetch/decode all NEW raw-count gzip artifacts, then a fresh source-bound custody join establish BACKED_UP.

All preparation/test/adapter outputs and this usage/receipt are LOCAL post4d inventory until a later exact checkpoint. No source/catalog/living/review/release mutation, production inventory/copy/stage/commit/push, or remote payload fetch occurred in this preparation.

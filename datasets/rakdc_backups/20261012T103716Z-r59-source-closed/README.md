# RAKDC private archival snapshot
This backup preserves original dataset and evidence/record bytes. It is an administrative archive, not a newly admitted dataset release. The research goal remains active. See CHECKPOINT.json for exact reviewed, provisional, pending and historical qualification status.

Archive release: https://github.com/LUNCY579/rineya-world-model-research/releases/tag/rakdc-backup-20261012T103716Z-r59-source-closed
Dedicated branch: backups/rakdc/20261012T103716Z-r59-source-closed
Selected DATA files: 401
Selected original DATA bytes: 58247806
The manifest records each original relative path, SHA256, size, permission mode, mtime_ns and archive shard. Compression changes storage only; restored member bytes must match the manifest exactly.

## Restore
Authenticate as the repository owner or an authorized collaborator. Retrieve MANIFEST_FILES.jsonl.gz at the recorded commit, select the required original path, download only its named release asset, verify the asset SHA256 from PREPARED.json, read that regular tar member, verify its original bytes/size/SHA256, and restore permission mode and mtime_ns from the manifest. Do not execute archived foreign programs or use protected evaluator material. Restored inode/device/ctime history is a new administrative event, not continuity with an earlier native-stat gate. A complete manifest is necessary to restore a file; a release URL alone is insufficient.

## Standing storage policy
Below 10 GiB free counts as tight storage. During active work, check before substantial generation/staging. Close or checkpoint writers; create a bounded original-byte inventory; back up selected datasets and relevant records to the verified private repository; freshly fetch and verify complete remote bytes, member coverage, sizes, hashes and archived modes; only then delete exact unchanged verified local copies. Never delete unverified, changed, extra, active-writer, protected-program/journal or unrelated files. Keep a small recovery index. Preserve Source access rules, sealed-evaluation boundaries, historical failures/HOLDs and scientific admission gates. No paid API, native weight update, main mutation or force push is authorized by this archival policy.

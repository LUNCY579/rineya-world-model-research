# Backup execution records

`PREFLIGHT.json` records the real independent Backup Manager preflight. The repository is private and writable; the dedicated worktree shares the existing clone. The original checkout remains on its existing branch.

`checkpoint_backup.py` requires an approved exact-byte inventory. Each entry lists a factory-relative path, SHA256, and size. The checkpoint specification must identify an approving agent and passing validation, with approval type `reviewed_release`, `provisional_checkpoint`, or `code_review_milestone`. These values describe checkpoint authorization; they do not confer scientific approval on individual samples.

Run safety tests without generating cache files:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 backup_logs/test_checkpoint_backup.py
```

Run a checkpoint validation with the spec file and a descriptive commit message, then run the same command without `--validate-only`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 backup_logs/checkpoint_backup.py --spec backup_logs/CHECKPOINT_SPEC.json --message "datasets: preserve reviewed pilot and reproducible factory" --validate-only
```

The helper rechecks repository privacy and permission, branch identity, source hashes, file sizes, secrets, free storage, and remote ancestry. It copies only listed files, verifies copied bytes, stages only factory paths, records a local commit before pushing, and compares every listed file against actual GitHub blob bytes. Files over 25 MiB are preserved locally and block the checkpoint until modest compressed shards are approved.

An incomplete push keeps the local commit and writes a receipt. Resume that exact commit using `--resume-commit COMMIT_SHA` with the same unchanged inventory. A failed copy keeps the dirty dedicated checkout for human or agent inspection; no automatic reset, clean, deletion, force push, or merge occurs.

Receipts are created after the verified commit and are therefore local until the next checkpoint. A `BACKED_UP` receipt certifies its listed inventory only. It does not falsely claim that its own later receipt bytes are included in that commit. Verified receipt files and release directories are immutable; create a new checkpoint/release version to correct them.

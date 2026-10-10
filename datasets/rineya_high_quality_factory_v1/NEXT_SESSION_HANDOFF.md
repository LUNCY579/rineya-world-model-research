# Next-session handoff

Read PROJECT_STATUS.md, quality_reports/CURRENT.json and backup_logs receipts first. Inspect original repo/worktree status; never switch or clean the research checkout.

- RBM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (all-pilots).
- RCM-001 v0001: 240 historical records, 240 active; ACTIVE, BACKED_UP (first-approved).
- RCM-001 v0002: 1200 historical records, 1200 active; ACTIVE, LOCAL_ONLY (verification pending).
- RDP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (all-pilots).
- REM-001 v0001: 120 historical records, 120 active; ACTIVE, BACKED_UP (all-pilots).
- RGM-001 v0001: 72 historical records, 72 active; ACTIVE, BACKED_UP (all-pilots).
- RGM-001 v0002: 360 historical records, 360 active; ACTIVE, LOCAL_ONLY (verification pending).
- RJP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (all-pilots).
- RMM-001 v0001: 76 historical records, 76 active; ACTIVE, BACKED_UP (all-pilots).
- ROM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (all-pilots).
- RSM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (all-pilots).
- RVM-001 v0001: 60 historical records, 0 active; SUPERSEDED, BACKED_UP (first-approved).
- RVM-001 v0002: 300 historical records, 300 active; ACTIVE, LOCAL_ONLY (verification pending).
- RVM-001 v0003: 60 historical records, 60 active; ACTIVE, LOCAL_ONLY (verification pending).

- all-pilots: BACKED_UP; commit 265f768f0658942b0004130eef626cdb122eddbd.
- first-approved: BACKED_UP; commit 8c2571e51e43ed206d111321482367802b79600a.
- foundation-r2: BACKED_UP; commit 517a51a60a00e6b7a625edeeb8e6bc126393856c.
- foundation: FAILED_LOCAL_ARTIFACTS_PRESERVED; commit pending.
- review-milestone: BACKED_UP; commit 7decf8635f37a9e0e514b05dcd5dc7135a7eeb57.

Run `/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'` from the factory. Audit each candidate batch with `-m factory.pipeline audit --batch NAME`. Reviewers must see no prior verdict before their first judgement. New hashes require new review receipts. Assemble only ACCEPT-gated immutable releases with `-m factory.pipeline release --batch NAME --version vNNNN`. Revisions require new batches and regression evidence; do not rewrite existing releases.

Expand P0 toward 150–250 reviewed parents by semantic gaps after pilots pass. Remaining seven tracks need small exact-verifiable pilots and independent reviews. Create no sealed evaluation under current exposed custody. Back up exact file inventories with checkpoint_backup.py; failed pushes preserve local commits and support --resume-commit. No background execution is promised.

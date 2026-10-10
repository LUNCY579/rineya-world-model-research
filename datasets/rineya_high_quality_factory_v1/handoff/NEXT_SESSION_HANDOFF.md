# Next-session handoff

Read PROJECT_STATUS.md, quality_reports/CURRENT.json and backup_logs receipts first. Inspect original repo/worktree status; never switch or clean the research checkout.

No quality-approved release yet.

- foundation-r2: BACKED_UP; commit 517a51a60a00e6b7a625edeeb8e6bc126393856c.
- foundation: FAILED_LOCAL_ARTIFACTS_PRESERVED; commit pending.

Run `/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'` from the factory. Audit each candidate batch with `-m factory.pipeline audit --batch NAME`. Reviewers must see no prior verdict before their first judgement. New hashes require new review receipts. Assemble only ACCEPT-gated immutable releases with `-m factory.pipeline release --batch NAME --version vNNNN`. Revisions require new batches and regression evidence; do not rewrite existing releases.

Expand P0 toward 150–250 reviewed parents by semantic gaps after pilots pass. Remaining seven tracks need small exact-verifiable pilots and independent reviews. Create no sealed evaluation under current exposed custody. Back up exact file inventories with checkpoint_backup.py; failed pushes preserve local commits and support --resume-commit. No background execution is promised.

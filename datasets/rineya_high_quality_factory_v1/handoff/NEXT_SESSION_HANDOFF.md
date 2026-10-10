# Next-session handoff

The first P0 production milestone is reached: each priority track has at least 150 reviewed source families, and all seven other tracks have independently reviewed exact-verifiable pilots. Further generation should follow diagnosed capability gaps.

| Track | Approved records | Distinct source families |
|---|---:|---:|
| RBM-001 | 24 | 12 |
| RCM-001 | 1440 | 180 |
| RDP-001 | 24 | 12 |
| REM-001 | 120 | 12 |
| RGM-001 | 432 | 180 |
| RJP-001 | 24 | 12 |
| RMM-001 | 76 | 12 |
| ROM-001 | 24 | 12 |
| RSM-001 | 24 | 12 |
| RVM-001 | 384 | 192 |


Read PROJECT_STATUS.md, manifests/ACTIVE_RELEASE_CATALOG.json, quality_reports/CURRENT.json and backup_logs receipts first. Inspect original repo/worktree status; never switch or clean the research checkout. Load approved rows through factory.catalog.catalog(); the old RVM pilot is superseded and retained solely as historical evidence.

- RBM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (production-milestone).
- RCM-001 v0001: 240 historical records, 240 active; ACTIVE, BACKED_UP (production-milestone).
- RCM-001 v0002: 1200 historical records, 0 active; SUPERSEDED, BACKED_UP (production-milestone).
- RCM-001 v0003: 1200 historical records, 1200 active; ACTIVE, LOCAL_ONLY (verification pending).
- RDP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (production-milestone).
- REM-001 v0001: 120 historical records, 120 active; ACTIVE, BACKED_UP (production-milestone).
- RGM-001 v0001: 72 historical records, 72 active; ACTIVE, BACKED_UP (production-milestone).
- RGM-001 v0002: 360 historical records, 360 active; ACTIVE, BACKED_UP (production-milestone).
- RJP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (production-milestone).
- RMM-001 v0001: 76 historical records, 76 active; ACTIVE, BACKED_UP (production-milestone).
- ROM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (production-milestone).
- RSM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (production-milestone).
- RVM-001 v0001: 60 historical records, 0 active; SUPERSEDED, BACKED_UP (production-milestone).
- RVM-001 v0002: 300 historical records, 300 active; ACTIVE, BACKED_UP (production-milestone).
- RVM-001 v0003: 60 historical records, 60 active; ACTIVE, BACKED_UP (production-milestone).
- RVM-001 v0004: 24 historical records, 24 active; ACTIVE, LOCAL_ONLY (verification pending).

- all-pilots: BACKED_UP; commit 265f768f0658942b0004130eef626cdb122eddbd.
- first-approved: BACKED_UP; commit 8c2571e51e43ed206d111321482367802b79600a.
- foundation-r2: BACKED_UP; commit 517a51a60a00e6b7a625edeeb8e6bc126393856c.
- foundation: FAILED_LOCAL_ARTIFACTS_PRESERVED; commit pending.
- production-milestone: BACKED_UP; commit ee7a91cf8cf4195238dafb0caa6f494a20c0b7ef.
- review-milestone: BACKED_UP; commit 7decf8635f37a9e0e514b05dcd5dc7135a7eeb57.

Run `/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'` from the factory, and `/opt/anaconda3/bin/python3 backup_logs/test_checkpoint_backup.py`. Audit fresh candidate batches with `-m factory.pipeline audit --batch NAME`. Full state and independent factual metadata checks must pass. Reviewers must see no prior verdict before their first judgement. New hashes require new review receipts. Assemble only ACCEPT-gated immutable releases with `-m factory.pipeline release --batch NAME --version vNNNN`. Corrections require fresh batches and an explicit supersedes argument through factory.pipeline.release; do not rewrite existing releases.

Prioritize research/research_gap_analysis/PRODUCTION_MILESTONE.md and unresolved research gaps. Add new semantic mechanisms before further surface variants. Critic matched controls still have a 2:1 global overclaim/underclaim mix, limited dev scope and residual plan-length correlation. Expand memory relevance/history conflicts, skill protection/composition and belief information-gathering coverage with fresh exact contracts and independent review.

The 80 expansion competency-tag errors and 6 historical pilot errors were fixed through fresh reviews and a superseding pilot; use feedback/resolved_issues/META-FIX-001.json. Claim-gap-v1 failed 12 renderer hashes and remains quarantined; v2 fixes the alias and has fresh review evidence. Historical review decisions and failed candidate bytes remain preserved and are excluded from active training counts. Initial reviews were blind; correction-topic guidance and same-model correlation are disclosed in research/ROLE_REGISTRY.json.

Create no sealed evaluation under current exposed custody. Back up exact file inventories with checkpoint_backup.py; failed pushes preserve local commits and support --resume-commit. Each post-push receipt is local until a later checkpoint. No model training or background execution is promised.

Source bytes are preserved under reproducibility/source_blobs and indexed in reproducibility/SOURCE_ARCHIVE_INDEX.json. Three original broad source snapshots for historical RCM v0002 could not be recovered; v0003 supersedes it with identical reviewed data, independently reproduced generation and available source bytes. The historical limitation remains documented in quality_reports/reproducibility/RCM-001-v0002.json. Machine audit reruns preserve prior bytes when results change.

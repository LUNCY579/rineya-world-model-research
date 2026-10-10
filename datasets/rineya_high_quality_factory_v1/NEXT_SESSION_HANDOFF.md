# Next-session handoff

The first P0 production milestone is reached: each priority track has at least 150 reviewed source families, and all seven other tracks have independently reviewed exact-verifiable pilots. Further generation should follow diagnosed capability gaps.

| Track | Approved records | Distinct source families |
|---|---:|---:|
| RBM-001 | 24 | 12 |
| RCM-001 | 1440 | 180 |
| RDP-001 | 24 | 12 |
| REM-001 | 888 | 24 |
| RGM-001 | 432 | 180 |
| RJP-001 | 24 | 12 |
| RMM-001 | 76 | 12 |
| ROM-001 | 24 | 12 |
| RSM-001 | 24 | 12 |
| RVM-001 | 384 | 192 |


Read PROJECT_STATUS.md, manifests/ACTIVE_RELEASE_CATALOG.json, quality_reports/CURRENT.json and backup_logs receipts first. Inspect original repo/worktree status; never switch or clean the research checkout. Load approved rows through factory.catalog.catalog(); the old RVM pilot is superseded and retained solely as historical evidence.

- RBM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- RCM-001 v0001: 240 historical records, 240 active; ACTIVE, BACKED_UP (session-final).
- RCM-001 v0002: 1200 historical records, 0 active; SUPERSEDED, BACKED_UP (session-final).
- RCM-001 v0003: 1200 historical records, 1200 active; ACTIVE, BACKED_UP (session-final).
- RDP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- REM-001 v0001: 120 historical records, 120 active; ACTIVE, BACKED_UP (session-final).
- REM-001 v0002: 768 historical records, 768 active; ACTIVE, BACKED_UP (memory-gap-v2-final).
- RGM-001 v0001: 72 historical records, 72 active; ACTIVE, BACKED_UP (session-final).
- RGM-001 v0002: 360 historical records, 360 active; ACTIVE, BACKED_UP (session-final).
- RJP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- RMM-001 v0001: 76 historical records, 76 active; ACTIVE, BACKED_UP (session-final).
- ROM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- RSM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- RVM-001 v0001: 60 historical records, 0 active; SUPERSEDED, BACKED_UP (session-final).
- RVM-001 v0002: 300 historical records, 300 active; ACTIVE, BACKED_UP (session-final).
- RVM-001 v0003: 60 historical records, 60 active; ACTIVE, BACKED_UP (session-final).
- RVM-001 v0004: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).

- all-pilots: BACKED_UP; commit 265f768f0658942b0004130eef626cdb122eddbd.
- first-approved: BACKED_UP; commit 8c2571e51e43ed206d111321482367802b79600a.
- foundation-r2: BACKED_UP; commit 517a51a60a00e6b7a625edeeb8e6bc126393856c.
- foundation: FAILED_LOCAL_ARTIFACTS_PRESERVED; commit pending.
- json-type-memory-provisional: BACKED_UP; commit 75934fb845af8c2e3a418a66cb2a812563e10f74.
- memory-gap-v2-final: BACKED_UP; commit 98cd4b3ad4f3d47af5a98e1d41936b67136e867b.
- production-milestone: BACKED_UP; commit ee7a91cf8cf4195238dafb0caa6f494a20c0b7ef.
- review-milestone: BACKED_UP; commit 7decf8635f37a9e0e514b05dcd5dc7135a7eeb57.
- session-final: BACKED_UP; commit eac47777c60387092bb10851b137adad36a6eb4b.
- session-handoff: BACKED_UP; commit 61530fc87b64db1254cb9447ea800a23048a563e.

Run `/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'` from the factory, and `/opt/anaconda3/bin/python3 backup_logs/test_checkpoint_backup.py`. Audit fresh candidate batches with `-m factory.pipeline audit --batch NAME`. Full state and independent factual metadata checks must pass. Reviewers must see no prior verdict before their first judgement. New hashes require new review receipts. Assemble only ACCEPT-gated immutable releases with `-m factory.pipeline release --batch NAME --version vNNNN`. Corrections require fresh batches and an explicit supersedes argument through factory.pipeline.release; do not rewrite existing releases.

Prioritize research/research_gap_analysis/PRODUCTION_MILESTONE.md and unresolved research gaps. Add new semantic mechanisms before further surface variants. Critic matched controls still have a 2:1 global overclaim/underclaim mix, limited dev scope and residual plan-length correlation. Expand memory relevance/history conflicts, skill protection/composition and belief information-gathering coverage with fresh exact contracts and independent review. Read research/research_gap_analysis/MEMORY_HISTORY_MILESTONE.md for the newer memory coverage and remaining controlled retrieval, composition and belief gaps.

The 80 expansion competency-tag errors and 6 historical pilot errors were fixed through fresh reviews and a superseding pilot; use feedback/resolved_issues/META-FIX-001.json. Claim-gap-v1 failed 12 renderer hashes and remains quarantined; v2 fixes the alias and has fresh review evidence. Historical review decisions and failed candidate bytes remain preserved and are excluded from active training counts. Initial reviews were blind; correction-topic guidance and same-model correlation are disclosed in research/ROLE_REGISTRY.json.

Create no sealed evaluation under current exposed custody. Back up exact file inventories with checkpoint_backup.py; failed pushes preserve local commits and support --resume-commit. Each post-push receipt is local until a later checkpoint. No model training or background execution is promised.

Source bytes are preserved under reproducibility/source_blobs and indexed in reproducibility/SOURCE_ARCHIVE_INDEX.json. Three original broad source snapshots for historical RCM v0002 could not be recovered; v0003 supersedes it with identical reviewed data, independently reproduced generation and available source bytes. The historical limitation remains documented in quality_reports/reproducibility/RCM-001-v0002.json. Machine audit reruns preserve prior bytes when results change.


## Belief evidence-gap preparation — 2026-10-10 UTC

The public RBM-GAP-007 design is frozen. Four proposed fresh families use six contexts and27 independent query ID/position/fact-name shifts with two field-order views (1,296 planned correlated records, not yet approved). Generation and a separately authored input-only control checker are in progress. Actual independent pre-generation belief properties passed14tests; all202tests present at this preparation checkpoint passed together. Expert review preparation is separate. First reviewer expectations will be frozen from stripped input-only copies before target/property-claim comparison. Existing approved counts remain3,340records/648sourcefamilies/all10tracks. See `handoff/BELIEF_PREPARATION_CHECKPOINT.json` for exact current stage and custody. New production/control/review artifacts still require their own frozen audits and approvals.


## Belief evidence-gap first review milestone — 2026-10-10 UTC

The first `belief-gap-v1` candidate contains 1,296 correlated records from four source families, 24 base contexts and 648 rotation-level tasks. Input-only expert and red predictions were frozen before gold/property comparisons. Expert final judgments are 1,296 ACCEPT; red final judgments are 864 ACCEPT and 432 REVISE. All exact typed targets agree. The red audit found that observation tokens `stone`/`unknown` predict emptiness perfectly on both 648-row splits, and a marker/world-count/observation-count lookup reproduces all 216 development empty-context target cores. This serious corpus defect remains unresolved; none of this candidate is approved or released. Independent final adjudication is being completed, including the full-family dependency hold.

The initial checker audit rejected 216 absorption locators and eight arbitrary probe references due to unpublished interface assumptions. An independently authored public-only correction preserves the old evidence, executes probes under explicitly bound arbitrary IDs, and passes all 1,296 rows with zero oracle, metadata or control errors and 68 actual probes. The corrected checker has 26 focused tests; three actual publication corruption tests pass. All 231 current factory tests passed together, with source hashes unchanged, plus seven backup-helper tests. The first failed gate-test fixture and its exact source were preserved before repairing a public output field name. See `quality_reports/BELIEF_V1_REVIEW_FULL_SUITE_EXECUTION.json` and `quality_reports/BELIEF_GATE_TEST_EXECUTION.json`.

New public revision-v2 design bytes are frozen separately in `schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md` and `research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md`. Prospective E/G and F/H pairs change only the second observation equality and match the first observation/prefix, at both observation lengths. F/H repeat their first observation. Planned counts are 1,728 correlated records/four existing families/32 contexts, with 75% nonempty class frequency; added contexts do not count as new source families. Production and independent pre-generation checks have started. Actual revised outputs still require new input-derived reviews, completed adjudication and verified release custody. Positive-only scalar and finite lookup risks remain explicit.

Active approved counts remain 3,340 records/648 source families across all ten tracks; consult `manifests/ACTIVE_RELEASE_CATALOG.json`. No training, efficacy result or sealed evaluation was performed. Actual separate executions share an underlying model; the public design author later adjudicates, with that role overlap disclosed. Oversized first/current checker and expert diagnostics are preserved as exact raw bytes and verified lossless gzip representations. The most recent verified preparation checkpoint is `e65f3c2a07b3f09572584befdbd2925fcd108ee1`; completed first-candidate reviews and corrections await the next verified archival checkpoint.

Final milestone update: independent adjudication is complete: all1,296 rows are REVISE (432 direct lexical defects and864 family dependency holds). The actual publication gate wrote the complete quarantine snapshot and created zero release directories; candidate bytes remain unchanged. See `quality_reports/BELIEF_GAP_V1_MILESTONE.json`, `reviews/adjudication/ADJUDICATION_BELIEF_GAP_V1.md` and `handoff/BELIEF_FIRST_REVIEW_FINAL.json`. The root froze a separate prospective marker-and-auxiliary-token diagnostic before v2 generation; its v2 result is unknown until actual execution.

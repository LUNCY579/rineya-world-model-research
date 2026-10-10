# Project status

CPU-only dataset production is active. Existing research is read-only. No model loading, training or weights/architecture changes.

3340 approved canonical records from 648 distinct source families. A family is one generated world/program instance, not a new mechanism class.

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


Historical candidate inventory (revisions must not be summed):

| Track | Batch | Candidate records | Source-family IDs |
|---|---|---:|---:|
| RBM-001 | extended-pilot-v1 | 24 | 12 |
| RBM-001 | extended-pilot-v2 | 24 | 12 |
| RCM-001 | expansion-v1 | 1200 | 150 |
| RCM-001 | pilot-v1 | 120 | 30 |
| RCM-001 | pilot-v2 | 240 | 30 |
| RDP-001 | extended-pilot-v1 | 24 | 12 |
| RDP-001 | extended-pilot-v2 | 24 | 12 |
| REM-001 | extended-pilot-v1 | 24 | 12 |
| REM-001 | extended-pilot-v2 | 24 | 12 |
| REM-001 | extended-pilot-v3 | 120 | 12 |
| REM-001 | memory-gap-v1 | 768 | 12 |
| REM-001 | memory-gap-v2 | 768 | 12 |
| RGM-001 | expansion-v1 | 360 | 150 |
| RGM-001 | expansion-v2 | 360 | 150 |
| RGM-001 | expansion-v3 | 360 | 150 |
| RGM-001 | pilot-v1 | 60 | 30 |
| RGM-001 | pilot-v2 | 60 | 30 |
| RGM-001 | pilot-v3 | 72 | 30 |
| RJP-001 | extended-pilot-v1 | 24 | 12 |
| RJP-001 | extended-pilot-v2 | 24 | 12 |
| RMM-001 | extended-pilot-v1 | 24 | 12 |
| RMM-001 | extended-pilot-v2 | 24 | 12 |
| RMM-001 | extended-pilot-v3 | 76 | 12 |
| ROM-001 | extended-pilot-v1 | 24 | 12 |
| ROM-001 | extended-pilot-v2 | 24 | 12 |
| RSM-001 | extended-pilot-v1 | 24 | 12 |
| RSM-001 | extended-pilot-v2 | 24 | 12 |
| RVM-001 | claim-gap-v1 | 24 | 12 |
| RVM-001 | claim-gap-v2 | 24 | 12 |
| RVM-001 | expansion-v1 | 300 | 150 |
| RVM-001 | expansion-v2 | 300 | 150 |
| RVM-001 | pilot-v1 | 60 | 30 |
| RVM-001 | pilot-v2 | 60 | 30 |
| RVM-001 | pilot-v4 | 60 | 30 |


Approved release inventory:

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

Backups:

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

All ten curricula exist. P0 pilot correctness/review precedes expansion. Counterfactual/rendering counts do not inflate parent counts. Exposed dev challenges are not sealed tests. Exact state correctness is a data property; model benefit remains untested.


## Belief evidence-gap preparation — 2026-10-10 UTC

The public RBM-GAP-007 design is frozen. Four proposed fresh families use six contexts and27 independent query ID/position/fact-name shifts with two field-order views (1,296 planned correlated records, not yet approved). Generation and a separately authored input-only control checker are in progress. Actual independent pre-generation belief properties passed14tests; all202tests present at this preparation checkpoint passed together. Expert review preparation is separate. First reviewer expectations will be frozen from stripped input-only copies before target/property-claim comparison. Existing approved counts remain3,340records/648sourcefamilies/all10tracks. See `handoff/BELIEF_PREPARATION_CHECKPOINT.json` for exact current stage and custody. New production/control/review artifacts still require their own frozen audits and approvals.


## Belief evidence-gap first review milestone — 2026-10-10 UTC

The first `belief-gap-v1` candidate contains 1,296 correlated records from four source families, 24 base contexts and 648 rotation-level tasks. Input-only expert and red predictions were frozen before gold/property comparisons. Expert final judgments are 1,296 ACCEPT; red final judgments are 864 ACCEPT and 432 REVISE. All exact typed targets agree. The red audit found that observation tokens `stone`/`unknown` predict emptiness perfectly on both 648-row splits, and a marker/world-count/observation-count lookup reproduces all 216 development empty-context target cores. This serious corpus defect remains unresolved; none of this candidate is approved or released. Independent final adjudication is being completed, including the full-family dependency hold.

The initial checker audit rejected 216 absorption locators and eight arbitrary probe references due to unpublished interface assumptions. An independently authored public-only correction preserves the old evidence, executes probes under explicitly bound arbitrary IDs, and passes all 1,296 rows with zero oracle, metadata or control errors and 68 actual probes. The corrected checker has 26 focused tests; three actual publication corruption tests pass. All 231 current factory tests passed together, with source hashes unchanged, plus seven backup-helper tests. The first failed gate-test fixture and its exact source were preserved before repairing a public output field name. See `quality_reports/BELIEF_V1_REVIEW_FULL_SUITE_EXECUTION.json` and `quality_reports/BELIEF_GATE_TEST_EXECUTION.json`.

New public revision-v2 design bytes are frozen separately in `schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md` and `research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md`. Prospective E/G and F/H pairs change only the second observation equality and match the first observation/prefix, at both observation lengths. F/H repeat their first observation. Planned counts are 1,728 correlated records/four existing families/32 contexts, with 75% nonempty class frequency; added contexts do not count as new source families. Production and independent pre-generation checks have started. Actual revised outputs still require new input-derived reviews, completed adjudication and verified release custody. Positive-only scalar and finite lookup risks remain explicit.

Active approved counts remain 3,340 records/648 source families across all ten tracks; consult `manifests/ACTIVE_RELEASE_CATALOG.json`. No training, efficacy result or sealed evaluation was performed. Actual separate executions share an underlying model; the public design author later adjudicates, with that role overlap disclosed. Oversized first/current checker and expert diagnostics are preserved as exact raw bytes and verified lossless gzip representations. The most recent verified preparation checkpoint is `e65f3c2a07b3f09572584befdbd2925fcd108ee1`; completed first-candidate reviews and corrections await the next verified archival checkpoint.

Final milestone update: independent adjudication is complete: all1,296 rows are REVISE (432 direct lexical defects and864 family dependency holds). The actual publication gate wrote the complete quarantine snapshot and created zero release directories; candidate bytes remain unchanged. See `quality_reports/BELIEF_GAP_V1_MILESTONE.json`, `reviews/adjudication/ADJUDICATION_BELIEF_GAP_V1.md` and `handoff/BELIEF_FIRST_REVIEW_FINAL.json`. The root froze a separate prospective marker-and-auxiliary-token diagnostic before v2 generation; its v2 result is unknown until actual execution.


## Current revised belief batch — 2026-10-10 UTC

`belief-gap-v2` is frozen at 1,728 PENDING records / 864 rotation tasks / four reused source families / 32 base contexts. Train and exposed development each contain 864 records; 1,296 rows are nonempty and 432 empty. Difficulty is measured at Level 3 for 1,242 rows and Level 2 for 486. Its exact candidate SHA is `589119d69290cf91f8cb00adf49d280f7499a6121e9064655ae523222e9e72df`. The original word marker now makes 432 status errors, but the separately frozen marker-and-`linen` rule remains perfect on both splits. Fresh expert and red input-only predictions are frozen before canonical comparisons; final reviews/adjudication are still pending. No revised record is approved.

Actual revised checker invocation finds zero target failures but 1,728 repeated metadata errors due to its assumption that a corpus evidence ID must identify only one property. The public contract permits the same matched pair to support both new properties; independent public-only reconciliation is queued. First failing report bytes and source are preserved. This interface question cannot resolve the independent lexical corpus defect. Root routing by canonical revision provenance now protects the new obligations even if G/H rows or internal claims are removed. Fourteen related publication, memory and typed-evidence tests pass.

Reproduction gives exact bytes for the candidate, parents and six of seven deterministic artifacts. One diagnostic JSON has different object key ordering from unsorted set iteration; parsed JSON is identical. The failed run and all outputs are preserved, and `RBM-REPRO-010.json` requires sorted iteration and separate-process byte checks for a fresh revision. Neither frozen producer nor candidate was edited. A new prospective scalar-transposition design is being developed; no future candidate or efficacy claim is made. See `handoff/BELIEF_V2_REVIEW_IN_PROGRESS.json`. The last verified archival checkpoint is `7a7d055bda9a155b0480ebad5bfb8080133b7585`; all 3,340 approved records / 648 source families remain unchanged.


## Revised belief batch final review — 2026-10-10 UTC

Current V2 status supersedes the earlier in-progress snapshots: all 1,728 records are REVISE and the actual publication gate quarantined the entire batch, creating zero releases. Expert and red each returned 1,296 ACCEPT / 432 REVISE with zero disagreement; the separate adjudicator applies 432 direct compound-token defects and 1,296 complete-family dependency holds. All exact targets, metadata and controls pass, with 84 actual executed probes. The compound marker-and-linen rule still predicts all 1,728 statuses and reconstructs all 432 empty-target cores without world facts. No new record is approved. Active approved counts remain 3,340 records / 648 source families across all ten tracks.

All 267 then-present V1/V2 factory tests passed with tested sources unchanged. V2 candidate bytes reproduce exactly; one of seven deterministic artifacts differs only in diagnostic JSON key ordering. The later actual publication audit changes six equally scoring diagnostic feature/polarity representatives while preserving every semantic result, histogram and accuracy statistic. Both complete reports and their raw/gzip histories are retained. RBM-REPRO-010 and ORACLE-REPRO-011 require deterministic ordering/tie handling and separate-process checks in new V3 artifacts; frozen V2 sources are unchanged. These custody limitations are separate from the measured lexical defect.

The NEW public V3 design/root freeze precede actual generation. Its complete product is 3,456 correlated rows from the same four families, 32 semantic contexts and 64 scalar-view cases. A fact-local swap of an unqueried auxiliary column must preserve full targets within each context, while 864 disjoint cross-table pairs require opposite statuses under identical observations/queries/first-prefix features. Actual producer core/archive reproduction and independently authored public-only checks are in progress; fresh prediction-before-gold reviews and adjudication remain required. Planned counts and bounds do not constitute approval, independent-family growth, sealed OOD or model efficacy. See `handoff/BELIEF_V2_FINAL.json` for exact hashes, history locations and next actions. The last verified archival commit is still `7a7d055bda9a155b0480ebad5bfb8080133b7585`; subsequent final V2 evidence and V3 work require the next actual-byte checkpoint.


## Belief v3 review discrepancy and verified checkpoint — 2026-10-10 UTC

Active approved corpus remains **3,340 records / 648 source families across ten tracks**. V3 has 3,456 pending rows: both initial independent reviews accepted the exact targets and controls, then separate supplements found 6,912 inherited claims without inline pair bindings and recommended metadata revision. All targets and underlying pairs still pass. The producer manifest contains actual length-specific pair evidence; the unresolved issue is the retained claim-binding interface, including an ambiguous optional-pair sentence in the v3 supplement. Frozen checker reports 3,456 metadata failures / 3,484 control errors. Actual separate adjudication is in progress; no v3 release is approved. First reviews, failed audit and test sources remain immutable.

The corrected root publication fixture passed all 21 applicable tests; it adds inherited pair bindings only to synthetic temporary rows before testing missing v3 variants. Real candidate bytes remain frozen. RGM prospective design adds 48 proposed families / 384 rows for measured matched-input and block-arrangement gaps; its public specification is frozen and independent formal work has started. Producer dispatch encountered actual thread limits, so no new RGM data is claimed yet.

Latest verified backup is `9a03758bf7343e911c9dbc618a7158e393e5f790` (1,344 files / 674,483,136 bytes). All 18 oversized archived files were additionally streamed from actual remote gzip blobs and matched 735,726,123 original raw bytes. New v3 final reviews/supplements, current gate code/tests, remote decode proof, RGM design and this handoff remain local until the next checkpoint. Storage is volatile due to system swap. Only redundant tracked research copies in the root-owned backup checkout were excluded by sparse checkout; full Git index and original research working files/main remain unchanged. Shared Git worktreeConfig metadata was intentionally enabled and exact before/after custody recorded.

See `handoff/BELIEF_V3_REVIEW_DISCREPANCY.json` for exact stage hashes and remaining work.


## Completed belief v3 adjudication — 2026-10-10 UTC

Actual adjudicator assigned **all 3,456 REVISE** for unresolved inherited claim encoding under ambiguous public wording; all targets and 1,728 named manifest entries still pass. Root mechanically quarantined the whole unchanged batch, verified every original record/hash join and retained candidate bytes. No release was created. The full applicable suite passed **289 tests in 258.652 seconds**, with source hashes unchanged. Approved corpus stays 3,340 records / 648 source families.

NEW prospective v4 clarification requires explicit inline inherited pair IDs and resolvable length-specific manifest/control bindings while retaining aggregate permission for the three v3 controls. It introduces no new semantic units and does not approve old or future candidates. RGM production has now actually started under its frozen 48-family/384-row design; independently authored formal handmade fixtures passed 12 tests and final source audit remains in progress. Reviewer preparation dispatch hit an actual thread limit and will resume after a completed worker frees capacity. Current RGM work remains local and provisional while the completed belief checkpoint is prepared. See `handoff/BELIEF_V3_FINAL.json`.

# Training readiness

Authoritative active inventory: manifests/ACTIVE_RELEASE_CATALOG.json. Load through factory.catalog.catalog(), which excludes exact-hash errata and superseded versions; do not concatenate historical release directories. Original release bytes remain preserved.

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

Source bytes are preserved under reproducibility/source_blobs and indexed in reproducibility/SOURCE_ARCHIVE_INDEX.json. Three original broad source snapshots for historical RCM v0002 could not be recovered; v0003 supersedes it with identical reviewed data, independently reproduced generation and available source bytes. The historical limitation remains documented in quality_reports/reproducibility/RCM-001-v0002.json. Machine audit reruns preserve prior bytes when results change.

Only independently approved active release bytes may be considered for a separately authorized future training experiment. Pending candidates and quarantine files are excluded from student exports. Claims are limited to the stated exact synthetic domains. The factory does not establish transfer, retention, larger-world success or intelligence improvement. Structural template overlap, symbolic AST presentation and same-model reviewer correlation limit interpretation. Train/dev records are exposed; separate custody is needed for a future sealed evaluation.


## Belief evidence-gap first review milestone — 2026-10-10 UTC

The first `belief-gap-v1` candidate contains 1,296 correlated records from four source families, 24 base contexts and 648 rotation-level tasks. Input-only expert and red predictions were frozen before gold/property comparisons. Expert final judgments are 1,296 ACCEPT; red final judgments are 864 ACCEPT and 432 REVISE. All exact typed targets agree. The red audit found that observation tokens `stone`/`unknown` predict emptiness perfectly on both 648-row splits, and a marker/world-count/observation-count lookup reproduces all 216 development empty-context target cores. This serious corpus defect remains unresolved; none of this candidate is approved or released. Independent final adjudication is being completed, including the full-family dependency hold.

The initial checker audit rejected 216 absorption locators and eight arbitrary probe references due to unpublished interface assumptions. An independently authored public-only correction preserves the old evidence, executes probes under explicitly bound arbitrary IDs, and passes all 1,296 rows with zero oracle, metadata or control errors and 68 actual probes. The corrected checker has 26 focused tests; three actual publication corruption tests pass. All 231 current factory tests passed together, with source hashes unchanged, plus seven backup-helper tests. The first failed gate-test fixture and its exact source were preserved before repairing a public output field name. See `quality_reports/BELIEF_V1_REVIEW_FULL_SUITE_EXECUTION.json` and `quality_reports/BELIEF_GATE_TEST_EXECUTION.json`.

New public revision-v2 design bytes are frozen separately in `schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md` and `research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md`. Prospective E/G and F/H pairs change only the second observation equality and match the first observation/prefix, at both observation lengths. F/H repeat their first observation. Planned counts are 1,728 correlated records/four existing families/32 contexts, with 75% nonempty class frequency; added contexts do not count as new source families. Production and independent pre-generation checks have started. Actual revised outputs still require new input-derived reviews, completed adjudication and verified release custody. Positive-only scalar and finite lookup risks remain explicit.

Active approved counts remain 3,340 records/648 source families across all ten tracks; consult `manifests/ACTIVE_RELEASE_CATALOG.json`. No training, efficacy result or sealed evaluation was performed. Actual separate executions share an underlying model; the public design author later adjudicates, with that role overlap disclosed. Oversized first/current checker and expert diagnostics are preserved as exact raw bytes and verified lossless gzip representations. The most recent verified preparation checkpoint is `e65f3c2a07b3f09572584befdbd2925fcd108ee1`; completed first-candidate reviews and corrections await the next verified archival checkpoint.

Final milestone update: independent adjudication is complete: all1,296 rows are REVISE (432 direct lexical defects and864 family dependency holds). The actual publication gate wrote the complete quarantine snapshot and created zero release directories; candidate bytes remain unchanged. See `quality_reports/BELIEF_GAP_V1_MILESTONE.json`, `reviews/adjudication/ADJUDICATION_BELIEF_GAP_V1.md` and `handoff/BELIEF_FIRST_REVIEW_FINAL.json`. The root froze a separate prospective marker-and-auxiliary-token diagnostic before v2 generation; its v2 result is unknown until actual execution.

# Actual curriculum coverage

The first P0 production milestone is reached: each priority track has at least 150 reviewed source families, and all seven other tracks have independently reviewed exact-verifiable pilots. Further generation should follow diagnosed capability gaps.

Active approved inventory:

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


Active difficulty counts (canonical records, including render variants):

```json
{
  "RBM-001": {
    "2": 16,
    "3": 8
  },
  "RCM-001": {
    "3": 384,
    "1": 192,
    "4": 864
  },
  "RDP-001": {
    "2": 4,
    "3": 20
  },
  "REM-001": {
    "3": 888
  },
  "RGM-001": {
    "3": 64,
    "4": 292,
    "1": 28,
    "2": 48
  },
  "RJP-001": {
    "2": 12,
    "3": 12
  },
  "RMM-001": {
    "3": 76
  },
  "ROM-001": {
    "3": 24
  },
  "RSM-001": {
    "2": 4,
    "3": 16,
    "4": 4
  },
  "RVM-001": {
    "4": 276,
    "3": 108
  }
}
```

Historical candidate inventory; revisions and renderings must not be added to active family counts:

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


Difficulty counts by historical batch:

```json
{
  "RBM-001/extended-pilot-v1": {
    "2": 16,
    "3": 8
  },
  "RBM-001/extended-pilot-v2": {
    "2": 16,
    "3": 8
  },
  "RCM-001/expansion-v1": {
    "3": 320,
    "1": 160,
    "4": 720
  },
  "RCM-001/pilot-v1": {
    "3": 32,
    "1": 16,
    "4": 48,
    "7": 24
  },
  "RCM-001/pilot-v2": {
    "3": 64,
    "1": 32,
    "4": 144
  },
  "RDP-001/extended-pilot-v1": {
    "2": 4,
    "3": 20
  },
  "RDP-001/extended-pilot-v2": {
    "2": 4,
    "3": 20
  },
  "REM-001/extended-pilot-v1": {
    "3": 24
  },
  "REM-001/extended-pilot-v2": {
    "3": 24
  },
  "REM-001/extended-pilot-v3": {
    "3": 120
  },
  "REM-001/memory-gap-v1": {
    "3": 768
  },
  "REM-001/memory-gap-v2": {
    "3": 768
  },
  "RGM-001/expansion-v1": {
    "3": 50,
    "4": 244,
    "1": 24,
    "2": 42
  },
  "RGM-001/expansion-v2": {
    "3": 50,
    "4": 244,
    "1": 24,
    "2": 42
  },
  "RGM-001/expansion-v3": {
    "3": 50,
    "4": 244,
    "1": 24,
    "2": 42
  },
  "RGM-001/pilot-v1": {
    "3": 24,
    "4": 24,
    "7": 12
  },
  "RGM-001/pilot-v2": {
    "3": 28,
    "4": 20,
    "1": 4,
    "2": 8
  },
  "RGM-001/pilot-v3": {
    "3": 14,
    "4": 48,
    "1": 4,
    "2": 6
  },
  "RJP-001/extended-pilot-v1": {
    "2": 12,
    "3": 12
  },
  "RJP-001/extended-pilot-v2": {
    "2": 12,
    "3": 12
  },
  "RMM-001/extended-pilot-v1": {
    "3": 24
  },
  "RMM-001/extended-pilot-v2": {
    "3": 24
  },
  "RMM-001/extended-pilot-v3": {
    "3": 76
  },
  "ROM-001/extended-pilot-v1": {
    "3": 24
  },
  "ROM-001/extended-pilot-v2": {
    "3": 24
  },
  "RSM-001/extended-pilot-v1": {
    "2": 4,
    "3": 16,
    "4": 4
  },
  "RSM-001/extended-pilot-v2": {
    "2": 4,
    "3": 16,
    "4": 4
  },
  "RVM-001/claim-gap-v1": {
    "4": 20,
    "3": 4
  },
  "RVM-001/claim-gap-v2": {
    "4": 20,
    "3": 4
  },
  "RVM-001/expansion-v1": {
    "4": 216,
    "3": 84
  },
  "RVM-001/expansion-v2": {
    "4": 216,
    "3": 84
  },
  "RVM-001/pilot-v1": {
    "4": 24,
    "5": 24,
    "7": 12
  },
  "RVM-001/pilot-v2": {
    "4": 40,
    "3": 20
  },
  "RVM-001/pilot-v4": {
    "4": 40,
    "3": 20
  }
}
```

Use research/curricula cards for all ten proposed programs. Actual mechanisms and split axes are in semantic_parents files, not inferred from prompt length. Unsupported latent representation, hidden-causal identification, unrestricted social inference and sealed OOD evidence remain gaps. Surface twins do not create new independent concepts.


## Belief evidence-gap first review milestone — 2026-10-10 UTC

The first `belief-gap-v1` candidate contains 1,296 correlated records from four source families, 24 base contexts and 648 rotation-level tasks. Input-only expert and red predictions were frozen before gold/property comparisons. Expert final judgments are 1,296 ACCEPT; red final judgments are 864 ACCEPT and 432 REVISE. All exact typed targets agree. The red audit found that observation tokens `stone`/`unknown` predict emptiness perfectly on both 648-row splits, and a marker/world-count/observation-count lookup reproduces all 216 development empty-context target cores. This serious corpus defect remains unresolved; none of this candidate is approved or released. Independent final adjudication is being completed, including the full-family dependency hold.

The initial checker audit rejected 216 absorption locators and eight arbitrary probe references due to unpublished interface assumptions. An independently authored public-only correction preserves the old evidence, executes probes under explicitly bound arbitrary IDs, and passes all 1,296 rows with zero oracle, metadata or control errors and 68 actual probes. The corrected checker has 26 focused tests; three actual publication corruption tests pass. All 231 current factory tests passed together, with source hashes unchanged, plus seven backup-helper tests. The first failed gate-test fixture and its exact source were preserved before repairing a public output field name. See `quality_reports/BELIEF_V1_REVIEW_FULL_SUITE_EXECUTION.json` and `quality_reports/BELIEF_GATE_TEST_EXECUTION.json`.

New public revision-v2 design bytes are frozen separately in `schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md` and `research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md`. Prospective E/G and F/H pairs change only the second observation equality and match the first observation/prefix, at both observation lengths. F/H repeat their first observation. Planned counts are 1,728 correlated records/four existing families/32 contexts, with 75% nonempty class frequency; added contexts do not count as new source families. Production and independent pre-generation checks have started. Actual revised outputs still require new input-derived reviews, completed adjudication and verified release custody. Positive-only scalar and finite lookup risks remain explicit.

Active approved counts remain 3,340 records/648 source families across all ten tracks; consult `manifests/ACTIVE_RELEASE_CATALOG.json`. No training, efficacy result or sealed evaluation was performed. Actual separate executions share an underlying model; the public design author later adjudicates, with that role overlap disclosed. Oversized first/current checker and expert diagnostics are preserved as exact raw bytes and verified lossless gzip representations. The most recent verified preparation checkpoint is `e65f3c2a07b3f09572584befdbd2925fcd108ee1`; completed first-candidate reviews and corrections await the next verified archival checkpoint.

Final milestone update: independent adjudication is complete: all1,296 rows are REVISE (432 direct lexical defects and864 family dependency holds). The actual publication gate wrote the complete quarantine snapshot and created zero release directories; candidate bytes remain unchanged. See `quality_reports/BELIEF_GAP_V1_MILESTONE.json`, `reviews/adjudication/ADJUDICATION_BELIEF_GAP_V1.md` and `handoff/BELIEF_FIRST_REVIEW_FINAL.json`. The root froze a separate prospective marker-and-auxiliary-token diagnostic before v2 generation; its v2 result is unknown until actual execution.

# Dataset quality report

Machine evidence: quality_reports/CURRENT.json. Candidate batches are retained even when superseded. Review decisions are bound to exact candidate hashes and independent execution identities.

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


Review decision counts:

```json
{
  "expert": {
    "records": 5176,
    "decisions": {
      "ACCEPT": 4294,
      "REVISE": 882
    }
  },
  "adversarial": {
    "records": 4408,
    "decisions": {
      "ACCEPT": 4220,
      "REVISE": 188
    }
  },
  "adjudication": {
    "records": 4168,
    "decisions": {
      "ACCEPT": 4028,
      "REVISE": 140
    }
  }
}
```

Exact oracle, schema, duplicates and family split audit summaries:

```json
[
  {
    "batch": "claim-gap-v1",
    "records": 24,
    "unique_parents": 12,
    "distinct_parent_semantic_signatures": 12,
    "oracle_failures": 12,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "4": 20,
      "3": 4
    },
    "tracks": {
      "RVM-001": 24
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "claim-gap-v2",
    "records": 24,
    "unique_parents": 12,
    "distinct_parent_semantic_signatures": 12,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "4": 20,
      "3": 4
    },
    "tracks": {
      "RVM-001": 24
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 0,
    "validation_failures": 0
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  },
  {
    "batch": "expansion-v1",
    "records": 1860,
    "unique_parents": 450,
    "distinct_parent_semantic_signatures": 450,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 454,
      "1": 184,
      "4": 1180,
      "2": 42
    },
    "tracks": {
      "RCM-001": 1200,
      "RGM-001": 360,
      "RVM-001": 300
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 80,
    "validation_failures": 80
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  },
  {
    "batch": "expansion-v2",
    "records": 660,
    "unique_parents": 300,
    "distinct_parent_semantic_signatures": 300,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 134,
      "4": 460,
      "1": 24,
      "2": 42
    },
    "tracks": {
      "RGM-001": 360,
      "RVM-001": 300
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 14,
    "validation_failures": 14
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  },
  {
    "batch": "expansion-v3",
    "records": 360,
    "unique_parents": 150,
    "distinct_parent_semantic_signatures": 150,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 50,
      "4": 244,
      "1": 24,
      "2": 42
    },
    "tracks": {
      "RGM-001": 360
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 0,
    "validation_failures": 0
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  },
  {
    "batch": "extended-pilot-v1",
    "records": 168,
    "unique_parents": 84,
    "distinct_parent_semantic_signatures": 74,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [
      [
        "RBM-001-extended-pilot-v1-p0000",
        "RBM-001-extended-pilot-v1-p0006"
      ],
      [
        "RBM-001-extended-pilot-v1-p0001",
        "RBM-001-extended-pilot-v1-p0007"
      ],
      [
        "RBM-001-extended-pilot-v1-p0004",
        "RBM-001-extended-pilot-v1-p0010"
      ],
      [
        "RBM-001-extended-pilot-v1-p0005",
        "RBM-001-extended-pilot-v1-p0011"
      ],
      [
        "RJP-001-extended-pilot-v1-p0004",
        "RJP-001-extended-pilot-v1-p0010"
      ],
      [
        "RMM-001-extended-pilot-v1-p0000",
        "RMM-001-extended-pilot-v1-p0006"
      ],
      [
        "RMM-001-extended-pilot-v1-p0001",
        "RMM-001-extended-pilot-v1-p0007"
      ],
      [
        "RMM-001-extended-pilot-v1-p0004",
        "RMM-001-extended-pilot-v1-p0010"
      ],
      [
        "RMM-001-extended-pilot-v1-p0005",
        "RMM-001-extended-pilot-v1-p0011"
      ],
      [
        "ROM-001-extended-pilot-v1-p0005",
        "ROM-001-extended-pilot-v1-p0011"
      ]
    ],
    "exact_prompt_duplicates": 20,
    "difficulty": {
      "2": 36,
      "3": 128,
      "4": 4
    },
    "tracks": {
      "RBM-001": 24,
      "RDP-001": 24,
      "REM-001": 24,
      "RJP-001": 24,
      "RMM-001": 24,
      "ROM-001": 24,
      "RSM-001": 24
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "extended-pilot-v2",
    "records": 168,
    "unique_parents": 84,
    "distinct_parent_semantic_signatures": 84,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "2": 36,
      "3": 128,
      "4": 4
    },
    "tracks": {
      "RBM-001": 24,
      "RDP-001": 24,
      "REM-001": 24,
      "RJP-001": 24,
      "RMM-001": 24,
      "ROM-001": 24,
      "RSM-001": 24
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "extended-pilot-v3",
    "records": 196,
    "unique_parents": 24,
    "distinct_parent_semantic_signatures": 24,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 196
    },
    "tracks": {
      "REM-001": 120,
      "RMM-001": 76
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "memory-gap-v1",
    "records": 768,
    "unique_parents": 12,
    "distinct_parent_semantic_signatures": 12,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 768
    },
    "tracks": {
      "REM-001": 768
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 768,
    "validation_failures": 768,
    "memory_control_error_count": 6156,
    "memory_control_counts": {
      "records": 768,
      "scheduled_records": 768,
      "source_families": 12,
      "base_controlled_histories": 48,
      "base_history_query_observations": 384,
      "rotation_level_observations": 3072,
      "rendered_observations": 6144
    }
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  },
  {
    "batch": "memory-gap-v2",
    "records": 768,
    "unique_parents": 12,
    "distinct_parent_semantic_signatures": 12,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 768
    },
    "tracks": {
      "REM-001": 768
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 0,
    "validation_failures": 0,
    "memory_control_error_count": 0,
    "memory_control_counts": {
      "records": 768,
      "scheduled_records": 768,
      "source_families": 12,
      "base_controlled_histories": 48,
      "base_history_query_observations": 384,
      "rotation_level_observations": 3072,
      "rendered_observations": 6144
    }
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  },
  {
    "batch": "pilot-v1",
    "records": 240,
    "unique_parents": 90,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 56,
      "1": 16,
      "4": 96,
      "7": 48,
      "5": 24
    },
    "tracks": {
      "RCM-001": 120,
      "RGM-001": 60,
      "RVM-001": 60
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "pilot-v2",
    "records": 360,
    "unique_parents": 90,
    "distinct_parent_semantic_signatures": 90,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 112,
      "1": 36,
      "4": 204,
      "2": 8
    },
    "tracks": {
      "RCM-001": 240,
      "RGM-001": 60,
      "RVM-001": 60
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "pilot-v3",
    "records": 72,
    "unique_parents": 30,
    "distinct_parent_semantic_signatures": 30,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "3": 14,
      "4": 48,
      "1": 4,
      "2": 6
    },
    "tracks": {
      "RGM-001": 72
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument"
  },
  {
    "batch": "pilot-v4",
    "records": 60,
    "unique_parents": 30,
    "distinct_parent_semantic_signatures": 30,
    "oracle_failures": 0,
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "exact_prompt_duplicates": 0,
    "difficulty": {
      "4": 40,
      "3": 20
    },
    "tracks": {
      "RVM-001": 60
    },
    "structural_template_holdout": false,
    "protected_benchmark_inspected": false,
    "scope": "fresh generated development data; not a sealed generalization instrument",
    "metadata_failures": 0,
    "validation_failures": 0
  },
  {
    "family_overlap": [],
    "split_leakage": [],
    "cross_parent_semantic_duplicates": [],
    "already_released_samples": []
  }
]
```

Template overlap remains a known scope limitation. All three P0 task families require independent expert/adversarial assessment and adjudication before release. Oracle equivalence and mechanized review do not establish native model performance. Reviewers share the underlying model, with correlated-error risk. Generation uses local CPU arithmetic/BFS, no paid external generation service; agent-token costs are not metered per sample.

Source bytes are preserved under reproducibility/source_blobs and indexed in reproducibility/SOURCE_ARCHIVE_INDEX.json. Three original broad source snapshots for historical RCM v0002 could not be recovered; v0003 supersedes it with identical reviewed data, independently reproduced generation and available source bytes. The historical limitation remains documented in quality_reports/reproducibility/RCM-001-v0002.json. Machine audit reruns preserve prior bytes when results change.


## Belief evidence-gap first review milestone — 2026-10-10 UTC

The first `belief-gap-v1` candidate contains 1,296 correlated records from four source families, 24 base contexts and 648 rotation-level tasks. Input-only expert and red predictions were frozen before gold/property comparisons. Expert final judgments are 1,296 ACCEPT; red final judgments are 864 ACCEPT and 432 REVISE. All exact typed targets agree. The red audit found that observation tokens `stone`/`unknown` predict emptiness perfectly on both 648-row splits, and a marker/world-count/observation-count lookup reproduces all 216 development empty-context target cores. This serious corpus defect remains unresolved; none of this candidate is approved or released. Independent final adjudication is being completed, including the full-family dependency hold.

The initial checker audit rejected 216 absorption locators and eight arbitrary probe references due to unpublished interface assumptions. An independently authored public-only correction preserves the old evidence, executes probes under explicitly bound arbitrary IDs, and passes all 1,296 rows with zero oracle, metadata or control errors and 68 actual probes. The corrected checker has 26 focused tests; three actual publication corruption tests pass. All 231 current factory tests passed together, with source hashes unchanged, plus seven backup-helper tests. The first failed gate-test fixture and its exact source were preserved before repairing a public output field name. See `quality_reports/BELIEF_V1_REVIEW_FULL_SUITE_EXECUTION.json` and `quality_reports/BELIEF_GATE_TEST_EXECUTION.json`.

New public revision-v2 design bytes are frozen separately in `schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md` and `research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md`. Prospective E/G and F/H pairs change only the second observation equality and match the first observation/prefix, at both observation lengths. F/H repeat their first observation. Planned counts are 1,728 correlated records/four existing families/32 contexts, with 75% nonempty class frequency; added contexts do not count as new source families. Production and independent pre-generation checks have started. Actual revised outputs still require new input-derived reviews, completed adjudication and verified release custody. Positive-only scalar and finite lookup risks remain explicit.

Active approved counts remain 3,340 records/648 source families across all ten tracks; consult `manifests/ACTIVE_RELEASE_CATALOG.json`. No training, efficacy result or sealed evaluation was performed. Actual separate executions share an underlying model; the public design author later adjudicates, with that role overlap disclosed. Oversized first/current checker and expert diagnostics are preserved as exact raw bytes and verified lossless gzip representations. The most recent verified preparation checkpoint is `e65f3c2a07b3f09572584befdbd2925fcd108ee1`; completed first-candidate reviews and corrections await the next verified archival checkpoint.

Final milestone update: independent adjudication is complete: all1,296 rows are REVISE (432 direct lexical defects and864 family dependency holds). The actual publication gate wrote the complete quarantine snapshot and created zero release directories; candidate bytes remain unchanged. See `quality_reports/BELIEF_GAP_V1_MILESTONE.json`, `reviews/adjudication/ADJUDICATION_BELIEF_GAP_V1.md` and `handoff/BELIEF_FIRST_REVIEW_FINAL.json`. The root froze a separate prospective marker-and-auxiliary-token diagnostic before v2 generation; its v2 result is unknown until actual execution.

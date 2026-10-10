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
    "records": 3616,
    "decisions": {
      "ACCEPT": 3502,
      "REVISE": 114
    }
  },
  "adversarial": {
    "records": 3640,
    "decisions": {
      "ACCEPT": 3452,
      "REVISE": 188
    }
  },
  "adjudication": {
    "records": 3376,
    "decisions": {
      "ACCEPT": 3236,
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
    "scope": "fresh generated development data; not a sealed generalization instrument"
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
    "scope": "fresh generated development data; not a sealed generalization instrument"
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

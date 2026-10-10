# Actual curriculum coverage

The first P0 production milestone is reached: each priority track has at least 150 reviewed source families, and all seven other tracks have independently reviewed exact-verifiable pilots. Further generation should follow diagnosed capability gaps.

Active approved inventory:

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
    "3": 120
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

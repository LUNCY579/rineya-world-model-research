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

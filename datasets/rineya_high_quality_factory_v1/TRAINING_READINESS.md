# Training readiness

Authoritative active inventory: manifests/ACTIVE_RELEASE_CATALOG.json. Load through factory.catalog.catalog(), which excludes exact-hash errata and superseded versions; do not concatenate historical release directories. Original release bytes remain preserved.

- RBM-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- RCM-001 v0001: 240 historical records, 240 active; ACTIVE, BACKED_UP (session-final).
- RCM-001 v0002: 1200 historical records, 0 active; SUPERSEDED, BACKED_UP (session-final).
- RCM-001 v0003: 1200 historical records, 1200 active; ACTIVE, BACKED_UP (session-final).
- RDP-001 v0001: 24 historical records, 24 active; ACTIVE, BACKED_UP (session-final).
- REM-001 v0001: 120 historical records, 120 active; ACTIVE, BACKED_UP (session-final).
- REM-001 v0002: 768 historical records, 768 active; ACTIVE, LOCAL_ONLY (verification pending).
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

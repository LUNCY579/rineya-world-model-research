# GCDL-001 production status

All original 2,152 v1 accepted records and eligible evidence/runtime payloads have verified GitHub custody at snapshot 7d08c8c3934a38653c88a0c7b940a427ea06d886. Local bulk payloads and archived Git blob objects are absent. Cold full release audit passed all 11 gates; producer recovery preserved TRAIN 1,320, DEV 297, OOD 319 and SEALED 216 without reading sealed gold. Next nonsealed seed is 1,000,066.

The exact 54-file source digest 3cc00cd2c73b37829c8488babe654296aba4df28fada67c459a3b8a2f7c58835 passed 329 tests in both root and independent runs. V2 generation is approved for new jobs; existing records retain v1 identities. Migration resume awaits the final confirmed control backup and normal resource gates. Five-minute low-space archival and a quiet two-hour follow-up are configured for GCDL-001 only. See ARCHIVE_POLICY.md and manifests/ARCHIVE_COMPLETION.json.

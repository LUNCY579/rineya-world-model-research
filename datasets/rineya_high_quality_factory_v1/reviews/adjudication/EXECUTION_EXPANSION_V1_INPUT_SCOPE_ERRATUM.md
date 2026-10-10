# Expansion-v1 execution input-scope erratum

The original execution manifest accidentally snapshotted two unrelated expansion-v2 work-in-progress files through an overbroad filename glob. Their later finalization changed those hashes. They were not inputs to expansion-v1 decisions. This erratum preserves the original manifest and adjudication bytes, excludes the two unrelated paths from the actual dependency scope, and records both historical and current hashes.

Every actual expansion-v1 dependency still matches its saved hash. All1,860 candidate hashes, typed visible payloads and independent-oracle targets were rechecked. The unchanged adjudication remains1,780 ACCEPT/80 REVISE. No sample decision, candidate or release was edited. Correction batches have separate complete manifests whose input/output hashes all match.

# Candidate process supervision v1

Each export contains the full student prompt, one submitted candidate, and its independently verified evaluation. Labels supervise the whole candidate: legality, claim correctness, goal achievement, remaining distance, progress, usefulness, first error and an optimal fresh replacement plan. These are not per-step scalar rewards. The full prompt states how to interpret every label. No artificial step rewards are inferred.

Candidate IDs are student-visible opaque response identifiers. Dataset parent/sample IDs, provenance, split metadata and reviewer receipts are excluded. Candidate quality does not follow input order or ID. Only strict, independently approved canonical labels produce exports.

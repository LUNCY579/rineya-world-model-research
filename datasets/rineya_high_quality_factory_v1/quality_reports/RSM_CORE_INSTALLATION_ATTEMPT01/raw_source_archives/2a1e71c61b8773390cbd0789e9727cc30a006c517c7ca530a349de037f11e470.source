# Memory control metadata interface

Actual author: `/root`, 2026-10-09. This describes internal coverage metadata and changes no episodic task semantics or student-visible output. Factual properties and scope requirements are defined in MEMORY_CONTROL_PROPERTIES.md.

`family_control_claims` is a list of objects. Each contains exactly one property identifier under either `property` or `name`, plus `scope` equal to `corpus_variant` or `audit_probe`. The identifier must be a public family-control property. An optional `source_family_key` must equal the record's family. Optional `evidence_manifest` supplies a relative provenance pointer; it cannot establish a witness or substitute for independent input-derived verification. Unknown properties/scopes, conflicting identifiers or families, and undeclared extra fields are errors.

A claim requires an actual witness within the same source family at the declared scope. A family claim asserts supplied coverage somewhere in that family, not that every row exhibits that relationship. Audit probes contribute zero training records and zero source families. Per-record skill tags remain subject to their separate in-record witness requirements.

`query_role_assignments` lists eight distinct declared role labels with exact `position` and `query_id` bindings to visible queries. Labels name controlled family slots. They must retain stable decoded `(key group, as_of)` signatures across a history's complete rotation schedule; a label alone does not certify an event operation or answer status. `key_role_assignments` maps the four groups delayed, lifecycle, index_tie and absent to four distinct equally styled strings, consistently renamed in events and queries. `query_rotation`, `key_rotation`, `renderer` and `controlled_variant` identify the declared schedule. Actual crossings and outcome histograms, rather than these annotations, certify coverage.

This interface accepts descriptive `name` encoding already used in memory-gap-v2 as well as the checker engineer's `property` encoding. Original candidate bytes and prior failed interface audits remain preserved. No private targets or reviewer verdicts define these fields.

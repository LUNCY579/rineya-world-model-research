# GCDL-001 nonsealed operation/composition audit

Measured on 11 October 2026 from the existing 66 nonsealed v1 shards: 1,936 accepted records. This is a read-only construction audit; no new canonical data, model training, model predictions, or sealed adaptation occurred. The machine receipt is `OPERATION_COMPOSITION_COVERAGE_20261011.json`, SHA-256 `3fdc62d5b434b335257da6910007c55c4bef3f6b71985246765e5bcbac90a374`.

There are 264 H records across 132 parent worlds. A fixed goal/constraint/rule-blind selector that prefers singleton settle, or the positive valid add in register cases, recovers every audited H winner (264/264). This measures a shortcut in the candidate construction, not learned performance.

The 66 H/register records expose 264 candidate plans: 198 add and 66 noop. Multiply, set, copy and swap occur zero times despite being enabled operations. Every plan has one step, one valid register is active, and each record has one goal atom. There are 66 active valid register slots among 165 declared slots.

TRAIN contains 176 ledger H and 44 register H rows. Its ledger portion has 44 rows in each of three individual-constraint strata and 44 in the simultaneous stratum. DEV and OOD each contain 11 ledger H and 11 register H rows. OOD forbidden-status rules do not bind because no void candidate is offered; protected-register lists are empty and maximum-cost constraints do not bind the preferred register action.

Reviewed v2 changes goal triplets and reserved vocabulary while preserving these H/register dynamics. A future composition experiment should add independently checked noncommuting operations, multiple active registers and matched goal/constraint interventions. `NEXT_OPERATION_COMPOSITION_DESIGN.md` is an unapproved design record containing rejected sketches; it is not a frozen specification or promotion approval. Existing accepted data, IDs, profiles and the sealed instrument remain immutable.

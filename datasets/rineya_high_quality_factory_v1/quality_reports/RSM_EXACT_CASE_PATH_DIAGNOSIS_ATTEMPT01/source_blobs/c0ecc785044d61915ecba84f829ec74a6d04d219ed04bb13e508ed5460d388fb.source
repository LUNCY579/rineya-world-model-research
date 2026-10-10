# Independent read-only diagnosis: exact role-registry case boundary

The diagnosis is confirmed by source and metadata reads. Frozen admission8f and living-completione929 use `research/role_registry.json`, while the actual directory entry and parent653 inventory use `research/ROLE_REGISTRY.json`. On this filesystem both lookup strings reach the same regular file; string-keyed inventory membership still differs.

Actual read-only tool `9a6e0d` observed the uppercase research entry, device16777234/inode130946201 under both lookup strings, 19279 bytes and SHA `984d8676e8967c61af0d484851d25625d22339497187d3ce28209e12d25eea72`. Device/inode are local observations, not prospective hard-coded restoration identities. The registry body was not interpreted.

Frozen metadata was independently rehashed in tool `7d54c3`:

- admission: `8f74d5d915c5d061efd5b20fe5efb9103d6d11689c3d44a0dd49fdfbdcf719bd`
- living completion: `e92965afdbb3bcca577717dd065a9a51abc7cba4869847a5c0926b78bbe78026`
- installed507 runtime freeze: `415cb144658406925f7d2939edcdc907a17f8dc1e9e62616ba6808202fb5a6ea`
- parent653 SPEC: `6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9`

`admission.py:22-29` stores the provenance row under its exact lowercase key; runtime evolution replaces its old bytes with the exact current tuple but retains that spelling. `inventory.py:159-175` builds names from actual directory traversal, then requires each returned admission row's literal key in those entries. Later admission-to-SPEC validation also needs coherent handling. Case-insensitive successful file lookup cannot establish literal membership in a case-sensitive inventory or a future restoration.

The histories must remain distinct: parent653 directly tracks uppercase at13913B/SHA `ed0b314b07fa4bbaeffde8bc2233f713d139681e213a01e1781d69e9aeefe447`; the later executed507/living transition's prior lowercase-label bytes are16187B/SHA `400789d3467ce6b33e7e134caa6645666a0bd58f586cbb2b2d8dff0bbce72a4f`, preserved in its separate archives; current bytes are19279B/SHA984d above. A canonical inventory byte-evolution entry uses the parent653 uppercase tuple, not the later507 prior tuple.

## Minimal prospective source recommendation

Introduce one source-bound canonical admission projection only for the exact JSON tuple `{path: research/role_registry.json, size:19279, sha256:984d...eea72}` to `{path: research/ROLE_REGISTRY.json, size:19279, same sha256}`. Preserve the original lowercase row as provenance and emit an explicit, derived case-identity witness. Return/store the canonical uppercase row in the operative closure so inventory and later admission/SPEC validation use one interpretation. Other paths, casings, sizes or hashes keep ordinary literal behavior; do not call generic `lower`, `casefold`, glob or suffix matching.

Require the exact prior653 uppercase binding and absence of a prior lowercase direct path. Require the actual uppercase directory entry and full current bytes, no symlinks or path aliases, and no separately named lowercase entry. If the lowercase lookup exists on the local filesystem, require it to be the same physical regular file as the canonical uppercase entry. A case-sensitive restoration may have only the canonical uppercase file: historical lowercase provenance is resolved through this explicit source-bound mapping, rather than requiring an invented second file. Store the observed identity, not a machine-specific hardcoded inode.

Do not rename, duplicate, recode or edit the registry, admission8f, living-completione929,507 freeze, archives, earlier sources, tests or parent3158 inventory. Do not add a raw omission, deferral, release exception or equality waiver. Preserve literal prior-path membership and require exact parent-bound uppercase byte evolution. Existing strict secret/whitespace/cap/release checks remain operative.

Suggested isolated new-source tests before inventory: exact positive tuple; wrong size/hash/Boolean size; arbitrary casing; missing canonical entry; wrong prior uppercase binding; symlink; distinct lowercase file (even equal bytes); two separately observed names/hard links; unchanged unrelated paths; preserved original lower-label proofs; and separate653-versus507 prior-chain checks. This recommendation supplies no implementation or test execution verdict.

## Scope and custody

Author `/root/formal_oracle`, same model family/workspace and root-guided diagnostic scope disclosed. Only read-only metadata/source/filename/stat/hash commands were used. No helper imports, tests, inventory, packing, Git/network, file rename/duplication or production semantic rerun. The future05 source was not read: a bounded filename query found its directory absent (tool472b25, exit2); no source/helper failure occurred. Earlier04 FINAL and actual tests are outside this fresh diagnostic and remain unchanged.

No data approval, future05 source approval or inventory-phase grant. This is a narrow recommendation; wait for actual05 FINAL and fresh source review before dependent execution. Writes stop after this diagnostic evidence is frozen.

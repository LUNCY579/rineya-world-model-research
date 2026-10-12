# Raw-byte preservation for new native records

New recovery and Ledger records/freeze JSON are SHA-bound to actual native
execution bytes. Some writers produce Windows CRLF; Git's inherited LF conversion
would otherwise change planning trace bytes while retaining the recorded trace
SHA. Before first publication, package-local attributes disable line-ending
conversion for these new result/freeze paths. JSON content and source/model
semantics are unchanged. Source files remain LF.

Previously published results and cold-pilot freezes are left unchanged. The
already-published B0 posthoc supplement is restored byte-identically from its
published Git blob if its local text normalization differs; it has no downstream
execution/hash dependency. New results keep their actual raw trace/freeze bytes,
and new Git blob SHA-256 is checked against working-file SHA-256 before pushing.
